"""RE-RUN the restart-count and cooling-schedule comparisons ON THE CORRECT RULER.

Both prior results (SEARCHPARAMS-1) were measured with the search objective set to the
shipped DEFAULT (bigram table) while the outcome was reported on the gauge. Since those two
rank layouts inverted, neither conclusion transfers automatically -- re-running them with the
objective SET TO the gauge is the point of this driver.

Design:
  * objective == reporting gauge == `keybo optimize --gauge-objective` (the code just shipped
    on this branch), so the search and the number it is graded on are the same measurement and
    best-of-N selection is not an oracle.
  * PAIRED seeds across the two alpha arms: the same seed is the same starting RNG stream, so
    the alpha contrast is within-seed and the search-seed spread (the dominant noise, median
    |d| = 0.883 over 32,640 pairs) largely cancels.
  * BOTH equal-N and equal-TIME comparisons, because alpha changes the cost per attempt: at a
    fixed wall-clock budget the cheaper arm buys more restarts, and that is the comparison a
    user actually faces.
  * power quoted against the SEARCH spread, not the 0.135 model-seed floor.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

MY_WORKTREE = "/local/home/zegertho/repos/keybo-wt-ruler"

# --- provenance, asserted not assumed: the shared .venv resolves `keybo` to whatever branch
# --- the SHARED checkout is on, and it moved twice in one session. Four agents hit this.
import keybo  # noqa: E402

if not keybo.__file__.startswith(MY_WORKTREE + "/"):
    raise SystemExit(f"WRONG TREE: keybo resolved to {keybo.__file__}, not {MY_WORKTREE}")

import numpy as np  # noqa: E402

from keybo.analysis.timecard import gauge_search_scorer  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.optimize.annealing import SimulatedAnnealing, stopping_point  # noqa: E402
from keybo.scoring import model_norm as MN  # noqa: E402
from keybo.scoring.table_trigram import TableTrigramScorer  # noqa: E402

OUT_TMPL = "/local/home/zegertho/agent/state/ruler/artifacts/rerun_matched_objective%s.json"
C30M = MN.S.C30M
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
OUT = OUT_TMPL % (f"_n{N_SEEDS}" if N_SEEDS != 48 else "")
ALPHAS = (0.999, 0.98)  # shipped default vs the prior art's equal-time winner
MODEL_SEED_FLOOR = 0.135  # the MODEL-seed resolution floor (NOT the binding constraint here)
SEARCH_SPREAD_MEDIAN_ABS_D = 0.883  # median |d| over 32,640 search-seed pairs (3 agents)

_SCORER: TableTrigramScorer | None = None


def _scorer():
    """One gauge scorer per process, built lazily (the surface build is ~1 s)."""
    global _SCORER
    if _SCORER is None:
        _SCORER = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    return _SCORER


class InstrumentedSA(SimulatedAnnealing):
    """SA + counters. Overrides nothing that affects the search path.

    `cool` and `optimize` are inherited verbatim; we only observe T0, the final temperature and
    the number of OUTER iterations, to test the mis-scaling claim (alpha is applied per outer
    iteration while `stopping_point` counts INNER steps, so alpha=0.999 may never get cold) on
    the CORRECT ruler rather than inheriting it from the wrong-ruler measurement.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.outer_count = 0
        self.t0 = None
        self.t_final = None

    def estimate_initial_temperature(self, layout, scorer, acceptance=None, samples=1000):
        t0 = super().estimate_initial_temperature(layout, scorer, acceptance, samples)
        if self.t0 is None:
            self.t0 = t0
        return t0

    def cool(self, temperature):
        self.outer_count += 1
        self.t_final = out = super().cool(temperature)
        return out


def one_attempt(job):
    """One `optimize --gauge-objective` attempt: SA + the 2-opt polish, exactly as the CLI."""
    alpha, seed = job
    scorer = _scorer()
    layout = Layout(C30M, ROW_STAGGERED_30)
    sa = InstrumentedSA(seed=seed, alpha=alpha, progress=False)
    t0 = time.perf_counter()
    best = sa.optimize(layout, scorer)
    t_sa = time.perf_counter() - t0
    fit_sa = scorer.fitness(best)
    t1 = time.perf_counter()
    from keybo.optimize.local_search import two_opt

    best = two_opt(best, scorer)
    t_2opt = time.perf_counter() - t1
    return {
        "alpha": alpha,
        "seed": seed,
        "layout": "".join(best.chars),
        "ms_per_char": scorer.ms_per_char(best),
        "ms_per_char_sa_only": fit_sa / scorer._covered,
        "sec": t_sa + t_2opt,
        "sec_sa": t_sa,
        "sec_2opt": t_2opt,
        "outer_count": sa.outer_count,
        "t0": sa.t0,
        "t_final": sa.t_final,
        "t_end_over_t0": (sa.t_final / sa.t0) if sa.t0 else None,
    }


def best_of_n_curve(mpc, sec_mean, ladder, rng, draws=4000):
    """E[min ms/char] over N independent attempts, bootstrapped WITH replacement.

    With replacement (not the prior art's without-replacement at N == pool) so N is set by the
    ladder and never silently clipped by pool size -- the "budget ceiling" defect the prior art
    corrected in its own first pass.
    """
    curve = {}
    for n in ladder:
        draw = rng.choice(len(mpc), size=(draws, n), replace=True)
        got = mpc[draw].min(axis=1)
        curve[str(n)] = {
            "N": n,
            "mean": float(got.mean()),
            "sd": float(got.std(ddof=1)),
            "median": float(np.median(got)),
            "p10": float(np.percentile(got, 10)),
            "p90": float(np.percentile(got, 90)),
            "min": float(got.min()),
            "expected_wall_sec": float(n * sec_mean),
        }
    return curve


def main():
    rng = np.random.default_rng(20260801)
    jobs = [(a, s) for a in ALPHAS for s in range(N_SEEDS)]
    workers = min(16, max(1, (os.cpu_count() or 8) // 8))
    print(f"running {len(jobs)} attempts on {workers} workers ({N_SEEDS} seeds x {len(ALPHAS)} alphas)")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(one_attempt, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    print(f"done in {wall:.0f}s")

    by_alpha = {a: [r for r in runs if r["alpha"] == a] for a in ALPHAS}
    for arm in by_alpha.values():
        arm.sort(key=lambda r: r["seed"])

    # --- parity re-assertion AFTER the search: the objective the runs were scored on is still
    # --- the published gauge (a cached-surface mutation elsewhere would show up here).
    scorer = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    parity = {}
    for label, board in (("C30M", C30M), ("best_found", min(runs, key=lambda r: r["ms_per_char"])["layout"])):
        parity[label] = scorer.parity_rel_dev(Layout(board, ROW_STAGGERED_30))
    worst_parity = max(parity.values())
    print(f"parity vs analyze: worst rel dev {worst_parity:.3e}")
    if worst_parity > 1e-12:
        raise SystemExit("parity gate FAILED after the run — refusing to report")

    result = {
        "design": (
            "objective == reporting gauge (analyze's ms/char, K31 T2+Tcond seed-mean over the "
            "blend-v1 trigram corpus) via the shipped `optimize --gauge-objective`; "
            "SimulatedAnnealing + two_opt, start=C30M, PAIRED seeds across alpha arms"
        ),
        "provenance": {
            "keybo_module": keybo.__file__,
            "worktree": MY_WORKTREE,
            "git_sha": os.popen(f"cd {MY_WORKTREE} && git rev-parse HEAD").read().strip(),
            "git_branch": os.popen(f"cd {MY_WORKTREE} && git branch --show-current").read().strip(),
            "thread_vars": {
                v: os.environ.get(v)
                for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
        "parity_gate": {
            "rel_dev_vs_analyze": parity,
            "tolerance": 1e-12,
            "verdict": "PASS — the objective these runs optimized IS analyze's ms/char",
        },
        "n_seeds": N_SEEDS,
        "alphas": list(ALPHAS),
        "wall_sec": wall,
        "workers": workers,
        "scales": {
            "model_seed_floor": MODEL_SEED_FLOOR,
            "search_spread_median_abs_d": SEARCH_SPREAD_MEDIAN_ABS_D,
            "note": (
                "power is quoted against the SEARCH spread; the 0.135 model-seed floor is a "
                "different (and here non-binding) scale — three agents found this independently"
            ),
        },
        "stopping_point_inner_steps": stopping_point(30),
        "arms": {},
    }

    ladder = [1, 2, 4, 8, 16, 32, 48]
    ladder = [n for n in ladder if n <= N_SEEDS]
    for alpha, arm in by_alpha.items():
        mpc = np.array([r["ms_per_char"] for r in arm])
        sec = np.array([r["sec"] for r in arm])
        sa_only = np.array([r["ms_per_char_sa_only"] for r in arm])
        result["arms"][f"alpha={alpha:g}"] = {
            "single_attempt": {
                "mean": float(mpc.mean()),
                "sd": float(mpc.std(ddof=1)),
                "median": float(np.median(mpc)),
                "min": float(mpc.min()),
                "max": float(mpc.max()),
                "range": float(mpc.max() - mpc.min()),
            },
            "sa_only_before_2opt": {"mean": float(sa_only.mean()), "sd": float(sa_only.std(ddof=1))},
            "polish_gain_mean": float((sa_only - mpc).mean()),
            "sec_per_attempt": {
                "mean": float(sec.mean()),
                "sd": float(sec.std(ddof=1)),
                "sa_mean": float(np.mean([r["sec_sa"] for r in arm])),
                "2opt_mean": float(np.mean([r["sec_2opt"] for r in arm])),
            },
            "cooling_instrument": {
                "outer_count_mean": float(np.mean([r["outer_count"] for r in arm])),
                "t_end_over_t0_mean": float(np.mean([r["t_end_over_t0"] for r in arm])),
                "t_end_over_t0_max": float(np.max([r["t_end_over_t0"] for r in arm])),
                "note": (
                    "alpha is applied ONCE PER OUTER iteration while stopping_point counts INNER "
                    "steps, so T_end/T0 near 1 means the annealer never got cold and the 2-opt "
                    "polish did the descent (cross-check: polish_gain_mean)"
                ),
            },
            "n_distinct_layouts": len({r["layout"] for r in arm}),
            "restart_curve": best_of_n_curve(mpc, float(sec.mean()), ladder, rng),
            "best_found": {
                "layout": arm[int(mpc.argmin())]["layout"],
                "ms_per_char": float(mpc.min()),
                "seed": arm[int(mpc.argmin())]["seed"],
            },
        }

    # --- saturation: the first doubling whose mean gain is below each scale ---
    for arm_out in result["arms"].values():
        curve = arm_out["restart_curve"]
        doublings = []
        keys = [int(k) for k in curve]
        for a, b in zip(keys, keys[1:], strict=False):
            delta = curve[str(a)]["mean"] - curve[str(b)]["mean"]
            doublings.append(
                {
                    "N": a,
                    "to_N": b,
                    "delta_ms_per_char": float(delta),
                    "below_model_seed_floor": bool(delta < MODEL_SEED_FLOOR),
                    "below_search_spread": bool(delta < SEARCH_SPREAD_MEDIAN_ABS_D),
                }
            )
        arm_out["doublings"] = doublings
        arm_out["saturation_N_vs_model_seed_floor"] = next(
            (d["N"] for d in doublings if d["below_model_seed_floor"]), None
        )
        arm_out["saturation_N_vs_search_spread"] = next(
            (d["N"] for d in doublings if d["below_search_spread"]), None
        )

    # --- THE A/B: alpha=0.98 vs the shipped alpha=0.999, paired and both ways ---
    a999 = np.array([r["ms_per_char"] for r in by_alpha[0.999]])
    a098 = np.array([r["ms_per_char"] for r in by_alpha[0.98]])
    sec999 = float(np.mean([r["sec"] for r in by_alpha[0.999]]))
    sec098 = float(np.mean([r["sec"] for r in by_alpha[0.98]]))
    paired = a999 - a098  # >0 means alpha=0.98 is BETTER (lower ms/char)
    boot = rng.choice(paired, size=(20000, len(paired)), replace=True).mean(axis=1)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    equal_time = {}
    for budget in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0):
        row = {}
        for alpha, mpc, sec in ((0.999, a999, sec999), (0.98, a098, sec098)):
            n = max(1, int(budget // sec))
            draw = rng.choice(len(mpc), size=(4000, n), replace=True)
            row[f"alpha={alpha:g}"] = {
                "N_affordable": n,
                "E_ms_per_char": float(mpc[draw].min(axis=1).mean()),
            }
        gain = row["alpha=0.999"]["E_ms_per_char"] - row["alpha=0.98"]["E_ms_per_char"]
        row["gain_for_alpha_0.98"] = float(gain)
        row["exceeds_model_seed_floor"] = bool(gain > MODEL_SEED_FLOOR)
        row["exceeds_search_spread_median"] = bool(gain > SEARCH_SPREAD_MEDIAN_ABS_D)
        equal_time[str(budget)] = row

    result["alpha_AB"] = {
        "hypothesis": (
            "PRE-REGISTERED by SEARCHPARAMS-1 on the WRONG ruler: alpha=0.98 beat the shipped "
            "0.999 on 7/8 equal-time budgets. Two arms only, both named in advance, so there is "
            "no winner's-curse selection over a sweep here."
        ),
        "equal_N_paired": {
            "n_pairs": len(paired),
            "mean_gain_for_alpha_0.98": float(paired.mean()),
            "sd_of_paired_difference": float(paired.std(ddof=1)),
            "median_gain": float(np.median(paired)),
            "wins_for_alpha_0.98": int((paired > 0).sum()),
            "losses": int((paired < 0).sum()),
            "bootstrap_ci95_of_mean_gain": ci,
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "gain_in_model_seed_floors": float(paired.mean() / MODEL_SEED_FLOOR),
            "gain_in_search_spread_units": float(paired.mean() / SEARCH_SPREAD_MEDIAN_ABS_D),
        },
        "equal_time": equal_time,
        "sec_per_attempt": {"alpha=0.999": sec999, "alpha=0.98": sec098},
    }

    with open(OUT, "w") as fh:
        json.dump({"meta": result, "runs": runs}, fh, indent=1)
    print(json.dumps(result, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
