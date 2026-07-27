"""FLAGSHIP-1 — is the between-layout speed difference RESOLVABLE, or is the field a tie?

The campaign's "resolution floor" (GEOMEAN-1 0.7186 ms/char on iWeb; REHUNT-1 re-measured
per corpus) is the **max per-seed spread WITHIN a layout** — i.e. how much one layout's own
predicted ms/char moves when you swap which of the three fitted model seeds you score on.
Comparing a BETWEEN-layout difference to that number is the conservative thing to do only
if the seed-to-seed variation is INDEPENDENT across layouts.

It is not obviously independent: all layouts are scored on the SAME three seed tables, so a
seed that predicts "everything is slow" shifts every layout together. That is a COMMON-MODE
offset, and a common-mode offset cancels in a difference. The right instrument for
"is layout A faster than layout B" is therefore the PAIRED (within-seed) difference — the
same logic as a paired t-test versus an unpaired one.

So this driver answers three separate questions, and keeps them separate:

1. **DECOMPOSITION.** Of the total variance in per-(layout,seed) ms/char, how much is the
   seed main effect (common mode), how much the layout main effect (the signal), how much
   the layout x seed interaction (the part that does NOT cancel)? If the seed main effect
   dominates, the unpaired floor is the wrong ruler and REHUNT-1's "resolvably slower"
   verdicts are conservative but the *field* verdict ("all within noise") is not safe.

2. **PAIRED SIGN TEST.** For every ordered pair of the 6 candidates, does the SAME sign of
   difference hold on all 3 seeds? With 3 seeds the strongest available statement is
   unanimity (a 3/3 sign agreement); it is weak evidence in isolation (p = 1/4 under a
   naive symmetric null) but it is exactly reproducible and it is the only within-seed
   statement the surviving artifacts can support. Reported as unanimity, never as a p-value.

3. **THE HONEST CAVEAT.** n=3 seeds. A 3/3 unanimous sign is NOT a significance claim, and
   the seeds are not independent draws from a population of typists — they are three fits of
   the same model family on the same data. So even a perfect paired result bounds only
   ESTIMATOR resolution, never realized typing speed. Phase-D is cancelled; nothing here is
   an empirical speed claim.

Also emitted, because the adoption question needs them and they are cheap here:
   * per-corpus ms/char and the paired matrix, for all three corpora;
   * a SEED-DROP jackknife: recompute the ranking with each seed removed (2 of 3), to see
     whether rank 1 survives losing any single seed.

MODELED ONLY. The time surface is a baked 90 WPM artifact; --target-wpm does not move the
three fitted model surfaces and 7 of 8 per-seed model surfaces are gone.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

# Worktree isolation: append, never insert(0, ...) — trap 35. And assert we got THIS tree.
_ROOT = Path(__file__).resolve()
sys.path.append(str(Path("/tmp/flagship/src")))

import keybo  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies  # noqa: E402

assert keybo.__file__.startswith("/tmp/flagship/"), f"NOT the worktree: {keybo.__file__}"

CANDIDATES = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
}
EXTERNALS = {
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}
CORPORA = {
    "iweb": "/tmp/flagship/data/corpus",
    "blend-v1": "/tmp/flagship/data/corpus/blend-v1",
    "blend-v1-no-anchor": "/tmp/flagship-corpora/blend-v1-no-anchor",
}
# REHUNT-1's per-corpus measured floors (rehunt-1/runs/rehunt-floor.json), re-measured below.
REHUNT_FLOORS = {
    "iweb": 0.7185664292632339,
    "blend-v1": 0.6654644286543601,
    "blend-v1-no-anchor": 0.6641431262198978,
}
# GEOMEAN-1's iWeb positive control (via rehunt_floor.py GEOMEAN1_TABLE) — ms/char, spread.
GEOMEAN1_IWEB = {
    "keybo-lsb": (253.2104, 0.5061),
    "keybo-lsb+lm": (253.2657, 0.5643),
    "lsb-sib": (253.2896, 0.7186),
    "archive-1843": (253.4523, 0.6281),
    "archive-1846": (253.4586, 0.6230),
    "qwerty30m": (262.4294, 0.9600),
}


def measure(surface: TimeSurface, lay30: str) -> dict:
    card = surface.card(lay30)
    chars = card.total_ms / card.ms_per_char
    per_seed = [t / chars for t in surface.seed_totals(lay30)]
    return {
        "ms_per_char": card.ms_per_char,
        "per_seed_ms_per_char": per_seed,
        "per_seed_spread": float(max(per_seed) - min(per_seed)),
        "coverage_pct": card.coverage_pct,
        "covered_chars": chars,
    }


def main() -> None:
    out: dict = {
        "what": "FLAGSHIP-1 paired-seed resolution of the 6-candidate predicted-time field",
        "wpm": 90.0,
        "n_seeds": 3,
        "seed_caveat": (
            "3 seeds are 3 fits of ONE model family on ONE dataset, not independent draws "
            "from a population of typists. A 3/3 unanimous paired sign bounds ESTIMATOR "
            "resolution only. Phase-D is cancelled; no claim here is about realized typing "
            "speed."
        ),
        "floor_definition": (
            "REHUNT-1/GEOMEAN-1 floor = max WITHIN-layout per-seed spread over the 5 "
            "incumbents. It is an UNPAIRED ruler; this driver also computes the PAIRED one."
        ),
        "corpora": {},
    }

    for corpus, path in CORPORA.items():
        tri = load_frequencies(str(Path(path) / "trigrams.txt"))
        surf = TimeSurface(tri, target_wpm=90.0, keep_seed_tables=True)
        rows = {n: measure(surf, lay) for n, lay in {**CANDIDATES, **EXTERNALS}.items()}

        cand = list(CANDIDATES)
        M = np.array([rows[n]["per_seed_ms_per_char"] for n in cand])  # 6 x 3

        grand = float(M.mean())
        layout_eff = M.mean(axis=1) - grand
        seed_eff = M.mean(axis=0) - grand
        resid = M - grand - layout_eff[:, None] - seed_eff[None, :]
        ss_layout = float((M.shape[1] * layout_eff**2).sum())
        ss_seed = float((M.shape[0] * seed_eff**2).sum())
        ss_resid = float((resid**2).sum())
        ss_tot = ss_layout + ss_seed + ss_resid

        # unpaired floor, re-measured (max within-layout spread over the 5 INCUMBENTS,
        # i.e. excluding flagship-c3, matching REHUNT-1's definition exactly)
        incumbent5 = [n for n in cand if n != "flagship-c3"]
        floor_unpaired = max(rows[n]["per_seed_spread"] for n in incumbent5)
        # paired resolution: the largest |layout x seed| deviation that does NOT cancel in a
        # within-seed difference. This is the honest paired analogue of the floor.
        floor_paired = float(np.abs(resid).max() * 2)

        pairs = {}
        for a, b in itertools.combinations(cand, 2):
            da = np.array(rows[a]["per_seed_ms_per_char"])
            db = np.array(rows[b]["per_seed_ms_per_char"])
            d = da - db  # >0 means a is SLOWER than b
            pairs[f"{a}|{b}"] = {
                "mean_delta_ms_per_char": float(d.mean()),
                "per_seed_delta": [float(x) for x in d],
                "sign_unanimous": bool(np.all(d > 0) or np.all(d < 0)),
                "min_abs_delta": float(np.abs(d).min()),
                "faster": (a if d.mean() < 0 else b),
                "clears_unpaired_floor": bool(abs(float(d.mean())) > floor_unpaired),
                "clears_paired_resolution": bool(abs(float(d.mean())) > floor_paired),
            }

        # seed-drop jackknife: does rank 1 survive losing any single seed?
        jack = {}
        for drop in range(3):
            keep = [s for s in range(3) if s != drop]
            means = {n: float(np.mean([rows[n]["per_seed_ms_per_char"][s] for s in keep])) for n in cand}
            order = sorted(means, key=means.get)
            jack[f"drop_seed{drop}"] = {"order": order, "rank1": order[0], "means": means}
        # and each single seed alone
        single = {}
        for s in range(3):
            means = {n: rows[n]["per_seed_ms_per_char"][s] for n in cand}
            order = sorted(means, key=means.get)
            single[f"seed{s}_only"] = {"order": order, "rank1": order[0], "means": means}

        order_mean = sorted(cand, key=lambda n: rows[n]["ms_per_char"])
        out["corpora"][corpus] = {
            "corpus_dir": path,
            "rows": rows,
            "order_by_mean_ms_per_char": order_mean,
            "span_ms_per_char": rows[order_mean[-1]]["ms_per_char"]
            - rows[order_mean[0]]["ms_per_char"],
            "variance_decomposition": {
                "ss_layout": ss_layout,
                "ss_seed": ss_seed,
                "ss_interaction_resid": ss_resid,
                "ss_total": ss_tot,
                "pct_layout": 100 * ss_layout / ss_tot,
                "pct_seed": 100 * ss_seed / ss_tot,
                "pct_interaction": 100 * ss_resid / ss_tot,
                "reading": (
                    "seed main effect is COMMON MODE and cancels in a within-seed "
                    "difference; only the interaction term limits a paired comparison"
                ),
            },
            "floor_unpaired_remeasured": floor_unpaired,
            "floor_unpaired_rehunt1": REHUNT_FLOORS[corpus],
            "floor_unpaired_reproduces": bool(
                abs(floor_unpaired - REHUNT_FLOORS[corpus]) < 5e-4
            ),
            "resolution_paired": floor_paired,
            "resolution_paired_note": (
                "2 x max|layout-seed interaction residual| — the worst-case amount a "
                "within-seed pairwise difference can be perturbed by seed choice"
            ),
            "pairs": pairs,
            "n_pairs": len(pairs),
            "n_pairs_sign_unanimous": sum(p["sign_unanimous"] for p in pairs.values()),
            "n_pairs_clear_unpaired_floor": sum(
                p["clears_unpaired_floor"] for p in pairs.values()
            ),
            "n_pairs_clear_paired_resolution": sum(
                p["clears_paired_resolution"] for p in pairs.values()
            ),
            "jackknife_drop_one_seed": jack,
            "single_seed": single,
        }

        # positive control on iWeb against GEOMEAN-1's frozen table
        if corpus == "iweb":
            ctrl = {}
            for n, (ms_ref, spread_ref) in GEOMEAN1_IWEB.items():
                got = rows[n]
                ctrl[n] = {
                    "geomean1_ms_per_char": ms_ref,
                    "measured_ms_per_char": got["ms_per_char"],
                    "delta": got["ms_per_char"] - ms_ref,
                    "geomean1_spread": spread_ref,
                    "measured_spread": got["per_seed_spread"],
                    "reproduces": bool(
                        abs(got["ms_per_char"] - ms_ref) < 5e-4
                        and abs(got["per_seed_spread"] - spread_ref) < 5e-4
                    ),
                }
            out["positive_control_iweb_vs_geomean1"] = ctrl
            out["positive_control_reproduces"] = f"{sum(c['reproduces'] for c in ctrl.values())}/{len(ctrl)}"

    dest = Path(sys.argv[1])
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")

    # human-readable
    print(f"\npositive control (iWeb vs GEOMEAN-1): {out['positive_control_reproduces']}")
    for corpus, c in out["corpora"].items():
        v = c["variance_decomposition"]
        print(f"\n=== {corpus} ===")
        print(
            f"  variance: layout {v['pct_layout']:.2f}%  seed {v['pct_seed']:.2f}%  "
            f"interaction {v['pct_interaction']:.4f}%"
        )
        print(
            f"  floor UNPAIRED re-measured {c['floor_unpaired_remeasured']:.4f} "
            f"(REHUNT-1 {c['floor_unpaired_rehunt1']:.4f}, reproduces={c['floor_unpaired_reproduces']})"
        )
        print(f"  resolution PAIRED         {c['resolution_paired']:.6f}")
        print(f"  order: {' < '.join(c['order_by_mean_ms_per_char'])}   span {c['span_ms_per_char']:.4f}")
        print(
            f"  pairs: {c['n_pairs_sign_unanimous']}/{c['n_pairs']} sign-unanimous | "
            f"{c['n_pairs_clear_unpaired_floor']}/{c['n_pairs']} clear UNPAIRED floor | "
            f"{c['n_pairs_clear_paired_resolution']}/{c['n_pairs']} clear PAIRED resolution"
        )
        for k, j in c["jackknife_drop_one_seed"].items():
            print(f"    {k}: rank1={j['rank1']}")
        for k, j in c["single_seed"].items():
            print(f"    {k}: rank1={j['rank1']}  order={' < '.join(j['order'])}")


if __name__ == "__main__":
    main()
