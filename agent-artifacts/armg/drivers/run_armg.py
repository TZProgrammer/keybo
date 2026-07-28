"""ARM G runner — n=5 seeds of the ARM G objective PLUS n=5 seeds of a same-seed,
same-budget BASELINE CONTROL.

See ../PREREGISTRATION.md. The control arm is not optional: it is (a) how ARM G measures
its OWN search-noise sd instead of borrowing another arm's (the standing
POOL x REPLICATE-STRUCTURE x SCALE x STATISTIC rule), and (b) the same-size same-seed
placebo that makes any ARM G gauge gain attributable to the OBJECTIVE rather than to the
draw (traps 17/32/34).

WHY NOT `search_placebo.py` / `run_budget.py` (the drivers the brief points at): they carry
hardcoded paths into OTHER agents' worktrees (traps 35/65) --
  * `search_placebo.py`  : `cwd="/tmp/optev"`, `timeout=3600`, writes into
                           `state/optevidence/artifacts` (another workspace).
  * `run_budget.py`      : `WORKTREE = Path("/tmp/speedtie")`.
  * `analyze_budget.py`  : `WORKTREE = Path("/tmp/speedtie")`.
REPOINTED: this file hardcodes `/tmp/armg` and writes only into my own state dir. The
engine (`search.py`) and evaluator (`evobj.py`) are the inherited ones; `evobj.py` is
byte-identical to the speedtie copy (md5 dc45ef503792576157a872a996d9e9d7) and `search.py`
differs ONLY by the added `armg` arm (its `baseline` path is untouched, so the control arm
is the same code that produced the reference band).

`unique_evals` is reported ACHIEVED, never requested: the engine stops on the EPOCH
schedule, so a run can fall short and still exit 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = Path("/tmp/armg")
OUTDIR = Path("/local/home/zegertho/agent/state/armg/artifacts/runs")

#: pre-registered in ../PREREGISTRATION.md section 2, before any run existed.
SEED_FORMULA = "20260728 + 7919 * r"
BUDGET = 1_000_000
ISLANDS = 20
EPOCHS = 12
OVERSHOOT = 1.95
GA_SHARE = 0.6
POLISH = 40
#: a seed below this fraction of the requested budget is EXCLUDED from the primary n and
#: reported as excluded, with a sensitivity analysis including it.
ACHIEVED_FLOOR = 0.80


def seed_of(r: int) -> int:
    return 20_260_728 + 7919 * r


def launch(arm: str, r: int, budget: int, epochs: int) -> subprocess.Popen:
    """One (arm, seed), detached. NO subprocess timeout — a killed run looks exactly like a
    missing sentinel (traps 22 + 1). cwd is MY worktree."""
    out = OUTDIR / f"{arm}-r{r}.json"
    log = OUTDIR / f"{arm}-r{r}.log"
    cmd = [
        "uv", "run", "--no-sync", "python", str(HERE / "search.py"),
        "--arm", arm,
        "--budget", str(budget),
        "--islands", str(ISLANDS),
        "--epochs", str(epochs),
        "--overshoot", str(OVERSHOOT),
        "--ga-share", str(GA_SHARE),
        "--polish-sweeps", str(POLISH),
        "--seed", str(seed_of(r)),
        "--out", str(out),
    ]
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    fh = open(log, "w")  # noqa: SIM115 - handed to a long-lived detached child
    return subprocess.Popen(cmd, cwd=str(WORKTREE), stdout=fh,
                            stderr=subprocess.STDOUT, env=env)


def main() -> int:
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["armg", "baseline"]
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else BUDGET
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else EPOCHS
    tag = sys.argv[5] if len(sys.argv) > 5 else ""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    per_epoch = int(budget * OVERSHOOT) // (epochs * ISLANDS)
    print(f"arms={arms} repeats={repeats} budget={budget:,} islands={ISLANDS} "
          f"epochs={epochs} => {per_epoch:,} calls/island/epoch", flush=True)
    print(f"seeds: {[seed_of(r) for r in range(repeats)]}  (formula {SEED_FORMULA})",
          flush=True)

    t0 = time.time()
    # All (arm, seed) cells launched together: they are independent, the box has 192 cores,
    # and each run is capped to 20 islands with BLAS pinned to 1 thread.
    procs = {(a, r): launch(a, r, budget, epochs)
             for a in arms for r in range(repeats)}
    rcs: dict[tuple[str, int], int] = {}
    for key, p in procs.items():
        rcs[key] = p.wait()
        print(f"[{time.time()-t0:8.1f}s] {key[0]} r={key[1]} seed={seed_of(key[1])} "
              f"rc={rcs[key]}", flush=True)

    rows = []
    for a in arms:
        for r in range(repeats):
            out = OUTDIR / f"{a}-r{r}.json"
            rc = rcs[(a, r)]
            row: dict = {"arm": a, "r": r, "seed": seed_of(r), "rc": rc, "out": str(out)}
            if rc == 0 and out.exists():
                with open(out) as fh:
                    blob = json.load(fh)
                achieved = blob["unique_evals"]
                row.update({
                    "layout": blob["champion"]["layout"],
                    "search_fitness": blob["champion"]["fitness"],
                    "unique_evals_ACHIEVED": achieved,
                    "budget_requested": blob["budget_requested"],
                    "achieved_frac": achieved / blob["budget_requested"],
                    "clears_floor": achieved >= ACHIEVED_FLOOR * blob["budget_requested"],
                    "epochs_run": blob["epochs_run"],
                    "elapsed_s": blob["elapsed_s"],
                    "armg_constants_check": blob.get("armg_constants_check"),
                    "top50": blob["top50"],
                    "ok": True,
                })
            else:
                row.update({"ok": False,
                            "why": f"rc={rc} json_exists={out.exists()}"})
            rows.append(row)
            if row["ok"]:
                print(f"  {a} r={r} {row['layout']} fit={row['search_fitness']:.6f} "
                      f"uniq={row['unique_evals_ACHIEVED']:,} "
                      f"({row['achieved_frac']:.1%}) "
                      f"{'' if row['clears_floor'] else '<<< BELOW 80% FLOOR'}", flush=True)
            else:
                print(f"  {a} r={r} FAILED {row['why']}", flush=True)

    summary = {
        "experiment": "ARM G",
        "prereg": "agent-artifacts/armg/PREREGISTRATION.md (committed 0f606d1)",
        "arms": arms,
        "budget_requested_per_run": budget,
        "islands": ISLANDS, "epochs": epochs, "overshoot": OVERSHOOT,
        "ga_share": GA_SHARE, "polish_sweeps": POLISH,
        "calls_per_island_per_epoch": per_epoch,
        "seed_formula": SEED_FORMULA,
        "achieved_floor": ACHIEVED_FLOOR,
        "wall_clock_s": time.time() - t0,
        "n_ok": sum(1 for x in rows if x["ok"]),
        "n_expected": len(arms) * repeats,
        "runs": rows,
        "modelled_only": ("MODELLED ONLY: g-frame, baked 90 WPM, blend-v1, skipgrams "
                          "1-skip31. Not a claim about realized human typing speed."),
        "note": ("unique_evals is ACHIEVED, not requested — the engine stops on the epoch "
                 "schedule, so a run can fall short and still exit 0."),
    }
    path = OUTDIR / f"armg-summary{tag}.json"
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nWROTE {path}  n_ok={summary['n_ok']}/{summary['n_expected']} "
          f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    return 0 if summary["n_ok"] == summary["n_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
