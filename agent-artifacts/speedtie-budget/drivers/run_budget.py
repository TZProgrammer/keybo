"""SPEEDTIE-BUDGET runner — the SAME search as the 1M placebo, continued 10x longer.

Answers: does SPEEDTIE-1's free gauge headroom survive at the full budget, or is it an
artifact of under-convergence?

WHY THIS EXISTS INSTEAD OF `search_placebo.py` (which the brief pointed at). That driver
carries four defects that are fatal to THIS experiment, and they are recorded in
../PREREGISTRATION.md section 6:
  1. `cwd="/tmp/optev"` — launches the subprocess in ANOTHER agent's worktree at a DIFFERENT
     commit. Trap 35 wearing a subprocess's clothes: the copied file's own sys.path hygiene
     does not save you when the child process starts in someone else's tree.
  2. `timeout=3600` — a killed run looks exactly like a missing sentinel (traps 22 + 1).
  3. writes into `state/optevidence/artifacts` — another workspace.
  4. hardcodes islands=20/epochs=12, so it cannot express a 10M run at all.
`search.py` and `evobj.py` themselves are used UNMODIFIED (md5-verified against the originals).

THE ONE FACTOR. `search.py:318-323` builds `init` as `islands x 64` uniformly random C30M
permutations from `default_rng(seed)` — no incumbent, no warm start. So `init` is a function of
(seed, islands) ONLY. Holding islands=20 (the placebo's value) makes the 10M initial population
BIT-IDENTICAL to the 1M one for the same seed, and choosing epochs=120 makes
`int(budget*overshoot)//(epochs*islands)` = 8,125 calls/island/epoch — EXACTLY the placebo's
per-epoch spend. Only the epoch COUNT differs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = Path("/tmp/speedtie")
OUTDIR = Path("/local/home/zegertho/agent/state/speedtie/artifacts/runs")

# The placebo's formula, reproduced verbatim so the two budgets are seed-matched run-for-run.
SEED_FORMULA = "900000 + 7919 * r"


def seed_of(r: int) -> int:
    return 900_000 + 7919 * r


def launch(r: int, budget: int, islands: int, epochs: int) -> subprocess.Popen:
    """One seed, detached. NO timeout (defect 2). cwd is MY worktree (defect 1)."""
    seed = seed_of(r)
    out = OUTDIR / f"b{budget}-r{r}.json"
    log = OUTDIR / f"b{budget}-r{r}.log"
    cmd = [
        "uv", "run", "--no-sync", "python", str(HERE / "search.py"),
        "--arm", "baseline",
        "--budget", str(budget),
        "--islands", str(islands),
        "--epochs", str(epochs),
        "--overshoot", "1.95",
        "--ga-share", "0.6",
        "--polish-sweeps", "40",
        "--seed", str(seed),
        "--out", str(out),
    ]
    env = dict(os.environ)
    # search.py already setdefaults these to 1; make it explicit so N parallel runs of 20
    # islands each cannot oversubscribe the box through BLAS threads.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    fh = open(log, "w")
    # cwd = MY worktree, verified by the caller's positive control on FastEval.corpus_dir.
    return subprocess.Popen(cmd, cwd=str(WORKTREE), stdout=fh, stderr=subprocess.STDOUT,
                            env=env)


def main() -> int:
    budget = int(sys.argv[1])
    repeats = int(sys.argv[2])
    islands = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    # `first` lets a later invocation launch only the seeds not yet run (the timing seed r=0 is
    # already complete and its JSON must not be overwritten). The harvest below still reads
    # ALL of r in [0, repeats), so the summary covers every seed regardless of who ran it.
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    per_epoch = int(budget * 1.95) // (epochs * islands)
    print(f"budget={budget:,} repeats={repeats} islands={islands} epochs={epochs} "
          f"=> {per_epoch:,} calls/island/epoch  (placebo 1M was 8,125)", flush=True)
    print(f"seeds: {[seed_of(r) for r in range(repeats)]}  (formula {SEED_FORMULA})", flush=True)
    print(f"launching r={first}..{repeats-1}; r<{first} assumed already complete on disk",
          flush=True)

    t0 = time.time()
    procs = {r: launch(r, budget, islands, epochs) for r in range(first, repeats)}
    rcs: dict[int, int] = {}
    for r, p in procs.items():
        rcs[r] = p.wait()
        print(f"[{time.time()-t0:8.1f}s] r={r} seed={seed_of(r)} rc={rcs[r]}", flush=True)

    # Harvest. A run that produced no JSON, or fell short, is reported as such — NEVER
    # silently substituted (brief hard constraint).
    rows = []
    for r in range(repeats):
        out = OUTDIR / f"b{budget}-r{r}.json"
        # A seed launched by an EARLIER invocation has no rc here; its own sentinel already
        # recorded rc=0 and its JSON exists. Absence of an rc in THIS process is not a failure
        # (trap 1: absence is not disproof) — but it is also not a pass, so the JSON must exist.
        if r not in rcs:
            rcs[r] = 0 if out.exists() else 99
        row: dict = {"r": r, "seed": seed_of(r), "rc": rcs[r], "out": str(out),
                     "launched_by_this_invocation": r >= first}
        if rcs[r] == 0 and out.exists():
            blob = json.load(open(out))
            row.update({
                "layout": blob["champion"]["layout"],
                "search_fitness": blob["champion"]["fitness"],
                "unique_evals": blob["unique_evals"],
                "budget_requested": blob["budget_requested"],
                "epochs_run": blob["epochs_run"],
                "elapsed_s": blob["elapsed_s"],
                "ok": True,
            })
        else:
            row["ok"] = False
            row["why"] = f"rc={rcs[r]} json_exists={out.exists()}"
        rows.append(row)
        print(f"  r={r} {row.get('layout','<FAILED>')} "
              f"uniq={row.get('unique_evals','-'):,}" if row["ok"] else f"  r={r} FAILED {row['why']}",
              flush=True)

    summary = {
        "experiment": "SPEEDTIE-BUDGET",
        "arm": "baseline (served ms/char, blend-v1, 90 WPM) — MODELLED ONLY",
        "budget_requested": budget,
        "repeats_launched": repeats,
        "islands": islands,
        "epochs": epochs,
        "overshoot": 1.95,
        "calls_per_island_per_epoch": per_epoch,
        "seed_formula": SEED_FORMULA,
        "wall_clock_s": time.time() - t0,
        "n_ok": sum(1 for x in rows if x["ok"]),
        "runs": rows,
        "note": ("unique_evals is the ACHIEVED count, not the request. A run below 80% of the "
                 "requested budget is a DIFFERENT experiment and is labelled as one."),
    }
    path = OUTDIR / f"budget-{budget}-summary.json"
    json.dump(summary, open(path, "w"), indent=1)
    print(f"\nWROTE {path}  n_ok={summary['n_ok']}/{repeats} "
          f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    return 0 if summary["n_ok"] == repeats else 1


if __name__ == "__main__":
    sys.exit(main())
