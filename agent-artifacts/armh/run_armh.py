"""ARM H runner — THREE PHASES, in a deliberate causal order.

  PHASE 1  `baseline` x 5 seeds (+1 reproduction control at ARM G's r=0 seed)
           => MEASURES sd_H. Must complete before phase 2 can start, because
              EPS := 2*sd_H and the SEARCH band must equal the VERDICT band. ARM G's
              registered failure was exactly a search band LOOSER than its verdict band
              (by 0.0251); every one of its champions landed in the gap.
  PHASE 2  `armh` COLD x 5 seeds  +  `armh` WARM x 5 seeds, at EPS from phase 1.
  PHASE 3  the champion gate (`gate_armh.py`), which re-checks feasibility through the
           SHIPPED analyze path, not through FastEval.

rc is written to a SENTINEL per run and the summary is gated on reading those files -- a
completion callback is best-effort notification only (trap 50: a `while pgrep` watcher died
three times in one session while the work itself completed fine).

`unique_evals` is reported ACHIEVED, never requested: the engine stops on the EPOCH
schedule, so a run can fall short and still exit 0.

`.keys.npy` sidecars are RETAINED so `--resume` works (SPEEDTIE-BUDGET-1 deleted 388MB of
them and permanently lost that ability for its runs).

Repointed paths, declared (trap 35): this file hardcodes /tmp/armh and writes only into my
own state dir. It does NOT reuse `search_placebo.py` (cwd=/tmp/optev), `run_budget.py` or
`analyze_budget.py` (WORKTREE=/tmp/speedtie).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import armh_constants as AH  # noqa: E402

WORKTREE = Path("/tmp/armh")
OUTDIR = Path("/local/home/zegertho/agent/state/armh/artifacts/runs")

BUDGET = 1_000_000
ISLANDS = 20
EPOCHS = 12
OVERSHOOT = 1.95
GA_SHARE = 0.6
POLISH = 40
ACHIEVED_FLOOR = 0.80
#: ARM G's r=0 seed, run as a BIT-EXACT reproduction control. Deliberately NOT in the sd_H
#: pool -- including it would share a draw with ARM G and contaminate my own ruler.
REPRO_SEED = 20_260_728


def launch(tag: str, seed: int, arm: str, extra: list[str]) -> subprocess.Popen:
    """One run, detached. NO subprocess timeout -- a killed run looks exactly like a missing
    sentinel (traps 22 + 1), and absence of a sentinel is NOT evidence of rc=0."""
    out = OUTDIR / f"{tag}.json"
    log = OUTDIR / f"{tag}.log"
    rcf = OUTDIR / f"{tag}.rc"
    cmd = ["uv", "run", "--no-sync", "python", str(HERE / "search.py"),
           "--arm", arm, "--budget", str(BUDGET), "--islands", str(ISLANDS),
           "--epochs", str(EPOCHS), "--overshoot", str(OVERSHOOT),
           "--ga-share", str(GA_SHARE), "--polish-sweeps", str(POLISH),
           "--seed", str(seed), "--out", str(out), *extra]
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    fh = open(log, "w")  # noqa: SIM115 - handed to a long-lived detached child
    # the SENTINEL: the shell that ran the work writes its own rc, so the only way to lose
    # the rc is to lose the work too.
    wrapped = ["bash", "-c",
               'set -o pipefail; "$@" ; echo $? > ' + str(rcf), "_"] + cmd
    return subprocess.Popen(wrapped, cwd=str(WORKTREE), stdout=fh,
                           stderr=subprocess.STDOUT, env=env)


def harvest(tag: str, seed: int, arm: str) -> dict:
    """Read the run's result, with rc taken from the SENTINEL file, never from a callback."""
    out = OUTDIR / f"{tag}.json"
    rcf = OUTDIR / f"{tag}.rc"
    row: dict = {"tag": tag, "arm": arm, "seed": seed,
                 "rc_sentinel_exists": rcf.exists(),
                 "rc": int(rcf.read_text().strip()) if rcf.exists() else None}
    if row["rc"] != 0 or not out.exists():
        row["ok"] = False
        row["why"] = f"rc={row['rc']} sentinel={rcf.exists()} json={out.exists()}"
        return row
    blob = json.load(open(out))
    achieved = blob["unique_evals"]
    keys = OUTDIR / f"{tag}.keys.npy"
    ckpt = OUTDIR / f"{tag}.ckpt.json"
    # TRIPLE reconciliation of unique_evals: run JSON == ckpt n_unique == keys.npy length.
    n_keys = int(np.load(keys).shape[0]) if keys.exists() else None
    n_ckpt = json.load(open(ckpt))["n_unique"] if ckpt.exists() else None
    row.update({
        "layout": blob["champion"]["layout"],
        "search_fitness": blob["champion"]["fitness"],
        "unique_evals_ACHIEVED": achieved,
        "budget_requested": blob["budget_requested"],
        "achieved_frac": achieved / blob["budget_requested"],
        "clears_floor": achieved >= ACHIEVED_FLOOR * blob["budget_requested"],
        "epochs_run": blob["epochs_run"],
        "elapsed_s": blob["elapsed_s"],
        "unique_evals_keys_npy": n_keys,
        "unique_evals_ckpt": n_ckpt,
        "unique_evals_triple_agree": (achieved == n_keys == n_ckpt),
        "keys_npy_retained": keys.exists(),
        "armh_constants_check": blob.get("armh_constants_check"),
        "armh_band": blob.get("armh_band"),
        "top50": blob["top50"],
        "ok": True,
    })
    return row


def wait_and_harvest(procs: dict, t0: float) -> list[dict]:
    rows = []
    for (tag, seed, arm), p in procs.items():
        p.wait()
        print(f"[{time.time() - t0:8.1f}s] {tag} seed={seed} waited", flush=True)
    for (tag, seed, arm) in procs:
        r = harvest(tag, seed, arm)
        rows.append(r)
        if r["ok"]:
            print(f"  {tag:22s} {r['layout']} fit={r['search_fitness']:.6f} "
                  f"uniq={r['unique_evals_ACHIEVED']:,} ({r['achieved_frac']:.1%})"
                  f"{'' if r['clears_floor'] else '  <<< BELOW 80% FLOOR'}"
                  f"{'' if r['unique_evals_triple_agree'] else '  <<< TRIPLE MISMATCH'}",
                  flush=True)
        else:
            print(f"  {tag:22s} FAILED {r['why']}", flush=True)
    return rows


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    per_epoch = int(BUDGET * OVERSHOOT) // (EPOCHS * ISLANDS)
    print(f"ARM H runner  phase={phase}  budget={BUDGET:,} islands={ISLANDS} "
          f"epochs={EPOCHS} => {per_epoch:,} calls/island/epoch", flush=True)
    print(f"seeds {AH.ARMH_SEEDS}  (formula {AH.ARMH_SEED_FORMULA})", flush=True)
    summary_path = OUTDIR / "armh-summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    # ---------------- PHASE 1: baseline control => sd_H ----------------
    if phase in ("all", "1"):
        procs = {}
        for r, seed in enumerate(AH.ARMH_SEEDS):
            procs[(f"baseline-r{r}", seed, "baseline")] = launch(
                f"baseline-r{r}", seed, "baseline", [])
        procs[("repro-armg-r0", REPRO_SEED, "baseline")] = launch(
            "repro-armg-r0", REPRO_SEED, "baseline", [])
        rows = wait_and_harvest(procs, t0)
        summary["phase1_baseline"] = rows
        prim = [x for x in rows if x["ok"] and x["clears_floor"]
                and x["tag"].startswith("baseline-r")]
        ms = [x["search_fitness"] for x in prim]
        sd_H = float(np.std(ms, ddof=1)) if len(ms) >= 2 else None
        summary["sd_H"] = sd_H
        summary["sd_H_pool"] = [x["tag"] for x in prim]
        summary["sd_H_ms_values"] = ms
        summary["sd_H_quadruple"] = {
            "POOL": "my 5 ARM-H-family baseline-control champions (near-optimal, cold start, blend-v1)",
            "REPLICATE_STRUCTURE": "independent cold-start search runs, one champion each",
            "SCALE": "raw ms/char",
            "STATISTIC": "sd, ddof=1, n=%d" % len(ms),
        }
        summary["eps"] = 2 * sd_H if sd_H else None
        summary["flip_threshold_sd"] = AH.ARMH_FLIP_SD
        summary["BALL1_in_band"] = (2 * sd_H >= AH.ARMH_BALL1_MS - AH.ARMH_REF_MS
                                   if sd_H else None)
        print(f"\n*** sd_H = {sd_H!r}   2*sd_H = {2 * sd_H if sd_H else None!r}", flush=True)
        print(f"*** flip threshold sd_H = {AH.ARMH_FLIP_SD!r}  => BALL-1 in band: "
              f"{summary['BALL1_in_band']}", flush=True)
        json.dump(summary, open(summary_path, "w"), indent=1)

    # ---------------- PHASE 2: ARM H cold + warm, at EPS = 2*sd_H ----------------
    if phase in ("all", "2"):
        eps = summary.get("eps")
        if eps is None:
            print("REFUSING: phase 2 needs sd_H from phase 1.", flush=True)
            return 1
        procs = {}
        for r, seed in enumerate(AH.ARMH_SEEDS):
            procs[(f"armh-cold-r{r}", seed, "armh")] = launch(
                f"armh-cold-r{r}", seed, "armh", ["--armh-eps", repr(eps)])
            procs[(f"armh-warm-r{r}", seed, "armh")] = launch(
                f"armh-warm-r{r}", seed, "armh", ["--armh-eps", repr(eps), "--armh-warm"])
        rows = wait_and_harvest(procs, t0)
        summary["phase2_armh"] = rows
        summary["eps_used"] = eps
        json.dump(summary, open(summary_path, "w"), indent=1)

    summary["wall_clock_s"] = time.time() - t0
    summary["config"] = {"budget": BUDGET, "islands": ISLANDS, "epochs": EPOCHS,
                         "overshoot": OVERSHOOT, "ga_share": GA_SHARE,
                         "polish_sweeps": POLISH, "achieved_floor": ACHIEVED_FLOOR,
                         "seed_formula": AH.ARMH_SEED_FORMULA,
                         "seeds": list(AH.ARMH_SEEDS), "repro_seed": REPRO_SEED}
    summary["prereg"] = "agent-artifacts/armh/PREREGISTRATION.md (committed 491138b)"
    summary["modelled_only"] = ("MODELLED ONLY: g-frame, baked 90 WPM, blend-v1, skipgrams "
                                "1-skip31, as-shipped NESTED bad_redirect oxey convention.")
    json.dump(summary, open(summary_path, "w"), indent=1)
    allrows = summary.get("phase1_baseline", []) + summary.get("phase2_armh", [])
    n_ok = sum(1 for x in allrows if x["ok"])
    print(f"\nWROTE {summary_path}  n_ok={n_ok}/{len(allrows)} "
          f"wall={summary['wall_clock_s']:.1f}s", flush=True)
    return 0 if n_ok == len(allrows) else 1


if __name__ == "__main__":
    sys.exit(main())
