"""GATEFOLDS-1 §6 — THE CORRECT FLOOR, MEASURED BY ME, AFTER MY FIRST FLOOR TURNED OUT TAUTOLOGICAL.

⚠ WHY THIS FILE EXISTS. My §3 floor measured the WITHIN-arm spread of an arm's per-bucket rho
around THAT ARM'S OWN MEAN (median sd 0.00244). The sibling `gatewhy` (landed at origin/main
d2a60bc) showed that the gate's whole baseline construction is a TAUTOLOGY: the incumbent baseline
is CUR's MEAN over the SAME seeds being scored, so the incumbent's per-cell deltas necessarily SUM
TO ZERO and can never ALL be negative -- while a structural refusal requires all three negative.
I REPRODUCED IT ON MY OWN ARTIFACT: max |sum of deltas| = 3.331e-16 over all 20 cells, and 0/20
cells have all three seed deltas negative. So both my "the gate passes the incumbent" control AND
my floor were measuring the same self-referential zero.

THE CORRECT FLOOR compares TWO INDEPENDENT TRAINING RUNS of the SAME frame: nothing changed but the
seeds. That is a RESEED arm, and it is the only floor that answers the question a candidate's Δrho
actually poses -- "is this bigger than what changing nothing produces?"

  CUR-RESEED   the served frame, seeds [3, 4, 5], scored against the REGISTERED CUR(0,1,2) baseline

This is a NEGATIVE CONTROL THAT CAN FAIL, which the tautological one could not. Registered reading,
written before the numbers exist:

  * If CUR-RESEED is structurally refused on any fold, then a refusal can be manufactured by
    changing NOTHING but the seed, and NO Δrho at that magnitude in this campaign -- mine included
    -- can be attributed to a frame. My §3 margin-vs-floor claims would then be WRONG AS WRITTEN
    and I say so.
  * The floor itself is the DISTRIBUTION of |Δrho| between two same-frame runs, reported as p50/p95
    and as the share of cells already exceeding the gate's 0.005 tolerance. Every structural row I
    reported gets re-read against THAT number rather than against the within-arm sd.

⚠ This does NOT re-adjudicate anything by fiat and it does NOT weaken the gate: it measures the
gate's own noise floor and reports which of MY OWN claims survive it. Choosing what to do about a
gate whose floor exceeds its tolerance is the human's call (GATESUPPORT-1 precedent).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, SCRATCH, STROKES, assert_tree, require  # noqa: E402

assert_tree()

import keybo.verdicts as VD  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

bucket_regression_report = require(VD, "bucket_regression_report")
TOL = float(require(VD, "HIGH_WPM_TOLERANCE"))
FLOOR_WPM = int(require(VD, "HIGH_WPM_FLOOR"))

RESEEDS = [3, 4, 5]
SENTINEL = f"{SCRATCH}/reseed.sentinel"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows")
GEO = ROW_STAGGERED_31

# --- FIRST: reproduce the tautology on the already-computed CUR arm ------------------------
# Done here, in the artifact, rather than only in a shell one-liner -- a claim this consequential
# should be reproducible from the tree.
control = json.load(open(f"{ARTIFACTS}/control.json"))
cur = control["arms"]["CUR"]
taut = {"cells": {}, "max_abs_sum_of_deltas": 0.0, "n_cells_all_seeds_negative": 0, "n_cells": 0}
for h, f in sorted(cur["folds"].items()):
    per: dict[int, list[float]] = {}
    for rec in f["seeds"]:
        for b, v in rec["bucket_rhos"].items():
            per.setdefault(int(b), []).append(float(v))
    for b, vals in sorted(per.items()):
        base = float(np.mean(vals))
        deltas = [v - base for v in vals]
        taut["cells"][f"{h}/{b}"] = {
            "deltas": deltas,
            "abs_sum": abs(float(sum(deltas))),
            "all_negative": bool(all(x < 0 for x in deltas)),
        }
        taut["max_abs_sum_of_deltas"] = max(taut["max_abs_sum_of_deltas"], abs(float(sum(deltas))))
        taut["n_cells_all_seeds_negative"] += int(all(x < 0 for x in deltas))
        taut["n_cells"] += 1
print()
print("=" * 100)
print("TAUTOLOGY CHECK on the gate's OWN incumbent control (reproducing gatewhy's finding)")
print("=" * 100)
print(f"  max |sum of the incumbent's 3 seed deltas| over {taut['n_cells']} cells: "
      f"{taut['max_abs_sum_of_deltas']:.3e}")
print(f"  cells where ALL THREE seed deltas are negative: "
      f"{taut['n_cells_all_seeds_negative']}/{taut['n_cells']}")
print("  => the incumbent CANNOT be structurally refused against its own mean. The control that")
print("     both INTERPFRAME-1 and I ran licensed nothing. CONFIRMED, independently.")

# --- THEN: the floor that CAN fail --------------------------------------------------------
log(f"ARM CUR-RESEED: validate() 4 folds x seeds {RESEEDS} (served frame, nothing else changed)")
reseed = validate(
    rows,
    seeds=RESEEDS,
    ngram="bigram",
    n_boot=10,
    geometry=GEO,
    train_params={"n_jobs": 8},
)
log("ARM CUR-RESEED: done")

out: dict = {
    "prereg": "agent-artifacts/gatefolds/GATEFOLDS-preregistration.md",
    "purpose": "the floor that CAN fail: two INDEPENDENT runs of the SAME frame",
    "tautology_check": taut,
    "reseeds": RESEEDS,
    "gate": {"tolerance": TOL, "floor_wpm": FLOOR_WPM},
    "arm": reseed,
}

# The registered CUR(0,1,2) baseline, per fold -- the same construction highwpm.py uses.
baseline = {}
for h, f in cur["folds"].items():
    acc: dict[int, list[float]] = {}
    for rec in f["seeds"]:
        for b, v in rec["bucket_rhos"].items():
            acc.setdefault(int(b), []).append(float(v))
    baseline[h] = {b: float(np.mean(v)) for b, v in sorted(acc.items())}

detail, all_abs, hi_abs = {}, [], []
for h, f in sorted(reseed["folds"].items()):
    hits: dict[int, int] = {}
    dl: dict[int, list[float]] = {}
    n = len(f["seeds"])
    for rec in f["seeds"]:
        blk = bucket_regression_report(
            {int(k): v for k, v in rec["bucket_rhos"].items()},
            baseline.get(h, {}),
            f"CUR-RESEED/{h}/seed{rec['seed']}",
        )
        for b in blk.get("regressing_high_buckets", []):
            hits[int(b)] = hits.get(int(b), 0) + 1
        for b, d in blk.get("deltas", {}).items():
            dl.setdefault(int(b), []).append(float(d))
            all_abs.append(abs(float(d)))
            if int(b) >= FLOOR_WPM:
                hi_abs.append(abs(float(d)))
    detail[h] = {
        "n_seeds": n,
        "structural": sorted(b for b, k in hits.items() if k == n),
        "noise": sorted(b for b, k in hits.items() if 0 < k < n),
        "per_bucket_seed_counts": {str(k): v for k, v in sorted(hits.items())},
        "mean_delta_per_bucket": {str(b): float(np.mean(v)) for b, v in sorted(dl.items())},
    }

structural = {h: d["structural"] for h, d in detail.items() if d["structural"]}
out["reseed_verdict"] = {
    "passed": not structural,
    "structural_regressions": structural,
    "n_folds_structural": len(structural),
    "detail": detail,
}
out["measured_reseed_floor"] = {
    "n_high_cells": len(hi_abs),
    "high_abs_delta_p50": float(np.percentile(hi_abs, 50)) if hi_abs else None,
    "high_abs_delta_p95": float(np.percentile(hi_abs, 95)) if hi_abs else None,
    "high_abs_delta_max": float(np.max(hi_abs)) if hi_abs else None,
    "share_high_cells_exceeding_tolerance": (
        float(np.mean([x > TOL for x in hi_abs])) if hi_abs else None
    ),
    "all_buckets_abs_delta_p95": float(np.percentile(all_abs, 95)) if all_abs else None,
}

print()
print("=" * 100)
print("CUR-RESEED: the served frame, seeds [3,4,5], vs the REGISTERED CUR(0,1,2) baseline")
print("=" * 100)
print(f"  verdict: {'PASS' if not structural else f'STRUCTURAL on {len(structural)}/4 folds -> {structural}'}")
for h, d in sorted(detail.items()):
    hi = {b: v for b, v in d["mean_delta_per_bucket"].items() if int(b) >= FLOOR_WPM}
    print(f"    {h:<8} structural {d['structural']}  noise {d['noise']}   high-bucket mean deltas "
          + "  ".join(f"b{b}:{v:+.4f}" for b, v in sorted(hi.items(), key=lambda kv: int(kv[0]))))
m = out["measured_reseed_floor"]
print()
print(f"  MEASURED RESEED FLOOR on the {m['n_high_cells']} high (fold,bucket,seed) cells:")
print(f"    |delta rho|  p50 {m['high_abs_delta_p50']:.4f}   p95 {m['high_abs_delta_p95']:.4f}   "
      f"max {m['high_abs_delta_max']:.4f}")
print(f"    share of high cells already exceeding the gate's {TOL} tolerance: "
      f"{m['share_high_cells_exceeding_tolerance']*100:.1f}%")
print()
print("  READ: this is what CHANGING NOTHING BUT THE SEED produces. Any of my structural rows whose")
print("  |delta rho| sits below this p95 is NOT attributable to the frame, and I must say so.")

with open(f"{ARTIFACTS}/reseed.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/reseed.json")
os.makedirs(SCRATCH, exist_ok=True)
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
log(f"SENTINEL {SENTINEL}")
