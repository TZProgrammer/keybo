"""GATEFOLDS-1 §4b — WHY HIGH WPM SPECIFICALLY: the ANCHOR argument, after my §4a metric failed.

⚠ THIS FILE EXISTS BECAUSE MY FIRST MECHANISM METRIC WAS CONFOUNDED AND I AM NOT KEEPING IT.
`reorder.py` compared "LOW-anchored" bucket pairs (both buckets < 80) against "HIGH-anchored" ones
(both >= 80) and found high agreement 0.0297 LOWER, 3/4 folds. That comparison is NOT
distance-matched: with five buckets the LOW set contains exactly ONE pair (40-60, adjacent) while
the HIGH set contains THREE (80-100 and 100-120 adjacent, 80-120 two apart). And agreement decays
steeply with bucket distance -- measured +0.8686 / +0.8107 / +0.7436 / +0.7001 at 1/2/3/4 buckets
apart. So most of that -0.0297 is the extra distance, not anything about speed.
DISTANCE-MATCHED (the adjacent pair at each end, 40-60 vs 100-120) the effect DISAPPEARS: azerty
0.8532 -> 0.8292 and qwertz 0.9125 -> 0.7911 (high worse) but dvorak 0.7889 -> 0.8622 and qwerty
0.9611 -> 0.9688 (high BETTER). 2/4 each way = no effect.
⇒ H1's invariance does NOT get to claim the high-wpm SPECIFICITY from that number, and the
registered honest reading is that the specificity needs a different explanation.

THE CORRECT ARGUMENT, and it follows from the distance decay rather than fighting it:

A wpm-invariant frame emits ONE ranking. Training minimizes loss over ALL buckets at once, weighted
by DATA VOLUME -- and the volume is concentrated at the BOTTOM of the range (azerty b40 has 28,713
raw samples vs b120's 1,340; qwerty b40 43,467 participants vs b120's 10,811). So the single
ranking is anchored near the low/mid buckets. Combined with the measured monotone distance decay,
the fixed ranking is then FURTHEST from the truth at the TOP of the range -- which is exactly where
the gate looks. The specificity comes from the ASYMMETRY OF THE DATA MASS, not from high buckets
being internally more chaotic.

THE MEASUREMENT (model-free, and LEAVE-ONE-BUCKET-OUT so it cannot be a tautology):

  For each fold and each bucket b: build the VOLUME-WEIGHTED pooled observed ordering from ALL
  BUCKETS EXCEPT b, then correlate it with b's own observed ordering.

⚠ THE LEAVE-ONE-OUT IS LOAD-BEARING. Pooling ALL buckets and correlating with bucket b would
include b's own observations in its own reference -- and since the pooling is volume-weighted, a
high-volume bucket would score high partly by predicting itself. That is precisely the
self-generated-target defect prereg invariant 3 forbids. Excluding b makes the reference an honest
out-of-bucket predictor, which is also what the trained model faces.

⚠ INVARIANT 3: every number comes from `Cell.obs` (IQR-mean of REAL durations). No model is
trained, nothing is predicted, no TimeSurface/TableBigramScorer is built.
⚠ INVARIANT 6: the quantity is agreement between two OBSERVED orderings. No rho delta, arm verdict
or model output enters it, so it cannot be an algebraic function of the outcome it explains.
⚠ EQUAL-n: the buckets differ in cell count, so the headline is also computed at a common n.

REGISTERED READING, stated before the numbers print:
  * If agreement with the out-of-bucket volume-weighted reference DECLINES toward high wpm, the
    anchor argument holds and the high-wpm specificity is explained.
  * If it is FLAT or rises, it does not, and I report that H1 explains the 4/4 refusal but NOT the
    high-wpm specificity -- leaving the specificity OPEN rather than narrated.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, STROKES, assert_tree, require  # noqa: E402

assert_tree()

from scipy.stats import spearmanr  # noqa: E402

import keybo.training.validate as V  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402

build_cells = require(V, "build_cells")

CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
BUCKETS = [40, 60, 80, 100, 120]
N_SUB = 200
RNG = np.random.default_rng(20260804)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
LAYOUTS = sorted({r.layout for r in rows})
log(f"{len(rows)} rows; layouts {LAYOUTS}")

out: dict = {
    "prereg": "agent-artifacts/gatefolds/GATEFOLDS-preregistration.md",
    "supersedes": "reorder.py's LOW-vs-HIGH summary, which was not distance-matched (see docstring)",
    "source": STROKES,
    "cell_config": CELL_KW,
    "method": "leave-one-bucket-out volume-weighted pooled OBSERVED ordering vs the bucket's own",
    "folds": {},
}

for layout in LAYOUTS:
    cells = build_cells([r for r in rows if r.layout == layout], **CELL_KW)
    obs: dict[int, dict[tuple, float]] = {}
    vol: dict[int, dict[tuple, int]] = {}
    for c in cells:
        k = (c.ngram, c.positions)
        obs.setdefault(c.bucket, {})[k] = float(c.obs)
        vol.setdefault(c.bucket, {})[k] = int(c.n)

    raw_vol = {b: int(sum(vol[b].values())) for b in sorted(vol)}
    per_bucket = {}
    for b in BUCKETS:
        if b not in obs:
            continue
        others = [o for o in BUCKETS if o != b and o in obs]
        # VOLUME-WEIGHTED pooled reference from every OTHER bucket. Weight = that bucket's raw
        # sample count for the cell, which is what makes the reference reflect where the training
        # loss actually has mass.
        num: dict[tuple, float] = {}
        den: dict[tuple, float] = {}
        for o in others:
            for k, v in obs[o].items():
                w = float(vol[o][k])
                num[k] = num.get(k, 0.0) + w * v
                den[k] = den.get(k, 0.0) + w
        ref = {k: num[k] / den[k] for k in num if den[k] > 0}
        keys = sorted(set(ref) & set(obs[b]))
        if len(keys) < 5:
            continue
        r = float(spearmanr(np.array([ref[k] for k in keys]),
                            np.array([obs[b][k] for k in keys])).statistic)
        per_bucket[b] = {
            "n_common_cells": len(keys),
            "spearman_vs_leave_one_out_reference": r,
            "raw_samples_in_bucket": raw_vol.get(b),
            "share_of_fold_raw_samples": raw_vol.get(b, 0) / sum(raw_vol.values()),
        }
    # EQUAL-n at the smallest common set in this fold.
    n_min = min(p["n_common_cells"] for p in per_bucket.values()) if per_bucket else 0
    for b, p in per_bucket.items():
        others = [o for o in BUCKETS if o != b and o in obs]
        num, den = {}, {}
        for o in others:
            for k, v in obs[o].items():
                w = float(vol[o][k])
                num[k] = num.get(k, 0.0) + w * v
                den[k] = den.get(k, 0.0) + w
        ref = {k: num[k] / den[k] for k in num if den[k] > 0}
        keys = sorted(set(ref) & set(obs[b]))
        vals = []
        for _ in range(N_SUB):
            idx = RNG.choice(len(keys), size=n_min, replace=False)
            sel = [keys[j] for j in idx]
            v = spearmanr(np.array([ref[k] for k in sel]), np.array([obs[b][k] for k in sel])).statistic
            if np.isfinite(v):
                vals.append(v)
        p["spearman_equal_n_mean"] = float(np.mean(vals)) if vals else float("nan")
        p["spearman_equal_n_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        p["equal_n"] = n_min
    out["folds"][layout] = {"buckets": {str(b): p for b, p in sorted(per_bucket.items())}}

print()
print("=" * 104)
print("AGREEMENT of each bucket's OBSERVED ordering with the VOLUME-WEIGHTED ordering of the")
print("OTHER buckets (leave-one-bucket-out, so no bucket helps predict itself)")
print("=" * 104)
print(f"  {'fold':<9}" + "".join(f"{'b'+str(b):>18}" for b in BUCKETS))
for layout, blk in sorted(out["folds"].items()):
    cells_row, share_row = [], []
    for b in BUCKETS:
        p = blk["buckets"].get(str(b))
        cells_row.append(f"{p['spearman_equal_n_mean']:+.4f}({p['equal_n']:>4})" if p else f"{'n/a':>18}")
        share_row.append(f"{p['share_of_fold_raw_samples']*100:5.1f}%" if p else "   -- ")
    print(f"  {layout:<9}" + "".join(f"{c:>18}" for c in cells_row))
    print(f"  {'':<9}" + "".join(f"{s:>18}" for s in share_row) + "   <- share of fold raw samples")

# The registered read: monotone decline toward high wpm?
print()
print("=" * 104)
print("VERDICT ON THE ANCHOR ARGUMENT")
print("=" * 104)
decl = {}
for layout, blk in sorted(out["folds"].items()):
    seq = [blk["buckets"][str(b)]["spearman_equal_n_mean"] for b in BUCKETS if str(b) in blk["buckets"]]
    bs = [b for b in BUCKETS if str(b) in blk["buckets"]]
    mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    top_vs_best = seq[-1] - max(seq)
    decl[layout] = {
        "sequence": [float(x) for x in seq],
        "buckets": bs,
        "monotone_declining": bool(mono),
        "top_bucket_minus_best": float(top_vs_best),
        "top_bucket_is_worst": bool(seq[-1] == min(seq)),
    }
    print(f"  {layout:<9} " + " -> ".join(f"{x:+.4f}" for x in seq)
          + f"   monotone-declining={mono}  top-is-worst={seq[-1] == min(seq)}"
          + f"  top-best={top_vs_best:+.4f}")
out["anchor_verdict"] = decl
n_top_worst = sum(1 for v in decl.values() if v["top_bucket_is_worst"])
out["n_folds_top_bucket_is_worst"] = n_top_worst
out["n_folds"] = len(decl)
print()
print(f"  folds where the TOP bucket has the WORST agreement with the rest: {n_top_worst}/{len(decl)}")
print("  READ: top-is-worst on most folds supports the anchor argument (a single volume-anchored")
print("  ranking is furthest from the truth at the top). Otherwise the high-wpm SPECIFICITY stays")
print("  OPEN and I say so rather than narrate it.")

with open(f"{ARTIFACTS}/anchor.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/anchor.json")
