"""GATEFOLDS-1 §4 — THE POSITIVE SIDE OF THE MECHANISM: does the TRUTH actually re-order with pace?

H1 establishes that a wpm-invariant frame emits ONE fixed pair-ranking for all buckets. That is a
statement about the MODEL. It only explains a HIGH-WPM-SPECIFIC failure if the OBSERVED ordering
re-orders more at high wpm than at low -- otherwise a single fixed ranking would be equally good
everywhere and the gate would fire uniformly, not at the top.

So this measures the DATA, with no model in the loop at all:

  For each fold (layout) and each pair of buckets, the Spearman correlation between the two
  buckets' OBSERVED cell durations, over the cells they have IN COMMON.

⚠ INVARIANT 3 (no self-generated targets) IS THE WHOLE DESIGN HERE. Every number is computed from
`Cell.obs` -- the IQR-mean of REAL recorded durations from the stroke TSV, built by
`keybo.training.validate.build_cells`. No model is trained, nothing is predicted, no
`TimeSurface`/`TableBigramScorer` is constructed. A "the truth re-orders" claim computed from a
frame's own predictions would be a tautology; this cannot be one, because no frame is involved.

⚠ INVARIANT 6 (no metric that is a function of its own outcome). The quantity here (agreement
between two buckets' OBSERVED orderings) is measured on data the gate's verdict does not enter. It
is not derived from any rho delta, any arm, or any model output.

⚠ AND THE CONFOUND THAT WOULD FAKE THIS RESULT, measured rather than assumed: high buckets have
FEWER cells, and a Spearman on fewer cells is noisier, which alone would make high-vs-high
agreement look lower. So the headline comparison is DOWN-SAMPLED TO A COMMON n: every bucket pair
is also evaluated on a random subsample of the size of the SMALLEST overlap, repeated, so the
sample-size difference cannot produce the effect. Both the raw and the equal-n numbers are
reported, and if they disagree the equal-n one is the answer.
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

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
BUCKETS = [40, 60, 80, 100, 120]
HIGH_FLOOR = 80
N_SUB = 200
RNG = np.random.default_rng(20260804)

log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
LAYOUTS = sorted({r.layout for r in rows})
log(f"{len(rows)} rows; layouts {LAYOUTS}")

out: dict = {
    "prereg": "agent-artifacts/gatefolds/GATEFOLDS-preregistration.md",
    "source": STROKES,
    "cell_config": CELL_KW,
    "buckets": BUCKETS,
    "high_wpm_floor": HIGH_FLOOR,
    "n_subsamples_for_equal_n": N_SUB,
    "note": "computed from OBSERVED Cell.obs only -- no model, no frame, no predicted target",
    "folds": {},
}


def obs_by_bucket(layout):
    """{bucket: {ngram-key: observed ms}} for one layout, from REAL durations only."""
    cells = build_cells([r for r in rows if r.layout == layout], **CELL_KW)
    d: dict[int, dict[tuple, float]] = {}
    part: dict[int, set] = {}
    for c in cells:
        d.setdefault(c.bucket, {})[(c.ngram, c.positions)] = float(c.obs)
        part.setdefault(c.bucket, set()).update(s[2] for s in c.samples)
    return d, {b: len(v) for b, v in part.items()}


for layout in LAYOUTS:
    D, PART = obs_by_bucket(layout)
    pairs: dict[str, dict] = {}
    for i, b1 in enumerate(BUCKETS):
        for b2 in BUCKETS[i + 1 :]:
            if b1 not in D or b2 not in D:
                continue
            keys = sorted(set(D[b1]) & set(D[b2]))
            if len(keys) < 5:
                continue
            v1 = np.array([D[b1][k] for k in keys])
            v2 = np.array([D[b2][k] for k in keys])
            rho = float(spearmanr(v1, v2).statistic)
            pairs[f"{b1}-{b2}"] = {
                "n_common_cells": len(keys),
                "spearman_observed": rho,
                "n_participants_b1": PART.get(b1),
                "n_participants_b2": PART.get(b2),
            }
    # EQUAL-n control: the smallest overlap in this fold, applied to every pair.
    n_common = [p["n_common_cells"] for p in pairs.values()]
    n_min = min(n_common) if n_common else 0
    for i, b1 in enumerate(BUCKETS):
        for b2 in BUCKETS[i + 1 :]:
            key = f"{b1}-{b2}"
            if key not in pairs:
                continue
            keys = sorted(set(D[b1]) & set(D[b2]))
            vals = []
            for _ in range(N_SUB):
                idx = RNG.choice(len(keys), size=n_min, replace=False)
                sel = [keys[j] for j in idx]
                vals.append(
                    spearmanr(
                        np.array([D[b1][k] for k in sel]), np.array([D[b2][k] for k in sel])
                    ).statistic
                )
            vals = [v for v in vals if np.isfinite(v)]
            pairs[key]["spearman_equal_n_mean"] = float(np.mean(vals)) if vals else float("nan")
            pairs[key]["spearman_equal_n_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            pairs[key]["equal_n"] = n_min

    # The two populations the mechanism claim needs separated:
    #   LOW-anchored  = pairs where BOTH buckets are below the gate's floor
    #   HIGH-anchored = pairs where BOTH buckets are at/above the floor
    lo = [p for k, p in pairs.items() if all(int(x) < HIGH_FLOOR for x in k.split("-"))]
    hi = [p for k, p in pairs.items() if all(int(x) >= HIGH_FLOOR for x in k.split("-"))]
    out["folds"][layout] = {
        "n_cells_per_bucket": {str(b): len(D.get(b, {})) for b in BUCKETS},
        "n_participants_per_bucket": {str(b): PART.get(b) for b in BUCKETS},
        "bucket_pairs": pairs,
        "low_anchored_mean_equal_n": (
            float(np.mean([p["spearman_equal_n_mean"] for p in lo])) if lo else None
        ),
        "high_anchored_mean_equal_n": (
            float(np.mean([p["spearman_equal_n_mean"] for p in hi])) if hi else None
        ),
        "low_anchored_mean_raw": float(np.mean([p["spearman_observed"] for p in lo])) if lo else None,
        "high_anchored_mean_raw": float(np.mean([p["spearman_observed"] for p in hi])) if hi else None,
    }

print()
print("=" * 104)
print("OBSERVED cross-bucket agreement (Spearman of two buckets' OBSERVED durations, common cells)")
print("no model, no frame, no predicted target -- invariant 3 by construction")
print("=" * 104)
for layout, blk in sorted(out["folds"].items()):
    print()
    print(f"  {layout}   cells/bucket " + " ".join(
        f"b{b}:{blk['n_cells_per_bucket'][str(b)]}" for b in BUCKETS))
    print(f"  {'':<9} participants " + " ".join(
        f"b{b}:{blk['n_participants_per_bucket'][str(b)]}" for b in BUCKETS))
    for k, p in blk["bucket_pairs"].items():
        eq = p.get("spearman_equal_n_mean")
        print(
            f"      {k:<8} n_common {p['n_common_cells']:<5} "
            f"rho_obs {p['spearman_observed']:+.4f}   "
            f"equal-n({p.get('equal_n')}) {eq:+.4f}" + (f" +- {p['spearman_equal_n_sd']:.4f}"
                                                        if p.get("spearman_equal_n_sd") else "")
        )
    print(
        f"      => LOW-anchored (both <{HIGH_FLOOR}) mean equal-n "
        f"{blk['low_anchored_mean_equal_n']}"
    )
    print(
        f"      => HIGH-anchored (both >={HIGH_FLOOR}) mean equal-n "
        f"{blk['high_anchored_mean_equal_n']}"
    )

# The cross-fold summary: is high-anchored agreement LOWER than low-anchored, and by how much?
lo_all = [b["low_anchored_mean_equal_n"] for b in out["folds"].values()
          if b["low_anchored_mean_equal_n"] is not None]
hi_all = [b["high_anchored_mean_equal_n"] for b in out["folds"].values()
          if b["high_anchored_mean_equal_n"] is not None]
out["summary"] = {
    "low_anchored_mean_equal_n_over_folds": float(np.mean(lo_all)) if lo_all else None,
    "high_anchored_mean_equal_n_over_folds": float(np.mean(hi_all)) if hi_all else None,
    "delta_high_minus_low": (
        float(np.mean(hi_all) - np.mean(lo_all)) if lo_all and hi_all else None
    ),
    "n_folds_high_below_low": sum(
        1 for b in out["folds"].values()
        if b["high_anchored_mean_equal_n"] is not None
        and b["low_anchored_mean_equal_n"] is not None
        and b["high_anchored_mean_equal_n"] < b["low_anchored_mean_equal_n"]
    ),
    "n_folds": len(out["folds"]),
}
print()
print("=" * 104)
print("SUMMARY (equal-n, so the difference cannot be a cell-count artifact)")
print("=" * 104)
s = out["summary"]
print(f"  LOW-anchored  bucket-pair agreement, mean over folds: {s['low_anchored_mean_equal_n_over_folds']}")
print(f"  HIGH-anchored bucket-pair agreement, mean over folds: {s['high_anchored_mean_equal_n_over_folds']}")
print(f"  delta (high - low): {s['delta_high_minus_low']}")
print(f"  folds where HIGH agreement < LOW agreement: {s['n_folds_high_below_low']}/{s['n_folds']}")
print()
print("  READ: if HIGH-anchored agreement is LOWER, the observed pair-ordering genuinely CHANGES")
print("  more across the fast buckets -- so ONE fixed ranking (all a wpm-invariant frame can emit)")
print("  must lose the most exactly there. If it is NOT lower, H1 explains the invariance but NOT")
print("  the high-wpm SPECIFICITY, and I must say so.")

with open(f"{ARTIFACTS}/reorder.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/reorder.json")
