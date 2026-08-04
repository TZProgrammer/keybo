"""GATEFOLDS-1 §1 — H1's STRUCTURAL claim, proved on the featurizers and then on the PREDICTIONS.

Two separate proofs, deliberately on DIFFERENT code paths so neither is an algebraic function of
the other (prereg invariant 6):

  A. FEATURIZER invariance. For a fixed position pair, is the feature vector bit-identical across
     the whole wpm range? Enumerated over ALL 961 position pairs of the 31-key geometry x the five
     bucket midpoints the gate actually uses, per frame. Path: keybo.features.ngram.
  B. PREDICTION invariance. Train ONE real model per frame and ask whether its RAW LOGRAT output
     varies with the bucket, and whether the ms output's WITHIN-BUCKET RANKING varies. Path:
     keybo.training.train + keybo.models.base.

B is the one that matters for the gate, because the gate scores a within-bucket Spearman. A is the
mechanism; B is the consequence. If A says "invariant" and B says "the within-bucket ranking is
IDENTICAL across buckets", then a wpm-invariant frame provably cannot track any cross-pace
re-ordering, and the high-wpm refusal is a statement about pace adaptation, not interpretability.

⚠ The RANK check is the honest one. `to_ms` multiplies by 12000/wpm, so the ms VALUES differ across
buckets even for an invariant frame -- a naive value comparison would report "not invariant" and
miss the point entirely. Spearman is invariant to that positive rescale, which is exactly why the
gate cannot see the pace factor either.
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

import keybo.features.ngram as NG  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

# Brief-decay defence: assert every symbol I lean on exists on THIS tree.
replacement_frame = require(NG, "replacement_frame")

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


GEO = ROW_STAGGERED_31
# The bucket MIDPOINTS validate() feeds the model: bucket start + bucket_width/2, for
# wpm_lo=40, wpm_hi=140, bucket_width=20 (the registered config of every arm on this line).
BUCKETS = [40, 60, 80, 100, 120]
MIDPOINTS = [b + 10.0 for b in BUCKETS]

KEYS = [(x, y) for x in list(range(-6, 0)) + list(range(1, 7)) for y in (-1, 0, 1)]
PAIRS = [(a, b) for a in KEYS for b in KEYS]

out: dict = {
    "geometry": "ROW_STAGGERED_31",
    "buckets": BUCKETS,
    "bucket_midpoints": MIDPOINTS,
    "n_pairs_enumerated": len(PAIRS),
    "A_featurizer": {},
    "B_prediction": {},
}

# =========================================================================================
# A. FEATURIZER INVARIANCE — enumerated over every position pair x every bucket midpoint
# =========================================================================================
print()
print("=" * 92)
print("A. FEATURIZER INVARIANCE across the gate's five wpm buckets, over ALL position pairs")
print("=" * 92)


def builder_for(flag):
    """(builder, names) for a frame flag; `False` means the served frame."""
    if flag is False:
        return (
            lambda g, p, wpm: bigram_features_from_positions(g, p, wpm=wpm),
            list(require(__import__("keybo.features.schema", fromlist=["x"]),
                         "BIGRAM_FEATURE_NAMES")),
        )
    b, names, _mono, _stamp, _tag = replacement_frame(flag)
    return (lambda g, p, wpm: b(g, p, wpm=wpm)), list(names)


FRAMES = [
    ("served", False),
    ("interp.1", True),
    ("interp-wpm", "wpm"),
    ("hybridb", "hybridb"),
]

for label, flag in FRAMES:
    build, names = builder_for(flag)
    n_invariant = 0
    max_abs_spread = 0.0
    varying_cols: set[str] = set()
    for pair in PAIRS:
        mats = np.vstack([np.asarray(build(GEO, pair, m), dtype=np.float64) for m in MIDPOINTS])
        spread = mats.max(axis=0) - mats.min(axis=0)
        if not spread.any():
            n_invariant += 1
        else:
            max_abs_spread = max(max_abs_spread, float(spread.max()))
            varying_cols.update(names[i] for i in np.nonzero(spread)[0])
    frac = n_invariant / len(PAIRS)
    out["A_featurizer"][label] = {
        "n_columns": len(names),
        "has_wpm_column": "wpm" in names,
        "n_pairs_invariant_across_all_buckets": n_invariant,
        "frac_pairs_invariant": frac,
        "max_abs_spread_over_buckets": max_abs_spread,
        "columns_that_vary_with_bucket": sorted(varying_cols),
    }
    print(
        f"  {label:<12} {len(names):2}c  has_wpm={str('wpm' in names):<5} "
        f"invariant pairs {n_invariant}/{len(PAIRS)} ({frac:.4f})  "
        f"varying cols: {sorted(varying_cols) or 'NONE'}"
    )

# =========================================================================================
# B. PREDICTION INVARIANCE — one real trained model per frame, on real rows
# =========================================================================================
print()
print("=" * 92)
print("B. PREDICTION INVARIANCE — raw LOGRAT spread and WITHIN-BUCKET RANK identity")
print("=" * 92)
log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")

# A fixed, frame-independent evaluation set: every position pair that appears in the data, so the
# comparison is on the surface the gate actually scores rather than on the full enumeration.
EVAL_PAIRS = sorted({tuple(r.positions) for r in rows})
log(f"{len(EVAL_PAIRS)} distinct position pairs present in the data")

for label, flag in FRAMES:
    log(f"ARM {label}: train_bigram_model")
    kw = {"interp": flag} if flag is not False else {}
    model = train_bigram_model(
        rows,
        target_wpm=90.0,
        geometry=GEO,
        random_state=0,
        n_jobs=8,
        **kw,
    )
    names = list(model.metadata.feature_names)
    has_wpm = "wpm" in names
    build, _ = builder_for(flag)

    raw_by_bucket = {}
    ms_by_bucket = {}
    for bucket, mid in zip(BUCKETS, MIDPOINTS, strict=True):
        X = np.vstack([np.asarray(build(GEO, p, mid), dtype=np.float64) for p in EVAL_PAIRS])
        assert X.shape[1] == len(names), f"{label}: {X.shape[1]} cols vs model's {len(names)}"
        raw = model.predict(X)
        # to_ms REFUSES an explicit wpm when the frame carries the column, and REQUIRES it when
        # it does not -- exactly the `needs_wpm` branch _predict_cells uses.
        ms = model.to_ms(raw, X, None if has_wpm else np.full(len(EVAL_PAIRS), mid))
        raw_by_bucket[bucket] = np.asarray(raw, dtype=np.float64)
        ms_by_bucket[bucket] = np.asarray(ms, dtype=np.float64)

    R = np.vstack([raw_by_bucket[b] for b in BUCKETS])
    raw_spread = float((R.max(axis=0) - R.min(axis=0)).max())

    # THE DECISIVE NUMBER: is the WITHIN-BUCKET ORDERING of pairs the same in every bucket?
    # argsort ties are broken deterministically, so compare the rank VECTORS directly.
    ranks = {b: np.argsort(np.argsort(ms_by_bucket[b])) for b in BUCKETS}
    ref = ranks[BUCKETS[0]]
    identical = {b: bool(np.array_equal(ranks[b], ref)) for b in BUCKETS}
    n_ident = sum(identical.values())
    # And the rank CORRELATION between the slowest and fastest bucket, as a continuous read.
    from scipy.stats import spearmanr

    rho_lo_hi = float(spearmanr(ms_by_bucket[BUCKETS[0]], ms_by_bucket[BUCKETS[-1]]).statistic)

    out["B_prediction"][label] = {
        "n_columns": len(names),
        "has_wpm_column": has_wpm,
        "feature_version": model.metadata.feature_version,
        "monotone_constraints": ((model.metadata.extra.get("training") or {}).get("frame") or {}),
        "n_eval_pairs": len(EVAL_PAIRS),
        "max_raw_lograt_spread_over_buckets": raw_spread,
        "within_bucket_rank_identical_to_b40": identical,
        "n_buckets_rank_identical": n_ident,
        "spearman_b40_vs_b120_ms": rho_lo_hi,
        "ms_mean_per_bucket": {str(b): float(ms_by_bucket[b].mean()) for b in BUCKETS},
    }
    print(
        f"  {label:<12} {len(names):2}c has_wpm={str(has_wpm):<5} "
        f"raw LOGRAT spread over buckets {raw_spread:.3e}  "
        f"rank-identical buckets {n_ident}/{len(BUCKETS)}  "
        f"rho(b40 ms, b120 ms) = {rho_lo_hi:.6f}"
    )
    print(f"               ms mean/bucket: " + "  ".join(
        f"b{b}:{ms_by_bucket[b].mean():.2f}" for b in BUCKETS))

with open(f"{ARTIFACTS}/invariance.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/invariance.json")

print()
print("=" * 92)
print("WHAT THIS DECIDES")
print("=" * 92)
inv = out["A_featurizer"]
prd = out["B_prediction"]
for label, _ in FRAMES:
    a = inv[label]
    b = prd[label]
    verdict = (
        "WPM-INVARIANT (cannot re-rank across pace)"
        if a["frac_pairs_invariant"] == 1.0 and b["n_buckets_rank_identical"] == len(BUCKETS)
        else "pace-adaptive"
    )
    print(f"  {label:<12} {verdict}")
