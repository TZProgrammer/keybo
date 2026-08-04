"""GATEFOLDS-1 §4c — ALIGNMENT: is the model-free anchor number PREDICTIVE of the gate's refusals?

The last link in the chain. §4b showed the top bucket agrees WORST with the volume-weighted rest
(3/4 folds). This asks the sharper question: across all 12 high (fold, bucket) cells, do the
buckets that agree LESS with the rest of the range take a MORE NEGATIVE rho hit?

The two quantities are computed on independent paths, which is what makes the question non-circular:
  * anchor agreement -- OBSERVED durations only, leave-one-bucket-out, no model (anchor.py)
  * rho delta        -- a trained model's per-bucket Spearman minus the incumbent's (rows.json)

⚠ AND THE FLOOR IS REPORTED BEFORE THE STATISTIC, because a Spearman on 12 cells has a wide null
and quoting +0.36 alone would oversell it. The null is built by permuting the DELTA labels WITHIN
each fold -- which destroys the bucket->delta association while PRESERVING each fold's rho level, so
a fold-level offset (dvorak sits near 0.70, qwertz near 0.92) cannot manufacture the effect.

⚠ A CORRECTION TO MY OWN FIRST PASS, recorded rather than quietly fixed: my first alignment check
zero-filled the rho delta for buckets that were NOT refused, which biased the statistic upward
(+0.43/+0.48/+0.56). Using every bucket's REAL delta gives +0.36/+0.35/+0.31, and those are the
numbers that stand.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

from scipy.stats import spearmanr  # noqa: E402

HIGH = [80, 100, 120]
ALL = [40, 60, 80, 100, 120]
N_PERM = 20000
RNG = np.random.default_rng(20260804)

anchor = json.load(open(f"{ARTIFACTS}/anchor.json"))
rows = json.load(open(f"{ARTIFACTS}/rows.json"))

ARMS = ["CUR-INVARIANT", "CUR-NOWPM", "HYBRIDB", "INTERP", "INTERP-NOMONO", "CUR"]

out: dict = {
    "prereg": "agent-artifacts/gatefolds/GATEFOLDS-preregistration.md",
    "question": "do buckets that agree LESS with the volume-weighted rest take a MORE NEGATIVE rho hit?",
    "independence": (
        "anchor agreement is OBSERVED-only leave-one-bucket-out (no model); rho delta is a trained "
        "model's per-bucket Spearman minus the incumbent's. Different code paths, so the "
        "correlation is not an identity (prereg invariant 6)."
    ),
    "null": "permute rho-delta labels WITHIN each fold, preserving fold-level rho levels",
    "n_perm": N_PERM,
    "correction": (
        "my first pass zero-filled deltas for non-refused buckets, biasing the statistic up to "
        "+0.43/+0.48/+0.56; these numbers use every bucket's REAL delta"
    ),
    "arms": {},
}


def cells(arm, buckets):
    per_fold = []
    for fold, blk in sorted(anchor["folds"].items()):
        anch = {int(b): v["spearman_equal_n_mean"] for b, v in blk["buckets"].items()}
        dl = {int(b): v for b, v in rows["arms"][arm]["detail"][fold]["mean_delta_per_bucket"].items()}
        xs = [anch[b] for b in buckets if b in anch and b in dl]
        ys = [dl[b] for b in buckets if b in anch and b in dl]
        if xs:
            per_fold.append((np.array(xs), np.array(ys)))
    return per_fold


print()
print("=" * 104)
print("ALIGNMENT: Spearman(anchor agreement, rho delta) over the HIGH (fold,bucket) cells")
print("floor FIRST: the within-fold permutation null sd, then the statistic")
print("=" * 104)
for arm in ARMS:
    if arm not in rows["arms"]:
        continue
    for label, buckets in (("high", HIGH), ("all", ALL)):
        pf = cells(arm, buckets)
        X = np.concatenate([p[0] for p in pf])
        Y = np.concatenate([p[1] for p in pf])
        obs = float(spearmanr(X, Y).statistic)
        null = np.array(
            [
                spearmanr(X, np.concatenate([RNG.permutation(p[1]) for p in pf])).statistic
                for _ in range(N_PERM)
            ]
        )
        sd = float(null.std())
        p_one = float(np.mean(null >= obs))
        out["arms"].setdefault(arm, {})[label] = {
            "n_cells": int(len(X)),
            "spearman": obs,
            "null_sd": sd,
            "obs_over_null_sd": obs / sd if sd else None,
            "one_sided_p": p_one,
        }
        if label == "high":
            print(
                f"  {arm:<15} n={len(X):2}  null sd {sd:.4f}   observed {obs:+.4f}   "
                f"obs/null_sd {obs / sd:5.2f}   one-sided p {p_one:.4f}"
            )

print()
print("=" * 104)
print("WHAT THIS DOES AND DOES NOT ESTABLISH")
print("=" * 104)
hi = {a: v["high"] for a, v in out["arms"].items() if "high" in v}
strong = [a for a, v in hi.items() if v["obs_over_null_sd"] and v["obs_over_null_sd"] >= 2.0]
print(f"  sign is POSITIVE on {sum(1 for v in hi.values() if v['spearman'] > 0)}/{len(hi)} arms "
      f"(the direction the anchor mechanism predicts)")
print(f"  but only {len(strong)}/{len(hi)} arms reach 2 null-sd: {strong or 'NONE'}")
print("  => DIRECTIONALLY CONSISTENT, NOT ESTABLISHED. n=12 cells per arm is too few to settle it,")
print("     and I report it as SUGGESTIVE rather than as the mechanism's proof. The mechanism's")
print("     actual evidence is the CUR-INVARIANT control (4/4, a decisive design), not this rho.")
out["verdict"] = {
    "n_arms_positive_sign": int(sum(1 for v in hi.values() if v["spearman"] > 0)),
    "n_arms": len(hi),
    "arms_at_or_above_2_null_sd": strong,
    "reading": (
        "directionally consistent with the anchor mechanism on every arm, but only "
        f"{len(strong)} of {len(hi)} arms reach 2 null-sd at n=12 -- SUGGESTIVE, NOT ESTABLISHED"
    ),
}

with open(f"{ARTIFACTS}/align.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print()
print(f"wrote {ARTIFACTS}/align.json")
