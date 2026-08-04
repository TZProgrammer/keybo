"""INTERPFRAME-1 — WHY does the 10-column frame lose MAGNITUDE accuracy? (the third hypothesis)

Two mechanism claims of mine are now REFUTED by measurement, and this driver exists because of
the second one:

  H1 "conditioning on same-hand makes travel monotone"  -> REFUTED (mechanism.py: a two-level
     Simpson's paradox; conditioning on same-hand alone leaves rho at -0.3490).
  H2 "the wmae cost is the dropped wpm column"           -> REFUTED (variant.py: restoring wpm
     changes wmae by +0.005 and does NOT recover the high-wpm gate).

H2 was inferred from CUR-NOWPM-fixed (+6.074 wmae vs CUR) sitting near INTERP (+5.765). That
inference was WRONG: ablating wpm from the SERVED frame costs ~6 ms, but ADDING it to the interp
frame buys ~0. Two different frames, two different roles for the same column — a coincidence of
magnitudes, not a shared cause.

H3, measured here: the interp frame COLLAPSES cells the served frame separates. 10 ordinal columns
cannot index as many distinct geometry cells as 20 columns containing one-hots, so distinct
position pairs become featurewise IDENTICAL and MUST receive the same prediction. That caps
MAGNITUDE accuracy (a collapsed group can only be predicted at its group mean) while leaving RANK
accuracy largely intact (the groups are still ordered correctly) — which is exactly the observed
signature: rho -0.0005, wmae +58%.

The test is decisive because the collapse is a property of the FEATURE MATRIX ALONE — no model, no
SHAP, no training. If the interp frame collapses many more cells than the served frame, H3 is
confirmed and the magnitude cost has a structural explanation rather than a mysterious one. And it
predicts a FLOOR: the best any model on that frame could do.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

from keybo.analysis.shap_diff import _char_weight_tables  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.features import (  # noqa: E402
    bigram_features_from_positions,
    interp_features_from_positions,
    interp_wpm_features_from_positions,
)

WPM = 90.0
surface = default_surface(WPM, None)
G = surface.geometry
pos = [*G.slots, G.space_position]
n = len(pos)
_, LAY_A = _resolve("flagship-c3")

FRAMES = {
    "served (20c)": lambda a, b: bigram_features_from_positions(G, (a, b), wpm=WPM),
    "interp (10c)": lambda a, b: interp_features_from_positions(G, (a, b), wpm=WPM),
    "interp-wpm (11c)": lambda a, b: interp_wpm_features_from_positions(G, (a, b), wpm=WPM),
}

# The corpus weight over POSITION cells, so the collapse is priced by how much typing it affects
# rather than by raw cell counts.
slot_a = surface._slot_of(LAY_A)
_w3, w2, covered = _char_weight_tables(surface, LAY_A)
perm = np.array([slot_a[c] for c in LAY_A] + [slot_a[" "]], dtype=np.intp)
w_pos = np.zeros((n, n))
np.add.at(w_pos, (perm[:, None], perm[None, :]), w2)
w_flat = w_pos.ravel()

ms = surface._T2.ravel()  # the SHIPPED per-cell time: the target a model on any frame must hit

out: dict = {"n_cells": int(n * n), "covered_mass": int(covered), "frames": {}}
print(f"[res] {n}x{n} = {n * n} position cells; covered mass {covered:,}")
print()
print(
    f"{'frame':<18} {'distinct rows':>14} {'collapsed cells':>16} {'wmass collapsed':>16} "
    f"{'FLOOR wmae':>11} {'FLOOR umae':>11}"
)

for label, feat in FRAMES.items():
    X = np.vstack([feat(a, b) for a in pos for b in pos])
    # Group cells by their EXACT feature row: cells in one group are indistinguishable to ANY model
    # on this frame and must receive the same prediction.
    _uniq, inv, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n_groups = len(counts)
    collapsed = int((counts[inv] > 1).sum())

    # THE FLOOR: the best achievable error is the within-group weighted mean, because the group's
    # single prediction can at best be its own (weighted) mean. This is a property of the FRAME, not
    # of any model — a genuine lower bound on magnitude error, measured rather than assumed.
    gw = np.bincount(inv, weights=w_flat, minlength=n_groups)
    gwm = np.bincount(inv, weights=w_flat * ms, minlength=n_groups)
    best_w = np.divide(gwm, gw, out=np.zeros_like(gwm), where=gw > 0)
    floor_wmae = float((w_flat * np.abs(ms - best_w[inv])).sum() / max(w_flat.sum(), 1))
    gu = np.bincount(inv, minlength=n_groups)
    gum = np.bincount(inv, weights=ms, minlength=n_groups)
    best_u = gum / np.maximum(gu, 1)
    floor_umae = float(np.abs(ms - best_u[inv]).mean())

    wmass_collapsed = float(w_flat[counts[inv] > 1].sum() / max(w_flat.sum(), 1))
    out["frames"][label] = {
        "n_columns": int(X.shape[1]),
        "distinct_feature_rows": n_groups,
        "collapsed_cells": collapsed,
        "collapsed_share": collapsed / (n * n),
        "corpus_mass_share_in_collapsed_groups": wmass_collapsed,
        "floor_wmae_ms": floor_wmae,
        "floor_umae_ms": floor_umae,
        "largest_group": int(counts.max()),
    }
    print(
        f"{label:<18} {n_groups:>14d} {collapsed:>16d} {wmass_collapsed:>15.1%} "
        f"{floor_wmae:>11.4f} {floor_umae:>11.4f}"
    )

# --- the VERDICT: does the floor explain the measured gap? ---------------------------------
lolo = json.load(open(f"{ARTIFACTS}/lolo.json"))


def mean_of(rep, key):
    return float(
        np.mean([m[key] for f in rep["folds"].values() for m in f["seeds"] if m.get(key) is not None])
    )


measured = {
    "CUR": mean_of(lolo["arms"]["CUR"], "wmae"),
    "INTERP": mean_of(lolo["arms"]["INTERP"], "wmae"),
}
variant = json.load(open(f"{ARTIFACTS}/variant.json"))
measured["INTERP-WPM"] = mean_of(variant["lolo"], "wmae")

f_served = out["frames"]["served (20c)"]["floor_wmae_ms"]
f_interp = out["frames"]["interp (10c)"]["floor_wmae_ms"]
d_floor = f_interp - f_served
d_measured = measured["INTERP"] - measured["CUR"]
out["verdict"] = {
    "floor_wmae_served": f_served,
    "floor_wmae_interp": f_interp,
    "floor_gap_ms": d_floor,
    "measured_lolo_wmae_gap_ms": d_measured,
    "share_of_measured_gap_explained_by_the_floor": (d_floor / d_measured) if d_measured else None,
    "h3_confirmed": bool(d_floor > 0),
}
print()
print("=" * 90)
print("H3 — is the magnitude cost a RESOLUTION floor of the frame itself?")
print("=" * 90)
print(f"  FLOOR wmae (best ANY model on the frame could do, on the shipped per-cell times):")
print(f"    served frame  {f_served:8.4f} ms")
print(f"    interp frame  {f_interp:8.4f} ms      difference {d_floor:+8.4f} ms")
print(f"  MEASURED held-out LOLO wmae gap (INTERP - CUR): {d_measured:+8.4f} ms")
if d_measured:
    print(f"  => the frame's own resolution floor explains {100 * d_floor / d_measured:.1f}% of it")
print()
print("  Measured held-out wmae per arm:")
for k, v in measured.items():
    print(f"    {k:<12} {v:8.4f} ms")
print()
print(f"H3 {'CONFIRMED' if d_floor > 0 else 'REFUTED'}: the interp frame makes "
      f"{out['frames']['interp (10c)']['collapsed_cells']} of {n * n} cells featurewise "
      f"indistinguishable (vs {out['frames']['served (20c)']['collapsed_cells']} on the served "
      f"frame), so a single prediction must cover each collapsed group.")

with open(f"{ARTIFACTS}/resolution.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"[res] wrote {ARTIFACTS}/resolution.json")
