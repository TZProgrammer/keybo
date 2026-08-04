"""EXPLOIT-1 §7 / INVARIANT 6 — the HYBRID frame: measure its COLLAPSE and FLOOR first.

Model-free, no training, seconds. The parent's proposal: served one-hots for RESOLUTION plus the
interp ordinals for INTERPRETATION. The registered gate is that the floor must look good BEFORE
any training is spent.

I add the test that actually matters given the R2 result, because it can be answered with no model
at all: a frame is exploitable-by-construction exactly to the extent that its feature rows COLLAPSE
cells the truth prices differently. R2 measured that a PERFECT model on interp.1 is still
exploitable in the bigram channel, so the FRAME's collapse -- not any model's accuracy -- is what
the search can walk into. So for each candidate frame I report:
  - distinct feature rows over the 961 cells, and collapsed corpus-mass share;
  - the within-group floor (wmae/umae) -- the best ANY model on the frame could do;
  - the SEARCHABLE null-space size: the corpus-mass-weighted within-group spread of the TRUTH,
    which is the quantity a search can exploit and a LOLO evaluation cannot see.

Frames compared: served (20c), interp.1 (10c), and three hybrids of increasing width.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
    bigram_features_from_positions,
    interp_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

WPM = 90.0
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


surface = default_surface(WPM, None)
T2 = surface._T2.copy()          # THE TRUTH: the shipped per-cell time a model must hit
ms = T2.ravel()

# corpus bigram mass per POSITION cell, on the C30M start board (the same convention
# INTERPFRAME-1's resolution.py used: weight the collapse by how much typing it affects)
tri = {k: v for k, v in load_frequencies(str(production_corpus_dir(None) / "trigrams.txt")).items()
       if len(k) == 3}
IDX = {c: i for i, c in enumerate(CHARS)}
IDX[" "] = NP - 1
F3 = np.zeros((NP, NP, NP))
for ng, f in tri.items():
    try:
        F3[IDX[ng[0]], IDX[ng[1]], IDX[ng[2]]] += f
    except KeyError:
        continue
# the C30M start board maps char index -> the SAME slot index, so char space IS position space here
w_flat = F3.sum(axis=2).ravel()

SERVED = np.vstack([bigram_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
INTERP = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
sn, inn = list(BIGRAM_FEATURE_NAMES), list(BIGRAM_INTERP_FEATURE_NAMES)
log(f"served {SERVED.shape} cols={len(sn)}  interp {INTERP.shape} cols={len(inn)}")

# The served one-hot blocks that carry RESOLUTION -- the parent's "served one-hots" half. Selected
# by NAME from the served frame's own list, and asserted to exist rather than assumed.
def cols(names):
    for n in names:
        if n not in sn:
            raise SystemExit(f"ABORT: {n!r} not in the served frame on this tree: {sn}")
    return SERVED[:, [sn.index(n) for n in names]]


ROW_ONEHOTS = [n for n in sn if n in ("bottom", "home", "top")]
FINGER_ONEHOTS = [n for n in sn if n in ("index", "middle", "ring", "pinky", "lateral")]
log(f"row one-hots found: {ROW_ONEHOTS}")
log(f"finger one-hots found: {FINGER_ONEHOTS}")

FRAMES = {
    "served (20c)": SERVED,
    "interp.1 (10c)": INTERP,
    "hybrid-A: interp + row one-hots": np.hstack([INTERP, cols(ROW_ONEHOTS)]),
    "hybrid-B: interp + row + finger one-hots": np.hstack([INTERP, cols(ROW_ONEHOTS + FINGER_ONEHOTS)]),
    "hybrid-C: interp + ALL served cols": np.hstack([INTERP, SERVED]),
}

out = {"n_cells": NP * NP, "covered_bigram_mass": float(w_flat.sum()), "frames": {}}
log("")
log(f"{'frame':<42} {'cols':>5} {'rows':>6} {'coll mass':>10} {'FLOOR wmae':>11} "
    f"{'FLOOR umae':>11} {'NULLSPACE ms':>13}")
for label, X in FRAMES.items():
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    ng = len(cnt)
    # weighted + unweighted within-group floors: the best ANY model on this frame could do
    gw = np.bincount(inv, weights=w_flat, minlength=ng)
    gwm = np.bincount(inv, weights=w_flat * ms, minlength=ng)
    bw = np.divide(gwm, gw, out=np.zeros_like(gwm), where=gw > 0)
    floor_w = float((w_flat * np.abs(ms - bw[inv])).sum() / w_flat.sum())
    bu = np.bincount(inv, weights=ms, minlength=ng) / cnt
    floor_u = float(np.abs(ms - bu[inv]).mean())
    coll = cnt[inv] > 1
    cmass = float(w_flat[coll].sum() / w_flat.sum())
    # THE SEARCHABLE NULL SPACE: mass-weighted within-group sd of the TRUTH. This is what a
    # search can exploit (pick the member the truth hates) and LOLO cannot see.
    sd = np.zeros(ng)
    for g in np.flatnonzero(cnt > 1):
        sd[g] = ms[inv == g].std()
    nullspace = float((w_flat * sd[inv]).sum() / w_flat.sum())
    out["frames"][label] = {
        "n_columns": int(X.shape[1]), "distinct_feature_rows": int(ng),
        "collapsed_cells": int(coll.sum()), "collapsed_mass_share": cmass,
        "floor_wmae_ms": floor_w, "floor_umae_ms": floor_u,
        "searchable_nullspace_ms": nullspace, "largest_group": int(cnt.max()),
    }
    log(f"{label:<42} {X.shape[1]:>5d} {ng:>6d} {cmass:>9.1%} {floor_w:>11.4f} "
        f"{floor_u:>11.4f} {nullspace:>13.4f}")

s, i = out["frames"]["served (20c)"], out["frames"]["interp.1 (10c)"]
log("")
log("=" * 96)
log("VERDICT on the hybrid (the registered gate: does the floor look good BEFORE training?)")
log("=" * 96)
for label in ("hybrid-A: interp + row one-hots", "hybrid-B: interp + row + finger one-hots",
              "hybrid-C: interp + ALL served cols"):
    h = out["frames"][label]
    log(f"  {label}")
    log(f"    rows {h['distinct_feature_rows']} (interp {i['distinct_feature_rows']}, "
        f"served {s['distinct_feature_rows']})   collapsed mass {h['collapsed_mass_share']:.1%} "
        f"(interp {i['collapsed_mass_share']:.1%}, served {s['collapsed_mass_share']:.1%})")
    log(f"    FLOOR wmae {h['floor_wmae_ms']:.4f} ms (interp {i['floor_wmae_ms']:.4f}, "
        f"served {s['floor_wmae_ms']:.4f})")
    log(f"    SEARCHABLE NULL SPACE {h['searchable_nullspace_ms']:.4f} ms "
        f"(interp {i['searchable_nullspace_ms']:.4f}, served {s['searchable_nullspace_ms']:.4f}) "
        f"=> {100 * h['searchable_nullspace_ms'] / max(i['searchable_nullspace_ms'], 1e-12):.1f}% of interp.1's")
    out["frames"][label]["floor_near_zero"] = bool(h["floor_wmae_ms"] < 0.05 * i["floor_wmae_ms"])
    log(f"    GATE (floor < 5% of interp.1's): "
        f"{'PASS -- training would be justified' if out['frames'][label]['floor_near_zero'] else 'FAIL'}")

out["interpretation_columns_retained"] = {
    "note": "every hybrid RETAINS all ten interp.1 ordinal columns, so the monotone/orthogonal "
            "interpretation columns survive; the added one-hots restore RESOLUTION only",
    "interp_columns": inn,
}
with open(f"{ARTIFACTS}/hybrid.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/hybrid.json")
