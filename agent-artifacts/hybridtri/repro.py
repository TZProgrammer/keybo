"""HYBRIDTRI step 0 — REPRODUCE the numbers my brief handed me, BEFORE registering anything.

Nothing here is a new claim. Every number is one a sibling published and my brief quoted; the
point is that a brief is a HYPOTHESIS (three siblings in a row corrected the parent) and a
measurement I did not run is not one I may build on.

Reproduced here, each against its published value:
  * hybrid-B's structure: 18 columns, 573 distinct feature rows, null space 0.9377 ms (28.8% of
    interp.1's), MAXCORR 0.7079  -- EXPLOIT-1 §g / hybrid.json + hybrid_cost.json
  * the frames it is built from: served 765 rows / MAXCORR 0.9813; interp.1 378 rows / 0.7037
  * the ARM-2 axis numbers: served-bigram resolution 0.7960 (765/961, largest group 4) vs
    TRIGRAM resolution 0.9401 (28006/29791, largest group 2)   -- FRAMEDIAG-1 §e
  * the split-pairs count the other axis rests on: served bigram 7, trigram 51 (3.0465 ms/char)
    -- INTERPFRAME-1 §a. That one needs SHAP attributions, so it is deferred to axis.py; here I
    only reproduce what is model-free.

Uses the SHIPPED diagnostic (`keybo.analysis.frame_collapse`, FRAMEDIAG-1) rather than
re-implementing np.unique, so the structure numbers come from the instrument the repo now owns.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, assert_tree, load_by_path, require  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import frame_collapse as FC  # noqa: E402
from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
    bigram_features_from_positions,
    interp_features_from_positions,
    trigram_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

# Every symbol I lean on, asserted to EXIST on this tree before use (rc=0 with all-None output is
# a key-not-present bug, not a measurement).
for _n in ("frame_collapse", "cell_positions", "feature_matrix", "group_cells"):
    require(FC, _n)

WPM = 90.0
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP_ = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


# ==========================================================================================
# The hybrid-B frame, defined ONCE here and imported by every later driver, so no two drivers
# can disagree about what hybrid-B is. Selected BY NAME from the served frame's own list and
# asserted to exist (EXPLOIT-1's hybrid.py convention).
# ==========================================================================================
SERVED_NAMES = list(BIGRAM_FEATURE_NAMES)
INTERP_NAMES = list(BIGRAM_INTERP_FEATURE_NAMES)
ROW_ONEHOTS = ["bottom", "home", "top"]
FINGER_ONEHOTS = ["pinky", "ring", "middle", "index", "lateral"]
for _n in ROW_ONEHOTS + FINGER_ONEHOTS:
    if _n not in SERVED_NAMES:
        raise SystemExit(f"ABORT: {_n!r} not in the served frame on this tree: {SERVED_NAMES}")
HYBRIDB_NAMES = INTERP_NAMES + ROW_ONEHOTS + FINGER_ONEHOTS


def hybridb_features_from_positions(geometry, positions, wpm: float = 0.0) -> np.ndarray:
    """hybrid-B: interp.1's 10 ordinals + the served ROW and FINGER one-hots (18 columns).

    Built by CONCATENATING the two shipped featurizers' outputs and slicing the served one by
    NAME, so the columns are provably the same objects the two published frames carry.
    """
    iv = interp_features_from_positions(geometry, positions, wpm=wpm)
    sv = bigram_features_from_positions(geometry, positions, wpm=wpm)
    take = [SERVED_NAMES.index(n) for n in ROW_ONEHOTS + FINGER_ONEHOTS]
    return np.concatenate([iv, sv[take]])


# ==========================================================================================
# (1) STRUCTURE via the SHIPPED diagnostic
# ==========================================================================================
FRAMES = {
    "served": (2, lambda g, c: bigram_features_from_positions(g, c, wpm=WPM)),
    "interp.1": (2, lambda g, c: interp_features_from_positions(g, c, wpm=WPM)),
    "hybrid-B": (2, lambda g, c: hybridb_features_from_positions(g, c, wpm=WPM)),
    "trigram": (3, lambda g, c: trigram_features_from_positions(g, c, wpm=WPM)),
}

# the corpus weights, on INTERPFRAME-1's OWN weighting board (flagship-c3) -- a MAXCORR read on
# qwerty-C30M gave 0.9556 vs the published 0.9813 purely from the grid, so the grid is part of
# the instrument (EXPLOIT-1's hybrid_cost.py note).
surface = default_surface(WPM, None)
from keybo.analysis.shap_diff import _char_weight_tables  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402

_, LAY_W = _resolve("flagship-c3")
_slot = surface._slot_of(LAY_W)
_w3, _w2, _covered = _char_weight_tables(surface, LAY_W)
_perm = np.array([_slot[c] for c in LAY_W] + [_slot[" "]], dtype=np.intp)
_wp = np.zeros((NP_, NP_))
np.add.at(_wp, (_perm[:, None], _perm[None, :]), _w2)
W_FLAG = _wp.ravel()
log(f"[grid] weighting board flagship-c3 = {LAY_W}   covered mass {_covered:,.0f}")

# the T2 truth, for the floor
T2 = surface._T2.copy()

out = {"published": {}, "measured": {}, "frames": {}}
log("")
log(f"{'frame':<12} {'ord':>3} {'cols':>5} {'rows':>7} {'cells':>7} {'resolution':>11} {'lgst':>5}")
for label, (order, fn) in FRAMES.items():
    r = FC.frame_collapse(
        fn,
        GEO,
        order=order,
        include_space=True,
        target=T2.ravel() if order == 2 else None,
        weights=W_FLAG if order == 2 else None,
    )
    d = r.as_dict()
    out["frames"][label] = d
    log(
        f"{label:<12} {order:>3} {d['n_columns']:>5} {d['distinct_feature_rows']:>7} "
        f"{d['n_cells']:>7} {d['resolution']:>11.4f} {d['largest_group']:>5}"
    )

# ==========================================================================================
# (2) MAXCORR via INTERPFRAME-1's OWN registered instrument, loaded BY PATH
# ==========================================================================================
M = load_by_path(
    "interpframe_metrics_hybridtri",
    "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/metrics.py",
)
for _fn in ("m1_maxcorr", "weighted_corr_matrix", "m3_splitpairs", "same_property_groups"):
    require(M, _fn)

SERVED_X = np.vstack(
    [bigram_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS]
)
INTERP_X = np.vstack(
    [interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS]
)
HYB_X = np.vstack([hybridb_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
if HYB_X.shape[1] != 18:
    raise SystemExit(f"ABORT: hybrid-B width {HYB_X.shape[1]} != 18")

log("")
log(f"{'frame':<12} {'cols':>5} {'MAXCORR':>9} {'>0.9':>5} {'>0.7':>5} {'MEANCORR':>9}  worst pair")
CORR = {}
for label, X, names in (
    ("served", SERVED_X, SERVED_NAMES),
    ("interp.1", INTERP_X, INTERP_NAMES),
    ("hybrid-B", HYB_X, HYBRIDB_NAMES),
):
    r = M.m1_maxcorr(X, W_FLAG, names)
    CORR[label] = r
    out["frames"][label]["maxcorr"] = r
    log(
        f"{label:<12} {X.shape[1]:>5} {r['maxcorr']:>9.4f} {r['n_pairs_over_0.9']:>5} "
        f"{r['n_pairs_over_0.7']:>5} {r['meancorr']:>9.4f}  {r['worst_pair']}"
    )

# ==========================================================================================
# (3) THE SEARCHABLE NULL SPACE -- EXPLOIT-1's own quantity, reproduced with its own formula
# (mass-weighted within-group sd of the TRUTH). Recomputed here rather than borrowed.
# ==========================================================================================
tri = {
    k: v
    for k, v in load_frequencies(str(production_corpus_dir(None) / "trigrams.txt")).items()
    if len(k) == 3
}
IDX = {c: i for i, c in enumerate(CHARS)}
IDX[" "] = NP_ - 1
F3 = np.zeros((NP_, NP_, NP_))
for ng, f in tri.items():
    try:
        F3[IDX[ng[0]], IDX[ng[1]], IDX[ng[2]]] += f
    except KeyError:
        continue
# EXPLOIT-1's hybrid.py weighted on the C30M START board (char space == position space there),
# NOT flagship-c3. Reproduced with ITS grid so the null-space numbers are comparable to its
# published ones; the MAXCORR above uses flagship-c3 because THAT is the grid its bar was set on.
W_C30M = F3.sum(axis=2).ravel()
ms = T2.ravel()

log("")
log(f"{'frame':<12} {'rows':>6} {'coll mass':>10} {'FLOOR wmae':>11} {'NULLSPACE ms':>13}")
NULL = {}
for label, X in (("served", SERVED_X), ("interp.1", INTERP_X), ("hybrid-B", HYB_X)):
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    ng = len(cnt)
    gw = np.bincount(inv, weights=W_C30M, minlength=ng)
    gwm = np.bincount(inv, weights=W_C30M * ms, minlength=ng)
    bw = np.divide(gwm, gw, out=np.zeros_like(gwm), where=gw > 0)
    floor_w = float((W_C30M * np.abs(ms - bw[inv])).sum() / W_C30M.sum())
    coll = cnt[inv] > 1
    cmass = float(W_C30M[coll].sum() / W_C30M.sum())
    sd = np.zeros(ng)
    for g in np.flatnonzero(cnt > 1):
        sd[g] = ms[inv == g].std()
    nullspace = float((W_C30M * sd[inv]).sum() / W_C30M.sum())
    NULL[label] = {
        "distinct_feature_rows": int(ng),
        "collapsed_mass_share_C30M": cmass,
        "floor_wmae_at_group_mean_C30M": floor_w,
        "searchable_nullspace_ms_C30M": nullspace,
    }
    out["frames"][label]["nullspace_C30M"] = NULL[label]
    log(f"{label:<12} {ng:>6} {cmass:>9.1%} {floor_w:>11.4f} {nullspace:>13.4f}")

# ==========================================================================================
# (4) THE SCORECARD: every published number vs mine, with |diff|
# ==========================================================================================
PUB = {
    # EXPLOIT-1 §g hybrid.json / hybrid_cost.json
    "hybridB_n_columns": (18, out["frames"]["hybrid-B"]["n_columns"]),
    "hybridB_distinct_rows": (573, NULL["hybrid-B"]["distinct_feature_rows"]),
    "hybridB_nullspace_ms": (0.9377, NULL["hybrid-B"]["searchable_nullspace_ms_C30M"]),
    "hybridB_maxcorr": (0.7079, CORR["hybrid-B"]["maxcorr"]),
    "hybridB_floor_wmae_at_mean": (0.2545, NULL["hybrid-B"]["floor_wmae_at_group_mean_C30M"]),
    # the two frames it is built from
    "served_distinct_rows": (765, NULL["served"]["distinct_feature_rows"]),
    "interp_distinct_rows": (378, NULL["interp.1"]["distinct_feature_rows"]),
    "served_maxcorr": (0.9813, CORR["served"]["maxcorr"]),
    "interp_maxcorr": (0.7037, CORR["interp.1"]["maxcorr"]),
    "served_meancorr": (0.1137, CORR["served"]["meancorr"]),
    "interp_meancorr": (0.1572, CORR["interp.1"]["meancorr"]),
    "interp_nullspace_ms": (3.2565, NULL["interp.1"]["searchable_nullspace_ms_C30M"]),
    "served_nullspace_ms": (0.0000, NULL["served"]["searchable_nullspace_ms_C30M"]),
    # FRAMEDIAG-1 §e -- the ARM-2 resolution axis
    "served_resolution": (0.7960, out["frames"]["served"]["resolution"]),
    "served_largest_group": (4, out["frames"]["served"]["largest_group"]),
    "trigram_resolution": (0.9401, out["frames"]["trigram"]["resolution"]),
    "trigram_largest_group": (2, out["frames"]["trigram"]["largest_group"]),
    "trigram_distinct_rows": (28006, out["frames"]["trigram"]["distinct_feature_rows"]),
    "trigram_n_cells": (29791, out["frames"]["trigram"]["n_cells"]),
    "trigram_n_columns": (46, out["frames"]["trigram"]["n_columns"]),
}
log("")
log("=" * 92)
log("SCORECARD -- every number my brief / the sibling reports handed me, vs MY measurement")
log("=" * 92)
log(f"{'quantity':<34} {'published':>12} {'measured':>14} {'|diff|':>12}  verdict")
bad = []
for k, (pub, mine) in PUB.items():
    diff = abs(float(pub) - float(mine))
    # 4-dp published values reproduce to <=5e-5; integer counts must be EXACT.
    tol = 0.0 if isinstance(pub, int) else 5e-5
    ok = diff <= tol
    if not ok:
        bad.append((k, pub, mine, diff))
    log(f"{k:<34} {pub:>12} {mine:>14.6f} {diff:>12.2e}  {'OK' if ok else '** MISMATCH **'}")
    out["published"][k] = {"published": pub, "measured": float(mine), "abs_diff": diff, "ok": ok}

log("")
if bad:
    log(
        f"!! {len(bad)} MISMATCH(ES) -- the brief's premise is WRONG on these and must be corrected:"
    )
    for k, pub, mine, diff in bad:
        log(f"   {k}: published {pub} vs measured {mine} (|diff| {diff:.4e})")
else:
    log(f"ALL {len(PUB)} published numbers REPRODUCE on my tree.")
out["measured"]["n_mismatches"] = len(bad)
out["measured"]["mismatches"] = [
    {"key": k, "published": p, "measured": float(m), "abs_diff": d} for k, p, m, d in bad
]

with open(f"{ARTIFACTS}/repro.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/repro.json")
