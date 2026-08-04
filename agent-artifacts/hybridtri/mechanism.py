"""HYBRIDB-1 §5 — the WITHIN-GROUP ADVERSE SELECTION quantity (EXPLOIT-1's M4), on hybrid-B.

Registered in the prereg §5. EXPLOIT-1's non-circular mechanism test, reused verbatim in DEFINITION
so the numbers are directly comparable to its published ones:

  M4(board) := mass-weighted (T2_served[cell] - classmean(T2_served)), the class being the frame's
               feature-row equivalence class.

It reads ONLY the truth's table and the grouping -- never T2_hybridb's values -- so no model
level/calibration error can move it. It asks the exploitation question directly: inside a class the
proxy cannot tell members apart, so DID the search land on the members the TRUTH prices ABOVE their
class average?

EXPLOIT-1 measured, on interp.1's grouping:
  B-INTERP-optimal +0.1882   vs   B-SERVED-optimal -0.1754   (OPPOSITE signs)
  G-INTERP-optimal +0.0878   vs   G-SERVED-optimal -0.1168
My registered prediction: the same sign split appears on hybrid-B but SMALLER in magnitude, roughly
in proportion to the 71% null-space cut. A NULL here (same sign on both arms, or |delta| at noise)
would be evidence the cut removed the mechanism.

⚠ INVARIANT 5, registered in advance: my bigram weight is the trigram table's own first-two-char
marginal, exactly as EXPLOIT-1's was, so M2 IS THE SAME ALGEBRAIC IDENTITY on my run --
M2(board) == trusted(board) - hybridb_score(board). It is kept as a DECOMPOSITION (it prices the
ILLUSION: what the proxy thinks it gained vs what the truth says it lost) and NEVER as evidence.
The identity is verified numerically below and its residual reported.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, SCRATCH, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import (  # noqa: E402
    FEATURE_VERSION_HYBRIDB,
    hybridb_features_from_positions,
    interp_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402

WPM, SEEDS = 90.0, (0, 1, 2)
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP_ = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


models = [
    XGBoostTypingModel.load(
        f"{SCRATCH}/hybridb_mono_seed{s}.json", expected_feature_version=FEATURE_VERSION_HYBRIDB
    )
    for s in SEEDS
]
surface = default_surface(WPM, None)
T2_SERVED = surface._T2.copy()
vec_h = np.vstack([hybridb_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
T2_HYB = np.mean([m.predict_ms(vec_h, wpm=WPM).reshape(NP_, NP_) for m in models], axis=0)
_, inv, cnt = np.unique(vec_h, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
log(f"hybrid-B grouping: {len(cnt)} classes over {NP_ * NP_} cells; largest {cnt.max()}")

# interp.1's grouping too, so the two frames' M4 are computed by ONE code path on the SAME truth --
# a separate implementation per frame is how an arm wins on a definition difference.
vec_i = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
_, inv_i, cnt_i = np.unique(vec_i, axis=0, return_inverse=True, return_counts=True)
inv_i = inv_i.ravel()
log(f"interp.1 grouping: {len(cnt_i)} classes over {NP_ * NP_} cells; largest {cnt_i.max()}")

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
F2_CHAR = F3.sum(axis=2)

flat_s = T2_SERVED.ravel()
# UNWEIGHTED class mean of the TRUTH: a fixed property of (truth, grouping), independent of any
# board. A board-dependent (mass-weighted) centre would move with the thing being measured.
gmean_h = np.bincount(inv, weights=flat_s, minlength=len(cnt)) / cnt
DEV_H = (flat_s - gmean_h[inv]).reshape(NP_, NP_)
gmean_i = np.bincount(inv_i, weights=flat_s, minlength=len(cnt_i)) / cnt_i
DEV_I = (flat_s - gmean_i[inv_i]).reshape(NP_, NP_)
COLL_H = cnt[inv] > 1
COLL_I = cnt_i[inv_i] > 1
DELTA = T2_SERVED - T2_HYB  # for the M2 decomposition only


def per_board(lay30: str) -> dict:
    slot = {pos: i for i, pos in enumerate(GEO.slots)}
    perm = np.empty(NP_, dtype=np.intp)
    for i, ch in enumerate(lay30):
        perm[IDX[ch]] = slot[GEO.slots[i]]
    perm[NP_ - 1] = NP_ - 1
    W = np.zeros((NP_, NP_))
    np.add.at(W, (perm[:, None], perm[None, :]), F2_CHAR)
    tot = W.sum()
    Wf = W.ravel()
    return {
        # M4 on HYBRID-B's grouping -- the non-circular smoking gun, over ALL mass
        "M4_hybridb_ms": float((W * DEV_H).sum() / tot),
        # the same restricted to COLLAPSED mass and normalized by it: the null space's own
        # contribution, undiluted by the singleton cells where no exploitation is possible
        "M4c_hybridb_within_collapsed_ms": float(
            (Wf * DEV_H.ravel())[COLL_H].sum() / Wf[COLL_H].sum()
        ),
        # M4 on INTERP.1's grouping, same code path -- so the two frames are comparable
        "M4_interp1_ms": float((W * DEV_I).sum() / tot),
        "M4c_interp1_within_collapsed_ms": float(
            (Wf * DEV_I.ravel())[COLL_I].sum() / Wf[COLL_I].sum()
        ),
        "M2_optimism_ms": float((W * DELTA).sum() / tot),
        "trusted_ms_per_char": float((W * T2_SERVED).sum() / tot),
        "hybridb_score_ms_per_char": float((W * T2_HYB).sum() / tot),
        "collapsed_mass_share_hybridb": float(Wf[COLL_H].sum() / tot),
        "collapsed_mass_share_interp1": float(Wf[COLL_I].sum() / tot),
    }


with open(f"{ARTIFACTS}/exploit.json") as _fh:
    ex = json.load(_fh)
boards = {}
for ch in ("G", "B"):
    boards[f"{ch}-HYBRIDB-optimal"] = ex["verdict"][ch]["hybridb_board"]
    boards[f"{ch}-SERVED-optimal"] = ex["verdict"][ch]["served_board"]
boards["qwerty-C30M (start)"] = CHARS
for name, lay in NAMED_LAYOUTS.items():
    if sorted(lay) == sorted(CHARS):
        boards[f"named:{name}"] = lay

out = {
    "prereg": "agent-artifacts/hybridtri/HYBRIDTRI-preregistration.md @ 5a5d3c3 §5",
    "EXPLOIT1_published": {
        "B-INTERP-optimal_M4": 0.1882,
        "B-SERVED-optimal_M4": -0.1754,
        "G-INTERP-optimal_M4": 0.0878,
        "G-SERVED-optimal_M4": -0.1168,
        "qwerty_M4": -0.2417,
    },
    "grouping": {
        "hybridb_classes": int(len(cnt)),
        "hybridb_largest": int(cnt.max()),
        "interp1_classes": int(len(cnt_i)),
        "interp1_largest": int(cnt_i.max()),
    },
    "boards": {},
}
log("")
log(f"{'board':<26} {'M4 hybridB':>11} {'M4c in-coll':>12} {'M4 interp.1':>12} {'M2 optim':>10}")
for label, lay in boards.items():
    r = per_board(lay)
    out["boards"][label] = {**r, "layout": lay}
    log(
        f"{label:<26} {r['M4_hybridb_ms']:>+11.4f} {r['M4c_hybridb_within_collapsed_ms']:>+12.4f} "
        f"{r['M4_interp1_ms']:>+12.4f} {r['M2_optimism_ms']:>+10.4f}"
    )

out["tests"] = {}
for ch in ("G", "B"):
    i = out["boards"][f"{ch}-HYBRIDB-optimal"]
    s = out["boards"][f"{ch}-SERVED-optimal"]
    v = ex["verdict"][ch]
    hyb_adv = i["hybridb_score_ms_per_char"] - s["hybridb_score_ms_per_char"]
    m4d = i["M4_hybridb_ms"] - s["M4_hybridb_ms"]
    out["tests"][ch] = {
        "gap_ms_per_char": v["gap_ms_per_char"],
        "M4_hybridb_optimal": i["M4_hybridb_ms"],
        "M4_served_optimal": s["M4_hybridb_ms"],
        "M4_delta": m4d,
        "M4_signs_are_OPPOSITE": bool(i["M4_hybridb_ms"] > 0 > s["M4_hybridb_ms"]),
        "M4_prediction_holds": bool(m4d > 0),
        "M4_share_of_gap": (m4d / v["gap_ms_per_char"]) if v["gap_ms_per_char"] else None,
        "M4c_hybridb_optimal": i["M4c_hybridb_within_collapsed_ms"],
        "M4c_served_optimal": s["M4c_hybridb_within_collapsed_ms"],
        "M4c_delta": i["M4c_hybridb_within_collapsed_ms"] - s["M4c_hybridb_within_collapsed_ms"],
        # EXPLOIT-1's own delta on interp.1, for the magnitude comparison my prereg predicted
        "EXPLOIT1_M4_delta": (0.1882 - (-0.1754)) if ch == "B" else (0.0878 - (-0.1168)),
        "proxy_thinks_it_GAINED_ms": -hyb_adv,
        "truth_says_it_LOST_ms": v["gap_ms_per_char"],
        "illusion_total_ms": v["gap_ms_per_char"] - hyb_adv,
        "M2_delta": i["M2_optimism_ms"] - s["M2_optimism_ms"],
    }
    t = out["tests"][ch]
    t["M4_delta_vs_EXPLOIT1_ratio"] = t["M4_delta"] / t["EXPLOIT1_M4_delta"]
    # INVARIANT 5: the identity check. M2_delta must EQUAL the illusion, to float precision.
    t["M2_identity_residual"] = abs(t["M2_delta"] - t["illusion_total_ms"])
    log("")
    log(f"### channel {ch} ###")
    log(
        f"  M4 adverse selection: hybrid-B-optimal {t['M4_hybridb_optimal']:+.4f} vs "
        f"served-optimal {t['M4_served_optimal']:+.4f}  (delta {t['M4_delta']:+.4f})  "
        f"PREDICTION {'HOLDS' if t['M4_prediction_holds'] else 'FAILS'}"
    )
    log(f"     signs OPPOSITE (EXPLOIT-1's signature): {t['M4_signs_are_OPPOSITE']}")
    log(
        f"     -> M4 accounts for {100 * (t['M4_share_of_gap'] or 0):.1f}% of the "
        f"{t['gap_ms_per_char']:+.4f} gap"
    )
    log(
        f"     -> delta is {t['M4_delta_vs_EXPLOIT1_ratio']:.3f}x EXPLOIT-1's interp.1 delta "
        f"({t['EXPLOIT1_M4_delta']:+.4f})  [prereg predicted SMALLER, i.e. < 1.0]"
    )
    log(
        f"  M4c (within collapsed mass only): hybrid-B-opt {t['M4c_hybridb_optimal']:+.4f} vs "
        f"served-opt {t['M4c_served_optimal']:+.4f}  (delta {t['M4c_delta']:+.4f})"
    )
    log(
        f"  THE ILLUSION (a DECOMPOSITION, not evidence): proxy thinks it GAINED "
        f"{t['proxy_thinks_it_GAINED_ms']:+.4f} ms/char; truth says it LOST "
        f"{t['truth_says_it_LOST_ms']:+.4f}  => total {t['illusion_total_ms']:+.4f}"
    )
    log(
        f"     INVARIANT 5 identity check: M2_delta {t['M2_delta']:+.6f} vs illusion "
        f"{t['illusion_total_ms']:+.6f}   residual {t['M2_identity_residual']:.3e}  "
        f"=> M2 IS an identity, kept as a decomposition only"
    )

with open(f"{ARTIFACTS}/mechanism.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log("")
log(f"wrote {ARTIFACTS}/mechanism.json")
