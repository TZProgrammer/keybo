"""EXPLOIT-1 §4 addendum — M2 IS AN IDENTITY, NOT A TEST. The non-circular replacement (M4).

A defect in my OWN registered mechanism metric, found by checking it rather than reporting it:

  M2(board) := mass-weighted (T2_served - T2_interp) over the cells the board uses.

Because the bigram weight I use is F3.sum(axis=2), its total EQUALS the covered mass, so in the
B channel M2(board) is EXACTLY `trusted(board) - interp_surface_score(board)` -- verified to
1.4e-14. Hence

  M2_delta = gap - [interp(I) - interp(S)]

and the bracket is NEGATIVE whenever the interp search beat the control on its OWN surface, which
is what a working search does BY CONSTRUCTION. So `M2_delta > 0` was very nearly guaranteed before
any exploitation existed. It is a DECOMPOSITION of the gap, not evidence for a mechanism.

Kept and reported as a decomposition (it is genuinely informative in that role -- it prices the
ILLUSION: how much the proxy thinks it gained vs how much the truth says it lost), and replaced as
a TEST by:

  M4 -- WITHIN-GROUP ADVERSE SELECTION. mass-weighted (T2_served[cell] - groupmean(T2_served)),
        the group being the interp.1 feature-row equivalence class. This asks the exploitation
        question directly: inside each class the proxy cannot tell members apart, so DID the
        search land on the members the TRUTH prices above their class average?

M4 is not circular: it reads only the TRUTH's table and the GROUPING. It never touches
`T2_interp`'s values, so it cannot be moved by the model's level/calibration error -- which is
exactly the part of M2 that made M2 tautological. It isolates the STRUCTURAL null space.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import ARTIFACTS, SCRATCH, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import (  # noqa: E402
    FEATURE_VERSION_INTERP,
    interp_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402

WPM, K31_SEEDS = 90.0, (0, 1, 2)
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


models = [XGBoostTypingModel.load(f"{SCRATCH}/interp_mono_seed{s}.json",
                                  expected_feature_version=FEATURE_VERSION_INTERP)
          for s in K31_SEEDS]
surface = default_surface(WPM, None)
T2_SERVED = surface._T2.copy()
vec_i = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
T2_INTERP = np.mean([m.predict_ms(vec_i, wpm=WPM).reshape(NP, NP) for m in models], axis=0)
_, inv, cnt = np.unique(vec_i, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()

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
F2_CHAR = F3.sum(axis=2)

flat_s = T2_SERVED.ravel()
COLLAPSED = (cnt[inv] > 1)

# UNWEIGHTED class mean of the TRUTH: a fixed property of (truth, grouping), independent of any
# board. A board-dependent (mass-weighted) centre would move with the thing being measured.
gmean = np.bincount(inv, weights=flat_s, minlength=len(cnt)) / cnt
DEV = (flat_s - gmean[inv]).reshape(NP, NP)      # how far above its class the TRUTH prices a cell
DELTA = T2_SERVED - T2_INTERP                    # kept for the M2 decomposition only


def per_board(lay30: str) -> dict:
    slot = {pos: i for i, pos in enumerate(GEO.slots)}
    perm = np.empty(NP, dtype=np.intp)
    for i, ch in enumerate(lay30):
        perm[IDX[ch]] = slot[GEO.slots[i]]
    perm[NP - 1] = NP - 1
    W = np.zeros((NP, NP))
    np.add.at(W, (perm[:, None], perm[None, :]), F2_CHAR)
    tot = W.sum()
    Wf = W.ravel()
    coll_mass = Wf[COLLAPSED].sum()
    return {
        # M4 -- the non-circular smoking gun, over ALL mass
        "M4_adverse_selection_ms": float((W * DEV).sum() / tot),
        # M4c -- the same, restricted to COLLAPSED mass and normalized by it: the null space's
        # own contribution, undiluted by the singleton cells where no exploitation is possible.
        "M4c_adverse_within_collapsed_ms": float((Wf * DEV.ravel())[COLLAPSED].sum() / coll_mass),
        "M2_optimism_ms": float((W * DELTA).sum() / tot),
        "trusted_ms_per_char": float((W * T2_SERVED).sum() / tot),
        "interp_score_ms_per_char": float((W * T2_INTERP).sum() / tot),
    }


ex = json.load(open(f"{ARTIFACTS}/exploit.json"))
boards = {}
for ch in ("G", "B"):
    boards[f"{ch}-INTERP-optimal"] = ex["verdict"][ch]["interp_board"]
    boards[f"{ch}-SERVED-optimal"] = ex["verdict"][ch]["served_board"]
boards["qwerty-C30M (start)"] = CHARS
for name, lay in NAMED_LAYOUTS.items():
    if sorted(lay) == sorted(CHARS):
        boards[f"named:{name}"] = lay

out = {"prereg": "EXPLOIT-preregistration.md @ da56139 §4 (M2 corrected to M4)", "boards": {}}
log(f"{'board':<24} {'M4 adverse ms':>14} {'M4c in-collapsed':>17} {'M2 optimism':>12}")
for label, lay in boards.items():
    r = per_board(lay)
    out["boards"][label] = {**r, "layout": lay}
    log(f"{label:<24} {r['M4_adverse_selection_ms']:>+14.4f} "
        f"{r['M4c_adverse_within_collapsed_ms']:>+17.4f} {r['M2_optimism_ms']:>+12.4f}")

out["tests"] = {}
for ch in ("G", "B"):
    i = out["boards"][f"{ch}-INTERP-optimal"]
    s = out["boards"][f"{ch}-SERVED-optimal"]
    v = ex["verdict"][ch]
    # THE DECOMPOSITION: gap = M2_delta + (interp advantage on its OWN surface).
    interp_adv = i["interp_score_ms_per_char"] - s["interp_score_ms_per_char"]
    m4d = i["M4_adverse_selection_ms"] - s["M4_adverse_selection_ms"]
    out["tests"][ch] = {
        "gap_ms_per_char": v["gap_ms_per_char"],
        "M4_interp": i["M4_adverse_selection_ms"], "M4_served": s["M4_adverse_selection_ms"],
        "M4_delta": m4d, "M4_prediction_holds": bool(m4d > 0),
        "M4_share_of_gap": (m4d / v["gap_ms_per_char"]) if v["gap_ms_per_char"] else None,
        "M4c_interp": i["M4c_adverse_within_collapsed_ms"],
        "M4c_served": s["M4c_adverse_within_collapsed_ms"],
        "M4c_delta": i["M4c_adverse_within_collapsed_ms"] - s["M4c_adverse_within_collapsed_ms"],
        "proxy_thinks_it_GAINED_ms": -interp_adv,
        "truth_says_it_LOST_ms": v["gap_ms_per_char"],
        "illusion_total_ms": v["gap_ms_per_char"] - interp_adv,
        "M2_delta_equals_illusion": i["M2_optimism_ms"] - s["M2_optimism_ms"],
    }
    t = out["tests"][ch]
    log("")
    log(f"### channel {ch} ###")
    log(f"  M4 adverse selection: interp {t['M4_interp']:+.4f} vs served {t['M4_served']:+.4f} ms  "
        f"(delta {t['M4_delta']:+.4f})  PREDICTION {'HOLDS' if t['M4_prediction_holds'] else 'FAILS'}")
    log(f"     -> M4 accounts for {100 * (t['M4_share_of_gap'] or 0):.1f}% of the {t['gap_ms_per_char']:+.4f} gap")
    log(f"  M4c (within collapsed mass only): interp {t['M4c_interp']:+.4f} vs served "
        f"{t['M4c_served']:+.4f}  (delta {t['M4c_delta']:+.4f})")
    log(f"  THE ILLUSION: proxy thinks it GAINED {t['proxy_thinks_it_GAINED_ms']:+.4f} ms/char; "
        f"truth says it LOST {t['truth_says_it_LOST_ms']:+.4f}  => total {t['illusion_total_ms']:+.4f}")
    log(f"     (identity check: M2_delta = {t['M2_delta_equals_illusion']:+.6f} vs illusion "
        f"{t['illusion_total_ms']:+.6f})")

with open(f"{ARTIFACTS}/mechanism2.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/mechanism2.json")
