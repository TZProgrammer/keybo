"""EXPLOIT-1 §6 / INVARIANT 5 — is the ATTRIBUTION use actually safe? MEASURED, not argued.

The easy version of this verdict is an argument: "explaining a gap between two GIVEN boards never
runs a search, so the search result cannot bear on it." True as far as it goes, and it is the
reason a condemning §3 does not retract the attribution use. But it is not a measurement, and this
project's own record says an untested argument is a hypothesis.

So: the collapse has a DIRECT consequence for attribution that can be measured with no model and
no search. If two boards' differing cells fall in the SAME interp.1 feature-row class, the frame
assigns them the SAME predicted time -- so the T2 component of the gap between those two boards is
reported as ZERO no matter how large the truth says it is. That is not a search artifact; it is an
attribution failure on GIVEN boards, which is exactly the use §6 licenses.

Measured per board PAIR, over the campaign's own named boards:
  A1 -- the TRUTH's T2 gap between the pair (mass-weighted ms/char), the quantity to be explained.
  A2 -- the gap interp.1's BEST POSSIBLE model would report (the class-mean surface from R2, which
        has zero model error and only the frame's collapse).
  A3 -- BLIND MASS: the share of the pair's DIFFERING corpus mass that lands in cells the frame
        cannot distinguish from the other board's cells (same class).
A2/A1 is the attribution FIDELITY of the frame on that pair. If it is ~1.0 the attribution use is
safe on real pairs; if it is far from 1.0 the frame misreports gaps between GIVEN boards too, and
§6's licence has to be narrowed. Either way it is now a number.
"""

from __future__ import annotations

import itertools
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
from keybo.features import interp_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402

WPM = 90.0
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


surface = default_surface(WPM, None)
T2 = surface._T2.copy()
ms = T2.ravel()
vec_i = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
_, inv, cnt = np.unique(vec_i, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
# the BEST POSSIBLE interp-frame surface: class mean of the truth (R2's construction)
BEST = (np.bincount(inv, weights=ms, minlength=len(cnt)) / cnt)[inv].reshape(NP, NP)
CLASS = inv.reshape(NP, NP)

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
F2 = F3.sum(axis=2)
COVERED = float(F3.sum())
SLOT = {pos: i for i, pos in enumerate(GEO.slots)}


def weights(lay30):
    perm = np.empty(NP, dtype=np.intp)
    for i, ch in enumerate(lay30):
        perm[IDX[ch]] = SLOT[GEO.slots[i]]
    perm[NP - 1] = NP - 1
    W = np.zeros((NP, NP))
    np.add.at(W, (perm[:, None], perm[None, :]), F2)
    return W


# The campaign's own boards, restricted to the C30M charset (a different charset covers different
# corpus rows and would be a different denominator's mean in the same column).
boards = {n: lay for n, lay in NAMED_LAYOUTS.items() if sorted(lay) == sorted(CHARS)}
boards["qwerty-C30M"] = CHARS
ex = json.load(open(f"{ARTIFACTS}/exploit.json"))
boards["B-INTERP-optimal"] = ex["verdict"]["B"]["interp_board"]
boards["B-SERVED-optimal"] = ex["verdict"]["B"]["served_board"]
log(f"boards on the C30M charset: {list(boards)}")

out = {"note": "attribution fidelity of interp.1 on GIVEN board pairs -- no search involved",
       "pairs": {}}
rows = []
for a, b in itertools.combinations(sorted(boards), 2):
    Wa, Wb = weights(boards[a]), weights(boards[b])
    D = (Wa - Wb) / COVERED            # signed mass difference per cell
    truth = float((D * T2).sum())      # A1: the TRUTH's T2 gap, ms/char
    best = float((D * BEST).sum())     # A2: what the frame's BEST model reports
    # A3: of the mass that DIFFERS between the boards, how much is frame-blind?
    diff_mass = np.abs(D)
    tot = diff_mass.sum()
    # a differing cell is BLIND if its class also carries differing mass of the opposite sign
    blind = 0.0
    for c in np.unique(CLASS[diff_mass > 0]):
        m = (CLASS == c) & (diff_mass > 0)
        pos, neg = D[m][D[m] > 0].sum(), -D[m][D[m] < 0].sum()
        blind += 2.0 * min(pos, neg)   # the cancelling part the frame cannot see
    rec = {"truth_T2_gap_ms_per_char": truth, "best_frame_reported_gap": best,
           "fidelity": (best / truth) if abs(truth) > 1e-12 else None,
           "abs_error_ms_per_char": best - truth,
           "blind_share_of_differing_mass": float(blind / tot) if tot else None}
    out["pairs"][f"{a} vs {b}"] = rec
    rows.append((f"{a} vs {b}", truth, best, rec["fidelity"], rec["blind_share_of_differing_mass"]))

log("")
log(f"{'pair':<44} {'TRUTH gap':>11} {'frame gap':>11} {'fidelity':>9} {'blind mass':>11}")
for label, tr, be, fi, bl in sorted(rows, key=lambda r: -abs(r[1])):
    log(f"{label:<44} {tr:>+11.4f} {be:>+11.4f} "
        f"{(f'{fi:.4f}' if fi is not None else 'n/a'):>9} "
        f"{(f'{bl:.1%}' if bl is not None else 'n/a'):>11}")

fids = [r[3] for r in rows if r[3] is not None]
errs = [abs(r[2] - r[1]) for r in rows]
signs = [np.sign(r[1]) == np.sign(r[2]) for r in rows if abs(r[1]) > 1e-12]
out["summary"] = {
    "n_pairs": len(rows), "fidelity_min": float(min(fids)), "fidelity_max": float(max(fids)),
    "fidelity_median": float(np.median(fids)),
    "abs_error_max_ms_per_char": float(max(errs)), "abs_error_median": float(np.median(errs)),
    "sign_agreement": float(np.mean(signs)),
    "blind_share_median": float(np.median([r[4] for r in rows if r[4] is not None])),
}
s = out["summary"]
log("")
log("=" * 92)
log("ATTRIBUTION VERDICT (§6 / INVARIANT 5) -- measured on GIVEN pairs, no search")
log("=" * 92)
log(f"  fidelity (frame-reported gap / true gap): median {s['fidelity_median']:.4f}, "
    f"range [{s['fidelity_min']:.4f}, {s['fidelity_max']:.4f}] over {s['n_pairs']} pairs")
log(f"  SIGN agreement (does it get the DIRECTION right?): {s['sign_agreement']:.1%}")
log(f"  absolute error: median {s['abs_error_median']:.4f}, worst {s['abs_error_max_ms_per_char']:.4f} ms/char")
log(f"  blind share of differing mass: median {s['blind_share_median']:.1%}")

with open(f"{ARTIFACTS}/attrib_safe.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/attrib_safe.json")
