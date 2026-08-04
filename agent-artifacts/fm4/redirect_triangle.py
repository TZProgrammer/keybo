"""FM4 step 2b: the THREE-WAY redirect comparison, because the campaign has THREE predicates.

The parent's brief (inheriting TCOND-1) states the frame's ``redirect``/``bad_redirect`` are
predicate-EQUAL to "the gauge's" over all 30^3 triples, citing ``analysis/redirects.py``. Read
that docstring: its exhaustive claim is

    "``kmstats._is_redirect`` ... ``_v1_pattern`` ... It is not a subset -- it is EQUAL.
     Exhaustively over all 30**3 = 27000 slot triples: 2808 satisfy both, 0 satisfy only one"

which is GAUGE vs GAUGE. Neither side of it is the model's FRAME column. This driver measures
all three pairings so the distinction is a number, not an argument:

  A = FRAME       keybo.features.ngram._trigram_level_from_positions["redirect"]
  B = GAUGE km    keybo.analysis.kmstats._is_redirect          (the ``redir`` gauge)
  C = GAUGE v1    keybo.analysis.community._v1_pattern         (the redirect FAMILY gauge)
"""

from __future__ import annotations

import itertools
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import keybo  # noqa: E402
from keybo.analysis import kmstats as KM  # noqa: E402
from keybo.analysis.community import _v1_pattern  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.features.ngram import _trigram_level_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

print("keybo.__file__ =", keybo.__file__)
G = ROW_STAGGERED_30
SLOTS = list(G.slots)

# RedirectFamily's own slot->finger map, used EXACTLY as it uses it (redirects.py:87,108).
FAM_FINGER = [key.finger for key in KM._KEYS]

_FAMILY = ("redirects", "redirects_sfs", "bad_redirects", "bad_redirects_sfs")
_BADFAMILY = ("bad_redirects", "bad_redirects_sfs")

rows = []
counts = {"A": 0, "B": 0, "C": 0, "A_bad": 0, "C_bad": 0}
pair_disagree = {"A_vs_B": 0, "A_vs_C": 0, "B_vs_C": 0, "Abad_vs_Cbad": 0}
only = {"A_not_B": 0, "B_not_A": 0, "A_not_C": 0, "C_not_A": 0, "B_not_C": 0, "C_not_B": 0}

for i, j, k in itertools.product(range(len(SLOTS)), repeat=3):
    a, b, c = SLOTS[i], SLOTS[j], SLOTS[k]
    tri = _trigram_level_from_positions(G, a, b, c)
    A = bool(tri["redirect"])
    A_bad = bool(tri["bad_redirect"])
    B = bool(KM._is_redirect(KM._KEYS[i], KM._KEYS[j], KM._KEYS[k]))
    pat = _v1_pattern(FAM_FINGER[i], FAM_FINGER[j], FAM_FINGER[k])
    Cc = pat in _FAMILY
    C_bad = pat in _BADFAMILY

    counts["A"] += A
    counts["B"] += B
    counts["C"] += Cc
    counts["A_bad"] += A_bad
    counts["C_bad"] += C_bad
    pair_disagree["A_vs_B"] += A != B
    pair_disagree["A_vs_C"] += Cc != A
    pair_disagree["B_vs_C"] += Cc != B
    pair_disagree["Abad_vs_Cbad"] += A_bad != C_bad
    only["A_not_B"] += A and not B
    only["B_not_A"] += B and not A
    only["A_not_C"] += A and not Cc
    only["C_not_A"] += Cc and not A
    only["B_not_C"] += B and not Cc
    only["C_not_B"] += Cc and not B
    if Cc != A and len(rows) < 6:
        rows.append(
            {
                "triple_positions": [a, b, c],
                "frame_redirect": A,
                "km_redir": B,
                "v1_family": Cc,
                "v1_label": pat,
                "same_finger_ab": C.same_finger(G, a, b),
                "same_finger_bc": C.same_finger(G, b, c),
            }
        )

n = len(SLOTS) ** 3
print(f"\n=== all {n} ordered slot triples of ROW_STAGGERED_30 ===")
print(f"  A  FRAME   redirect  fires : {counts['A']}")
print(f"  B  GAUGE   km redir  fires : {counts['B']}")
print(f"  C  GAUGE   v1 family fires : {counts['C']}")
print(f"  A  FRAME   bad_redirect    : {counts['A_bad']}")
print(f"  C  GAUGE   v1 bad family   : {counts['C_bad']}")
print("\n=== pairwise DISAGREEMENTS ===")
for key, val in pair_disagree.items():
    verdict = "EQUAL" if val == 0 else "DIFFERENT"
    print(f"  {key:<16} disagree={val:<6} ({100.0*val/n:5.2f}%)  -> {verdict}")
print("\n=== asymmetry (who fires when the other does not) ===")
for key, val in only.items():
    print(f"  {key:<10} {val}")

print("\n=== the claim in redirects.py's own words, re-measured ===")
both = sum(
    1
    for i, j, k in itertools.product(range(len(SLOTS)), repeat=3)
    if KM._is_redirect(KM._KEYS[i], KM._KEYS[j], KM._KEYS[k])
    and _v1_pattern(FAM_FINGER[i], FAM_FINGER[j], FAM_FINGER[k]) in _FAMILY
)
print(f"  triples satisfying BOTH km_redir and v1-family : {both}   (docstring says 2808)")
print(f"  triples satisfying exactly ONE                 : {pair_disagree['B_vs_C']}   (docstring says 0)")

print("\n=== first frame-vs-v1 disagreements (mechanism) ===")
for r in rows:
    print(f"  {r}")

out = {
    "keybo_file": keybo.__file__,
    "n_triples": n,
    "firings": counts,
    "pairwise_disagreements": pair_disagree,
    "asymmetry": only,
    "redirects_docstring_recheck": {"both": both, "exactly_one": pair_disagree["B_vs_C"]},
    "sample_frame_vs_v1_disagreements": rows,
}
with open(os.path.join(os.path.dirname(__file__), "redirect_triangle.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote redirect_triangle.json")
