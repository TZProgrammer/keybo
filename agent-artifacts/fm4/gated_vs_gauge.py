"""FM4 step 2d: is the ALREADY-BUILT gated column the gauge's predicate?

If `redirect_sfgated` (REDIRGATE-1, in the WIDENED frame) is predicate-EQUAL to the reported
redirect gauge, then the correct naming is settled without inventing anything: the GAUGE's
predicate already has a column name in this repo (`redirect_sfgated`), and the served column is
the ungated one -- so the served column is the one whose name must say so.
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
from keybo.features.ngram import (  # noqa: E402
    _trigram_level_from_positions,
    trigram_direction_row,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

print("keybo.__file__ =", keybo.__file__)
G = ROW_STAGGERED_30
SLOTS = list(G.slots)
FAM_FINGER = [key.finger for key in KM._KEYS]
_FAMILY = ("redirects", "redirects_sfs", "bad_redirects", "bad_redirects_sfs")
_BAD = ("bad_redirects", "bad_redirects_sfs")

stats = {k: 0 for k in ("gated", "gated_bad", "v1", "v1_bad", "d_gated_v1", "d_gatedbad_v1bad",
                        "d_km_gated", "ungated", "ungated_bad")}
for i, j, k in itertools.product(range(len(SLOTS)), repeat=3):
    a, b, c = SLOTS[i], SLOTS[j], SLOTS[k]
    tri = _trigram_level_from_positions(G, a, b, c)
    gate = trigram_direction_row(G, a, b, c)
    gd = bool(gate["redirect_sfgated"])
    gdb = bool(gate["bad_redirect_sfgated"])
    pat = _v1_pattern(FAM_FINGER[i], FAM_FINGER[j], FAM_FINGER[k])
    v1 = pat in _FAMILY
    v1b = pat in _BAD
    km = bool(KM._is_redirect(KM._KEYS[i], KM._KEYS[j], KM._KEYS[k]))
    stats["gated"] += gd
    stats["gated_bad"] += gdb
    stats["v1"] += v1
    stats["v1_bad"] += v1b
    stats["ungated"] += bool(tri["redirect"])
    stats["ungated_bad"] += bool(tri["bad_redirect"])
    stats["d_gated_v1"] += gd != v1
    stats["d_gatedbad_v1bad"] += gdb != v1b
    stats["d_km_gated"] += km != gd

n = len(SLOTS) ** 3
print(f"\n=== all {n} ordered slot triples ===")
print(f"  frame  redirect        (UNGATED, served) : {stats['ungated']}")
print(f"  frame  redirect_sfgated (GATED, widened) : {stats['gated']}")
print(f"  gauge  v1 redirect family                : {stats['v1']}")
print(f"  gauge  kmstats redir                     : {stats['gated'] - stats['d_km_gated'] if False else 'see below'}")
print(f"\n  redirect_sfgated  vs v1 family  disagree : {stats['d_gated_v1']}  -> "
      f"{'EQUAL' if stats['d_gated_v1'] == 0 else 'DIFFERENT'}")
print(f"  redirect_sfgated  vs kmstats redir       : {stats['d_km_gated']}  -> "
      f"{'EQUAL' if stats['d_km_gated'] == 0 else 'DIFFERENT'}")
print(f"\n  frame  bad_redirect       (UNGATED)      : {stats['ungated_bad']}")
print(f"  frame  bad_redirect_sfgated (GATED)      : {stats['gated_bad']}")
print(f"  gauge  v1 bad family                     : {stats['v1_bad']}")
print(f"  bad_redirect_sfgated vs v1 bad  disagree : {stats['d_gatedbad_v1bad']}  -> "
      f"{'EQUAL' if stats['d_gatedbad_v1bad'] == 0 else 'DIFFERENT'}")

with open(os.path.join(os.path.dirname(__file__), "gated_vs_gauge.json"), "w") as fh:
    json.dump({"keybo_file": keybo.__file__, "n": n, "stats": stats}, fh, indent=2)
print("\nwrote gated_vs_gauge.json")
