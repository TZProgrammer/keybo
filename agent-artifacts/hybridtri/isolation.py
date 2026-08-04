"""SERVED-PATH ISOLATION — my registry refactor deleted 108 src/ lines; check the SHIPPED VALUES.

"Every deletion was a line I widened" is a claim about MY OWN EDITS, so it is not evidence. This
checks the served path against values three siblings PUBLISHED, the route INTERPFRAME-1's
isolation.py took for exactly the same reason. A changed served column could not survive the
feature-matrix checksum.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis.shap_diff import shap_diff  # noqa: E402
from keybo.analysis.surfaces import C30M  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    TRIGRAM_FEATURE_NAMES,
    bigram_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402

WPM = 90.0
POS = [*G.slots, G.space_position]
out: dict = {"checks": {}}


def check(name, published, mine, tol):
    ok = abs(float(published) - float(mine)) <= tol if tol else published == mine
    out["checks"][name] = {"published": published, "mine": mine, "ok": bool(ok), "tol": tol}
    print(
        f"  {name:<34} published {published!s:>14}  mine {mine!s:>14}  {'OK' if ok else '** FAIL **'}"
    )
    return ok


print("SERVED-PATH ISOLATION vs published values")
allok = True
allok &= check("FEATURE_VERSION", "2026-07-05.3", FEATURE_VERSION, None)
allok &= check("served bigram n_columns", 20, len(BIGRAM_FEATURE_NAMES), None)
allok &= check("served trigram n_columns", 46, len(TRIGRAM_FEATURE_NAMES), None)
allok &= check("served bigram last column", "wpm", BIGRAM_FEATURE_NAMES[-1], None)
allok &= check("served trigram last column", "wpm", TRIGRAM_FEATURE_NAMES[-1], None)

X = np.vstack([bigram_features_from_positions(G, (a, b), wpm=WPM) for a in POS for b in POS])
allok &= check("served bigram matrix shape", "(961, 20)", str(X.shape), None)
# INTERPFRAME-1 isolation.json: a checksum a changed column could not survive
allok &= check("served bigram matrix checksum", 97940.954121, float(X.sum()), 1e-6)

surface = default_surface(WPM, None)
_, LAY_A = _resolve("flagship-c3")
_, LAY_B = _resolve("graphite")
# INTERPFRAME-1 negctl.json published these two as flagship-c3 / graphite (NOT C30M -- C30M/qwerty
# is 264.1389, which is exploit.py's own inv4.1 anchor; keeping both so the labels cannot be mixed).
allok &= check(
    "card() flagship-c3 ms/char", 254.9761, round(surface.card(LAY_A).ms_per_char, 4), 5e-5
)
allok &= check("card() graphite ms/char", 258.1696, round(surface.card(LAY_B).ms_per_char, 4), 5e-5)
allok &= check(
    "card() C30M/qwerty ms/char", 264.1389, round(surface.card(C30M).ms_per_char, 4), 5e-5
)

r = shap_diff(LAY_A, LAY_B, channel="both", target_wpm=WPM)
allok &= check("gap_total", 3.193444, round(r.gap_total, 6), 5e-6)
t2 = {c.feature: c.ms_per_char for c in r.t2.contributions}
tc = {c.feature: c.ms_per_char for c in r.tcond.contributions}
for k, pub in (("bottom", 0.7453), ("wpm", -0.0922), ("lateral", -0.1362), ("dx", 0.1678)):
    allok &= check(f"T2 {k}", pub, round(t2[k], 4), 5e-5)
for k, pub in (("bg2_bottom", 0.7382), ("bg1_bottom", -0.2337)):
    allok &= check(f"Tcond {k}", pub, round(tc[k], 4), 5e-5)

out["all_passed"] = bool(allok)
print()
print(f"ALL {len(out['checks'])} CHECKS {'PASSED' if allok else 'DID NOT PASS'}")
with open(f"{ARTIFACTS}/isolation.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"wrote {ARTIFACTS}/isolation.json")
