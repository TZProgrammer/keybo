"""FM4 step 1: enumerate the NAME INTERSECTION between served frame columns and reported gauges.

Mechanical, not judged: the set of frame column names that literally match a reported gauge
name, under the campaign's own naming conventions. Both the exact-match set AND the
hyphen/underscore-normalized set are reported, because a reader does not distinguish
``bad_redirect`` from ``bad-redirect`` when reading two tables side by side.
"""

from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import keybo  # noqa: E402
from keybo.analysis.redirects import REDIRECT_CLASSES  # noqa: E402
from keybo.cli.analyze import GAUGE_NAMES  # noqa: E402
from keybo.features.schema import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.scoring.comfort import DEFAULT_COMFORT  # noqa: E402
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS, ORDERED_ROLL_SHARES  # noqa: E402

print("keybo.__file__ =", keybo.__file__)


def norm(name: str) -> str:
    """The form a READER conflates: case-folded, hyphen == underscore, bg1_/bg2_ stripped."""
    n = name.lower().replace("-", "_")
    for pref in ("bg1_", "bg2_"):
        if n.startswith(pref):
            n = n[len(pref) :]
    return n


# --- the REPORTED gauge universe ----------------------------------------------------------
# GAUGE_NAMES is the analyze board's own frame. The oxey pattern_shares block and the redirect
# family are ALSO printed by analyze (`--redirects`, the oxey-style component block), and the
# comfort term names are printed in its weight table, so a name colliding with any of those is
# equally readable-as-the-same-thing. All four sources are included and attributed.
GAUGE_SOURCES = {
    "analyze GAUGE_NAMES (the board)": list(GAUGE_NAMES),
    "oxey pattern_shares": [*DEFAULT_OXEY_WEIGHTS, *ORDERED_ROLL_SHARES],
    "redirect family (analyze --redirects)": list(REDIRECT_CLASSES),
    "comfort terms": list(DEFAULT_COMFORT),
}
gauge_by_norm: dict[str, list[str]] = {}
for src, names in GAUGE_SOURCES.items():
    for g in names:
        gauge_by_norm.setdefault(norm(g), []).append(f"{g} [{src}]")

FRAMES = {
    "BIGRAM_FEATURE_NAMES": list(BIGRAM_FEATURE_NAMES),
    "TRIGRAM_FEATURE_NAMES": list(TRIGRAM_FEATURE_NAMES),
}

rows = []
for frame, cols in FRAMES.items():
    for col in cols:
        hits = gauge_by_norm.get(norm(col))
        if hits:
            exact = col in {g.split(" [")[0] for g in hits}
            rows.append(
                {
                    "frame": frame,
                    "column": col,
                    "normalized": norm(col),
                    "exact_name_match": exact,
                    "gauges": sorted(set(hits)),
                }
            )

by_norm: dict[str, list[str]] = {}
for r in rows:
    by_norm.setdefault(r["normalized"], []).append(f'{r["frame"]}::{r["column"]}')

print(f"\n=== {len(by_norm)} DISTINCT colliding names, {len(rows)} (frame, column) instances ===")
for n in sorted(by_norm):
    hit = next(r for r in rows if r["normalized"] == n)
    print(f"\n  {n}   exact_match={hit['exact_name_match']}")
    print(f"    frame columns : {', '.join(by_norm[n])}")
    for g in hit["gauges"]:
        print(f"    gauge         : {g}")

print("\n=== gauge names with NO frame column of that name (for completeness) ===")
frame_norms = {norm(c) for cols in FRAMES.values() for c in cols}
print(
    "   ",
    ", ".join(sorted(g for g in gauge_by_norm if g not in frame_norms)),
)

out = {
    "keybo_file": keybo.__file__,
    "gauge_sources": GAUGE_SOURCES,
    "frames": FRAMES,
    "collisions": rows,
    "distinct_colliding_names": sorted(by_norm),
}
with open(os.path.join(os.path.dirname(__file__), "collide.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote collide.json")
