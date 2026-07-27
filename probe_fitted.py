"""lmscissor (d), independent evidence path: what do the FITTED surfaces say?

The raw-Aalto arbiter (probe_arbiter2.py) is one evidence path. This is a different one: the
production per-position-pair bigram time table `_T2`, built from the trained models. It is
available per source pool (AALTO / COMMUNITY / POOL) via analysis.surfaces, so it answers the
brief's "report per-source, and note POOL is not independent".

Two questions:
  1. What does the fitted table charge for the specific position pairs `bl` and `ld` occupy,
     on each layout — and does it charge MORE for the 2-row middle-pinky reach than for the
     dy=1 pinky descent?
  2. Scored over the whole corpus, which layout does each source prefer?

⚠ This is NOT independent of the campaign's ms/char numbers — it is the same surface family the
flagship comparison used. It IS independent of my raw-TSV cell aggregates, which is the point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}

print(f"surface families: {S.FAMILIES}, pools: {S.POOLS}, default {S.DEFAULT_FAMILY}")
print(f"surface names: {S.surface_names()}")

corpus_dir = production_corpus_dir(None)
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"corpus = {corpus_dir.name}")

# ---- the production K31 bigram time table (seed-mean), used to price position pairs ----
tri = load_frequencies(str(corpus_dir / "trigrams.txt"))
surf = TimeSurface(tri, target_wpm=90.0, geometry=G)
T2 = surf._T2  # (n, n) ms per ordered position pair, seed-mean over the 6 bigram models
positions = (*G.slots, G.space_position)
idx_of_pos = {p: i for i, p in enumerate(positions)}
print(f"T2 shape {T2.shape}")


def ms(a, b):
    return float(T2[idx_of_pos[a], idx_of_pos[b]])


print(f"\n{'='*100}\n1. WHAT THE FITTED BIGRAM TABLE CHARGES FOR THE CRUX POSITION PAIRS\n{'='*100}")
print("   (ms for the ordered position pair at 90 WPM; both orders shown)")
CASES = {
    "bl on keybo-lsb    : b(3,1)BOTTOM-middle -> l(5,3)TOP-pinky   dy2 nonadj": ((3, 1), (5, 3)),
    "bl on keybo-lsb+lm : b(3,1)BOTTOM-middle -> l(5,2)HOME-pinky  dy1 nonadj": ((3, 1), (5, 2)),
    "ld on keybo-lsb+lm : l(5,2)HOME-pinky    -> d(3,3)TOP-middle  dy1 nonadj": ((5, 2), (3, 3)),
    "ld on keybo-lsb    : l(5,3)TOP-pinky     -> d(3,3)TOP-middle  dy0 SAMEROW": ((5, 3), (3, 3)),
    "md on keybo-lsb    : m(5,2)HOME-pinky    -> d(3,3)TOP-middle  dy1 nonadj": ((5, 2), (3, 3)),
    "same-row ref       : (5,2)HOME-pinky     -> (3,2)HOME-middle  dy0": ((5, 2), (3, 2)),
}
for label, (a, b) in CASES.items():
    print(f"  {label}\n      {a}->{b} {ms(a,b):8.3f} ms    {b}->{a} {ms(b,a):8.3f} ms")

print(f"\n{'='*100}\n2. THE ORIENTATION CONTRAST IN THE FITTED TABLE (middle<->pinky, dy2)\n{'='*100}")
print("  pinky LOWER (bad-scissor FLAGS) vs pinky UPPER (bad-scissor EXCLUDES), same |dcol|=2:")
contrasts = [
    ("pinky BOTTOM(5,1) <-> middle TOP(3,3)   [flagged: weak lower]", (5, 1), (3, 3)),
    ("pinky TOP(5,3)    <-> middle BOTTOM(3,1) [EXCLUDED: weak on top]", (5, 3), (3, 1)),
    ("pinky HOME(5,2)   <-> middle TOP(3,3)   [flagged: weak lower, dy1]", (5, 2), (3, 3)),
    ("pinky TOP(5,3)    <-> middle HOME(3,2)  [EXCLUDED: weak on top, dy1]", (5, 3), (3, 2)),
]
for label, a, b in contrasts:
    avg = (ms(a, b) + ms(b, a)) / 2
    print(f"  {label}\n      mean of both orders: {avg:8.3f} ms")

print(f"\n{'='*100}\n3. PER-SOURCE CORPUS SCORE (ms/char) — which layout does each source prefer?\n{'='*100}")
per_source = {}
for name in S.surface_names():
    try:
        sf = S.default_surface(90.0, None) if False else None
    except Exception:
        sf = None
    per_source[name] = None

# Use the documented API path for per-pool surfaces.
print(f"  (surfaces dir contents: {[p.name for p in sorted(Path('data/surfaces').iterdir())]})")
print(f"  surfaces module API: {[n for n in dir(S) if not n.startswith('_')]}")

json.dump({"note": "see stdout"}, open("/tmp/lmscissor_fitted.json", "w"))
print("\n--- section 3 needs the per-pool loader; inspecting API above ---")
