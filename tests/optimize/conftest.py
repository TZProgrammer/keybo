"""Shared test scorer for the optimizer tests.

A real char->position assignment landscape (not a position-only plateau): earlier letters
in the weight string are "more frequent" and prefer low-effort central home keys, so
swapping characters genuinely changes fitness. This gives SA and local search a gradient to
follow and a non-trivial optimum to find.
"""

from keybo.layout import Layout
from keybo.scoring.base import IScorer

_CHAR_WEIGHT = {c: w for w, c in enumerate("etaoinshrdlcumwfgypbvkjxqz',.-", start=1)}


class CharPlacementScorer(IScorer):
    def _effort(self, x: int, y: int) -> float:
        row_effort = {2: 0.0, 3: 1.0, 1: 2.0}[y]  # home cheapest, bottom worst
        return abs(x) + row_effort

    def fitness(self, layout: Layout) -> float:
        return sum(_CHAR_WEIGHT[ch] * self._effort(*layout.pos(ch)) for ch in layout.chars)
