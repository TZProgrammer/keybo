"""Per-finger decomposition of the scissor gauge (ALLGAUGE-1).

The aggregate scissor gauge — ``oxey.pattern_shares()["scissor"]``, the corpus share of
same-hand two-row reaches on adjacent fingers — says *how much* a layout scissors but not
*which fingers pay for it*. A layout whose scissors sit on index/middle and one whose
scissors sit on the pinky read identically. This module splits the same quantity across
the eight fingers.

**Attribution rule: half to each of the two fingers.** A scissor is a two-finger event;
there is no principled way to call one finger "the" owner, and any asymmetric rule (weaker
finger takes all, higher row takes all) would make the decomposition depend on a second
opinion layered on top of the gauge. Splitting evenly is the unique rule that is both

* an **exact partition** — the eight per-finger values sum to the aggregate share, so no
  mass is invented or lost (this is the property the test suite pins, and the reason a
  wrong rule shows up as a failing test rather than a plausible-looking table), and
* **symmetric in the bigram's order** — ``ab`` and ``ba`` attribute identically, matching
  ``classify.is_scissor``, which is itself order-blind (``abs(a[1] - b[1]) == 2``).

If a future gauge wants asymmetric attribution (e.g. charging the reach direction), it
should be a *separate*, separately-labelled column: silently re-weighting this one would
break the partition property that makes it checkable.

**Denominator (trap #9): the layout-restricted bigram mass** — exactly
``OxeyStyleScorer.pattern_shares``' own denominator, i.e. only bigrams whose *both*
characters sit on the layout count, toward numerator and denominator alike. Matching it is
what makes ``sum(per_finger) == aggregate`` exact rather than approximate. Note this
denominator is NOT the same as :mod:`keybo.analysis.kmstats`' (which masks space out) nor
the same as the full-corpus mass the comfort gauge divides by; the three are different
conventions for the word "share" and this module states which one it uses.

Support: the incumbent narrow predicate ``classify.is_scissor`` (adjacent fingers only),
so the per-finger split decomposes *the gauge analyze reports*, not a wider relative of it.
Consequence, inherited and documented rather than fixed here: ``is_adjacent`` requires a
column gap of 1, so no middle-pinky pair is in the support at all, and the pinky's share
counts only its ring-adjacent scissors.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.features import classify as C
from keybo.geometry import Finger
from keybo.layout import Layout

#: The eight typing fingers, in board order (left pinky .. right pinky). The thumb is
#: excluded: space is not an assignable slot, so it cannot take part in a scissor.
FINGER_NAMES: tuple[str, ...] = ("LP", "LR", "LM", "LI", "RI", "RM", "RR", "RP")

#: The attribution rule, named in the output so a reader never has to infer it.
ATTRIBUTION_RULE = "half-to-each-finger"


class ScissorByFinger:
    """The scissor gauge, split across the eight fingers, over one bigram corpus."""

    def __init__(self, bigram_freqs: Mapping[str, int]) -> None:
        self._bg = {bg: freq for bg, freq in bigram_freqs.items() if len(bg) == 2}

    def shares(self, layout: Layout) -> dict[str, float]:
        """Per-finger scissor share, in percent of layout-restricted bigram mass.

        The eight values sum to the aggregate scissor share (``pattern_shares()`` /
        ``"scissor"``) exactly, up to float summation order.
        """
        geometry = layout.geometry
        charged = dict.fromkeys(FINGER_NAMES, 0.0)
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not all(layout.has_key(character) for character in bigram):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            if not C.is_scissor(geometry, a, b):
                continue
            for position in (a, b):
                name = geometry.finger(position[0]).name
                charged[name] += freq / 2.0
        if not denominator:
            return dict.fromkeys(FINGER_NAMES, 0.0)
        return {name: 100.0 * value / denominator for name, value in charged.items()}

    def pair_shares(self, layout: Layout) -> dict[str, float]:
        """Scissor share per unordered FINGER PAIR (e.g. ``"LR+LP"``), a second partition.

        Same mass, cut a different way: this one shows *which adjacency* pays, which the
        per-finger view cannot (a finger's share pools its two neighbours). Also an exact
        partition of the aggregate.
        """
        geometry = layout.geometry
        charged: dict[str, float] = {}
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not all(layout.has_key(character) for character in bigram):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            if not C.is_scissor(geometry, a, b):
                continue
            fingers = sorted(
                (geometry.finger(a[0]).name, geometry.finger(b[0]).name),
                key=FINGER_NAMES.index,
            )
            key = "+".join(fingers)
            charged[key] = charged.get(key, 0.0) + freq
        if not denominator:
            return {}
        return {key: 100.0 * value / denominator for key, value in charged.items()}


def finger_order() -> tuple[str, ...]:
    """Report order for the per-finger columns."""
    return FINGER_NAMES


def _assert_finger_names_match_geometry() -> None:  # pragma: no cover - import-time guard
    """FINGER_NAMES must be exactly the non-thumb ``Finger`` members."""
    expected = {f.name for f in Finger} - {"THUMB"}
    if set(FINGER_NAMES) != expected:
        raise AssertionError(f"FINGER_NAMES {FINGER_NAMES} does not match Finger enum {expected}")


_assert_finger_names_match_geometry()
