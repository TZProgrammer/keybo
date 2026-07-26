"""Severity-weighted scissor gauge — a declared PREFERENCE, not a measurement.

The incumbent scissor gauge is a flat corpus share: ``oxey.pattern_shares()["scissor"]`` counts
every scissor as 1.0, and the only weight anywhere is the single ``"scissor": 4.0`` applied to the
aggregate. Two distinctions it cannot draw:

1. **Which fingers.** A two-row reach involving the pinky counts the same as one on index/middle.
2. **Which direction.** ``classify.is_scissor`` is ``abs(a[1] - b[1]) == 2`` — symmetric in its
   two arguments, hence direction-BLIND. Reaching DOWN to the bottom row scores identically to
   reaching UP to the top row.

This module adds both as **independently togglable, independently sweepable** multiplicative
weights, so a sensitivity sweep can attribute an effect to one component rather than to the bundle.

Epistemic status, load-bearing
------------------------------
There is **no human severity data** to calibrate against. So the weights here are a
**preference**, declared in advance in ``docs/scissor-severity-preregistration.md``, and every
result computed from them must be read as *"under stated preference P, layout X scores better on
scissors"* — never as *"X is ergonomically better"*. That is the same honesty convention
:mod:`keybo.scoring.oxey` documents for its own community weights, for the same reason.

At **all weights 1.0 on the narrow support the gauge reproduces the incumbent flat share
exactly** (test-pinned, max abs err 0.0). It is a strict generalization, not a rival metric.

Two structural facts about this board, both established by exhaustive enumeration and pinned by
test, because they decide what the gauge *can* express:

* **Rows are** ``y in {3 top, 2 home, 1 bottom}`` **and a scissor spans exactly two of them**, so
  every scissor is a top(3) <-> bottom(1) pair. A *static* "involves the bottom row" term is
  therefore identically true on the whole support — zero variance, so it can weight nothing. This
  is the degeneracy DIST-1 proved for ``scissor-vdist`` (row span always 2 => a rigid x2 rescale).
  The non-degenerate reading is the **signed** one: a bigram is ordered, so ``top_to_bottom``
  (reaching down second) and ``bottom_to_top`` (reaching up second) are genuinely different
  events that a corpus distinguishes. That is what :attr:`SeverityWeights.down` prices.
* **The narrow support contains no middle-pinky pair.** ``is_adjacent`` requires
  ``|dcolumn| = 1``, and middle (col 3) to pinky (col 5) is a gap of 2. So a pinky weighting on
  the narrow support *cannot* reach the middle-pinky mass at all; only the wide support can.

Direction convention matches the campaign's AXIS-2 objective (``bottom_to_top`` iff
``source_row < target_row``) so numbers are comparable across the two.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from keybo.features import classify as C
from keybo.geometry import Finger, Geometry, Position
from keybo.layout import Layout

#: Finger -> severity tier name. The tier is set by the WEAKEST finger in the pair.
_TIER_OF_FINGER: Mapping[Finger, str] = {
    Finger.LP: "pinky",
    Finger.RP: "pinky",
    Finger.LR: "ring",
    Finger.RR: "ring",
    Finger.LM: "other",
    Finger.RM: "other",
    Finger.LI: "other",
    Finger.RI: "other",
    Finger.THUMB: "other",
}

#: The two supports for the scissor predicate (component (c)).
SUPPORTS = ("narrow", "wide")

#: Rows this gauge's direction semantics are defined for (top, home, bottom).
_EXPECTED_ROWS = frozenset({1, 2, 3})


@dataclass(frozen=True)
class SeverityWeights:
    """One point in the preference space — every field is a knob the sweep varies.

    All weights are ``>= 1.0`` multipliers on a nonnegative indicator, so the gauge is monotone
    non-decreasing in each of them and a weight of 1.0 switches its component OFF. A weight below
    1.0 would turn a penalty into a reward, which is never the intent here, so it is rejected.

    Attributes:
        pinky: severity multiplier when either key is on the pinky (component (a)).
        ring_ratio: the ring tier's share of the pinky tier's *excess* severity, so
            ``ring = 1 + ring_ratio * (pinky - 1)``. One knob instead of two, and the
            monotonicity ``1 <= ring <= pinky`` holds by construction.
        down: severity multiplier when the second keystroke reaches DOWN to the bottom row
            (component (b)). ``bottom_to_top`` is the 1.0 reference.
        support: ``"narrow"`` = the incumbent ``is_scissor`` (adjacent fingers only);
            ``"wide"`` = DIST-1's ``wscissor``, the same predicate with the column-adjacency gate
            dropped, which is the only support where middle-pinky mass is visible (component (c)).
    """

    pinky: float = 1.0
    ring_ratio: float = 0.5
    down: float = 1.0
    support: str = "narrow"

    def __post_init__(self) -> None:
        for name in ("pinky", "down"):
            value = getattr(self, name)
            if not value >= 1.0:
                raise ValueError(
                    f"{name} weight must be >= 1.0 (a severity, not a reward): {value!r}"
                )
        if not 0.0 <= self.ring_ratio <= 1.0:
            raise ValueError(f"ring_ratio must lie in [0, 1]: {self.ring_ratio!r}")
        if self.support not in SUPPORTS:
            raise ValueError(f"support must be one of {SUPPORTS}: {self.support!r}")

    @property
    def ring(self) -> float:
        """The ring tier's weight, derived from ``pinky`` and ``ring_ratio``."""
        return 1.0 + self.ring_ratio * (self.pinky - 1.0)

    def tier_weight(self, tier: str) -> float:
        if tier == "pinky":
            return self.pinky
        if tier == "ring":
            return self.ring
        return 1.0

    def label(self) -> str:
        """Compact identity of this preference point, for artifact keys and report tables."""
        return f"{self.support}/pinky={self.pinky:g}/ratio={self.ring_ratio:g}/down={self.down:g}"


#: The PREREGISTERED headline preference P (docs/scissor-severity-preregistration.md §3).
DEFAULT_SEVERITY = SeverityWeights(pinky=2.0, ring_ratio=0.5, down=1.5, support="narrow")

#: All weights off: the positive control, which must reproduce the incumbent flat share exactly.
FLAT = SeverityWeights()


def _tier(geometry: Geometry, a: Position, b: Position) -> str:
    """The pair's severity tier — set by the weakest finger involved."""
    tiers = {_TIER_OF_FINGER[geometry.finger(a[0])], _TIER_OF_FINGER[geometry.finger(b[0])]}
    if "pinky" in tiers:
        return "pinky"
    if "ring" in tiers:
        return "ring"
    return "other"


def _direction(a: Position, b: Position) -> str:
    """Signed reach direction of the ordered bigram ``a -> b``.

    ``top_to_bottom`` means the SECOND keystroke lands on the lower row, i.e. the hand reaches
    DOWN. Convention matches the campaign's AXIS-2 objective.
    """
    return "bottom_to_top" if a[1] < b[1] else "top_to_bottom"


def bigram_severity(
    geometry: Geometry,
    a: Position,
    b: Position,
    weights: SeverityWeights = DEFAULT_SEVERITY,
) -> float:
    """Severity of the ordered bigram ``a -> b``: 0.0 if not a scissor, else the weight product.

    At ``SeverityWeights()`` this returns exactly 1.0 on the incumbent ``is_scissor`` support and
    0.0 elsewhere — i.e. the flat indicator the current gauge uses.
    """
    if not ScissorSeverity._in_support(geometry, a, b, weights.support):
        return 0.0
    severity = weights.tier_weight(_tier(geometry, a, b))
    if _direction(a, b) == "top_to_bottom":
        severity *= weights.down
    return severity


class ScissorSeverity:
    """Severity-weighted scissor share over one bigram corpus.

    The share is a percentage of the **layout-restricted** bigram mass — only bigrams whose both
    characters sit on the layout count, toward numerator or denominator. That is
    :meth:`keybo.scoring.oxey.OxeyStyleScorer.pattern_shares`' convention, and matching it exactly
    is what makes the positive control exact rather than approximate.
    """

    def __init__(self, bigram_freqs: Mapping[str, int]) -> None:
        self._bg = {bg: f for bg, f in bigram_freqs.items() if len(bg) == 2}

    # -- predicates ---------------------------------------------------------------------------

    @staticmethod
    def _in_wide_support(geometry: Geometry, a: Position, b: Position) -> bool:
        """DIST-1's ``wscissor``: same hand, two distinct fingers, row span 2, NO adjacency gate."""
        if not C.same_hand(geometry, a, b):
            return False
        if C.same_finger(geometry, a, b):
            return False
        return abs(a[1] - b[1]) == 2

    @staticmethod
    def _in_support(geometry: Geometry, a: Position, b: Position, support: str) -> bool:
        if support == "wide":
            return ScissorSeverity._in_wide_support(geometry, a, b)
        if support == "narrow":
            return C.is_scissor(geometry, a, b)
        raise ValueError(f"support must be one of {SUPPORTS}: {support!r}")

    # -- gauges -------------------------------------------------------------------------------

    def share(self, layout: Layout, weights: SeverityWeights = DEFAULT_SEVERITY) -> float:
        """Severity-weighted scissor mass as a percentage of layout-restricted bigram mass."""
        numerator, denominator = self._totals(layout, weights)
        return 100.0 * numerator / denominator if denominator else 0.0

    def breakdown(
        self, layout: Layout, weights: SeverityWeights = DEFAULT_SEVERITY
    ) -> dict[str, float]:
        """The share decomposed by ``tier|direction|adjacency`` class (an exact partition).

        Exact by construction: the classes are disjoint and every charged bigram lands in one, so
        the values sum to :meth:`share`. That is what lets the report attribute a movement to a
        class without the attribution silently losing or double-counting mass.
        """
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        parts: dict[str, float] = {}
        denominator = 0.0
        for bg, freq in self._bg.items():
            if not all(layout.has_key(ch) for ch in bg):
                continue
            denominator += freq
            a, b = layout.pos(bg[0]), layout.pos(bg[1])
            severity = bigram_severity(geometry, a, b, weights)
            if not severity:
                continue
            adjacency = "adjacent" if C.is_adjacent(geometry, a, b) else "nonadjacent"
            key = f"{_tier(geometry, a, b)}|{_direction(a, b)}|{adjacency}"
            parts[key] = parts.get(key, 0.0) + severity * freq
        if not denominator:
            return {}
        return {k: 100.0 * v / denominator for k, v in parts.items()}

    def class_masses(self, layout: Layout, support: str = "narrow") -> dict[str, float]:
        """UNWEIGHTED corpus share per ``tier|direction|adjacency`` class.

        This is the measurement the preference is applied *to*, reported separately so a reader
        can see how much of any conclusion is the data and how much is the weighting.
        """
        return self.breakdown(layout, SeverityWeights(support=support))

    # -- internals ---------------------------------------------------------------------------

    def _totals(self, layout: Layout, weights: SeverityWeights) -> tuple[float, float]:
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        numerator = 0.0
        denominator = 0.0
        for bg, freq in self._bg.items():
            if not all(layout.has_key(ch) for ch in bg):
                continue
            denominator += freq
            a, b = layout.pos(bg[0]), layout.pos(bg[1])
            severity = bigram_severity(geometry, a, b, weights)
            if severity:
                numerator += severity * freq
        return numerator, denominator

    @staticmethod
    def _check_geometry(geometry: Geometry) -> None:
        """Fail loudly on a board this gauge's direction semantics are not defined for.

        The ``down`` component means "reaches the BOTTOM row", which is only equivalent to
        "lands on the lower row" because a two-row span on a three-row board is always
        top <-> bottom. On a four-row board a home->top reach would also span two rows and would
        be silently mislabelled, so refuse instead.
        """
        rows = {y for _x, y in geometry.slots}
        if not rows <= _EXPECTED_ROWS:
            raise ValueError(
                "scissor severity is defined for the three-row board "
                f"(rows {sorted(_EXPECTED_ROWS)}); got rows {sorted(rows)}. The 'down' component "
                "assumes a two-row span is always top<->bottom, which fails on more rows."
            )
