"""``bad-scissor`` — a same-hand reach in which the WEAKER finger must descend.

Implements the specification in ``state/badscissor/badscissor-spec.md`` (the ``badscissor``
agent, 2026-07-26) exactly. That document derived the predicate, the severity decision, the
denominator and the attribution rule from the Aalto keystroke frame and pinned expected
values before this code existed; ``tests/analysis/test_bad_scissor.py`` asserts them.
Nothing here is re-decided, and nothing is invented.

    bad-scissor fires  <=>  same hand AND different fingers AND different rows
                            AND the WEAKER finger of the pair is on the LOWER row

**Why this and not the incumbent gauges.** ``classify.is_scissor`` (narrow) and the served
objective's wide support both gate on ``abs(dy) == 2``. This predicate gates on *which
finger descends* instead, which makes it a **cross-cut** of both rather than a superset:

============================================  =====
ordered position pairs on ``ROW_STAGGERED_30``  900
narrow ``is_scissor``                            24
wide (same-hand, distinct-finger, dy == 2)       72
``bad-scissor``                                 108
  ...of which dy == 1                            72
  ...of which dy == 2                            36
  ...of which middle-pinky                       12
narrow \\ bad  (excluded: all weak-on-TOP)       12
wide   \\ bad  (excluded: all weak-on-TOP)       36
============================================  =====

The excluded pairs are the ones the spec's fit measured as *not* strained (the weak-on-top
wide class measures -0.0179, i.e. faster than the same-row baseline, at n=1.64M). So this
gauge drops half of the incumbent's own support and adds 72 single-row descents neither
incumbent can see. Because the supports are **not nested**, comparing it against narrow or
wide is a meaningful check rather than trap #11's nested-guard mistake — but per the spec's
§6.4, correlation with them is still not independent corroboration of anything.

**Severity: FLAT** (1.0 per qualifying bigram). Per-pair graded weights are derivable but
rest on 2-5 bigram identities per pair and do not produce a robust board ordering, so the
spec ships flat and this module implements only flat. Notably ``bad-scissor-dist`` is
deliberately absent: the spec tested vertical distance as the severity axis and **refuted**
it (the distance coefficients come out negative), so there is no distance-weighted variant
to wire.

**Denominator (trap #9): layout-restricted, space-EXCLUDED bigram mass** — the
``kmstats``/``sfb``/``lsb`` convention, NOT ``oxey.pattern_shares``'. ``Layout.has_key(" ")``
is True, so the oxey convention silently counts space-touching bigrams in the denominator.
Space is in no bad-scissor pair (``hand(0) == 0``), so choosing wrong leaves the
**numerator bit-identical** and inflates every share by a plausible ~1.497x constant. That
is exactly the failure the campaign's trap #9 describes, and
``test_the_space_including_denominator_would_inflate_every_share_by_about_1_497x`` pins it.

**Attribution: the whole of a pair's mass to the DESCENDING (weaker) finger** — not to both,
not split. The predicate is an asymmetric statement about one finger: in the spec's fit the
descending weak finger measures +0.5453 while the strong finger's position measures -0.1083,
so charging both would credit the strong finger with strain the data says it does not bear.
It also keeps the decomposition an exact partition. Structural consequence, expected and
tested rather than a bug: **both index fingers are always 0.0**, because the index is the
most dextrous finger and so is never the weaker member of any pair.

⚠ **This is a MEASUREMENT/DIAGNOSIS gauge — not a search objective.** WSCISSOR-GEN-1
(ledger ``44d282b``) showed that optimizing a scissor-severity axis is optimizing the ruler:
champions won 1 of 19 gauges with a negative normalized floor, and an arm with no severity
axis behaved identically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from keybo.features import classify as C
from keybo.geometry import Geometry, Position
from keybo.layout import Layout

#: Dexterity rank, least -> most dextrous. Matches the campaign's ``_DEXTERITY_RANK`` and is
#: the reverse of ``tb_objective_ref._KIND_ORDER``, so "weaker finger" means the same thing
#: across the campaign's artifacts. Inverting this silently moves mass onto the index
#: fingers, which the test suite catches.
_DEX: Mapping[str, int] = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}

#: The attribution rule, named in the output so a reader never has to infer it.
ATTRIBUTION_RULE = "all-to-descending-weaker-finger"

#: Report order for the per-finger columns (both index entries are structurally 0.0).
FINGER_ORDER: tuple[str, ...] = (
    "L-pinky",
    "L-ring",
    "L-middle",
    "L-index",
    "R-index",
    "R-middle",
    "R-ring",
    "R-pinky",
)

#: Rows the direction semantics are defined for (3 top, 2 home, 1 bottom).
_EXPECTED_ROWS = frozenset({1, 2, 3})


def _kind(geometry: Geometry, x: int) -> str:
    """``'right-index' -> 'index'``. Hand-independent."""
    return geometry.finger(x).value.split("-")[1]


def _weak_and_strong(
    geometry: Geometry, a: Position, b: Position
) -> tuple[str, int, int, int, str]:
    """``(weak kind, weak x, weak row, strong row, strong kind)`` for a two-finger pair."""
    ka, kb = _kind(geometry, a[0]), _kind(geometry, b[0])
    if _DEX[ka] <= _DEX[kb]:
        return ka, a[0], a[1], b[1], kb
    return kb, b[0], b[1], a[1], ka


def bad_scissor(geometry: Geometry, a: Position, b: Position) -> bool:
    """Whether the ordered bigram ``a -> b`` is a bad scissor.

    A property of the two POSITIONS, so it is symmetric: ``bad_scissor(g, a, b) ==
    bad_scissor(g, b, a)`` for all 900 pairs (asserted exhaustively). Direction of travel is
    priced by the corpus supplying both orderings, not by the flag.
    """
    if not C.same_hand(geometry, a, b):
        return False
    if C.same_finger(geometry, a, b):  # index cols 1&2, K31 pinky cols 5&6
        return False
    if a[1] == b[1]:  # no row travel
        return False
    _weak, _weak_x, weak_y, strong_y, _strong = _weak_and_strong(geometry, a, b)
    return weak_y < strong_y  # rows: 3 = top, 2 = home, 1 = bottom


def bad_scissor_finger(geometry: Geometry, a: Position, b: Position) -> str | None:
    """Hand-qualified name of the finger the pair's mass is attributed to, or ``None``.

    Always the descending (weaker) finger, e.g. ``"L-pinky"``. Never an index finger.
    """
    if not bad_scissor(geometry, a, b):
        return None
    weak, weak_x, _weak_y, _strong_y, _strong = _weak_and_strong(geometry, a, b)
    return f"{'L' if weak_x < 0 else 'R'}-{weak}"


def bad_scissor_cell(geometry: Geometry, a: Position, b: Position) -> str | None:
    """The pair's ``"<finger-pair> dy<n>"`` class, e.g. ``"index-pinky dy1"``, or ``None``.

    Finger pairs are named most-dextrous-first so the label matches the specification's
    tables (``index-pinky``, not ``pinky-index``).
    """
    if not bad_scissor(geometry, a, b):
        return None
    ka, kb = _kind(geometry, a[0]), _kind(geometry, b[0])
    first, second = sorted((ka, kb), key=lambda kind: -_DEX[kind])
    return f"{first}-{second} dy{abs(a[1] - b[1])}"


class BadScissor:
    """The ``bad-scissor`` share, and its two exact decompositions, over one bigram corpus."""

    def __init__(self, bigram_freqs: Mapping[str, int]) -> None:
        self._bg = {bg: freq for bg, freq in bigram_freqs.items() if len(bg) == 2}

    # -- gauges ---------------------------------------------------------------------------

    def share(self, layout: Layout, *, exclude_space: bool = True) -> float:
        """Bad-scissor mass as a percent of layout-restricted, space-excluded bigram mass.

        ``exclude_space=False`` selects the ``oxey`` denominator, which is **wrong for this
        gauge** and exists only so the trap-#9 regression test can measure the ~1.497x
        inflation it causes. Production callers leave it alone.
        """
        return self.share_of(layout, bad_scissor, exclude_space=exclude_space)

    def share_of(
        self,
        layout: Layout,
        predicate: Callable[[Geometry, Position, Position], bool],
        *,
        exclude_space: bool = True,
    ) -> float:
        """This gauge's scoring loop and denominator, run on an arbitrary pair predicate.

        Exposed so the ``sfb`` positive control can drive OUR denominator over a predicate
        whose value is independently known from :mod:`keybo.analysis.kmstats`. Because
        ``sfb``'s support is disjoint from bad-scissor's, any disagreement isolates the
        denominator rather than the predicate — which is the whole point of trap #9.
        """
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        numerator = 0.0
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            if predicate(geometry, a, b):
                numerator += freq
        return 100.0 * numerator / denominator if denominator else 0.0

    def by_finger(self, layout: Layout, *, exclude_space: bool = True) -> dict[str, float]:
        """Share attributed to each finger — an exact partition of :meth:`share`.

        Both index entries are always 0.0 (see the module docstring).
        """
        return self._partition(layout, bad_scissor_finger, FINGER_ORDER, exclude_space)

    def by_cell(self, layout: Layout, *, exclude_space: bool = True) -> dict[str, float]:
        """Share per ``"<finger-pair> dy<n>"`` class — a second exact partition.

        The ``dy2`` subtotal is the number that motivates the predicate: it is under a tenth
        of the priced mass, and ``dy == 2`` is the *only* thing the incumbent gauges see.
        """
        return self._partition(layout, bad_scissor_cell, (), exclude_space)

    # -- internals ------------------------------------------------------------------------

    @staticmethod
    def _counts(layout: Layout, bigram: str, exclude_space: bool) -> bool:
        """Whether a bigram is inside the denominator (and so eligible for the numerator)."""
        if exclude_space and " " in bigram:
            return False
        return all(layout.has_key(character) for character in bigram)

    def _partition(
        self,
        layout: Layout,
        classifier: Callable[[Geometry, Position, Position], str | None],
        preset_keys: tuple[str, ...],
        exclude_space: bool,
    ) -> dict[str, float]:
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        charged = dict.fromkeys(preset_keys, 0.0)
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            key = classifier(geometry, a, b)
            if key is not None:
                charged[key] = charged.get(key, 0.0) + freq
        if not denominator:
            return dict.fromkeys(preset_keys, 0.0)
        return {key: 100.0 * value / denominator for key, value in charged.items()}

    @staticmethod
    def _check_geometry(geometry: Geometry) -> None:
        """Refuse a board this gauge's row semantics are not defined for.

        "The weaker finger is on the lower row" is well defined on any board, but the
        spec's expected values, its dy census and its severity evidence are all derived on
        the three-row block. A four-row board would score silently against a support the
        specification never examined, so refuse instead — the same stance
        :mod:`keybo.scoring.scissor_severity` takes.
        """
        rows = {y for _x, y in geometry.slots}
        if not rows <= _EXPECTED_ROWS:
            raise ValueError(
                "bad-scissor is defined for the three-row board "
                f"(rows {sorted(_EXPECTED_ROWS)}); got rows {sorted(rows)}."
            )
