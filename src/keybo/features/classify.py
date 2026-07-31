"""Bigram classification and geometric predicates, derived purely from key positions.

Everything here is computed from the two positions passed in — there is no hidden reference
to a separate "real" keyboard. That matters: in the original code some features were taken
from the layout under evaluation while others came from the typed-on keyboard, an
inconsistency that only a single-source pipeline removes.

Positions are ``(x, y)`` with signed ``x`` (see :mod:`keybo.geometry`).
"""

from __future__ import annotations

from enum import Enum
from math import atan2, degrees

from keybo.geometry import Geometry, Position
from keybo.layout import Layout


class BigramClass(Enum):
    """The three motion classes the analysis found most predictive.

    ALTERNATE (different hands) is fastest; SAME_HAND (same hand, different fingers) is
    slower; SAME_FINGER (a true single-finger bigram) is slowest.
    """

    ALTERNATE = "alt"
    SAME_HAND = "shb"
    SAME_FINGER = "sfb"


def _positions(layout: Layout, bigram: str) -> tuple[Position, Position]:
    return layout.pos(bigram[0]), layout.pos(bigram[1])


def same_hand(geometry: Geometry, a: Position, b: Position) -> bool:
    ha, hb = geometry.hand(a[0]), geometry.hand(b[0])
    return ha != 0 and ha == hb


def same_finger(geometry: Geometry, a: Position, b: Position) -> bool:
    """A genuine single-finger bigram: same hand and same finger (incl. index cols 1 & 2)."""
    return geometry.same_finger(a[0], b[0])


def classify_positions(geometry: Geometry, a: Position, b: Position) -> BigramClass:
    """Classify a bigram from its two key positions."""
    if same_finger(geometry, a, b):
        return BigramClass.SAME_FINGER
    if same_hand(geometry, a, b):
        return BigramClass.SAME_HAND
    return BigramClass.ALTERNATE


def classify_bigram(layout: Layout, bigram: str) -> BigramClass:
    a, b = _positions(layout, bigram)
    return classify_positions(layout.geometry, a, b)


def is_adjacent(geometry: Geometry, a: Position, b: Position) -> bool:
    """Same-hand keys on neighbouring but *distinct* fingers.

    Columns 1 and 2 are both the index finger (see ``Geometry.same_finger``), so although
    they differ by one in |x| they are NOT adjacent fingers -- a bigram on them is a
    single-finger reach, not a two-finger roll/scissor. Excluding the same-finger case keeps
    ``is_adjacent``/``is_scissor`` from contradicting ``same_finger``.

    K31: the quote-slot column (|x| = 6) is the pinky, so {6, 4} is a pinky-ring pair --
    adjacent fingers despite the column gap of 2 (registered in the K31 charter). This
    branch cannot fire for 30-key positions (no |x| = 6 there).
    """
    if not same_hand(geometry, a, b):
        return False
    if same_finger(geometry, a, b):
        return False
    if {abs(a[0]), abs(b[0])} == {6, 4}:
        return True
    return abs(abs(a[0]) - abs(b[0])) == 1


def is_lateral(x: int) -> bool:
    """A key in a lateral-stretch column: inner index (|x| == 1) or, on K31, the outer
    pinky quote slot (|x| == 6). Both are the finger's off-home extra column, so the flag
    plays the same role it does for the index: it disambiguates the stretch column from
    the finger's home column (which the finger one-hot alone cannot)."""
    return abs(x) in (1, 6)


def is_lsb(geometry: Geometry, a: Position, b: Position) -> bool:
    """Lateral stretch bigram: adjacent index/middle fingers pulled apart horizontally."""
    if not same_hand(geometry, a, b):
        return False
    ax, ay = a
    bx, by = b
    index_middle = (abs(ax) in (1, 2) and abs(bx) == 3) or (abs(bx) in (1, 2) and abs(ax) == 3)
    return index_middle and geometry.stagger_adjusted_dx(a, b) > 1.5


def is_scissor(geometry: Geometry, a: Position, b: Position) -> bool:
    """A same-hand bigram spanning two rows on adjacent fingers (top<->bottom reach)."""
    if not is_adjacent(geometry, a, b):
        return False
    return abs(a[1] - b[1]) == 2


def rotation_angle(geometry: Geometry, a: Position, b: Position) -> float:
    """Signed roll angle (degrees) from the outer to the inner key.

    Rolls are TWO-finger motions: a single finger travelling between its own columns is a
    same-finger reach, not a roll, so same-finger pairs (like cross-hand pairs) have no roll
    angle. Their geometry is already captured by dx/dy/distance.
    """
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return 0.0
    if abs(a[0]) == abs(b[0]):
        return 0.0
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    ox, oy = outer
    ix, iy = inner
    off_o = geometry.row_offsets.get(oy, 0.0)
    off_i = geometry.row_offsets.get(iy, 0.0)
    hand = geometry.hand(a[0]) or 1
    angle = atan2((oy - iy), ((ox + off_o) - (ix + off_i)) * hand)
    return round(degrees(angle), 2)


def _roll_eligible(geometry: Geometry, a: Position, b: Position) -> bool:
    """Whether a position pair is a two-finger cross-column roll at all.

    The shared gate of all four roll-direction predicates, so the ordered and unordered
    readings below are guaranteed to describe the same universe of pairs and differ only in
    what they say ABOUT it. Three exclusions, each load-bearing:

    * cross-hand pairs are not rolls (no single hand travels);
    * same-FINGER pairs are not rolls but single-finger reaches — and this is the one that
      matters, because :meth:`keybo.geometry.Geometry.same_finger` counts index columns 1
      and 2 as one finger. Without it, ``(-2,2) -> (-1,2)`` would read as an inward step
      when it is the index finger reaching to its own second column. :mod:`keybo.scoring.oxey`
      records this exact error being fixed once already in its trigram path (its docstring:
      "the direction step was ``abs(b.column) - abs(a.column)``, which reads the index
      finger's two columns as a direction STEP"); this gate is what keeps it out of the
      bigram path.
    * equal |column| pairs have no horizontal direction to report. On the shipped geometries
      this is unreachable after the same-finger gate (equal |column| implies equal finger),
      so it is a guard against a future geometry, not live logic here.
    """
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return False
    return abs(a[0]) != abs(b[0])


# --- the UNORDERED pair: a property of the two KEYS ---------------------------------------
#
# ⚠ These two are SWAP-INVARIANT BY CONSTRUCTION and that is deliberate, not a bug left in
# place: they sort the pair by column magnitude and compare ROWS, so ``is_inwards(g, a, b) ==
# is_inwards(g, b, a)`` for all 870 ordered K30 pairs (pinned in
# ``tests/features/test_roll_direction_order.py``). What they actually measure is "is the
# key further from the index on the higher row" — a real geometric distinction, but NOT a
# direction of travel.
#
# They are kept unchanged, rather than fixed in place, because three callers depend on the
# unordered reading and two of those dependencies are irreversible:
#
# 1. ``features/ngram.py`` serves them as columns 18/19 (``inwards``/``outwards``) of
#    ``FEATURE_VERSION = "2026-07-05.3"``. Six trained models under ``data/models/k31/``
#    carry that stamp and ``models/base.py`` hard-errors on a mismatch. Changing what these
#    columns MEASURE while leaving the version string alone is exactly the train/serve skew
#    the stamp exists to prevent: the models would keep loading and score on a frame whose
#    column 18 no longer means what it meant during training. Bumping the version instead
#    would invalidate all six.
# 2. ``tests/features/test_k31_geometry.py`` asserts the K30 feature matrix bit-identical
#    against a frozen golden ``.npz``. These columns are in it.
# 3. ``analysis/effect_curves.py`` reads them as ``outer_high``/``outer_low`` — names it
#    chose precisely BECAUSE the predicates are unordered, with the proof in its docstring.
#    It wants the unordered semantics and would be wrong to receive ordered ones.
#
# The ordered predicates below are therefore ADDITIVE. That keeps every published
# ``inroll``/``outroll`` number in the ledger reproducible; a silent renumbering of shipped
# gauges is the outcome this split exists to avoid.


def is_inwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """Outer key on the higher row (UNORDERED — see the note above; not a stroke direction).

    Despite the name this cannot distinguish a stroke from its reverse. For the
    direction the stroke actually travelled, use :func:`is_inwards_ordered`.
    """
    if not _roll_eligible(geometry, a, b):
        return False
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] > inner[1]


def is_outwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """Outer key on the lower row (UNORDERED — see the note above; not a stroke direction).

    See :func:`is_outwards_ordered` for the direction of travel.
    """
    if not _roll_eligible(geometry, a, b):
        return False
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] < inner[1]


# --- the ORDERED stroke: which key was struck FIRST ----------------------------------------


def is_inwards_ordered(geometry: Geometry, a: Position, b: Position) -> bool:
    """A roll that TRAVELLED toward the index finger: struck ``a`` first, then ``b`` inboard.

    The honest direction-of-travel predicate. ``is_inwards_ordered(g, a, b)`` is the negation
    of ``is_inwards_ordered(g, b, a)`` on every roll-eligible pair — all 324 of them on K30 —
    which is the property the unordered pair above provably lacks.

    Two differences from :func:`is_inwards`, both consequences of asking the right question:

    * **It reads COLUMNS, not rows.** Direction of travel across the hand is horizontal, so
      the row is irrelevant to it; ``|column|`` decreasing means the stroke moved toward the
      index. The row's contribution to a roll's difficulty is a separate quantity, already
      carried by ``dy``/``distance``/``scissor`` and by the unordered predicates.
    * **Flat rolls now have a direction.** The row comparison left every same-row roll
      (108 of K30's 324 eligible pairs — a third of the roll universe) in neither class, even
      though a flat pinky-to-index roll is the least ambiguous inward stroke there is. The
      ordered pair partitions the eligible set exactly: 162 inward, 162 outward, none left
      over.
    """
    if not _roll_eligible(geometry, a, b):
        return False
    return abs(b[0]) < abs(a[0])


def is_outwards_ordered(geometry: Geometry, a: Position, b: Position) -> bool:
    """A roll that TRAVELLED toward the pinky: struck ``a`` first, then ``b`` outboard.

    The exact complement of :func:`is_inwards_ordered` on the roll-eligible set, so the two
    partition it rather than merely being disjoint.
    """
    if not _roll_eligible(geometry, a, b):
        return False
    return abs(b[0]) > abs(a[0])
