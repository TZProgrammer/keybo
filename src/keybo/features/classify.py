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


def is_inwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """Whether the outer-column key sits on the HIGHER row. Two fingers, one hand.

    ⚠ SWAP-INVARIANT despite the name and the ordered signature (THEORY-1, ledger f4d126e):
    "outer" and "inner" are chosen by |column|, a property of the UNORDERED pair, so this
    cannot express which key was typed first. It is an orientation, not a direction. For
    the directional quantity the name suggests — "the motion ran toward the index finger" —
    use :func:`is_directed_inwards`.
    """
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return False
    if abs(a[0]) == abs(b[0]):
        return False
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] > inner[1]


def is_outwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """Whether the outer-column key sits on the LOWER row. Also SWAP-INVARIANT — see
    :func:`is_inwards`; the directional twin is :func:`is_directed_outwards`."""
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return False
    if abs(a[0]) == abs(b[0]):
        return False
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] < inner[1]


# --- direction of travel (v2 features) --------------------------------------------------
# Everything above is a function of the UNORDERED pair (except the landing-key one-hots
# built in ngram.py). Everything below is order-DEPENDENT by construction: swapping a and b
# changes the value. These back the opt-in BIGRAM_DIRECTION_NAMES block.


def signed_dx(geometry: Geometry, a: Position, b: Position) -> float:
    """Stagger-adjusted horizontal displacement WITH SIGN, from ``a`` to ``b``.

    :meth:`Geometry.stagger_adjusted_dx` takes an absolute value and is therefore a function
    of the unordered pair; this keeps the sign, making it the finest-grained statement of
    "which way did the hand move". Positive = rightward on the board. NOT hand-relative —
    :func:`dir_dx_inward` is the hand-relative version.
    """
    ax, ay = a
    bx, by = b
    return (bx + geometry.row_offsets.get(by, 0.0)) - (ax + geometry.row_offsets.get(ay, 0.0))


def dir_dx_inward(geometry: Geometry, a: Position, b: Position) -> float:
    """Column steps travelled TOWARD the index finger (positive) or the pinky (negative).

    Measured in |column| space, so it is hand-relative: on either hand a positive value
    means the motion ran inward. Cross-hand pairs have no such notion and give 0.0 — the
    convention :func:`rotation_angle` already uses where a roll is undefined.
    """
    if not same_hand(geometry, a, b):
        return 0.0
    return float(abs(a[0]) - abs(b[0]))


def directed_angle(geometry: Geometry, a: Position, b: Position) -> float:
    """The roll angle measured FROM ``a`` TO ``b`` — the directed twin of :func:`rotation_angle`.

    ``rotation_angle`` measures outer-key-to-inner-key, an unordered notion (hence its
    swap-invariance). Measuring a->b instead puts the direction of travel in the SIGN:
    reversing the bigram rotates the vector by 180 degrees.

    Undefined in the same cases (cross-hand, same finger, same column), returning 0.0, so
    this column is non-zero on exactly the pairs ``rotation_angle`` is non-zero on.
    """
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return 0.0
    ax, ay = a
    bx, by = b
    if abs(ax) == abs(bx):
        return 0.0
    off_a = geometry.row_offsets.get(ay, 0.0)
    off_b = geometry.row_offsets.get(by, 0.0)
    hand = geometry.hand(ax) or 1
    return round(degrees(atan2((by - ay), ((bx + off_b) - (ax + off_a)) * hand)), 2)


def is_directed_inwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """A TRUE inward roll: the SECOND key is nearer the index finger than the first.

    This is what the community means by an "inroll". It is not what :func:`is_inwards`
    computes. Two fingers on one hand only.
    """
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return False
    if abs(a[0]) == abs(b[0]):
        return False
    return abs(b[0]) < abs(a[0])


def is_directed_outwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """A TRUE outward roll: the second key is nearer the pinky. See :func:`is_directed_inwards`."""
    if not same_hand(geometry, a, b) or same_finger(geometry, a, b):
        return False
    if abs(a[0]) == abs(b[0]):
        return False
    return abs(b[0]) > abs(a[0])
