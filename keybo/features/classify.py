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


def classify_bigram(layout: Layout, bigram: str) -> BigramClass:
    a, b = _positions(layout, bigram)
    g = layout.geometry
    if same_finger(g, a, b):
        return BigramClass.SAME_FINGER
    if same_hand(g, a, b):
        return BigramClass.SAME_HAND
    return BigramClass.ALTERNATE


def is_adjacent(geometry: Geometry, a: Position, b: Position) -> bool:
    """Same-hand keys on neighbouring fingers (adjacent columns)."""
    if not same_hand(geometry, a, b):
        return False
    return abs(abs(a[0]) - abs(b[0])) == 1


def is_lateral(x: int) -> bool:
    """A key in the inner index column (|x| == 1), reached by a lateral stretch."""
    return abs(x) == 1


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
    """Signed angle (degrees) from the outer to the inner key, or 0 for cross-hand."""
    if not same_hand(geometry, a, b):
        return 0.0
    if abs(a[0]) == abs(b[0]):
        return 0.0
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    ox, oy = outer
    ix, iy = inner
    off_o = geometry.row_offsets[oy]
    off_i = geometry.row_offsets[iy]
    hand = geometry.hand(a[0]) or 1
    angle = atan2((oy - iy), ((ox + off_o) - (ix + off_i)) * hand)
    return round(degrees(angle), 2)


def is_inwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """Rolling toward the index finger (outer key on the higher row)."""
    if not same_hand(geometry, a, b) or abs(a[0]) == abs(b[0]):
        return False
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] > inner[1]


def is_outwards(geometry: Geometry, a: Position, b: Position) -> bool:
    """Rolling toward the pinky (outer key on the lower row)."""
    if not same_hand(geometry, a, b) or abs(a[0]) == abs(b[0]):
        return False
    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] < inner[1]
