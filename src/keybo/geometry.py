"""The physical keyboard: key slots, finger assignment, and distances.

A :class:`Geometry` is a value object describing *where the keys physically are* and *which
finger presses each column* — independent of which character sits on which key (that is the
job of :class:`keybo.layout.Layout`). Isolating the geometry here keeps the row-stagger
constants and the finger map in one place, so features query the board instead of hardcoding
offsets.

The package ships a single instance, :data:`ROW_STAGGERED_30`: the standard 30-key,
three-row, row-staggered block that the training data was collected on. It is the only
geometry the learned models are valid for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import sqrt

Position = tuple[int, int]


class Finger(Enum):
    """The ten fingers, plus the thumb used for the space bar.

    ``L``/``R`` prefix the hand; ``P``/``R``/``M``/``I`` are pinky/ring/middle/index.
    """

    LP = "left-pinky"
    LR = "left-ring"
    LM = "left-middle"
    LI = "left-index"
    RI = "right-index"
    RM = "right-middle"
    RR = "right-ring"
    RP = "right-pinky"
    THUMB = "thumb"


# Absolute column (1..5) -> finger, per hand. Columns 1 and 2 are both the index finger.
_ABS_COLUMN_TO_FINGER = {
    5: ("LP", "RP"),
    4: ("LR", "RR"),
    3: ("LM", "RM"),
    2: ("LI", "RI"),
    1: ("LI", "RI"),
}


@dataclass(frozen=True)
class Geometry:
    """A physical board layout.

    Attributes:
        slots: the key positions in canonical order (top row left-to-right, then home,
            then bottom). ``Layout`` assigns characters to these slots by index.
        row_offsets: horizontal stagger applied to a key's column when measuring
            stagger-adjusted horizontal distance, keyed by row (y).
    """

    slots: tuple[Position, ...]
    row_offsets: dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.0, 3: -0.25})

    def finger(self, x: int) -> Finger:
        """Return the finger that presses column ``x`` (sign = hand, 0 = thumb)."""
        if x == 0:
            return Finger.THUMB
        left, right = _ABS_COLUMN_TO_FINGER[abs(x)]
        return Finger[left] if x < 0 else Finger[right]

    def hand(self, x: int) -> int:
        """Return -1 for the left hand, +1 for the right, 0 for the thumb/space."""
        return (x > 0) - (x < 0)

    def same_finger(self, x1: int, x2: int) -> bool:
        """Whether columns ``x1`` and ``x2`` are pressed by the same finger.

        Same finger means same hand AND same finger assignment — so index columns 1 and 2
        on one hand count as the same finger, while a column and its mirror on the other
        hand do not.
        """
        if x1 == 0 or x2 == 0:
            return False
        return self.finger(x1) == self.finger(x2)

    def stagger_adjusted_dx(self, a: Position, b: Position) -> float:
        """Absolute horizontal distance between two keys, accounting for row stagger.

        Rows without a stagger entry (notably the space/thumb row, y=0) contribute no
        offset.
        """
        ax, ay = a
        bx, by = b
        return abs((ax + self.row_offsets.get(ay, 0.0)) - (bx + self.row_offsets.get(by, 0.0)))

    def distance(self, a: Position, b: Position, ex: float = 2.0) -> float:
        """Euclidean (``ex=2``) distance between two key positions, ignoring stagger."""
        ax, ay = a
        bx, by = b
        if ex == 2.0:
            return sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        return (abs(ax - bx) ** ex + abs(ay - by) ** ex) ** (1.0 / ex)


def _row_staggered_30_slots() -> tuple[Position, ...]:
    """Build the canonical 30 slots: three rows of ten, top to bottom, left to right.

    Columns map to signed x as in the original project: the left five keys are -5..-1 and
    the right five are +1..+5 (there is no column 0 among the letter keys).
    """
    xs = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    slots: list[Position] = []
    for y in (3, 2, 1):  # top, home, bottom
        slots.extend((x, y) for x in xs)
    return tuple(slots)


ROW_STAGGERED_30 = Geometry(slots=_row_staggered_30_slots())
