"""Tests for the physical-board geometry.

Coordinate system (inherited from the original project, verified against it):
- x is signed: left hand < 0, right hand > 0, |x| in 1..5. Columns 1 and 2 are BOTH the
  index finger. Space/thumb is x == 0.
- y is the row: top = 3, home = 2, bottom = 1. Space is y == 0.
- Row-stagger offsets applied to x when measuring horizontal distance:
  top row -0.25, home 0.0, bottom +0.5.
"""

from math import sqrt

import pytest

from keybo.geometry import ROW_STAGGERED_30, Finger, Geometry


def test_ships_a_row_staggered_30_key_geometry():
    assert isinstance(ROW_STAGGERED_30, Geometry)


def test_has_30_slots_in_canonical_row_order():
    slots = ROW_STAGGERED_30.slots
    assert len(slots) == 30
    # Top row (y=3), left-to-right.
    assert slots[0] == (-5, 3)
    assert slots[4] == (-1, 3)
    assert slots[5] == (1, 3)
    assert slots[9] == (5, 3)
    # Home row (y=2).
    assert slots[10] == (-5, 2)
    assert slots[19] == (5, 2)
    # Bottom row (y=1).
    assert slots[20] == (-5, 1)
    assert slots[29] == (5, 1)


def test_slots_are_all_distinct():
    assert len(set(ROW_STAGGERED_30.slots)) == 30


@pytest.mark.parametrize(
    "x,expected",
    [
        (-5, Finger.LP),
        (-4, Finger.LR),
        (-3, Finger.LM),
        (-2, Finger.LI),
        (-1, Finger.LI),
        (1, Finger.RI),
        (2, Finger.RI),
        (3, Finger.RM),
        (4, Finger.RR),
        (5, Finger.RP),
        (0, Finger.THUMB),
    ],
)
def test_finger_maps_signed_column_to_finger(x, expected):
    assert ROW_STAGGERED_30.finger(x) == expected


@pytest.mark.parametrize(
    "x,expected_hand",
    [(-5, -1), (-1, -1), (1, 1), (5, 1), (0, 0)],
)
def test_hand_is_the_sign_of_x(x, expected_hand):
    assert ROW_STAGGERED_30.hand(x) == expected_hand


def test_same_finger_is_same_finger_same_hand():
    # Columns 1 and 2 are both the (same) index finger -> same finger.
    assert ROW_STAGGERED_30.same_finger(-1, -2) is True
    assert ROW_STAGGERED_30.same_finger(1, 2) is True
    # Index vs middle -> different fingers (this is the SFB bug the old code got wrong).
    assert ROW_STAGGERED_30.same_finger(-2, -3) is False
    # Same column number but opposite hands -> different finger (mirror, not same).
    assert ROW_STAGGERED_30.same_finger(-3, 3) is False


def test_row_offsets_match_the_row_stagger():
    assert ROW_STAGGERED_30.row_offsets == {1: 0.5, 2: 0.0, 3: -0.25}


def test_stagger_adjusted_dx_applies_row_offsets():
    # home index (-2,2) to top index (-1,3):
    # (-2 + 0.0) - (-1 + -0.25) = -2 - (-1.25) = -0.75 -> 0.75
    assert ROW_STAGGERED_30.stagger_adjusted_dx((-2, 2), (-1, 3)) == pytest.approx(0.75)
    # top pinky (-5,3) to bottom pinky (-5,1):
    # (-5 + -0.25) - (-5 + 0.5) = -0.75 -> 0.75
    assert ROW_STAGGERED_30.stagger_adjusted_dx((-5, 3), (-5, 1)) == pytest.approx(0.75)


def test_stagger_adjusted_dx_handles_the_space_key():
    # Space is at (0, 0); its row has no stagger (offset 0). A bigram involving space must
    # not raise (space is the most frequent character in real data).
    # home index (-2,2) to space (0,0): (-2 + 0.0) - (0 + 0.0) = -2 -> 2.0
    assert ROW_STAGGERED_30.stagger_adjusted_dx((-2, 2), (0, 0)) == pytest.approx(2.0)
    assert ROW_STAGGERED_30.stagger_adjusted_dx((0, 0), (-1, 3)) == pytest.approx(1.25)


def test_distance_is_raw_euclidean_without_stagger():
    # Same column, two rows apart: dx=0, dy=2 -> 2.0
    assert ROW_STAGGERED_30.distance((-5, 3), (-5, 1)) == pytest.approx(2.0)
    # One column, one row apart: sqrt(2)
    assert ROW_STAGGERED_30.distance((-2, 2), (-1, 3)) == pytest.approx(sqrt(2))


def test_distance_is_symmetric():
    a, b = (-3, 3), (2, 1)
    assert ROW_STAGGERED_30.distance(a, b) == pytest.approx(ROW_STAGGERED_30.distance(b, a))


def test_geometry_is_immutable():
    with pytest.raises((AttributeError, TypeError)):
        ROW_STAGGERED_30.row_offsets = {}  # type: ignore[misc]
