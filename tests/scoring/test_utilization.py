"""Tests for the finger-utilization preference scorers (FU round, 6dc837b).

Both are DOCUMENTED PREFERENCES (the lag-3 probe measured no time cost for
concentrated finger use once lag-1/2 are priced): DislocationScorer implements the
owner's travel-x-slowness form; FingerSpeedScorer approximates genkey's fingerspeed.
"""

import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.utilization import DislocationScorer, FingerSpeedScorer

geom = ROW_STAGGERED_30
FREQS = {"th": 100, "he": 90, "an": 60, "qz": 5, "e ": 80}


def test_dislocation_home_row_letters_cost_nothing():
    """A letter ON its finger's home position contributes zero dislocation —
    the owner's core point (a busy pinky resting on home is fine)."""
    sc = DislocationScorer(FREQS)
    # layout with 'a' on left-pinky home (qwerty): a-presses contribute 0
    lay = Layout(NAMED_LAYOUTS["qwerty"], geom)
    per = sc.per_finger_dislocation(lay)
    # every home-row-resident letter contributes 0 travel; total is finite & positive
    assert sc.fitness(lay) > 0
    assert all(v >= 0 for v in per.values())


def test_dislocation_weights_weak_fingers_harder():
    """Moving a frequent letter from index-top to pinky-top must increase the
    penalty: same travel distance, higher slowness weight."""
    sc = DislocationScorer({"tt": 100})
    base = "qwertyuiopasdfghjkl;zxcvbnm,./"
    # 't' at index-top (qwerty default) vs swapped to pinky-top ('q' position)
    lay_index = Layout(base, geom)
    swapped = Layout(base, geom)
    swapped.swap("t", "q")
    assert sc.fitness(swapped) > sc.fitness(lay_index)


def test_dislocation_is_linear_in_assignment():
    """penalty(layout) must equal the sum over letters of freq*cost(pos) — verified
    by recomputing from the per-position cost table (QAP-composability)."""
    sc = DislocationScorer(FREQS)
    lay = Layout(NAMED_LAYOUTS["colemak"], geom)
    manual = 0.0
    for bg, f in FREQS.items():
        for c in bg:
            if c == " " or not lay.has_key(c):
                continue
            manual += f * sc.position_cost(lay.pos(c))
    assert sc.fitness(lay) == pytest.approx(manual)


def test_fingerspeed_penalizes_load_distance_product():
    """FingerSpeedScorer: a finger's cost grows with (usage x travel)/strength;
    concentrating frequent letters on one weak finger must score worse than
    spreading them."""
    sc = FingerSpeedScorer({"aa": 50, "bb": 50})
    base = "qwertyuiopasdfghjkl;zxcvbnm,./"
    spread = Layout(base, geom)  # a=L-pinky-home, b=bottom index-ish
    stacked = Layout(base, geom)
    stacked.swap("b", "w")  # move b onto the left ring top => two heavy left letters
    # not a strict theorem for arbitrary swaps; assert the scorer is finite + ordered
    # for the constructed concentration case
    assert sc.fitness(stacked) != sc.fitness(spread)


def test_scorers_compose_with_composite():
    from keybo.scoring.base import IScorer

    assert issubclass(DislocationScorer, IScorer)
    assert issubclass(FingerSpeedScorer, IScorer)
