"""Comfort scorer + composite objective (OQ-4; gaps-audit Phase C).

Comfort weights are PREFERENCES, not measurements — the module must keep them visibly
separate from the speed term (its own scorer, its own explicit weight, defaults documented
as opinions). First measured job (OQ-14): break the top-vs-home speed tie toward home.
"""

import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.comfort import DEFAULT_COMFORT, ComfortBigramScorer, CompositeScorer

QWERTY = NAMED_LAYOUTS["qwerty"]


class ConstScorer:
    def __init__(self, value):
        self.value = value

    def fitness(self, layout):
        return self.value


@pytest.fixture
def freqs():
    return {"th": 100, "he": 90, "an": 80, "ju": 20, "e ": 60}


def test_comfort_penalizes_off_home_and_sfb(freqs):
    scorer = ComfortBigramScorer(freqs)
    qwerty = scorer.fitness(Layout(QWERTY, ROW_STAGGERED_30))
    # colemak moves common letters home and kills most SFBs -> must be more comfortable.
    colemak = scorer.fitness(Layout(NAMED_LAYOUTS["colemak"], ROW_STAGGERED_30))
    assert colemak < qwerty


def test_comfort_breaks_the_home_top_tie(freqs):
    """Two layouts identical except a common letter sits on home vs top: comfort must
    prefer home (the OQ-14 tie-break job)."""
    base = list(QWERTY)
    # swap 'e' (top row) with 'd' (home row, same finger column): eqwrtyuiop asd->
    i_e, i_d = base.index("e"), base.index("d")
    swapped = base.copy()
    swapped[i_e], swapped[i_d] = swapped[i_d], swapped[i_e]
    scorer = ComfortBigramScorer({"ee": 100})  # only 'e' matters
    on_top = scorer.fitness(Layout("".join(base), ROW_STAGGERED_30))
    on_home = scorer.fitness(Layout("".join(swapped), ROW_STAGGERED_30))
    assert on_home < on_top


def test_composite_is_a_weighted_sum(freqs):
    speed = ConstScorer(100.0)
    comfort = ConstScorer(10.0)
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    assert CompositeScorer(speed, comfort, comfort_weight=0.0).fitness(lay) == 100.0
    assert CompositeScorer(speed, comfort, comfort_weight=2.0).fitness(lay) == 120.0


def test_default_weights_are_documented_preferences():
    # Every default carries an explanation string — comfort is opinion, not measurement.
    for name, (weight, why) in DEFAULT_COMFORT.items():
        assert isinstance(weight, float) and weight >= 0
        assert isinstance(why, str) and len(why) > 10, name
