from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import keybo.analysis.timecard as timecard
from keybo.analysis.timecard import TimeSurface
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31

QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"


def _surface(trigrams, geometry=ROW_STAGGERED_30):
    surface = object.__new__(TimeSurface)
    surface.geometry = geometry
    surface._n = len(geometry.slots) + 1
    surface._T2 = np.zeros((surface._n, surface._n))
    surface._Tc = np.zeros((surface._n, surface._n, surface._n))
    surface._T2s = None
    surface._Tcs = None
    surface.tri = dict(trigrams)
    surface.total_mass = sum(trigrams.values())
    return surface


class _ZeroModel:
    metadata = SimpleNamespace(extra={})

    @staticmethod
    def predict_ms(features):
        return np.zeros(len(features))


def test_k31_constructor_builds_31_slots_plus_space(monkeypatch):
    monkeypatch.setattr(timecard, "_load_gz_model", lambda _stem: _ZeroModel())

    surface = TimeSurface({}, geometry=ROW_STAGGERED_31)

    assert surface._n == 32
    assert surface._T2.shape == (32, 32)
    assert surface._Tc.shape == (32, 32, 32)


def test_card_pins_units_denominators_attribution_and_coverage():
    surface = _surface({"qwe": 2, "qw ": 3, "q#q": 5})
    slot = {char: i for i, char in enumerate(QWERTY30M)}
    space = surface._n - 1
    surface._T2[slot["q"], slot["w"]] = 10.0
    surface._Tc[slot["q"], slot["w"], slot["e"]] = 1.0
    surface._Tc[slot["q"], slot["w"], space] = 2.0

    card = surface.card(QWERTY30M, ref_total_ms=100.0)

    assert card.total_ms == pytest.approx(58.0)
    assert card.ms_per_char == pytest.approx(11.6)
    assert card.saved_vs_ref_pct == pytest.approx(42.0)
    assert card.coverage_pct == pytest.approx(50.0)
    assert card.per_key_ms["w"] == pytest.approx(50.0)
    assert card.per_key_ms["e"] == pytest.approx(2.0)
    assert card.per_key_ms[" "] == pytest.approx(6.0)
    assert card.per_finger_ms["LR"] == pytest.approx(50.0)
    assert card.per_finger_ms["LM"] == pytest.approx(2.0)
    assert card.per_finger_ms["THUMB"] == pytest.approx(6.0)
    assert sum(card.per_key_ms.values()) == pytest.approx(card.total_ms)
    assert sum(card.per_finger_ms.values()) == pytest.approx(card.total_ms)
    assert card.top_bigrams == [("qw", 50.0)]


def test_card_reports_space_transitions_in_costliest_bigrams():
    surface = _surface({" qw": 2, "q w": 1})
    slot = {char: i for i, char in enumerate(QWERTY30M)}
    space = surface._n - 1
    surface._T2[space, slot["q"]] = 4.0
    surface._T2[slot["q"], space] = 3.0

    card = surface.card(QWERTY30M)

    assert card.total_ms == pytest.approx(11.0)
    assert card.per_key_ms["q"] == pytest.approx(8.0)
    assert card.per_key_ms[" "] == pytest.approx(3.0)
    assert card.top_bigrams == [(" q", 8.0), ("q ", 3.0)]


def test_seed_totals_pin_each_seed_instead_of_the_mean_table():
    surface = _surface({"qwe": 2, "qw ": 3, "q#q": 5})
    slot = {char: i for i, char in enumerate(QWERTY30M)}
    q, w, e, space = slot["q"], slot["w"], slot["e"], surface._n - 1
    t2_first = np.zeros_like(surface._T2)
    tc_first = np.zeros_like(surface._Tc)
    t2_second = np.zeros_like(surface._T2)
    tc_second = np.zeros_like(surface._Tc)
    t2_first[q, w] = 10.0
    tc_first[q, w, e] = 1.0
    tc_first[q, w, space] = 2.0
    t2_second[q, w] = 20.0
    tc_second[q, w, e] = 3.0
    tc_second[q, w, space] = 4.0
    surface._T2s = [t2_first, t2_second]
    surface._Tcs = [tc_first, tc_second]

    assert surface.seed_totals(QWERTY30M) == pytest.approx([58.0, 118.0])


def test_seed_totals_fail_loud_when_seed_tables_were_not_retained():
    with pytest.raises(ValueError, match="keep_seed_tables"):
        _surface({"qwe": 1}).seed_totals(QWERTY30M)


def test_k31_quote_slot_time_is_attributed_to_the_right_pinky():
    layout31 = QWERTY30M + ";"
    surface = _surface({"q;;": 1}, geometry=ROW_STAGGERED_31)
    slot = {char: i for i, char in enumerate(layout31)}
    surface._T2[slot["q"], slot[";"]] = 2.0
    surface._Tc[slot["q"], slot[";"], slot[";"]] = 3.0

    card = surface.card(layout31)

    assert card.total_ms == pytest.approx(5.0)
    assert card.per_key_ms[";"] == pytest.approx(5.0)
    assert card.per_finger_ms["RP"] == pytest.approx(5.0)
    assert card.per_finger_ms["LP"] == pytest.approx(0.0)
