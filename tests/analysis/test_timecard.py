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


# --- the layout-string guard: a short or repeating layout must not score silently ---------


def test_card_REFUSES_a_short_layout_instead_of_silently_skipping_the_corpus():
    """The real incident: a driver passed the 6-character literal ``"qwerty"`` to ``card``.

    ``slot_of`` was built from whatever it was handed, so the 24 missing keys turned every
    n-gram touching them into a ``KeyError`` swallowed by ``continue`` — ~95% of corpus mass
    skipped while a well-formed TimeCard came back with a plausible ms/char. It was caught only
    as a 19.6x ``total_ms`` discrepancy between two runs.

    The message must name the ACTUAL length and duplicate count, because "invalid layout" alone
    would not have told that driver's author which of the two mistakes they had made.
    """
    surface = _surface({"qwe": 1})

    with pytest.raises(ValueError, match="30 DISTINCT characters") as excinfo:
        surface.card("qwerty")

    message = str(excinfo.value)
    assert "got 6 characters" in message, "the actual length is named"
    assert "6 distinct" in message and "0 duplicate" in message


def test_card_REFUSES_a_full_length_layout_with_a_DUPLICATE_character():
    """The likelier authoring mistake: 30 characters, one of them repeated.

    Length alone cannot catch this, and it fails the same silent way — two characters mapping
    to one slot drops the shadowed key's corpus mass. So the duplicate count is reported
    separately rather than folded into the length complaint.
    """
    duplicated = QWERTY30M[:-1] + QWERTY30M[0]  # 30 chars, 29 distinct: "q" twice, "-" gone
    assert len(duplicated) == 30 and len(set(duplicated)) == 29

    surface = _surface({"qwe": 1})

    with pytest.raises(ValueError, match="30 DISTINCT characters") as excinfo:
        surface.card(duplicated)

    message = str(excinfo.value)
    assert "got 30 characters" in message
    assert "29 distinct" in message and "1 duplicate" in message


def test_the_guard_sizes_itself_from_the_GEOMETRY_and_not_from_the_literal_30():
    """A 31-slot geometry must REQUIRE 31 and REJECT 30 — the mirror of the K30 case.

    This is the constraint that makes a hardcoded ``!= 30`` wrong: ``ROW_STAGGERED_31`` is
    supported and ``test_k31_quote_slot_time_is_attributed_to_the_right_pinky`` above passes a
    31-character layout, which such a guard would have broken. Asserting both directions is
    what distinguishes "sized from the geometry" from "happens to allow 31 as well".
    """
    layout31 = QWERTY30M + ";"
    surface31 = _surface({"qwe": 1}, geometry=ROW_STAGGERED_31)

    surface31.card(layout31)  # the valid 31-char case is unaffected

    with pytest.raises(ValueError, match="31 DISTINCT characters"):
        surface31.card(QWERTY30M)  # 30 is now the WRONG length

    with pytest.raises(ValueError, match="30 DISTINCT characters"):
        _surface({"qwe": 1}).card(layout31)  # ...and 31 is wrong on the 30-slot geometry


def test_a_VALID_layout_is_completely_unaffected_by_the_guard():
    """The guard must be a pure precondition: same numbers, byte for byte.

    Re-computes the pinned card from ``test_card_pins_units_denominators_attribution_and_coverage``
    so a regression that made the guard perturb ``slot_of`` (reordering it, dropping space)
    fails here rather than silently shifting every published ms.
    """
    surface = _surface({"qwe": 2, "qw ": 3, "q#q": 5})
    slot = {char: i for i, char in enumerate(QWERTY30M)}
    surface._T2[slot["q"], slot["w"]] = 10.0
    surface._Tc[slot["q"], slot["w"], slot["e"]] = 1.0
    surface._Tc[slot["q"], slot["w"], surface._n - 1] = 2.0

    card = surface.card(QWERTY30M, ref_total_ms=100.0)

    assert card.total_ms == pytest.approx(58.0)
    assert card.ms_per_char == pytest.approx(11.6)
    assert card.coverage_pct == pytest.approx(50.0)
    assert card.per_key_ms[" "] == pytest.approx(6.0)


def test_seed_totals_gets_the_SAME_guard_as_card():
    """The other method with the identical unguarded ``slot_of`` construction.

    ``seed_totals`` backs the SELECT-1 estimator-stability instrument, so a short layout there
    would understate every seed's total by the same ~95% and the SPREAD — the quantity being
    read — would look reassuringly tight. Guarding only ``card`` would have left that live.
    """
    surface = _surface({"qwe": 1})
    surface._T2s = [np.zeros_like(surface._T2)]
    surface._Tcs = [np.zeros_like(surface._Tc)]

    with pytest.raises(ValueError, match="30 DISTINCT characters"):
        surface.seed_totals("qwerty")

    assert surface.seed_totals(QWERTY30M) == pytest.approx([0.0]), "valid layout still works"
