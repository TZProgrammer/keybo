"""SELECT-1 instrument unit tests (IO-free: RawSupport math via injected sets)."""

from __future__ import annotations

import pytest

from keybo.analysis.select import QWERTY30M, RawSupport, hand_balance_pct, switching_costs


def test_switching_costs_identity():
    c = switching_costs(QWERTY30M)
    assert c == {
        "unchanged_keys": 30,
        "same_finger_keys": 30,
        "same_hand_keys": 30,
        "zxcv_preserved": 4,
    }


def test_switching_costs_move_one_key():
    # swap q and p (opposite corners): both keys change slot, finger and hand
    lay = "pwertyuioqasdfghjkl'zxcvbnm,.-"
    c = switching_costs(lay)
    assert c["unchanged_keys"] == 28
    assert c["same_finger_keys"] == 28
    assert c["same_hand_keys"] == 28
    assert c["zxcv_preserved"] == 4


def test_hand_balance():
    freqs = {"q": 3.0, "p": 1.0}
    assert hand_balance_pct(QWERTY30M, freqs) == pytest.approx(75.0)
    assert hand_balance_pct("p" + QWERTY30M[1:9] + "q" + QWERTY30M[10:], freqs) == pytest.approx(
        25.0
    )


def test_raw_support_math():
    # positions31: 31 distinct fake coordinates; observed sets built by hand
    positions = [(i, 0) for i in range(31)]
    lay = QWERTY30M
    q, w, e = (0, 0), (1, 0), (2, 0)
    rs = RawSupport(
        bi_serve={(q, w)},
        bi_any={(q, w), (w, e)},
        tri_serve=set(),
        tri_any={(q, w, e)},
        positions31=positions,
    )
    sup = rs.support(lay, {"qw": 3.0, "we": 1.0, "zz": 6.0}, {"qwe": 1.0, "qww": 3.0})
    assert sup["bi_serve_pct"] == pytest.approx(30.0)  # qw only, of 10 mass
    assert sup["bi_any_pct"] == pytest.approx(40.0)  # qw + we
    assert sup["tri_serve_pct"] == pytest.approx(0.0)
    assert sup["tri_any_pct"] == pytest.approx(25.0)  # qwe of 4 mass
    # ngrams with off-layout chars are excluded from the denominator
    sup2 = rs.support(lay, {"qw": 1.0, "q#": 9.0}, {"qwe": 1.0})
    assert sup2["bi_serve_pct"] == pytest.approx(100.0)
