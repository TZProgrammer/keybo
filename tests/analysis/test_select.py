"""SELECT-1 instrument unit tests (IO-free: RawSupport math via injected sets)."""

from __future__ import annotations

import math

import pytest

from keybo.analysis.select import (
    QWERTY30M,
    RawSupport,
    behavior_stats,
    hand_balance_pct,
    switching_costs,
    usage_stats,
)
from keybo.geometry import ROW_STAGGERED_31


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


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (
            "q",
            "a",
            {
                "unchanged_keys": 28,
                "same_finger_keys": 30,
                "same_hand_keys": 30,
                "zxcv_preserved": 4,
            },
        ),
        (
            "q",
            "w",
            {
                "unchanged_keys": 28,
                "same_finger_keys": 28,
                "same_hand_keys": 30,
                "zxcv_preserved": 4,
            },
        ),
        (
            "z",
            "b",
            {
                "unchanged_keys": 28,
                "same_finger_keys": 28,
                "same_hand_keys": 30,
                "zxcv_preserved": 3,
            },
        ),
    ],
)
def test_switching_costs_distinguish_finger_hand_and_shortcut_retention(first, second, expected):
    layout = list(QWERTY30M)
    i, j = layout.index(first), layout.index(second)
    layout[i], layout[j] = layout[j], layout[i]
    assert switching_costs("".join(layout)) == expected


def test_hand_balance():
    freqs = {"q": 3.0, "p": 1.0}
    assert hand_balance_pct(QWERTY30M, freqs) == pytest.approx(75.0)
    assert hand_balance_pct("p" + QWERTY30M[1:9] + "q" + QWERTY30M[10:], freqs) == pytest.approx(
        25.0
    )


def test_raw_support_math():
    # positions: 30 slot coords + a distinct space coord at the LAST index (32 total,
    # mirroring [*geometry.slots, space_position]); index 30 must NOT be treated as space.
    positions = [(i, 0) for i in range(30)] + [(99, 99), (0, 0)]  # [30]=quote-slot, [31]=space
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


def test_raw_support_space_maps_to_last_index():
    """Regression (space-index bug, 2026-07-20): ' ' must resolve to the appended
    space_position (last index), not a hardcoded 30 (the 31-slot quote-slot coord)."""
    positions = [*ROW_STAGGERED_31.slots, ROW_STAGGERED_31.space_position]
    assert len(positions) == len(set(positions)) == 32
    assert positions[0] == (-5, 3)
    assert positions[30] == (6, 2)
    assert positions[31] == (0, 0)

    # Production has three distinct coordinates here. This also rejects a wrong index 0,
    # which the old synthetic fixture aliased to the space coordinate.
    rs = RawSupport(
        bi_serve={(positions[0], positions[31])},
        bi_any=set(),
        tri_serve=set(),
        tri_any=set(),
        positions31=positions,
    )
    sup = rs.support(QWERTY30M, {"q ": 5.0}, {})
    assert sup["bi_serve_pct"] == pytest.approx(100.0)  # space resolved to index 31, matched


def _stroke_row(layout, positions, ngram, frequency, wpms):
    samples = [repr((wpm, 100 + i, i + 1, 50)) for i, wpm in enumerate(wpms)]
    return "\t".join((layout, repr(positions), ngram, str(frequency), *samples))


def test_raw_support_from_tsvs_uses_raw_samples_and_serve_bucket(tmp_path):
    bi_high_freq = ((-5, 3), (-4, 3))
    bi_enough_samples = ((-5, 3), (-3, 3))
    bi_wrong_bucket = ((-5, 3), (-2, 3))
    tri_high_freq = ((-5, 3), (-4, 3), (-3, 3))
    tri_enough_samples = ((-5, 3), (-3, 3), (-2, 3))
    bi_tsv = tmp_path / "bigrams.tsv"
    tri_tsv = tmp_path / "trigrams.tsv"
    bi_tsv.write_text(
        "\n".join(
            (
                _stroke_row("qwerty", bi_high_freq, "qw", 9999, [85, 86]),
                _stroke_row("qwerty", bi_enough_samples, "qe", 1, [85, 86, 87]),
                _stroke_row("qwerty", bi_wrong_bucket, "qr", 1, [65, 66, 67, 68]),
            )
        )
        + "\n"
    )
    tri_tsv.write_text(
        "\n".join(
            (
                _stroke_row("qwerty", tri_high_freq, "qwe", 9999, [85, 86]),
                _stroke_row("qwerty", tri_enough_samples, "qer", 1, [85, 86, 87]),
            )
        )
        + "\n"
    )

    support = RawSupport.from_tsvs(
        bi_tsv,
        tri_tsv,
        wpm_lo=40,
        wpm_hi=140,
        w=20,
        min_cell=3,
        serve_bucket=80,
    )

    assert support.bi_any == {bi_high_freq, bi_enough_samples, bi_wrong_bucket}
    assert support.bi_serve == {bi_enough_samples}
    assert support.tri_any == {tri_high_freq, tri_enough_samples}
    assert support.tri_serve == {tri_enough_samples}


def test_usage_stats_qwerty_corner():
    u = usage_stats(QWERTY30M, {"a": 3.0, "p": 1.0})  # a: home-row LP; p: top-row RP
    assert u["left_pct"] == pytest.approx(75.0)
    assert u["home_row_pct"] == pytest.approx(75.0)
    assert u["pinky_pct"] == pytest.approx(100.0)
    assert u["fingers"]["LP"] == pytest.approx(75.0)
    assert u["fingers"]["RP"] == pytest.approx(25.0)


def test_usage_stats_pins_every_finger_bucket():
    u = usage_stats(QWERTY30M, dict.fromkeys(QWERTY30M[:10], 1.0))
    assert u["left_pct"] == pytest.approx(50.0)
    assert u["home_row_pct"] == pytest.approx(0.0)
    assert u["pinky_pct"] == pytest.approx(20.0)
    assert u["fingers"] == pytest.approx(
        {
            "LP": 10.0,
            "LR": 10.0,
            "LM": 10.0,
            "LI": 20.0,
            "RI": 20.0,
            "RM": 10.0,
            "RR": 10.0,
            "RP": 10.0,
        }
    )


def test_usage_stats_pins_hand_home_and_off_layout_boundaries():
    assert usage_stats(QWERTY30M, {"t": 1.0, "y": 1.0})["left_pct"] == pytest.approx(50.0)
    assert usage_stats(QWERTY30M, {"a": 1.0, "z": 1.0})["home_row_pct"] == pytest.approx(50.0)
    u = usage_stats(QWERTY30M, {"q": 1.0, "#": 99.0})
    assert u["left_pct"] == pytest.approx(100.0)
    assert u["home_row_pct"] == pytest.approx(0.0)
    assert u["pinky_pct"] == pytest.approx(100.0)
    assert u["fingers"] == pytest.approx(
        {"LP": 100.0, "LR": 0.0, "LM": 0.0, "LI": 0.0, "RI": 0.0, "RM": 0.0, "RR": 0.0, "RP": 0.0}
    )


def test_usage_stats_zero_mass_keeps_the_complete_schema():
    u = usage_stats(QWERTY30M, {})
    assert set(u) == {"left_pct", "home_row_pct", "pinky_pct", "fingers"}
    assert math.isnan(u["left_pct"])
    assert math.isnan(u["home_row_pct"])
    assert math.isnan(u["pinky_pct"])
    assert set(u["fingers"]) == {"LP", "LR", "LM", "LI", "RI", "RM", "RR", "RP"}
    assert all(math.isnan(value) for value in u["fingers"].values())


def test_behavior_stats_structure():
    from keybo.analysis.community import community_suite

    _, v1, _ = community_suite(";")
    q = behavior_stats(QWERTY30M, v1)
    s = behavior_stats("flhvz'wuoysrntkcdeaixjbmqpg,.-", v1)  # semimak
    for r in (q, s):
        for k, v in r.items():
            assert 0.0 <= v <= 100.0, (k, v)
    # same-char reuse is layout-invariant (same char -> same key everywhere)
    for k in ("sk1_samekey_pct", "sk2_samekey_pct", "sk3_samekey_pct"):
        assert q[k] == pytest.approx(s[k])
    # qwerty's same-finger skip-travel dwarfs semimak's (sfs 11.35 vs 5.83)
    assert q["sk1_sftravel_pct"] > s["sk1_sftravel_pct"] * 1.5
    assert q["bad_redirect_pct"] > s["bad_redirect_pct"]
    assert q == pytest.approx(
        {
            "bad_redirect_pct": 1.4475817038556225,
            "sk1_samekey_pct": 3.431607337289121,
            "sk1_sftravel_pct": 11.25380060338811,
            "sk2_samekey_pct": 5.408923153943775,
            "sk2_sftravel_pct": 8.999093101864323,
            "sk3_samekey_pct": 5.823438909883514,
            "sk3_sftravel_pct": 9.47557805843017,
        },
        abs=1e-12,
    )
