from __future__ import annotations

import pytest

from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.cli.analyze import _shared_kmstats

QWERTY = "qwertyuiopasdfghjkl;zxcvbnm,./"


def test_kmstats_small_corpus_pins_every_metric_and_denominator():
    stats = KmStats(
        {"aa": 1, "qa": 1, "et": 1},
        {"qa": 1},
        {"aja": 1, "asj": 1, "sad": 1},
    ).stats(QWERTY)

    assert stats == pytest.approx(
        {
            "sfr": 33.333333333333336,
            "sfb": 33.333333333333336,
            "sfs": 100.0,
            "sfb-dist": 34.35921354681384,
            "sfs-dist": 103.07764064044152,
            "lsb": 33.333333333333336,
            "lsb-dist": 66.66666666666667,
            "alt": 33.333333333333336,
            "roll": 33.333333333333336,
            "sr-roll": 33.333333333333336,
            "redir": 33.333333333333336,
        },
        abs=1e-12,
    )


def test_kmstats_empty_corpora_return_the_complete_zero_schema():
    assert KmStats({}, {}, {}).stats(QWERTY) == dict.fromkeys(STAT_NAMES, 0.0)


def test_production_corpus_wiring_has_a_value_oracle():
    assert _shared_kmstats().stats(QWERTY) == pytest.approx(
        {
            "sfr": 2.8385205258856523,
            "sfb": 6.638478872558484,
            "sfs": 11.37683803688819,
            "sfb-dist": 9.483735823380075,
            "sfs-dist": 15.648916897054898,
            "lsb": 3.024213475781101,
            "lsb-dist": 6.720650752657646,
            "alt": 26.583470480629522,
            "roll": 37.749794327437385,
            "sr-roll": 4.980928620686516,
            "redir": 13.447247585573997,
        },
        abs=1e-12,
    )
