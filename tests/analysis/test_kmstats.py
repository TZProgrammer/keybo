from __future__ import annotations

import pytest

from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.cli.analyze import _shared_corpora
from keybo.data.corpus import IWEB, resolve_corpus_dir

QWERTY = "qwertyuiopasdfghjkl;zxcvbnm,./"


def _shared_kmstats() -> KmStats:
    """kmstats wired exactly as `analyze` wires it, on iWeb — NAMED, not the default.

    The oracle below is an iWeb number. CORPUS-SWAP-1 made ``blend-v1`` the default, so a
    test that took the default was asserting *the default* rather than *the value*. Naming
    the corpus is what keeps the frozen board reproducible across a default change.
    """
    return KmStats(*_shared_corpora(resolve_corpus_dir(IWEB)))


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
    """Oracle for the production wiring, on the ``1-skip31.txt`` skipgram table.

    ALLGAUGE-1 moved ``analyze`` from ``1-skip.txt`` to ``1-skip31.txt`` — the table every
    frozen campaign gauge board was computed on, and the one ``data/build_corpus.py``
    documents as the reproducible trigram marginalization ``skip(a,c) = sum_b tri(a,b,c)``
    (``1-skip.txt`` is a different, unreproducible pass). Exactly the two skipgram-derived
    cells moved, which is the change's own positive control:

        sfs       11.37683803688819  -> 11.380475122064565
        sfs-dist  15.648916897054898 -> 15.653519527778622

    The nine bigram/trigram cells are unchanged, as they must be — the swap touches only
    the skipgram table. See ``tests/cli/test_analyze_allgauge.py`` for the end-to-end
    control that the new convention reproduces the frozen boards bit-for-bit.
    """
    assert _shared_kmstats().stats(QWERTY) == pytest.approx(
        {
            "sfr": 2.8385205258856523,
            "sfb": 6.638478872558484,
            "sfs": 11.380475122064565,
            "sfb-dist": 9.483735823380075,
            "sfs-dist": 15.653519527778622,
            "lsb": 3.024213475781101,
            "lsb-dist": 6.720650752657646,
            "alt": 26.583470480629522,
            "roll": 37.749794327437385,
            "sr-roll": 4.980928620686516,
            "redir": 13.447247585573997,
        },
        abs=1e-12,
    )


def test_only_the_skipgram_cells_moved_when_the_skipgram_table_changed():
    """The convention swap's own positive control: 9 of 11 cells MUST be untouched.

    A change to the skipgram table can only reach ``sfs`` and ``sfs-dist``. If any other
    cell moves, something other than the intended table swap happened.
    """
    from pathlib import Path

    from keybo.data.corpus import load_frequencies

    corpus = Path(__file__).resolve().parents[2] / "data" / "corpus"
    bigrams = load_frequencies(str(corpus / "bigrams.txt"))
    trigrams = load_frequencies(str(corpus / "trigrams.txt"))
    old = KmStats(bigrams, load_frequencies(str(corpus / "1-skip.txt")), trigrams).stats(QWERTY)
    new = KmStats(bigrams, load_frequencies(str(corpus / "1-skip31.txt")), trigrams).stats(QWERTY)
    moved = {name for name in STAT_NAMES if old[name] != new[name]}
    assert moved == {"sfs", "sfs-dist"}
