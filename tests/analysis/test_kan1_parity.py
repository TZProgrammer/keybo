"""KAN-1 parity gates (rule b330ab4): the analyzer's gauges vs golden tool outputs.

Goldens (``golden_kan1.json``) were frozen from:
* community scores — the P17 campaign board (runs/p17_coopt.json), whose values
  come from the binary-parity-gated campaign ports (genkey rank corr 1.0 /
  spread <=2%; oxeylyzer repl rank corr 1.0 / spread <=5%, o2 exact x100);
* kmstats — kmrun (the keymeow scoring harness) run on the vendored shared
  corpus (gate G3: per-stat abs diff <= 0.02pp);
* speed — the P17 board's fit_speed totals (gate G4: rel err <= 1e-6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from keybo.analysis.community import community_suite, pinned_char
from keybo.analysis.kmstats import STAT_NAMES

GOLD = json.loads((Path(__file__).parent / "golden_kan1.json").read_text())


@pytest.fixture(scope="module")
def suite():
    return community_suite(";")  # every golden layout is C30M-charset


@pytest.mark.parametrize("name", sorted(GOLD["community"]))
def test_g1_genkey_exact(name, suite):
    gk, _, _ = suite
    lay = GOLD["layouts"][name]
    assert gk.score(lay) == pytest.approx(GOLD["community"][name]["genkey"], abs=1e-9)


@pytest.mark.parametrize("name", sorted(GOLD["community"]))
def test_g2_oxeylyzer_exact(name, suite):
    _, v1, o2 = suite
    lay = GOLD["layouts"][name]
    assert v1.score(lay) == GOLD["community"][name]["oxey1"]
    assert o2.score(lay) == GOLD["community"][name]["oxey2"]
    assert o2.wfd(lay) == GOLD["community"][name]["wfd"]


def test_pinned_char_convention():
    assert pinned_char(GOLD["layouts"]["semimak"]) == ";"  # C30M board pins ;
    assert pinned_char("qwertyuiopasdfghjkl;zxcvbnm,./") == "'"  # classic pins '


@pytest.mark.slow
def test_g3_kmstats_vs_kmrun():
    import gzip

    from keybo.analysis.kmstats import KmStats

    vend = Path(__file__).parents[2] / "data" / "community" / "vendored"
    with gzip.open(vend / "keymeow-keybo.json.gz", "rt") as fh:
        d = json.load(fh)
    ks = KmStats(d["bigrams"], d["skipgrams"], d["trigrams"])
    for lay, want in GOLD["kmstats"].items():
        got = ks.stats(lay)
        for stat in STAT_NAMES:
            if stat in want:
                assert got[stat] == pytest.approx(want[stat], abs=0.02), (lay, stat)


@pytest.mark.slow
def test_g4_time_surface_reproduces_p17_board():
    from keybo.analysis.timecard import default_surface

    surf = default_surface(90.0)
    for name, want in GOLD["speed"].items():
        card = surf.card(GOLD["layouts"][name])
        assert card.total_ms == pytest.approx(want, rel=1e-6), name
    # coverage: every golden layout is full-charset for the C30M corpus subset
    card = surf.card(GOLD["layouts"]["semimak"])
    assert card.coverage_pct > 90.0
