"""OxeyStyleScorer — a documented approximation of oxeylyzer-family heuristics (7.2).

Community analyzers score layouts by weighted pattern counts (SFB%, DSFB%, LSB, scissors,
rolls, redirects, alternation, finger balance). This scorer reproduces that STYLE of
judgment as an IScorer so it can (a) crosswalk our layouts against community judgment and
(b) be jointly optimized with measured speed via CompositeScorer. It is explicitly a
PREFERENCE term: our own data measured redirects time-neutral and lag-2 reuse
speed-neutral — oxeylyzer penalizes both — so joint optimization deliberately
re-introduces community doctrine, at a user-chosen weight.

Tests pin the judgment DIRECTION on known layouts: community consensus ranks
semimak/colemak far above qwerty (qwerty is pattern-horrible: high SFB, bad redirects,
low rolls). If our approximation disagrees with that ordering, it's not approximating.
"""

import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS, OxeyStyleScorer


@pytest.fixture(scope="module")
def corpora():
    # Tiny but structured: enough mass to exercise every pattern class.
    bigrams = {"th": 100, "he": 90, "ed": 40, "ju": 25, "ws": 20, "de": 35, "wd": 30}
    skipgrams = {"te": 30, "hd": 15, "jm": 10}
    trigrams = {"the": 80, "eds": 25, "was": 20, "ded": 15}
    return bigrams, skipgrams, trigrams


def test_community_consensus_ordering(corpora):
    """The whole point: qwerty must score WORSE (higher penalty) than colemak and
    semimak under community-style judgment, using the real corpus files' top patterns."""
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load(path):
        out = {}
        with open(os.path.join(root, path), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 500:  # top-500 rows are plenty for ordering
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    out[parts[0]] = int(parts[1])
        return out

    scorer = OxeyStyleScorer(
        load("data/corpus/bigrams.txt"),
        load("data/corpus/1-skip.txt"),
        load("data/corpus/trigrams.txt"),
    )
    scores = {
        name: scorer.fitness(Layout(NAMED_LAYOUTS[name], ROW_STAGGERED_30))
        for name in ("qwerty", "colemak", "semimak")
    }
    assert scores["colemak"] < scores["qwerty"]
    assert scores["semimak"] < scores["qwerty"]


def test_sfb_and_roll_terms_move_the_score(corpora):
    bigrams, skipgrams, trigrams = corpora
    lay = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    base = OxeyStyleScorer(bigrams, skipgrams, trigrams).fitness(lay)
    no_sfb = OxeyStyleScorer(bigrams, skipgrams, trigrams, weights={"sfb": 0.0}).fitness(lay)
    # 'ju' and 'ed'/'de' are qwerty SFBs -> zeroing the sfb weight must lower the penalty.
    assert no_sfb < base


def test_rolls_are_rewarded_not_penalized(corpora):
    bigrams, skipgrams, trigrams = corpora
    lay = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    boosted_rolls = OxeyStyleScorer(
        bigrams, skipgrams, trigrams, weights={"inroll": -100.0}
    ).fitness(lay)
    base = OxeyStyleScorer(bigrams, skipgrams, trigrams).fitness(lay)
    # Rewards are NEGATIVE weights: making inroll more negative lowers the score.
    assert boosted_rolls < base


def test_unknown_weight_rejected(corpora):
    bigrams, skipgrams, trigrams = corpora
    with pytest.raises(ValueError, match="unknown oxey weight"):
        OxeyStyleScorer(bigrams, skipgrams, trigrams, weights={"nope": 1.0})


def test_default_weights_documented():
    for name, (w, why) in DEFAULT_OXEY_WEIGHTS.items():
        assert isinstance(w, float)
        assert isinstance(why, str) and len(why) > 10, name
