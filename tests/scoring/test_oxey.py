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


def test_pattern_shares_and_weighted_fitness_are_value_pinned(corpora):
    """Value pin on the tiny fixture corpus.

    ⚠️ ``redirect`` is 0.0 and ``bad_redirect`` is 14.2857 — that is the point, not a gap.
    Exactly one fixture trigram is classified at all: ``was`` on qwerty is L-ring ->
    L-pinky -> L-ring, a same-hand reversal with no index finger, i.e. a BAD redirect. The
    two redirect classes are mutually exclusive (upstream dispatches one class per trigram),
    so it counts once, in the worse class. It used to count in BOTH, which is why this pin
    read 14.2857 twice and the fitness was 598.6822: the difference,
    ``598.6822 - 570.1108 = 28.5714``, is exactly ``redirect`` weight 2.0 x the 14.2857
    share it should never have had. See tests/scoring/test_oxey_trigram_partition.py.

    The two ``*_ordered`` shares are the order-aware roll channel. They are pinned here with
    every other share while the FITNESS keeps its pre-existing value — the pair of assertions
    that together say "a share was added and nothing was repriced". On this fixture they
    happen to equal ``inroll``/``outroll``, because its only same-hand cross-column bigrams
    (``ws``, ``wd``, ``de``) are all cross-row; ``test_oxey_corpus_reversal.py`` is where the
    two pairs diverge, on a real corpus.
    """
    bigrams, skipgrams, trigrams = corpora
    lay = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    scorer = OxeyStyleScorer(bigrams, skipgrams, trigrams)

    assert scorer.pattern_shares(lay) == pytest.approx(
        {
            "sfb": 35.294117647058826,
            "dsfb": 18.181818181818183,
            "lsb": 0.0,
            "scissor": 0.0,
            "inroll": 8.823529411764707,
            "outroll": 0.0,
            "onehand": 0.0,
            "redirect": 0.0,
            "bad_redirect": 14.285714285714286,
            "alternate": 55.88235294117647,
            "imbalance": 29.41176470588235,
            "inroll_ordered": 8.823529411764707,
            "outroll_ordered": 0.0,
        },
        abs=1e-12,
    )
    # UNCHANGED from before the ordered shares existed: they ship at weight 0.
    assert scorer.fitness(lay) == pytest.approx(570.1107715813598, abs=1e-12)


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
