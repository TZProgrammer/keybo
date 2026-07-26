"""Per-finger scissor decomposition: it must PARTITION the aggregate gauge exactly.

The partition property is the whole test suite for an attribution rule: any rule that
loses, invents, or double-counts mass fails it, and a rule that merely disagrees about
*which* finger pays still passes — which is why the rule itself is pinned separately.
"""

from __future__ import annotations

import pytest

from keybo.analysis.scissor_fingers import (
    ATTRIBUTION_RULE,
    FINGER_NAMES,
    ScissorByFinger,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.oxey import OxeyStyleScorer

KEYBO_LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
GRAPHITE = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
FLAGSHIP_C3 = "pyou'vgdnmheai.cstrlkjz,-wfbxq"
ALL = [KEYBO_LSB, QWERTY30M, GRAPHITE, FLAGSHIP_C3]


def _layout(lay30: str) -> Layout:
    return Layout(lay30, ROW_STAGGERED_30)


@pytest.mark.parametrize("lay30", ALL)
def test_per_finger_shares_sum_to_the_aggregate_scissor_gauge(corpora, lay30):
    """THE test: the eight parts sum to `oxey.pattern_shares()['scissor']`.

    Both sides use the layout-restricted bigram denominator (trap #9), which is what makes
    this exact rather than off by a plausible constant.
    """
    bigrams, skipgrams, trigrams = corpora
    aggregate = OxeyStyleScorer(bigrams, skipgrams, trigrams).pattern_shares(_layout(lay30))[
        "scissor"
    ]
    per_finger = ScissorByFinger(bigrams).shares(_layout(lay30))
    assert sum(per_finger.values()) == pytest.approx(aggregate, rel=0, abs=1e-12)


@pytest.mark.parametrize("lay30", ALL)
def test_pair_shares_are_a_second_exact_partition(corpora, lay30):
    bigrams, skipgrams, trigrams = corpora
    aggregate = OxeyStyleScorer(bigrams, skipgrams, trigrams).pattern_shares(_layout(lay30))[
        "scissor"
    ]
    pairs = ScissorByFinger(bigrams).pair_shares(_layout(lay30))
    assert sum(pairs.values()) == pytest.approx(aggregate, rel=0, abs=1e-12)


@pytest.mark.parametrize("lay30", ALL)
def test_every_finger_is_present_and_nonnegative(corpora, lay30):
    bigrams, *_ = corpora
    shares = ScissorByFinger(bigrams).shares(_layout(lay30))
    assert tuple(shares) == FINGER_NAMES
    assert all(value >= 0.0 for value in shares.values())


def test_the_attribution_rule_is_half_to_each_finger(corpora):
    """A single scissor bigram charges exactly half its mass to each of its two fingers."""
    bigrams, *_ = corpora
    # A hand-built corpus with ONE scissor: qwerty 'x' (LR, bottom) then 'e' (LM, top).
    layout = _layout(QWERTY30M)
    from keybo.features import classify as C

    a, b = layout.pos("x"), layout.pos("e")
    assert C.is_scissor(ROW_STAGGERED_30, a, b), "fixture bigram must be a scissor"
    shares = ScissorByFinger({"xe": 100, "as": 900}).shares(layout)
    # 100 of 1000 mass is the scissor => 10% aggregate, split 5/5.
    assert shares["LR"] == pytest.approx(5.0)
    assert shares["LM"] == pytest.approx(5.0)
    assert sum(shares.values()) == pytest.approx(10.0)
    assert ATTRIBUTION_RULE == "half-to-each-finger"


def test_attribution_is_symmetric_in_bigram_order(corpora):
    """`is_scissor` is order-blind, so the decomposition must be too."""
    layout = _layout(QWERTY30M)
    forward = ScissorByFinger({"xe": 100, "as": 900}).shares(layout)
    backward = ScissorByFinger({"ex": 100, "as": 900}).shares(layout)
    assert forward == backward


def test_pair_key_names_both_fingers_in_board_order(corpora):
    layout = _layout(QWERTY30M)
    pairs = ScissorByFinger({"xe": 100, "as": 900}).pair_shares(layout)
    assert set(pairs) == {"LR+LM"}
    assert pairs["LR+LM"] == pytest.approx(10.0)


@pytest.mark.parametrize("lay30", ALL)
def test_qwerty_scissors_more_than_every_campaign_layout(corpora, lay30):
    bigrams, *_ = corpora
    scorer = ScissorByFinger(bigrams)
    qwerty_total = sum(scorer.shares(_layout(QWERTY30M)).values())
    if lay30 == QWERTY30M:
        pytest.skip("qwerty is the reference")
    assert qwerty_total > sum(scorer.shares(_layout(lay30)).values())


def test_no_middle_pinky_pair_is_in_the_narrow_support(corpora):
    """A documented LIMITATION of `is_scissor`, pinned so it cannot change silently.

    `is_adjacent` requires a column gap of 1, and middle (col 3) to pinky (col 5) is 2, so
    no middle-pinky scissor exists on this support at all. The pinky's per-finger share
    therefore counts only its ring-adjacent scissors.
    """
    bigrams, *_ = corpora
    for lay30 in ALL:
        pairs = ScissorByFinger(bigrams).pair_shares(_layout(lay30))
        assert not {"LM+LP", "RM+RP"} & set(pairs), pairs


def test_empty_corpus_returns_zeros_not_a_crash():
    shares = ScissorByFinger({}).shares(_layout(QWERTY30M))
    assert tuple(shares) == FINGER_NAMES
    assert all(value == 0.0 for value in shares.values())
    assert ScissorByFinger({}).pair_shares(_layout(QWERTY30M)) == {}
