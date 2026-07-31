"""CORPUS REVERSAL: a direction-sensitive gauge must move when every n-gram is reversed.

THE PIN. Reversing every n-gram in the corpus turns each stroke into its opposite: ``th``
becomes ``ht``, so every inward roll becomes an outward one and vice versa. A gauge that
claims to measure a DIRECTION of travel must therefore move; a gauge that is secretly
measuring the unordered key pair cannot move at all.

Before the ordered predicates existed, ``inroll`` and ``outroll`` moved by **exactly
0.00e+00** under this transform (CYANO-1, ledger ``dbc8970``, independently re-measured on
this branch) — which is what a swap-invariant predicate has to do. That is the failure this
file was written to produce and then fix.

Reversal is the right transform rather than, say, a shuffle: it is an involution that
preserves every unordered pair's total mass exactly, so it holds the entire unordered gauge
family fixed BY CONSTRUCTION. Any share that moves under it is order-sensitive, and any
share that does not is not — there is no third explanation, and no need to argue about
whether some confounder moved too. That makes the same transform a two-sided instrument: it
proves the new shares are directional AND proves the old ones are the control.
"""

from __future__ import annotations

import os

import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS, OxeyStyleScorer

#: The gauges that MUST NOT move under reversal — they are functions of the unordered pair.
#: ``inroll``/``outroll`` are in this list deliberately: they are the version-locked
#: outer-row shares, kept bit-stable so no published number is renumbered.
ORDER_INVARIANT_SHARES = (
    "sfb",
    "dsfb",
    "lsb",
    "scissor",
    "inroll",
    "outroll",
    "alternate",
    "imbalance",
)

#: The gauges that MUST move — the ordered roll shares this branch adds.
ORDER_SENSITIVE_SHARES = ("inroll_ordered", "outroll_ordered")


def _load(path: str, limit: int = 4000) -> dict[str, int]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out: dict[str, int] = {}
    with open(os.path.join(root, path), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def _reverse(freqs: dict[str, int]) -> dict[str, int]:
    """Reverse every n-gram, summing collisions (palindromes map onto themselves)."""
    out: dict[str, int] = {}
    for gram, f in freqs.items():
        out[gram[::-1]] = out.get(gram[::-1], 0) + f
    return out


@pytest.fixture(scope="module")
def corpora() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    return (
        _load("data/corpus/bigrams.txt"),
        _load("data/corpus/1-skip.txt"),
        _load("data/corpus/trigrams.txt"),
    )


@pytest.fixture(scope="module")
def shares(corpora):
    """Pattern shares on the real corpus and on the fully reversed corpus."""
    bigrams, skipgrams, trigrams = corpora
    layout = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    forward = OxeyStyleScorer(bigrams, skipgrams, trigrams).pattern_shares(layout)
    reverse = OxeyStyleScorer(
        _reverse(bigrams), _reverse(skipgrams), _reverse(trigrams)
    ).pattern_shares(layout)
    return forward, reverse


# ------------------------------------------------------------------ the failing pin
# This is the test that produced `delta 0.00e+00` on main, for both shares.


@pytest.mark.parametrize("name", ORDER_SENSITIVE_SHARES)
def test_corpus_reversal_moves_a_direction_sensitive_roll_share(name, shares):
    """The headline: reversing every n-gram must CHANGE the ordered roll shares.

    Asserted as a relative move against the share's own magnitude, so the test states a real
    effect size rather than merely ``!=`` — a 1e-12 float wobble would satisfy inequality and
    tell us nothing.
    """
    forward, reverse = shares
    before, after = forward[name], reverse[name]
    assert before > 0.0, f"{name} must fire on the real corpus or the test proves nothing"
    delta = abs(after - before)
    assert delta > 0.0, f"{name} moved by exactly {delta:.2e} — the gauge is order-blind"
    assert delta / before > 0.01, (
        f"{name} moved only {100 * delta / before:.4f}% — suspiciously inert"
    )


def test_reversal_swaps_the_two_ordered_roll_shares(shares):
    """The exact structural prediction, which is stronger than "it moved".

    Reversal maps every inward stroke onto an outward one and vice versa, so the ordered
    inroll after reversal must equal the ordered outroll before it — to floating-point
    equality, not approximately. A gauge that merely moved could have moved for any reason;
    this pins the mechanism. (Bigram totals are identical under reversal, so the
    normalisation denominator cancels exactly.)
    """
    forward, reverse = shares
    assert reverse["inroll_ordered"] == pytest.approx(forward["outroll_ordered"], rel=1e-12)
    assert reverse["outroll_ordered"] == pytest.approx(forward["inroll_ordered"], rel=1e-12)


# --------------------------------------------------- the control: the old gauges must NOT move


@pytest.mark.parametrize("name", ORDER_INVARIANT_SHARES)
def test_corpus_reversal_leaves_the_unordered_shares_bit_identical(name, shares):
    """The control half, and the anti-renumbering guard.

    Every share built on the unordered pair must be BIT-identical under reversal — asserted
    with ``==``, not ``approx``. Two things break if this fails: the CYANO-1 finding (all
    shipped shares move 0.00e+00 under reversal) stops reproducing, and ``inroll``/``outroll``
    silently change meaning under a frozen ``FEATURE_VERSION``.
    """
    forward, reverse = shares
    assert forward[name] == reverse[name], f"{name}: {forward[name]} -> {reverse[name]}"


def test_the_ordered_shares_are_additive_over_the_unordered_ones(shares):
    """The ordered pair must cover strictly MORE roll mass than the unordered pair.

    The unordered predicates skip same-row rolls entirely (108 of 324 eligible K30 pairs), so
    ``inroll + outroll`` understates total roll mass. Pinning the inequality documents that
    the new shares are a superset rather than a re-partition of the same mass — the reason
    they cannot be compared to the old ones cell-for-cell.
    """
    forward, _ = shares
    old_total = forward["inroll"] + forward["outroll"]
    new_total = forward["inroll_ordered"] + forward["outroll_ordered"]
    assert new_total > old_total


def test_the_ordered_shares_are_not_weighted_by_default(shares):
    """Measured, not priced: adding the shares must not change any layout's oxey score.

    ``DEFAULT_OXEY_WEIGHTS`` is a published preference table whose values reproduce the
    community's layout ORDERING. Giving the new shares a nonzero default weight would move
    every ``oxey-style`` number in the ledger, so they ship at weight 0 and a caller opts in.
    """
    assert "inroll_ordered" not in DEFAULT_OXEY_WEIGHTS
    assert "outroll_ordered" not in DEFAULT_OXEY_WEIGHTS


def test_oxey_fitness_is_unchanged_by_the_new_shares(corpora):
    """The end-to-end anti-renumbering guard: fitness is a weighted sum over the OLD keys.

    Pinned on the whole named-layout field rather than one board, and with the reversed
    corpus too — so a future weight added for an ordered share cannot slip in unnoticed.
    """
    bigrams, skipgrams, trigrams = corpora
    scorer = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    reversed_scorer = OxeyStyleScorer(_reverse(bigrams), _reverse(skipgrams), _reverse(trigrams))
    for name, chars in NAMED_LAYOUTS.items():
        layout = Layout(chars, ROW_STAGGERED_30)
        assert scorer.fitness(layout) == reversed_scorer.fitness(layout), name
