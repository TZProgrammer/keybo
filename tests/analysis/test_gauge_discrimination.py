"""Which gauges can DISCRIMINATE, pinned as a contract — because three of them cannot.

GAUGEAUDIT-1 measured it: over within-hand permutations of a fixed layout (hand partition held
constant, characters never crossing hands) `sfr` and `alt` are EXACTLY constant — spread 0.000e+00 —
while every other kmstats gauge moves. `imbalance` behaves the same way in the oxey frame.

That is why `tests/cli/test_analyze_allgauge.py` check-ins `sfr = 2.8187069323648957`,
`alt = 43.61506539321432` and `imbalance = 0.9321038278010377` as IDENTICAL literals for THREE
different frozen boards: those boards all share the left-hand charset "',-.aehijkopuyz". Pinning the
same number for three layouts is a correct regression pin and a useless discrimination test, and
nothing distinguished the two.

⚠ THE PERTURBATION MATTERS, and my earlier probe got it wrong. Random FULL-charset shuffles move
characters between hands, so `alt` separates and the invariance HIDES (only `sfr` shows). Only
WITHIN-HAND permutation exposes it — and that is exactly the candidate set a local search around a
fixed hand partition explores, i.e. what our own optimizer does.

`keybo.verdicts.all_distinct` was written for precisely this, and its docstring names these gauges by
name — it had ZERO production callers. These tests wire it to the fact it describes.
"""

from __future__ import annotations

import random

import pytest

from keybo.analysis.kmstats import KmStats
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.verdicts import all_distinct

_G = ROW_STAGGERED_30
_BASE = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"  # keybo-lsb
#: Measured in GAUGEAUDIT-1: exactly constant under within-hand permutation.
_HAND_PARTITION_INVARIANT = frozenset({"sfr", "alt"})


@pytest.fixture(scope="module")
def km() -> KmStats:
    d = production_corpus_dir()
    return KmStats(
        load_frequencies(str(d / "bigrams.txt")),
        load_frequencies(str(d / "1-skip31.txt")),
        load_frequencies(str(d / "trigrams.txt")),
    )


def _within_hand_variants(n: int = 12, seed: int = 0) -> list[str]:
    """Permute inside each hand only, so the hand partition is IDENTICAL across all variants."""
    slots = list(_G.slots)
    left = [i for i, s in enumerate(slots) if s[0] < 0]
    right = [i for i, s in enumerate(slots) if s[0] > 0]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ch = list(_BASE)
        for side in (left, right):
            vals = [ch[i] for i in side]
            rng.shuffle(vals)
            for k, i in enumerate(side):
                ch[i] = vals[k]
        out.append("".join(ch))
    return out


def test_the_hand_partition_is_actually_held_constant(km: KmStats) -> None:
    """Guard the guard: if the variants moved a character across hands, the test below is vacuous."""
    slots = list(_G.slots)
    left_of = lambda s: frozenset(ch for ch, sl in zip(s, slots, strict=True) if sl[0] < 0)  # noqa: E731
    partitions = {left_of(v) for v in _within_hand_variants()}
    assert len(partitions) == 1, "variants must share ONE hand partition"
    assert partitions.pop() == frozenset("',-.aehijkopuyz"), "and it must be keybo-lsb's"


def test_sfr_and_alt_CANNOT_discriminate_within_a_hand_partition(km: KmStats) -> None:
    """The invariance, pinned. If a future change makes these vary, that is NEWS — read the docstring."""
    variants = _within_hand_variants()
    rows = [km.stats(v) for v in variants]
    for name in sorted(_HAND_PARTITION_INVARIANT):
        values = [r[name] for r in rows]
        assert max(values) - min(values) == 0.0, f"{name} moved; the invariance may be fixed"
        assert not all_distinct(values, name), "all_distinct must FLAG it"


def test_every_other_gauge_DOES_discriminate(km: KmStats) -> None:
    """The contrast is the point — otherwise the test above could pass on a broken corpus."""
    rows = [km.stats(v) for v in _within_hand_variants()]
    movers = [n for n in rows[0] if n not in _HAND_PARTITION_INVARIANT]
    assert movers, "no gauges left to compare"
    for name in movers:
        values = [r[name] for r in rows]
        assert max(values) - min(values) > 0.0, f"{name} is ALSO invariant — extend the frozen set"


def test_a_FULL_shuffle_HIDES_the_invariance_which_is_why_the_probe_must_be_within_hand(
    km: KmStats,
) -> None:
    """Pin the methodological trap, not just the fact.

    Under full-charset shuffles `alt` separates, so a full-shuffle distinctness probe reports the
    frame as healthy. This test fails if someone "simplifies" the fixture to a plain shuffle.
    """
    rng = random.Random(0)
    full = []
    for _ in range(12):
        ch = list(_BASE)
        rng.shuffle(ch)
        full.append("".join(ch))
    rows = [km.stats(v) for v in full]
    alt = [r["alt"] for r in rows]
    sfr = [r["sfr"] for r in rows]
    assert max(alt) - min(alt) > 0.0, "alt SHOULD separate under a full shuffle — that is the trap"
    assert max(sfr) - min(sfr) == 0.0, (
        "sfr is invariant under ANY permutation (global charset stat)"
    )


def test_all_distinct_is_WIRED_and_flags_the_documented_case() -> None:
    """`all_distinct`'s docstring names alt/imbalance/sfr; it had zero production callers.

    This is the smallest honest wiring: the guard now has a caller that exercises the exact fact it
    documents, so a change that silently makes it unreachable fails here.
    """
    assert all_distinct([1.0, 2.0, 3.0], "distinct")
    assert not all_distinct([2.8187069323648957] * 3, "the three frozen boards' sfr")


def test_the_frozen_boards_share_a_hand_partition_which_is_WHY_three_literals_repeat() -> None:
    """Documents the CAUSE, so nobody "fixes" the duplicate literals by editing the numbers.

    The three boards frozen in ``tests/cli/test_analyze_allgauge.py`` share one left-hand charset, so
    the hand-partition invariants MUST agree across them. The duplicate literals are a SYMPTOM of a
    real invariance, not a copy-paste error, and editing them would hide the finding.
    """
    slots = list(_G.slots)
    left = frozenset(ch for ch, sl in zip(_BASE, slots, strict=True) if sl[0] < 0)
    assert left == frozenset("',-.aehijkopuyz"), "keybo-lsb's left-hand charset"
