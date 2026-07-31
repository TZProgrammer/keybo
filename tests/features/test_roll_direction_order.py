"""DIRECTION-ORDER: the roll-direction predicates must depend on WHICH KEY WAS STRUCK FIRST.

Written BEFORE the implementation and watched fail. The four failures on ``main`` were:
``test_ordered_inward_is_not_swap_invariant`` (0 != 324), ``..._outward...`` (0 != 324),
``test_ordered_predicates_partition_every_roll_eligible_pair`` (108 unclassed), and
``test_corpus_reversal_moves_a_direction_sensitive_roll_share`` (delta exactly 0.00e+00) —
plus the ``inroll_ordered`` KeyError from the share dict not existing yet.

WHY THIS FILE EXISTS. ``is_inwards``/``is_outwards`` compute

    outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
    return outer[1] > inner[1]

which sorts the two keys by COLUMN MAGNITUDE and then compares ROWS. Both operations discard
the argument order, so the predicates are swap-invariant BY CONSTRUCTION: over the 870
ordered position pairs of ``ROW_STAGGERED_30`` there are ZERO pairs whose verdict changes
when ``(a, b)`` becomes ``(b, a)``. A gauge named "inroll" that cannot tell ``as`` from
``sa`` is measuring the KEYS, not the STROKE. This file pins the distinction in both
directions: the unordered predicates must STAY unordered (callers depend on it), and the new
ordered ones must be maximally order-dependent.

The expected asymmetry count is 324 and that number is derived, not chosen: 324 is the
number of roll-eligible ordered pairs (same hand, different finger, different |column|), and
``inward_ordered`` is ``abs(b[0]) < abs(a[0])`` — which negates under swap for EVERY eligible
pair, because eligibility already excludes ``abs(a[0]) == abs(b[0])`` (the only way the
comparison could tie). So asymmetry == eligibility, exactly, and a count below 324 means the
predicate lost order-sensitivity somewhere.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31, Geometry

#: (geometry, ordered-pair count, roll-eligible count) — the eligible count is what the
#: asymmetry must equal. Both geometries are pinned so a K31 regression cannot hide.
GEOMETRIES = [
    pytest.param(ROW_STAGGERED_30, 870, 324, id="k30"),
    pytest.param(ROW_STAGGERED_31, 930, 348, id="k31"),
]


def _ordered_pairs(geometry: Geometry) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return list(permutations(geometry.slots, 2))


def _eligible(geometry: Geometry, a, b) -> bool:
    """The roll-eligibility gate both the old and new predicates share."""
    return (
        C.same_hand(geometry, a, b) and not C.same_finger(geometry, a, b) and abs(a[0]) != abs(b[0])
    )


# --------------------------------------------------------------- 1. the defect, pinned
# These four assert the CURRENT (unordered) predicates keep their current semantics. They
# pass on main and must keep passing: three callers depend on the unordered reading, so the
# fix is additive and a change here would be a silent renumbering of shipped gauges.


@pytest.mark.parametrize(("geometry", "n_pairs", "n_eligible"), GEOMETRIES)
def test_unordered_predicates_are_swap_invariant_by_design(geometry, n_pairs, n_eligible):
    """``is_inwards``/``is_outwards`` are a property of the KEY PAIR, not the stroke.

    This is the defect that motivated the file, but it is pinned as INTENDED behaviour
    because ``features/ngram.py`` serves these two as version-locked columns 18/19 of
    ``FEATURE_VERSION 2026-07-05.3`` and ``analysis/effect_curves.py`` deliberately reads
    them as ``outer_high``/``outer_low``. Changing them in place would move every shipped
    ``inroll``/``outroll`` number and desynchronise 6 trained models from their frame.
    """
    pairs = _ordered_pairs(geometry)
    assert len(pairs) == n_pairs
    assert (
        sum(1 for a, b in pairs if C.is_inwards(geometry, a, b) != C.is_inwards(geometry, b, a))
        == 0
    )
    assert (
        sum(1 for a, b in pairs if C.is_outwards(geometry, a, b) != C.is_outwards(geometry, b, a))
        == 0
    )
    # the roll ANGLE is swap-invariant too — the same outer/inner sort drives it
    assert (
        sum(
            1
            for a, b in pairs
            if C.rotation_angle(geometry, a, b) != C.rotation_angle(geometry, b, a)
        )
        == 0
    )


def test_unordered_predicates_miss_flat_rolls_entirely():
    """The second defect, pinned: a SAME-ROW roll has no direction under the row comparison.

    108 of the 324 roll-eligible ordered pairs on K30 have ``a[1] == b[1]``, so ``outer[1] >
    inner[1]`` and ``outer[1] < inner[1]`` are BOTH false and the pair is classed as neither
    inward nor outward — yet a flat pinky-to-index roll is the purest inward stroke there is.
    Known since ``agent-artifacts/bigram-experiment-backlog.md:26``; the ordered predicates
    are what fixes it, which is why the count is pinned here rather than left implicit.
    """
    geometry = ROW_STAGGERED_30
    eligible = [(a, b) for a, b in _ordered_pairs(geometry) if _eligible(geometry, a, b)]
    unclassed = [
        (a, b)
        for a, b in eligible
        if not C.is_inwards(geometry, a, b) and not C.is_outwards(geometry, a, b)
    ]
    assert len(eligible) == 324
    assert len(unclassed) == 108
    assert all(a[1] == b[1] for a, b in unclassed), "every unclassed eligible pair is same-row"


# ------------------------------------------------- 2. the new ordered predicates
# These are the tests that FAILED before the implementation existed.


@pytest.mark.parametrize(("geometry", "n_pairs", "n_eligible"), GEOMETRIES)
def test_ordered_inward_is_not_swap_invariant(geometry, n_pairs, n_eligible):
    """The headline requirement: ``is_inwards_ordered(g,a,b) != is_inwards_ordered(g,b,a)``.

    The count must equal the roll-eligible count EXACTLY (see module docstring for why
    asymmetry == eligibility by construction), so this is a two-sided pin: too low means
    order-sensitivity was lost, too high means the eligibility gate leaked.
    """
    pairs = _ordered_pairs(geometry)
    asym = sum(
        1
        for a, b in pairs
        if C.is_inwards_ordered(geometry, a, b) != C.is_inwards_ordered(geometry, b, a)
    )
    assert asym == n_eligible > 0


@pytest.mark.parametrize(("geometry", "n_pairs", "n_eligible"), GEOMETRIES)
def test_ordered_outward_is_not_swap_invariant(geometry, n_pairs, n_eligible):
    pairs = _ordered_pairs(geometry)
    asym = sum(
        1
        for a, b in pairs
        if C.is_outwards_ordered(geometry, a, b) != C.is_outwards_ordered(geometry, b, a)
    )
    assert asym == n_eligible > 0


@pytest.mark.parametrize(("geometry", "n_pairs", "n_eligible"), GEOMETRIES)
def test_ordered_predicates_partition_every_roll_eligible_pair(geometry, n_pairs, n_eligible):
    """Exactly one of inward/outward fires on every eligible pair — including flat rolls.

    A partition (not merely disjoint classes) is the property the unordered pair fails: it
    leaves the 108 same-row pairs in neither class. Splitting the eligible set exactly in
    half also falls out of the definition — reversing a stroke maps inward onto outward
    bijectively — so ``n_eligible / 2`` each is a derived expectation, not an observed one.
    """
    pairs = _ordered_pairs(geometry)
    inward = [(a, b) for a, b in pairs if C.is_inwards_ordered(geometry, a, b)]
    outward = [(a, b) for a, b in pairs if C.is_outwards_ordered(geometry, a, b)]
    eligible = [(a, b) for a, b in pairs if _eligible(geometry, a, b)]

    assert len(eligible) == n_eligible
    assert not [p for p in inward if p in set(outward)], "classes must be disjoint"
    assert len(inward) + len(outward) == n_eligible, "classes must COVER the eligible set"
    assert len(inward) == len(outward) == n_eligible // 2
    # and the reverse of every inward stroke is an outward one
    assert all(C.is_outwards_ordered(geometry, b, a) for a, b in inward)


def test_ordered_predicates_agree_with_the_unordered_ones_on_the_eligibility_GATE():
    """Same universe, different verdict inside it — the fix is a re-reading, not a re-scoping.

    Anything the unordered predicates fire on must be eligible for the ordered ones. This is
    what makes the two comparable: if the gates differed, an ``inroll`` share change could be
    a scope change masquerading as a direction effect.
    """
    geometry = ROW_STAGGERED_30
    for a, b in _ordered_pairs(geometry):
        if C.is_inwards(geometry, a, b) or C.is_outwards(geometry, a, b):
            assert C.is_inwards_ordered(geometry, a, b) or C.is_outwards_ordered(geometry, a, b)


@pytest.mark.parametrize(
    ("bigram", "inward", "outward", "why"),
    [
        ("as", 1, 0, "a(-5,2) pinky -> s(-4,2) ring: flat, travels toward index = INWARD"),
        ("sa", 0, 1, "the reverse stroke of 'as' must be OUTWARD, not the same class"),
        ("sd", 1, 0, "ring -> middle, flat: inward (unordered pair calls this NEITHER)"),
        ("ds", 0, 1, "middle -> ring, flat: outward"),
        ("qs", 1, 0, "q(-5,3) pinky top -> s(-4,2) ring home: inward and cross-row"),
        ("sq", 0, 1, "s -> q reverses it; the UNORDERED predicate calls both 'inwards'"),
        ("kj", 1, 0, "right hand: k(3,2) middle -> j(2,2) index is inward"),
        ("jk", 0, 1, "right hand: j -> k is outward — the hand's sign must not confuse it"),
        ("un", 0, 0, "u(2,3)/n(1,1) are BOTH right index: a same-finger reach is not a roll"),
        ("jf", 0, 0, "cross-hand: not a roll at all"),
    ],
)
def test_ordered_direction_on_concrete_qwerty_strokes(bigram, inward, outward, why):
    """Readable spot-checks, including the four cases the unordered predicate gets wrong.

    ``as``/``sa``/``sd``/``ds`` are flat rolls the old predicate scores 0/0; ``qs``/``sq``
    are a stroke and its reverse the old predicate puts in the SAME class.
    """
    from keybo.layout import Layout
    from keybo.layouts import NAMED_LAYOUTS

    layout = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
    assert int(C.is_inwards_ordered(ROW_STAGGERED_30, a, b)) == inward, why
    assert int(C.is_outwards_ordered(ROW_STAGGERED_30, a, b)) == outward, why


def test_index_two_columns_are_not_a_direction_step():
    """HAZARD (``geometry.same_finger``): index columns 1 and 2 are ONE finger.

    A bare ``abs(b[0]) - abs(a[0])`` step on column index reads ``(-2,2) -> (-1,2)`` as an
    inward move, but that is the index finger reaching to its own second column — a
    same-finger bigram, not a two-finger roll. ``oxey.py``'s module docstring records this
    exact mistake being fixed once already in its trigram path; the ordered predicates must
    not reintroduce it in the bigram path.
    """
    g = ROW_STAGGERED_30
    for a, b in (((-2, 2), (-1, 2)), ((-1, 2), (-2, 2)), ((1, 2), (2, 2)), ((2, 3), (1, 2))):
        assert g.same_finger(a[0], b[0]) is True
        assert C.is_inwards_ordered(g, a, b) is False
        assert C.is_outwards_ordered(g, a, b) is False


def test_ordered_predicates_ignore_the_row_entirely():
    """Direction of travel is a COLUMN fact; the row must not enter it.

    The old predicate's whole defect was reading rows for a horizontal question. Holding the
    columns fixed and sweeping both rows must not change the verdict.
    """
    g = ROW_STAGGERED_30
    for ay in (1, 2, 3):
        for by in (1, 2, 3):
            assert C.is_inwards_ordered(g, (-5, ay), (-4, by)) is True
            assert C.is_outwards_ordered(g, (-4, by), (-5, ay)) is True
