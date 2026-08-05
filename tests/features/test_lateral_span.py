"""`lateral_span` — the graded lateral-stretch measure, and the OLD predicate's blind spot.

Written before the implementation (LSBWIDEN-1, 2026-08-01). Two jobs:

1. **Pin the MOTIVATION so it cannot be lost.** ``test_the_old_predicate_*`` assert
   ``is_lsb``'s blind spot and its LAYOUT-DEPENDENCE directly. If someone later "simplifies"
   ``lateral_span`` away, these tests state on the record why it existed.
2. **Pin the new measure's behaviour**, including the two properties that make it a
   generalization rather than a rival: it reproduces ``is_lsb`` exactly on the index-middle
   support, and it is a superset of it.

Every expected count here is from an exhaustive enumeration of ``ROW_STAGGERED_30``'s 870
ordered position pairs, recomputed in-test rather than quoted, except the census totals
(72/64/28/4/4) which are the independently published figures from ``state/closeout-unknown``
D7 and so are genuine positive controls.
"""

from __future__ import annotations

import itertools

import pytest

from keybo.data.corpus import load_frequencies
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as GEOM
from keybo.geometry import ROW_STAGGERED_31, Geometry
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS

PAIRS = list(itertools.permutations(GEOM.slots, 2))
_RANK = {"index": 3, "middle": 2, "ring": 1, "pinky": 0}


def _finger(x: int) -> str:
    return GEOM.finger(x).value.split("-")[1]


def _pair_label(a, b) -> str:
    first, second = sorted((_finger(a[0]), _finger(b[0])), key=lambda k: -_RANK[k])
    return f"{first}-{second}"


def _two_finger_same_hand(a, b) -> bool:
    return C.same_hand(GEOM, a, b) and not C.same_finger(GEOM, a, b)


# --- 1. the OLD predicate's defect, pinned so the motivation survives -------------------


def test_the_old_predicate_only_ever_fires_on_index_middle() -> None:
    """``is_lsb`` hardcodes the ('index','middle') pair — the defect, stated as a test."""
    labels = {_pair_label(a, b) for a, b in PAIRS if C.is_lsb(GEOM, a, b)}
    assert labels == {"index-middle"}


def test_the_old_predicate_can_never_flag_these_wide_classes() -> None:
    """The five classes with real lateral span that ``is_lsb`` structurally cannot see.

    Counts are ``state/closeout-unknown`` D7's independently published census.
    """
    unflaggable: dict[str, int] = {}
    for a, b in PAIRS:
        if C.is_lsb(GEOM, a, b):
            continue
        if _two_finger_same_hand(a, b) and GEOM.stagger_adjusted_dx(a, b) > 1.5:
            label = _pair_label(a, b)
            unflaggable[label] = unflaggable.get(label, 0) + 1
    assert unflaggable == {
        "index-pinky": 72,
        "index-ring": 64,
        "middle-pinky": 28,
        "middle-ring": 4,
        "ring-pinky": 4,
    }
    assert sum(unflaggable.values()) == 172


def test_the_old_predicates_blind_spot_is_LAYOUT_DEPENDENT() -> None:
    """THE WHOLE ARGUMENT: the flagged fraction differs per layout, so the gauge cannot rank
    layouts consistently even in principle.

    Asserts the *shape* (a large fold spread, qwerty best-covered, graphite worst) rather than
    the exact ratios, so a corpus refresh cannot silently delete the motivation.

    Deliberately NOT ``NAMED_LAYOUTS.items()`` — this is a claim about the specific five
    layouts registered when the test was written (2026-08-01), not about whatever the
    registry happens to hold later. A later-added layout could legitimately score worse than
    graphite on this metric without that saying anything about graphite's own coverage.
    """
    original_named_layouts = {
        name: NAMED_LAYOUTS[name]
        for name in ("qwerty", "dvorak", "colemak", "graphite", "semimak")
    }
    coverage = {}
    for name, chars in original_named_layouts.items():
        layout = Layout(chars, GEOM)
        flagged = phenomenon = 0.0
        for bigram, freq in _CORPUS.items():
            if not all(layout.has_key(ch) for ch in bigram):
                continue
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            if _two_finger_same_hand(a, b) and GEOM.stagger_adjusted_dx(a, b) > 1.5:
                phenomenon += freq
                if C.is_lsb(GEOM, a, b):
                    flagged += freq
        coverage[name] = flagged / phenomenon
    assert max(coverage.values()) / min(coverage.values()) > 3.0
    assert max(coverage, key=coverage.get) == "qwerty"
    assert min(coverage, key=coverage.get) == "graphite"


# --- 2. the new measure ------------------------------------------------------------------


def test_lateral_span_is_zero_off_support() -> None:
    """Cross-hand, same-finger, and at-or-inside-neutral pairs all score 0.0."""
    left, right = (-2, 2), (2, 2)
    assert C.lateral_span(GEOM, left, right) == 0.0  # cross-hand
    assert C.lateral_span(GEOM, (-1, 2), (-2, 2)) == 0.0  # index cols 1&2 = same finger
    assert C.lateral_span(GEOM, (-2, 2), (-3, 2)) == 0.0  # index-middle at neutral
    assert C.lateral_span(GEOM, (-3, 2), (-4, 2)) == 0.0  # middle-ring at neutral


def test_lateral_span_measures_EXCESS_over_the_pairs_neutral_separation() -> None:
    """The load-bearing design choice: a raw ``dx`` threshold fires on index-pinky's REST
    posture (they sit ~3 columns apart), so it would measure "two far fingers were used"
    rather than a stretch. Excess-over-neutral is what makes the classes comparable."""
    # index home col 2, pinky home col 5 -> neutral separation 3, so a home-row
    # index-pinky pair is NOT a stretch even though its raw dx is 3.
    assert GEOM.stagger_adjusted_dx((-2, 2), (-5, 2)) == pytest.approx(3.0)
    assert C.lateral_span(GEOM, (-2, 2), (-5, 2)) == 0.0
    # the index STRETCH column (|x| == 1) against the pinky is a genuine one-column stretch
    assert C.lateral_span(GEOM, (-1, 2), (-5, 2)) == pytest.approx(1.0)


def test_lateral_span_reproduces_is_lsb_EXACTLY_on_the_index_middle_support() -> None:
    """A generalization, not a rival: on ``is_lsb``'s own support the two agree on all 870
    pairs. ``index-middle`` neutral separation is 1, so ``excess > 0.5`` IS ``dx > 1.5``."""
    disagreements = [
        (a, b)
        for a, b in PAIRS
        if _pair_label(a, b) == "index-middle"
        and _two_finger_same_hand(a, b)
        and C.is_lsb(GEOM, a, b) != (C.lateral_span(GEOM, a, b) > 0.5)
    ]
    assert disagreements == []


def test_every_is_lsb_pair_has_positive_lateral_span() -> None:
    """Strict superset: nothing the incumbent flags becomes invisible."""
    assert [
        (a, b) for a, b in PAIRS if C.is_lsb(GEOM, a, b) and not C.lateral_span(GEOM, a, b)
    ] == []


def test_lateral_span_equalizes_the_per_class_firing_rate() -> None:
    """The fix, measured: a raw-dx gate fires on 100% of index-pinky pairs and 11% of
    middle-ring ones. Excess-over-neutral gives every class the same rate on equal geometry —
    44.4% for the three index classes, 11.1% for the three narrow ones."""
    naive: dict[str, list[bool]] = {}
    excess: dict[str, list[bool]] = {}
    for a, b in PAIRS:
        if not _two_finger_same_hand(a, b):
            continue
        label = _pair_label(a, b)
        naive.setdefault(label, []).append(GEOM.stagger_adjusted_dx(a, b) > 1.5)
        excess.setdefault(label, []).append(C.lateral_span(GEOM, a, b) > 0.5)
    assert sum(naive["index-pinky"]) == len(naive["index-pinky"])  # 100% — unconditional
    rates = {k: sum(v) / len(v) for k, v in excess.items()}
    assert rates["index-middle"] == pytest.approx(rates["index-ring"])
    assert rates["index-ring"] == pytest.approx(rates["index-pinky"])
    assert rates["middle-ring"] == pytest.approx(rates["ring-pinky"])


def test_lateral_span_is_symmetric_on_every_pair() -> None:
    """A property of the two KEYS, like ``is_lsb`` and ``bad_scissor``. Direction is priced by
    the corpus supplying both orderings, not by the measure."""
    assert all(C.lateral_span(GEOM, a, b) == C.lateral_span(GEOM, b, a) for a, b in PAIRS)


def test_lateral_span_is_never_negative() -> None:
    assert all(C.lateral_span(GEOM, a, b) >= 0.0 for a, b in PAIRS)


def test_lateral_span_handles_the_K31_quote_slot() -> None:
    """|x| == 6 is the pinky's stretch column (``is_lateral``'s own reading), so its neutral
    separation is the pinky's home column 5 — and a col-6/col-5 pair is the SAME finger."""
    assert C.lateral_span(ROW_STAGGERED_31, (6, 2), (5, 2)) == 0.0  # same pinky
    assert C.lateral_span(ROW_STAGGERED_31, (6, 2), (4, 2)) == pytest.approx(1.0)


def test_lateral_span_class_bands_the_graded_value() -> None:
    """The banded reading is kept as a LABEL for reporting; the graded value is the measure."""
    assert C.lateral_span_class(GEOM, (-2, 2), (-3, 2)) == 0
    assert C.lateral_span_class(GEOM, (-1, 2), (-5, 2)) == 1
    assert C.lateral_span_class(GEOM, (-1, 1), (-5, 3)) == 2


def test_lateral_span_class_is_zero_exactly_when_span_is_subthreshold() -> None:
    for a, b in PAIRS:
        banded = C.lateral_span_class(GEOM, a, b)
        assert (banded == 0) == (C.lateral_span(GEOM, a, b) <= 0.5)


def test_lateral_span_refuses_a_geometry_it_has_no_neutral_column_for() -> None:
    """Neutral separation is read from a column->home-column table, so an unknown column must
    fail loudly rather than score against an invented neutral."""
    weird = Geometry(slots=((-9, 2), (9, 2), (-3, 2), (3, 2)))
    with pytest.raises(KeyError):
        C.lateral_span(weird, (-9, 2), (-3, 2))


_CORPUS = {
    bigram: freq
    for bigram, freq in load_frequencies("data/corpus/bigrams.txt").items()
    if len(bigram) == 2 and " " not in bigram
}
