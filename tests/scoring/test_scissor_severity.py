"""Severity-weighted scissor gauge — a PREFERENCE, declared and swept (SCISSOR-SEVERITY).

The incumbent scissor gauge (``oxey.pattern_shares()["scissor"]``) is a flat corpus share: every
scissor counts 1.0. Two things it cannot express, both registered in
``docs/scissor-severity-preregistration.md``:

1. **which fingers** are involved — a pinky two-row reach counts the same as an index/middle one;
2. **which direction** the reach goes — ``is_scissor`` is ``abs(a[1] - b[1]) == 2``, which is
   direction-BLIND, so reaching DOWN to the bottom row scores identically to reaching UP.

This module pins the severity gauge that adds both as independently togglable weights.

The load-bearing test is :func:`test_positive_control_all_weights_one_reproduces_flat_share`:
at all weights 1.0 on the narrow support the gauge must reproduce the existing flat share
**exactly**, which is the proof that it is a strict generalization rather than a new metric that
merely correlates with the old one.

Two structural facts pinned here as tests, because a future change to ``is_scissor`` would
silently invalidate the report's conclusions otherwise:

* every scissor spans top(3) <-> bottom(1) on a three-row board, so a *static* bottom-row term is
  constant on the support and can weight nothing (only the SIGNED direction has variance);
* the narrow support contains no middle-pinky pair at all (``|dcol| = 2`` fails ``is_adjacent``),
  which is why the disputed middle-pinky sub-bin is invisible to the incumbent share.
"""

import pytest

from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.oxey import OxeyStyleScorer
from keybo.scoring.scissor_severity import (
    DEFAULT_SEVERITY,
    ScissorSeverity,
    SeverityWeights,
    bigram_severity,
)

# --- geometries built by hand so each case pins ONE thing -------------------------------------
# Signed x: left hand -5..-1. Columns 5=pinky, 4=ring, 3=middle, 2/1=index. y: 3 top, 2 home,
# 1 bottom. A tiny geometry is enough: the predicates read only positions.
PINKY_TOP = (-5, 3)
PINKY_HOME = (-5, 2)
PINKY_BOTTOM = (-5, 1)
RING_TOP = (-4, 3)
RING_HOME = (-4, 2)
RING_BOTTOM = (-4, 1)
MIDDLE_TOP = (-3, 3)
MIDDLE_BOTTOM = (-3, 1)
INDEX_TOP = (-2, 3)
INDEX_BOTTOM = (-2, 1)
RIGHT_INDEX_TOP = (2, 3)


@pytest.fixture(scope="module")
def geom():
    return ROW_STAGGERED_30


# --------------------------------------------------------------------------------------------
# 1. The three cases the brief names explicitly
# --------------------------------------------------------------------------------------------


def test_same_row_pair_scores_zero(geom):
    """A same-row adjacent-finger pair is not a scissor under any weighting."""
    w = SeverityWeights(pinky=4.0, down=3.0)
    assert bigram_severity(geom, PINKY_HOME, RING_HOME, w) == 0.0
    assert bigram_severity(geom, MIDDLE_TOP, RING_TOP, w) == 0.0
    # ... and a one-row reach is deliberately zero too (span must be exactly 2).
    assert bigram_severity(geom, PINKY_HOME, RING_TOP, w) == 0.0


def test_pinky_pair_outweighs_non_pinky_pair_at_equal_geometry(geom):
    """Equal geometry, different fingers: the pinky pair must carry more severity.

    Both pairs are adjacent-finger, both span top->bottom, both go the same direction. The ONLY
    difference is which fingers press them -- so any difference in severity is attributable to
    component (a) alone.
    """
    w = SeverityWeights(pinky=2.0, down=1.0)  # direction off, so this isolates the pinky term
    pinky_pair = bigram_severity(geom, PINKY_TOP, RING_BOTTOM, w)
    index_pair = bigram_severity(geom, MIDDLE_TOP, INDEX_BOTTOM, w)
    assert pinky_pair == pytest.approx(2.0)
    assert index_pair == pytest.approx(1.0)
    assert pinky_pair > index_pair


def test_downward_reach_outweighs_upward_reach(geom):
    """Same two keys, opposite order: reaching DOWN must cost more than reaching UP.

    This is the case the incumbent predicate provably cannot express -- ``is_scissor`` is
    symmetric in its arguments, so it returns the same value for both orders.
    """
    w = SeverityWeights(pinky=1.0, down=1.5)  # pinky off, so this isolates the direction term
    down = bigram_severity(geom, MIDDLE_TOP, INDEX_BOTTOM, w)  # y 3 -> 1, reaching down
    up = bigram_severity(geom, INDEX_BOTTOM, MIDDLE_TOP, w)  # y 1 -> 3, reaching up
    assert down == pytest.approx(1.5)
    assert up == pytest.approx(1.0)
    assert down > up
    # And the incumbent predicate really is blind to the distinction being drawn:
    assert C.is_scissor(geom, MIDDLE_TOP, INDEX_BOTTOM) == C.is_scissor(
        geom, INDEX_BOTTOM, MIDDLE_TOP
    )


# --------------------------------------------------------------------------------------------
# 2. Tiering: pinky > ring > neither, monotone by construction
# --------------------------------------------------------------------------------------------


def test_ring_tier_sits_strictly_between_pinky_and_neither(geom):
    """The preregistered tiering: weakest finger sets the tier, monotone in weakness."""
    w = SeverityWeights(pinky=3.0, ring_ratio=0.5, down=1.0)
    assert w.ring == pytest.approx(2.0)  # 1 + 0.5*(3-1)
    pinky = bigram_severity(geom, PINKY_TOP, RING_BOTTOM, w)  # pinky involved
    ring = bigram_severity(geom, RING_TOP, MIDDLE_BOTTOM, w)  # ring, no pinky
    neither = bigram_severity(geom, MIDDLE_TOP, INDEX_BOTTOM, w)  # index/middle only
    assert pinky == pytest.approx(3.0)
    assert ring == pytest.approx(2.0)
    assert neither == pytest.approx(1.0)
    assert pinky > ring > neither


def test_ring_ratio_endpoints(geom):
    """ratio 0 collapses ring to the base tier; ratio 1 promotes it to the pinky tier."""
    ring_pair = (RING_TOP, MIDDLE_BOTTOM)
    at_zero = SeverityWeights(pinky=4.0, ring_ratio=0.0)
    at_one = SeverityWeights(pinky=4.0, ring_ratio=1.0)
    assert bigram_severity(geom, *ring_pair, at_zero) == pytest.approx(1.0)
    assert bigram_severity(geom, *ring_pair, at_one) == pytest.approx(4.0)


def test_pinky_tier_wins_when_pair_is_pinky_ring(geom):
    """A pinky-ring pair involves BOTH tiers; the weakest finger (pinky) must decide."""
    w = SeverityWeights(pinky=4.0, ring_ratio=0.5)
    assert bigram_severity(geom, PINKY_TOP, RING_BOTTOM, w) == pytest.approx(4.0)


def test_components_multiply_and_are_independently_togglable(geom):
    """Separability: each component can be switched off without touching the other."""
    both = SeverityWeights(pinky=2.0, down=1.5)
    assert bigram_severity(geom, PINKY_TOP, RING_BOTTOM, both) == pytest.approx(3.0)
    pinky_only = SeverityWeights(pinky=2.0, down=1.0)
    assert bigram_severity(geom, PINKY_TOP, RING_BOTTOM, pinky_only) == pytest.approx(2.0)
    down_only = SeverityWeights(pinky=1.0, down=1.5)
    assert bigram_severity(geom, PINKY_TOP, RING_BOTTOM, down_only) == pytest.approx(1.5)
    off = SeverityWeights()
    assert bigram_severity(geom, PINKY_TOP, RING_BOTTOM, off) == pytest.approx(1.0)


def test_weights_reject_values_below_one():
    """A severity weight below 1.0 would turn a penalty into a reward -- refuse it loudly."""
    with pytest.raises(ValueError):
        SeverityWeights(pinky=0.5)
    with pytest.raises(ValueError):
        SeverityWeights(down=0.0)
    with pytest.raises(ValueError):
        SeverityWeights(ring_ratio=-0.1)
    with pytest.raises(ValueError):
        SeverityWeights(ring_ratio=1.5)


# --------------------------------------------------------------------------------------------
# 3. Support (component c): narrow vs wide
# --------------------------------------------------------------------------------------------


def test_cross_hand_and_same_finger_pairs_are_zero_on_both_supports(geom):
    """Neither support may leak cross-hand pairs or same-finger reaches into scissor mass."""
    for support in ("narrow", "wide"):
        w = SeverityWeights(pinky=4.0, down=3.0, support=support)
        # cross-hand, spans two rows
        assert bigram_severity(geom, INDEX_TOP, RIGHT_INDEX_TOP, w) == 0.0
        assert bigram_severity(geom, MIDDLE_TOP, (3, 1), w) == 0.0
        # same finger (pinky column to itself), spans two rows -- an SFB, not a scissor
        assert bigram_severity(geom, PINKY_TOP, PINKY_BOTTOM, w) == 0.0
        # index columns 1 and 2 are the SAME finger
        assert bigram_severity(geom, (-1, 3), (-2, 1), w) == 0.0


def test_wide_support_admits_the_skipped_finger_pair_narrow_rejects(geom):
    """The middle-pinky pair: |dcol| = 2, so ``is_adjacent`` rejects it and the narrow gauge
    scores it 0. The wide support (DIST-1's ``wscissor``) prices it."""
    narrow = SeverityWeights(support="narrow")
    wide = SeverityWeights(support="wide")
    assert bigram_severity(geom, MIDDLE_TOP, PINKY_BOTTOM, narrow) == 0.0
    assert bigram_severity(geom, MIDDLE_TOP, PINKY_BOTTOM, wide) == pytest.approx(1.0)
    # and it picks up the pinky tier + downward direction when those are switched on
    weighted = SeverityWeights(pinky=2.0, down=1.5, support="wide")
    assert bigram_severity(geom, MIDDLE_TOP, PINKY_BOTTOM, weighted) == pytest.approx(3.0)


def test_narrow_support_is_a_strict_subset_of_wide(geom):
    """Exhaustive over all 900 ordered slot pairs: anything the narrow gauge charges, the wide
    gauge charges identically. The supports may not disagree on shared pairs."""
    narrow = SeverityWeights(pinky=2.0, down=1.5, support="narrow")
    wide = SeverityWeights(pinky=2.0, down=1.5, support="wide")
    slots = ROW_STAGGERED_30.slots
    shared = 0
    for a in slots:
        for b in slots:
            n = bigram_severity(geom, a, b, narrow)
            if n:
                assert bigram_severity(geom, a, b, wide) == pytest.approx(n)
                shared += 1
    assert shared == 24  # the narrow support, as enumerated in the preregistration


def test_unknown_support_is_rejected():
    with pytest.raises(ValueError):
        SeverityWeights(support="everything")


# --------------------------------------------------------------------------------------------
# 4. Structural facts, pinned so a future predicate change breaks loudly
# --------------------------------------------------------------------------------------------


def test_every_scissor_spans_top_to_bottom_so_static_bottom_row_term_is_degenerate(geom):
    """Registered structural constraint: on a three-row board ``abs(dy) == 2`` forces
    top(3) <-> bottom(1). A *static* "involves the bottom row" flag is therefore constant on the
    entire support and cannot weight anything -- the same degeneracy DIST-1 proved for
    ``scissor-vdist``. If a future geometry adds a row, this test fails and the report's
    direction-based design must be revisited."""
    row_pairs = set()
    for a in ROW_STAGGERED_30.slots:
        for b in ROW_STAGGERED_30.slots:
            if C.is_scissor(geom, a, b) or ScissorSeverity._in_wide_support(geom, a, b):
                row_pairs.add((a[1], b[1]))
    assert row_pairs == {(1, 3), (3, 1)}
    assert all(1 in rp and 3 in rp for rp in row_pairs)


def test_narrow_support_contains_no_middle_pinky_pair(geom):
    """Registered structural constraint, and the reason component (a) alone cannot move the
    disputed veto: middle (col 3) to pinky (col 5) is a column gap of 2, so ``is_adjacent``
    rejects it and the incumbent flat share never sees it."""
    kinds = {5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
    narrow_pairs, wide_pairs = set(), set()
    for a in ROW_STAGGERED_30.slots:
        for b in ROW_STAGGERED_30.slots:
            pair = tuple(sorted((kinds[abs(a[0])], kinds[abs(b[0])])))
            if C.is_scissor(geom, a, b):
                narrow_pairs.add(pair)
            if ScissorSeverity._in_wide_support(geom, a, b):
                wide_pairs.add(pair)
    assert ("middle", "pinky") not in narrow_pairs
    assert ("index", "pinky") not in narrow_pairs
    assert ("index", "ring") not in narrow_pairs
    assert ("middle", "pinky") in wide_pairs


# --------------------------------------------------------------------------------------------
# 5. THE POSITIVE CONTROL — the proof of strict generalization
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_corpus():
    """Top rows of the real iWeb tables: enough mass for a meaningful share comparison."""
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load(path, limit=4000):
        out = {}
        with open(os.path.join(root, path), encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    out[parts[0]] = int(parts[1])
        return out

    return (
        load("data/corpus/bigrams.txt"),
        load("data/corpus/1-skip.txt"),
        load("data/corpus/trigrams.txt"),
    )


def test_positive_control_all_weights_one_reproduces_flat_share(real_corpus):
    """THE control: at all weights 1.0 on the narrow support, the severity gauge must equal the
    incumbent flat ``oxey`` scissor share EXACTLY (max abs err 0.0).

    Anything less than exact equality means the gauge is a *different* metric that merely
    correlates with the old one, and every severity number would be confounded by the
    reimplementation. Exactness is what makes it a strict generalization.
    """
    bigrams, skipgrams, trigrams = real_corpus
    oxey = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    gauge = ScissorSeverity(bigrams)
    flat = SeverityWeights()  # all 1.0, narrow
    worst = 0.0
    for name, s in NAMED_LAYOUTS.items():
        layout = Layout(s, ROW_STAGGERED_30)
        incumbent = oxey.pattern_shares(layout)["scissor"]
        severity = gauge.share(layout, flat)
        worst = max(worst, abs(incumbent - severity))
        assert severity == incumbent, f"{name}: {severity!r} != {incumbent!r}"
    assert worst == 0.0


def test_severity_share_is_monotone_nondecreasing_in_each_weight(real_corpus):
    """Raising a severity weight can never lower the gauge: it is a positive reweighting of a
    fixed nonnegative support."""
    bigrams, _, _ = real_corpus
    gauge = ScissorSeverity(bigrams)
    layout = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    base = gauge.share(layout, SeverityWeights())
    assert gauge.share(layout, SeverityWeights(pinky=2.0)) >= base
    assert gauge.share(layout, SeverityWeights(down=2.0)) >= base
    assert gauge.share(layout, SeverityWeights(support="wide")) >= base
    # strictly greater, in fact -- qwerty has mass in all three
    assert gauge.share(layout, SeverityWeights(pinky=2.0)) > base
    assert gauge.share(layout, SeverityWeights(down=2.0)) > base
    assert gauge.share(layout, SeverityWeights(support="wide")) > base


def test_default_severity_is_the_preregistered_preference():
    """The headline preference P, pinned so it cannot drift after the fact."""
    assert DEFAULT_SEVERITY.pinky == 2.0
    assert DEFAULT_SEVERITY.ring_ratio == 0.5
    assert DEFAULT_SEVERITY.ring == 1.5
    assert DEFAULT_SEVERITY.down == 1.5
    assert DEFAULT_SEVERITY.support == "narrow"


def test_breakdown_masses_sum_to_the_share(real_corpus):
    """The per-class breakdown is an exact decomposition of the headline number, so the report's
    attribution cannot silently lose or double-count mass."""
    bigrams, _, _ = real_corpus
    gauge = ScissorSeverity(bigrams)
    layout = Layout(NAMED_LAYOUTS["colemak"], ROW_STAGGERED_30)
    for w in (
        SeverityWeights(),
        DEFAULT_SEVERITY,
        SeverityWeights(pinky=4.0, down=3.0, support="wide"),
    ):
        bd = gauge.breakdown(layout, w)
        assert sum(bd.values()) == pytest.approx(gauge.share(layout, w), abs=1e-12)


def test_empty_corpus_is_zero_not_a_crash():
    gauge = ScissorSeverity({})
    layout = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    assert gauge.share(layout, DEFAULT_SEVERITY) == 0.0


def test_gauge_ignores_bigrams_not_fully_on_the_layout(real_corpus):
    """Denominator convention must match oxey's: only bigrams whose BOTH chars are on the
    layout count, toward either numerator or denominator."""
    bigrams = {"qc": 100, "q9": 5000, "9!": 900}  # only 'qc' is on a letters-only layout
    gauge = ScissorSeverity(bigrams)
    only_qc = ScissorSeverity({"qc": 100})
    layout = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    w = SeverityWeights(support="wide")
    assert gauge.share(layout, w) == pytest.approx(only_qc.share(layout, w))


def test_three_row_geometry_assumption_is_asserted(geom):
    """The gauge's direction semantics assume a 3-row board. A 4-row geometry must fail loudly
    rather than silently mislabel a home-to-top reach as a bottom-row reach."""
    four_row = Geometry(slots=tuple((x, y) for y in (4, 3, 2, 1) for x in (-5, -4, -3)))
    with pytest.raises(ValueError, match="three-row"):
        ScissorSeverity({"ab": 1}).share(Layout("abcdefghijkl", four_row), DEFAULT_SEVERITY)
