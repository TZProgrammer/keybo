"""Tests for the model-free frame-collapse diagnostic (:mod:`keybo.analysis.frame_collapse`).

Bars registered in ``agent-artifacts/framediag/FRAMEDIAG-preregistration.md`` §4 BEFORE any number
this module asserts existed. Every assertion here was MUTATION-TESTED: the mutation that should break
it was applied and the test confirmed RED. Three tests in this campaign were green while not testing
their own names and none was findable by re-reading, so a green suite is not evidence — the mutation
log lives in ``state/framediag/report.md`` §(f).

The synthetic featurizers below are the point of the design, not test scaffolding: the diagnostic must
work for a frame that does not exist yet (INVARIANT 2), and a featurizer whose collapse structure and
floor are known BY HAND is the only way to test a floor against a known answer (INVARIANT 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis.frame_collapse import (
    cell_positions,
    feature_matrix,
    format_report,
    frame_collapse,
    group_cells,
    sweep_verdict,
    tolerance_sweep,
)
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31, Geometry

# --- tiny synthetic geometries and frames ------------------------------------------------------
#
# A 2-slot board with the space slot EXCLUDED gives a 2-cell order-1 space: the smallest setting in
# which "two cells forced to share one prediction" is expressible at all, which is what makes the
# known-answer floor derivable by hand rather than by running the code.

TWO_SLOT = Geometry(slots=((-1, 2), (1, 2)))


def _constant(_geometry, _cell):
    """Every cell gets the same row: TOTAL collapse, one group."""
    return np.array([1.0, 2.0])


def _injective(_geometry, cell):
    """Every cell gets a distinct row: ZERO collapse. Keyed on the positions themselves."""
    return np.array([float(p) for pos in cell for p in pos])


def _second_key_only(_geometry, cell):
    """Row depends only on the LAST key — the shape of a real frame's blindness to the first key."""
    return np.asarray(cell[-1], dtype=np.float64)


# --- INVARIANT 4: the known-answer floor -------------------------------------------------------


def test_known_answer_floor_distinguishes_median_from_mean_estimator():
    """The registered known-answer test: hand-derived floors on a 2-cell total collapse.

    Both cells share one row, so one prediction ``p`` must cover targets ``t = (t0, t0+d)`` carrying
    weights ``(w0, w1)``. Derived by hand, not by running the code (prereg §4 T-KNOWN-FLOOR):

    * The L1 minimizer is the WEIGHTED MEDIAN. With ``w = (3, 1)`` more than half the weight sits on
      ``t0``, so ``p = t0`` and the cost is ``d * w1 / (w0 + w1) = 4 * 1/4 = 1.0``.
    * The GROUP MEAN is ``t0 + d * w1/(w0+w1) = t0 + 1``, giving cost
      ``2 * d * w0 * w1 / (w0+w1)**2 = 2*4*3*1/16 = 1.5``.

    **1.0 != 1.5 is the whole point**: this is the case that would not distinguish the two estimators
    if the weights were equal (both give ``d/2 = 2``), which is exactly why INTERPFRAME-1's driver
    could use the mean for a wmae floor without the substitution being visible.
    """
    d, t0 = 4.0, 10.0
    r = frame_collapse(
        _constant,
        TWO_SLOT,
        order=1,
        include_space=False,
        target=np.array([t0, t0 + d]),
        weights=np.array([3.0, 1.0]),
    )
    assert r.n_cells == 2
    assert r.distinct_feature_rows == 1
    assert r.collapsed_cells == 2

    assert r.floor_wmae == pytest.approx(1.0, abs=1e-12)
    assert r.floor_wmae_at_group_mean == pytest.approx(1.5, abs=1e-12)
    # The published estimator is strictly WORSE than the true floor here — the inequality that makes
    # "2.2399 ms is a floor" an over-statement on the real frame too.
    assert r.floor_wmae > 0
    assert r.floor_wmae_at_group_mean > r.floor_wmae

    # L2: the mean IS the minimizer, so wrmse is the weighted RMS about t0+1 = sqrt(3/4*1+1/4*9).
    assert r.floor_wrmse == pytest.approx(np.sqrt(3.0), abs=1e-12)


def test_known_answer_floor_equal_weights_collapses_the_two_estimators():
    """The companion the test above needs to be meaningful: at EQUAL weights both give ``d/2``.

    Without this, the previous test's inequality could come from a bug that inflates the mean-based
    number everywhere. Here the two must AGREE exactly, so only the unequal-weight case separates
    them — which pins the mechanism (which within-group constant is used) rather than a magnitude.
    """
    d, t0 = 4.0, 10.0
    r = frame_collapse(
        _constant, TWO_SLOT, order=1, include_space=False, target=np.array([t0, t0 + d])
    )
    assert r.floor_wmae == pytest.approx(d / 2, abs=1e-12)
    assert r.floor_wmae_at_group_mean == pytest.approx(d / 2, abs=1e-12)
    assert r.floor_wrmse == pytest.approx(d / 2, abs=1e-12)


def test_known_answer_floor_three_cells_median_is_not_the_mean():
    """A 3-cell group where mean and median differ even at UNIFORM weights: t = (0, 1, 10).

    Hand-derived: median 1 -> cost (1 + 0 + 9)/3 = 10/3; mean 11/3 -> cost
    (11/3 + 8/3 + 19/3)/3 = 38/9. ``10/3 < 38/9`` — an outlier is exactly where the estimator choice
    bites, and the interp.1 frame's largest group holds 16 cells spanning 93 ms.
    """
    r = frame_collapse(
        _constant,
        Geometry(slots=((-1, 2), (1, 2), (2, 2))),
        order=1,
        include_space=False,
        target=np.array([0.0, 1.0, 10.0]),
    )
    assert r.floor_wmae == pytest.approx(10 / 3, abs=1e-12)
    assert r.floor_wmae_at_group_mean == pytest.approx(38 / 9, abs=1e-12)


def test_zero_collapse_frame_has_perfect_resolution_and_exactly_zero_floors():
    """An injective featurizer: nothing is forced to share, so both floors are EXACTLY 0.0."""
    r = frame_collapse(
        _injective, ROW_STAGGERED_30, order=2, target=np.arange(961, dtype=np.float64)
    )
    assert r.n_cells == 961
    assert r.distinct_feature_rows == 961
    assert r.collapsed_cells == 0
    assert r.resolution == 1.0
    assert r.collapsed_share == 0.0
    assert r.mass_share_collapsed == 0.0
    assert r.largest_group == 1
    assert r.floor_wmae == 0.0
    assert r.floor_wmae_at_group_mean == 0.0
    assert r.floor_wrmse == 0.0
    # ⚠ and the flag must NOT fire: there are no collapse groups, so "all groups have zero spread" is
    # vacuously true and a naive implementation would flag a PERFECTLY RESOLVED frame as tautological.
    assert r.target_is_self_generated is False
    assert r.n_collapse_groups == 0


def test_total_collapse_floor_at_group_mean_is_the_mean_absolute_deviation():
    """One group over everything: the mean-based floor IS the target's MAD about its weighted mean."""
    t = np.array([1.0, 5.0, 6.0, 100.0])
    r = frame_collapse(
        _constant,
        Geometry(slots=((-2, 2), (-1, 2), (1, 2), (2, 2))),
        order=1,
        include_space=False,
        target=t,
    )
    assert r.distinct_feature_rows == 1
    assert r.collapsed_cells == 4
    assert r.floor_wmae_at_group_mean == pytest.approx(np.abs(t - t.mean()).mean(), abs=1e-12)
    # and the median-based floor is the MAD about the (lower) median, 5.0 — strictly smaller.
    assert r.floor_wmae == pytest.approx(np.abs(t - 5.0).mean(), abs=1e-12)
    assert r.floor_wmae < r.floor_wmae_at_group_mean


def test_floors_are_none_without_a_target_because_zero_is_a_real_answer():
    """No target -> ``None``, never 0.0: a zero floor is the strongest real result this can return."""
    r = frame_collapse(_constant, TWO_SLOT, order=1, include_space=False)
    assert r.floor_wmae is None
    assert r.floor_wmae_at_group_mean is None
    assert r.floor_wrmse is None
    assert r.distinct_feature_rows == 1  # structure is still measured


# --- INVARIANT 3: the tolerance ----------------------------------------------------------------


def test_quantization_never_exceeds_the_exact_count_and_the_bar_is_not_vacuous():
    """T-COARSENING (prereg §8): the ONE tolerance guarantee that is true.

    Exact-equal rows quantize equally, so every quantized partition coarsens the EXACT one. This is
    the clause the 765-vs-775 resolution rests on: no tolerance can raise a count.
    """
    tols = [0.0, 1e-12, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    rs = tolerance_sweep(
        lambda g, c: np.asarray(_second_key_only(g, c)) / 3.0, ROW_STAGGERED_30, tols=tols, order=2
    )
    exact = rs[0].distinct_feature_rows
    assert rs[0].tol == 0.0
    counts = [r.distinct_feature_rows for r in rs]
    assert all(c <= exact for c in counts), counts
    # NOT VACUOUS: at least one tolerance must actually merge something, or this asserts nothing.
    assert min(counts) < exact, counts
    v = sweep_verdict(rs)
    assert v["exceeds_exact"] is False
    assert v["flat"] is False


def test_distinct_rows_is_NOT_monotone_between_two_nonzero_tolerances():
    """T-NONMONOTONE (prereg §8): pin the counterexample so the false claim cannot come back.

    ``round(x/tol)`` bin boundaries MOVE with ``tol``, so a coarser grid can SPLIT a pair a finer grid
    merged. Registered after my own T-MONO bar was refuted by measurement on the served frame
    (0.5 -> 701 but 0.75 -> 709). This test exists so a future reader who "fixes" the docstring back
    to the intuitive monotone claim gets a RED test instead of a plausible falsehood.
    """
    # The minimal scalar instance, asserted directly on the arithmetic.
    assert round(0.3 / 0.5) == round(0.4 / 0.5)  # merged at the FINER tolerance
    assert round(0.3 / 0.75) != round(0.4 / 0.75)  # SPLIT at the COARSER one

    # And the same effect through the shipped grouping rule, on a 2-cell frame.
    X = np.array([[0.3], [0.4]])
    assert len(group_cells(X, tol=0.5)[1]) == 1
    assert len(group_cells(X, tol=0.75)[1]) == 2

    # Reported as a RISE, and explicitly NOT as an exact-count violation: the two are different.
    rs = tolerance_sweep(
        lambda _g, c: np.array([c[0][0] * 0.1]),
        TWO_SLOT,
        tols=[0.0, 0.5, 0.75],
        order=1,
        include_space=False,
    )
    v = sweep_verdict(rs)
    assert v["exceeds_exact"] is False


def test_exceeds_exact_detects_a_count_above_the_exact_count():
    """``exceeds_exact`` must be able to fire, or every ``is False`` assertion on it is VACUOUS.

    ⚠ FOUND BY MUTATION, NOT BY READING: hard-coding ``exceeds_exact`` to ``False`` left the whole
    suite GREEN (mutation M20), because every other test asserts it is False — the exact
    green-test-that-tests-nothing shape this campaign hit three times. ``exceeds_exact`` is an
    IMPOSSIBILITY check, so no real frame can make it fire; the only way to test it is to hand
    :func:`sweep_verdict` a fabricated sweep in which a ``tol>0`` count exceeds the ``tol=0`` count,
    which is what this does.
    """
    from dataclasses import replace

    base = frame_collapse(_constant, TWO_SLOT, order=1, include_space=False)
    exact = replace(base, tol=0.0, distinct_feature_rows=5)
    coarse = replace(base, tol=1e-6, distinct_feature_rows=7)  # impossible in real data

    v = sweep_verdict([exact, coarse])
    assert v["exceeds_exact"] is True
    assert v["rises"] == [(0.0, 1e-6)]
    assert v["flat"] is False

    # A step that stays EQUAL is not a rise. ⚠ Found by mutation M21b: relaxing ``>`` to ``>=`` in the
    # rise detector left the suite green, because every real sweep either rises strictly or is flat —
    # and on a FLAT sweep nothing asserted that ``rises`` stays empty.
    equal = sweep_verdict(
        [
            replace(base, tol=0.0, distinct_feature_rows=5),
            replace(base, tol=1e-6, distinct_feature_rows=5),
        ]
    )
    assert equal["rises"] == []
    assert equal["flat"] is True
    assert equal["exceeds_exact"] is False
    # ...and a strict FALL is not a rise either.
    assert (
        sweep_verdict(
            [
                replace(base, tol=0.0, distinct_feature_rows=5),
                replace(base, tol=1e-6, distinct_feature_rows=4),
            ]
        )["rises"]
        == []
    )

    # and it stays False on the same pair the legal way round, so it is not a constant either.
    assert (
        sweep_verdict(
            [
                replace(base, tol=0.0, distinct_feature_rows=7),
                replace(base, tol=1e-6, distinct_feature_rows=5),
            ]
        )["exceeds_exact"]
        is False
    )


def test_served_and_interp_frames_are_tolerance_flat_so_the_headline_needs_no_tolerance():
    """Both shipped bigram frames give ONE number across 12 orders of magnitude of tolerance."""
    from keybo.features import bigram_features_from_positions, interp_features_from_positions

    for feat, expected in (
        (lambda g, c: bigram_features_from_positions(g, c, wpm=90.0), 765),
        (lambda g, c: interp_features_from_positions(g, c, wpm=90.0), 378),
    ):
        rs = tolerance_sweep(feat, ROW_STAGGERED_30, order=2)
        v = sweep_verdict(rs)
        assert v["flat"] is True, v["counts"]
        assert {r.distinct_feature_rows for r in rs} == {expected}
        # A FLAT sweep must report NO rises: every step is equal, and an equal step is not a rise.
        # (Mutation M21b: without this, ``>=`` in the rise detector passes the whole suite.)
        assert v["rises"] == []
        assert v["exceeds_exact"] is False


def test_grouping_refuses_non_finite_rows_rather_than_reporting_them_as_resolved():
    """``nan != nan`` would report each NaN cell as its own group, i.e. as perfectly RESOLVED."""
    with pytest.raises(ValueError, match="non-finite"):
        group_cells(np.array([[1.0], [np.nan]]))
    with pytest.raises(ValueError, match="non-finite"):
        frame_collapse(lambda _g, _c: np.array([np.nan]), TWO_SLOT, order=1, include_space=False)


def test_negative_weights_are_refused_because_errors_would_cancel():
    with pytest.raises(ValueError, match="must be >= 0"):
        frame_collapse(
            _constant,
            TWO_SLOT,
            order=1,
            include_space=False,
            target=np.array([1.0, 2.0]),
            weights=np.array([1.0, -1.0]),
        )


def test_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="tol must be >= 0"):
        group_cells(np.array([[1.0]]), tol=-1e-9)


# --- weights ------------------------------------------------------------------------------------


def test_weights_move_the_mass_share_away_from_the_cell_share():
    """A frame where cell share and mass share provably differ, so the weighting is load-bearing."""
    # 3 cells: two collapse (indices 0,1 share a row), one is alone.
    geom = Geometry(slots=((-1, 2), (1, 2), (2, 2)))
    feat = lambda _g, c: np.array([0.0 if c[0][0] < 2 else 1.0])  # noqa: E731
    t = np.array([1.0, 2.0, 3.0])
    r_unw = frame_collapse(feat, geom, order=1, include_space=False, target=t)
    assert r_unw.collapsed_cells == 2
    assert r_unw.mass_share_collapsed == pytest.approx(2 / 3)
    assert r_unw.weighted is False

    # Put almost all mass on the SINGLETON: the collapsed mass share must fall well below 2/3.
    r_w = frame_collapse(
        feat, geom, order=1, include_space=False, target=t, weights=np.array([1.0, 1.0, 98.0])
    )
    assert r_w.collapsed_cells == 2  # structure is weight-INDEPENDENT
    assert r_w.mass_share_collapsed == pytest.approx(0.02)
    assert r_w.weighted is True
    # and the floor shrinks with the collapsed group's weight, while the UNWEIGHTED floor does not.
    assert r_w.floor_wmae < r_unw.floor_wmae
    assert r_w.floor_umae == pytest.approx(r_unw.floor_wmae)


def test_zero_weight_removes_a_cell_from_the_floor_entirely():
    """A cell weighted 0 contributes nothing: the floor equals the remaining cells' own floor."""
    r = frame_collapse(
        _constant,
        Geometry(slots=((-1, 2), (1, 2), (2, 2))),
        order=1,
        include_space=False,
        target=np.array([0.0, 1.0, 1000.0]),
        weights=np.array([1.0, 1.0, 0.0]),
    )
    assert r.floor_wmae == pytest.approx(0.5, abs=1e-12)


def test_mismatched_target_and_weight_lengths_are_refused():
    with pytest.raises(ValueError, match="target has"):
        frame_collapse(_constant, TWO_SLOT, order=1, include_space=False, target=np.zeros(5))
    with pytest.raises(ValueError, match="weights has"):
        frame_collapse(
            _constant,
            TWO_SLOT,
            order=1,
            include_space=False,
            target=np.zeros(2),
            weights=np.zeros(7),
        )


# --- the cell space, and the 765-vs-775 reconciliation -----------------------------------------


def test_cell_space_identity_is_what_separates_765_from_775():
    """The measured resolution of INTERPFRAME-1's 10-row discrepancy, pinned as a test.

    Both cell spaces have 31 positions and 961 cells, so the cell COUNT cannot tell them apart —
    which is exactly how the two runs disagreed while both reporting "961 cells on ROW_STAGGERED_31".
    Rounding was blamed; rounding provably cannot do it (a coarsening cannot RAISE a count) and the
    tolerance-flatness test above shows it does not.
    """
    from keybo.features import bigram_features_from_positions, interp_features_from_positions

    served = lambda g, c: bigram_features_from_positions(g, c, wpm=90.0)  # noqa: E731
    interp = lambda g, c: interp_features_from_positions(g, c, wpm=90.0)  # noqa: E731

    # K30 slots + space: the surface's own cell space, and the published numbers.
    with_space = frame_collapse(served, ROW_STAGGERED_30, order=2, include_space=True)
    assert (with_space.n_positions, with_space.n_cells) == (31, 961)
    assert with_space.includes_space is True
    assert with_space.distinct_feature_rows == 765
    assert frame_collapse(interp, ROW_STAGGERED_30, order=2).distinct_feature_rows == 378

    # K31 slots, NO space: also 31 positions and 961 cells, DIFFERENT answers.
    no_space = frame_collapse(served, ROW_STAGGERED_31, order=2, include_space=False)
    assert (no_space.n_positions, no_space.n_cells) == (31, 961)
    assert no_space.includes_space is False
    assert no_space.distinct_feature_rows == 775
    assert (
        frame_collapse(interp, ROW_STAGGERED_31, order=2, include_space=False).distinct_feature_rows
        == 422
    )


def test_cell_positions_includes_space_only_when_asked():
    assert len(cell_positions(ROW_STAGGERED_30)) == 31
    assert cell_positions(ROW_STAGGERED_30)[-1] == ROW_STAGGERED_30.space_position
    assert len(cell_positions(ROW_STAGGERED_30, include_space=False)) == 30
    assert len(cell_positions(ROW_STAGGERED_31, include_space=False)) == 31


def test_feature_matrix_row_order_matches_a_ravelled_surface_table():
    """Cell ``i`` must be the ``i``-th entry of a C-ordered ``(P, P)`` table, or every floor pairs a
    cell with ANOTHER cell's target — a silent, plausible-looking wrong answer."""
    pos = cell_positions(ROW_STAGGERED_30)
    n = len(pos)
    X = feature_matrix(_injective, ROW_STAGGERED_30, order=2)
    assert X.shape[0] == n * n
    grid = np.arange(n * n).reshape(n, n)
    for idx in (0, 1, n - 1, n, n + 1, 7 * n + 3, n * n - 1):
        i, j = divmod(idx, n)
        assert grid[i, j] == idx
        expected = np.array([*pos[i], *pos[j]], dtype=np.float64)
        np.testing.assert_array_equal(X[idx], expected)


def test_dict_returning_featurizers_are_accepted_in_their_own_key_order():
    """``keybo.features`` also exposes ``*_row_from_positions`` dict builders; both shapes work."""
    from keybo.features import interp_features_from_positions, interp_row_from_positions

    as_dict = frame_collapse(
        lambda g, c: interp_row_from_positions(g, c[0], c[1]), ROW_STAGGERED_30, order=2
    )
    as_array = frame_collapse(
        lambda g, c: interp_features_from_positions(g, c, wpm=90.0), ROW_STAGGERED_30, order=2
    )
    assert as_dict.n_columns == as_array.n_columns == 10
    assert as_dict.distinct_feature_rows == as_array.distinct_feature_rows == 378


def test_order_one_and_order_three_cell_spaces_have_the_right_size():
    assert frame_collapse(_injective, ROW_STAGGERED_30, order=1).n_cells == 31
    assert frame_collapse(_injective, ROW_STAGGERED_30, order=3).n_cells == 31**3
    with pytest.raises(ValueError, match="order must be >= 1"):
        frame_collapse(_injective, ROW_STAGGERED_30, order=0)


def test_trigram_frame_runs_at_order_three_over_its_own_cell_space():
    """INVARIANT 2: the 46-column trigram frame, on 29791 cells."""
    from keybo.features import trigram_features_from_positions

    r = frame_collapse(
        lambda g, c: trigram_features_from_positions(g, c, wpm=90.0), ROW_STAGGERED_30, order=3
    )
    assert r.n_cells == 31**3 == 29791
    assert r.n_columns == 46
    assert r.order == 3
    assert r.distinct_feature_rows == 28006
    assert r.collapsed_cells == 3570
    assert r.largest_group == 2
    # Better resolved than the SERVED BIGRAM frame (0.796) despite a 31x larger cell space.
    assert r.resolution == pytest.approx(0.9400825752744117)
    assert (
        r.resolution
        > frame_collapse(
            lambda g, c: __import__(
                "keybo.features", fromlist=["bigram_features_from_positions"]
            ).bigram_features_from_positions(g, c, wpm=90.0),
            ROW_STAGGERED_30,
            order=2,
        ).resolution
    )


def test_the_trigram_direction_channel_adds_columns_but_ZERO_resolution():
    """52 and 69 columns give the SAME 28006 rows as 46 — the ``interp-wpm`` lesson, at order 3.

    This is the diagnostic's cheapest possible use: it says "do not expect accuracy from these extra
    columns" without fitting anything.
    """
    from keybo.features import trigram_features_from_positions

    def _diagnose(**kw):
        # A real function, not a lambda closing over a loop variable: ruff B023 flags that shape, and
        # it is correct-by-luck here only because generator unpacking happens to evaluate eagerly
        # enough. If it ever bound late, all three calls would featurize the SAME frame and the
        # 46/52/69-column assertion below would pass for entirely the wrong reason.
        return frame_collapse(
            lambda g, c: trigram_features_from_positions(g, c, wpm=90.0, **kw),
            ROW_STAGGERED_30,
            order=3,
        )

    base = _diagnose()
    direction = _diagnose(direction=True)
    sink = _diagnose(kitchensink=True)
    assert (base.n_columns, direction.n_columns, sink.n_columns) == (46, 52, 69)
    assert base.distinct_feature_rows == direction.distinct_feature_rows == 28006
    assert sink.distinct_feature_rows == 28006


# --- the self-generated-target tautology -------------------------------------------------------


def test_self_generated_target_flag_fires_on_a_frame_scored_against_its_own_output():
    """A target that IS a function of the frame's rows: the flag must fire, and the floor must be 0.

    Built synthetically rather than from the shipped surface so the test is fast and the mechanism is
    unmistakable: ``target = 10 * row`` is exactly the "identical rows -> identical target" relation
    that makes the served frame's 0.0000 an identity.
    """
    geom = Geometry(slots=((-2, 2), (-1, 2), (1, 2), (2, 2)))
    feat = lambda _g, c: np.array([float(abs(c[0][0]))])  # noqa: E731  cells 0,3 and 1,2 collapse
    rows = np.array(
        [feature_matrix(feat, geom, order=1, include_space=False)[i, 0] for i in range(4)]
    )
    r = frame_collapse(feat, geom, order=1, include_space=False, target=10.0 * rows)
    assert r.collapsed_cells == 4
    assert r.n_collapse_groups == 2
    assert r.floor_wmae == 0.0
    assert r.max_group_target_spread == 0.0
    assert r.groups_with_target_spread == 0
    assert r.target_is_self_generated is True


def test_self_generated_target_flag_does_NOT_fire_on_an_independent_target():
    """The flag must be able to stay silent, or it is a constant and tests nothing."""
    geom = Geometry(slots=((-2, 2), (-1, 2), (1, 2), (2, 2)))
    feat = lambda _g, c: np.array([float(abs(c[0][0]))])  # noqa: E731
    r = frame_collapse(
        feat, geom, order=1, include_space=False, target=np.array([1.0, 2.0, 30.0, 40.0])
    )
    assert r.collapsed_cells == 4
    assert r.groups_with_target_spread == 2
    assert r.max_group_target_spread == pytest.approx(39.0)
    assert r.target_is_self_generated is False
    assert r.floor_wmae > 0


def test_flag_is_silent_without_a_target_and_on_a_fully_resolved_frame():
    """Two more ways the flag must stay False, both of which a naive check gets wrong."""
    assert (
        frame_collapse(_constant, TWO_SLOT, order=1, include_space=False).target_is_self_generated
        is False
    )
    assert (
        frame_collapse(
            _injective, TWO_SLOT, order=1, include_space=False, target=np.array([1.0, 2.0])
        ).target_is_self_generated
        is False
    )


# --- the report -----------------------------------------------------------------------------


def test_report_names_the_cell_space_and_warns_on_a_self_generated_target():
    geom = Geometry(slots=((-2, 2), (-1, 2), (1, 2), (2, 2)))
    feat = lambda _g, c: np.array([float(abs(c[0][0]))])  # noqa: E731
    rows = np.array([1.0, 2.0, 2.0, 1.0])
    text = format_report(
        {
            "selfgen": frame_collapse(feat, geom, order=1, include_space=False, target=10.0 * rows),
            "honest": frame_collapse(
                feat, geom, order=1, include_space=False, target=np.array([1.0, 2.0, 30.0, 40.0])
            ),
        },
        target_name="a synthetic target",
    )
    assert "4 cells" in text
    assert "a synthetic target" in text
    assert "selfgen" in text and "honest" in text
    # the warning names the offending frame and only it
    assert "⚠ selfgen" in text
    assert "⚠ honest" not in text
    assert "IDENTITY, not a measurement" in text
    # and the necessary-condition caveat is always present
    assert "NECESSARY condition only" in text


def test_report_omits_floor_columns_when_no_target_was_supplied():
    """Assert on the COLUMN HEADERS, not on the substring "FLOOR".

    ⚠ Written the naive way first (``assert "FLOOR" not in text``) and it FAILED — the word appears in
    the always-printed explanatory prose. Had the prose not contained it, that assertion would have
    passed while testing nothing about the columns, which is this campaign's own three-vacuous-tests
    failure mode. Anchored on the header cells instead.
    """
    header = format_report(
        {"f": frame_collapse(_constant, TWO_SLOT, order=1, include_space=False)}
    ).splitlines()[1]
    assert "distinct" in header
    assert "FLOOR wmae" not in header
    assert "at grp mean" not in header
    assert "FLOOR wrmse" not in header

    # and the same header DOES carry them once a target is supplied — so the assertion above is not
    # passing for the trivial reason that the header never contains them.
    with_floor = format_report(
        {
            "f": frame_collapse(
                _constant, TWO_SLOT, order=1, include_space=False, target=np.array([1.0, 2.0])
            )
        }
    ).splitlines()[1]
    assert "FLOOR wmae" in with_floor
    assert "at grp mean" in with_floor
    assert "FLOOR wrmse" in with_floor


def test_as_dict_round_trips_every_reported_field():
    d = frame_collapse(
        _constant,
        TWO_SLOT,
        order=1,
        include_space=False,
        target=np.array([1.0, 5.0]),
        weights=np.array([1.0, 3.0]),
    ).as_dict()
    assert d["n_cells"] == 2
    assert d["distinct_feature_rows"] == 1
    assert d["weighted"] is True
    assert d["includes_space"] is False
    assert d["tol"] == 0.0
    for key in (
        "floor_wmae",
        "floor_wmae_at_group_mean",
        "floor_wrmse",
        "floor_umae",
        "target_is_self_generated",
        "max_group_target_spread",
        "resolution",
        "mass_share_collapsed",
        "n_positions",
        "order",
        "largest_group",
    ):
        assert key in d, key
