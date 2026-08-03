"""The finite-operand guard: a comparison over uncomputed operands must not return a verdict.

Generalizes three defects found in this repo in two days — ``analyze --help`` exiting 0 with
no output, a liveness probe reporting "done" for a process that never existed, and an A/B
printing ``ARGMAX MOVES: False`` over two all-``-inf`` arms. Each returned the answer that
means "no problem" because its operands were never computed.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

from keybo.training.validate import calibration_slope
from keybo.verdicts import (
    CALIBRATION_SLOPE_RECOMMENDED_BAND,
    CalibrationCompressed,
    EmptyComparison,
    MarginTooSmall,
    all_distinct,
    argmax_finite,
    calibration_report,
    compare_finite,
    require_calibration,
    require_finite,
    require_margin,
    reweighting_margin_bound,
)


def test_require_finite_passes_real_numbers_through_as_floats() -> None:
    assert require_finite([1, 2.5, -3], "x") == [1.0, 2.5, -3.0]


@pytest.mark.parametrize("bad", [float("-inf"), float("inf"), float("nan")])
def test_require_finite_rejects_every_non_finite_kind(bad: float) -> None:
    with pytest.raises(EmptyComparison) as exc:
        require_finite([1.0, bad], "candidate scores")
    assert "candidate scores" in str(exc.value)
    assert "1 of 2" in str(exc.value)


def test_require_finite_rejects_the_empty_case_separately() -> None:
    """Zero operands is its own failure — 'no values at all', not 'no difference'."""
    with pytest.raises(EmptyComparison) as exc:
        require_finite([], "gauge spread")
    assert "no values at all" in str(exc.value)


def test_the_message_tells_the_reader_not_to_trust_the_verdict() -> None:
    with pytest.raises(EmptyComparison) as exc:
        require_finite([float("-inf")], "arm scores")
    assert "do not" in str(exc.value).lower()


def test_compare_finite_guards_BOTH_sides() -> None:
    a, b = compare_finite([1.0, 2.0], [3.0, 4.0], "arm A vs arm B")
    assert (a, b) == ([1.0, 2.0], [3.0, 4.0])
    for left, right, side in (
        ([float("-inf")], [1.0], "side A"),
        ([1.0], [float("nan")], "side B"),
    ):
        with pytest.raises(EmptyComparison) as exc:
            compare_finite(left, right, "arm A vs arm B")
        assert side in str(exc.value)


def test_compare_finite_catches_the_exact_degenerate_AB_that_bit_this_campaign() -> None:
    """Two arms that 'agree' only because neither was evaluated."""
    all_inf = [float("-inf")] * 4
    with pytest.raises(EmptyComparison):
        compare_finite(all_inf, all_inf, "tune argmax A/B")


def test_compare_finite_requires_equal_lengths() -> None:
    with pytest.raises(EmptyComparison) as exc:
        compare_finite([1.0, 2.0], [1.0], "paired")
    assert "equal lengths" in str(exc.value)


def test_argmax_finite_refuses_instead_of_returning_index_zero() -> None:
    """``np.argmax`` over all -inf returns 0, which reads as a selected winner."""
    assert argmax_finite([0.1, 0.9, 0.4], "scores") == 1
    with pytest.raises(EmptyComparison):
        argmax_finite([float("-inf")] * 3, "scores")
    # a SINGLE non-finite entry is still a refusal: it silently loses every comparison
    with pytest.raises(EmptyComparison):
        argmax_finite([0.5, float("-inf")], "scores")


def test_all_distinct_is_the_cheap_hidden_invariant_check() -> None:
    """How `alt`, `imbalance`, `sfr` and index_imbalance_pct were each caught."""
    assert all_distinct([1.0, 2.0, 3.0], "gauge over perturbations")
    assert not all_distinct([2.0777, 2.0777, 2.0777], "imbalance over four layouts")
    # tolerance is opt-in, because exact ties are the thing being hunted
    assert all_distinct([1.0, 1.0 + 1e-9], "near-tie", tol=0.0)
    assert not all_distinct([1.0, 1.0 + 1e-9], "near-tie", tol=1e-6)


def test_all_distinct_guards_its_own_operands_too() -> None:
    with pytest.raises(EmptyComparison):
        all_distinct([1.0, float("nan")], "gauge")


def test_a_real_no_difference_result_is_NOT_an_error() -> None:
    """The guard must distinguish 'measured, equal' from 'never measured'."""
    a, b = compare_finite([1.0, 2.0], [1.0, 2.0], "before vs after")
    assert a == b, "identical finite operands are a legitimate null result"
    assert math.isclose(sum(a), sum(b))


# --- the minimum-margin rule -------------------------------------------------------------


def test_the_bound_is_the_weights_relative_half_range_in_closed_form() -> None:
    """Derived, not sampled — so the shipped constant has a justification, not a provenance."""
    # Spearman-Brown weight is (1+c)/2 per fold; this ledger's ceilings span [0.709, 0.815].
    ledger = reweighting_margin_bound([(1 + c) / 2 for c in (0.709, 0.815)])
    assert ledger == pytest.approx(0.0301, abs=1e-4)
    # a wider assumed ceiling range gives a strictly larger bound
    wide = reweighting_margin_bound([(1 + c) / 2 for c in (0.50, 0.95)])
    assert wide == pytest.approx(0.1304, abs=1e-4)
    assert wide > ledger
    # identical weights cannot move anything
    assert reweighting_margin_bound([0.9, 0.9]) == 0.0


def test_the_bound_rejects_nonpositive_weights() -> None:
    with pytest.raises(EmptyComparison):
        reweighting_margin_bound([0.0, 0.9])
    with pytest.raises(EmptyComparison):
        reweighting_margin_bound([0.5, float("nan")])


def test_require_margin_passes_a_clear_win_and_refuses_a_narrow_one() -> None:
    assert require_margin([0.90, 0.50, 0.10], "sel", min_margin=0.03) == 0
    with pytest.raises(MarginTooSmall) as exc:
        require_margin([0.900, 0.899], "lolo selection", min_margin=0.03)
    msg = str(exc.value)
    assert "lolo selection" in msg
    assert "do not read it as a winner" in msg


def test_require_margin_is_RELATIVE_by_default_and_absolute_on_request() -> None:
    # gap 0.02 on a winner of 1.0 -> relative 2%, below a 3% bar
    with pytest.raises(MarginTooSmall):
        require_margin([1.00, 0.98], "sel", min_margin=0.03)
    # the same gap passes an ABSOLUTE 0.01 bar
    assert require_margin([1.00, 0.98], "sel", min_margin=0.01, relative=False) == 0
    # and relative scales with magnitude: the same 0.02 gap on a winner of 0.10 is 20%
    assert require_margin([0.10, 0.08], "sel", min_margin=0.03) == 0


def test_require_margin_inherits_the_finite_guard() -> None:
    with pytest.raises(EmptyComparison):
        require_margin([float("-inf")] * 2, "sel", min_margin=0.03)


def test_a_single_candidate_has_no_margin_to_check() -> None:
    assert require_margin([0.5], "sel", min_margin=0.99) == 0


def test_min_margin_zero_disables_the_gate_and_negatives_are_rejected() -> None:
    assert require_margin([0.9000, 0.8999], "sel", min_margin=0.0) == 0
    with pytest.raises(ValueError):
        require_margin([0.9, 0.5], "sel", min_margin=-0.1)


def test_the_documented_shipped_margin_CLEARS_the_shipped_threshold() -> None:
    """Why this gate guards future selections rather than retracting a past one."""
    from keybo.training.tune import LOLO_MIN_MARGIN

    # tune_lolo's docstring: depth-5 lost ~0.06 rho/ceiling to depth-3, on scores near 0.93
    assert require_margin([0.93, 0.87], "shipped", min_margin=LOLO_MIN_MARGIN) == 0
    assert LOLO_MIN_MARGIN < 0.06 / 0.93


# --- the calibration gate (backlog E4 / roadmap 4.2) -------------------------------------------
# Every test below asserts on the SPECIFIC failure it means to provoke -- the exception type AND a
# substring of its message -- because a test going red proves nothing until you know WHICH failure
# you got. CALIB-1 nearly shipped a vacuously-green guard whose mock patched the wrong module.


def test_calibration_report_is_a_report_and_never_raises() -> None:
    r = calibration_report({"80": 2.5, "100": 0.1}, "wildly miscalibrated")
    assert r["gated"] is True
    assert r["passed"] is False
    assert r["out_of_band"] == ["100", "80"]


def test_an_absent_slope_is_UNGATED_not_passing() -> None:
    """The TAUGATE-1 rule: a missing verdict must not read like a passing one."""
    r = calibration_report({"80": float("nan")}, "nothing measurable")
    assert r["gated"] is False
    assert r["passed"] is None, "an unmeasured slice must be None, never True"
    assert r["n_slices_missing"] == 1


def test_require_calibration_refuses_the_unmeasured_case_rather_than_passing_it() -> None:
    with pytest.raises(CalibrationCompressed) as e:
        require_calibration({}, "empty", band=(0.9, 1.1))
    assert "'Not measured' is not 'is calibrated'" in str(e.value)


def test_require_calibration_passes_a_calibrated_surface() -> None:
    r = require_calibration({"80": 1.02, "100": 0.97}, "calibrated", band=(0.9, 1.1))
    assert r["passed"] is True
    assert r["worst_abs_deviation_from_1"] == pytest.approx(0.03)


def test_require_calibration_refuses_a_COMPRESSED_surface_and_says_which_slice() -> None:
    with pytest.raises(CalibrationCompressed) as e:
        require_calibration({"80": 1.00, "120": 1.46}, "compressed at speed", band=(0.9, 1.1))
    msg = str(e.value)
    assert "1 of 2 slices" in msg
    assert "120" in msg and "1.46" in msg
    assert "COMPRESS" in msg, "the message must name the DIRECTION, not just the deviation"
    assert "rho and tau are invariant" in msg


def test_the_band_is_REQUIRED_so_no_agent_can_install_a_retroactive_threshold() -> None:
    """The band decides which PAST results stand, so it must arrive from the call site."""
    with pytest.raises(TypeError):
        require_calibration({"80": 1.0}, "no band given")  # type: ignore[call-arg]


def test_support_travels_with_the_verdict_but_never_changes_it() -> None:
    slopes = {"80": 1.46, "120": 1.46}
    thin = {"80": {"n_cells": 900, "n_participants": 120},
            "120": {"n_cells": 12, "n_participants": 3}}
    with_support = calibration_report(slopes, "x", support=thin)
    without = calibration_report(slopes, "x")
    assert with_support["passed"] == without["passed"]
    assert with_support["support"]["120"]["n_participants"] == 3
    assert without["support"] is None, "None distinguishes 'not supplied' from 'supplied and thin'"


def test_low_r2_does_not_license_a_slope_away_from_1() -> None:
    """The identity behind the constant: an MSE-optimal predictor has slope 1 at ANY r^2.

    Built from data rather than asserted: fit the least-squares predictor of `obs` on a noisy
    feature, and its slope(obs~pred) is 1.0 however weak the correlation is. This is why a low
    r^2 is a SEPARATE defect from a bad slope, and why the target is 1.0 and not "whatever r^2
    allows".
    """
    rng = np.random.default_rng(20260803)
    obs = rng.normal(0, 40, 4000)
    for noise in (10.0, 60.0, 200.0):          # r^2 from ~0.94 down to ~0.04
        feature = obs + rng.normal(0, noise, obs.size)
        b, a = np.polyfit(feature, obs, 1)     # the MSE-optimal linear predictor
        pred = a + b * feature
        r2 = np.corrcoef(pred, obs)[0, 1] ** 2
        assert calibration_slope(pred, obs) == pytest.approx(1.0, abs=0.02), (
            f"an MSE-optimal predictor must be calibrated even at r2={r2:.3f}"
        )
        assert r2 < 0.99


def test_the_shipped_surfaces_measured_compression_would_be_REFUSED_by_the_recommended_band() -> None:
    """CALIB-1's measurement, pinned: the pair-level expansion needed is 1.4618.

    Kept as a test so the recommended band's retroactive force is visible in the suite rather than
    only in the ledger -- this is the number that makes the band a human decision.
    """
    with pytest.raises(CalibrationCompressed):
        require_calibration({"pairs-wpm80": 1.4618}, "shipped k31 bigram surface",
                            band=CALIBRATION_SLOPE_RECOMMENDED_BAND)
