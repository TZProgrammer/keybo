"""The finite-operand guard: a comparison over uncomputed operands must not return a verdict.

Generalizes three defects found in this repo in two days — ``analyze --help`` exiting 0 with
no output, a liveness probe reporting "done" for a process that never existed, and an A/B
printing ``ARGMAX MOVES: False`` over two all-``-inf`` arms. Each returned the answer that
means "no problem" because its operands were never computed.
"""

from __future__ import annotations

import math

import pytest

from keybo.verdicts import (
    EmptyComparison,
    MarginTooSmall,
    all_distinct,
    argmax_finite,
    compare_finite,
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
