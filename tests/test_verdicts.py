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
    all_distinct,
    argmax_finite,
    compare_finite,
    require_finite,
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
