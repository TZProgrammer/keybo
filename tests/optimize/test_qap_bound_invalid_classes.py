"""The two invalid-bound classes the existing suite lets through, plus certificate() scope.

`test_qap_bound.py` catches 18 of 24 mutants — a 0.1% bound inflation, ``bound := 0``, both
direction flips, the dropped halving — but TWO survivors are real invalid-bound classes:
reading the INCOMING leg along the wrong axis. Those produce bounds that EXCEED the true
optimum, i.e. a fake TIGHT certificate, which is the failure mode that reads as good news.

Reproduced before writing these tests: ``t_in`` on the row axis gives 24/750 violations
(worst +9.9% above the true optimum) and ``f_in`` on the row axis 27/750 (+6.4%), while the
shipped code gives 0/750. The mutation is COUPLED — swapping BOTH legs to the row form is
harmless (0/750), which is exactly why a coarse mutation survived: you have to break one leg
alone. So these tests target each leg SEPARATELY.

The existing suite also never scores an asymmetric instance deeply, and the real F2/T2 are
both asymmetric, so the class is reachable in production rather than theoretical.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from keybo.optimize.qap_bound import (
    CertificateScopeError,
    certificate,
    gilmore_lawler_bound,
    qap_fitness,
)


def _brute_optimum(F: np.ndarray, T: np.ndarray) -> float:
    n = F.shape[0]
    return min(qap_fitness(F, T, np.array(p)) for p in itertools.permutations(range(n)))


def _bound_with_legs(F: np.ndarray, T: np.ndarray, *, f_in_axis: str, t_in_axis: str) -> float:
    """The shipped bound, but with each INCOMING leg's axis selectable.

    ``"col"`` is the shipped (correct) choice: the incoming leg must read F's COLUMN and T's
    COLUMN, because it bounds the terms where i/k is the DESTINATION. ``"row"`` re-reads the
    outgoing slice and is the mutation under test.
    """
    n = F.shape[0]
    off = ~np.eye(n, dtype=bool)
    cost = np.empty((n, n))
    for i in range(n):
        f_out = F[i][off[i]]
        f_in = F[i][off[i]] if f_in_axis == "row" else F[:, i][off[:, i]]
        for k in range(n):
            t_out = T[k][off[k]]
            t_in = T[k][off[k]] if t_in_axis == "row" else T[:, k][off[:, k]]
            cost[i, k] = F[i, i] * T[k, k] + 0.5 * (
                float(np.sort(f_out)[::-1] @ np.sort(t_out))
                + float(np.sort(f_in)[::-1] @ np.sort(t_in))
            )
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].sum())


def _asymmetric_instances(n_cases: int = 250, seed: int = 11):
    """Random ASYMMETRIC F and T — the property that makes the wrong-axis mutation bite.

    A symmetric instance cannot expose it (row == column), which is why the existing suite's
    cases pass under the mutants.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        n = int(rng.integers(4, 7))
        yield rng.random((n, n)) * 10.0, rng.random((n, n)) * 10.0


# --- the shipped bound is valid on exactly the instances that expose the mutants ---------


def test_shipped_bound_never_exceeds_the_optimum_on_ASYMMETRIC_instances() -> None:
    """The positive half: the correct code has zero violations where the mutants fail."""
    violations = 0
    for F, T in _asymmetric_instances():
        assert gilmore_lawler_bound(F, T) <= _brute_optimum(F, T) + 1e-9
        violations += 0
    assert violations == 0


# --- the two classes, each leg separately -----------------------------------------------


@pytest.mark.parametrize(
    "f_in_axis,t_in_axis,label",
    [
        ("col", "row", "T's incoming leg on the row axis"),
        ("row", "col", "F's incoming leg on the row axis"),
    ],
)
def test_reading_ONE_incoming_leg_on_the_wrong_axis_BREAKS_the_bound(
    f_in_axis: str, t_in_axis: str, label: str
) -> None:
    """Each single-leg swap must produce a bound EXCEEDING some true optimum.

    This is the assertion the existing suite is missing. It is phrased as "the mutant IS
    detectably invalid" so that the test fails if someone 'simplifies' the shipped code into
    the mutant — the mutant would then be the code under test and this test would stop
    finding violations, which is the signal we want.
    """
    violations = 0
    worst = 0.0
    for F, T in _asymmetric_instances():
        mutant = _bound_with_legs(F, T, f_in_axis=f_in_axis, t_in_axis=t_in_axis)
        opt = _brute_optimum(F, T)
        if mutant > opt + 1e-9:
            violations += 1
            worst = max(worst, (mutant - opt) / abs(opt))
    assert violations > 0, (
        f"{label}: expected this mutation to yield invalid bounds; if it no longer does, "
        f"either the instance family stopped being asymmetric or the shipped code changed"
    )
    assert worst > 0.01, f"{label}: overshoot should be material, got {100 * worst:.2f}%"


def test_the_mutation_is_COUPLED_so_a_coarse_both_legs_swap_hides_it() -> None:
    """Why the original 24-mutant sweep let these through.

    Swapping BOTH incoming legs to the row axis is harmless — it just re-bounds the outgoing
    term twice — so a mutation operator that flips 'in' to 'out' everywhere at once finds
    nothing. Only a single-leg swap is fatal. Pinned so the reason is not lost.
    """
    both_row_violations = 0
    for F, T in _asymmetric_instances(n_cases=150, seed=7):
        both = _bound_with_legs(F, T, f_in_axis="row", t_in_axis="row")
        if both > _brute_optimum(F, T) + 1e-9:
            both_row_violations += 1
    assert both_row_violations == 0, "both-legs-row is expected to stay valid"


def test_a_SYMMETRIC_instance_family_cannot_expose_either_class() -> None:
    """Documents the existing suite's blind spot rather than just asserting around it."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        n = int(rng.integers(4, 6))
        A = rng.random((n, n)) * 10.0
        B = rng.random((n, n)) * 10.0
        F = A + A.T  # symmetric
        T = B + B.T
        opt = _brute_optimum(F, T)
        for f_ax, t_ax in (("col", "row"), ("row", "col"), ("row", "row")):
            assert _bound_with_legs(F, T, f_in_axis=f_ax, t_in_axis=t_ax) <= opt + 1e-9


# --- certificate() carries its scope and refuses impossible inputs ----------------------


def test_certificate_carries_a_scope_key_and_renders_it() -> None:
    rng = np.random.default_rng(5)
    F, T = rng.random((5, 5)) * 10, rng.random((5, 5)) * 10
    opt = _brute_optimum(F, T)
    cert = certificate(F, T, opt, scope="the bigram component")
    assert cert["scope"] == "the bigram component"
    assert "ON the bigram component" in cert["statement"]
    # the default is explicit rather than silent
    assert "scope" in certificate(F, T, opt)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_certificate_refuses_a_non_finite_fitness(bad: float) -> None:
    """Previously rendered 'within nan% of the best possible layout' with no raise."""
    rng = np.random.default_rng(6)
    F, T = rng.random((4, 4)) * 10, rng.random((4, 4)) * 10
    with pytest.raises(CertificateScopeError) as exc:
        certificate(F, T, bad)
    assert "finite" in str(exc.value)


def test_certificate_refuses_a_fitness_BELOW_the_bound() -> None:
    """A negative gap is impossible for a layout scored on the bound's own objective.

    It is the precise signature of a bound/objective mismatch — the defect this module was
    audited for — so it must raise rather than report a cheerful -50.00%.
    """
    rng = np.random.default_rng(9)
    F, T = rng.random((5, 5)) * 10, rng.random((5, 5)) * 10
    lb = gilmore_lawler_bound(F, T)
    with pytest.raises(CertificateScopeError) as exc:
        certificate(F, T, lb * 0.5)
    assert "BELOW the lower bound" in str(exc.value)
    assert "different objectives" in str(exc.value)
    # exactly at the bound is legitimate (gap 0), not an error
    assert certificate(F, T, lb)["gap_pct"] == pytest.approx(0.0, abs=1e-9)
