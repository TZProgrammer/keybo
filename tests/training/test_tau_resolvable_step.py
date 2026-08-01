"""``tau_resolvable_step`` is the tau gate's tolerance, and its formula was ALREADY WRONG once.

The gate in ``apply_tau_gate`` keeps a candidate iff ``t >= best_tau - (step + 1e-9)``, so this
one number decides whether a ranking difference is a measurement or a rounding artifact. It was
first written as ``2 / (n * (n - 1))`` — the normalized-concordance step, which understates the
real tau-b spacing by **half** — and corrected to ``4 / (n * (n - 1))`` in ledger ``cb907aa``.
Kendall tau is ``(concordant - discordant) / total_pairs``, so moving ONE pair from concordant to
discordant changes the numerator by two pair-units, not one.

That correction shipped with a test of the *enumeration* at n=4
(``test_tune_grouped_cv.py::test_kendall_tau_over_four_layouts_takes_only_seven_values``) but with
**zero direct tests of the function itself** — verified by ``grep -rl tau_resolvable_step tests/``
returning nothing before this file. A formula that has been wrong once and is unpinned is the
cheapest place in the selection path for the same halving to come back:

* too SMALL (the old ``2/(n(n-1))``) makes the gate a **tripwire** again — a one-inversion edge,
  the finest distinction four layouts can draw, sets the best-rho candidates to ``-inf``;
* too LARGE makes it a **no-op** that passes a genuinely collapsed ranking.

Both failures are silent: the leaderboard looks the same either way. So the tests below pin the
value from an independent direction — exhaustive enumeration of every permutation's tau, in exact
rational arithmetic — rather than restating the closed form and agreeing with itself.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest
from scipy.stats import kendalltau

from keybo.training.tune import apply_tau_gate, tau_resolvable_step

#: The group count that motivated the whole gate: four layouts in the training frame.
_SHIPPED_N_GROUPS = 4


def _achievable_taus_exact(n: int) -> list[Fraction]:
    """Every Kendall tau reachable by ranking ``n`` items, in EXACT rational arithmetic.

    Computed from the definition — ``(concordant - discordant) / total_pairs`` over all ``n!``
    permutations — deliberately NOT via the closed form under test, and with ``Fraction`` rather
    than floats so a claim about the *spacing* is not a claim about float noise (a float version
    of this check reports spurious unequal gaps at n=3, 4, 6 and 7).
    """
    base = list(range(n))
    total_pairs = Fraction(n * (n - 1), 2)
    taus = set()
    for perm in itertools.permutations(base):
        concordant = discordant = 0
        for i, j in itertools.combinations(range(n), 2):
            if (perm[i] - perm[j]) * (base[i] - base[j]) > 0:
                concordant += 1
            else:
                discordant += 1
        taus.add(Fraction(concordant - discordant) / total_pairs)
    return sorted(taus)


# --- the value itself, pinned against enumeration ----------------------------------------


def test_the_step_at_the_shipped_four_layouts_is_one_third_not_one_sixth() -> None:
    """The exact regression: 4/(4*3) = 0.3333, and the old 2/(4*3) = 0.1667 must NOT pass.

    Named for the wrong formula on purpose. If someone reinstates ``2 / (n * (n - 1))``, this is
    the assertion that names it, and the second half of the test says out loud which value is the
    bug rather than leaving a bare number for the next reader to re-derive.
    """
    step = tau_resolvable_step(_SHIPPED_N_GROUPS)
    assert step == pytest.approx(1 / 3), f"expected 4/(4*3) = 0.3333, got {step}"
    wrong = 2 / (_SHIPPED_N_GROUPS * (_SHIPPED_N_GROUPS - 1))
    assert step != pytest.approx(wrong), (
        f"{step} equals the normalized-concordance step {wrong} — that is the ledger's original "
        f"defect, which halves the tolerance and turns the gate back into a tripwire"
    )
    assert step == pytest.approx(2 * wrong), "the real tau-b spacing is exactly twice that step"


def test_the_step_equals_the_gap_between_the_two_best_achievable_taus_at_four_groups() -> None:
    """1.0 vs 0.6667 IS one inversion — the gate's tolerance must reach exactly that far.

    This is the operative case: ``apply_tau_gate`` compares against ``best_tau``, so the only gap
    that matters for the tripwire is the one immediately below the maximum.
    """
    taus = _achievable_taus_exact(_SHIPPED_N_GROUPS)
    assert taus[-1] == 1
    assert float(taus[-1] - taus[-2]) == pytest.approx(tau_resolvable_step(_SHIPPED_N_GROUPS))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_the_step_is_the_exact_lattice_spacing_of_every_achievable_tau(n: int) -> None:
    """Property, not a table: the achievable taus form a UNIFORM lattice of width ``step``.

    Stronger than "step == the minimum positive gap", and true: in exact arithmetic every
    consecutive gap is the same single value, so one wrong step cannot hide behind an
    irregularly-spaced set. n=3..6 brackets the shipped 4 on both sides; 6! = 720 permutations
    keeps the enumeration cheap.
    """
    taus = _achievable_taus_exact(n)
    gaps = {b - a for a, b in zip(taus[:-1], taus[1:], strict=True)}
    assert len(gaps) == 1, f"n={n}: achievable taus are not uniformly spaced: {sorted(gaps)}"
    assert gaps.pop() == Fraction(4, n * (n - 1)), f"n={n}: lattice spacing != 4/(n(n-1))"
    assert float(tau_resolvable_step(n)) == pytest.approx(4 / (n * (n - 1)))


def test_the_enumerated_count_of_achievable_taus_matches_the_lattice_the_step_implies() -> None:
    """A cross-check on the lattice claim: the values are ``2/step + 1`` points from -1 to +1.

    At n=4 that is seven — the number quoted throughout the ledger — and it follows from the step
    rather than being asserted alongside it.
    """
    for n in (3, 4, 5, 6):
        step = tau_resolvable_step(n)
        assert len(_achievable_taus_exact(n)) == round(2 / step) + 1


def test_scipys_own_kendalltau_agrees_with_the_exact_enumeration() -> None:
    """Guards the enumeration helper itself, so these tests pin tau and not my arithmetic.

    Without this, a bug in ``_achievable_taus_exact`` would make every property above vacuous
    while still passing.
    """
    base = list(range(_SHIPPED_N_GROUPS))
    from_scipy = sorted(
        {round(float(kendalltau(base, list(p)).statistic), 6) for p in itertools.permutations(base)}
    )
    from_exact = [round(float(t), 6) for t in _achievable_taus_exact(_SHIPPED_N_GROUPS)]
    assert from_scipy == from_exact
    assert from_exact == [-1.0, -0.666667, -0.333333, 0.0, 0.333333, 0.666667, 1.0]


# --- the fallback, and WHY it is 0.0 rather than something permissive --------------------


@pytest.mark.parametrize("n_groups", [None, 0, 1])
def test_an_unknown_or_unrankable_group_count_falls_back_to_ZERO_tolerance(n_groups) -> None:
    """0.0 restores the historical exact-max gate; any positive guess would WIDEN it silently.

    ``n_groups=None`` is the default, i.e. every caller that does not thread the group count
    through. The safe direction for a *guard* under missing information is to keep gating, not to
    start tolerating gaps whose size nothing measured — and with fewer than two items there is no
    pair to invert, so no step exists to return.
    """
    assert tau_resolvable_step(n_groups) == 0.0


def test_the_default_call_of_apply_tau_gate_therefore_still_gates_a_one_step_edge() -> None:
    """The consequence of the 0.0 fallback, at the gate: no ``n_groups`` means exact-max.

    Pinned because the fallback is only defensible if it makes the gate STRICTER. If someone
    "helpfully" returns a nonzero default step, this test fails — which is the intent: widening
    the tolerance for a caller who never said how many groups there are is a silent change to
    every selection that omits the argument.
    """
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, 0.666667)]
    gated, saturated = apply_tau_gate(scored)  # no n_groups — the historical call shape
    assert saturated is False, "two distinct taus are not saturated"
    assert [s for _p, s in gated] == [0.90, float("-inf")], (
        "with no group count the gate must fall back to exact-max, not tolerate one inversion"
    )


def test_a_LARGER_group_count_tightens_the_step_monotonically() -> None:
    """More layouts resolve finer rankings, so the tolerance must shrink, never grow.

    A gate whose tolerance grew with the group count would get *looser* exactly as the data got
    more informative. The direction is the property worth pinning; the values are covered above.
    """
    steps = [tau_resolvable_step(n) for n in range(2, 12)]
    assert steps == sorted(steps, reverse=True)
    assert all(s > 0.0 for s in steps)
    assert all(a > b for a, b in zip(steps[:-1], steps[1:], strict=True))


def test_the_step_never_exceeds_the_full_tau_range_above_two_groups() -> None:
    """Sanity bound: a step wider than 2.0 would make the gate unable to reject anything.

    At n=2 the step IS 2.0 — the whole range — because two groups admit only tau in {-1, +1}: a
    single inversion is a total reversal. That is the degenerate case, and it is honest: with two
    layouts the ranking carries one bit, so the gate legitimately cannot discriminate. Above two
    groups the step must be strictly inside the range.
    """
    assert tau_resolvable_step(2) == pytest.approx(2.0)
    for n in range(3, 12):
        assert 0.0 < tau_resolvable_step(n) < 2.0
