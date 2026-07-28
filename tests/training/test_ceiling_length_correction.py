"""The noise ceiling is a FULL-length reliability, not a half-length one.

``split_half_ceiling`` correlates two halves of the participants, which measures the
reliability of a HALF-length instrument — but every consumer divides a FULL-sample model
rho by it. The missing Spearman-Brown step made ``rho/ceiling`` too large, and unevenly
so, because ``2r/(1+r)/r`` is decreasing in ``r``.

These tests pin the correction, the registered gate it moves, and the one place a ceiling
is not merely displayed (``tune.py`` selects on a mean of ``rho_frac_ceiling``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from keybo.training.validate import spearman_brown, split_half_ceiling

# --- the correction itself --------------------------------------------------------------


def test_spearman_brown_lengthens_and_is_idempotent_at_the_fixed_points() -> None:
    assert spearman_brown(1.0) == pytest.approx(1.0)  # perfect stays perfect
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(0.6) == pytest.approx(0.75)
    # strictly lengthening in (0, 1)
    for r in (0.05, 0.3, 0.6, 0.709, 0.815, 0.99):
        assert spearman_brown(r) > r


def test_the_inflation_factor_is_DECREASING_so_a_noisier_arm_is_flattered_more() -> None:
    """This is the whole reason a per-arm ``rho/ceiling`` comparison could invert.

    The factor is exactly ``2/(1+r)``. NOTE the audit digest that surfaced this defect
    quoted the pair 1.4434 (r=0.60) and 1.0076 (r=0.99); neither is right — 2/(1+r) gives
    1.2500 and 1.0050, and 1.4434 corresponds to r=0.3856, no arm in the register. The
    CONCLUSION (monotone decreasing, so the noisier arm is flattered more) is unaffected,
    which is exactly why the wrong constants survived: they pointed the right way.
    """
    ratio = lambda r: spearman_brown(r) / r  # noqa: E731
    for r in (0.10, 0.30, 0.60, 0.709, 0.815, 0.99):
        assert ratio(r) == pytest.approx(2.0 / (1.0 + r), rel=1e-12)
    assert ratio(0.60) == pytest.approx(1.2500, abs=1e-4)
    assert ratio(0.99) == pytest.approx(1.0050, abs=1e-4)
    factors = [ratio(r) for r in (0.10, 0.30, 0.60, 0.80, 0.99)]
    assert factors == sorted(factors, reverse=True), "must be monotone decreasing"
    # the two arms that matter: F5M (c=0.709) is flattered more than BASE (c=0.815)
    assert ratio(0.709) > ratio(0.815)


def test_domain_guard_leaves_nonpositive_and_nan_alone() -> None:
    assert spearman_brown(0.0) == 0.0
    assert spearman_brown(-0.3) == -0.3  # lengthening a negative r is not meaningful
    assert math.isnan(spearman_brown(float("nan")))


# --- the registered gate this moves -----------------------------------------------------


# The four Q-OBJ arms as registered at PREREGISTRATIONS.md:1049-1052 (2026-07-08):
# (own-ceiling c, rho/own-ceiling as shipped).
_QOBJ = {
    "BASE": (0.815, 0.994),
    "Q25": (0.803, 0.941),
    "Q20": (0.795, 0.937),
    "F5M": (0.709, 0.974),
}
_GATE = -0.02  # F5M's rho-fraction must be within this of BASE's to be an ADOPT-CANDIDATE


def _fractions(lengthen) -> dict[str, float]:
    """Re-score each arm's rho against a ceiling transformed by ``lengthen``."""
    out = {}
    for arm, (ceiling, shipped_frac) in _QOBJ.items():
        rho = shipped_frac * ceiling  # recover the raw rho that was scored
        out[arm] = rho / lengthen(ceiling)
    return out


def test_the_shipped_gate_passed_only_because_the_ceiling_was_half_length() -> None:
    as_shipped = _fractions(lambda c: c)
    delta = as_shipped["F5M"] - as_shipped["BASE"]
    assert delta == pytest.approx(-0.0200, abs=1e-4), "reproduces the registered margin"

    corrected = _fractions(spearman_brown)
    delta_corrected = corrected["F5M"] - corrected["BASE"]
    assert delta_corrected == pytest.approx(-0.0698, abs=1e-4)
    assert delta_corrected < _GATE, "F5M FAILS the -0.02 gate once the ceiling is correct"
    # and it fails by a wide margin, not another hair
    assert delta_corrected < 3 * _GATE


@pytest.mark.parametrize(
    "name,lengthen",
    [
        ("spearman-brown", spearman_brown),
        ("sqrt", math.sqrt),
        ("c**0.75", lambda c: c**0.75),
        ("c**0.5", lambda c: c**0.5),
    ],
)
def test_the_gate_FAILS_under_every_candidate_length_correction(name, lengthen) -> None:
    """The gate failure is robust to the FORM of the correction, not an artifact of one."""
    fr = _fractions(lengthen)
    assert fr["F5M"] - fr["BASE"] < _GATE, f"{name} should still fail the gate"


def test_the_ARM_ORDERING_inverts_under_spearman_brown_but_NOT_under_every_form() -> None:
    """Scope guard: the gate failure is universal, the ordering inversion is not.

    Registered as "the arm ordering inverts". It does under Spearman-Brown and sqrt — F5M
    falls BELOW both quantile arms the same entry refuted as objectives — but under
    ``c**0.75`` F5M stays second. Quote the gate failure unconditionally; quote the
    inversion only with its form named.
    """
    order = lambda fr: sorted(fr, key=lambda a: -fr[a])  # noqa: E731

    assert order(_fractions(lambda c: c)) == ["BASE", "F5M", "Q25", "Q20"]
    assert order(_fractions(spearman_brown)) == ["BASE", "Q25", "Q20", "F5M"]
    assert order(_fractions(math.sqrt)) == ["BASE", "Q25", "Q20", "F5M"]
    # ... but a gentler form does not invert it:
    assert order(_fractions(lambda c: c**0.75)) == ["BASE", "F5M", "Q25", "Q20"]


def test_the_other_F5M_gate_moves_FAVOURABLY_so_the_fix_is_not_oversold() -> None:
    """F5M's own-ceiling ratio vs BASE (>= 0.85 gate) IMPROVES from 0.870 to 0.924."""
    raw = 0.709 / 0.815
    corrected = spearman_brown(0.709) / spearman_brown(0.815)
    assert raw == pytest.approx(0.8699, abs=1e-4)
    assert corrected == pytest.approx(0.9239, abs=1e-4)
    assert corrected > raw and corrected >= 0.85


# --- the one consumer that is not display-only ------------------------------------------


def test_a_per_fold_ceiling_reweighting_CAN_move_tune_pys_argmax() -> None:
    """``tune.py`` picks argmax over candidates of ``mean_folds(rho_fold / ceiling_fold)``.

    The ceiling takes no ``train_params`` (it is a property of the data fold), so the
    correction is a candidate-INDEPENDENT positive reweighting. That does NOT make the
    argmax invariant: a mean of ratios is not scale-invariant, so re-weighting folds can
    flip which candidate wins when candidates trade folds off against each other.
    """
    rng = np.random.default_rng(0)
    n_folds, n_candidates = 5, 6
    moved = 0
    trials = 4000
    for _ in range(trials):
        ceilings = rng.uniform(0.55, 0.95, n_folds)
        rho = rng.uniform(0.50, 0.95, (n_candidates, n_folds))
        raw = int(np.argmax((rho / ceilings[None, :]).mean(axis=1)))
        sb_ceilings = np.array([spearman_brown(c) for c in ceilings])
        corrected = int(np.argmax((rho / sb_ceilings[None, :]).mean(axis=1)))
        moved += raw != corrected
    assert moved > 0, "the correction is NOT provably confined to reporting"
    # empirically a few percent; assert only that it is neither impossible nor pervasive
    assert 0.0 < moved / trials < 0.25


def test_split_half_ceiling_signature_takes_no_train_params() -> None:
    """Why the tune.py reweighting is candidate-independent: the ceiling never sees them."""
    import inspect

    params = set(inspect.signature(split_half_ceiling).parameters)
    assert "train_params" not in params
    assert "correct_length" in params, "the escape hatch must stay discoverable"


# --- end to end, through the real function ----------------------------------------------


def _synthetic_rows(n_participants: int = 12, n_ngrams: int = 24, seed: int = 3):
    """Rows with a real per-ngram signal plus participant noise, so rho is in (0, 1)."""
    from keybo.data.strokes import StrokeRow

    rng = np.random.default_rng(seed)
    truth = rng.uniform(120.0, 320.0, n_ngrams)
    rows = []
    for i in range(n_ngrams):
        samples = []
        for pid in range(n_participants):
            for _ in range(6):
                dur = int(truth[i] + rng.normal(0.0, 55.0))
                samples.append((90, dur, pid, 0))
        rows.append(
            StrokeRow(
                layout="synthetic",
                positions=(),
                ngram=f"n{i:02d}",
                frequency=1,
                samples=samples,
            )
        )
    return rows


def test_correct_length_is_ON_by_default_and_raises_the_ceiling() -> None:
    rows = _synthetic_rows()
    raw = split_half_ceiling(rows, n_boot=8, seed=0, correct_length=False)
    corrected = split_half_ceiling(rows, n_boot=8, seed=0)
    assert np.isfinite(raw) and np.isfinite(corrected)
    assert 0.0 < raw < 1.0
    assert corrected > raw, "the default must be the lengthened value"
    assert corrected <= 1.0 + 1e-12


def test_the_correction_is_applied_PER_BISECTION_not_to_the_mean() -> None:
    """``mean(f(r)) != f(mean(r))`` for a non-linear f, and per-bisection is correct."""
    rows = _synthetic_rows(seed=11)
    raw_mean = split_half_ceiling(rows, n_boot=8, seed=0, correct_length=False)
    per_bisection = split_half_ceiling(rows, n_boot=8, seed=0)
    naive_on_the_mean = spearman_brown(raw_mean)
    assert per_bisection != pytest.approx(naive_on_the_mean, abs=1e-9), (
        "if these agree the correction was applied to the mean, which is the wrong order"
    )
