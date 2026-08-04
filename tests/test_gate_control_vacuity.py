"""GATEWHY-1 — the high-wpm gate's SELF-COMPARISON is structurally un-failable.

Both INTERPFRAME-1 and HYBRIDB-1 ran a "mandatory gate control": the gate must PASS the incumbent
against the incumbent's own per-fold per-bucket rho, and both reported "GATE CONTROL PASSED =>
verdicts readable". The control cannot fail. When the baseline is the MEAN of the same seeds being
scored, each seed's delta is a deviation from that mean, so the deltas sum to zero -- and a STRUCTURAL
refusal requires EVERY seed to regress, which three numbers summing to zero cannot do.

These tests pin that as a property of the arithmetic, and pin the two facts that make the gate's
verdicts readable or not:

  * a self-referential baseline can never produce a structural refusal (the vacuity), and
  * an INDEPENDENT baseline (one the candidate did not contribute to) CAN -- which is what makes the
    SEEDNOISE control non-vacuous and therefore informative.

The second is what stops these tests from being vacuous themselves (INVARIANT 5): the subject varies,
because the same code refuses when the baseline is independent and cannot refuse when it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.verdicts import (
    HIGH_WPM_FLOOR,
    HIGH_WPM_TOLERANCE,
    HighWpmRegression,
    bucket_regression_report,
    require_no_high_wpm_regression,
)

BUCKETS = (40, 60, 80, 100, 120)


def _self_baseline(seed_rhos: list[dict[int, float]]) -> dict[int, float]:
    """The baseline the published gate controls used: the mean over the SAME seeds being scored."""
    return {
        b: float(np.mean([s[b] for s in seed_rhos if b in s]))
        for b in sorted({b for s in seed_rhos for b in s})
    }


def _structural(seed_rhos: list[dict[int, float]], baseline: dict[int, float]) -> list[int]:
    """Buckets regressing on EVERY seed -- validate()'s own STRUCTURAL rule."""
    counts: dict[int, int] = {}
    for i, s in enumerate(seed_rhos):
        for b in bucket_regression_report(s, baseline, f"s{i}")["regressing_high_buckets"]:
            counts[int(b)] = counts.get(int(b), 0) + 1
    return sorted(b for b, c in counts.items() if c == len(seed_rhos))


@pytest.mark.parametrize("n_seeds", [2, 3, 5])
def test_a_SELF_baseline_can_NEVER_produce_a_structural_refusal(n_seeds: int) -> None:
    """The vacuity, over adversarial inputs including pathological spreads.

    Not one observed pass: a search over inputs designed to break it. A seed set whose deltas are all
    negative does not exist, because they are deviations from their own mean.
    """
    rng = np.random.default_rng(20260804)
    for _ in range(4000):
        # Adversarial spreads: from degenerate to the full rho range, and heavily skewed draws.
        scale = float(rng.choice([1e-9, 1e-4, 0.01, 0.5, 1.5]))
        vals = rng.normal(0.0, scale, n_seeds) ** float(rng.choice([1.0, 3.0]))
        seeds = [{b: float(v) for b in BUCKETS} for v in vals]
        assert _structural(seeds, _self_baseline(seeds)) == []


def test_b_the_self_baseline_deltas_sum_to_zero_which_is_WHY() -> None:
    """The mechanism behind the vacuity, stated as the identity it is."""
    rng = np.random.default_rng(11)
    for _ in range(500):
        vals = rng.uniform(-1.0, 1.0, 3)
        base = _self_baseline([{120: float(v)} for v in vals])
        assert sum(float(v) - base[120] for v in vals) == pytest.approx(0.0, abs=1e-12)


def test_c_leave_one_seed_out_scales_the_delta_by_exactly_1p5_so_it_CANNOT_fix_it() -> None:
    """A symmetric LOSO baseline does not repair the vacuity -- it rescales it by exactly n/(n-1).

    This is why "compare each incumbent seed against the mean of the OTHERS" is not the fix: for 3
    seeds every deviation is exactly 1.5x the published one, so the SIGNS -- and therefore every
    verdict -- are identical.
    """
    rng = np.random.default_rng(5)
    for _ in range(500):
        x = rng.normal(0.8, 0.05, 3)
        published = x - x.mean()
        loso = np.array([x[i] - np.delete(x, i).mean() for i in range(3)])
        assert np.allclose(loso, 1.5 * published, rtol=0, atol=1e-12)
        # the ratio n/(n-1) generalizes, so the sign can never flip for any n >= 2
        for n in (2, 4, 7):
            y = rng.normal(0.8, 0.05, n)
            pub_n = y - y.mean()
            loso_n = np.array([y[i] - np.delete(y, i).mean() for i in range(n)])
            assert np.allclose(loso_n, (n / (n - 1)) * pub_n, rtol=0, atol=1e-12)


def test_d_an_INDEPENDENT_baseline_CAN_refuse_so_the_control_is_not_vacuous_by_nature() -> None:
    """The subject VARIES: the same gate refuses when the baseline is independent (INVARIANT 5).

    This is the SEEDNOISE construction -- a candidate scored against a baseline built from runs it did
    not contribute to -- and it is what makes that control informative where the self-comparison is not.
    """
    independent = {b: 0.80 for b in BUCKETS}
    reseeded = [{b: 0.80 for b in BUCKETS} | {120: 0.80 - d} for d in (0.02, 0.03, 0.04)]
    assert _structural(reseeded, independent) == [120]
    # and it does NOT fire when the same values are scored against their own mean
    assert _structural(reseeded, _self_baseline(reseeded)) == []


def test_e_the_gate_refuses_only_at_or_above_the_high_wpm_floor() -> None:
    """A low-bucket collapse must not fire the gate; the floor is a real boundary, not decoration."""
    independent = {b: 0.80 for b in BUCKETS}
    below = [{b: 0.80 for b in BUCKETS} | {60: 0.20} for _ in range(3)]
    assert _structural(below, independent) == []
    at_floor = [{b: 0.80 for b in BUCKETS} | {HIGH_WPM_FLOOR: 0.20} for _ in range(3)]
    assert _structural(at_floor, independent) == [HIGH_WPM_FLOOR]


def test_f_a_regression_exactly_AT_the_tolerance_does_not_refuse_but_beyond_it_does() -> None:
    """The tolerance boundary, pinned on both sides so a change to it cannot pass silently.

    The gate's rule is ``delta < -tolerance``, i.e. STRICT -- a deviation of exactly the tolerance
    passes. Constructed so the subtraction is exact in binary: ``0.0 - 0.005`` is representable, while
    the natural-looking ``0.80 - 0.005`` is not (it yields -0.005000000000000004, which IS beyond the
    tolerance). That is a property of float arithmetic rather than of the gate, and it means the
    boundary is knife-edge in practice -- worth pinning explicitly rather than approximately.
    """
    independent = {b: 0.0 for b in BUCKETS}
    at_tol = {b: 0.0 for b in BUCKETS} | {120: -HIGH_WPM_TOLERANCE}
    assert at_tol[120] - independent[120] == -HIGH_WPM_TOLERANCE  # exact, no rounding
    assert bucket_regression_report(at_tol, independent, "at")["regressing_high_buckets"] == []
    beyond = {b: 0.0 for b in BUCKETS} | {120: -HIGH_WPM_TOLERANCE * 1.001}
    assert bucket_regression_report(beyond, independent, "beyond")["regressing_high_buckets"] == [
        120
    ]
    # The knife-edge, recorded as a measurement: the same INTENT written the obvious way DOES refuse.
    naive = {b: 0.80 for b in BUCKETS} | {120: 0.80 - HIGH_WPM_TOLERANCE}
    assert naive[120] - 0.80 < -HIGH_WPM_TOLERANCE
    assert bucket_regression_report(naive, {b: 0.80 for b in BUCKETS}, "naive")[
        "regressing_high_buckets"
    ] == [120]


def test_g_require_no_high_wpm_regression_raises_on_an_independent_baseline_regression() -> None:
    """The enforcing wrapper, exercised on the failing side (not only the passing one)."""
    independent = {b: 0.80 for b in BUCKETS}
    bad = {b: 0.80 for b in BUCKETS} | {120: 0.70}
    with pytest.raises(HighWpmRegression, match="regresses in 1 of 3 high-wpm buckets"):
        require_no_high_wpm_regression(bad, independent, "candidate")
    # and it returns the passing report rather than raising when there is no regression
    ok = {b: 0.80 for b in BUCKETS}
    assert require_no_high_wpm_regression(ok, independent, "candidate")["passed"] is True


def test_i_NOT_MEASURED_is_not_DID_NOT_REGRESS_so_an_unreachable_verdict_is_ungated() -> None:
    """The absence-is-not-disproof rule, pinned on the side that matters.

    Closes mutation M11 (``gated = True`` unconditionally), which SURVIVED my first battery: nothing
    asserted that a verdict which could NOT be reached is reported as un-reached. This is the exact
    ambiguity TAUGATE-1 was: an artifact that merely omits a verdict reads the same whether the gate
    ran and passed or never ran. ``gated`` must be False, and ``passed`` must be ``None`` -- NOT
    ``True`` -- in every unreachable case.
    """
    candidate = {b: 0.80 for b in BUCKETS}
    # (a) no baseline at all
    r = bucket_regression_report(candidate, {}, "no baseline")
    assert r["gated"] is False and r["passed"] is None
    # (b) the baseline's TOP bucket is missing from the candidate -- "not measured"
    top_missing = {b: 0.80 for b in BUCKETS if b != 120}
    r = bucket_regression_report(top_missing, {b: 0.80 for b in BUCKETS}, "top missing")
    assert r["gated"] is False and r["passed"] is None
    # (c) a non-finite rho is not a measurement either
    r = bucket_regression_report(
        {b: 0.80 for b in BUCKETS} | {120: float("nan")}, {b: 0.80 for b in BUCKETS}, "nan"
    )
    assert r["gated"] is False and r["passed"] is None
    # (d) NO high bucket exists at all (every bucket below the floor)
    low_only = {40: 0.8, 60: 0.8}
    r = bucket_regression_report(low_only, low_only, "low only")
    assert r["gated"] is False and r["passed"] is None
    # ... and the reachable case is genuinely gated, so (a)-(d) are not trivially true
    r = bucket_regression_report(candidate, {b: 0.80 for b in BUCKETS}, "reachable")
    assert r["gated"] is True and r["passed"] is True


def test_j_the_enforcing_wrapper_REFUSES_an_unreachable_verdict_rather_than_passing_it() -> None:
    """Closes mutation M18 (deleting the ungated raise), which SURVIVED my first battery.

    ``require_no_high_wpm_regression`` must RAISE when the gate could not run. Without this, a
    candidate whose top bucket was never measured passes silently -- which is the failure the
    docstring's "'not measured' is not 'did not regress'" exists to prevent.
    """
    baseline = {b: 0.80 for b in BUCKETS}
    top_missing = {b: 0.80 for b in BUCKETS if b != 120}
    with pytest.raises(HighWpmRegression, match="could not run"):
        require_no_high_wpm_regression(top_missing, baseline, "unmeasured")
    with pytest.raises(HighWpmRegression, match="could not run"):
        require_no_high_wpm_regression({b: 0.80 for b in BUCKETS}, {}, "no baseline")
    # the message must name the bucket a caller has to supply, or it is not actionable
    with pytest.raises(HighWpmRegression, match=r"top wpm bucket \(120\)"):
        require_no_high_wpm_regression(top_missing, baseline, "unmeasured")


def test_h_support_is_recorded_but_provably_does_NOT_change_the_verdict() -> None:
    """GATESUPPORT-1's deliberate choice, pinned: the thinnest possible cell still refuses."""
    independent = {b: 0.80 for b in BUCKETS}
    bad = {b: 0.80 for b in BUCKETS} | {120: 0.70}
    thin = {120: {"n_cells": 1, "n_participants": 1}}
    thick = {120: {"n_cells": 100_000, "n_participants": 50_000}}
    r_thin = bucket_regression_report(bad, independent, "x", support=thin)
    r_thick = bucket_regression_report(bad, independent, "x", support=thick)
    assert r_thin["passed"] is False and r_thick["passed"] is False
    assert r_thin["regressing_high_buckets"] == r_thick["regressing_high_buckets"] == [120]
    assert r_thin["min_regressing_support"] == 1
    assert r_thick["min_regressing_support"] == 50_000
    # ... and absent support is distinguishable from thin support (absence-is-not-disproof)
    assert bucket_regression_report(bad, independent, "x")["support"] is None
