"""The high-wpm non-regression gate must FIRE on the case that motivated it.

HIGHWPM-1 (ledger ``11f5ae7``) measured a blended objective against the shipped ``ms/char`` on the
AALTO leave-one-layout-out frame, per wpm bucket, over 16,281 scoreable cells:

    bucket     ms/char   drop-pool     delta
    40-60      +0.4215    +0.3913    -0.0303
    60-80      +0.4964    +0.4484    -0.0480
    80-100     +0.5321    +0.4674    -0.0647
    100-120    +0.5367    +0.4787    -0.0580
    120-140    +0.5268    +0.4536    -0.0733   <- the FASTEST bucket is the WORST

``_per_bucket_rho`` had computed these all along and ``validate.py`` even derived a worst bucket,
but **nothing gated on it**: ``grep -c bucket training/tune.py`` was 2, both passthrough. A user had
to notice. These tests pin the gate on the real numbers so the regression cannot return silently.

The values below are the MEASURED ones, not illustrative — if the gate stops firing on them, either
the gate broke or the measurement changed, and both need a human to look.
"""

from __future__ import annotations

import math

import pytest

from keybo.verdicts import (
    HighWpmRegression,
    bucket_regression_report,
    require_no_high_wpm_regression,
)

#: Measured per-bucket rho, HIGHWPM-1. Bucket start wpm -> rho.
_MS_PER_CHAR = {40: 0.4215, 60: 0.4964, 80: 0.5321, 100: 0.5367, 120: 0.5268}
_DROP_POOL = {40: 0.3913, 60: 0.4484, 80: 0.4674, 100: 0.4787, 120: 0.4536}


def test_the_gate_FIRES_on_the_measured_blend_regression() -> None:
    """The whole point: the -0.0733 at 120-140 must be refused, not reported."""
    with pytest.raises(HighWpmRegression) as exc:
        require_no_high_wpm_regression(_DROP_POOL, _MS_PER_CHAR, "drop-pool 50/50")
    message = str(exc.value)
    assert "120" in message, "the message must name the offending bucket"
    assert "drop-pool" in message, "and the candidate, so a log line is actionable"


def test_the_gate_is_SILENT_on_the_incumbent_against_itself() -> None:
    """A candidate identical to the baseline regresses nothing; the gate must not fire."""
    require_no_high_wpm_regression(_MS_PER_CHAR, _MS_PER_CHAR, "ms/char vs itself")


def test_the_gate_is_SILENT_on_an_improvement_at_the_top() -> None:
    better = dict(_MS_PER_CHAR)
    better[120] += 0.05
    require_no_high_wpm_regression(better, _MS_PER_CHAR, "faster-at-the-top candidate")


def test_a_LOW_bucket_regression_alone_does_NOT_fire_the_TOP_bucket_gate() -> None:
    """Scope discipline: this gate is about the fast buckets, by the user's ask.

    A candidate that gives up accuracy on slow typists but holds the top bucket is a different
    tradeoff, and this gate must not silently also decide that one.
    """
    slow_only = dict(_MS_PER_CHAR)
    slow_only[40] -= 0.20
    require_no_high_wpm_regression(slow_only, _MS_PER_CHAR, "slow-bucket sacrifice")


def test_the_tolerance_lets_a_negligible_wobble_through_but_not_the_measured_case() -> None:
    """The gate needs a tolerance or search noise trips it; it must still catch -0.0733."""
    wobble = dict(_MS_PER_CHAR)
    wobble[120] -= 0.001
    require_no_high_wpm_regression(wobble, _MS_PER_CHAR, "1e-3 wobble", tolerance=0.005)
    with pytest.raises(HighWpmRegression):
        require_no_high_wpm_regression(_DROP_POOL, _MS_PER_CHAR, "measured", tolerance=0.005)


def test_a_MISSING_top_bucket_is_refused_not_treated_as_a_pass() -> None:
    """An absent bucket is 'not measured', which is not 'did not regress'.

    This is the campaign's absence-is-not-disproof rule as code: a candidate whose fastest bucket
    was dropped for having too few cells must not be waved through as if it had passed.
    """
    missing_top = {k: v for k, v in _MS_PER_CHAR.items() if k != 120}
    with pytest.raises(HighWpmRegression) as exc:
        require_no_high_wpm_regression(missing_top, _MS_PER_CHAR, "candidate missing 120-140")
    assert "not measured" in str(exc.value)


def test_a_non_finite_top_bucket_is_refused() -> None:
    nan_top = dict(_MS_PER_CHAR)
    nan_top[120] = float("nan")
    with pytest.raises(HighWpmRegression):
        require_no_high_wpm_regression(nan_top, _MS_PER_CHAR, "nan at the top")


def test_an_empty_baseline_is_refused_rather_than_vacuously_passing() -> None:
    with pytest.raises(HighWpmRegression):
        require_no_high_wpm_regression(_MS_PER_CHAR, {}, "no baseline at all")


# --- the serialized verdict ---------------------------------------------------------------


def test_the_report_SERIALIZES_the_verdict_so_ungated_is_distinguishable_from_gated() -> None:
    """An artifact that omits the verdict reads identically whether or not a gate ran.

    That ambiguity is exactly how the tau gate spent a whole campaign reporting a pass while
    checking nothing (TAUGATE-1), so the report carries the verdict, the tolerance and the
    offending bucket explicitly.
    """
    report = bucket_regression_report(_DROP_POOL, _MS_PER_CHAR, "drop-pool 50/50")
    assert report["gated"] is True
    assert report["passed"] is False
    assert report["top_bucket"] == 120
    assert math.isclose(
        report["top_bucket_delta"], _DROP_POOL[120] - _MS_PER_CHAR[120], abs_tol=1e-9
    )
    assert report["worst_bucket"] == 120, "the measured worst bucket IS the fastest one"
    assert report["candidate"] == "drop-pool 50/50"
    # every bucket's delta is carried, so a reader can see the monotone structure
    assert set(report["deltas"]) == set(_MS_PER_CHAR)
    assert report["deltas"][120] < report["deltas"][40] < 0.0, "regression deepens with speed"


def test_the_report_records_a_PASS_as_explicitly_as_a_failure() -> None:
    report = bucket_regression_report(_MS_PER_CHAR, _MS_PER_CHAR, "incumbent")
    assert report["gated"] is True and report["passed"] is True
    assert report["top_bucket_delta"] == 0.0


def test_the_report_marks_an_UNGATED_result_as_ungated() -> None:
    """A caller with no baseline gets gated=False -- never a silent 'passed'."""
    report = bucket_regression_report(_MS_PER_CHAR, {}, "no baseline")
    assert report["gated"] is False
    assert report["passed"] is None, "None means 'no verdict', which is not True"


# --- ALL high-wpm buckets, not just the top one --------------------------------------------


def test_a_regression_one_bucket_BELOW_the_top_is_also_refused() -> None:
    """The user said "high WPM buckets" -- plural. A top-bucket-only gate has a hole.

    Measured on the shipped top-bucket-only form: a -0.20 collapse in 100-120 PASSED while the
    120-140 bucket was held flat. That is a large regression in expert-typing territory sneaking
    through a gate whose whole purpose is to catch exactly that.
    """
    sneak = dict(_MS_PER_CHAR)
    sneak[100] -= 0.20
    with pytest.raises(HighWpmRegression) as exc:
        require_no_high_wpm_regression(sneak, _MS_PER_CHAR, "regresses 100-120 only")
    assert "100" in str(exc.value), "the message must name the bucket that actually regressed"


def test_the_high_wpm_FLOOR_is_what_defines_scope_and_slow_buckets_stay_out() -> None:
    """Scope is the FLOOR, not the single top bucket: below it, a trade is still allowed.

    A candidate that gives up accuracy on 40-60 typists but holds every fast bucket is a different
    decision, and this gate must not silently settle it.
    """
    slow_only = dict(_MS_PER_CHAR)
    slow_only[40] -= 0.30
    require_no_high_wpm_regression(slow_only, _MS_PER_CHAR, "slow-bucket sacrifice")


def test_every_gated_bucket_at_or_above_the_floor_is_checked() -> None:
    """Each of 80/100/120 must be able to fire on its own."""
    for bucket in (80, 100, 120):
        candidate = dict(_MS_PER_CHAR)
        candidate[bucket] -= 0.10
        with pytest.raises(HighWpmRegression) as exc:
            require_no_high_wpm_regression(candidate, _MS_PER_CHAR, f"regresses {bucket}")
        assert str(bucket) in str(exc.value)


def test_the_report_names_EVERY_regressing_high_bucket_not_just_the_worst() -> None:
    """A reader fixing this needs the full list, not one example."""
    report = bucket_regression_report(_DROP_POOL, _MS_PER_CHAR, "drop-pool 50/50")
    assert set(report["regressing_high_buckets"]) == {80, 100, 120}
    assert report["high_wpm_floor"] == 80
