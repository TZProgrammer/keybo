"""`valid_domain` as a HARD constraint (OPTEVIDENCE-1, ledger 9fd5c7b).

An unclamped fitted curve is an UNBOUNDED objective: a search manufactured 96.5% of its
apparent win by walking two CORRECTLY-signed gauges outside their fitted bands. Flagging that
on the artifact did nothing, because a maximizer does not read flags. These tests pin the two
levels actually measured.
"""

from __future__ import annotations

import pytest

from keybo.analysis.evidence_scorer import (
    CLAMP,
    EXTRAPOLATE,
    REJECT,
    SEARCH_DOMAIN_POLICY,
    LossCurve,
    OutOfDomainError,
)

# The two measured exploits, from OPTEVIDENCE-1's arm A champion.
COMFORT_LEVEL, COMFORT_DOMAIN = 2.9592, (6.5236, 11.5644)  # BELOW the floor
SR_ROLL_LEVEL, SR_ROLL_DOMAIN = 17.8343, (1.9997, 8.3369)  # 2.14x the ceiling


def _hinge(metric: str, domain: tuple[float, float], knot: float) -> LossCurve:
    """A hinge whose slope keeps paying outside the domain — the shape that was exploited."""
    return LossCurve(
        metric=metric,
        form="hinge",
        coeffs=[0.0, -1.0, -2.0],
        knot=knot,
        domain=domain,
        observed_range=domain,
        weight=-1.0,
        weight_ci=(-2.0, 0.0),
        r2=0.9,
        r2_linear=0.5,
        mean_abs_shap=1.0,
        shap_share_pct=10.0,
    )


def test_search_policy_is_clamp_not_extrapolate():
    """The constant an optimizer must reach for; naming it is what stops a silent default."""
    assert SEARCH_DOMAIN_POLICY == CLAMP
    assert SEARCH_DOMAIN_POLICY != EXTRAPOLATE


def test_comfort_below_the_floor_is_clamped_to_the_edge():
    curve = _hinge("comfort", COMFORT_DOMAIN, knot=9.2622)
    assert not curve.in_domain(COMFORT_LEVEL)
    clamped = curve.price(COMFORT_LEVEL, policy=CLAMP)
    at_edge = curve.price(COMFORT_DOMAIN[0], policy=EXTRAPOLATE)
    assert clamped == pytest.approx(at_edge)


def test_sr_roll_above_the_ceiling_is_clamped_to_the_edge():
    curve = _hinge("sr-roll", SR_ROLL_DOMAIN, knot=5.0)
    assert not curve.in_domain(SR_ROLL_LEVEL)
    clamped = curve.price(SR_ROLL_LEVEL, policy=CLAMP)
    at_edge = curve.price(SR_ROLL_DOMAIN[1], policy=EXTRAPOLATE)
    assert clamped == pytest.approx(at_edge)


def test_clamping_removes_the_unbounded_reward():
    """The actual defect: extrapolating paid MORE the further out the search pushed."""
    curve = _hinge("sr-roll", SR_ROLL_DOMAIN, knot=5.0)
    near = curve.price(SR_ROLL_DOMAIN[1] + 1.0, policy=EXTRAPOLATE)
    far = curve.price(SR_ROLL_DOMAIN[1] + 50.0, policy=EXTRAPOLATE)
    assert far < near, "extrapolation must be the unbounded case these tests exist to stop"
    # Under CLAMP the reward saturates: pushing 50x further buys exactly nothing.
    assert curve.price(SR_ROLL_DOMAIN[1] + 1.0, policy=CLAMP) == pytest.approx(
        curve.price(SR_ROLL_DOMAIN[1] + 50.0, policy=CLAMP)
    )


def test_in_domain_prices_are_bit_identical_across_policies():
    """The fix must not perturb any supported level — otherwise it changes frozen numbers."""
    curve = _hinge("comfort", COMFORT_DOMAIN, knot=9.2622)
    for level in (6.6, 8.0, 9.2622, 11.0, 11.5):
        assert curve.price(level, policy=CLAMP) == curve.price(level, policy=EXTRAPOLATE)
        assert curve.price(level, policy=REJECT) == curve.price(level, policy=EXTRAPOLATE)


def test_reject_raises_and_names_the_gauge_and_domain():
    curve = _hinge("sr-roll", SR_ROLL_DOMAIN, knot=5.0)
    with pytest.raises(OutOfDomainError) as excinfo:
        curve.price(SR_ROLL_LEVEL, policy=REJECT)
    message = str(excinfo.value)
    assert "sr-roll" in message and "17.83" in message


def test_unknown_policy_is_refused():
    curve = _hinge("comfort", COMFORT_DOMAIN, knot=9.2622)
    with pytest.raises(ValueError, match="unknown domain policy"):
        curve.price(7.0, policy="soft")


# --- the vectorized path must be the SAME function (ARM D found it was not) ----------------


def test_price_many_matches_price_exactly_under_every_policy():
    """The defect ARM D caught: the optimizer's fast path had its OWN vectorized price, so the
    policy never reached a search. Pin the two at EXACT float equality, not approximately —
    anything looser lets a reimplementation drift back apart.
    """
    import numpy as np

    curve = _hinge("sr-roll", SR_ROLL_DOMAIN, knot=5.0)
    levels = np.array([0.5, 1.9997, 3.0, 5.0, 8.3369, 12.0, SR_ROLL_LEVEL, 60.0])
    for policy in (EXTRAPOLATE, CLAMP):
        vector = curve.price_many(levels, policy=policy)
        scalar = [curve.price(float(x), policy=policy) for x in levels]
        assert list(vector) == scalar, policy


def test_price_many_clamp_saturates_like_the_scalar_path():
    import numpy as np

    curve = _hinge("comfort", COMFORT_DOMAIN, knot=9.2622)
    far = curve.price_many(np.array([COMFORT_DOMAIN[0] - 50.0]), policy=CLAMP)[0]
    edge = curve.price(COMFORT_DOMAIN[0], policy=EXTRAPOLATE)
    assert far == pytest.approx(edge)


def test_price_many_rejects_out_of_domain_under_reject():
    import numpy as np

    curve = _hinge("sr-roll", SR_ROLL_DOMAIN, knot=5.0)
    with pytest.raises(OutOfDomainError, match="sr-roll"):
        curve.price_many(np.array([3.0, SR_ROLL_LEVEL]), policy=REJECT)


def test_price_many_refuses_an_unknown_policy():
    import numpy as np

    curve = _hinge("comfort", COMFORT_DOMAIN, knot=9.2622)
    with pytest.raises(ValueError, match="unknown domain policy"):
        curve.price_many(np.array([7.0]), policy="soft")
