"""Tests for keybo.analysis.los — the LOS instrument's structural guarantees.

These are the guarantees that make it an instrument rather than a plausible formula: the null must
return 0.5, an at-chance flip hazard must force 0.5, a sub-floor margin must not clear the DECIDED
bar however tiny its p-value, and a huge margin must clear it. They are the calibration invariants
from the preregistration, encoded so they cannot silently regress.
"""

import numpy as np
import pytest

from keybo.analysis.los import (
    apply_flip_hazard,
    compute_los,
    flip_hazard,
    scale_floor_to_n,
    split_half_floor,
)


def test_flip_hazard_strata():
    # PICK2-1's measured co-observed sign-flip rates, by |gap| stratum.
    assert flip_hazard(0.1) == 0.81
    assert flip_hazard(0.41) == 0.81
    assert flip_hazard(0.42) == 0.74
    assert flip_hazard(0.96) == 0.74
    assert flip_hazard(0.97) == 0.30
    assert flip_hazard(3.03) == 0.30
    assert flip_hazard(3.04) == 0.12
    assert flip_hazard(9.7) == 0.12
    # symmetric in sign
    assert flip_hazard(-1.5) == flip_hazard(1.5)


def test_apply_flip_hazard_registered_properties():
    # q=0 leaves los unchanged; q=0.5 forces 0.5 for ANY los (the anti-laundering guarantee).
    for los in (0.5, 0.9, 0.99, 0.999, 1.0, 0.01):
        assert apply_flip_hazard(los, 0.0) == pytest.approx(los)
        assert apply_flip_hazard(los, 0.5) == pytest.approx(0.5)
    # monotone: more hazard pulls a confident los toward 0.5
    assert apply_flip_hazard(0.99, 0.12) > apply_flip_hazard(0.99, 0.30) > apply_flip_hazard(0.99, 0.74)
    with pytest.raises(ValueError):
        apply_flip_hazard(0.9, 1.5)


def test_null_board_against_itself_is_exactly_half():
    # NULL-1: d_s == 0 for all s. LOS must be 0.5, not 1.0.
    rng = np.random.default_rng(0)
    ms = 254.0 + rng.normal(0, 0.4, size=25)
    r = compute_los(ms, ms.copy(), floor=0.29, a_name="X", b_name="X")
    assert r.mean_margin == 0.0
    assert r.los_seed == pytest.approx(0.5)
    assert r.los_design == pytest.approx(0.5)
    assert r.los_typist == pytest.approx(0.5)
    assert r.verdict == "UNDECIDED"


def test_subfloor_margin_cannot_be_decided_however_tiny_the_pvalue():
    # The pathology case: a tiny, extremely consistent margin far BELOW the floor. p can be ~0
    # (huge t) yet LOS_design must sit near equipoise 0.5 because all its mass is in the tie region.
    n = 25
    d = np.full(n, -0.03) + np.random.default_rng(1).normal(0, 0.001, n)  # margin 0.03, floor 0.29
    ms_b = 254.0 + np.random.default_rng(2).normal(0, 0.4, n)
    ms_a = ms_b + d
    r = compute_los(ms_a, ms_b, floor=0.29, a_name="A", b_name="B")
    assert r.p_two_sided < 1e-10          # p-value screams "significant"
    assert r.los_seed > 0.999             # seed-only confidence is near 1 (the pathology)
    assert 0.45 < r.los_design < 0.55     # ...but ROPE confidence is at equipoise (it's a tie)
    assert r.p_tie > 0.99                 # essentially all posterior mass is within resolution
    assert r.p_exceed < 0.01              # no resolvable difference
    assert r.verdict == "UNDECIDED"       # and it is NOT decided


def test_known_big_margin_is_decided_and_near_one():
    # BIG: qwerty-scale gap, ~9.7 ms/char, 0/25 signs. LOS_design ~ 1, and even with the >3.0 gap
    # flip hazard (0.12) LOS_typist stays high.
    n = 25
    ms_b = 263.0 + np.random.default_rng(3).normal(0, 0.4, n)  # qwerty-ish
    ms_a = ms_b - 9.7 + np.random.default_rng(4).normal(0, 0.4, n)  # tuned board
    r = compute_los(ms_a, ms_b, floor=0.29, a_name="tuned", b_name="qwerty")
    assert r.signs_a_faster == 25
    assert r.los_design >= 0.99
    assert r.verdict == "A-DECIDED"
    assert r.los_typist >= 0.85           # 0.12 hazard caps it a little, by construction
    assert apply_flip_hazard(r.los_design, 0.12) == pytest.approx(r.los_typist)


def test_los_monotone_in_margin():
    n = 25
    ms_b = 254.0 + np.random.default_rng(5).normal(0, 0.4, n)
    prev = -1.0
    for gap in (0.0, 0.1, 0.3, 0.5, 1.0, 3.0):
        ms_a = ms_b - gap
        r = compute_los(ms_a, ms_b, floor=0.29)
        assert r.los_design >= prev - 1e-12
        prev = r.los_design


def test_split_half_floor_truth_is_zero_and_positive():
    # Placebo margins are |mean(H1)-mean(H2)| on the SAME board: truth 0, spread = noise > 0.
    rng = np.random.default_rng(6)
    panel = 254.0 + rng.normal(0, 0.4, size=(5, 25))
    f = split_half_floor(panel, n_partitions=500, rng=rng)
    assert f["p90"] > 0
    assert f["p50"] < f["p90"] < f["p99"] <= f["max"]
    assert f["half_n"] == 12
    # scaling to full n shrinks it (a verdict at n=25 has finer resolution than a half at n=12)
    scaled = scale_floor_to_n(f["p90"], f["half_n"], 25)
    assert 0 < scaled < f["p90"]


def test_refuses_nonfinite():
    ms = np.array([254.0] * 25)
    bad = ms.copy(); bad[0] = np.nan
    with pytest.raises(ValueError):
        compute_los(bad, ms, floor=0.29)


def test_degeneracy_zero_variance_does_not_manufacture_confidence():
    # D1: zero seed-variance must NOT manufacture a decided verdict on a sub-floor margin.
    # Every seed gives the identical margin 0.2 (< floor 0.29): a point mass entirely inside the
    # tie region -> los_design = 0.5 (equipoise), NOT 1.0.
    ms_b = np.array([254.0] * 25)
    ms_a = ms_b - 0.2
    r = compute_los(ms_a, ms_b, floor=0.29)
    assert r.sd_margin == pytest.approx(0.0)
    assert r.p_tie == pytest.approx(1.0)
    assert r.los_design == pytest.approx(0.5)
    assert r.verdict == "UNDECIDED"
    # ...but a zero-variance margin well BEYOND the floor is legitimately decided.
    r2 = compute_los(ms_b - 1.0, ms_b, floor=0.29)
    assert r2.p_a_beyond == pytest.approx(1.0)
    assert r2.los_design == pytest.approx(1.0)
    assert r2.verdict == "A-DECIDED"


def test_los_is_directional_and_complementary():
    # LOS(B vs A) == 1 - LOS(A vs B), exactly (fishtest's LOS is directional the same way).
    n = 25
    ms_b = 254.0 + np.random.default_rng(11).normal(0, 0.4, n)
    ms_a = ms_b - 0.5
    rab = compute_los(ms_a, ms_b, floor=0.29, a_name="A", b_name="B")
    rba = compute_los(ms_b, ms_a, floor=0.29, a_name="B", b_name="A")
    assert rab.los_design == pytest.approx(1 - rba.los_design)
    assert rab.los_seed == pytest.approx(1 - rba.los_seed)
    assert rab.p_exceed == pytest.approx(rba.p_exceed)   # non-directional: symmetric


# ==================================================================================================
# LOSVAR-1: the FOURTH estimand, los_valid — los_design on a posterior widened by the MEASURED
# layout-differential validation error. These encode the registered structural properties.
# ==================================================================================================


def _pair(n=25, margin=-1.046, sd=0.18, seed=7):
    """A synthetic paired sample with a controlled mean margin and seed-scatter sd."""
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, 1.0, size=n)
    d = (d - d.mean()) / d.std(ddof=1) * sd + margin      # exact mean and ddof=1 sd
    base = 254.0 + rng.normal(0, 0.4, size=n)
    return base + d, base


def test_los_valid_with_zero_sigma_diff_is_los_design_bit_for_bit():
    """NC3, as a test: sigma_diff=0 must be a STRICT generalization, not merely close."""
    a, b = _pair()
    r = compute_los(a, b, floor=0.2929, sigma_diff=0.0)
    assert r.los_valid == r.los_design            # bit-for-bit, not approx
    assert r.sigma_diff == 0.0
    # and the default (no sigma_diff passed at all) behaves identically
    assert compute_los(a, b, floor=0.2929).los_valid == r.los_design


def test_los_valid_goes_to_equipoise_as_sigma_diff_grows():
    a, b = _pair()
    prev = compute_los(a, b, floor=0.2929, sigma_diff=0.0).los_valid
    assert prev > 0.99                                    # decided on seed noise alone
    for s in (0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1e4):
        cur = compute_los(a, b, floor=0.2929, sigma_diff=s).los_valid
        assert cur <= prev + 1e-12                        # monotone non-increasing
        prev = cur
    assert compute_los(a, b, floor=0.2929, sigma_diff=1e6).los_valid == pytest.approx(0.5, abs=1e-3)


def test_los_valid_null_board_against_itself_is_exactly_half_for_any_sigma_diff():
    """NULL-1 must hold for the new estimand at EVERY sigma_diff — a widened scale cannot
    manufacture a direction out of a zero margin."""
    rng = np.random.default_rng(0)
    ms = 254.0 + rng.normal(0, 0.4, size=25)
    for s in (0.0, 0.05, 0.5, 5.0, 50.0):
        r = compute_los(ms, ms.copy(), floor=0.2929, sigma_diff=s)
        assert r.los_valid == pytest.approx(0.5)
        assert r.verdict_valid == "UNDECIDED"


def test_los_valid_is_symmetric_under_swapping_the_boards():
    """Directionality: LOS_valid(B vs A) == 1 - LOS_valid(A vs B), like every other estimand."""
    a, b = _pair()
    fwd = compute_los(a, b, floor=0.2929, sigma_diff=0.4).los_valid
    rev = compute_los(b, a, floor=0.2929, sigma_diff=0.4).los_valid
    assert fwd + rev == pytest.approx(1.0)


def test_los_valid_scale_is_the_quadrature_sum():
    a, b = _pair(sd=0.18)
    r = compute_los(a, b, floor=0.2929, sigma_diff=0.4)
    assert r.scale_valid == pytest.approx(np.hypot(r.sem_margin, 0.4))
    assert r.scale_valid > r.sem_margin                   # strictly wider than seed-only


def test_los_valid_rejects_negative_sigma_diff():
    a, b = _pair()
    with pytest.raises(ValueError):
        compute_los(a, b, floor=0.2929, sigma_diff=-0.1)


def test_los_valid_subfloor_margin_stays_at_equipoise():
    """The anti-pathology guarantee must survive the widening: a sub-floor margin cannot become
    decided by ADDING uncertainty (that would be nonsense) nor drift away from 0.5."""
    a, b = _pair(margin=-0.011, sd=0.02)                  # arm-B-vs-candidate-like: deep sub-floor
    for s in (0.0, 0.1, 0.5, 2.0):
        r = compute_los(a, b, floor=0.2929, sigma_diff=s)
        assert 0.45 <= r.los_valid <= 0.55
