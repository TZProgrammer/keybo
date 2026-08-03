"""INVARIANT 1: interrogate the ESTIMAND. What does OLS slope(obs~pred) miss, and what can it
falsely flag? Everything here is a CONSTRUCTED case with a known ground truth, so each claim is a
demonstration rather than an assertion -- the same discipline the gate's own tests use.

Six probes:
  A. NEGATIVE CONTROL for the identity: an MSE-optimal predictor has slope(obs~pred)=1 at ANY r^2,
     and the two directions multiply to r^2 (the attenuation identity the parent verified).
  B. MISS-1: a pure MONOTONE nonlinearity with slope==1. OLS slope passes; the surface is wrong.
  C. MISS-2: two buckets miscalibrated in OPPOSITE directions cancel in the pooled slope.
  D. MISS-3: slope==1 says nothing about the INTERCEPT/level, and nothing about scatter.
  E. FALSE-FLAG-1: THIN-SLICE SAMPLING NOISE. With n cells and a given r, how often does a
     perfectly-calibrated slice land outside [0.90,1.10] by chance alone? This is the FLOOR the
     per-bucket scope must clear, and per the standing rule it is MEASURED here, not borrowed.
  F. FALSE-FLAG-2: errors-in-variables. If pred is measured with noise, OLS slope(obs~pred) is
     attenuated BELOW 1 even when the underlying surface is perfect -> a low slope can be a
     property of the ESTIMATOR, not the model. Deming/TLS is the alternative functional.
"""
import json
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

OUT = "/tmp/gateaudit/run/g03_estimand.json"
BAND = (0.90, 1.10)
res = {}


def ols_slope(pred, obs):
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    var = ((pred - pred.mean()) ** 2).sum()
    if var <= 0:
        return float("nan")
    return float(((pred - pred.mean()) * (obs - obs.mean())).sum() / var)


def deming_slope(pred, obs, lam=1.0):
    """Total-least-squares / Deming slope with error-variance ratio lam = var(e_obs)/var(e_pred)."""
    x = np.asarray(pred, float)
    y = np.asarray(obs, float)
    sxx = ((x - x.mean()) ** 2).mean()
    syy = ((y - y.mean()) ** 2).mean()
    sxy = ((x - x.mean()) * (y - y.mean())).mean()
    d = syy - lam * sxx
    return float((d + math.sqrt(d * d + 4 * lam * sxy * sxy)) / (2 * sxy))


# ---- A. the identity, as a NEGATIVE CONTROL ----------------------------------------------------
rng = np.random.default_rng(20260803)
n = 20000
truth = rng.normal(0, 1, n)
A = {}
for r_target in (0.3, 0.5, 0.657889, 0.9):
    noise = rng.normal(0, 1, n)
    # a feature with correlation ~r_target to truth
    feat = r_target * truth + math.sqrt(max(1 - r_target**2, 0)) * noise
    # the MSE-OPTIMAL predictor of truth given feat = E[truth|feat] = r * feat (both unit variance)
    b = ((feat - feat.mean()) * (truth - truth.mean())).sum() / ((feat - feat.mean()) ** 2).sum()
    pred = b * feat
    s_fwd = ols_slope(pred, truth)          # slope(obs ~ pred)
    s_rev = ols_slope(truth, pred)          # slope(pred ~ obs)
    r = float(np.corrcoef(pred, truth)[0, 1])
    A[f"r={r_target}"] = {
        "slope_obs_on_pred": s_fwd,
        "slope_pred_on_obs": s_rev,
        "product": s_fwd * s_rev,
        "r2": r * r,
        "product_minus_r2": s_fwd * s_rev - r * r,
        "sd_pred_over_sd_obs": float(pred.std() / truth.std()),
        "in_band": BAND[0] <= s_fwd <= BAND[1],
    }
res["A_identity_negative_control"] = A

# ---- B. MISS: monotone nonlinearity at slope 1 -------------------------------------------------
x = np.linspace(-2, 2, 4000)
obs = x.copy()
# a predictor that is a strictly increasing but strongly S-shaped function of truth, then rescaled
# so its OLS slope is EXACTLY 1. Rank metrics: tau = +1.0 (perfect). Magnitudes: badly wrong.
warped = np.tanh(2.2 * x)
b = ols_slope(warped, obs)
pred_b = warped / b * 1.0  # rescale so slope(obs~pred)=1
res["B_miss_monotone_nonlinearity"] = {
    "slope_obs_on_pred": ols_slope(pred_b, obs),
    "in_band": bool(BAND[0] <= ols_slope(pred_b, obs) <= BAND[1]),
    "kendall_tau": 1.0,
    "note": "tanh warp, rescaled to slope 1",
    "max_abs_gap_error": float(np.abs(pred_b - obs).max()),
    "worst_local_slope_ratio_center_vs_edge": float(
        (np.gradient(pred_b, x)[len(x) // 2]) / (np.gradient(pred_b, x)[10])
    ),
    "r2": float(np.corrcoef(pred_b, obs)[0, 1] ** 2),
}

# ---- C. MISS: opposite-direction buckets cancel in the pooled slope -----------------------------
rng = np.random.default_rng(7)
m = 3000
t1 = rng.normal(0, 1, m)
t2 = rng.normal(0, 1, m)
# bucket 1 COMPRESSED (pred range too small -> slope(obs~pred) > 1)
p1 = t1 / 1.45
# bucket 2 EXPANDED by the mirror factor
p2 = t2 * 1.45
pooled_pred = np.concatenate([p1, p2])
pooled_obs = np.concatenate([t1, t2])
res["C_miss_opposite_buckets_cancel"] = {
    "bucket1_slope": ols_slope(p1, t1),
    "bucket2_slope": ols_slope(p2, t2),
    "pooled_slope": ols_slope(pooled_pred, pooled_obs),
    "pooled_in_band": bool(BAND[0] <= ols_slope(pooled_pred, pooled_obs) <= BAND[1]),
    "buckets_in_band": [bool(BAND[0] <= ols_slope(p1, t1) <= BAND[1]),
                        bool(BAND[0] <= ols_slope(p2, t2) <= BAND[1])],
}

# ---- D. MISS: level/intercept and scatter are invisible ------------------------------------------
rng = np.random.default_rng(11)
t = rng.normal(50, 10, 5000)
res["D_miss_level_and_scatter"] = {
    "slope_with_50ms_offset": ols_slope(t + 50.0, t),
    "offset_ms": 50.0,
    "slope_with_huge_scatter": ols_slope(t + rng.normal(0, 30, 5000) * 0, t),
    "note_scatter": "OLS slope is invariant to an additive constant; a +50ms level error is "
                    "INVISIBLE to it. Level matters for ms-denominated claims, not rankings.",
}

# ---- E. FALSE FLAG: thin-slice sampling noise. MEASURE the floor. --------------------------------
# Ground truth: a PERFECTLY calibrated predictor (slope exactly 1 in expectation). How often does a
# slice of n cells land outside the band purely by chance? Uses the analytic sd as a cross-check.
E = {}
rng = np.random.default_rng(4242)
TRIALS = 20000
for r in (0.5, 0.657889, 0.8):
    for n_cells in (12, 20, 40, 64, 100, 200, 400, 900):
        # generate pred ~ N(0,1); obs = r_eff*pred + noise scaled so slope(obs~pred)=1 exactly in
        # expectation and corr = r  =>  obs = pred + e, var(e) = (1-r^2)/r^2
        sd_e = math.sqrt((1 - r * r) / (r * r))
        pred = rng.normal(0, 1, (TRIALS, n_cells))
        obs = pred + rng.normal(0, sd_e, (TRIALS, n_cells))
        pc = pred - pred.mean(axis=1, keepdims=True)
        oc = obs - obs.mean(axis=1, keepdims=True)
        slopes = (pc * oc).sum(axis=1) / (pc * pc).sum(axis=1)
        oob = float(np.mean((slopes < BAND[0]) | (slopes > BAND[1])))
        E[f"r={r}/n={n_cells}"] = {
            "false_flag_rate": oob,
            "slope_sd_measured": float(slopes.std()),
            "slope_sd_analytic": sd_e / math.sqrt(max(n_cells - 2, 1)),
            "slope_mean": float(slopes.mean()),
            "band_halfwidth_in_sds": float(0.10 / slopes.std()),
        }
res["E_false_flag_thin_slice_noise"] = E

# ---- F. FALSE FLAG: errors-in-variables attenuates the slope BELOW 1 ----------------------------
rng = np.random.default_rng(99)
n = 40000
true_surface = rng.normal(0, 1, n)
obs_f = true_surface + rng.normal(0, 0.3, n)      # observation noise
F = {}
for pred_noise in (0.0, 0.1, 0.2, 0.4):
    pred_f = true_surface + rng.normal(0, pred_noise, n)
    F[f"pred_noise_sd={pred_noise}"] = {
        "ols_slope_obs_on_pred": ols_slope(pred_f, obs_f),
        "in_band": bool(BAND[0] <= ols_slope(pred_f, obs_f) <= BAND[1]),
        "deming_slope_lam1": deming_slope(pred_f, obs_f),
        "truth": 1.0,
    }
res["F_false_flag_errors_in_variables"] = F

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, sort_keys=True)
print(json.dumps(res, indent=2, sort_keys=True)[:6000])
print("WROTE", OUT)
