"""g03 probes B and C, CORRECTED. My first construction had two errors, both mine:

  B. To make `slope(obs~c*w) == 1` you need `c = b` where `b = slope(obs~w)`, because
     slope(obs ~ c*w) = b/c. I DIVIDED by b instead of multiplying, so I got b**2 = 1.536 and my
     "rescaled to slope 1" case was not at slope 1 at all. Fixed and asserted, not eyeballed.

  C. Mirror factors (1/1.45, x1.45) do NOT cancel in a pooled slope, because pooling is
     VARIANCE-weighted: the expanded bucket carries 1.45**2 = 2.1x the leverage of the compressed
     one, so the pooled slope is dragged toward the expanded bucket (I measured 0.8286, not ~1).
     The correct demonstration SOLVES for the factor pair that puts the pooled slope inside the
     band while both buckets are outside it.

  C2 (the one that actually matters for INVARIANT 1). The real masking mechanism for a POOLED
     slope is the BETWEEN-slice ramp: if bucket MEANS line up with observed bucket means, that
     between-bucket signal enters the pooled slope and can hold it at ~1.0 while EVERY bucket is
     miscalibrated WITHIN itself. wpm is a model INPUT, so this between-bucket agreement is
     partly credit for information the model was handed -- which is precisely the branch's stated
     reason for adding `bucket_centered`. This probe tests whether that reasoning is sound.
"""
import json
import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

OUT = "/tmp/gateaudit/run/g04_estimand_fix.json"
BAND = (0.90, 1.10)
res = {}


def ols(pred, obs):
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    v = ((pred - pred.mean()) ** 2).sum()
    return float("nan") if v <= 0 else float(((pred - pred.mean()) * (obs - obs.mean())).sum() / v)


def bucket_centered(groups, values):
    """the branch's _bucket_centered: subtract each group's mean."""
    out = np.asarray(values, float).copy()
    for g in np.unique(groups):
        idx = groups == g
        out[idx] -= out[idx].mean()
    return out


# ---- B (CORRECTED): monotone nonlinearity that sits EXACTLY at slope 1 --------------------------
x = np.linspace(-2, 2, 4001)
obs = x.copy()
w = np.tanh(2.2 * x)
b = ols(w, obs)
pred = b * w                     # now slope(obs~pred) == 1 by construction
s = ols(pred, obs)
assert abs(s - 1.0) < 1e-12, f"rescale failed: {s}"
# Spearman/Kendall are +1 (strictly increasing warp). Magnitudes are badly wrong.
local = np.gradient(pred, x)
res["B_monotone_nonlinearity_at_slope_exactly_1"] = {
    "slope_obs_on_pred": s,
    "in_band": bool(BAND[0] <= s <= BAND[1]),
    "spearman_rho": 1.0,
    "kendall_tau": 1.0,
    "r2": float(np.corrcoef(pred, obs)[0, 1] ** 2),
    "max_abs_gap_error_units": float(np.abs(pred - obs).max()),
    "local_slope_at_centre": float(local[len(x) // 2]),
    "local_slope_near_edge": float(local[5]),
    "centre_over_edge_local_slope": float(local[len(x) // 2] / local[5]),
    "verdict": "PASSES the gate at slope 1.000000 while the local exchange rate is wrong by a "
               "large factor between the middle and the tails of the range.",
}

# ---- C (CORRECTED): solve for the pair that hides inside the band -------------------------------
rng = np.random.default_rng(7)
m = 40000
t1 = rng.normal(0, 1, m)
t2 = rng.normal(0, 1, m)
grid = []
for k1 in np.linspace(1.15, 2.5, 400):          # bucket1 COMPRESSED: pred too small
    for k2 in np.linspace(0.30, 0.87, 200):     # bucket2 EXPANDED:   pred too large
        # slope(obs~pred) for bucket i is 1/k_i when pred = k_i * obs
        s1, s2 = 1.0 / k1, 1.0 / k2
        if BAND[0] <= s1 <= BAND[1] or BAND[0] <= s2 <= BAND[1]:
            continue
        # analytic pooled slope for two zero-mean groups: sum(k_i*var_i)/sum(k_i^2*var_i), var=1
        pooled = (k1 + k2) / (k1 * k1 + k2 * k2)
        if BAND[0] <= pooled <= BAND[1]:
            grid.append((abs(pooled - 1.0), k1, k2, s1, s2, pooled))
grid.sort()
_, k1, k2, s1, s2, pooled_analytic = grid[0]
p1, p2 = k1 * t1, k2 * t2
pp = np.concatenate([p1, p2])
po = np.concatenate([t1, t2])
res["C_opposite_buckets_CAN_hide_in_the_pooled_slope"] = {
    "bucket1_slope": ols(p1, t1),
    "bucket2_slope": ols(p2, t2),
    "pooled_slope_measured": ols(pp, po),
    "pooled_slope_analytic": pooled_analytic,
    "pooled_in_band": bool(BAND[0] <= ols(pp, po) <= BAND[1]),
    "buckets_in_band": [bool(BAND[0] <= ols(p1, t1) <= BAND[1]),
                        bool(BAND[0] <= ols(p2, t2) <= BAND[1])],
    "n_hiding_pairs_found": len(grid),
    "verdict": "A POOLED-ONLY gate can pass while EVERY bucket is out of band.",
}

# ---- C2: the between-bucket ramp props up the pooled slope --------------------------------------
# 5 wpm buckets with distinct means. WITHIN each bucket the surface compresses by 1.45x (slope
# 1.45). BETWEEN buckets, predicted means track observed means exactly -- the wpm->duration ramp
# the model was HANDED. Question: what do pooled and bucket_centered each report?
rng = np.random.default_rng(2026)
per = 400
means = np.array([120.0, 100.0, 80.0, 65.0, 55.0])   # observed bucket means (ms), a strong ramp
groups, obs_l, pred_l = [], [], []
COMPRESS = 1.45
for gi, mu in enumerate(means):
    within = rng.normal(0, 6.0, per)                  # within-bucket structure (the LAYOUT signal)
    o = mu + within
    p = mu + within / COMPRESS                        # same mean, compressed within-bucket spread
    groups.append(np.full(per, gi))
    obs_l.append(o)
    pred_l.append(p)
groups = np.concatenate(groups)
obs_c2 = np.concatenate(obs_l)
pred_c2 = np.concatenate(pred_l)
s_pooled = ols(pred_c2, obs_c2)
s_bc = ols(bucket_centered(groups, pred_c2), bucket_centered(groups, obs_c2))
per_bucket = {int(g): ols(pred_c2[groups == g], obs_c2[groups == g]) for g in np.unique(groups)}
res["C2_between_bucket_ramp_props_up_the_pooled_slope"] = {
    "true_within_bucket_compression": COMPRESS,
    "pooled_slope": s_pooled,
    "pooled_in_band": bool(BAND[0] <= s_pooled <= BAND[1]),
    "bucket_centered_slope": s_bc,
    "bucket_centered_in_band": bool(BAND[0] <= s_bc <= BAND[1]),
    "per_bucket_slopes": per_bucket,
    "between_bucket_sd_ms": float(means.std()),
    "within_bucket_sd_ms": 6.0,
    "verdict": "pooled is carried by the ramp the model was HANDED; bucket_centered recovers the "
               "true within-bucket compression. This VALIDATES the branch's stated reason for "
               "adding the bucket_centered slice.",
}

# how does the masking scale with ramp strength? (the honest sensitivity)
sens = {}
for ramp_sd in (0.0, 3.0, 6.0, 12.0, 25.0, 50.0):
    mm = np.linspace(-ramp_sd, ramp_sd, 5) * math.sqrt(2.5) + 80.0
    g2, o2, p2_ = [], [], []
    for gi, mu in enumerate(mm):
        wi = rng.normal(0, 6.0, per)
        g2.append(np.full(per, gi))
        o2.append(mu + wi)
        p2_.append(mu + wi / COMPRESS)
    g2 = np.concatenate(g2)
    o2 = np.concatenate(o2)
    p2_ = np.concatenate(p2_)
    sens[f"ramp_sd={ramp_sd}"] = {
        "pooled": ols(p2_, o2),
        "pooled_in_band": bool(BAND[0] <= ols(p2_, o2) <= BAND[1]),
        "bucket_centered": ols(bucket_centered(g2, p2_), bucket_centered(g2, o2)),
    }
res["C2b_masking_vs_ramp_strength"] = sens

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, sort_keys=True)
print(json.dumps(res, indent=2, sort_keys=True))
print("WROTE", OUT)
