"""E-CONTROL + INVARIANT 2: is the compression UNIFORM or DIFFERENTIAL?

E-control   reproduce all six published (n, raw, model, ratio) from the 486 per-pair records.
K1          the estimand reconciliation: slope(raw~pred) * slope(pred~raw) == r^2, exactly.
K2          THE REGISTERED DECISION RULE -- fit ONE global affine map raw ~ a + b*pred, then test
            each class contrast IN THE RESIDUALS. Pair-level bootstrap, Holm over 6 classes.
            UNIFORM iff no class survives Holm; DIFFERENTIAL iff >=1 does.
K3          the secondary, in the parent's own currency: is the observed spread of the six RATIOS
            larger than the spread expected under a uniform-truth null?
"""
import json
import time

import numpy as np
from _guard import (ART, BOOT_SEED, CLASS_ORDER, PUBLISHED, SFBPRICE_C02, assert_d5, class_masks,
                    sha)

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

from keybo.verdicts import require_finite  # noqa: E402

N_BOOT = 10_000            # registered in CALIB-1 PREREG
RNG = np.random.default_rng(BOOT_SEED)
out = {"n_boot": N_BOOT, "boot_seed": BOOT_SEED,
       "source": SFBPRICE_C02, "source_sha256": sha(SFBPRICE_C02)}

# ---------------------------------------------------------------- the 486 pairs (REUSED, not re-read)
src = json.load(open(SFBPRICE_C02))
SAME, OTHER = src["pairs_same"], src["pairs_other"]
PAIRS = SAME + OTHER
same_ids = {id(r) for r in SAME}
MASKS = class_masks(PAIRS, same_ids)
log(f"loaded {len(PAIRS)} pairs ({len(SAME)} same-finger, {len(OTHER)} other) "
    f"from sfbprice c02, sha256 {out['source_sha256'][:16]}")

raw = np.array([r["raw"] for r in PAIRS], float)
pred = np.array([r["pred"] for r in PAIRS], float)
pred_b = np.array([r["pred_b"] for r in PAIRS], float)   # practice term restored
nsamp = np.array([r["n"] for r in PAIRS], float)
require_finite(list(raw) + list(pred) + list(pred_b), "pair arrays")


def contrast(values, mask):
    """median(values | in class) - median(values | not in class) -- the parent's estimand."""
    m = np.asarray(mask, bool)
    return float(np.median(values[m]) - np.median(values[~m]))


# ============================================================== E-CONTROL: reproduce the published six
log("E-CONTROL: reproducing the six published contrasts")
econ = {}
worst = 0.0
for name in CLASS_ORDER:
    m = np.asarray(MASKS[name], bool)
    cr, cm = contrast(raw, m), contrast(pred, m)
    pn, praw, pmod, prat = PUBLISHED[name]
    d = {"n": int(m.sum()), "raw": cr, "model": cm, "ratio": cm / cr,
         "published": {"n": pn, "raw": praw, "model": pmod, "ratio": prat},
         "diff_n": int(m.sum()) - pn, "diff_raw": cr - praw, "diff_model": cm - pmod,
         "diff_ratio": cm / cr - prat}
    econ[name] = d
    worst = max(worst, abs(d["diff_raw"]), abs(d["diff_model"]))
    log(f"  {name:20s} n={d['n']:4d} raw={cr:+8.2f} model={cm:+7.2f} ratio={cm / cr:.4f}  "
        f"(published {praw:+.2f}/{pmod:+.2f}/{prat:.3f}; dn={d['diff_n']})")
out["e_control"] = econ
out["e_control_worst_abs_diff_ms"] = worst
# published values are rounded to 2dp, so 0.005 is the exact-match tolerance
out["e_control_pass"] = bool(worst < 0.01 and all(d["diff_n"] == 0 for d in econ.values()))
log(f"  E-CONTROL {'PASS' if out['e_control_pass'] else 'FAIL'}: worst |diff| = {worst:.4f} ms")

# ============================================================ K1: the estimand reconciliation, exactly
log("K1: estimand reconciliation")
b_fwd = float(np.polyfit(pred, raw, 1)[0])        # raw ~ pred   (the 'expansion' slope)
a_fwd = float(np.polyfit(pred, raw, 1)[1])
b_rev = float(np.polyfit(raw, pred, 1)[0])        # pred ~ raw   (the repo's calibration_slope form)
r = float(np.corrcoef(pred, raw)[0, 1])
out["k1"] = {"slope_raw_on_pred": b_fwd, "intercept_raw_on_pred": a_fwd,
             "slope_pred_on_raw": b_rev, "r": r, "r2": r * r,
             "product": b_fwd * b_rev, "product_minus_r2": b_fwd * b_rev - r * r,
             "reciprocal_of_fwd": 1.0 / b_fwd,
             "sd_raw": float(raw.std(ddof=1)), "sd_pred": float(pred.std(ddof=1)),
             "mean_published_ratio": float(np.mean([PUBLISHED[c][3] for c in CLASS_ORDER]))}
log(f"  slope(raw~pred)={b_fwd:.6f}  slope(pred~raw)={b_rev:.6f}  r2={r * r:.6f}  "
    f"product-r2={b_fwd * b_rev - r * r:.3e}")
log(f"  1/slope(raw~pred)={1 / b_fwd:.6f}  != slope(pred~raw)  => attenuation, not a scale error")

# ============================== K2: THE REGISTERED DECISION RULE -- residual contrasts after ONE map
# The global affine map is fitted ONCE on all 486 pairs; a UNIFORM compression means this single map
# explains every class's gap, i.e. leaves no class contrast in the residuals.
log("K2: residual class contrasts after ONE global affine map (the registered rule)")
resid = raw - (a_fwd + b_fwd * pred)


def boot_residual_contrasts(n_boot):
    """Pair-level bootstrap: refit the affine map INSIDE each draw (the map is estimated, so its
    sampling error must propagate -- the FLOOR-D lesson from sfbprice)."""
    n = len(PAIRS)
    acc = {c: np.empty(n_boot) for c in CLASS_ORDER}
    acc_ratio = {c: np.empty(n_boot) for c in CLASS_ORDER}
    idx_all = np.arange(n)
    for i in range(n_boot):
        d = RNG.choice(idx_all, n, replace=True)
        rr, pp = raw[d], pred[d]
        bb, aa = np.polyfit(pp, rr, 1)
        res_d = rr - (aa + bb * pp)
        for c in CLASS_ORDER:
            m = np.asarray(MASKS[c], bool)[d]
            if m.sum() < 2 or (~m).sum() < 2:
                acc[c][i] = np.nan
                acc_ratio[c][i] = np.nan
                continue
            acc[c][i] = np.median(res_d[m]) - np.median(res_d[~m])
            cr = np.median(rr[m]) - np.median(rr[~m])
            cm = np.median(pp[m]) - np.median(pp[~m])
            acc_ratio[c][i] = cm / cr if cr != 0 else np.nan
    return acc, acc_ratio


log(f"  bootstrapping {N_BOOT} draws (affine map refitted inside every draw)")
boot_res, boot_ratio = boot_residual_contrasts(N_BOOT)

k2 = {}
for c in CLASS_ORDER:
    m = np.asarray(MASKS[c], bool)
    point = float(np.median(resid[m]) - np.median(resid[~m]))
    bs = boot_res[c][np.isfinite(boot_res[c])]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # two-sided bootstrap p: fraction of draws on the other side of 0, doubled
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / len(bs))))          # floor at resolution, never report 0
    rb = boot_ratio[c][np.isfinite(boot_ratio[c])]
    k2[c] = {"n": int(m.sum()), "residual_contrast_ms": point,
             "ci95": [float(lo), float(hi)], "p_boot": p,
             "ratio_point": econ[c]["ratio"],
             "ratio_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]}

# Holm over the 6 classes, at alpha = 0.05 (registered)
order = sorted(CLASS_ORDER, key=lambda c: k2[c]["p_boot"])
alpha, k = 0.05, len(CLASS_ORDER)
rejected, still = [], True
for i, c in enumerate(order):
    thresh = alpha / (k - i)
    k2[c]["holm_threshold"] = thresh
    if still and k2[c]["p_boot"] <= thresh:
        k2[c]["holm_reject"] = True
        rejected.append(c)
    else:
        k2[c]["holm_reject"] = False
        still = False

for c in CLASS_ORDER:
    d = k2[c]
    log(f"  {c:20s} resid={d['residual_contrast_ms']:+7.2f} ms  "
        f"CI95[{d['ci95'][0]:+7.2f},{d['ci95'][1]:+7.2f}]  p={d['p_boot']:.4f}  "
        f"holm<={d['holm_threshold']:.4f}  {'REJECT-NULL' if d['holm_reject'] else 'not distinguishable'}")
    log(f"  {'':20s}   ratio={d['ratio_point']:.4f} CI95[{d['ratio_ci95'][0]:.4f},{d['ratio_ci95'][1]:.4f}]")

out["k2"] = k2
out["k2_holm_rejected"] = rejected
out["k2_verdict"] = "DIFFERENTIAL" if rejected else "UNIFORM"
log(f"  ==> REGISTERED VERDICT: {out['k2_verdict']}  (Holm-surviving classes: {rejected or 'none'})")

# ================= K3: the secondary -- is the RATIO spread bigger than a uniform-truth null predicts?
# Uniform-truth null: the single global affine map IS the whole story. Simulate raw* = a + b*pred +
# eps, resampling eps from the pooled residuals (so the null keeps the real noise magnitude but
# destroys any class structure), then recompute the six ratios and their spread.
log("K3: is the six-ratio spread larger than a uniform-truth null predicts?")
obs_ratios = np.array([econ[c]["ratio"] for c in CLASS_ORDER])
obs_spread = float(obs_ratios.max() - obs_ratios.min())
obs_sd = float(obs_ratios.std(ddof=1))
null_spread, null_sd = np.empty(N_BOOT), np.empty(N_BOOT)
for i in range(N_BOOT):
    eps = RNG.choice(resid, len(resid), replace=True)
    raw_star = a_fwd + b_fwd * pred + eps
    rs = []
    for c in CLASS_ORDER:
        m = np.asarray(MASKS[c], bool)
        cr = np.median(raw_star[m]) - np.median(raw_star[~m])
        cm = np.median(pred[m]) - np.median(pred[~m])
        rs.append(cm / cr if cr != 0 else np.nan)
    rs = np.array(rs, float)
    null_spread[i] = np.nanmax(rs) - np.nanmin(rs)
    null_sd[i] = np.nanstd(rs, ddof=1)
p_spread = float(max((null_spread >= obs_spread).mean(), 1.0 / N_BOOT))
p_sd = float(max((null_sd >= obs_sd).mean(), 1.0 / N_BOOT))
out["k3"] = {"observed_ratio_spread": obs_spread, "observed_ratio_sd": obs_sd,
             "null_spread_p50": float(np.percentile(null_spread, 50)),
             "null_spread_p95": float(np.percentile(null_spread, 95)),
             "null_sd_p50": float(np.percentile(null_sd, 50)),
             "null_sd_p95": float(np.percentile(null_sd, 95)),
             "p_spread": p_spread, "p_sd": p_sd,
             "ratios": {c: econ[c]["ratio"] for c in CLASS_ORDER}}
log(f"  observed spread {obs_spread:.4f} (sd {obs_sd:.4f}) vs null p50 "
    f"{out['k3']['null_spread_p50']:.4f} / p95 {out['k3']['null_spread_p95']:.4f}  "
    f"=> p={p_spread:.4f} (spread), p={p_sd:.4f} (sd)")

# ------------------------------------------------- also report the practice-term-restored ratios (H-ESTIMAND preview)
pb = {}
for c in CLASS_ORDER:
    m = np.asarray(MASKS[c], bool)
    cr, cm, cmb = contrast(raw, m), contrast(pred, m), contrast(pred_b, m)
    pb[c] = {"raw": cr, "model": cm, "model_with_b": cmb,
             "ratio": cm / cr, "ratio_with_b": cmb / cr,
             "gap_closed_frac": (cmb - cm) / (cr - cm) if cr != cm else float("nan")}
    log(f"  [b] {c:20s} ratio {cm / cr:.4f} -> {cmb / cr:.4f}  "
        f"(closes {100 * pb[c]['gap_closed_frac']:+.1f}% of its gap)")
out["practice_term_preview"] = pb

out["wall_s"] = time.time() - t0
p = f"{ART}/k01_uniform.json"
json.dump(out, open(p, "w"), indent=1)
log(f"wrote {p}  ({out['wall_s']:.1f}s)")
