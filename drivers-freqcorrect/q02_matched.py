"""Q02 — INVARIANT B: the MATCHED-GEOMETRY contamination test, and what `b` encodes.

THE DESIGN (registered in FREQCORRECT-1 PREREG §3): at (near-)identical geometry, timing must not
depend on corpus frequency IF the surface is purely biomechanical. Group cells by the EXACT served
geometric feature vector (the 19 non-wpm columns) WITHIN a wpm bucket AND WITHIN a layout; inside
each group regress the observed target on log-frequency; pool the within-group slopes.

PRIMARY ESTIMAND: pooled within-group slope beta_freq, bootstrap CI95 over GROUPS (groups are the
independent unit, not cells).
DECISION RULE: CI95 excluding 0 => frequency-dependent timing at matched geometry EXISTS => a
practice effect is real and the estimand is legitimate. CI95 containing 0 => NULL, registered in
advance as a POSITIVE result for the current model's correctness.

THE SECOND HALF, which is what decides CORRECTNESS: compare beta_freq against the
frequency-dependence the FITTED b actually encodes (slope of b on log-freq over the same support).
R_encode = slope(b ~ logfreq) / beta_freq; R_encode >= 2 or <= 0.5 => MIS-ATTRIBUTION.

N4 (registered): the same design on a PERMUTED log-frequency column must give beta_freq = 0 within
CI -- guards against the grouping itself manufacturing a slope.

FLOOR (registered §9): I MEASURE the floor of this design rather than borrowing a constant. The
floor here is the spread of beta_freq under a design-preserving null (permuted frequency WITHIN
group), which is exactly N4's distribution -- so N4 doubles as the measured floor.
"""
import json
import time
from collections import defaultdict

import numpy as np
from _guard import ART, BOOT_SEED, CELL_KW, assert_d5, load_rows

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log("D5:")
assert_d5()

from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import build_cells  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
WPM_I = BIGRAM_FEATURE_NAMES.index("wpm")
GEO_COLS = [i for i in range(len(BIGRAM_FEATURE_NAMES)) if i != WPM_I]
out = {"cell_kw": CELL_KW, "boot_seed": BOOT_SEED, "geometry": "ROW_STAGGERED_31",
       "n_geo_cols": len(GEO_COLS)}

log("loading rows")
rows = load_rows()
cells = build_cells(rows, **CELL_KW)
log(f"  {len(rows)} rows -> {len(cells)} cells")

# The TARGET must be in the model's own space to be comparable with b (which is a LOGRAT
# log-ratio). LOGRAT = log(ms * wpm / 12000).
y_lograt = np.array([np.log(max(c.obs, 1.0) * max(c.wpm, 1.0) / 12000.0) for c in cells])
logf = np.log(np.array([max(c.frequency, 1) for c in cells], float))
require_finite(list(y_lograt) + list(logf), "targets and log-freq")

# ============================================================== build the matched-geometry groups
geo = np.vstack([bigram_features_from_positions(G, c.positions, wpm=c.wpm)[GEO_COLS] for c in cells])
key_of = [(c.layout, c.bucket, geo[i].tobytes()) for i, c in enumerate(cells)]
groups = defaultdict(list)
for i, k in enumerate(key_of):
    groups[k].append(i)
# a group is USABLE iff it has >=2 members AND non-degenerate log-freq variation
usable = {k: v for k, v in groups.items() if len(v) >= 2 and np.ptp(logf[v]) > 0}
out["design"] = {
    "n_groups_total": len(groups),
    "n_groups_usable": len(usable),
    "n_cells_in_usable": int(sum(len(v) for v in usable.values())),
    "group_size_hist": np.bincount([len(v) for v in usable.values()]).tolist(),
    "median_logfreq_spread_in_usable": float(np.median([np.ptp(logf[v]) for v in usable.values()])),
    "total_logfreq_range": float(np.ptp(logf)),
    "grouping": "(layout, wpm_bucket, EXACT 19-col served geometry vector)",
}
log(f"groups: {len(groups)} total, {len(usable)} usable (>=2 cells, freq varies); "
    f"{out['design']['n_cells_in_usable']} cells")
log(f"  median within-group log-freq spread {out['design']['median_logfreq_spread_in_usable']:.3f} "
    f"of total range {out['design']['total_logfreq_range']:.3f}")


def pooled_slope(y, x, grp, weights=None):
    """The WITHIN-GROUP pooled OLS slope of y on x: sum_g Sxy_g / sum_g Sxx_g.

    This is the fixed-effects (group-demeaned) estimator: it uses ONLY within-group variation,
    so every between-group difference in geometry is differenced out by construction.
    `weights` optionally weights each group (used for the per-group bootstrap).
    """
    num = 0.0
    den = 0.0
    for gi, idx in enumerate(grp):
        w = 1.0 if weights is None else weights[gi]
        if w == 0:
            continue
        xs, ys = x[idx], y[idx]
        xc, yc = xs - xs.mean(), ys - ys.mean()
        num += w * float((xc * yc).sum())
        den += w * float((xc * xc).sum())
    return num / den if den > 0 else np.nan


glist = [np.array(v) for v in usable.values()]
beta = pooled_slope(y_lograt, logf, glist)
log(f"PRIMARY: pooled within-group slope beta_freq = {beta:.6f} log-units per log-freq")

# bootstrap CI95 over GROUPS (the independent unit)
rng = np.random.default_rng(BOOT_SEED)
NB = 10000
ng = len(glist)
boots = np.empty(NB)
for bi in range(NB):
    cnt = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(float)
    boots[bi] = pooled_slope(y_lograt, logf, glist, weights=cnt)
require_finite(list(boots), "bootstrap betas")
ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
out["invariant_b_primary"] = {
    "beta_freq": float(beta), "ci95": list(ci), "n_boot": NB, "n_groups": ng,
    "boot_sd": float(boots.std(ddof=1)),
    "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    "sign": "negative (more frequent => faster)" if beta < 0 else "positive",
}
log(f"  CI95 over {ng} groups = [{ci[0]:.6f}, {ci[1]:.6f}]  => "
    f"{'EXCLUDES 0 (contamination/practice REAL)' if (ci[0] > 0 or ci[1] < 0) else 'CONTAINS 0 (NULL)'}")

# ============================================================== N4 — permuted-frequency null = FLOOR
log("N4: permuted log-frequency WITHIN group (the design-preserving null = the MEASURED FLOOR)")
NP = 2000
nulls = np.empty(NP)
rng2 = np.random.default_rng(BOOT_SEED + 1)
for pi in range(NP):
    lp = logf.copy()
    for idx in glist:
        lp[idx] = lp[rng2.permutation(idx)]
    nulls[pi] = pooled_slope(y_lograt, lp, glist)
require_finite(list(nulls), "null betas")
null_sd = float(nulls.std(ddof=1))
p_perm = float((np.abs(nulls) >= abs(beta)).mean())
out["n4_null_floor"] = {
    "n_perm": NP, "null_mean": float(nulls.mean()), "null_sd": null_sd,
    "null_p95_abs": float(np.percentile(np.abs(nulls), 95)),
    "MEASURED_FLOOR_p95_abs_beta": float(np.percentile(np.abs(nulls), 95)),
    "beta_over_floor": float(abs(beta) / np.percentile(np.abs(nulls), 95)),
    "permutation_p_two_sided": p_perm,
    "passes_null_centered_on_zero": bool(abs(nulls.mean()) < 2 * null_sd / np.sqrt(NP)),
}
log(f"  null: mean {nulls.mean():+.3e} sd {null_sd:.3e} p95|beta| "
    f"{np.percentile(np.abs(nulls), 95):.3e}")
log(f"  |beta| / MEASURED FLOOR = {abs(beta) / np.percentile(np.abs(nulls), 95):.2f}x   "
    f"permutation p = {p_perm:.4g}  (floor stated beside p, per §9)")

# ====================================== THE SECOND HALF: what does the FITTED b encode, same support?
log("fitting b (seed-mean over 3 seeds) and measuring what IT encodes on the SAME support")
bmaps = []
for s in (0, 1, 2):
    m = train_bigram_model(rows, target_wpm=90.0, geometry=G, random_state=s, n_jobs=48)
    bmaps.append(m.metadata.extra["training"]["practice_term"]["values"])
allng = set().union(*[set(b) for b in bmaps])
b_sm = {ng: float(np.mean([b.get(ng, 0.0) for b in bmaps])) for ng in allng}
bvec = np.array([b_sm.get(c.ngram, 0.0) for c in cells], float)

beta_b = pooled_slope(bvec, logf, glist)
boots_b = np.empty(NB)
rng3 = np.random.default_rng(BOOT_SEED + 2)
for bi in range(NB):
    cnt = np.bincount(rng3.integers(0, ng, ng), minlength=ng).astype(float)
    boots_b[bi] = pooled_slope(bvec, logf, glist, weights=cnt)
ci_b = (float(np.percentile(boots_b, 2.5)), float(np.percentile(boots_b, 97.5)))
R_encode = float(beta_b / beta) if beta != 0 else np.nan
# bootstrap the RATIO with the SAME group resample (paired), so the ratio's CI is honest
rng4 = np.random.default_rng(BOOT_SEED + 3)
rboots = np.empty(NB)
for bi in range(NB):
    cnt = np.bincount(rng4.integers(0, ng, ng), minlength=ng).astype(float)
    bb = pooled_slope(bvec, logf, glist, weights=cnt)
    by = pooled_slope(y_lograt, logf, glist, weights=cnt)
    rboots[bi] = bb / by if by != 0 else np.nan
rboots = rboots[np.isfinite(rboots)]
ci_r = (float(np.percentile(rboots, 2.5)), float(np.percentile(rboots, 97.5)))
out["invariant_b_encode"] = {
    "beta_b_on_logfreq": float(beta_b), "ci95": list(ci_b),
    "beta_obs_on_logfreq": float(beta), "R_encode": R_encode, "R_encode_ci95": list(ci_r),
    "registered_misattribution_if": "R_encode >= 2 or <= 0.5",
    "misattribution": bool(R_encode >= 2 or R_encode <= 0.5),
    "n_boot_ratio": int(len(rboots)),
}
log(f"  beta(b ~ logfreq) = {beta_b:.6f} CI95 [{ci_b[0]:.6f}, {ci_b[1]:.6f}]")
log(f"  R_encode = {R_encode:.4f} CI95 [{ci_r[0]:.4f}, {ci_r[1]:.4f}] => "
    f"{'MIS-ATTRIBUTION' if (R_encode >= 2 or R_encode <= 0.5) else 'CONSISTENT'}")

# ====================================== CONFOUND CONTROLS (INVARIANT B's named confounds)
log("confound controls: sample count, participant mix")
n_of = np.array([c.n for c in cells], float)
log_n = np.log(n_of)
# (1) does sample count co-vary with log-freq within group? if so it is a candidate confound
beta_n = pooled_slope(log_n, logf, glist)
# (2) partial out log(n): within-group multivariate OLS of y on [logfreq, log n]
def pooled_multi(y, Xs, grp):
    """Within-group (group-demeaned) multivariate OLS. Returns the coefficient vector."""
    Xc, yc = [], []
    for idx in grp:
        Xg_ = Xs[idx]
        Xc.append(Xg_ - Xg_.mean(0))
        yc.append(y[idx] - y[idx].mean())
    Xc = np.vstack(Xc)
    yc = np.concatenate(yc)
    coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return coef


coef_yn = pooled_multi(y_lograt, np.column_stack([logf, log_n]), glist)
# (3) participant overlap: fraction of each group's cell pairs sharing >=50% of participants
pids = [set(s[2] for s in c.samples) for c in cells]
jacc = []
for idx in glist:
    for a in range(len(idx)):
        for bq in range(a + 1, len(idx)):
            A, Bs = pids[idx[a]], pids[idx[bq]]
            u = len(A | Bs)
            if u:
                jacc.append(len(A & Bs) / u)
out["confounds"] = {
    "beta_logn_on_logfreq_within_group": float(beta_n),
    "beta_freq_partialling_out_logn": float(coef_yn[0]),
    "beta_logn_partialling_out_freq": float(coef_yn[1]),
    "beta_freq_shift_from_partialling_logn": float(coef_yn[0] - beta),
    "participant_jaccard_median": float(np.median(jacc)) if jacc else None,
    "participant_jaccard_mean": float(np.mean(jacc)) if jacc else None,
    "n_pairs": len(jacc),
    "NOT_CONTROLLED": ["bigram position-within-word", "word-level context",
                       "per-participant skill x frequency interaction"],
}
log(f"  beta(log n ~ logfreq | group) = {beta_n:.4f}  (sample count DOES track frequency)")
log(f"  beta_freq partialling out log n = {coef_yn[0]:.6f} (was {beta:.6f}, shift "
    f"{coef_yn[0] - beta:+.6f})")
log(f"  participant Jaccard within group: median "
    f"{np.median(jacc) if jacc else float('nan'):.3f} over {len(jacc)} pairs")

out["wall_s"] = time.time() - t0
path = f"{ART}/q02_matched.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
