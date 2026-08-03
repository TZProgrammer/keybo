"""Q05 — the EQUAL-n FALSIFIER for my own INVARIANT B result (registered in ADDENDUM 2, b51e7e1).

THE PROBLEM (measured in q02, published in my report): within a matched-geometry group,
beta(log sample-count ~ log-frequency) = 1.0051. Frequency and per-cell sample count are therefore
NOT separately identified, and a noise-attenuation mechanism -- thin rare cells => attenuated
IQR-means => apparent slope -- cannot be refuted by partialling.

THE DESIGN: subsample every cell in a group to a COMMON n (the group's minimum raw-sample count),
recompute each cell's IQR-mean target from the drawn samples ONLY, and re-run the identical
within-group pooled slope. Every cell in a group then carries the SAME sample count, so the
attenuation channel is closed BY CONSTRUCTION rather than by regression adjustment.

REGISTERED RULE (it can only go against me):
  CI95 still excludes 0 AND |beta| > this design's OWN permutation floor => my §(b) SURVIVES.
  CI95 contains 0 / beta collapses into its floor                        => my §(b) FALLS.
REGISTERED PREDICTION: beta stays negative and clears its floor but SHRINKS vs -0.065084, because
subsampling discards data and adds target noise (so an attenuated estimate cannot later be sold as
a clean confirmation).
FLOOR: re-MEASURED for this design (equal-n targets are noisier), never reused from N4's 1.385e-02.
PLACEBO: the equal-n design on PERMUTED frequency must give 0 within CI.
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

from keybo.data.strokes import iqr_average  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import build_cells  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
WPM_I = BIGRAM_FEATURE_NAMES.index("wpm")
GEO_COLS = [i for i in range(len(BIGRAM_FEATURE_NAMES)) if i != WPM_I]
out = {"cell_kw": CELL_KW, "boot_seed": BOOT_SEED, "geometry": "ROW_STAGGERED_31",
       "registered_in": "PREREGISTRATIONS.md FREQCORRECT-1 ADDENDUM 2 (commit b51e7e1)"}

log("loading rows")
rows = load_rows()
cells = build_cells(rows, **CELL_KW)
logf = np.log(np.array([max(c.frequency, 1) for c in cells], float))

# ---- rebuild the IDENTICAL grouping q02 used ------------------------------------------------
geo = np.vstack([bigram_features_from_positions(G, c.positions, wpm=c.wpm)[GEO_COLS] for c in cells])
groups = defaultdict(list)
for i, c in enumerate(cells):
    groups[(c.layout, c.bucket, geo[i].tobytes())].append(i)
usable = {k: v for k, v in groups.items() if len(v) >= 2 and np.ptp(logf[v]) > 0}
glist = [np.array(v) for v in usable.values()]
log(f"  {len(cells)} cells; {len(usable)} usable groups (identical to q02's design)")

# the full-sample baseline, recomputed here so the comparison is within one process
y_full = np.array([np.log(max(c.obs, 1.0) * max(c.wpm, 1.0) / 12000.0) for c in cells])


def pooled_slope(y, x, grp, weights=None):
    """Within-group pooled OLS slope: sum_g Sxy_g / sum_g Sxx_g (the fixed-effects estimator)."""
    num = den = 0.0
    for gi, idx in enumerate(grp):
        w = 1.0 if weights is None else weights[gi]
        if w == 0:
            continue
        xs, ys = x[idx], y[idx]
        xc, yc = xs - xs.mean(), ys - ys.mean()
        num += w * float((xc * yc).sum())
        den += w * float((xc * xc).sum())
    return num / den if den > 0 else np.nan


beta_full = pooled_slope(y_full, logf, glist)
out["baseline_full_sample"] = {"beta_freq": float(beta_full), "q02_published": -0.065084212,
                              "note": "recomputed in-process; must match q02 to ~1e-6"}
log(f"  baseline (full-sample) beta_freq = {beta_full:.6f}  [q02 published -0.065084]")

# ---- the group-wise common-n and the per-cell sample pools ---------------------------------
n_of = np.array([c.n for c in cells])
group_nmin = {}
for gi, idx in enumerate(glist):
    group_nmin[gi] = int(n_of[idx].min())
nmins = np.array(list(group_nmin.values()))
out["equal_n_design"] = {
    "n_groups": len(glist),
    "common_n_min": int(nmins.min()), "common_n_median": float(np.median(nmins)),
    "common_n_max": int(nmins.max()),
    "cells_total_in_usable": int(sum(len(v) for v in glist)),
    "mean_frac_samples_retained": float(np.mean(
        [group_nmin[gi] * len(idx) / n_of[idx].sum() for gi, idx in enumerate(glist)])),
    "design": ("every cell in a group subsampled WITHOUT replacement to the group's minimum raw "
               "sample count; target = IQR-mean of the DRAWN samples only"),
}
log(f"  common n per group: min {nmins.min()} median {np.median(nmins):.1f} max {nmins.max()}; "
    f"mean fraction of samples retained {out['equal_n_design']['mean_frac_samples_retained']:.3f}")

# per-cell duration arrays (only for cells inside usable groups -- the rest are never drawn)
in_use = sorted({int(i) for idx in glist for i in idx})
dur = {i: np.array([s[1] for s in cells[i].samples], dtype=np.int64) for i in in_use}
wpm_of = np.array([max(c.wpm, 1.0) for c in cells])


def equal_n_targets(rng):
    """One equal-n draw: y[i] = log(IQR-mean(drawn) * wpm / 12000) for every cell in a usable group."""
    y = y_full.copy()
    for gi, idx in enumerate(glist):
        k = group_nmin[gi]
        for i in idx:
            d = dur[int(i)]
            pick = d if len(d) <= k else rng.choice(d, size=k, replace=False)
            y[i] = np.log(max(iqr_average(list(pick)), 1.0) * wpm_of[i] / 12000.0)
    return y


R = 200
log(f"running {R} equal-n draws")
rng = np.random.default_rng(BOOT_SEED)
betas = np.empty(R)
for r in range(R):
    betas[r] = pooled_slope(equal_n_targets(rng), logf, glist)
    if r < 3 or r == R - 1:
        log(f"  draw {r}: beta_equal_n = {betas[r]:.6f}")
require_finite(list(betas), "equal-n betas")
ci = (float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5)))
out["equal_n_primary"] = {
    "n_draws": R, "beta_mean": float(betas.mean()), "beta_sd": float(betas.std(ddof=1)),
    "beta_median": float(np.median(betas)), "ci95_over_draws": list(ci),
    "shrinkage_vs_full_sample": float(betas.mean() / beta_full),
    "registered_prediction_shrinks": bool(abs(betas.mean()) < abs(beta_full)),
}
log(f"EQUAL-n beta_freq = {betas.mean():.6f} +- {betas.std(ddof=1):.6f}  "
    f"CI95 over draws [{ci[0]:.6f}, {ci[1]:.6f}]  "
    f"(shrinkage vs full-sample x{betas.mean() / beta_full:.4f})")

# ---- the FLOOR for THIS design: permuted frequency on equal-n targets -----------------------
NP = 400
log(f"MEASURED FLOOR for the equal-n design: {NP} within-group frequency permutations")
rngf = np.random.default_rng(BOOT_SEED + 11)
nulls = np.empty(NP)
for p in range(NP):
    y = equal_n_targets(rngf)
    lp = logf.copy()
    for idx in glist:
        lp[idx] = lp[rngf.permutation(idx)]
    nulls[p] = pooled_slope(y, lp, glist)
require_finite(list(nulls), "equal-n null betas")
floor95 = float(np.percentile(np.abs(nulls), 95))
out["equal_n_floor"] = {
    "n_perm": NP, "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)),
    "MEASURED_FLOOR_p95_abs_beta": floor95,
    "beta_over_floor": float(abs(betas.mean()) / floor95),
    "permutation_p_two_sided": float((np.abs(nulls) >= abs(betas.mean())).mean()),
    "q02_full_sample_floor_for_reference": 1.385e-02,
    "note": "re-MEASURED for this design; equal-n targets are noisier so N4's floor cannot be reused",
}
log(f"  null mean {nulls.mean():+.3e} sd {nulls.std(ddof=1):.3e}  MEASURED FLOOR p95|beta| "
    f"{floor95:.3e}  (q02's full-sample floor was 1.385e-02)")
log(f"  |beta_equal_n| / floor = {abs(betas.mean()) / floor95:.2f}x  permutation p = "
    f"{(np.abs(nulls) >= abs(betas.mean())).mean():.4g}")

# ---- REGISTERED VERDICT --------------------------------------------------------------------
excl0 = bool(ci[0] > 0 or ci[1] < 0)
clears = bool(abs(betas.mean()) > floor95)
out["registered_verdict"] = {
    "ci95_excludes_zero": excl0, "clears_own_floor": clears,
    "verdict": ("INVARIANT B SURVIVES the equal-n falsifier" if (excl0 and clears)
                else "INVARIANT B FALLS -- beta collapses into its floor"),
    "rule": ("registered in ADDENDUM 2: CI95 excludes 0 AND |beta| > this design's own permutation "
             "floor => survives; otherwise falls, and the CORRECTNESS verdict then rests on the "
             "geometric-R2 nulls (P2/P3/N3) alone"),
}
log(f"REGISTERED VERDICT: {out['registered_verdict']['verdict']}")

# ---- the within-design placebo (subsampling must not manufacture a slope) -------------------
out["placebo_permuted_freq_is_the_floor_run"] = {
    "null_centered_on_zero": bool(abs(nulls.mean()) < 2 * nulls.std(ddof=1) / np.sqrt(NP)),
    "null_mean": float(nulls.mean()),
    "note": "the floor run IS the registered placebo: equal-n targets + permuted frequency",
}
log(f"  placebo (= the floor run): null centered on zero = "
    f"{out['placebo_permuted_freq_is_the_floor_run']['null_centered_on_zero']}")

out["wall_s"] = time.time() - t0
path = f"{ART}/q05_equaln.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
