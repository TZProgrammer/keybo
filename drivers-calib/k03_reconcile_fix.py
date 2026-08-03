"""RECONCILE the two slope estimands on ONE dataset, then run INVARIANT 3's held-out fix test.

R1  the repo's cell-level estimand vs my pair-level one, computed side by side on the SAME cells,
    so the 0.84-vs-1.4618 disagreement is explained by measurement rather than by argument.
R2  the OPTIMALITY identity, verified numerically: for an MSE-optimal predictor sd(pred)=r*sd(obs)
    hence slope(obs~pred)=1 at ANY r^2. Quantifies how far the surface sits from that.
F1  INVARIANT 3, the registered adoption test: fit the affine map on HELD-IN folds only, apply to
    HELD-OUT predictions, and measure paired per-fold wmae/umae deltas + the post-fix slope.
    This is a real LOLO run through the reviewed validate() path (not a re-implementation).
F2  the same test for the practice-term restoration (the biggest single mechanism from k02).
"""
import json
import time

import numpy as np
from _guard import ART, BI, BOOT_SEED, assert_d5

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import (_predict_cells, build_cells, calibration_slope,  # noqa: E402
                                     leave_one_layout_out, uniform_mae, weighted_mae)
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
SEEDS = [0, 1, 2]
HOLDOUTS = ["azerty", "dvorak", "qwerty", "qwertz"]
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
out = {"seeds": SEEDS, "holdouts": HOLDOUTS, "cell_kw": CELL_KW, "boot_seed": BOOT_SEED,
       "geometry": "ROW_STAGGERED_31"}

log(f"loading {BI}")
rows = load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
assert len(rows) == 2202, f"frame drift: {len(rows)} != 2202"
log(f"  {len(rows)} rows; layouts = {sorted({r.layout for r in rows})}")


def affine_fit(pred, obs):
    """OLS obs ~ a + b*pred. Returns (a, b). b == calibration_slope(pred, obs) by construction."""
    b, a = np.polyfit(np.asarray(pred, float), np.asarray(obs, float), 1)
    return float(a), float(b)


def bucket_centered_slope(cells, pred, obs):
    """slope(obs~pred) AFTER removing each wpm bucket's mean from BOTH sides.

    The wpm axis is a model INPUT (validate.py:_bucket_centered), so a pooled slope is dominated by
    the wpm->duration ramp the model was handed. Centering isolates the STRUCTURAL part -- the part
    a layout comparison actually consumes.
    """
    from collections import defaultdict
    by = defaultdict(list)
    for i, c in enumerate(cells):
        by[c.bucket].append(i)
    p, o = np.asarray(pred, float).copy(), np.asarray(obs, float).copy()
    for idx in by.values():
        p[idx] -= p[idx].mean()
        o[idx] -= o[idx].mean()
    return calibration_slope(p, o), p, o


# ==================================================================== the LOLO run (one per seed/fold)
records = []
for holdout in HOLDOUTS:
    train_rows, test_rows = leave_one_layout_out(rows, holdout)
    test_cells = build_cells(test_rows, **CELL_KW)
    train_cells = build_cells(train_rows, **CELL_KW)
    obs_te = np.array([c.obs for c in test_cells])
    obs_tr = np.array([c.obs for c in train_cells])
    for seed in SEEDS:
        log(f"fold {holdout} seed {seed}: training (n_train_cells={len(train_cells)}, "
            f"n_test_cells={len(test_cells)})")
        model = train_bigram_model(train_rows, target_wpm=90.0, geometry=G,
                                   random_state=seed, n_jobs=48)
        pred_te = _predict_cells(model, test_cells, G)
        pred_tr = _predict_cells(model, train_cells, G)
        require_finite(list(pred_te) + list(pred_tr), f"{holdout}/{seed} predictions")

        # --- the practice term, restored on the MODEL side (the k02 mechanism) ------------------
        bmap = (model.metadata.extra.get("training", {}).get("practice_term") or {}).get("values", {})
        bt = np.array([bmap.get(c.ngram, 0.0) for c in test_cells], float)
        # LOGRAT: b is a log-ratio, so restoring it is multiplicative on ms
        pred_te_b = pred_te * np.exp(bt)

        # --- R1: the two estimands, side by side on the SAME cells -----------------------------
        slope_pooled = calibration_slope(pred_te, obs_te)
        slope_centered, pc, oc = bucket_centered_slope(test_cells, pred_te, obs_te)
        slope_rev_pooled = calibration_slope(obs_te, pred_te)
        r_pooled = float(np.corrcoef(pred_te, obs_te)[0, 1])
        r_centered = float(np.corrcoef(pc, oc)[0, 1])

        # --- R2: how far from the MSE-optimal conditional mean? --------------------------------
        # optimal: sd(pred) = r * sd(obs). shortfall = required/actual.
        def shrink_factor(p, o):
            rr = float(np.corrcoef(p, o)[0, 1])
            need = rr * float(np.std(o, ddof=1))
            have = float(np.std(p, ddof=1))
            return {"r": rr, "sd_obs": float(np.std(o, ddof=1)), "sd_pred": have,
                    "sd_pred_optimal": need, "over_shrink_factor": need / have if have else np.nan}

        rec = {
            "holdout": holdout, "seed": seed,
            "n_test_cells": len(test_cells), "n_train_cells": len(train_cells),
            "r1": {"slope_pooled_obs_on_pred": slope_pooled,
                   "slope_pooled_pred_on_obs": slope_rev_pooled,
                   "product": slope_pooled * slope_rev_pooled, "r2_pooled": r_pooled ** 2,
                   "slope_bucket_centered": slope_centered, "r2_centered": r_centered ** 2},
            "r2_optimality": {"pooled": shrink_factor(pred_te, obs_te),
                              "bucket_centered": shrink_factor(pc, oc)},
        }

        # --- F1: the REGISTERED adoption test -- map fitted on HELD-IN, applied to HELD-OUT ----
        a_in, b_in = affine_fit(pred_tr, obs_tr)
        pred_te_cal = a_in + b_in * pred_te
        # the ORACLE map (fitted on held-out itself) bounds what any affine map could buy
        a_or, b_or = affine_fit(pred_te, obs_te)
        pred_te_oracle = a_or + b_or * pred_te

        variants = {"base": pred_te, "affine_heldin": pred_te_cal,
                    "affine_oracle": pred_te_oracle, "practice_b": pred_te_b}
        # b + affine, both: does restoring b make the affine map unnecessary?
        a_b, b_b = affine_fit(pred_tr * np.exp(np.array([bmap.get(c.ngram, 0.0)
                                                         for c in train_cells], float)), obs_tr)
        variants["practice_b_then_affine_heldin"] = a_b + b_b * pred_te_b

        rec["variants"] = {}
        for vname, pv in variants.items():
            sl_c, pvc, ovc = bucket_centered_slope(test_cells, pv, obs_te)
            rec["variants"][vname] = {
                "wmae": weighted_mae(test_cells, pv, obs_te),
                "umae": uniform_mae(pv, obs_te),
                "slope_pooled": calibration_slope(pv, obs_te),
                "slope_bucket_centered": sl_c,
                "mae": float(np.mean(np.abs(pv - obs_te))),
            }
        rec["heldin_map"] = {"a": a_in, "b": b_in}
        rec["oracle_map"] = {"a": a_or, "b": b_or}
        log(f"  slope pooled={slope_pooled:.4f} centered={slope_centered:.4f} | "
            f"held-in map b={b_in:.4f} oracle b={b_or:.4f} | "
            f"wmae base={rec['variants']['base']['wmae']:.4f} "
            f"aff={rec['variants']['affine_heldin']['wmae']:.4f} "
            f"b={rec['variants']['practice_b']['wmae']:.4f}")
        records.append(rec)

out["records"] = records

# ================================================================ paired per-fold deltas (MOR-FIX-1)
log("PAIRED PER-FOLD DELTAS (MOR-FIX-1: paired deltas, never a mean of ratios)")
VARIANTS = ["affine_heldin", "affine_oracle", "practice_b", "practice_b_then_affine_heldin"]
summary = {}
for v in VARIANTS:
    d_w = np.array([r["variants"][v]["wmae"] - r["variants"]["base"]["wmae"] for r in records])
    d_u = np.array([r["variants"][v]["umae"] - r["variants"]["base"]["umae"] for r in records])
    sl_p = np.array([r["variants"][v]["slope_pooled"] for r in records])
    sl_c = np.array([r["variants"][v]["slope_bucket_centered"] for r in records])
    # per-fold means over seeds, then the fold-level sign count (the registered >=3/4 rule)
    per_fold = {}
    for h in HOLDOUTS:
        sel = [i for i, r in enumerate(records) if r["holdout"] == h]
        per_fold[h] = {"d_wmae": float(d_w[sel].mean()), "d_umae": float(d_u[sel].mean()),
                       "slope_pooled": float(sl_p[sel].mean()),
                       "slope_centered": float(sl_c[sel].mean())}
    folds_nonworse_w = sum(1 for h in HOLDOUTS if per_fold[h]["d_wmae"] <= 0)
    folds_nonworse_u = sum(1 for h in HOLDOUTS if per_fold[h]["d_umae"] <= 0)
    summary[v] = {
        "mean_d_wmae": float(d_w.mean()), "sd_d_wmae": float(d_w.std(ddof=1)),
        "mean_d_umae": float(d_u.mean()), "sd_d_umae": float(d_u.std(ddof=1)),
        "cells_negative_wmae": int((d_w < 0).sum()), "n_cells": len(d_w),
        "folds_nonworse_wmae": folds_nonworse_w, "folds_nonworse_umae": folds_nonworse_u,
        "mean_slope_pooled": float(sl_p.mean()), "mean_slope_centered": float(sl_c.mean()),
        "per_fold": per_fold,
        # the registered rule: mean delta <= 0 AND >=3 of 4 folds non-worse, on BOTH wmae and umae
        "adoptable_registered_rule": bool(d_w.mean() <= 0 and folds_nonworse_w >= 3
                                          and d_u.mean() <= 0 and folds_nonworse_u >= 3),
    }
    s = summary[v]
    log(f"  {v:32s} d_wmae={s['mean_d_wmae']:+.4f} (sd {s['sd_d_wmae']:.4f}, "
        f"{s['cells_negative_wmae']}/{s['n_cells']} better) d_umae={s['mean_d_umae']:+.4f} "
        f"folds_nonworse {s['folds_nonworse_wmae']}/4 (w) {s['folds_nonworse_umae']}/4 (u) "
        f"slope {s['mean_slope_pooled']:.4f}/{s['mean_slope_centered']:.4f} "
        f"=> {'ADOPTABLE' if s['adoptable_registered_rule'] else 'REJECTED'}")

base_sl_p = float(np.mean([r["variants"]["base"]["slope_pooled"] for r in records]))
base_sl_c = float(np.mean([r["variants"]["base"]["slope_bucket_centered"] for r in records]))
out["baseline_slopes"] = {"mean_slope_pooled": base_sl_p, "mean_slope_bucket_centered": base_sl_c}
out["summary"] = summary
log(f"  BASELINE mean slope: pooled={base_sl_p:.4f}  bucket-centered={base_sl_c:.4f}")

os_p = float(np.mean([r["r2_optimality"]["pooled"]["over_shrink_factor"] for r in records]))
os_c = float(np.mean([r["r2_optimality"]["bucket_centered"]["over_shrink_factor"] for r in records]))
out["mean_over_shrink_factor"] = {"pooled": os_p, "bucket_centered": os_c}
log(f"  MEAN OVER-SHRINK vs the MSE-optimal conditional mean: pooled x{os_p:.4f}  "
    f"bucket-centered x{os_c:.4f}")

out["wall_s"] = time.time() - t0
path = f"{ART}/k03_reconcile_fix.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
