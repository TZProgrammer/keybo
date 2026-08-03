"""Q01 — NEGATIVE CONTROL N1, then H-SATURATED (P1/P2/P3) and the A1 double-count discriminator.

N1  reproduce FREQGEO-1's B(candidate) = -0.12673429794113286 from my own code path (bar 1e-9).
P1  within a layout, is ngram -> positions a BIJECTION? (if yes, a per-ngram intercept is an
    unconstrained function of geometry within that layout)
P2  how much of the fitted `b` is PREDICTABLE FROM GEOMETRY ALONE, out-of-fold? Registered
    threshold: R2 >= 0.30 => `b` is materially geometric => the "practice" label is wrong for
    that share. R2 < 0.10 => H-SATURATED refuted.
P3  does the geometric share survive the k=100 shrinkage?
N3  the PLACEBO practice term: `b` refitted on SHUFFLED ngram labels. Must show near-zero
    geometric R2, or my P2 instrument is measuring shrinkage arithmetic rather than contamination.
A1  is CALIB-1's `practice_b` arm a DOUBLE-COUNT? `_predict_cells` already adds b, so compare a
    full-path prediction against g-alone + b recomputed by hand. Bar: worst |diff| < 1e-6 ms.
"""
import json
import time

import numpy as np
from _guard import ART, BOOT_SEED, CELL_KW, SEEDS, assert_d5, load_rows

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log("D5:")
assert_d5()

from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import fit_practice_term, train_bigram_model  # noqa: E402
from keybo.training.validate import _predict_cells, build_cells  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
out = {"cell_kw": CELL_KW, "boot_seed": BOOT_SEED, "geometry": "ROW_STAGGERED_31",
       "served_feature_names": list(BIGRAM_FEATURE_NAMES),
       "n_served_features": len(BIGRAM_FEATURE_NAMES)}
log(f"served frame: {len(BIGRAM_FEATURE_NAMES)} columns, log_freq present = "
    f"{'log_freq' in BIGRAM_FEATURE_NAMES}")

log("loading rows")
rows = load_rows()
log(f"  {len(rows)} rows; layouts {sorted({r.layout for r in rows})}")

# ============================================================ P1 — bijection ngram -> geometry
from collections import defaultdict  # noqa: E402

by_ln = defaultdict(set)
by_n_geom = defaultdict(set)
by_n_lay = defaultdict(set)
for r in rows:
    by_ln[(r.layout, r.ngram)].add(tuple(r.positions))
    by_n_geom[r.ngram].add(tuple(r.positions))
    by_n_lay[r.ngram].add(r.layout)
max_geom_per_layout_ngram = max(len(v) for v in by_ln.values())
geom_hist = np.bincount([len(v) for v in by_n_geom.values()]).tolist()
lay_hist = np.bincount([len(v) for v in by_n_lay.values()]).tolist()
out["p1_bijection"] = {
    "max_geometries_per_layout_ngram": int(max_geom_per_layout_ngram),
    "bijection_holds": bool(max_geom_per_layout_ngram == 1),
    "n_distinct_ngrams": len(by_n_geom),
    "ngram_distinct_geometry_count_hist": geom_hist,
    "ngram_layout_count_hist": lay_hist,
    "n_ngrams_single_geometry": int(geom_hist[1]) if len(geom_hist) > 1 else 0,
    "frac_ngrams_single_geometry": float(geom_hist[1] / len(by_n_geom)) if len(geom_hist) > 1 else 0.0,
}
log(f"P1: max geometries per (layout,ngram) = {max_geom_per_layout_ngram} => bijection "
    f"{'HOLDS' if max_geom_per_layout_ngram == 1 else 'REFUTED'}")
log(f"  ngram distinct-geometry hist {geom_hist}; layout-count hist {lay_hist}")

# ==================================================== train the FULL-DATA model (b to audit)
log("training full-data bigram model (seed 0) to obtain the fitted b")
model_full = train_bigram_model(rows, target_wpm=90.0, geometry=G, random_state=0, n_jobs=48)
tr = model_full.metadata.extra["training"]
bmap_full = tr["practice_term"]["values"]
out["b_full"] = {"n_ngrams": tr["practice_term"]["n_ngrams"],
                 "shrinkage_k": tr["practice_term"]["shrinkage_k"],
                 "backfit_iters": tr["practice_term"]["backfit_iters"],
                 "target_space": tr["target_space"],
                 "mean_b": float(np.mean(list(bmap_full.values()))),
                 "sd_b": float(np.std(list(bmap_full.values()), ddof=1)),
                 "min_b": float(min(bmap_full.values())), "max_b": float(max(bmap_full.values()))}
log(f"  b: n={len(bmap_full)} mean={out['b_full']['mean_b']:.6f} sd={out['b_full']['sd_b']:.6f} "
    f"range [{out['b_full']['min_b']:.4f}, {out['b_full']['max_b']:.4f}]")

# ============================================================ N1 — reproduce FREQGEO-1's B
# B = the FREQUENCY-WEIGHTED mean of b over the ngrams a board can type, using the trigram
# first-transition marginal weighting freqgeo used, over the SEED-MEAN b (its stated recipe).
log("N1: reproducing FREQGEO-1's B(candidate)")
bmaps = [bmap_full]
for s in (1, 2):
    m = train_bigram_model(rows, target_wpm=90.0, geometry=G, random_state=s, n_jobs=48)
    bmaps.append(m.metadata.extra["training"]["practice_term"]["values"])
allng = set().union(*[set(b) for b in bmaps])
b_seedmean = {ng: float(np.mean([b.get(ng, 0.0) for b in bmaps])) for ng in allng}
out["b_seedmean"] = {"n_ngrams": len(b_seedmean),
                     "mean": float(np.mean(list(b_seedmean.values())))}

d = production_corpus_dir(None)
tri = {k: v for k, v in load_frequencies(str(d / "trigrams.txt")).items() if len(k) == 3}
from keybo.layouts import LAYOUTS  # noqa: E402

cand = LAYOUTS["candidate"] if "candidate" in LAYOUTS else None
out["n1_layout_keys_sample"] = sorted(LAYOUTS)[:40]


def B_of_board(lay30, bm):
    """freq-weighted mean b over the corpus trigrams this board can type, weighting each
    trigram by its frequency and keying b on the FIRST TRANSITION (a->b) -- freqgeo's
    trigram-marginal convention, which is the one that reproduced calib's published value."""
    chars = set(lay30) | {" "}
    num = 0.0
    den = 0.0
    for ng, f in tri.items():
        if not set(ng) <= chars:
            continue
        num += f * bm.get(ng[:2], 0.0)
        den += f
    return num / den, den


if cand is not None:
    B_cand, mass = B_of_board(cand, b_seedmean)
    published = -0.12673429794113286
    out["n1_neg_control"] = {"B_candidate": B_cand, "published": published,
                             "abs_diff": abs(B_cand - published), "bar": 1e-9,
                             "passes": bool(abs(B_cand - published) < 1e-9)}
    log(f"N1: B(candidate) = {B_cand:.17f} vs published {published:.17f} "
        f"|diff| = {abs(B_cand - published):.3e} => "
        f"{'PASS' if abs(B_cand - published) < 1e-9 else 'FAIL'}")
else:
    out["n1_neg_control"] = {"error": "candidate not in LAYOUTS"}
    log("N1: candidate NOT FOUND in LAYOUTS")

# ================================================== P2 — is b predictable from GEOMETRY alone?
# One row per distinct (layout, ngram): the served 19 non-wpm geometric columns -> b.
# Out-of-fold by GroupKFold on the ngram, so an ngram never trains and tests together.
log("P2: regressing b on the SERVED geometric frame, out-of-fold")
WPM_I = BIGRAM_FEATURE_NAMES.index("wpm")
GEO_COLS = [i for i in range(len(BIGRAM_FEATURE_NAMES)) if i != WPM_I]
seen = {}
for r in rows:
    seen.setdefault((r.layout, r.ngram), r)
keys = sorted(seen)
Xg = np.vstack([bigram_features_from_positions(G, seen[k].positions, wpm=90.0)[GEO_COLS]
                for k in keys])
yb = np.array([b_seedmean.get(k[1], 0.0) for k in keys], float)
ng_of = np.array([k[1] for k in keys], dtype=object)
lay_of = np.array([k[0] for k in keys], dtype=object)
require_finite(list(yb), "b targets")
log(f"  {Xg.shape[0]} (layout,ngram) rows x {Xg.shape[1]} geometric columns")

from sklearn.model_selection import GroupKFold  # noqa: E402


def oof_r2(X, y, groups, n_splits=5, seed=0, model="gbm"):
    """Out-of-fold R2, grouped so an ngram never appears in both train and test."""
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_i, te_i in gkf.split(X, y, groups):
        if model == "gbm":
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 n_jobs=48, random_state=seed)
            m.fit(X[tr_i], y[tr_i])
            pred[te_i] = m.predict(X[te_i])
        else:
            A = np.column_stack([X[tr_i], np.ones(len(tr_i))])
            coef, *_ = np.linalg.lstsq(A, y[tr_i], rcond=None)
            pred[te_i] = np.column_stack([X[te_i], np.ones(len(te_i))]) @ coef
    require_finite(list(pred), f"oof pred ({model})")
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot, pred


r2_gbm, pred_gbm = oof_r2(Xg, yb, ng_of, model="gbm")
r2_ols, pred_ols = oof_r2(Xg, yb, ng_of, model="ols")
# in-sample GBM too, as the upper bound a saturated learner could reach
import xgboost as xgb  # noqa: E402

m_in = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, n_jobs=48, random_state=0)
m_in.fit(Xg, yb)
p_in = m_in.predict(Xg)
r2_in = 1.0 - float(((yb - p_in) ** 2).sum()) / float(((yb - yb.mean()) ** 2).sum())
out["p2_b_on_geometry"] = {
    "n_rows": int(Xg.shape[0]), "n_geo_cols": int(Xg.shape[1]),
    "R2_oof_GBM": r2_gbm, "R2_oof_OLS": r2_ols, "R2_insample_GBM": r2_in,
    "registered_threshold_geometric": 0.30, "registered_threshold_refuted": 0.10,
    "verdict": ("MATERIALLY GEOMETRIC" if r2_gbm >= 0.30
                else "H-SATURATED REFUTED" if r2_gbm < 0.10 else "INTERMEDIATE"),
    "sd_b_rows": float(np.std(yb, ddof=1)),
    "sd_geometric_component": float(np.std(pred_gbm, ddof=1)),
}
log(f"P2: R2(b ~ geometry) oof GBM = {r2_gbm:.4f}  oof OLS = {r2_ols:.4f}  "
    f"in-sample GBM = {r2_in:.4f} => {out['p2_b_on_geometry']['verdict']}")

# ================================================== N3 — the PLACEBO practice term (shuffled ngrams)
log("N3: placebo practice term (shuffled ngram labels), same shrinkage and counts")
from keybo.training.train import _build_matrix_full  # noqa: E402

X_all, y_all, ngrams_all, layouts_all, counts_all = _build_matrix_full(
    rows, ngram="bigram", geometry=G, target_space="LOGRAT")
log(f"  training matrix {X_all.shape}")
# the REAL residual the real b was fit on, from a b-free model
m_g0 = train_bigram_model(rows, target_wpm=90.0, geometry=G, random_state=0, n_jobs=48,
                          practice_term=False)
resid = y_all - m_g0.predict(X_all)
rng = np.random.default_rng(BOOT_SEED)
perm = rng.permutation(len(ngrams_all))
b_placebo = fit_practice_term(ngrams_all[perm], resid, counts_all)
yb_pl = np.array([b_placebo.get(k[1], 0.0) for k in keys], float)
r2_pl_gbm, _ = oof_r2(Xg, yb_pl, ng_of, model="gbm")
r2_pl_ols, _ = oof_r2(Xg, yb_pl, ng_of, model="ols")
out["n3_placebo"] = {
    "n_ngrams": len(b_placebo), "mean_b": float(np.mean(list(b_placebo.values()))),
    "sd_b": float(np.std(list(b_placebo.values()), ddof=1)),
    "R2_oof_GBM": r2_pl_gbm, "R2_oof_OLS": r2_pl_ols,
    "passes_near_zero": bool(r2_pl_gbm < 0.10),
}
log(f"N3 placebo: R2(b_placebo ~ geometry) oof GBM = {r2_pl_gbm:.4f} OLS = {r2_pl_ols:.4f} "
    f"=> {'PASS (near-zero)' if r2_pl_gbm < 0.10 else 'FAIL -- instrument suspect'}")

# also: the REAL b fit through my own code path on the SAME residual, as a same-pipeline control
b_real_same = fit_practice_term(ngrams_all, resid, counts_all)
yb_rs = np.array([b_real_same.get(k[1], 0.0) for k in keys], float)
r2_rs_gbm, _ = oof_r2(Xg, yb_rs, ng_of, model="gbm")
out["n3_real_same_pipeline"] = {"R2_oof_GBM": r2_rs_gbm, "n_ngrams": len(b_real_same),
                               "mean_b": float(np.mean(list(b_real_same.values())))}
log(f"  same-pipeline REAL b (one pass, b-free g): R2 oof GBM = {r2_rs_gbm:.4f}")

# ================================================== P3 — does the geometric share survive shrinkage?
log("P3: geometric R2 of b across shrinkage k")
p3 = {}
for k in (0.0, 10.0, 100.0, 1000.0, 10000.0):
    bk = fit_practice_term(ngrams_all, resid, counts_all, k=k)
    ybk = np.array([bk.get(kk[1], 0.0) for kk in keys], float)
    r2k, _ = oof_r2(Xg, ybk, ng_of, model="gbm")
    p3[str(k)] = {"R2_oof_GBM": r2k, "sd_b": float(np.std(list(bk.values()), ddof=1)),
                  "mean_b": float(np.mean(list(bk.values())))}
    log(f"  k={k:>8.0f}  R2={r2k:.4f}  sd(b)={p3[str(k)]['sd_b']:.6f}")
out["p3_shrinkage"] = p3

# ================================================== A1 — the DOUBLE-COUNT discriminator
log("A1: is CALIB-1's practice_b arm a double-count?")
cells_all = build_cells(rows, **CELL_KW)
pred_path = _predict_cells(model_full, cells_all, G)          # the harness path
# g-alone: same model, practice block stripped from a COPY of the metadata
import copy  # noqa: E402

m_strip = copy.deepcopy(model_full)
m_strip.metadata.extra["training"] = {**tr, "practice_term": None}
pred_g = _predict_cells(m_strip, cells_all, G)
bt = np.array([bmap_full.get(c.ngram, 0.0) for c in cells_all], float)
recon = pred_g * np.exp(bt)
worst = float(np.max(np.abs(pred_path - recon)))
n_bhit = int(np.sum([c.ngram in bmap_full for c in cells_all]))
out["a1_double_count"] = {
    "n_cells": len(cells_all),
    "n_cells_with_b": n_bhit,
    "frac_cells_with_b": float(n_bhit / len(cells_all)),
    "worst_abs_diff_pred_path_vs_g_times_expb": worst,
    "bar": 1e-6,
    "double_count_confirmed": bool(worst < 1e-6),
    "mean_pred_path_ms": float(pred_path.mean()),
    "mean_pred_g_ms": float(pred_g.mean()),
    "mean_ratio_path_over_g": float((pred_path / pred_g).mean()),
    "practice_block_present_in_metadata": bool(tr.get("practice_term") is not None),
    "n_ngrams_in_metadata": int(tr["practice_term"]["n_ngrams"]),
}
log(f"A1: worst |pred_path - g*exp(b)| = {worst:.3e} ms (bar 1e-6) => DOUBLE-COUNT "
    f"{'CONFIRMED' if worst < 1e-6 else 'REFUTED'}")
log(f"  cells {len(cells_all)}, {n_bhit} carry a b ({100 * n_bhit / len(cells_all):.1f}%); "
    f"mean pred path {pred_path.mean():.3f} ms vs g-alone {pred_g.mean():.3f} ms")

out["wall_s"] = time.time() - t0
path = f"{ART}/q01_saturated.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
