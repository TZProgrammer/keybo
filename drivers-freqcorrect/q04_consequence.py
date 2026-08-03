"""Q04 — INVARIANT D/E: the FALSIFIABLE CONSEQUENCE, and what a correct decomposition changes.

The brief's question 5 asks for something a correct decomposition would predict that the current one
does not, tested rather than argued. THE CONSEQUENCE I TEST:

  If `b` is genuinely a LAYOUT-INDEPENDENT practice term, then the SAME ngram typed at DIFFERENT
  geometry on a DIFFERENT layout must carry the SAME practice component. A per-ngram intercept
  fitted on pooled data cannot distinguish "this ngram is practised" from "this ngram sits on fast
  keys in the layout that dominates the data" -- so the falsifiable prediction is:

  H0 (decomposition CORRECT): b estimated from QWERTY-only data and b estimated from NON-QWERTY-only
     data agree, up to noise. They are estimates of the same layout-independent quantity.
  H1 (decomposition CONTAMINATED): they DISAGREE systematically, and the disagreement is PREDICTABLE
     FROM THE GEOMETRY DIFFERENCE between where the ngram sits in the two populations.

  H1 is the sharp one: a pure practice term has no reason to differ, and certainly no reason for its
  difference to track a geometric difference. This is a DIRECT test of "is g still carrying
  qwerty-specific muscle memory" (the brief's question 1) that does not depend on ranking at all.

  MEASURED FLOOR (registered §9): the floor for "do two b estimates agree?" is the spread between
  two estimates that differ ONLY by resampling noise -- so I split the QWERTY data in half and
  compare b_half1 vs b_half2 at matched sample sizes. That is the design-matched floor, and it is
  measured here rather than borrowed.

Also computed: the ABSOLUTE ms/char consequence (INVARIANT D) of the geometric share of b, and the
hours-per-year translation, both stated with the arithmetic exposed.
"""
import json
import time
from collections import defaultdict

import numpy as np
from _guard import ART, BOOT_SEED, CELL_KW, SHIPPED, assert_d5, load_rows

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log("D5:")
assert_d5()

from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import _build_matrix_full, fit_practice_term, train_bigram_model  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
WPM_I = BIGRAM_FEATURE_NAMES.index("wpm")
GEO_COLS = [i for i in range(len(BIGRAM_FEATURE_NAMES)) if i != WPM_I]
out = {"cell_kw": CELL_KW, "boot_seed": BOOT_SEED, "geometry": "ROW_STAGGERED_31"}

log("loading rows")
rows = load_rows()
qrows = [r for r in rows if r.layout == "qwerty"]
nqrows = [r for r in rows if r.layout != "qwerty"]
log(f"  {len(rows)} rows: qwerty {len(qrows)}, non-qwerty {len(nqrows)}")


def fit_b_on(subset, seed=0, k=100.0):
    """Fit a b on `subset` using the SHIPPED recipe: a b-free g, then the shrunk residual mean.

    One backfit pass (not two) so every arm here is the same estimator; the two-pass shipped
    version is reproduced separately by train_bigram_model in the cross-checks below.
    """
    X, y, ngrams, layouts, counts = _build_matrix_full(subset, ngram="bigram", geometry=G,
                                                       target_space="LOGRAT")
    m = train_bigram_model(subset, target_wpm=90.0, geometry=G, random_state=seed, n_jobs=48,
                           practice_term=False)
    resid = y - m.predict(X)
    return fit_practice_term(ngrams, resid, counts, k=k), ngrams, counts


log("fitting b on QWERTY-only and on NON-QWERTY-only")
b_q, ng_q, cnt_q = fit_b_on(qrows)
b_nq, ng_nq, cnt_nq = fit_b_on(nqrows)
shared = sorted(set(b_q) & set(b_nq))
log(f"  b_qwerty n={len(b_q)}  b_nonqwerty n={len(b_nq)}  SHARED n={len(shared)}")

vq = np.array([b_q[ng] for ng in shared])
vnq = np.array([b_nq[ng] for ng in shared])
require_finite(list(vq) + list(vnq), "b_q / b_nq on shared ngrams")
diff = vnq - vq
r_agree = float(np.corrcoef(vq, vnq)[0, 1])
out["h1_transfer"] = {
    "n_shared_ngrams": len(shared),
    "corr_b_qwerty_vs_b_nonqwerty": r_agree,
    "mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)),
    "rms_diff": float(np.sqrt((diff ** 2).mean())),
    "sd_b_qwerty": float(vq.std(ddof=1)), "sd_b_nonqwerty": float(vnq.std(ddof=1)),
    "slope_nq_on_q": float(np.polyfit(vq, vnq, 1)[0]),
}
log(f"  corr(b_qwerty, b_nonqwerty) = {r_agree:.4f}  rms diff = "
    f"{out['h1_transfer']['rms_diff']:.6f}  sd(b_q) = {vq.std(ddof=1):.6f}")

# ============================================== the MEASURED FLOOR: split-half of QWERTY, matched n
log("MEASURED FLOOR: split-half of the QWERTY data (design-matched, truth = same quantity)")
# ⚠ THE SPLIT MUST BE OVER SAMPLES WITHIN EACH ROW, NOT OVER ROWS. A StrokeRow is unique per
# (layout, ngram) -- P1's bijection -- so splitting the ROW LIST gives the two halves DISJOINT
# ngram sets and `set(b1) & set(b2)` is EMPTY BY CONSTRUCTION. The first version of this driver
# did exactly that, skipped all 40 splits on its own `len(sh) < 20` guard, and died on
# np.percentile of an empty list. Splitting each row's SAMPLES keeps every ngram in both halves,
# which is what "the two halves estimate the SAME quantity" requires.
import copy as _copy  # noqa: E402

rng = np.random.default_rng(BOOT_SEED)
NS = 12          # each split costs 2 b-fits (~23 s); 12 gives a usable floor distribution
floor_r = []
floor_rms = []


def split_samples(subset, rng):
    """Two row-lists with the SAME ngrams, each carrying a random half of every row's samples."""
    h1, h2 = [], []
    for r in subset:
        if len(r.samples) < 4:          # too thin to halve into two estimable cells
            continue
        idx = rng.permutation(len(r.samples))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
        r1, r2 = _copy.copy(r), _copy.copy(r)
        r1.samples = [r.samples[j] for j in a]
        r2.samples = [r.samples[j] for j in b]
        h1.append(r1)
        h2.append(r2)
    return h1, h2


for i in range(NS):
    h1, h2 = split_samples(qrows, rng)
    b1, _, _ = fit_b_on(h1, seed=i)
    b2, _, _ = fit_b_on(h2, seed=i)
    sh = sorted(set(b1) & set(b2))
    if len(sh) < 20:
        log(f"  split {i}: SKIPPED, only {len(sh)} shared ngrams")
        continue
    a = np.array([b1[n] for n in sh])
    bb = np.array([b2[n] for n in sh])
    floor_r.append(float(np.corrcoef(a, bb)[0, 1]))
    floor_rms.append(float(np.sqrt(((bb - a) ** 2).mean())))
    log(f"  split {i}: n_shared={len(sh)} corr={floor_r[-1]:.4f} rms={floor_rms[-1]:.6f}")
assert floor_r, "FLOOR EMPTY -- every split was skipped; refusing to publish a floorless comparison"
floor_r = np.array(floor_r)
floor_rms = np.array(floor_rms)
out["measured_floor_split_half_qwerty"] = {
    "n_splits": int(len(floor_r)),
    "corr_mean": float(floor_r.mean()), "corr_sd": float(floor_r.std(ddof=1)),
    "corr_p05": float(np.percentile(floor_r, 5)),
    "rms_mean": float(floor_rms.mean()), "rms_sd": float(floor_rms.std(ddof=1)),
    "rms_p95": float(np.percentile(floor_rms, 95)),
    "design": ("random halves of the QWERTY rows, b refit on each half by the same estimator. "
               "TRUTH = the two halves estimate the SAME quantity, so any disagreement is pure "
               "sampling noise. This is the design-matched floor for 'do two b estimates agree?'"),
}
log(f"FLOOR: split-half corr {floor_r.mean():.4f} +- {floor_r.std(ddof=1):.4f} "
    f"(p05 {np.percentile(floor_r, 5):.4f}); rms {floor_rms.mean():.6f} "
    f"(p95 {np.percentile(floor_rms, 95):.6f})")
out["h1_transfer"]["corr_vs_floor"] = {
    "cross_layout_corr": r_agree, "same_layout_floor_corr_mean": float(floor_r.mean()),
    "gap": float(floor_r.mean() - r_agree),
    "cross_layout_rms": out["h1_transfer"]["rms_diff"],
    "same_layout_floor_rms_mean": float(floor_rms.mean()),
    "rms_ratio_cross_over_floor": float(out["h1_transfer"]["rms_diff"] / floor_rms.mean()),
    "verdict": ("DISAGREE BEYOND NOISE (contamination)" if r_agree < np.percentile(floor_r, 5)
                else "AGREE WITHIN NOISE (transfer OK)"),
}
log(f"  ⇒ cross-layout corr {r_agree:.4f} vs same-layout floor p05 "
    f"{np.percentile(floor_r, 5):.4f} => {out['h1_transfer']['corr_vs_floor']['verdict']}")

# ================================= H1's SHARP half: is the DISAGREEMENT predictable from GEOMETRY?
log("H1 sharp: is (b_nonqwerty - b_qwerty) predictable from the GEOMETRY DIFFERENCE?")
# for each shared ngram: qwerty geometry, and the mean non-qwerty geometry it occupies
geo_q = {}
geo_nq = defaultdict(list)
for r in rows:
    v = bigram_features_from_positions(G, r.positions, wpm=90.0)[GEO_COLS]
    if r.layout == "qwerty":
        geo_q[r.ngram] = v
    else:
        geo_nq[r.ngram].append(v)
use = [ng for ng in shared if ng in geo_q and ng in geo_nq]
Xq = np.vstack([geo_q[ng] for ng in use])
Xnq = np.vstack([np.mean(geo_nq[ng], axis=0) for ng in use])
Xd = Xnq - Xq
yd = np.array([b_nq[ng] - b_q[ng] for ng in use])
log(f"  {len(use)} ngrams with both geometries; geometry-difference matrix {Xd.shape}")

from sklearn.model_selection import GroupKFold  # noqa: E402
import xgboost as xgb  # noqa: E402


def oof_r2(X, y, groups, n_splits=5, seed=0, kind="gbm"):
    pred = np.full(len(y), np.nan)
    for tr_i, te_i in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if kind == "gbm":
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 n_jobs=48, random_state=seed)
            m.fit(X[tr_i], y[tr_i])
            pred[te_i] = m.predict(X[te_i])
        else:
            A = np.column_stack([X[tr_i], np.ones(len(tr_i))])
            coef, *_ = np.linalg.lstsq(A, y[tr_i], rcond=None)
            pred[te_i] = np.column_stack([X[te_i], np.ones(len(te_i))]) @ coef
    require_finite(list(pred), f"oof {kind}")
    return 1.0 - float(((y - pred) ** 2).sum()) / float(((y - y.mean()) ** 2).sum()), pred


grp = np.array(use, dtype=object)
r2_d_gbm, _ = oof_r2(Xd, yd, grp, kind="gbm")
r2_d_ols, _ = oof_r2(Xd, yd, grp, kind="ols")
# PLACEBO: shuffle the pairing between geometry-difference and b-difference
rngp = np.random.default_rng(BOOT_SEED + 7)
r2_pl = []
for _ in range(20):
    r2p, _ = oof_r2(Xd, yd[rngp.permutation(len(yd))], grp, kind="gbm")
    r2_pl.append(r2p)
out["h1_sharp_geometry_predicts_disagreement"] = {
    "n_ngrams": len(use),
    "R2_oof_GBM": r2_d_gbm, "R2_oof_OLS": r2_d_ols,
    "placebo_R2_mean": float(np.mean(r2_pl)), "placebo_R2_p95": float(np.percentile(r2_pl, 95)),
    "exceeds_placebo": bool(r2_d_gbm > np.percentile(r2_pl, 95)),
    "interpretation": ("a PURE practice term has no reason for its cross-layout disagreement to "
                       "track a GEOMETRY difference; R2 above the placebo means the 'practice' "
                       "estimate is absorbing layout-specific geometry"),
}
log(f"  R2(delta_b ~ delta_geometry) oof GBM = {r2_d_gbm:.4f} OLS = {r2_d_ols:.4f}; "
    f"placebo p95 = {np.percentile(r2_pl, 95):.4f} => "
    f"{'EXCEEDS PLACEBO' if r2_d_gbm > np.percentile(r2_pl, 95) else 'within placebo'}")

# ============================================== INVARIANT D: the ABSOLUTE ms/char consequence
log("INVARIANT D: absolute ms/char and hours-per-year arithmetic")
import gzip  # noqa: E402

b_sh = [json.loads(gzip.open(f"{SHIPPED}/bigram_reg31_seed{s}.meta.json.gz").read())
        ["extra"]["training"]["practice_term"]["values"] for s in (0, 1, 2)]
allk = set().union(*[set(b) for b in b_sh])
b_ship = {k: float(np.mean([b.get(k, 0.0) for b in b_sh])) for k in allk}

from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

d = production_corpus_dir(None)
tri = {k: v for k, v in load_frequencies(str(d / "trigrams.txt")).items() if len(k) == 3}
bi_marg = defaultdict(int)
for ng, f in tri.items():
    bi_marg[ng[:2]] += f
CANDIDATE = "pyu.,vdfnlhieaocstrmkj'-qgwbzx"
QWERTY = "qwertyuiopasdfghjkl;zxcvbnm,./"


def B_of(board, bm):
    ch = set(board) | {" "}
    c = {ng: f for ng, f in bi_marg.items() if all(x in ch for x in ng)}
    den = sum(c.values())
    return sum(f * bm.get(ng, 0.0) for ng, f in c.items()) / den if den else 0.0


# the geometric share of b, at the SHIPPED b, measured the same way q01 does
seen1 = {}
for r in rows:
    seen1.setdefault((r.layout, r.ngram), r)
keys1 = sorted(seen1)
Xg1 = np.vstack([bigram_features_from_positions(G, seen1[k].positions, wpm=90.0)[GEO_COLS]
                 for k in keys1])
yb1 = np.array([b_ship.get(k[1], 0.0) for k in keys1], float)
r2_ship, pred_ship = oof_r2(Xg1, yb1, np.array([k[1] for k in keys1], dtype=object), kind="gbm")
# b split into its geometric projection and the residual "true practice" part
b_geo_part = {}
b_prac_part = {}
for i, k in enumerate(keys1):
    b_geo_part[k[1]] = float(pred_ship[i])
    b_prac_part[k[1]] = float(yb1[i] - pred_ship[i])
B_cand_full = B_of(CANDIDATE, b_ship)
B_cand_geo = B_of(CANDIDATE, b_geo_part)
B_qwer_full = B_of(QWERTY, b_ship)
B_qwer_geo = B_of(QWERTY, b_geo_part)
out["invariant_d_magnitude"] = {
    "R2_oof_b_shipped_on_geometry": r2_ship,
    "B_candidate_full_b": B_cand_full, "B_candidate_geometric_part": B_cand_geo,
    "B_qwerty_full_b": B_qwer_full, "B_qwerty_geometric_part": B_qwer_geo,
    "geometric_share_of_B_candidate": float(B_cand_geo / B_cand_full) if B_cand_full else None,
    "cross_coverage_residual_full": float(B_cand_full - B_qwer_full),
    "cross_coverage_residual_geometric": float(B_cand_geo - B_qwer_geo),
    "note": ("B is a log-ratio, so exp(B) is the multiplicative level. The share of B that is "
             "PREDICTABLE FROM GEOMETRY is the part mislabelled 'practice'."),
    "exp_B_candidate_full": float(np.exp(B_cand_full)),
    "exp_B_candidate_geometric": float(np.exp(B_cand_geo)),
}
log(f"  R2(b_shipped ~ geometry) oof = {r2_ship:.4f}")
log(f"  B(candidate) full {B_cand_full:.8f}  geometric part {B_cand_geo:.8f}  "
    f"share {B_cand_geo / B_cand_full if B_cand_full else float('nan'):.4f}")
log(f"  B(qwerty)    full {B_qwer_full:.8f}  geometric part {B_qwer_geo:.8f}")
log(f"  cross-coverage residual: full {B_cand_full - B_qwer_full:+.8f} log  "
    f"geometric-only {B_cand_geo - B_qwer_geo:+.8f} log")

out["wall_s"] = time.time() - t0
path = f"{ART}/q04_consequence.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
