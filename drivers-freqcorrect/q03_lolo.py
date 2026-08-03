"""Q03 — INVARIANT C: EXPLAIN THE +9.906 d_wmae, as a decomposition rather than a narrative.

A real 12-model LOLO (4 folds x 3 seeds) through the reviewed validate() path, reproducing CALIB-1's
k03 configuration exactly so my numbers are directly comparable.

N2   reproduce CALIB-1's practice_b mean d_wmae = +9.906533936853519 and its four per-fold values.
     Bar: within 0.10 absolute. THIS IS THE NEGATIVE CONTROL FOR THE WHOLE ARM.
A1/A2 the CORRECT three-way comparison. `_predict_cells` already adds b, so:
        g       = practice block STRIPPED  (geometry alone)
        g+b     = the harness/shipped path (what k03 called `base`)
        g+2b    = pred_path * exp(b)       (what k03 called `practice_b`)
     The sign of "does b help or hurt held-out magnitude" must be read off g vs g+b.
C1   LEVEL SHIFT: re-center b to zero mean and re-measure. Registered prediction: >=60% of the
     penalty is level.
C2   MIS-TRANSFER: split held-out cells by whether their ngram was SEEN in the held-in folds
     (unseen get b=0, so they are a built-in placebo). Registered prediction: the penalty
     concentrates on SEEN ngrams.
C3   handled in q01 (P2's R2).
S5   the qwerty-worst-fold MECHANISM CHECK (n=4, explicitly NOT a test): does the fold's
     calibration excess track its b-EXPOSURE or its SUPPORT?
"""
import copy
import json
import time

import numpy as np
from _guard import ART, BOOT_SEED, CELL_KW, HOLDOUTS, SEEDS, assert_d5, load_rows

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log("D5:")
assert_d5()

from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import (_predict_cells, build_cells, calibration_slope,  # noqa: E402
                                     leave_one_layout_out, uniform_mae, weighted_mae)
from keybo.verdicts import require_finite  # noqa: E402

G = ROW_STAGGERED_31
out = {"seeds": SEEDS, "holdouts": HOLDOUTS, "cell_kw": CELL_KW, "boot_seed": BOOT_SEED,
       "geometry": "ROW_STAGGERED_31"}

log("loading rows")
rows = load_rows()
log(f"  {len(rows)} rows; layouts {sorted({r.layout for r in rows})}")


def bucket_centered_slope(cells, pred, obs):
    """slope(obs~pred) after removing each wpm bucket's mean from BOTH sides (validate.py's
    _bucket_centered convention -- the STRUCTURAL slope a layout comparison consumes)."""
    from collections import defaultdict
    by = defaultdict(list)
    for i, c in enumerate(cells):
        by[c.bucket].append(i)
    p, o = np.asarray(pred, float).copy(), np.asarray(obs, float).copy()
    for idx in by.values():
        p[idx] -= p[idx].mean()
        o[idx] -= o[idx].mean()
    return calibration_slope(p, o)


records = []
for holdout in HOLDOUTS:
    train_rows, test_rows = leave_one_layout_out(rows, holdout)
    test_cells = build_cells(test_rows, **CELL_KW)
    train_cells = build_cells(train_rows, **CELL_KW)
    obs_te = np.array([c.obs for c in test_cells])
    seen_ngrams = {r.ngram for r in train_rows}
    for seed in SEEDS:
        log(f"fold {holdout} seed {seed}: training "
            f"(n_train_cells={len(train_cells)}, n_test_cells={len(test_cells)})")
        model = train_bigram_model(train_rows, target_wpm=90.0, geometry=G,
                                  random_state=seed, n_jobs=48)
        tr = model.metadata.extra["training"]
        bmap = (tr.get("practice_term") or {}).get("values", {})

        # ---- the three arms. g is the harness path with the practice block STRIPPED ------------
        m_strip = copy.deepcopy(model)
        m_strip.metadata.extra["training"] = {**tr, "practice_term": None}
        pred_g = _predict_cells(m_strip, test_cells, G)          # g alone
        pred_gb = _predict_cells(model, test_cells, G)           # g + b  == k03's `base`
        bt = np.array([bmap.get(c.ngram, 0.0) for c in test_cells], float)
        pred_g2b = pred_gb * np.exp(bt)                          # g + 2b == k03's `practice_b`
        require_finite(list(pred_g) + list(pred_gb) + list(pred_g2b), f"{holdout}/{seed} preds")

        # A1: is g+b really g*exp(b)? (the double-count discriminator, per fold)
        worst_recon = float(np.max(np.abs(pred_gb - pred_g * np.exp(bt))))

        # ---- C1: b RE-CENTERED to zero mean (the level-vs-structure discriminator) -------------
        bbar = float(np.mean(list(bmap.values()))) if bmap else 0.0
        # centered b applied ONCE on top of g  ->  g + (b - bbar)
        pred_g_bc = pred_g * np.exp(bt - bbar)
        # and the DOUBLED arm with b centered -> g+b then *exp(b-bbar)
        pred_g2b_c = pred_gb * np.exp(bt - bbar)

        # ---- C2: seen vs unseen ngrams in the held-out fold ------------------------------------
        is_seen = np.array([c.ngram in seen_ngrams for c in test_cells])
        has_b = np.array([c.ngram in bmap for c in test_cells])

        arms = {"g": pred_g, "g_plus_b": pred_gb, "g_plus_2b": pred_g2b,
                "g_plus_b_centered": pred_g_bc, "g_plus_2b_centered": pred_g2b_c}
        rec = {"holdout": holdout, "seed": seed,
               "n_test_cells": len(test_cells), "n_train_cells": len(train_cells),
               "n_b_ngrams": len(bmap), "mean_b": bbar,
               "sd_b": float(np.std(list(bmap.values()), ddof=1)) if len(bmap) > 1 else 0.0,
               "worst_abs_recon_gb_vs_g_expb": worst_recon,
               "frac_test_cells_seen": float(is_seen.mean()),
               "frac_test_cells_with_b": float(has_b.mean()),
               "arms": {}}
        for name, pv in arms.items():
            rec["arms"][name] = {
                "wmae": weighted_mae(test_cells, pv, obs_te),
                "umae": uniform_mae(pv, obs_te),
                "slope_pooled": calibration_slope(pv, obs_te),
                "slope_bucket_centered": bucket_centered_slope(test_cells, pv, obs_te),
            }
        # C2: wmae restricted to SEEN vs UNSEEN subsets (own cell lists so weights are right)
        for sub_name, mask in (("seen", is_seen), ("unseen", ~is_seen),
                              ("has_b", has_b), ("no_b", ~has_b)):
            if mask.sum() == 0:
                continue
            sub_cells = [c for c, m in zip(test_cells, mask) if m]
            rec.setdefault("subsets", {})[sub_name] = {
                "n": int(mask.sum()),
                "wmae_g": weighted_mae(sub_cells, pred_g[mask], obs_te[mask]),
                "wmae_g_plus_b": weighted_mae(sub_cells, pred_gb[mask], obs_te[mask]),
                "wmae_g_plus_2b": weighted_mae(sub_cells, pred_g2b[mask], obs_te[mask]),
                "corpus_mass_frac": float(sum(c.frequency for c in sub_cells)
                                          / sum(c.frequency for c in test_cells)),
            }
        # S5: b-EXPOSURE of this fold = corpus-weighted mean |b| over test cells
        w = np.array([c.frequency for c in test_cells], float)
        rec["b_exposure"] = {
            "wmean_abs_b": float((w * np.abs(bt)).sum() / w.sum()),
            "wmean_b": float((w * bt).sum() / w.sum()),
            "umean_abs_b": float(np.abs(bt).mean()),
        }
        log(f"  wmae  g={rec['arms']['g']['wmae']:.4f}  g+b={rec['arms']['g_plus_b']['wmae']:.4f}"
            f"  g+2b={rec['arms']['g_plus_2b']['wmae']:.4f}"
            f"  | slope_c g={rec['arms']['g']['slope_bucket_centered']:.4f}"
            f" g+b={rec['arms']['g_plus_b']['slope_bucket_centered']:.4f}"
            f" | recon {worst_recon:.2e}")
        records.append(rec)

out["records"] = records

# ================================================================ paired per-fold deltas (MOR-FIX-1)
log("PAIRED PER-FOLD DELTAS (MOR-FIX-1: paired deltas, never a mean of ratios)")


def deltas(arm, base):
    d_w = np.array([r["arms"][arm]["wmae"] - r["arms"][base]["wmae"] for r in records])
    d_u = np.array([r["arms"][arm]["umae"] - r["arms"][base]["umae"] for r in records])
    per_fold = {}
    for h in HOLDOUTS:
        sel = [i for i, r in enumerate(records) if r["holdout"] == h]
        per_fold[h] = {"d_wmae": float(d_w[sel].mean()), "d_umae": float(d_u[sel].mean())}
    return {"mean_d_wmae": float(d_w.mean()), "sd_d_wmae": float(d_w.std(ddof=1)),
            "mean_d_umae": float(d_u.mean()), "sd_d_umae": float(d_u.std(ddof=1)),
            "cells_negative_wmae": int((d_w < 0).sum()), "n_cells": len(d_w),
            "folds_nonworse_wmae": sum(1 for h in HOLDOUTS if per_fold[h]["d_wmae"] <= 0),
            "folds_nonworse_umae": sum(1 for h in HOLDOUTS if per_fold[h]["d_umae"] <= 0),
            "per_fold": per_fold}


cmp = {
    # THE CORRECTED QUESTION: does b help or hurt, at its FITTED magnitude?
    "b_ONCE__g_plus_b_vs_g": deltas("g_plus_b", "g"),
    # k03's arm, re-derived: b applied TWICE
    "b_TWICE__g_plus_2b_vs_g_plus_b": deltas("g_plus_2b", "g_plus_b"),
    "b_TWICE_vs_g__g_plus_2b_vs_g": deltas("g_plus_2b", "g"),
    # C1: the level-stripped versions
    "C1_centered_b_once__vs_g": deltas("g_plus_b_centered", "g"),
    "C1_centered_b_twice__vs_g_plus_b": deltas("g_plus_2b_centered", "g_plus_b"),
}
for k, v in cmp.items():
    log(f"  {k:38s} d_wmae={v['mean_d_wmae']:+8.4f} (sd {v['sd_d_wmae']:6.4f}, "
        f"{v['cells_negative_wmae']}/{v['n_cells']} better) d_umae={v['mean_d_umae']:+8.4f} "
        f"folds_nonworse {v['folds_nonworse_wmae']}/4 w, {v['folds_nonworse_umae']}/4 u")
out["comparisons"] = cmp

# ============================================================ N2 — reproduce CALIB-1's +9.906
pub_mean = 9.906533936853519
pub_fold = {"azerty": 12.565848804197481, "dvorak": 7.604408050330945,
            "qwerty": 6.223, "qwertz": 13.24}   # azerty/dvorak exact from k03; others from report
mine = cmp["b_TWICE__g_plus_2b_vs_g_plus_b"]
out["n2_neg_control"] = {
    "published_mean_d_wmae": pub_mean,
    "my_mean_d_wmae_same_arm": mine["mean_d_wmae"],
    "abs_diff": abs(mine["mean_d_wmae"] - pub_mean), "bar": 0.10,
    "passes": bool(abs(mine["mean_d_wmae"] - pub_mean) < 0.10),
    "published_per_fold": pub_fold,
    "my_per_fold": {h: mine["per_fold"][h]["d_wmae"] for h in HOLDOUTS},
    "note": ("reproduces k03's `practice_b` arm EXACTLY as k03 computed it, i.e. pred*exp(b) on a "
             "prediction that already contains b -- so a PASS here confirms the double-count "
             "rather than validating the arm's interpretation"),
}
log(f"N2: my g+2b-vs-g+b mean d_wmae = {mine['mean_d_wmae']:+.6f} vs published {pub_mean:+.6f} "
    f"|diff| = {abs(mine['mean_d_wmae'] - pub_mean):.4f} => "
    f"{'PASS' if abs(mine['mean_d_wmae'] - pub_mean) < 0.10 else 'FAIL'}")
log(f"   per fold mine {out['n2_neg_control']['my_per_fold']}")
log(f"   per fold pub  {pub_fold}")

# ================================================== C1 share-of-penalty attributable to the LEVEL
raw = cmp["b_TWICE__g_plus_2b_vs_g_plus_b"]["mean_d_wmae"]
cen = cmp["C1_centered_b_twice__vs_g_plus_b"]["mean_d_wmae"]
out["c1_level_share"] = {
    "d_wmae_b_twice": raw, "d_wmae_b_twice_centered": cen,
    "level_share_of_penalty": float((raw - cen) / raw) if raw else np.nan,
    "registered_prediction": ">=0.60 of the penalty is level",
    "prediction_held": bool(raw and (raw - cen) / raw >= 0.60),
}
log(f"C1: penalty {raw:+.4f} -> {cen:+.4f} when b is re-centered => level share "
    f"{out['c1_level_share']['level_share_of_penalty']:.3f} "
    f"({'prediction HELD' if out['c1_level_share']['prediction_held'] else 'prediction FAILED'})")

# ================================================== C2 seen vs unseen concentration
c2 = {}
for sub in ("seen", "unseen", "has_b", "no_b"):
    have = [r for r in records if sub in r.get("subsets", {})]
    if not have:
        c2[sub] = None
        continue
    c2[sub] = {
        "n_records": len(have),
        "mean_n_cells": float(np.mean([r["subsets"][sub]["n"] for r in have])),
        "mean_corpus_mass_frac": float(np.mean([r["subsets"][sub]["corpus_mass_frac"] for r in have])),
        "mean_d_wmae_b_once": float(np.mean([r["subsets"][sub]["wmae_g_plus_b"]
                                             - r["subsets"][sub]["wmae_g"] for r in have])),
        "mean_d_wmae_b_twice": float(np.mean([r["subsets"][sub]["wmae_g_plus_2b"]
                                              - r["subsets"][sub]["wmae_g_plus_b"] for r in have])),
    }
out["c2_seen_unseen"] = c2
for k, v in c2.items():
    if v:
        log(f"C2 {k:8s} n~{v['mean_n_cells']:7.1f} mass {v['mean_corpus_mass_frac']:.4f} "
            f"d_wmae(b once) {v['mean_d_wmae_b_once']:+.4f}  d_wmae(b twice) "
            f"{v['mean_d_wmae_b_twice']:+.4f}")

# ================================================== S5 qwerty-fold MECHANISM CHECK (n=4, NOT a test)
s5 = {}
for h in HOLDOUTS:
    sel = [r for r in records if r["holdout"] == h]
    s5[h] = {
        "slope_centered_g": float(np.mean([r["arms"]["g"]["slope_bucket_centered"] for r in sel])),
        "slope_centered_g_plus_b": float(np.mean([r["arms"]["g_plus_b"]["slope_bucket_centered"]
                                                  for r in sel])),
        "n_train_cells": sel[0]["n_train_cells"], "n_test_cells": sel[0]["n_test_cells"],
        "b_exposure_wmean_abs": float(np.mean([r["b_exposure"]["wmean_abs_b"] for r in sel])),
        "b_exposure_wmean": float(np.mean([r["b_exposure"]["wmean_b"] for r in sel])),
        "frac_seen": float(np.mean([r["frac_test_cells_seen"] for r in sel])),
        "train_test_ratio": sel[0]["n_train_cells"] / sel[0]["n_test_cells"],
    }
xs_ex = np.array([s5[h]["b_exposure_wmean_abs"] for h in HOLDOUTS])
xs_sup = np.array([s5[h]["train_test_ratio"] for h in HOLDOUTS])
ys_sl = np.array([s5[h]["slope_centered_g_plus_b"] for h in HOLDOUTS])
out["s5_qwerty_fold"] = {
    "per_fold": s5, "n_folds": 4,
    "corr_slope_vs_b_exposure": float(np.corrcoef(xs_ex, ys_sl)[0, 1]),
    "corr_slope_vs_support_ratio": float(np.corrcoef(xs_sup, ys_sl)[0, 1]),
    "EXPLICIT_CAVEAT": ("n = 4 folds. A correlation over 4 points is NOT a test; reported as a "
                        "MECHANISM CHECK only. If the two candidate explanations are "
                        "indistinguishable at n=4 the verdict is COINCIDENTAL-OR-UNRESOLVED."),
}
log(f"S5 (n=4, mechanism check only): corr(slope_c, b-exposure) = "
    f"{out['s5_qwerty_fold']['corr_slope_vs_b_exposure']:+.4f}; "
    f"corr(slope_c, train/test ratio) = {out['s5_qwerty_fold']['corr_slope_vs_support_ratio']:+.4f}")
for h in HOLDOUTS:
    log(f"   {h:8s} slope_c(g)={s5[h]['slope_centered_g']:.4f} "
        f"slope_c(g+b)={s5[h]['slope_centered_g_plus_b']:.4f} "
        f"train/test={s5[h]['train_test_ratio']:.2f} b-expo={s5[h]['b_exposure_wmean_abs']:.4f} "
        f"seen={s5[h]['frac_seen']:.3f}")

out["wall_s"] = time.time() - t0
path = f"{ART}/q03_lolo.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
