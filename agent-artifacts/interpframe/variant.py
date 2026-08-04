"""INTERPFRAME-1 §11 — the interp-wpm VARIANT: does restoring `wpm` recover the high-wpm gate?

Registered AFTER the first result, and the ordering is stated rather than hidden: the 10-column
frame won every interpretability bar but was REFUSED by the high-wpm non-regression gate
structurally, in the b100/b120 buckets, on all four folds. That refusal traces to exactly one
design choice — dropping `wpm` — so the variant isolates it.

THE TRADE, registered before this runs:
  * interp-wpm CANNOT win M2 (CONSTFRAC). `wpm` is constant on any fixed-WPM serve grid, so
    TreeSHAP can credit it again. If CONSTFRAC comes back > 0, that is the EXPECTED result and not
    a failure of the variant.
  * The QUESTION is whether it recovers the high-wpm gate and the wmae, while keeping M1/M3/M4/M5/M6.
  * DECISION RULE (registered now): the variant is the better PROPOSAL iff it passes the high-wpm
    gate AND keeps M1, M3, M4 winning, EVEN IF M2 > 0. Rationale: a structurally-refused arm is
    "worse than a plain null" (SRROLL-1), and a constant-column artifact is a TOOL-fixable defect
    (report constant columns apart) whereas a high-wpm regression is not.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import attrib as A  # noqa: E402
import metrics as M  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from keybo.analysis.shap_diff import _shap_tables, block_map  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_INTERP_WPM_FEATURE_NAMES,
    BIGRAM_INTERP_WPM_MONOTONE,
    FEATURE_VERSION_INTERP_WPM,
    interp_wpm_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import validate  # noqa: E402
from keybo.verdicts import bucket_regression_report  # noqa: E402

WPM = 90.0
SEEDS = [0, 1, 2]
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SCRATCH = "/tmp/interpframe_wk/models"
SENTINEL = "/tmp/interpframe_wk/variant.sentinel"
os.makedirs(SCRATCH, exist_ok=True)
NAMES = list(BIGRAM_INTERP_WPM_FEATURE_NAMES)
MONO = dict(zip(NAMES, BIGRAM_INTERP_WPM_MONOTONE, strict=True))
GEO = ROW_STAGGERED_31
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows")
out: dict = {"frame": NAMES, "monotone": dict(MONO), "stamp": FEATURE_VERSION_INTERP_WPM}

# --- train -------------------------------------------------------------------------------
models = []
for s in SEEDS:
    path = f"{SCRATCH}/interp_wpm_seed{s}.json"
    if os.path.exists(path):
        m = XGBoostTypingModel.load(path, expected_feature_version=FEATURE_VERSION_INTERP_WPM)
        log(f"  REUSED interp_wpm seed{s}")
    else:
        m = train_bigram_model(
            rows, target_wpm=WPM, geometry=GEO, interp="wpm", random_state=s, n_jobs=8
        )
        m.save(path)
        log(f"  trained interp_wpm seed{s}")
    assert m.metadata.feature_version == FEATURE_VERSION_INTERP_WPM
    assert list(m.metadata.feature_names) == NAMES
    tag = m.metadata.extra["training"]["interp_frame"]
    assert tag["frame"] == "interp-wpm", tag
    assert tuple(tag["monotone_constraints"]) == tuple(BIGRAM_INTERP_WPM_MONOTONE)
    models.append(m)

# --- the METRICS, by exactly the same code the other arms used ---------------------------
surface = default_surface(WPM, None)
GS = surface.geometry
serve_pos = [*GS.slots, GS.space_position]
_, LAY_A = _resolve("flagship-c3")
_, LAY_B = _resolve("graphite")
w2_char, covered = A.char_bigram_weight(surface, LAY_A)
slot_a = surface._slot_of(LAY_A)
perm = np.array([slot_a[c] for c in LAY_A] + [slot_a[" "]], dtype=np.intp)
n_pos = len(serve_pos)
w2_pos = np.zeros((n_pos, n_pos))
np.add.at(w2_pos, (perm[:, None], perm[None, :]), w2_char)

X = np.vstack(
    [interp_wpm_features_from_positions(GS, (a, b), wpm=WPM) for a in serve_pos for b in serve_pos]
)
r = A.t2_attribution(models, GS, surface, LAY_A, LAY_B, w2_char, covered, WPM, "interp-wpm")
log(f"attribution gap {r['gap']:+.4f} reconciles {r['reconciles']}")
ms = r["ms"]

# monotone verification, same three checks
tables = _shap_tables(models, GS, WPM, 2, "interp-wpm")
shap = np.mean(tables[0], axis=0).reshape(-1, len(NAMES))
base = np.median(X, axis=0)
honored, detail = {}, {}
for j, name in enumerate(NAMES):
    vals = np.unique(X[:, j])
    if len(vals) > 25:
        vals = np.unique(np.quantile(X[:, j], np.linspace(0, 1, 25)))
    Xs = np.tile(base, (len(vals), 1))
    Xs[:, j] = vals
    pred = np.mean([m.predict(Xs) for m in models], axis=0)
    d = np.diff(pred)
    want = MONO[name]
    if d.size == 0:
        # ⚠ `wpm` is CONSTANT on the serve grid, so its sweep has ONE point and there is no step to
        # check. This is not a pass and not a violation: the constraint is UNVERIFIABLE on the
        # population being explained, which is exactly the defect this variant re-admits. Scored as
        # zero range and zero violation, and the `alive`/`rho` checks below then keep it OUT of M4.
        viol = 0.0
        pred = np.array([pred[0], pred[0]])
    else:
        viol = max(float(-d.min()) if want > 0 else float(d.max()), 0.0)
    col, sv = X[:, j], shap[:, j]
    rho = float(spearmanr(col, sv).statistic) if float(np.ptp(col)) > 0 else float("nan")
    alive = float(np.abs(sv).mean()) > 1e-6
    sign_ok = (rho >= -1e-12) if want > 0 else (rho <= 1e-12)
    if np.isnan(rho):
        # ⚠ `wpm` is CONSTANT on the serve grid, so rho is undefined there. That is not a
        # violation, it is the artifact this frame re-admits: the column cannot be verified on a
        # population where it does not vary, so it CANNOT count toward M4. Recorded explicitly.
        sign_ok = False
    honored[name] = bool(viol <= 1e-9 and sign_ok and alive)
    detail[name] = {
        "constraint": want,
        "booster_monotone": bool(viol <= 1e-9),
        "worst_violation": viol,
        "response_range": float(pred.max() - pred.min()),
        "rho": rho,
        "mean_abs_shap": float(np.abs(sv).mean()),
        "alive": bool(alive),
        "honored": honored[name],
        "constant_on_serve_grid": bool(float(np.ptp(col)) == 0.0),
    }
out["honored_detail"] = detail
print()
print(f"{'column':<20} {'con':>4} {'booster':>8} {'rho':>9} {'mean|SHAP|':>11} {'const?':>7} {'HONORED':>8}")
for n in NAMES:
    d = detail[n]
    print(
        f"{n:<20} {MONO[n]:>+4d} {str(d['booster_monotone']):>8} {d['rho']:>9.4f} "
        f"{d['mean_abs_shap']:>11.3e} {str(d['constant_on_serve_grid']):>7} {str(d['honored']):>8}"
    )

surface_iweb = default_surface(WPM, "iweb")
w2_i, cov_i = A.char_bigram_weight(surface_iweb, LAY_A)
r_iweb = A.t2_attribution(models, GS, surface_iweb, LAY_A, LAY_B, w2_i, cov_i, WPM, "interp-wpm")
seed_attribs = [
    A.t2_attribution([m], GS, surface, LAY_A, LAY_B, w2_char, covered, WPM, "interp-wpm")["ms"]
    for m in models
]
agree, rhos = [], []
for i in range(len(seed_attribs)):
    for j in range(i + 1, len(seed_attribs)):
        sg = M.sign_agreement(seed_attribs[i], seed_attribs[j])
        agree.append(sg["sign_agree_frac"])
        rhos.append(sg["rho"])
unan = np.array(
    [len({np.sign(sa[k]) for sa in seed_attribs if abs(sa[k]) >= 1e-4}) <= 1 for k in range(len(NAMES))]
)
spec = block_map(NAMES)
blk: dict[str, float] = {}
for name, v in zip(NAMES, ms, strict=True):
    blk[spec[name][0]] = blk.get(spec[name][0], 0.0) + float(v)

out["metrics"] = {
    "n_columns": len(NAMES),
    "gap_t2": r["gap"],
    "M1": M.m1_maxcorr(X, w2_pos.ravel(), NAMES),
    "M2": M.m2_constfrac(X, w2_pos.ravel(), NAMES, ms),
    "M3": M.m3_splitpairs(NAMES, ms),
    "M4": M.m4_monofrac(NAMES, ms, honored),
    "M5": M.sign_agreement(ms, r_iweb["ms"]),
    "M6": {
        "unanimous_sign_frac": float(unan.mean()),
        "mean_pairwise_sign_agree": float(np.mean(agree)),
        "mean_pairwise_rho": float(np.mean(rhos)),
    },
    "attribution": dict(zip(NAMES, [float(v) for v in ms], strict=True)),
    "blocks": blk,
    "residuals": {k: v for k, v in r.items() if k.startswith("resid") or k == "reconciles"},
}

# --- the LOLO arm + the high-wpm gate against CUR's PER-FOLD baseline --------------------
log("LOLO: interp-wpm, 4 folds x 3 seeds")
rep = validate(
    rows, seeds=SEEDS, ngram="bigram", n_boot=10, geometry=GEO,
    interp="wpm", train_params={"n_jobs": 8},
)
log("LOLO done")
out["lolo"] = rep

lolo = json.load(open(f"{ARTIFACTS}/lolo.json"))
cur = lolo["arms"]["CUR"]
baseline_per_fold = {}
for holdout, fold in cur["folds"].items():
    acc: dict[int, list[float]] = {}
    for rec in fold["seeds"]:
        for b, rho in (rec.get("bucket_rhos") or {}).items():
            if rho is not None:
                acc.setdefault(int(b), []).append(float(rho))
    baseline_per_fold[holdout] = {b: float(np.mean(v)) for b, v in sorted(acc.items())}

hw: dict = {}
for holdout, fold in rep["folds"].items():
    n_seeds = len(fold["seeds"])
    hits: dict[int, int] = {}
    for rec in fold["seeds"]:
        block = bucket_regression_report(
            {int(k): v for k, v in (rec.get("bucket_rhos") or {}).items()},
            baseline_per_fold.get(holdout, {}),
            f"interp-wpm/{holdout}/seed{rec['seed']}",
            support=rec.get("bucket_support"),
        )
        for b in block.get("regressing_high_buckets", []):
            hits[int(b)] = hits.get(int(b), 0) + 1
    hw[holdout] = {
        "structural": sorted(b for b, h in hits.items() if h == n_seeds),
        "noise": sorted(b for b, h in hits.items() if 0 < h < n_seeds),
    }
structural = {h: d["structural"] for h, d in hw.items() if d["structural"]}
out["high_wpm"] = {"passed": not structural, "structural": structural, "detail": hw}


def flat(r_, key):
    return [m[key] for f in r_["folds"].values() for m in f["seeds"] if m.get(key) is not None]


def per_fold(r_, key):
    d = {}
    for h, f in r_["folds"].items():
        for m in f["seeds"]:
            d.setdefault(h, {})[m["seed"]] = m.get(key)
    return d


def paired(a, b, key):
    A_, B_ = per_fold(a, key), per_fold(b, key)
    per, cells = {}, []
    for h in sorted(set(A_) & set(B_)):
        ds = [B_[h][s] - A_[h][s] for s in sorted(set(A_[h]) & set(B_[h]))
              if A_[h][s] is not None and B_[h][s] is not None]
        if ds:
            per[h] = {"mean_delta": float(np.mean(ds)),
                      "sign_consistent": bool(all(x > 0 for x in ds) or all(x < 0 for x in ds))}
            cells.extend(ds)
    return {"per_fold": per, "mean_paired_delta": float(np.mean(cells)),
            "wins": int(sum(1 for x in cells if x > 0)), "losses": int(sum(1 for x in cells if x < 0)),
            "n_folds_sign_consistent": sum(1 for v in per.values() if v["sign_consistent"]),
            "n_folds": len(per)}


out["deltas_vs_CUR"] = {k: paired(cur, rep, k) for k in ("rho", "rho_frac_ceiling", "mae_model", "wmae", "umae")}
out["deltas_vs_INTERP"] = {
    k: paired(lolo["arms"]["INTERP"], rep, k) for k in ("rho", "rho_frac_ceiling", "wmae")
}
with open(f"{ARTIFACTS}/variant.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)

# =========================================================================================
print()
print("=" * 92)
print("THE VARIANT: interp-wpm (11 cols) vs interp (10 cols) vs CUR (20 cols)")
print("=" * 92)
base_j = json.load(open(f"{ARTIFACTS}/baseline.json"))["frames"]["served_bigram"]
poc = json.load(open(f"{ARTIFACTS}/poc.json"))["interp"]
v = out["metrics"]
print(f"{'metric':<26} {'CUR (20c)':>12} {'interp (10c)':>14} {'interp-wpm (11c)':>18}")
for label, key, sub in [
    ("M1  MAXCORR", "M1", "maxcorr"),
    ("M1b MEANCORR", "M1", "meancorr"),
    ("M2  CONSTFRAC", "M2", "constfrac"),
    ("M3  SPLITPAIRS", "M3", "splitpairs"),
    ("M4  MONOFRAC", "M4", "monofrac"),
    ("M5  SIGNSTAB", "M5", "sign_agree_frac"),
    ("M5  rho", "M5", "rho"),
    ("M6  unanimous", "M6", "unanimous_sign_frac"),
]:
    print(f"{label:<26} {base_j[key][sub]:>12.4f} {poc[key][sub]:>14.4f} {v[key][sub]:>18.4f}")
print()
print("ACCURACY (paired per-fold, MOR-FIX-1):")
for k, d in out["deltas_vs_CUR"].items():
    print(f"  vs CUR    {k:<18} {d['mean_paired_delta']:+10.5f}  W/L {d['wins']}/{d['losses']}  sc {d['n_folds_sign_consistent']}/{d['n_folds']}")
for k, d in out["deltas_vs_INTERP"].items():
    print(f"  vs INTERP {k:<18} {d['mean_paired_delta']:+10.5f}  W/L {d['wins']}/{d['losses']}  sc {d['n_folds_sign_consistent']}/{d['n_folds']}")
print()
print(f"ABSOLUTE  rho {np.mean(flat(rep, 'rho')):.4f}  rho/ceil {np.mean(flat(rep, 'rho_frac_ceiling')):.4f}  "
      f"wmae {np.mean(flat(rep, 'wmae')):.4f}  tau {[p['tau_heldout'] for p in rep['pooled']]}")
print()
print(f"HIGH-WPM GATE: {'PASS' if out['high_wpm']['passed'] else 'STRUCTURAL REGRESSION ' + str(structural)}")
print(f"  (noise-only: { {h: d['noise'] for h, d in hw.items() if d['noise']} })")
log(f"wrote {ARTIFACTS}/variant.json")
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
