"""INTERPFRAME-1 — CUR-NOWPM, RE-RUN with the conversion confound REMOVED.

⚠ THE ARM I FIRST RAN WAS CONFOUNDED, and I caught it after reading the result rather than before.
The ablation neutralized the `wpm` COLUMN to a constant, intending "the tree cannot split on pace".
But `TypingModel.to_ms` RECOVERS the pace from that same column to convert LOGRAT -> ms:

    wpm = np.asarray(X)[:, self.metadata.feature_names.index("wpm")]      (models/base.py:123)

so the neutralized arm ALSO converted every cell at one constant pace instead of its own bucket
midpoint. Its wmae (+15.54 vs CUR) and its tau collapse (1.0 -> 0.333) therefore mix TWO changes:
the intended feature ablation, and an unintended mis-scaled ms conversion. That is not the
quantity §6 registered, and the tau collapse in particular is the signature of a broken conversion
rather than of a weaker model.

THE FIX, and why it is the honest one: neutralize the column the tree SEES while converting at the
TRUE per-cell pace. Both are achieved by patching the featurizer for TRAINING/prediction input and
passing the true wpm to `to_ms` explicitly — which the `wpm=` parameter added for the interp frame
already supports. So the re-run isolates exactly one thing: whether the model can USE pace as a
feature.

Everything else is identical to the original arm: same rows, same seeds, same folds, same cells.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import keybo.training.train as T  # noqa: E402
import keybo.training.validate as V  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import bigram_features_from_positions as REAL  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402

SEEDS = [0, 1, 2]
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SENTINEL = "/tmp/interpframe_wk/nowpm.sentinel"
J_WPM = BIGRAM_FEATURE_NAMES.index("wpm")
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows")
GEO = ROW_STAGGERED_31
WPM_CONST = float(np.mean([s[0] for r in rows for s in r.samples]))
log(f"neutralizing the wpm COLUMN to {WPM_CONST:.4f} (column index {J_WPM})")


def neutralized(geometry, positions, wpm=0.0, direction=False, kitchensink=False):
    vec = REAL(geometry, positions, wpm=wpm, direction=direction, kitchensink=kitchensink)
    vec[J_WPM] = WPM_CONST
    return vec


# --- the ms conversion must use the TRUE per-cell pace, not the neutralized column ---------
# _predict_cells' last line is `model.to_ms(pred, X, <wpm or None>)`; with the column neutralized,
# `None` would make to_ms read the CONSTANT. Wrapping _predict_cells to force the true per-cell
# wpm is what separates the feature ablation from the conversion.
REAL_PREDICT_CELLS = V._predict_cells


def predict_cells_true_pace(model, cells, geometry, direction=False, kitchensink=False, interp=False):
    X = np.vstack(
        [
            neutralized(geometry, c.positions, wpm=c.wpm, direction=direction, kitchensink=kitchensink)
            for c in cells
        ]
    )
    pred = model.predict(X)
    practice = (model.metadata.extra.get("training") or {}).get("practice_term")
    if practice:
        values = practice.get("values", {})
        pred = pred + np.array([values.get(c.ngram, 0.0) for c in cells])
    # THE FIX: convert at each cell's OWN bucket midpoint. to_ms would otherwise read the
    # neutralized column and rescale every prediction by a single wrong pace.
    true_wpm = np.array([c.wpm for c in cells], dtype=np.float64)
    return np.exp(pred) * 12000.0 / true_wpm


V.bigram_features_from_positions = neutralized
T.bigram_features_from_positions = neutralized
V._predict_cells = predict_cells_true_pace
try:
    log("ARM CUR-NOWPM-FIXED: validate() 4 folds x 3 seeds")
    rep = V.validate(
        rows,
        seeds=SEEDS,
        ngram="bigram",
        n_boot=10,
        geometry=GEO,
        train_params={"n_jobs": 8},
    )
finally:
    V.bigram_features_from_positions = REAL
    T.bigram_features_from_positions = REAL
    V._predict_cells = REAL_PREDICT_CELLS
log("done")

# --- SANITY: the conversion fix must be VISIBLE, or the patch did not take -----------------
prev = json.load(open(f"{ARTIFACTS}/lolo.json"))
old = prev["arms"]["CUR-NOWPM"]


def flat(r, key):
    return [m[key] for f in r["folds"].values() for m in f["seeds"] if m.get(key) is not None]


print()
print("CUR-NOWPM: the CONFOUNDED arm vs the FIXED arm (the conversion, not the features)")
print(f"{'metric':<20} {'confounded':>12} {'fixed':>12} {'CUR':>12}")
cur = prev["arms"]["CUR"]
for key in ("rho", "rho_frac_ceiling", "wmae", "mae_model"):
    print(
        f"{key:<20} {np.mean(flat(old, key)):>12.4f} {np.mean(flat(rep, key)):>12.4f} "
        f"{np.mean(flat(cur, key)):>12.4f}"
    )
print(
    f"{'tau_heldout':<20} {[round(p['tau_heldout'], 4) for p in old['pooled']]!s:>12} "
    f"{[round(p['tau_heldout'], 4) for p in rep['pooled']]!s:>12} "
    f"{[round(p['tau_heldout'], 4) for p in cur['pooled']]!s:>12}"
)

out = {
    "note": "CUR-NOWPM re-run with the LOGRAT->ms conversion using each cell's TRUE bucket "
    "midpoint instead of the neutralized column. Isolates the FEATURE ablation.",
    "wpm_const_used_for_the_column": WPM_CONST,
    "arm": rep,
    "comparison": {
        "confounded": {k: float(np.mean(flat(old, k))) for k in ("rho", "rho_frac_ceiling", "wmae", "mae_model")},
        "fixed": {k: float(np.mean(flat(rep, k))) for k in ("rho", "rho_frac_ceiling", "wmae", "mae_model")},
        "cur": {k: float(np.mean(flat(cur, k))) for k in ("rho", "rho_frac_ceiling", "wmae", "mae_model")},
        "tau_confounded": [p["tau_heldout"] for p in old["pooled"]],
        "tau_fixed": [p["tau_heldout"] for p in rep["pooled"]],
        "tau_cur": [p["tau_heldout"] for p in cur["pooled"]],
    },
}


def per_fold(r, key):
    d = {}
    for h, f in r["folds"].items():
        for m in f["seeds"]:
            d.setdefault(h, {})[m["seed"]] = m.get(key)
    return d


def paired(a, b, key):
    A, B = per_fold(a, key), per_fold(b, key)
    per, cells = {}, []
    for h in sorted(set(A) & set(B)):
        ds = [
            B[h][s] - A[h][s]
            for s in sorted(set(A[h]) & set(B[h]))
            if A[h][s] is not None and B[h][s] is not None
        ]
        if ds:
            per[h] = {
                "mean_delta": float(np.mean(ds)),
                "deltas": [float(x) for x in ds],
                "sign_consistent": bool(all(x > 0 for x in ds) or all(x < 0 for x in ds)),
            }
            cells.extend(ds)
    return {
        "per_fold": per,
        "mean_paired_delta": float(np.mean(cells)),
        "wins": int(sum(1 for x in cells if x > 0)),
        "losses": int(sum(1 for x in cells if x < 0)),
        "n_folds_sign_consistent": sum(1 for v in per.values() if v["sign_consistent"]),
        "n_folds": len(per),
    }


out["deltas_vs_CUR"] = {
    k: paired(cur, rep, k) for k in ("rho", "rho_frac_ceiling", "mae_model", "wmae", "umae")
}
print()
print("PAIRED PER-FOLD DELTAS, FIXED CUR-NOWPM vs CUR (MOR-FIX-1):")
for k, d in out["deltas_vs_CUR"].items():
    print(
        f"  {k:<18} mean paired delta {d['mean_paired_delta']:+10.5f}   W/L {d['wins']}/{d['losses']}"
        f"   sign-consistent {d['n_folds_sign_consistent']}/{d['n_folds']}"
    )
    for h, pf in sorted(d["per_fold"].items()):
        print(f"      {h:<10} {pf['mean_delta']:+10.5f}  {'consistent' if pf['sign_consistent'] else 'mixed'}")

with open(f"{ARTIFACTS}/lolo_nowpm_fixed.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/lolo_nowpm_fixed.json")
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
