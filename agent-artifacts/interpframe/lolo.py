"""INTERPFRAME-1 §6 — the ACCURACY COST: four-arm matched leave-one-layout-out.

The cost side of the trade, reported honestly and with NO pre-registered bar (§0: interpretability
is the maximand and a negative delta is an acceptable, expected outcome — CLOSING-2 measured nine
feature-frame arms to NULL on accuracy).

ARMS (registered §6 before any number existed):
  CUR            the served 20-column frame — the incumbent, MEASURED here rather than borrowed
  INTERP         the 10-column frame, monotone constraints ON
  INTERP-NOMONO  the same frame, constraints OFF        (isolates §5d: the constraint's own cost)
  CUR-NOWPM      the served frame minus wpm             (isolates the cost of the drop that
                                                         makes CONSTFRAC == 0)

MATCHED: same seeds, same folds, same cell construction, same hyperparameters — the ONLY
difference is the frame. Deltas are PAIRED PER-FOLD per MOR-FIX-1 (a mean-of-ratios can reorder).

⚠ CUR-NOWPM cannot be built by dropping the column (the LOGRAT->ms conversion reads it), so it is
built by NEUTRALIZING wpm to a CONSTANT — which is byte-equivalent to a drop for a tree (a constant
column is unsplittable) while leaving the ms conversion intact. That is the same ablation-by-
neutralization convention the sg_distance ablation used, and it is stated because the alternative
reading (a real drop) would be a different experiment.

Detached-friendly: writes a SENTINEL file when finished so a poller never has to `wait $PID`.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

SEEDS = [0, 1, 2]
WPM = 90.0
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SENTINEL = "/tmp/interpframe_wk/lolo.sentinel"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")
GEO = ROW_STAGGERED_31

ARMS = [
    ("CUR", dict()),
    ("INTERP", dict(interp=True, monotone=True)),
    ("INTERP-NOMONO", dict(interp=True, monotone=False)),
    # The neutralization: `wpm` is pinned to its own mean so the column exists (the ms conversion
    # needs it) but carries no information a split could use.
    ("CUR-NOWPM", dict(_neutralize_wpm=True)),
]

out: dict = {"strokes": STROKES, "n_rows": len(rows), "seeds": SEEDS, "arms": {}}


def run_arm(name, kw):
    kw = dict(kw)
    neutralize = kw.pop("_neutralize_wpm", False)
    if neutralize:
        # Patch the featurizer the harness uses, for THIS arm only, then restore. Done by
        # wrapping rather than editing the shipped function so the other arms are provably
        # unaffected (and the restore is in a finally).
        import keybo.training.validate as V
        from keybo.features import bigram_features_from_positions as real

        wpm_const = float(np.mean([s[0] for r in rows for s in r.samples]))
        log(f"  CUR-NOWPM: neutralizing wpm to the constant {wpm_const:.3f}")

        def neutralized(geometry, positions, wpm=0.0, direction=False, kitchensink=False):
            vec = real(geometry, positions, wpm=wpm, direction=direction, kitchensink=kitchensink)
            # index 19 == 'wpm' (the schema pins wpm LAST on this frame); asserted, not assumed.
            from keybo.features.schema import BIGRAM_FEATURE_NAMES

            vec[BIGRAM_FEATURE_NAMES.index("wpm")] = wpm_const
            return vec

        saved_v = V.bigram_features_from_positions
        import keybo.training.train as T

        saved_t = T.bigram_features_from_positions
        V.bigram_features_from_positions = neutralized
        T.bigram_features_from_positions = neutralized
        try:
            return _validate(name, kw)
        finally:
            V.bigram_features_from_positions = saved_v
            T.bigram_features_from_positions = saved_t
    return _validate(name, kw)


def _validate(name, kw):
    log(f"ARM {name}: validate() 4 folds x {len(SEEDS)} seeds")
    rep = validate(
        rows,
        seeds=SEEDS,
        ngram="bigram",
        n_boot=10,
        geometry=GEO,
        train_params={"n_jobs": 8},
        **kw,
    )
    log(f"ARM {name}: done")
    return rep


for name, kw in ARMS:
    rep = run_arm(name, kw)
    out["arms"][name] = rep
    with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)  # checkpoint after EVERY arm
    log(f"checkpointed {name} -> lolo.json")

# =========================================================================================
# PAIRED PER-FOLD DELTAS (MOR-FIX-1)
# =========================================================================================
def per_fold(rep, key):
    """{holdout: {seed: value}} for one metric."""
    d = {}
    for holdout, fold in rep["folds"].items():
        for rec in fold["seeds"]:
            d.setdefault(holdout, {})[rec["seed"]] = rec.get(key)
    return d


def paired(rep_a, rep_b, key):
    """PAIRED per-(fold, seed) delta b - a, then the per-fold means. Never a mean of ratios."""
    A, B = per_fold(rep_a, key), per_fold(rep_b, key)
    per, cells = {}, []
    for holdout in sorted(set(A) & set(B)):
        deltas = [
            B[holdout][s] - A[holdout][s]
            for s in sorted(set(A[holdout]) & set(B[holdout]))
            if A[holdout][s] is not None and B[holdout][s] is not None
        ]
        if deltas:
            per[holdout] = {
                "mean_delta": float(np.mean(deltas)),
                "deltas": [float(x) for x in deltas],
                "sign_consistent": bool(all(x > 0 for x in deltas) or all(x < 0 for x in deltas)),
                "direction": "up" if np.mean(deltas) > 0 else "down",
            }
            cells.extend(deltas)
    return {
        "per_fold": per,
        "mean_paired_delta": float(np.mean(cells)) if cells else float("nan"),
        "n_cells": len(cells),
        "n_folds_sign_consistent": sum(1 for v in per.values() if v["sign_consistent"]),
        "n_folds": len(per),
        "wins": int(sum(1 for x in cells if x > 0)),
        "losses": int(sum(1 for x in cells if x < 0)),
    }


cur = out["arms"]["CUR"]
out["deltas"] = {}
for arm in ("INTERP", "INTERP-NOMONO", "CUR-NOWPM"):
    out["deltas"][f"{arm}_vs_CUR"] = {
        k: paired(cur, out["arms"][arm], k)
        for k in ("rho", "rho_frac_ceiling", "mae_model", "wmae", "umae")
    }
# the constraint's OWN cost: INTERP vs INTERP-NOMONO (§5d)
out["deltas"]["INTERP_vs_INTERP-NOMONO"] = {
    k: paired(out["arms"]["INTERP-NOMONO"], out["arms"]["INTERP"], k)
    for k in ("rho", "rho_frac_ceiling", "mae_model", "wmae", "umae")
}
out["pooled_tau"] = {
    name: [p["tau_heldout"] for p in rep["pooled"]] for name, rep in out["arms"].items()
}
out["ceilings"] = {name: rep["ceilings"] for name, rep in out["arms"].items()}

with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)

print()
print("=" * 90)
print("PAIRED PER-FOLD DELTAS (MOR-FIX-1) — positive = the ARM is better than CUR on rho")
print("=" * 90)
for comp, block in out["deltas"].items():
    print(f"\n{comp}")
    for metric, d in block.items():
        if np.isnan(d["mean_paired_delta"]):
            continue
        print(
            f"  {metric:<18} mean paired delta {d['mean_paired_delta']:+10.5f}   "
            f"W/L {d['wins']}/{d['losses']}   sign-consistent folds "
            f"{d['n_folds_sign_consistent']}/{d['n_folds']}"
        )
        for holdout, pf in sorted(d["per_fold"].items()):
            print(
                f"      {holdout:<10} {pf['mean_delta']:+10.5f}  "
                f"{'consistent' if pf['sign_consistent'] else 'mixed':<11} {pf['deltas']}"
            )

print()
print("POOLED tau_heldout per arm (out-of-sample layout ranking):")
for name, taus in out["pooled_tau"].items():
    print(f"  {name:<16} {taus}")
print()
print("ABSOLUTE rho per arm (mean over folds x seeds):")
for name, rep in out["arms"].items():
    vals = [m["rho"] for f in rep["folds"].values() for m in f["seeds"] if m["rho"] is not None]
    frac = [
        m["rho_frac_ceiling"]
        for f in rep["folds"].values()
        for m in f["seeds"]
        if m.get("rho_frac_ceiling") is not None
    ]
    print(f"  {name:<16} rho {np.mean(vals):.4f}   rho/ceiling {np.mean(frac):.4f}")

# --- the HIGH-WPM gate, run explicitly (§6's one registered refusal) ----------------------
print()
print("HIGH-WPM non-regression gate (baseline = CUR's own per-bucket rhos):")
from keybo.training.validate import require_no_high_wpm_regression_in_report  # noqa: E402
from keybo.verdicts import HighWpmRegression  # noqa: E402

out["high_wpm"] = {}
for name, rep in out["arms"].items():
    try:
        out["high_wpm"][name] = require_no_high_wpm_regression_in_report(rep, name)
        print(f"  {name:<16} {out['high_wpm'][name].get('passed')}")
    except HighWpmRegression as exc:
        out["high_wpm"][name] = {"passed": False, "error": str(exc)[:600]}
        print(f"  {name:<16} REFUSED: {str(exc)[:200]}")

with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/lolo.json")
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
log(f"SENTINEL {SENTINEL}")
