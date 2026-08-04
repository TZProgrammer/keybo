"""HYBRIDB-1 §2 — hybrid-B's ACCURACY: three-arm matched leave-one-layout-out.

Registered at HYBRIDTRI-preregistration.md §2 before any accuracy number existed. This is the
number my brief says is UNMEASURED, and the arm's headline.

ARMS (§2, registered):
  CUR       served 20c   -- the incumbent, MEASURED here rather than borrowed
  HYBRIDB   hybrid-B 18c -- the candidate
  INTERP    interp.1 10c -- the published reference point, RE-measured on the same folds/seeds so
                            the three-way comparison is not a comparison against a quoted constant

MATCHED: same seeds, same folds, same cell construction, same hyperparameters -- the ONLY
difference is the frame. Deltas are PAIRED PER-FOLD per MOR-FIX-1 (a mean of ratios can reorder).

Detached-friendly: writes a SENTINEL file when finished so a poller never has to `wait $PID`, and
checkpoints lolo.json after EVERY arm (six tmux crashes have hit this fleet).
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, assert_tree, require  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    BIGRAM_HYBRIDB_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training import validate as V  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

# Every symbol asserted to exist BEFORE the 4-hour run leans on it (brief-decay defence: rc=0 with
# all-None output is a key-not-present bug, not a measurement).
require(V, "validate")
require(V, "require_no_high_wpm_regression_in_report")
for _name, _n in (
    ("served", len(BIGRAM_FEATURE_NAMES)),
    ("interp", len(BIGRAM_INTERP_FEATURE_NAMES)),
    ("hybridb", len(BIGRAM_HYBRIDB_FEATURE_NAMES)),
):
    print(f"[frame] {_name}: {_n} columns")
if len(BIGRAM_HYBRIDB_FEATURE_NAMES) != 18:
    raise SystemExit(f"ABORT: hybrid-B is {len(BIGRAM_HYBRIDB_FEATURE_NAMES)} columns, not 18")

SEEDS = [0, 1, 2]
WPM = 90.0
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SENTINEL = "/tmp/hybridtri_wk/lolo.sentinel"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")
GEO = ROW_STAGGERED_31

# Identical to INTERPFRAME-1's arm list minus its two ablations (which answered ITS question about
# the wpm column) plus the candidate. The reference arm is RE-RUN rather than read out of
# lolo.json, so all three arms share fold construction, seeds and ceilings exactly.
ARMS = [
    ("CUR", dict()),
    ("HYBRIDB", dict(interp="hybridb", monotone=True)),
    ("INTERP", dict(interp=True, monotone=True)),
]

out: dict = {
    "prereg": "agent-artifacts/hybridtri/HYBRIDTRI-preregistration.md @ 5a5d3c3 §2",
    "strokes": STROKES,
    "n_rows": len(rows),
    "seeds": SEEDS,
    "arms": {},
}

for name, kw in ARMS:
    log(f"ARM {name}: validate() 4 folds x {len(SEEDS)} seeds  kw={kw}")
    rep = validate(
        rows,
        seeds=SEEDS,
        ngram="bigram",
        n_boot=10,
        geometry=GEO,
        train_params={"n_jobs": 8},
        **kw,
    )
    out["arms"][name] = rep
    # ⚠ VERIFY THE ARM TRAINED THE FRAME IT CLAIMS. The report's own config block records the flag
    # that was PASSED; that is a label, not the referent. INTERPFRAME-1's near-miss was exactly a
    # validate() that forwarded a flag whose frame did not match what was featurized.
    cfg = rep.get("config", {})
    log(f"ARM {name}: done. config interp={cfg.get('interp')!r} monotone={cfg.get('monotone')}")
    with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)  # checkpoint after EVERY arm
    log(f"checkpointed {name} -> lolo.json")


# =========================================================================================
# PAIRED PER-FOLD DELTAS (MOR-FIX-1) -- lifted verbatim from INTERPFRAME-1's lolo.py so the
# delta CONVENTION is identical and the two arms' numbers are directly comparable.
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
METRICS = ("rho", "rho_frac_ceiling", "mae_model", "wmae", "umae")
out["deltas"] = {}
for arm in ("HYBRIDB", "INTERP"):
    out["deltas"][f"{arm}_vs_CUR"] = {k: paired(cur, out["arms"][arm], k) for k in METRICS}
# and the candidate against the frame it is a hybrid OF -- the "did the one-hots buy anything?" pair
out["deltas"]["HYBRIDB_vs_INTERP"] = {
    k: paired(out["arms"]["INTERP"], out["arms"]["HYBRIDB"], k) for k in METRICS
}
out["pooled_tau"] = {
    name: [p["tau_heldout"] for p in rep["pooled"]] for name, rep in out["arms"].items()
}
out["ceilings"] = {name: rep["ceilings"] for name, rep in out["arms"].items()}

with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)

print()
print("=" * 92)
print("PAIRED PER-FOLD DELTAS (MOR-FIX-1) — positive = the ARM is HIGHER than the reference")
print("=" * 92)
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
    print(f"  {name:<10} {taus}")
print()
print("ABSOLUTE per arm (mean over folds x seeds):")
out["absolute"] = {}
for name, rep in out["arms"].items():
    vals = [m["rho"] for f in rep["folds"].values() for m in f["seeds"] if m["rho"] is not None]
    frac = [
        m["rho_frac_ceiling"]
        for f in rep["folds"].values()
        for m in f["seeds"]
        if m.get("rho_frac_ceiling") is not None
    ]
    wm = [m["wmae"] for f in rep["folds"].values() for m in f["seeds"] if m.get("wmae") is not None]
    um = [m["umae"] for f in rep["folds"].values() for m in f["seeds"] if m.get("umae") is not None]
    out["absolute"][name] = {
        "rho": float(np.mean(vals)),
        "rho_frac_ceiling": float(np.mean(frac)),
        "wmae": float(np.mean(wm)),
        "umae": float(np.mean(um)),
    }
    a = out["absolute"][name]
    print(
        f"  {name:<10} rho {a['rho']:.4f}   rho/ceiling {a['rho_frac_ceiling']:.4f}   "
        f"wmae {a['wmae']:.4f}   umae {a['umae']:.4f}"
    )

# --- B1: THE HIGH-WPM GATE, with CUR's OWN PER-FOLD rhos as the incumbent baseline ---------
# ⚠ PER FOLD, not pooled. Pooling CUR's bucket rhos across folds made the gate refuse CUR ITSELF
# in INTERPFRAME-1's first pass, because dvorak's absolute rho (~0.70) sits below any cross-fold
# average BY CONSTRUCTION -- that measures fold heterogeneity, not the candidate. This is
# `bucket_regression_report`'s own documented failure mode.
from keybo.verdicts import bucket_regression_report  # noqa: E402

baseline_per_fold: dict[str, dict[int, float]] = {}
for holdout, fold in cur["folds"].items():
    acc: dict[int, list[float]] = {}
    for rec in fold["seeds"]:
        for bucket, rho in (rec.get("bucket_rhos") or {}).items():
            if rho is not None:
                acc.setdefault(int(bucket), []).append(float(rho))
    baseline_per_fold[holdout] = {b: float(np.mean(v)) for b, v in sorted(acc.items())}
if not baseline_per_fold:
    raise SystemExit("no bucket_rhos in the CUR report -- cannot form an incumbent baseline")

print()
print("HIGH-WPM GATE (B1). Incumbent = CUR's own per-bucket rho, PER FOLD (mean over its 3 seeds):")
for holdout, b in sorted(baseline_per_fold.items()):
    print(f"  {holdout:<8} " + "  ".join(f"b{k}:{v:.4f}" for k, v in b.items()))

hw: dict = {"incumbent_baseline_bucket_rhos_per_fold": baseline_per_fold, "arms": {}}
print()
print("  STRUCTURAL = regresses on EVERY seed of a fold => REFUSAL. noise-only = reported, no veto")
for name, rep in out["arms"].items():
    detail: dict[str, dict] = {}
    for holdout, fold in rep["folds"].items():
        n_seeds = len(fold["seeds"])
        hits: dict[int, int] = {}
        for rec in fold["seeds"]:
            block = bucket_regression_report(
                {int(k): v for k, v in (rec.get("bucket_rhos") or {}).items()},
                baseline_per_fold.get(holdout, {}),
                f"{name}/{holdout}/seed{rec['seed']}",
                support=rec.get("bucket_support"),
            )
            for bucket in block.get("regressing_high_buckets", []):
                hits[int(bucket)] = hits.get(int(bucket), 0) + 1
        detail[holdout] = {
            "n_seeds": n_seeds,
            "structural": sorted(b for b, h in hits.items() if h == n_seeds),
            "noise": sorted(b for b, h in hits.items() if 0 < h < n_seeds),
            "per_bucket_seed_counts": {str(k): v for k, v in sorted(hits.items())},
        }
    structural = {h: d["structural"] for h, d in detail.items() if d["structural"]}
    noise = {h: d["noise"] for h, d in detail.items() if d["noise"]}
    hw["arms"][name] = {
        "passed": not structural,
        "structural_regressions": structural,
        "noise_only": noise,
        "detail": detail,
    }
    verdict = "PASS" if not structural else f"STRUCTURAL REGRESSION {structural}"
    print(f"  {name:<10} {verdict}" + (f"   (noise-only: {noise})" if noise else ""))

# --- THE GATE'S OWN CONTROL (mandatory, §2): it must PASS the incumbent it is built from ---
hw["gate_control_incumbent_passes"] = bool(hw["arms"]["CUR"]["passed"])
if not hw["arms"]["CUR"]["passed"]:
    print()
    print("!! GATE CONTROL FAILED: the gate refuses the INCUMBENT against its own per-fold rhos.")
    print("!! Every candidate verdict from it is measuring seed noise. REPORT THAT, not a verdict.")
else:
    print()
    print("  GATE CONTROL: the gate PASSES the incumbent it is built from => verdicts readable.")
out["high_wpm"] = hw

# --- the registered decision rules, EVALUATED (so the verdict is in the artifact) ----------
dr = out["deltas"]["HYBRIDB_vs_CUR"]
out["registered_verdict"] = {
    "B1_high_wpm_gate_passed": hw["arms"]["HYBRIDB"]["passed"],
    "B1_gate_control_ok": hw["gate_control_incumbent_passes"],
    "B2_mean_paired_drho_vs_CUR": dr["rho"]["mean_paired_delta"],
    "B2_bar_drho_ge": -0.005,
    "B2_tau_heldout": out["pooled_tau"]["HYBRIDB"],
    "B2_rank_neutral": bool(
        dr["rho"]["mean_paired_delta"] >= -0.005
        and all(abs(t - 1.0) < 1e-12 for t in out["pooled_tau"]["HYBRIDB"])
    ),
    "B3_mean_paired_dwmae_vs_CUR": dr["wmae"]["mean_paired_delta"],
    "H_B_bar_dwmae_lt": 2.0,
    "H_B_holds": bool(dr["wmae"]["mean_paired_delta"] < 2.0),
    "interp_reference_dwmae_vs_CUR": out["deltas"]["INTERP_vs_CUR"]["wmae"]["mean_paired_delta"],
    "interp_reference_gate_passed": hw["arms"]["INTERP"]["passed"],
}
print()
print("=" * 92)
print("THE REGISTERED VERDICT (HYBRIDTRI-preregistration.md §2/§4)")
print("=" * 92)
for k, v in out["registered_verdict"].items():
    print(f"  {k:<38} {v}")

with open(f"{ARTIFACTS}/lolo.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/lolo.json")
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
log(f"SENTINEL {SENTINEL}")
