"""ARM-M — monotone_constraints from physical priors, judged on the RANKING frame.

Registered in agent-artifacts/kaggle-techniques/PREREG-kaggle-techniques.md @ 8168a82.

The NGRAM-FE precedent is the thing to beat: a technique can improve fit and DESTROY the
layout ranking. So this driver reports, per arm:
  - LOLO per-fold rho (bucket-centered Spearman)  + rho/ceiling where the ceiling is finite
  - pooled tau_heldout  <- THE RANKING GUARD. Every layout scored only by the fold that
    held it out. This is the number that kills the arm if it degrades.
  - wmae / umae (the ledger's incumbent-protection clause is wmae within +0.91%)

Arms:
  BASELINE   -- production-shaped params, unconstrained
  MONOTONE   -- same params + monotone_constraints from the priors registered in the prereg

Both go through keybo.training.validate.validate(), i.e. the SHIPPED harness, so the
numbers are the harness's own and not a re-implementation. Nothing shipped is overwritten.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from keybo.testkit import assert_module_under

assert_module_under("keybo", "/tmp/kaggle")  # wrong-tree guard

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

TSV = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
OUT = "/tmp/kaggle-work/arm_m_results.json"
SEEDS = [0, 1, 2]

# Production-shaped params (per README/train.py prose: depth 3, subsample .7, colsample .7).
BASE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
}

# The priors registered in the prereg. +1 = predicted time NON-DECREASING in the feature.
# These are MY priors, defended from physics -- EXPECTED_SIGN does not exist in this repo.
PRIORS = {
    "distance": +1,     # farther travel is slower
    "dy": +1,           # more vertical (row-crossing) travel is slower
    "same_finger": +1,  # same-finger bigrams are slower
    "scissor": +1,      # scissors are slower
    "lsb": +1,          # lateral stretch is slower
    "wpm": -1,          # a faster session means shorter keystroke times
}


def constraint_tuple() -> tuple[int, ...]:
    """monotone_constraints aligned to BIGRAM_FEATURE_NAMES column order."""
    return tuple(PRIORS.get(name, 0) for name in BIGRAM_FEATURE_NAMES)


def summarize(report: dict) -> dict:
    """Pull the decision-relevant numbers out of a validate() report."""
    folds = {}
    for layout, fold in report["folds"].items():
        rhos = [m["rho"] for m in fold["seeds"]]
        fracs = [
            float(m["rho_frac_ceiling"])
            for m in fold["seeds"]
            if m["rho_frac_ceiling"] is not None
            and math.isfinite(float(m["rho_frac_ceiling"]))
        ]
        folds[layout] = {
            "ceiling": fold["seeds"][0]["ceiling"],
            "rho_mean": float(np.mean(rhos)),
            "rho_per_seed": [float(r) for r in rhos],
            "rho_frac_ceiling_mean": float(np.mean(fracs)) if fracs else None,
            "wmae_mean": float(np.mean([m["wmae"] for m in fold["seeds"]])),
            "umae_mean": float(np.mean([m["umae"] for m in fold["seeds"]])),
            "beats_baseline": [m["beats_baseline"] for m in fold["seeds"]],
            "n_cells": fold["n_cells"],
        }
    taus = [p["tau_heldout"] for p in report["pooled"]]
    all_fracs = [
        float(m["rho_frac_ceiling"])
        for fold in report["folds"].values()
        for m in fold["seeds"]
        if m["rho_frac_ceiling"] is not None and math.isfinite(float(m["rho_frac_ceiling"]))
    ]
    return {
        "folds": folds,
        "tau_heldout_per_seed": [float(t) for t in taus],
        "tau_heldout_mean": float(np.mean(taus)),
        "tau_heldout_min": float(np.min(taus)),
        "lolo_mean_rho_frac_ceiling": float(np.mean(all_fracs)) if all_fracs else None,
        "n_finite_frac_cells": len(all_fracs),
        "wmae_overall": float(np.mean([f["wmae_mean"] for f in folds.values()])),
        "umae_overall": float(np.mean([f["umae_mean"] for f in folds.values()])),
    }


def main() -> None:
    t0 = time.time()
    rows = load_strokes(TSV, ngram_len=2, wpm_threshold=0, min_samples=1)
    print(f"loaded {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)

    cons = constraint_tuple()
    print(f"feature order : {list(BIGRAM_FEATURE_NAMES)}", flush=True)
    print(f"constraints   : {cons}", flush=True)
    print(f"constrained   : {[n for n, c in zip(BIGRAM_FEATURE_NAMES, cons) if c]}", flush=True)
    assert len(cons) == len(BIGRAM_FEATURE_NAMES)
    assert sum(1 for c in cons if c != 0) == len(PRIORS), "a prior name did not match a feature"

    arms = {
        "BASELINE_unconstrained": dict(BASE_PARAMS),
        "MONOTONE_priors": {**BASE_PARAMS, "monotone_constraints": cons},
    }

    results: dict = {
        "config": {
            "tsv": TSV, "seeds": SEEDS, "base_params": BASE_PARAMS,
            "priors": PRIORS, "constraint_tuple": list(cons),
            "feature_order": list(BIGRAM_FEATURE_NAMES),
            "n_rows": len(rows),
        },
        "arms": {},
    }

    for name, params in arms.items():
        t1 = time.time()
        print(f"\n=== {name} ===", flush=True)
        rep = validate(rows, seeds=SEEDS, ngram="bigram", train_params=params, n_boot=10)
        s = summarize(rep)
        s["secs"] = round(time.time() - t1, 1)
        results["arms"][name] = s
        print(f"  tau_heldout per seed = {s['tau_heldout_per_seed']}", flush=True)
        print(f"  tau_heldout mean={s['tau_heldout_mean']:.4f} min={s['tau_heldout_min']:.4f}",
              flush=True)
        print(f"  LOLO mean rho/ceiling = {s['lolo_mean_rho_frac_ceiling']} "
              f"(finite cells: {s['n_finite_frac_cells']})", flush=True)
        print(f"  wmae={s['wmae_overall']:.4f} umae={s['umae_overall']:.4f}", flush=True)
        for la, f in sorted(s["folds"].items()):
            print(f"    {la:8s} ceiling={f['ceiling']!s:>8.8} rho={f['rho_mean']:+.4f} "
                  f"frac={f['rho_frac_ceiling_mean']} wmae={f['wmae_mean']:.4f}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)

    # ---- the pre-registered decision ----
    b = results["arms"]["BASELINE_unconstrained"]
    m = results["arms"]["MONOTONE_priors"]
    d_tau = m["tau_heldout_mean"] - b["tau_heldout_mean"]
    d_wmae_pct = 100.0 * (m["wmae_overall"] - b["wmae_overall"]) / b["wmae_overall"]
    d_frac = (
        None
        if (m["lolo_mean_rho_frac_ceiling"] is None or b["lolo_mean_rho_frac_ceiling"] is None)
        else m["lolo_mean_rho_frac_ceiling"] - b["lolo_mean_rho_frac_ceiling"]
    )
    verdict = {
        "delta_tau_heldout_mean": d_tau,
        "delta_wmae_pct": d_wmae_pct,
        "delta_lolo_rho_frac_ceiling": d_frac,
        "gate1_ranking_not_degraded": bool(d_tau >= -1e-9),
        "gate2_rho_frac_not_worse_than_0.005": (None if d_frac is None else bool(d_frac > -0.005)),
        "gate3_wmae_within_+0.91pct": bool(d_wmae_pct <= 0.91),
    }
    gates = [v for k, v in verdict.items() if k.startswith("gate") and v is not None]
    verdict["ADOPT"] = bool(all(gates))
    results["verdict"] = verdict
    json.dump(results, open(OUT, "w"), indent=2)
    print("\n=== PRE-REGISTERED VERDICT ===", flush=True)
    for k, v in verdict.items():
        print(f"  {k} = {v}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
