"""ARM-G robustness: is the grouped path's advantage REAL, or a one-seed artifact?

The v2 run showed a single candidate-set draw (seed 42) in which the grouped selection had
0.81% lower honest LOLO MAE. 0.81% on ONE draw is not a result -- the selection is a
best-of-12 argmin, so a different candidate set could reverse it.

This repeats the ENTIRE selection experiment over several independent candidate-set seeds and
reports the distribution. Registered addition to the prereg's ARM-G; because this adds a
family of tests, the prereg's multiplicity clause applies -- I report per-seed outcomes and a
sign test, NOT a best-of-N.

The pre-registered SUCCESS criterion for ARM-G was the OPTIMISM GAP, which v2 already
established. This run answers the separate, weaker question: does the grouped SELECTION also
transfer better, reliably?
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from keybo.testkit import assert_module_under

assert_module_under("keybo", "/tmp/kaggle")

CACHE = "/tmp/kaggle-work/matrix_bigram_lograt.npz"
OUT = "/tmp/kaggle-work/arm_g3_results.json"
SEEDS = [42, 7, 13, 99, 2024]
N_CAND = 8
NTHREAD = 8


def mae_over(params, X, y, splits) -> float:
    errs = []
    for tr, te in splits:
        est = XGBRegressor(objective="reg:squarederror", verbosity=0,
                           random_state=0, n_jobs=NTHREAD, **params)
        est.fit(X[tr], y[tr])
        errs.append(float(np.mean(np.abs(est.predict(X[te]) - y[te]))))
    return float(np.mean(errs))


def main() -> None:
    d = np.load(CACHE, allow_pickle=False)
    X, y, layouts = d["X"], d["y"], d["layouts"].astype(str)
    uniq = sorted(set(layouts))

    kf = list(KFold(n_splits=4, shuffle=False).split(X))
    gk = list(GroupKFold(n_splits=4).split(X, groups=layouts))
    lolo = [(np.where(layouts != h)[0], np.where(layouts == h)[0]) for h in uniq]

    out = {"config": {"seeds": SEEDS, "n_candidates": N_CAND, "n_examples": int(X.shape[0])},
           "per_seed": {}}

    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        cands = [
            {
                "n_estimators": int(rng.integers(100, 500)),
                "max_depth": int(rng.integers(2, 6)),
                "learning_rate": float(rng.uniform(0.02, 0.2)),
                "min_child_weight": int(rng.integers(1, 6)),
                "subsample": float(rng.uniform(0.6, 1.0)),
            }
            for _ in range(N_CAND)
        ]
        # honest transfer of EVERY candidate (so we can also report the ORACLE best)
        honest = [mae_over(c, X, y, lolo) for c in cands]
        believed_kf = [mae_over(c, X, y, kf) for c in cands]
        believed_gk = [honest[i] for i in range(len(cands))]  # GroupKFold(4)==LOLO by construction

        i_kf = int(np.argmin(believed_kf))
        i_gk = int(np.argmin(believed_gk))
        i_oracle = int(np.argmin(honest))

        rec = {
            "candidates": cands,
            "honest_lolo_mae": honest,
            "believed_kfold_mae": believed_kf,
            "pick_kfold": i_kf,
            "pick_grouped": i_gk,
            "pick_oracle": i_oracle,
            "honest_of_kfold_pick": honest[i_kf],
            "honest_of_grouped_pick": honest[i_gk],
            "honest_of_oracle": honest[i_oracle],
            "grouped_better_by": honest[i_kf] - honest[i_gk],
            "grouped_better_pct": 100.0 * (honest[i_kf] - honest[i_gk]) / honest[i_kf],
            "kfold_regret_vs_oracle": honest[i_kf] - honest[i_oracle],
            "grouped_regret_vs_oracle": honest[i_gk] - honest[i_oracle],
            "picks_agree": i_kf == i_gk,
            "optimism_kfold": honest[i_kf] - believed_kf[i_kf],
            "secs": round(time.time() - t0, 1),
        }
        out["per_seed"][str(seed)] = rec
        print(f"seed {seed}: kfold picks #{i_kf} (honest {honest[i_kf]:.6f}) | "
              f"grouped picks #{i_gk} (honest {honest[i_gk]:.6f}) | oracle #{i_oracle} "
              f"({honest[i_oracle]:.6f}) | grouped better by {rec['grouped_better_pct']:+.3f}% | "
              f"kfold optimism {rec['optimism_kfold']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(OUT, "w"), indent=2)

    # ---- aggregate, with multiplicity honesty ----
    diffs = [r["grouped_better_by"] for r in out["per_seed"].values()]
    pcts = [r["grouped_better_pct"] for r in out["per_seed"].values()]
    agree = sum(1 for r in out["per_seed"].values() if r["picks_agree"])
    n_grouped_wins = sum(1 for x in diffs if x > 0)
    n_ties = sum(1 for x in diffs if x == 0)
    # exact two-sided sign test over the NON-TIED seeds
    n_eff = len(diffs) - n_ties
    from math import comb
    if n_eff:
        k = n_grouped_wins
        p_two = sum(comb(n_eff, i) for i in range(0, n_eff + 1)
                    if abs(i - n_eff / 2) >= abs(k - n_eff / 2)) / 2 ** n_eff
    else:
        p_two = 1.0
    out["aggregate"] = {
        "n_seeds": len(diffs),
        "picks_agree_count": agree,
        "grouped_wins": n_grouped_wins,
        "ties": n_ties,
        "mean_grouped_better_pct": float(np.mean(pcts)),
        "median_grouped_better_pct": float(np.median(pcts)),
        "min_pct": float(np.min(pcts)),
        "max_pct": float(np.max(pcts)),
        "sign_test_p_two_sided": p_two,
        "mean_kfold_optimism": float(np.mean([r["optimism_kfold"] for r in out["per_seed"].values()])),
        "mean_kfold_regret_vs_oracle": float(
            np.mean([r["kfold_regret_vs_oracle"] for r in out["per_seed"].values()])),
        "mean_grouped_regret_vs_oracle": float(
            np.mean([r["grouped_regret_vs_oracle"] for r in out["per_seed"].values()])),
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n=== AGGREGATE ===", flush=True)
    for k, v in out["aggregate"].items():
        print(f"  {k} = {v}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
