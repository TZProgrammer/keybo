"""ARM-G (v2) — identical registered design, bounded threads + cached matrix.

Only the COMPUTE BUDGET changed vs arm_g_groupcv.py; the measured quantity, splitters,
candidate set (seed 42) and decision rule are byte-identical to the prereg @ 8168a82.
v1 was killed after 27min/0-of-3 splitters: n_jobs=-1 on a box at load 745 across 192
cores was self-starving. n_jobs is now pinned and the parsed matrix is loaded from an
.npz cache instead of re-paying the 236s TSV parse.
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

assert_module_under("keybo", "/tmp/kaggle")  # wrong-tree guard, per prereg section 5

CACHE = "/tmp/kaggle-work/matrix_bigram_lograt.npz"
OUT = "/tmp/kaggle-work/arm_g_results.json"
N_CAND = 12
SEED = 42
NTHREAD = 8


def cv_mae(params: dict, X, y, splits) -> float:
    """Mean held-out MAE over the given (train, test) index pairs."""
    errs = []
    for tr, te in splits:
        est = XGBRegressor(objective="reg:squarederror", verbosity=0,
                           random_state=SEED, n_jobs=NTHREAD, **params)
        est.fit(X[tr], y[tr])
        errs.append(float(np.mean(np.abs(est.predict(X[te]) - y[te]))))
    return float(np.mean(errs))


def main() -> None:
    d = np.load(CACHE, allow_pickle=False)
    X, y, layouts = d["X"], d["y"], d["layouts"].astype(str)
    uniq = sorted(set(layouts))
    print(f"matrix X={X.shape} layouts={uniq}", flush=True)

    rng = np.random.default_rng(SEED)
    candidates = [
        {
            "n_estimators": int(rng.integers(100, 500)),
            "max_depth": int(rng.integers(2, 6)),
            "learning_rate": float(rng.uniform(0.02, 0.2)),
            "min_child_weight": int(rng.integers(1, 6)),
            "subsample": float(rng.uniform(0.6, 1.0)),
        }
        for _ in range(N_CAND)
    ]

    splitters = {
        "kfold_noshuffle_SHIPPED": list(KFold(n_splits=4, shuffle=False).split(X)),
        "kfold_shuffle_naivefix": list(
            KFold(n_splits=4, shuffle=True, random_state=SEED).split(X)
        ),
        "groupkfold_layout_FIX": list(
            GroupKFold(n_splits=4).split(X, groups=layouts)
        ),
    }
    # Sanity: the grouped splitter must put each layout in exactly one test fold.
    for tr, te in splitters["groupkfold_layout_FIX"]:
        assert len(set(layouts[te])) == 1, "GroupKFold test fold spans >1 layout"
        assert not (set(layouts[te]) & set(layouts[tr])), "layout on both sides of a grouped split"
    # And the shipped splitter must NOT have that property (that IS the defect).
    straddle = sum(
        1 for tr, te in splitters["kfold_noshuffle_SHIPPED"]
        if set(layouts[te]) & set(layouts[tr])
    )
    print(f"shipped KFold: {straddle}/4 folds share >=1 layout across train/test "
          f"(this is the leak)", flush=True)

    results: dict = {
        "config": {"cache": CACHE, "n_candidates": N_CAND, "seed": SEED,
                   "target_space": "LOGRAT", "n_examples": int(X.shape[0]),
                   "n_features": int(X.shape[1]), "layouts": uniq,
                   "nthread": NTHREAD},
        "leak_diagnostic": {"shipped_kfold_folds_straddling_a_layout": straddle,
                            "n_folds": 4},
        "splitters": {},
    }

    for name, splits in splitters.items():
        t1 = time.time()
        believed = [cv_mae(p, X, y, splits) for p in candidates]
        bi = int(np.argmin(believed))
        results["splitters"][name] = {
            "believed_cv_mae": believed,
            "best_index": bi,
            "best_believed_cv_mae": believed[bi],
            "best_params": candidates[bi],
            "secs": round(time.time() - t1, 1),
        }
        print(f"{name}: best cand #{bi} believed CV MAE={believed[bi]:.6f} "
              f"({time.time() - t1:.0f}s)", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)

    # HONEST number: leave-one-LAYOUT-out MAE for each splitter's selected params.
    lolo = [(np.where(layouts != h)[0], np.where(layouts == h)[0]) for h in uniq]
    for name, res in results["splitters"].items():
        params = res["best_params"]
        per = {}
        for (tr, te), ho in zip(lolo, uniq):
            est = XGBRegressor(objective="reg:squarederror", verbosity=0,
                               random_state=SEED, n_jobs=NTHREAD, **params)
            est.fit(X[tr], y[tr])
            per[ho] = float(np.mean(np.abs(est.predict(X[te]) - y[te])))
        honest = float(np.mean(list(per.values())))
        res["lolo_mae_per_layout"] = per
        res["lolo_mae_mean"] = honest
        res["optimism"] = honest - res["best_believed_cv_mae"]
        print(f"{name}: honest LOLO MAE={honest:.6f} OPTIMISM={res['optimism']:+.6f}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)

    sel = {n: r["best_index"] for n, r in results["splitters"].items()}
    results["selection_indices"] = sel
    results["selection_differs"] = len(set(sel.values())) > 1
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nselection indices {sel} differs={results['selection_differs']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
