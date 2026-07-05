"""Hyperparameter search for the XGBoost typing-time model.

A randomized search over the usual XGBoost knobs, scored by cross-validated MAE. Returns the
best parameter dict (which the ``train`` step can then be given). Uses the modern ``device``
parameter, not the removed ``gpu_hist``/``gpu_id`` (bug #12).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

_PARAM_DISTRIBUTIONS = {
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.005, 0.1),
    "min_child_weight": randint(1, 6),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 0.5),
    "reg_alpha": uniform(0, 2),
    "reg_lambda": uniform(0, 2),
    "n_estimators": randint(200, 900),
}


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 50,
    cv: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Randomized search for XGBoost params minimizing cross-validated MAE."""
    base = XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,
        device=device,
        random_state=seed,
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)
    return dict(search.best_params_)


def tune_lolo(
    rows,
    candidates: list[dict],
    seeds: list[int],
    ngram: str = "bigram",
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
) -> tuple[dict, list[tuple[dict, float]]]:
    """Hyperparameter selection scored by TRANSFER, not fit (backlog C1).

    The randomized-CV search above optimizes pooled CV MAE, which *rewards* memorizing
    training-family idiosyncrasies — the exact failure the LOLO harness exists to catch
    (measured: default depth-5 lost ~0.06 rho/ceiling to depth-3 while winning CV fit).
    This selector runs each candidate through the leave-one-layout-out harness and scores
    it by mean held-out rho/ceiling, GATED on the pooled layout-ranking tau staying at
    the maximum achieved by any candidate (a candidate that wins rho by breaking the
    ranking loses; the tau gate is the same principle as the arm-matrix decision rule).

    Returns (best_params, leaderboard) with the leaderboard sorted best-first as
    (params, gated_score) pairs. Candidates are explicit — reproducible and testable;
    callers wanting a random search generate the candidate list themselves.
    """
    from keybo.training.validate import validate

    results: list[tuple[dict, float, float]] = []  # (params, mean_frac, min_tau)
    for params in candidates:
        report = validate(
            rows,
            seeds=seeds,
            ngram=ngram,
            wpm_lo=wpm_lo,
            wpm_hi=wpm_hi,
            bucket_width=bucket_width,
            min_cell_samples=min_cell_samples,
            n_boot=10,  # ceilings are shared context here, not the contest
            train_params=params,
        )
        fracs = [
            m["rho_frac_ceiling"]
            for fold in report["folds"].values()
            for m in fold["seeds"]
            if m["rho_frac_ceiling"] is not None
        ]
        taus = [p["tau_heldout"] for p in report["pooled"]]
        mean_frac = float(np.mean(fracs)) if fracs else float("-inf")
        min_tau = float(min(taus)) if taus else float("-inf")
        results.append((params, mean_frac, min_tau))

    best_tau = max(r[2] for r in results)
    # tau gate: only candidates achieving the best observed ranking quality compete on rho.
    gated = [(p, f if t >= best_tau - 1e-9 else float("-inf")) for p, f, t in results]
    leaderboard = sorted(gated, key=lambda pf: -pf[1])
    return leaderboard[0][0], leaderboard
