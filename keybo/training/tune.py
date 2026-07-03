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
