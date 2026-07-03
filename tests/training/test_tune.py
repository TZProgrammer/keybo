"""Smoke tests for hyperparameter tuning (tiny search, synthetic data)."""

import numpy as np

from keybo.training.tune import tune_hyperparameters


def test_tune_returns_param_dict():
    rng = np.random.default_rng(0)
    X = rng.random((50, 6))
    y = X[:, 0] * 5 + rng.random(50)
    best = tune_hyperparameters(X, y, n_iter=2, cv=2, seed=0)
    assert isinstance(best, dict)
    # Params the search ranges over should appear.
    assert "max_depth" in best
    assert "n_estimators" in best
