"""XGBoost implementation of :class:`~keybo.models.base.TypingModel`.

Two things the original code got wrong are fixed here:

- Persistence uses XGBoost's native ``Booster`` JSON format, not Python ``pickle``. The old
  pickles only loaded because the unpickling namespace happened to contain the wrapper
  class, and they break across XGBoost versions.
- The regressor is configured with the modern ``device`` parameter. The old
  ``tree_method="gpu_hist"`` / ``gpu_id=...`` params were removed in XGBoost 2.x/3.x and now
  raise on construction.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from keybo.models.base import ModelMetadata, TypingModel

# Sensible defaults; callers (training) can override via kwargs.
_DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    # depth 3, not 5: the LOLO feature-arm matrix measured depth-5 trees spending their
    # extra splits memorizing training-family idiosyncrasies (held-out rho/ceiling .94 ->
    # ~1.0 at depth 3, layout tau intact; depth 2 over-regularizes and breaks tau).
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    # Explicit regularization (REG-LOLO, 2026-07-13): swept under the transfer-scored
    # (LOLO rho/ceiling, tau-gated) standard for the first time — every top-8 candidate
    # carried high gamma, i.e. split pruning is the lever the implicit recipe was
    # missing. Adopted: gated rho 1.0174 -> 1.0236, mean wmae +0.89% (inside the 0.91%
    # noise floor, dvorak fold IMPROVES), argmax plateau-stable (P10 regret -0.009%).
    "gamma": 0.957,
    "reg_alpha": 0.141,
    "reg_lambda": 0.011,
    "min_child_weight": 4,
    "verbosity": 0,
}


class XGBoostTypingModel(TypingModel):
    def __init__(self, metadata: ModelMetadata, device: str = "cpu", **params) -> None:
        super().__init__(metadata)
        self.params = {**_DEFAULT_PARAMS, **params, "device": device}
        self._regressor = xgb.XGBRegressor(**self.params)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostTypingModel:
        self._regressor.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("model is not fitted; call fit() or load() first")
        return np.asarray(self._regressor.predict(X), dtype=np.float64)

    def _save_artifact(self, artifact_path: str) -> None:
        # Native, version-portable JSON serialization of the booster.
        self._regressor.get_booster().save_model(artifact_path)

    @classmethod
    def _load_artifact(cls, artifact_path: str, metadata: ModelMetadata) -> XGBoostTypingModel:
        model = cls(metadata)
        # Public, version-stable API: load the booster JSON straight into the regressor.
        model._regressor.load_model(artifact_path)
        model._fitted = True
        return model
