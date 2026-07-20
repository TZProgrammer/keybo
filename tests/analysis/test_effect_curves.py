from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from keybo.analysis.effect_curves import EffectCurves, compute_effect_curves
from keybo.features.schema import BIGRAM_FEATURE_NAMES
from keybo.geometry import Geometry
from keybo.models.base import TypingModel


class _ZeroShapBooster:
    def predict(self, matrix, pred_contribs=False):
        assert pred_contribs is True
        return np.zeros((matrix.num_row(), len(BIGRAM_FEATURE_NAMES) + 1))


class _FakeModel:
    metadata = SimpleNamespace(ngram="bigram", feature_names=list(BIGRAM_FEATURE_NAMES))
    target_space = "MS"
    _regressor = SimpleNamespace(get_booster=lambda: _ZeroShapBooster())

    def predict_ms(self, features):
        same_hand = BIGRAM_FEATURE_NAMES.index("same_hand")
        return 100.0 + 20.0 * features[:, same_hand]


class _CalibratedLogratModel:
    metadata = SimpleNamespace(
        ngram="bigram",
        feature_names=list(BIGRAM_FEATURE_NAMES),
        extra={
            "training": {
                "target_space": "LOGRAT",
                "calibration": {"deltas_ms": {"pinky_first": 62.0}},
            }
        },
    )
    target_space = "LOGRAT"
    _regressor = SimpleNamespace(get_booster=lambda: _ZeroShapBooster())
    to_ms = TypingModel.to_ms

    def predict(self, features):
        wpm = features[:, BIGRAM_FEATURE_NAMES.index("wpm")]
        return np.log(138.0 * wpm / 12000.0)

    def predict_ms(self, features):
        return self.to_ms(self.predict(features), features)


def test_effect_curves_pin_class_means_contrasts_and_percent_units():
    geometry = Geometry(slots=((-5, 2), (-4, 2), (1, 2)))

    curves = compute_effect_curves([_FakeModel()], wpms=[90.0], geometry=geometry)

    assert curves.class_mean_ms["alternate"] == pytest.approx([100.0])
    assert curves.class_mean_ms["same_hand_other"] == pytest.approx([120.0])
    assert curves.contrast_ms["alternate"] == pytest.approx([0.0])
    assert curves.contrast_ms["same_hand_other"] == pytest.approx([20.0])
    assert curves.contrast_pct()["same_hand_other"] == pytest.approx([20.0])
    assert curves.shap_ms["same_hand_other"] == pytest.approx([0.0])
    assert curves.n_pairs["alternate"] == 4
    assert curves.n_pairs["same_hand_other"] == 2
    assert curves.weighted_by == "uniform"
    assert curves.to_dict()["contrast_vs_alternate_pct"]["same_hand_other"] == pytest.approx([20.0])


def test_contrast_pct_preserves_sign_and_uses_each_wpm_reference():
    curves = EffectCurves(
        wpms=[60.0, 120.0],
        class_mean_ms={"alternate": [100.0, 200.0]},
        contrast_ms={"sfb": [25.0, -50.0]},
        shap_ms={},
        n_pairs={},
    )
    assert curves.contrast_pct() == {"sfb": [25.0, -25.0]}


def test_effect_curves_apply_position_calibration_to_the_served_surface():
    geometry = Geometry(slots=((-5, 2), (-4, 2), (1, 2)))

    curves = compute_effect_curves(
        [_CalibratedLogratModel()],
        wpms=[100.0],
        geometry=geometry,
    )

    # The same-hand class has q->w (pinky-first, 200 ms) and w->q
    # (uncalibrated, 138 ms). Alternate pairs remain at the 138 ms baseline.
    assert curves.class_mean_ms["alternate"] == pytest.approx([138.0])
    assert curves.class_mean_ms["same_hand_other"] == pytest.approx([169.0])
    assert curves.contrast_ms["same_hand_other"] == pytest.approx([31.0])
