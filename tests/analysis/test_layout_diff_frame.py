"""layout_diff trigram_frame: the F3 double-count regression (audit 2026-07-13).

A trigram model trained on ABSOLUTE times must not have T2 added on top (measured
1.48x inflation on the shipped models); a conditioned model must. The frame is the
caller's declaration — these tests pin both constructions and the fail-loud default.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis.layout_diff import diff_layouts
from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION, TRIGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel

QWERTY = "qwertyuiopasdfghjkl;zxcvbnm,./"
SWAPPED = "qwertyuiopasdfghjkl;zxcvbnm,/."  # . and / swapped


def _tiny(ngram: str, names) -> XGBoostTypingModel:
    rng = np.random.default_rng(0)
    X = rng.random((300, len(names)))
    y = 50.0 + 30.0 * X[:, 0] + rng.normal(0, 0.5, 300)
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=list(names),
        wpm_range=(0, 300),
        ngram=ngram,
        extra={"training": {"target_space": "MS"}},
    )
    return XGBoostTypingModel(meta, n_estimators=10, max_depth=2).fit(X, y)


@pytest.fixture(scope="module")
def models():
    return (
        [_tiny("bigram", BIGRAM_FEATURE_NAMES)],
        [_tiny("trigram", TRIGRAM_FEATURE_NAMES)],
    )


def _diff(models, frame):
    bi, tri = models
    return diff_layouts(
        Layout(QWERTY, ROW_STAGGERED_30),
        Layout(SWAPPED, ROW_STAGGERED_30),
        bi,
        {"the": 100, "and": 50},
        trigram_models=tri,
        bigram_freqs={"th": 100, "he": 90},
        trigram_frame=frame,
    )


def test_frame_is_required_for_trigram_diffs(models):
    with pytest.raises(ValueError, match="trigram_frame"):
        _diff(models, None)


def test_absolute_frame_excludes_t2(models):
    """conditioned total = absolute total + the T2 mass — the exact F3 double-count."""
    da = _diff(models, "absolute")
    dc = _diff(models, "conditioned")
    # same models, so the conditioned construction is strictly larger by the T2 sums
    assert dc.total_a > da.total_a
    for ia, ic in zip(da.top(5), dc.top(5), strict=False):
        if ia.ngram == ic.ngram:
            assert ic.t_a_ms > ia.t_a_ms
