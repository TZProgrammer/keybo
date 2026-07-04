"""Train typing-time models from stroke data.

Feature vectors are built with the SAME pipeline used for scoring (via the
``*_from_positions`` entry points), using the physical positions recorded in the data. This
is the guarantee against train/serve skew: there is exactly one feature computation, and a
model's metadata records the ``FEATURE_VERSION`` it was trained under.

Each stroke row contributes one training example per WPM group: the target is the IQR-mean
of that group's durations, and the WPM enters as a feature so a single model spans the range.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from keybo.data.strokes import StrokeRow, iqr_average
from keybo.features import (
    bigram_features_from_positions,
    trigram_features_from_positions,
)
from keybo.features.schema import FEATURE_VERSION
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel


def _rows_to_examples(row: StrokeRow, geometry: Geometry, ngram: str):
    """Yield (feature_vector, target_time) per WPM group in a stroke row."""
    by_wpm: dict[int, list[int]] = defaultdict(list)
    for wpm, duration in row.samples:
        by_wpm[wpm].append(duration)

    for wpm, durations in by_wpm.items():
        target = iqr_average(durations)
        if ngram == "bigram":
            vec = bigram_features_from_positions(
                geometry, row.positions, freq=row.frequency, wpm=wpm
            )
        else:
            vec = trigram_features_from_positions(
                geometry, row.positions, tg_freq=row.frequency, wpm=wpm
            )
        yield vec, target


def build_training_matrix(
    rows: list[StrokeRow],
    ngram: str,
    target_wpm: float,
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn stroke rows into (X, y) using the shared feature pipeline.

    ``target_wpm`` is unused for the matrix itself (WPM is taken per-sample) but kept in the
    signature so callers pass their intended scoring WPM explicitly; it is recorded in model
    metadata by the ``train_*`` helpers. ``progress`` shows a tqdm bar over the stroke rows
    (feature building is the visible-latency stage on a real-sized table).
    """
    iterator = rows
    if progress:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="building features", unit="row", leave=False)
    features: list[np.ndarray] = []
    targets: list[float] = []
    for row in iterator:
        for vec, target in _rows_to_examples(row, geometry, ngram):
            features.append(vec)
            targets.append(target)
    if not features:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack(features), np.array(targets, dtype=np.float64)


def _train(
    rows, ngram, target_wpm, wpm_range, geometry, progress=False, **params
) -> XGBoostTypingModel:
    from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES

    X, y = build_training_matrix(
        rows, ngram=ngram, target_wpm=target_wpm, geometry=geometry, progress=progress
    )
    names = BIGRAM_FEATURE_NAMES if ngram == "bigram" else TRIGRAM_FEATURE_NAMES
    metadata = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=names,
        wpm_range=wpm_range,
        ngram=ngram,
    )
    model = XGBoostTypingModel(metadata, **params)
    model.fit(X, y)
    return model


def train_bigram_model(
    rows: list[StrokeRow],
    target_wpm: float,
    wpm_range: tuple[int, int] = (60, 120),
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    **params,
) -> XGBoostTypingModel:
    """Fit a bigram typing-time model from bistroke rows.

    ``progress`` is consumed here (feature-build bar), never forwarded into ``**params`` --
    XGBoost silently ignores unknown keyword params, so a leak would be invisible.
    """
    return _train(rows, "bigram", target_wpm, wpm_range, geometry, progress=progress, **params)


def train_trigram_model(
    rows: list[StrokeRow],
    target_wpm: float,
    wpm_range: tuple[int, int] = (60, 120),
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    **params,
) -> XGBoostTypingModel:
    """Fit a trigram typing-time model from tristroke rows. See train_bigram_model re progress."""
    return _train(rows, "trigram", target_wpm, wpm_range, geometry, progress=progress, **params)
