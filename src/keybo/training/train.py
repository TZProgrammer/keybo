"""Train typing-time models from stroke data.

Feature vectors are built with the SAME pipeline used for scoring (via the
``*_from_positions`` entry points), using the physical positions recorded in the data. This
is the guarantee against train/serve skew: there is exactly one feature computation, and a
model's metadata records the ``FEATURE_VERSION`` it was trained under.

Each stroke row contributes one training example per WPM group: the target is the IQR-mean
of that group's durations, and the WPM enters as a feature so a single model spans the range.

Two corrections found by the real-data LOLO harness (2026-07-04/05, arm R1W — see
``agent-artifacts/OQ1-frequency-feature.md``) are built in for bigram training:

- **Additive practice term.** Frequent bigrams are fast partly because they are practiced —
  a layout-independent effect. Left unmodeled, the geometry model absorbs it (frequent
  bigrams' qwerty positions look "fast" — omitted-variable bias); modeled as a raw freq
  feature it becomes a per-position memorization key (98.7% of the data is qwerty). The fix:
  fit ``time = g(geometry, wpm) + b(bigram)`` by backfitting — b is the shrunk per-bigram
  mean residual, g refits on the residualized target ``y − b̂``. b is keyed by bigram
  identity, so it cancels exactly in layout comparisons; its only job is cleaning g's
  training target. Measured: pooled out-of-sample layout tau +0.667 → +1.0.
- **Layout balance weights.** Inverse-layout-share example weights (capped) stop the 98.7%
  qwerty majority from dominating the fit. Measured: composes with the practice term
  (rho/ceiling .928 → .931).

Both are on by default and controllable (``practice_term=False`` / ``layout_weights=False``).
The fitted practice term is stored in the model metadata (``extra["practice_term"]``) for
inspection; scoring deliberately ignores it (layout-independent ⇒ ranking-irrelevant).

A third correction (T-REL, 2026-07-10): models train in the **LOGRAT target space** —
``log(ms * wpm / 12000)``, time as a log-multiple of the typist's session-mean keystroke.
The ms label carries the typist-pace scale, so every geometry leaf must re-learn the wpm
hyperbola; pre-factoring it out cut cross-layout wmae 37% (bigram) with the rare-ngram
guards held (an additive DIFF control moved nothing ⇒ the multiplicative scale structure
is the mechanism), and the conditioned-trigram A/B reproduced it (wmae −30.7%, every
guard improved). ``predict()`` therefore returns log-ratios for these models; consumers
convert back via ``TypingModel.predict_ms`` / ``to_ms``. The practice term is backfit in
the training space, so for LOGRAT models the stored b values are log-scale.
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

#: practice-term shrinkage: b(ngram) = sum(w·resid) / (sum(w) + K). Pre-registered at 100
#: raw samples; the LOLO conclusion is robust across k ∈ [10, 1000] (audit 2026-07-05).
PRACTICE_SHRINKAGE_K = 100.0
#: backfit iterations (b and g alternate); 2 sufficed in every measured arm.
PRACTICE_BACKFIT_ITERS = 2
#: cap on inverse-layout-share weights, so a 64-participant layout can't dominate.
LAYOUT_WEIGHT_CAP = 50.0

#: target-space transforms: name -> (ms -> training target, given the example's wpm).
#: LOGRAT is the adopted bigram space (T-REL 2026-07-10, -37% cross-layout wmae).
_TARGET_TRANSFORMS = {
    "MS": lambda ms, wpm: ms,
    "LOGRAT": lambda ms, wpm: np.log(np.maximum(ms, 1.0) * wpm / 12000.0),
}


def _rows_to_examples(row: StrokeRow, geometry: Geometry, ngram: str):
    """Yield (feature_vector, target_time) per WPM group in a stroke row."""
    by_wpm: dict[int, list[int]] = defaultdict(list)
    for wpm, duration, _pid, _hold in row.samples:
        by_wpm[wpm].append(duration)

    for wpm, durations in by_wpm.items():
        target = iqr_average(durations)
        if ngram == "bigram":
            vec = bigram_features_from_positions(geometry, row.positions, wpm=wpm)
        else:
            vec = trigram_features_from_positions(geometry, row.positions, wpm=wpm)
        yield vec, target, len(durations)


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
    X, y, _ngrams, _layouts, _n = _build_matrix_full(
        rows, ngram=ngram, geometry=geometry, progress=progress
    )
    return X, y


def _build_matrix_full(rows, ngram, geometry, progress=False):
    """(X, y, example ngram ids, example layouts, example raw-sample counts)."""
    iterator = rows
    if progress:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="building features", unit="row", leave=False)
    features: list[np.ndarray] = []
    targets: list[float] = []
    ngrams: list[str] = []
    layouts: list[str] = []
    counts: list[float] = []
    for row in iterator:
        for vec, target, n in _rows_to_examples(row, geometry, ngram):
            features.append(vec)
            targets.append(target)
            ngrams.append(row.ngram)
            layouts.append(row.layout)
            counts.append(float(n))
    if not features:
        return (
            np.empty((0, 0)),
            np.empty((0,)),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            np.empty((0,)),
        )
    return (
        np.vstack(features),
        np.array(targets, dtype=np.float64),
        np.array(ngrams, dtype=object),
        np.array(layouts, dtype=object),
        np.array(counts, dtype=np.float64),
    )


def layout_balance_weights(layouts: np.ndarray, cap: float = LAYOUT_WEIGHT_CAP) -> np.ndarray:
    """Inverse-layout-share example weights, capped, normalized to mean 1."""
    share: dict[str, float] = defaultdict(float)
    for la in layouts:
        share[la] += 1.0
    total = float(len(layouts))
    w = np.array([min(cap, total / (len(share) * share[la])) for la in layouts])
    return w / w.mean()


def fit_practice_term(
    ngrams: np.ndarray,
    residuals: np.ndarray,
    counts: np.ndarray,
    k: float = PRACTICE_SHRINKAGE_K,
) -> dict[str, float]:
    """Shrunk per-ngram mean residual: b = Σ(count·resid) / (Σcount + k).

    ``counts`` are raw-sample counts per example, so a bigram seen 10,000 times gets its
    full residual while a bigram seen 5 times is shrunk hard toward 0 (no practice claim
    from noise).
    """
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    for ng, r, c in zip(ngrams, residuals, counts, strict=True):
        num[ng] += c * r
        den[ng] += c
    return {ng: num[ng] / (den[ng] + k) for ng in num}


def _train(
    rows,
    ngram,
    target_wpm,
    wpm_range,
    geometry,
    progress=False,
    practice_term=True,
    layout_weights=True,
    target_space="MS",
    **params,
) -> XGBoostTypingModel:
    from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES

    target_space = str(target_space).upper()
    if target_space not in _TARGET_TRANSFORMS:
        raise ValueError(
            f"unknown target_space {target_space!r} (known: {sorted(_TARGET_TRANSFORMS)})"
        )

    X, y, ngrams, layouts, counts = _build_matrix_full(
        rows, ngram=ngram, geometry=geometry, progress=progress
    )
    names = BIGRAM_FEATURE_NAMES if ngram == "bigram" else TRIGRAM_FEATURE_NAMES
    metadata = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=names,
        wpm_range=wpm_range,
        ngram=ngram,
    )

    # The training target lives in the model's target space; the per-example wpm needed
    # for the transform is the schema's own wpm column (the same one to_ms reads back).
    if len(y):
        wpm_col = np.maximum(X[:, names.index("wpm")], 1.0)
        y = _TARGET_TRANSFORMS[target_space](y, wpm_col)

    weights = layout_balance_weights(layouts) if layout_weights and len(y) else None

    def fit(target):
        model = XGBoostTypingModel(metadata, **params)
        model._regressor.fit(X, target, sample_weight=weights)
        model._fitted = True
        return model

    if not practice_term or not len(y):
        model = fit(y)
        model.metadata.extra["training"] = {
            "target_space": target_space,
            "practice_term": None,
            "layout_weights": bool(weights is not None),
        }
        return model

    # Backfit: b absorbs the shrunk per-ngram residual mean; g refits on y - b. Runs in
    # the target space, so a LOGRAT model's b values are log-ratios (stored at higher
    # precision — rounding log-scale values to 3 decimals would destroy them).
    model = fit(y)
    bmap: dict[str, float] = {}
    for _ in range(PRACTICE_BACKFIT_ITERS):
        bmap = fit_practice_term(ngrams, y - model.predict(X), counts)
        bvec = np.array([bmap.get(ng, 0.0) for ng in ngrams])
        model = fit(y - bvec)
    b_digits = 3 if target_space == "MS" else 6
    model.metadata.extra["training"] = {
        "target_space": target_space,
        "practice_term": {
            "shrinkage_k": PRACTICE_SHRINKAGE_K,
            "backfit_iters": PRACTICE_BACKFIT_ITERS,
            "n_ngrams": len(bmap),
            "values": {ng: round(float(v), b_digits) for ng, v in bmap.items()},
        },
        "layout_weights": bool(weights is not None),
    }
    return model


def train_bigram_model(
    rows: list[StrokeRow],
    target_wpm: float,
    wpm_range: tuple[int, int] = (60, 120),
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    practice_term: bool = True,
    layout_weights: bool = True,
    target_space: str = "LOGRAT",
    **params,
) -> XGBoostTypingModel:
    """Fit a bigram typing-time model from bistroke rows (R1W + LOGRAT recipe by default).

    ``progress`` is consumed here (feature-build bar), never forwarded into ``**params`` --
    XGBoost silently ignores unknown keyword params, so a leak would be invisible.
    """
    return _train(
        rows,
        "bigram",
        target_wpm,
        wpm_range,
        geometry,
        progress=progress,
        practice_term=practice_term,
        layout_weights=layout_weights,
        target_space=target_space,
        **params,
    )


def train_trigram_model(
    rows: list[StrokeRow],
    target_wpm: float,
    wpm_range: tuple[int, int] = (60, 120),
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    practice_term: bool = True,
    layout_weights: bool = True,
    target_space: str = "LOGRAT",
    **params,
) -> XGBoostTypingModel:
    """Fit a trigram typing-time model from tristroke rows. See train_bigram_model.

    LOGRAT by default per the conditioned-trigram A/B (2026-07-10): wmae −30.7% with
    umae/dec3/taus all improved — the bigram mechanism carries.
    """
    return _train(
        rows,
        "trigram",
        target_wpm,
        wpm_range,
        geometry,
        progress=progress,
        practice_term=practice_term,
        layout_weights=layout_weights,
        target_space=target_space,
        **params,
    )
