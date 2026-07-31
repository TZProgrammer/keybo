"""The training and validation paths must be able to reach the widened direction frame.

`DIRECTION-1` and `REDIRGATE-1` added order-aware direction columns to the FEATURE side and
deliberately stopped there. These tests pin the training side of that seam, and they exist
because the failure mode is silent in both directions:

- a widened matrix stamped with the NARROW ``FEATURE_VERSION`` loads fine against a served
  model's expectation while carrying columns that version never described — the exact
  train/serve skew `DIRECTION-1` refused to create;
- and a widened model evaluated through a narrow featurizer is fed a 20-column matrix where it
  wants 22, which XGBoost may absorb rather than refuse.

So the version stamp and the eval-path featurizer are both asserted, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
)
from keybo.training.train import build_training_matrix, train_bigram_model, train_trigram_model


def _bigram_rows() -> list[StrokeRow]:
    """Two layouts so the layout-balance weights and a LOLO split both have something to do."""
    rows = []
    specs = [
        ("qwerty", "as", ((-5, 2), (-4, 2))),
        ("qwerty", "sa", ((-4, 2), (-5, 2))),
        ("qwerty", "df", ((-3, 2), (-2, 2))),
        ("dvorak", "ae", ((-5, 2), (-4, 2))),
        ("dvorak", "ea", ((-4, 2), (-5, 2))),
        ("dvorak", "ou", ((-3, 2), (-2, 2))),
    ]
    for layout, ngram, positions in specs:
        samples = [(wpm, 100 + 3 * i, 500 + i, 0) for i, wpm in enumerate([60, 80, 100, 120] * 4)]
        rows.append(
            StrokeRow(
                layout=layout, positions=positions, ngram=ngram, frequency=1000, samples=samples
            )
        )
    return rows


def _trigram_rows() -> list[StrokeRow]:
    rows = []
    specs = [
        ("qwerty", "asd", ((-5, 2), (-4, 2), (-3, 2))),
        ("qwerty", "dsa", ((-3, 2), (-4, 2), (-5, 2))),
        ("dvorak", "aoe", ((-5, 2), (-4, 2), (-3, 2))),
        ("dvorak", "eoa", ((-3, 2), (-4, 2), (-5, 2))),
    ]
    for layout, ngram, positions in specs:
        samples = [(wpm, 150 + 4 * i, 700 + i, 0) for i, wpm in enumerate([60, 80, 100, 120] * 4)]
        rows.append(
            StrokeRow(
                layout=layout, positions=positions, ngram=ngram, frequency=900, samples=samples
            )
        )
    return rows


# --- the matrix -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ngram", "rows_fn", "narrow", "wide"),
    [
        ("bigram", _bigram_rows, BIGRAM_FEATURE_NAMES, BIGRAM_DIRECTION_FEATURE_NAMES),
        ("trigram", _trigram_rows, TRIGRAM_FEATURE_NAMES, TRIGRAM_DIRECTION_FEATURE_NAMES),
    ],
)
def test_build_training_matrix_widens_only_when_asked(ngram, rows_fn, narrow, wide):
    rows = rows_fn()
    X_narrow, _ = build_training_matrix(rows, ngram=ngram, target_wpm=90.0)
    X_wide, _ = build_training_matrix(rows, ngram=ngram, target_wpm=90.0, direction=True)

    assert X_narrow.shape[1] == len(narrow)
    assert X_wide.shape[1] == len(wide)
    assert X_wide.shape[0] == X_narrow.shape[0], "widening must not change the example count"


def test_the_widened_bigram_matrix_is_the_narrow_one_plus_the_new_columns():
    """The added columns are ADDITIVE: every shared column is bit-identical between frames.

    This is the property that makes the A/B a clean single-variable comparison. If widening
    perturbed an existing column, a rho delta could not be attributed to direction at all.
    """
    rows = _bigram_rows()
    X_narrow, _ = build_training_matrix(rows, ngram="bigram", target_wpm=90.0)
    X_wide, _ = build_training_matrix(rows, ngram="bigram", target_wpm=90.0, direction=True)

    for name in BIGRAM_FEATURE_NAMES:
        col_narrow = X_narrow[:, BIGRAM_FEATURE_NAMES.index(name)]
        col_wide = X_wide[:, BIGRAM_DIRECTION_FEATURE_NAMES.index(name)]
        assert np.array_equal(col_narrow, col_wide), f"column {name!r} moved when the frame widened"


def test_the_new_bigram_columns_actually_vary_in_the_widened_matrix():
    """A column of constant zeros would make the A/B vacuous while looking like it ran."""
    rows = _bigram_rows()
    X_wide, _ = build_training_matrix(rows, ngram="bigram", target_wpm=90.0, direction=True)
    for name in ("inwards_ordered", "outwards_ordered"):
        col = X_wide[:, BIGRAM_DIRECTION_FEATURE_NAMES.index(name)]
        assert len(np.unique(col)) > 1, f"{name} is constant — the widened arm tests nothing"


# --- the version stamp ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ngram", "rows_fn", "train_fn", "narrow", "wide"),
    [
        (
            "bigram",
            _bigram_rows,
            train_bigram_model,
            BIGRAM_FEATURE_NAMES,
            BIGRAM_DIRECTION_FEATURE_NAMES,
        ),
        (
            "trigram",
            _trigram_rows,
            train_trigram_model,
            TRIGRAM_FEATURE_NAMES,
            TRIGRAM_DIRECTION_FEATURE_NAMES,
        ),
    ],
)
def test_a_widened_model_is_stamped_with_the_direction_version(
    ngram, rows_fn, train_fn, narrow, wide
):
    """The whole point of the stamp: a widened model must never claim to be a served one.

    ``models.base`` hard-errors on a feature_version mismatch, so this stamp is what makes a
    widened artifact refuse to load where a narrow one is expected — instead of loading and
    silently scoring a frame whose columns mean something else.
    """
    rows = rows_fn()
    narrow_model = train_fn(rows, target_wpm=90.0)
    wide_model = train_fn(rows, target_wpm=90.0, direction=True)

    assert narrow_model.metadata.feature_version == FEATURE_VERSION
    assert wide_model.metadata.feature_version == FEATURE_VERSION_DIRECTION
    assert narrow_model.metadata.feature_names == narrow
    assert wide_model.metadata.feature_names == wide


def test_the_two_versions_can_never_be_equal():
    """If they collided the load-time guard could not tell the two frames apart at all."""
    assert FEATURE_VERSION_DIRECTION != FEATURE_VERSION


# --- the eval path ----------------------------------------------------------------------


def test_validate_predicts_widened_models_through_the_widened_featurizer():
    """A widened model must be EVALUATED on 22 columns too, or the A/B scores a skewed model.

    Without direction reaching ``_predict_cells``, the widened model is handed a narrow matrix
    at eval time. That is the same train/serve skew the version stamp exists to prevent, just
    relocated into the measurement harness — so it is pinned here rather than trusted.
    """
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.training.validate import _predict_cells, build_cells

    rows = _bigram_rows()
    cells = build_cells(rows, min_cell_samples=1)
    assert cells, "fixture produced no cells"

    wide_model = train_bigram_model(rows, target_wpm=90.0, direction=True)
    pred = _predict_cells(wide_model, cells, geometry=ROW_STAGGERED_30, direction=True)
    assert pred.shape == (len(cells),)
    assert np.all(np.isfinite(pred))


def test_validate_threads_direction_into_both_training_and_prediction():
    """End to end through the public harness: a widened LOLO run must complete and be stamped."""
    from keybo.training.validate import validate

    rows = _bigram_rows()
    report = validate(
        rows,
        seeds=[0],
        ngram="bigram",
        min_cell_samples=1,
        n_boot=5,
        direction=True,
        progress=False,
    )
    assert report["config"]["direction"] is True
    assert set(report["folds"]) == {"qwerty", "dvorak"}
    for fold in report["folds"].values():
        assert fold["seeds"], "a fold with no seed records would read as a pass"
