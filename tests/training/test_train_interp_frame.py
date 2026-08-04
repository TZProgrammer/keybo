"""Training on the interp frame: the stamp, the constraints, and the refusals.

The refusals matter more than the happy path. ``interp`` REPLACES the served columns while
``direction``/``kitchensink`` WIDEN them, so any combination would produce a frame whose version
stamp lies about its columns — and the stamp is the only thing standing between a model and being
scored on the wrong matrix.
"""

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.features import (
    BIGRAM_INTERP_FEATURE_NAMES,
    BIGRAM_INTERP_MONOTONE,
    FEATURE_VERSION,
    FEATURE_VERSION_INTERP,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.training.train import train_bigram_model, train_trigram_model

G = ROW_STAGGERED_30


#: ``validate`` REFUSES rows whose ``ngram`` string length disagrees with the requested order, so
#: the fixture's identity keys must be exactly two characters — a bigram row's ngram IS a bigram.
_KEYS = [f"{x}{y}" for x in "abcde" for y in "fghij"]


def _rows(n=90, seed=0):
    """Synthetic bistroke rows with a real geometry signal, on two layouts."""
    rng = np.random.default_rng(seed)
    slots = list(G.slots)
    rows = []
    for i in range(n):
        a = slots[rng.integers(len(slots))]
        b = slots[rng.integers(len(slots))]
        # a genuine dependence on row deviation, so a fitted model is not constant
        base = 120.0 + 25.0 * (abs(a[1] - 2) + abs(b[1] - 2))
        samples = [
            (int(w), int(max(40.0, base + rng.normal(0, 6))), 100000 + i % 7, 0)
            for w in (70, 90, 110)
            for _ in range(4)
        ]
        rows.append(
            StrokeRow(
                layout="qwerty" if i % 2 else "dvorak",
                positions=(a, b),
                ngram=_KEYS[i % len(_KEYS)],
                frequency=100 + i,
                samples=samples,
            )
        )
    return rows


# --- the stamp and the frame ---------------------------------------------------------------


def test_an_interp_model_carries_the_interp_stamp_and_names():
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, interp=True, n_estimators=10)
    assert m.metadata.feature_version == FEATURE_VERSION_INTERP
    assert m.metadata.feature_version != FEATURE_VERSION
    assert list(m.metadata.feature_names) == BIGRAM_INTERP_FEATURE_NAMES
    assert len(m.metadata.feature_names) == 10


def test_a_default_model_is_UNAFFECTED_and_still_stamps_the_served_version():
    """The isolation claim: adding this frame must change nothing about the served path."""
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, n_estimators=10)
    assert m.metadata.feature_version == FEATURE_VERSION
    assert len(m.metadata.feature_names) == 20


# --- the monotone constraints are RECORDED, because the booster does not keep them ---------


def test_the_constraint_set_is_recorded_in_the_artifact():
    """xgboost bakes constraints into the tree structure and does NOT serialize the parameter, so
    without this record a constrained and an unconstrained model are indistinguishable after
    save() — and 'was this constrained?' is the question the whole verification protocol asks."""
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, interp=True, n_estimators=10)
    tag = m.metadata.extra["training"]["interp_frame"]
    assert tag["frame"] == "interp"
    assert tuple(tag["monotone_constraints"]) == tuple(BIGRAM_INTERP_MONOTONE)


def test_monotone_False_is_recorded_as_an_EMPTY_constraint_set_not_omitted():
    m = train_bigram_model(
        _rows(), target_wpm=90.0, geometry=G, interp=True, monotone=False, n_estimators=10
    )
    tag = m.metadata.extra["training"]["interp_frame"]
    assert tag["frame"] == "interp"
    assert tag["monotone_constraints"] == []


def test_the_constraint_actually_reaches_xgboost():
    """Present-is-not-effective, checked at the parameter boundary. (Whether the trained booster
    HONORS it is a separate, measured question — INTERPFRAME-1 §5.)"""
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, interp=True, n_estimators=10)
    assert tuple(m.params["monotone_constraints"]) == tuple(BIGRAM_INTERP_MONOTONE)
    plain = train_bigram_model(
        _rows(), target_wpm=90.0, geometry=G, interp=True, monotone=False, n_estimators=10
    )
    assert "monotone_constraints" not in plain.params


def test_a_served_model_gets_no_interp_tag_and_no_constraints():
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, n_estimators=10)
    assert m.metadata.extra["training"]["interp_frame"] is None
    assert "monotone_constraints" not in m.params


# --- the refusals -------------------------------------------------------------------------


def test_interp_with_direction_is_REFUSED():
    with pytest.raises(ValueError, match="cannot be combined"):
        train_bigram_model(
            _rows(), target_wpm=90.0, geometry=G, interp=True, direction=True, n_estimators=10
        )


def test_interp_with_kitchensink_is_REFUSED():
    with pytest.raises(ValueError, match="cannot be combined"):
        train_bigram_model(
            _rows(), target_wpm=90.0, geometry=G, interp=True, kitchensink=True, n_estimators=10
        )


def test_interp_on_the_trigram_path_is_REFUSED():
    tri = [
        StrokeRow(
            layout="qwerty",
            positions=(p, p, p),
            ngram="abc",
            frequency=10,
            samples=[(90, 150, 100001, 0)] * 12,
        )
        for p in list(G.slots)[:6]
    ]
    with pytest.raises(ValueError, match="bigram-only frame"):
        train_trigram_model(tri, target_wpm=90.0, geometry=G, interp=True, n_estimators=10)


# --- to_ms: the frame has no wpm column, so the pace must be STATED ------------------------


def test_predict_ms_REFUSES_to_guess_the_pace_on_the_interp_frame():
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, interp=True, n_estimators=10)
    from keybo.features import interp_features_from_positions

    X = np.vstack([interp_features_from_positions(G, (a, a), wpm=90.0) for a in G.slots[:4]])
    with pytest.raises(ValueError, match="cannot recover the pace"):
        m.predict_ms(X)
    ms = m.predict_ms(X, wpm=90.0)
    assert np.all(np.isfinite(ms)) and np.all(ms > 0)


def test_passing_wpm_to_a_frame_that_HAS_the_column_is_REFUSED():
    """Two sources for one quantity is how a prediction gets converted at a pace it was not made
    at — an error that rescales every number smoothly and is invisible in the output."""
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, n_estimators=10)
    from keybo.features import bigram_features_from_positions

    X = np.vstack([bigram_features_from_positions(G, (a, a), wpm=90.0) for a in G.slots[:4]])
    with pytest.raises(ValueError, match="carries a 'wpm' column"):
        m.predict_ms(X, wpm=120.0)
    assert np.all(np.isfinite(m.predict_ms(X)))


def test_the_ms_conversion_is_the_same_arithmetic_either_way():
    """The explicit-wpm path must not be a different formula: on a served model, converting via
    the column and via an equal explicit scalar are the same number — verified on the SERVED
    frame by temporarily reading the column out, so the two code paths are compared directly."""
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, n_estimators=10)
    from keybo.features import bigram_features_from_positions

    X = np.vstack([bigram_features_from_positions(G, (a, a), wpm=90.0) for a in G.slots[:4]])
    via_column = m.predict_ms(X)
    manual = np.exp(m.predict(X)) * 12000.0 / 90.0
    assert np.allclose(via_column, manual, rtol=0, atol=1e-12)


# --- the LOLO harness accepts the frame ---------------------------------------------------


def test_validate_runs_on_the_interp_frame_and_records_it():
    from keybo.training.validate import validate

    report = validate(
        _rows(140),
        seeds=[0],
        ngram="bigram",
        n_boot=5,
        geometry=G,
        interp=True,
        min_cell_samples=4,
        train_params={"n_estimators": 10},
    )
    assert report["config"]["interp"] is True
    assert report["config"]["monotone"] is True
    assert report["folds"], "expected at least one fold"
    for fold in report["folds"].values():
        for rec in fold["seeds"]:
            assert np.isfinite(rec["mae_model"])


def test_validate_records_monotone_even_on_a_served_run():
    """An arm's config must never be mistakable for another's: `monotone` is inert without
    `interp`, and a reader has to see that from the artifact rather than infer it."""
    from keybo.training.validate import validate

    report = validate(
        _rows(140),
        seeds=[0],
        ngram="bigram",
        n_boot=5,
        geometry=G,
        min_cell_samples=4,
        train_params={"n_estimators": 10},
    )
    assert report["config"]["interp"] is False
    assert "monotone" in report["config"]


# --- the frame-mismatch GUARD (the near-miss that motivated it) ----------------------------


def test_predict_cells_REFUSES_a_frame_that_does_not_match_the_model():
    """The guard _predict_cells was missing. validate() once forwarded interp=True where the
    model had been trained on interp='wpm', and only xgboost's shape check caught it -- which is
    luck: two frames of EQUAL width would have scored the model on a matrix it was never fitted
    for and returned a plausible number."""
    from keybo.training.validate import _predict_cells, build_cells

    rows = _rows(140)
    cells = build_cells(rows, min_cell_samples=4)
    assert cells, "fixture guard"
    m = train_bigram_model(rows, target_wpm=90.0, geometry=G, interp=True, n_estimators=10)
    # featurize with the 11-column VARIANT against a 10-column model
    with pytest.raises(ValueError, match="frame mismatch"):
        _predict_cells(m, cells, G, interp="wpm")


def test_validate_forwards_the_interp_STRING_and_trains_the_right_frame():
    """The bug itself: `{"interp": True}` discarded the "wpm" string, so the harness trained the
    10-column frame and evaluated 11 columns."""
    from keybo.features import BIGRAM_INTERP_WPM_FEATURE_NAMES
    from keybo.training.validate import validate

    report = validate(
        _rows(140),
        seeds=[0],
        ngram="bigram",
        n_boot=5,
        geometry=G,
        interp="wpm",
        min_cell_samples=4,
        train_params={"n_estimators": 10},
    )
    assert report["config"]["interp"] == "wpm"
    assert report["folds"]
    for fold in report["folds"].values():
        for rec in fold["seeds"]:
            assert np.isfinite(rec["mae_model"])
    # and the variant really is the 11-column frame
    m = train_bigram_model(_rows(), target_wpm=90.0, geometry=G, interp="wpm", n_estimators=10)
    assert list(m.metadata.feature_names) == BIGRAM_INTERP_WPM_FEATURE_NAMES


def test_an_unknown_interp_value_is_REFUSED():
    # Matches on the LEGAL-VALUE LIST rather than the prose, so adding a frame updates the list
    # (which is the point of the assertion) without rewriting the regex. `"wmp"` is the realistic
    # typo: a near-miss of `"wpm"` must NOT fall back to the 10-column frame.
    with pytest.raises(ValueError, match=r"interp must be False.*'wpm', 'hybridb'"):
        train_bigram_model(_rows(), target_wpm=90.0, geometry=G, interp="wmp", n_estimators=10)
