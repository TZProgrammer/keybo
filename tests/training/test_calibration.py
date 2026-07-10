"""Tests for the first-finger calibration seam (PINKY-CAL, adopted 2026-07-10).

The measured physics: same-hand same-row adjacent-finger bigrams are slower when
initiated by the OUTER finger (pinky-first +42ms, ring-first +21ms), but within one
layout that identity effect is collinear with the per-ngram practice term, so a free
fit mis-attributes it (sign inversion, PREREGISTRATIONS 485be17). The seam installs
the probe-measured deltas as a FIXED offset: training subtracts it from calibrated
classes' targets, serving adds it back per position pair.
"""

import numpy as np
import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.training.calibration import (
    CALIBRATION_VERSION,
    DELTA_MS,
    delta_log,
    finger_class,
)

geom = ROW_STAGGERED_30


# --- finger_class: exactly the probe's measured configuration ------------------------------


def test_pinky_first_into_ring_same_row_is_calibrated():
    # qwerty 'as' shape: pinky (−5) -> ring (−4), home row; and the right-hand mirror.
    assert finger_class(geom, (-5, 2), (-4, 2)) == "pinky_first"
    assert finger_class(geom, (5, 2), (4, 2)) == "pinky_first"
    assert finger_class(geom, (5, 3), (4, 3)) == "pinky_first"  # top row (po)


def test_ring_first_into_middle_same_row_is_calibrated():
    # qwerty 'we' shape: ring -> middle.
    assert finger_class(geom, (-4, 3), (-3, 3)) == "ring_first"
    assert finger_class(geom, (4, 2), (3, 2)) == "ring_first"


def test_inner_first_directions_are_not_calibrated():
    # ds (middle -> ring) and re (index-ish -> middle) are the REFERENCE directions.
    assert finger_class(geom, (-3, 2), (-4, 2)) is None
    assert finger_class(geom, (-2, 3), (-3, 3)) is None


def test_cross_row_cross_hand_and_same_finger_are_not_calibrated():
    assert finger_class(geom, (-5, 2), (-4, 3)) is None  # cross-row (probe didn't measure)
    assert finger_class(geom, (-5, 2), (4, 2)) is None  # cross-hand
    assert finger_class(geom, (-1, 2), (-2, 2)) is None  # index-index = same finger
    assert finger_class(geom, (-3, 2), (-1, 2)) is None  # non-adjacent fingers


def test_middle_first_into_index_is_not_calibrated():
    # The probe measured pinky->ring and ring->middle families only; middle->index was
    # not measured, so it must NOT be extrapolated.
    assert finger_class(geom, (-3, 2), (-2, 2)) is None


# --- delta_log: the offset in LOGRAT units --------------------------------------------------


def test_delta_log_zero_for_uncalibrated():
    assert delta_log(None, 90.0) == 0.0


def test_delta_log_positive_and_ordered():
    # pinky delta (42ms) > ring delta (21ms) at any pace.
    for wpm in (50.0, 90.0, 130.0):
        dp = delta_log("pinky_first", wpm)
        dr = delta_log("ring_first", wpm)
        assert dp > dr > 0


def test_delta_log_roundtrip_recovers_the_ms_delta():
    # exp(base + delta_log) * 12000/wpm  -  exp(base) * 12000/wpm  ==  DELTA_MS
    # when base corresponds to the class-typical time used in the construction.
    wpm = 90.0
    t_typ = 12000.0 / wpm * 1.15
    base = np.log(t_typ * wpm / 12000.0)
    served = np.exp(base + delta_log("pinky_first", wpm)) * 12000.0 / wpm
    assert served - t_typ == pytest.approx(DELTA_MS["pinky_first"], abs=1e-6)


def test_calibration_version_exported():
    # Consumers record this in sidecars; it must exist and be dated.
    assert CALIBRATION_VERSION.startswith("2026-")


# --- train/serve integration ----------------------------------------------------------------


def test_train_bigram_model_applies_and_records_calibration():
    from keybo.data.strokes import StrokeRow
    from keybo.training.train import train_bigram_model

    rng = np.random.default_rng(0)
    rows = []
    # a calibrated-class row (pinky->ring home) and a plain row
    for positions, ngram in [(((-5, 2), (-4, 2)), "as"), (((-1, 3), (1, 2)), "th")]:
        samples = [(90, 140 + int(rng.integers(0, 30)), i, 50) for i in range(30)]
        rows.append(
            StrokeRow(
                layout="qwerty", positions=positions, ngram=ngram, frequency=30, samples=samples
            )
        )
    model = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2)
    training = model.metadata.extra["training"]
    assert training["calibration"] == CALIBRATION_VERSION


def test_calibration_can_be_disabled():
    from keybo.data.strokes import StrokeRow
    from keybo.training.train import train_bigram_model

    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram="th",
            frequency=5,
            samples=[(90, 140 + i, i, 50) for i in range(10)],
        )
    ]
    model = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2, calibration=False)
    assert model.metadata.extra["training"]["calibration"] is None


def test_served_sign_end_to_end():
    """The whole point: a calibrated model must serve outer-first SLOWER than
    inner-first for the same-row adjacent pair, even when trained on data where
    both had identical times."""
    from keybo.data.strokes import StrokeRow
    from keybo.features import bigram_features_from_positions
    from keybo.training.train import train_bigram_model

    rng = np.random.default_rng(1)
    rows = []
    # IDENTICAL time distributions for the pair — any served gap comes from the seam.
    for positions, ngram in [(((-5, 2), (-4, 2)), "as"), (((-3, 2), (-4, 2)), "ds")]:
        samples = [(90, 150 + int(rng.integers(0, 10)), i, 50) for i in range(60)]
        rows.append(
            StrokeRow(
                layout="qwerty", positions=positions, ngram=ngram, frequency=60, samples=samples
            )
        )
    model = train_bigram_model(
        rows, target_wpm=90, n_estimators=10, max_depth=2, practice_term=False
    )
    va = bigram_features_from_positions(geom, ((-5, 2), (-4, 2)), wpm=90.0).reshape(1, -1)
    vd = bigram_features_from_positions(geom, ((-3, 2), (-4, 2)), wpm=90.0).reshape(1, -1)
    # NOTE the two vectors are byte-identical (the collision) — predict_ms alone cannot
    # separate them; the position-aware serve path must.
    assert np.array_equal(va, vd)
    t_as = model.predict_ms_at(va, ((-5, 2), (-4, 2)))
    t_ds = model.predict_ms_at(vd, ((-3, 2), (-4, 2)))
    assert float(t_as[0]) > float(t_ds[0]) + 20  # ~42ms planted by the seam


def test_table_scorer_applies_calibration_per_position_pair():
    """TableBigramScorer entries for outer-first vs inner-first pairs must differ by
    ~DELTA even though the feature vectors collide."""
    from keybo.data.strokes import StrokeRow
    from keybo.layouts import NAMED_LAYOUTS
    from keybo.scoring.table_scorer import TableBigramScorer
    from keybo.training.train import train_bigram_model

    rng = np.random.default_rng(2)
    bigrams = ["th", "he", "an", "in", "er", "re", "on", "at"]
    rows = []
    for i in range(80):
        bg = bigrams[i % len(bigrams)]
        samples = [(90, 120 + int(rng.integers(0, 40)), i, 50)]
        rows.append(
            StrokeRow(
                layout="qwerty", positions=((-1, 3), (1, 2)), ngram=bg, frequency=5, samples=samples
            )
        )
    model = train_bigram_model(rows, target_wpm=90, n_estimators=10, max_depth=2)
    sc = TableBigramScorer(model, {"th": 10}, target_wpm=90.0, chars=NAMED_LAYOUTS["qwerty"])
    positions = [*geom.slots, geom.space_position]
    pidx = {p: i for i, p in enumerate(positions)}
    t_pinky_first = sc._T[pidx[(-5, 2)], pidx[(-4, 2)]]
    t_middle_first = sc._T[pidx[(-3, 2)], pidx[(-4, 2)]]
    assert t_pinky_first > t_middle_first + 20
