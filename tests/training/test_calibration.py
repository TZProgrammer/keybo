"""Tests for the first-finger calibration seam (PINKY-FIT, 2026-07-10).

The measured physics: same-hand same-row adjacent-finger bigrams are slower when
initiated by the OUTER finger, but within one layout that identity effect is collinear
with the per-ngram practice term, so a free fit mis-attributes it (sign inversion,
PREREGISTRATIONS 485be17). The seam fits the deltas IN-PIPELINE with the identifying
matched-cell estimator, subtracts them at training, records them in the sidecar, and
adds them back per position pair at serve.
"""

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.geometry import ROW_STAGGERED_30
from keybo.training.calibration import (
    CALIBRATION_VERSION,
    delta_log,
    finger_class,
    fit_first_finger_deltas,
)

geom = ROW_STAGGERED_30


# --- finger_class: exactly the estimator's matched configuration ---------------------------


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
    # ds (middle -> ring) and re (index -> middle) are the REFERENCE directions.
    assert finger_class(geom, (-3, 2), (-4, 2)) is None
    assert finger_class(geom, (-2, 3), (-3, 3)) is None


def test_cross_row_cross_hand_and_same_finger_are_not_calibrated():
    assert finger_class(geom, (-5, 2), (-4, 3)) is None  # cross-row (not measured)
    assert finger_class(geom, (-5, 2), (4, 2)) is None  # cross-hand
    assert finger_class(geom, (-1, 2), (-2, 2)) is None  # index-index = same finger
    assert finger_class(geom, (-3, 2), (-1, 2)) is None  # non-adjacent fingers


def test_middle_first_into_index_is_not_calibrated():
    # middle->index has no matched inner control (cols 1-2 are one finger): out of scope.
    assert finger_class(geom, (-3, 2), (-2, 2)) is None


# --- the estimator: fits a planted effect, practice-controlled -----------------------------


def _matched_world(gap_ms=40.0, n_per_cell=80, base=150.0):
    """Rows where outer-first cells are exactly ``gap_ms`` slower than the matched
    inner-first control into the same key (half gap for the ring family), plus filler
    same-row cells so the practice regression has support. Counts are equal within
    each matched pair, so the practice adjustment is ~zero by construction."""
    rng = np.random.default_rng(0)
    rows = []

    def add(positions, ngram, mean, n=n_per_cell):
        samples = [(70, int(mean + rng.integers(0, 8)), i, 50) for i in range(n)]
        rows.append(
            StrokeRow(
                layout="qwerty", positions=positions, ngram=ngram, frequency=n, samples=samples
            )
        )

    # matched pair 1: pinky->ring (as) vs middle->ring (ds), home row
    add(((-5, 2), (-4, 2)), "as", base + gap_ms)
    add(((-3, 2), (-4, 2)), "ds", base)
    # matched pair 2: ring->middle (we) vs index->middle (re), top row, half the gap
    add(((-4, 3), (-3, 3)), "we", base + gap_ms / 2)
    add(((-2, 3), (-3, 3)), "re", base)
    # filler same-row same-hand cells (practice-regression support), varied counts
    for k, (positions, ngram) in enumerate(
        [
            (((-4, 2), (-5, 2)), "sa"),
            (((-4, 2), (-3, 2)), "sd"),
            (((-3, 3), (-4, 3)), "ew"),
            (((-3, 3), (-2, 3)), "er"),
            (((4, 2), (3, 2)), "lk"),
            (((3, 2), (4, 2)), "kl"),
        ]
    ):
        add(positions, ngram, base + 3 * k, n=n_per_cell + 10 * k)
    return rows


def test_estimator_recovers_a_planted_gap():
    deltas = fit_first_finger_deltas(_matched_world(gap_ms=40.0), geom)
    assert deltas["pinky_first"] == pytest.approx(40.0, abs=8)
    assert deltas["ring_first"] == pytest.approx(20.0, abs=8)


def test_estimator_returns_empty_without_matched_cells():
    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram="th",
            frequency=5,
            samples=[(90, 140 + i, i, 50) for i in range(60)],
        )
    ]
    assert fit_first_finger_deltas(rows, geom) == {}


# --- delta_log: the offset in LOGRAT units --------------------------------------------------


def test_delta_log_zero_for_uncalibrated_or_unfitted():
    assert delta_log(None, 90.0, {"pinky_first": 42.0}) == 0.0
    assert delta_log("pinky_first", 90.0, {}) == 0.0  # class not fitted => no offset


def test_delta_log_roundtrip_recovers_the_ms_delta():
    wpm = 90.0
    deltas = {"pinky_first": 42.0}
    t_typ = 12000.0 / wpm * 1.15
    base = np.log(t_typ * wpm / 12000.0)
    served = np.exp(base + delta_log("pinky_first", wpm, deltas)) * 12000.0 / wpm
    assert served - t_typ == pytest.approx(42.0, abs=1e-6)


def test_calibration_version_exported():
    assert CALIBRATION_VERSION.startswith("2026-")


# --- train/serve integration ----------------------------------------------------------------


def test_train_bigram_model_fits_and_records_calibration():
    from keybo.training.train import train_bigram_model

    model = train_bigram_model(
        _matched_world(gap_ms=40.0), target_wpm=90, n_estimators=5, max_depth=2,
        calibration=True,
    )
    cal = model.metadata.extra["training"]["calibration"]
    assert cal["version"] == CALIBRATION_VERSION
    assert cal["deltas_ms"]["pinky_first"] == pytest.approx(40.0, abs=8)


def test_calibration_off_by_default():
    """CAL-REMOVE (2026-07-12): the seam is opt-in; the default recipe records None."""
    from keybo.training.train import train_bigram_model

    model = train_bigram_model(
        _matched_world(), target_wpm=90, n_estimators=5, max_depth=2
    )
    assert model.metadata.extra["training"]["calibration"] is None


def test_calibration_can_be_disabled():
    from keybo.training.train import train_bigram_model

    model = train_bigram_model(
        _matched_world(), target_wpm=90, n_estimators=5, max_depth=2, calibration=False
    )
    assert model.metadata.extra["training"]["calibration"] is None


def test_calibration_absent_when_data_has_no_matched_cells():
    from keybo.training.train import train_bigram_model

    rows = [
        StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram="th",
            frequency=5,
            samples=[(90, 140 + i, i, 50) for i in range(60)],
        )
    ]
    model = train_bigram_model(rows, target_wpm=90, n_estimators=5, max_depth=2)
    assert model.metadata.extra["training"]["calibration"] is None


def test_served_sign_end_to_end():
    """A calibrated model must serve outer-first SLOWER than inner-first for colliding
    pairs — the whole point of the seam. The training data plants the gap; the collision
    means only the position-aware path can express it."""
    from keybo.features import bigram_features_from_positions
    from keybo.training.train import train_bigram_model

    model = train_bigram_model(
        _matched_world(gap_ms=40.0),
        target_wpm=90,
        n_estimators=10,
        max_depth=2,
        practice_term=False,
        calibration=True,
    )
    va = bigram_features_from_positions(geom, ((-5, 2), (-4, 2)), wpm=70.0).reshape(1, -1)
    vd = bigram_features_from_positions(geom, ((-3, 2), (-4, 2)), wpm=70.0).reshape(1, -1)
    assert np.array_equal(va, vd)  # the collision — feature path cannot separate
    t_as = model.predict_ms_at(va, ((-5, 2), (-4, 2)))
    t_ds = model.predict_ms_at(vd, ((-3, 2), (-4, 2)))
    assert float(t_as[0]) > float(t_ds[0]) + 20  # ~40ms planted, fitted, served


def test_table_scorer_applies_calibration_per_position_pair():
    from keybo.layouts import NAMED_LAYOUTS
    from keybo.scoring.table_scorer import TableBigramScorer
    from keybo.training.train import train_bigram_model

    model = train_bigram_model(
        _matched_world(gap_ms=40.0), target_wpm=90, n_estimators=10, max_depth=2,
        calibration=True,
    )
    sc = TableBigramScorer(model, {"th": 10}, target_wpm=90.0, chars=NAMED_LAYOUTS["qwerty"])
    positions = [*geom.slots, geom.space_position]
    pidx = {p: i for i, p in enumerate(positions)}
    t_pinky_first = sc._T[pidx[(-5, 2)], pidx[(-4, 2)]]
    t_middle_first = sc._T[pidx[(-3, 2)], pidx[(-4, 2)]]
    assert t_pinky_first > t_middle_first + 20
