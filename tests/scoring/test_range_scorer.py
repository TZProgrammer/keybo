"""RangeBigramScorer: the band aggregations, and the degeneracy guard that motivates them."""

from __future__ import annotations

import numpy as np
import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.range_scorer import RangeBigramScorer
from keybo.scoring.table_scorer import TableBigramScorer

QWERTY = NAMED_LAYOUTS["qwerty"]
FREQS = {"th": 100, "he": 90, "in": 80, "er": 70, "e ": 60, " t": 50, "qz": 1}


def _model():
    """A tiny LOGRAT bigram model on the served 20-feature frame.

    Fitted, not stubbed: the scorer's whole job is to route real predictions through
    ``predict_ms``, and a stub returning ms directly would not exercise the LOGRAT ->
    ms conversion where the ``1/wpm`` factor (the degeneracy's cause) lives.
    """
    from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION

    rng = np.random.default_rng(0)
    n_feat = len(BIGRAM_FEATURE_NAMES)
    X = rng.normal(size=(400, n_feat))
    X[:, BIGRAM_FEATURE_NAMES.index("wpm")] = rng.uniform(40, 200, size=400)
    y = 0.1 * X[:, 0] + 0.05 * X[:, 3] + rng.normal(scale=0.01, size=400)
    meta = ModelMetadata(
        feature_version=FEATURE_VERSION,
        feature_names=list(BIGRAM_FEATURE_NAMES),
        wpm_range=(60, 120),
        ngram="bigram",
        extra={"training": {"target_space": "LOGRAT"}},
    )
    model = XGBoostTypingModel(meta, n_estimators=8, max_depth=3)
    model.fit(X, y)
    return model


def test_single_pace_band_reproduces_the_shipped_objective():
    """A length-1 mean band IS the shipped single-point objective, bit for bit.

    This is what lets the control arm run through the same code path as the range arms:
    if it were not exact, the control would be a second implementation of the incumbent.
    """
    model = _model()
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    shipped = TableBigramScorer(model, FREQS, target_wpm=90.0, chars=QWERTY)
    for aggregation in ("mean", "endpoint"):
        band = RangeBigramScorer(model, FREQS, [90.0], aggregation=aggregation, chars=QWERTY)
        assert band.fitness(lay) == shipped.fitness(lay)


def test_mean_is_the_mean_of_the_per_pace_totals():
    model = _model()
    lay = Layout(QWERTY, ROW_STAGGERED_30)
    wpms = [90.0, 100.0, 110.0, 120.0]
    band = RangeBigramScorer(model, FREQS, wpms, aggregation="mean", chars=QWERTY)
    singles = [TableBigramScorer(model, FREQS, target_wpm=w, chars=QWERTY).fitness(lay) for w in wpms]
    assert band.fitness(lay) == pytest.approx(float(np.mean(singles)))
    assert band.per_wpm(lay) == pytest.approx(np.array(singles))


def test_minimax_refuses_to_run_unnormalized():
    """The degeneracy guard: total_ms falls with wpm, so a raw max collapses to min(band).

    Refusing is the point — an un-normalized minimax would silently be the single-point
    objective at the band's lowest pace while presenting as a band objective.
    """
    model = _model()
    with pytest.raises(ValueError, match="monotone decreasing"):
        RangeBigramScorer(model, FREQS, [90.0, 120.0], aggregation="minimax", chars=QWERTY)


def test_reference_is_refused_for_non_minimax():
    """A mean of reference-normalized ratios is not a time; the two knobs are not independent."""
    model = _model()
    with pytest.raises(ValueError, match="only meaningful for minimax"):
        RangeBigramScorer(
            model, FREQS, [90.0, 120.0], aggregation="mean", chars=QWERTY, reference=QWERTY
        )


def test_minimax_is_the_max_reference_normalized_ratio():
    model = _model()
    lay = Layout(NAMED_LAYOUTS["colemak"], ROW_STAGGERED_30)
    ref = Layout(QWERTY, ROW_STAGGERED_30)
    wpms = [90.0, 100.0, 110.0, 120.0]
    band = RangeBigramScorer(
        model, FREQS, wpms, aggregation="minimax", chars=QWERTY, reference=QWERTY
    )
    singles = [TableBigramScorer(model, FREQS, target_wpm=w, chars=QWERTY) for w in wpms]
    expected = max(s.fitness(lay) / s.fitness(ref) for s in singles)
    assert band.fitness(lay) == pytest.approx(expected)
    # the reference board itself sits exactly at 1.0 under its own normalization
    assert band.fitness(ref) == pytest.approx(1.0)


def test_reference_normalization_cannot_reorder_within_a_pace():
    """The divisor is constant per pace, so a fixed-pace ranking is untouched by it.

    This is the property that makes the normalization a reweighting ACROSS paces rather than
    a different objective at each pace.
    """
    model = _model()
    single = TableBigramScorer(model, FREQS, target_wpm=110.0, chars=QWERTY)
    band = RangeBigramScorer(
        model, FREQS, [110.0], aggregation="minimax", chars=QWERTY, reference=QWERTY
    )
    rng = np.random.default_rng(3)
    boards = []
    for _ in range(12):
        chars = list(QWERTY)
        rng.shuffle(chars)
        boards.append(Layout("".join(chars), ROW_STAGGERED_30))
    raw = [single.fitness(b) for b in boards]
    normed = [band.fitness(b) for b in boards]
    assert np.argsort(raw).tolist() == np.argsort(normed).tolist()


def test_endpoint_takes_exactly_one_pace():
    model = _model()
    with pytest.raises(ValueError, match="endpoint takes exactly one pace"):
        RangeBigramScorer(model, FREQS, [90.0, 120.0], aggregation="endpoint", chars=QWERTY)


def test_rejects_empty_band_unknown_aggregation_and_nonpositive_pace():
    model = _model()
    with pytest.raises(ValueError, match="at least one pace"):
        RangeBigramScorer(model, FREQS, [], chars=QWERTY)
    with pytest.raises(ValueError, match="unknown aggregation"):
        RangeBigramScorer(model, FREQS, [90.0], aggregation="median", chars=QWERTY)
    with pytest.raises(ValueError, match="must be > 0"):
        RangeBigramScorer(model, FREQS, [90.0, 0.0], chars=QWERTY)


def test_describe_names_the_objective():
    model = _model()
    mean = RangeBigramScorer(model, FREQS, [90.0, 120.0], aggregation="mean", chars=QWERTY)
    assert mean.describe() == "mean of total_ms over wpm in {90/120}"
    mm = RangeBigramScorer(
        model, FREQS, [90.0, 120.0], aggregation="minimax", chars=QWERTY, reference=QWERTY
    )
    assert "minimax" in mm.describe() and "90/120" in mm.describe()


def test_total_ms_is_monotone_decreasing_in_wpm_on_this_model():
    """The empirical fact the minimax guard encodes, asserted on a real fitted model.

    If a future model breaks monotonicity the guard's rationale weakens, and this test is
    where that shows up rather than in a silently-degenerate optimizer arm.
    """
    model = _model()
    scorers = [TableBigramScorer(model, FREQS, target_wpm=w, chars=QWERTY) for w in (90, 100, 110, 120)]
    rng = np.random.default_rng(11)
    for _ in range(8):
        chars = list(QWERTY)
        rng.shuffle(chars)
        lay = Layout("".join(chars), ROW_STAGGERED_30)
        totals = [s.fitness(lay) for s in scorers]
        assert totals == sorted(totals, reverse=True), totals
