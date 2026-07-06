"""Tests for the leave-one-layout-out validation harness (OQ-5).

The harness is the thing that licenses (or revokes) every cross-layout claim, so its own
correctness is tested against synthetic worlds where the right answer is known:

- a LAWFUL world where duration is a clean function of geometry (distance) — the harness
  must report high transfer (rho near the noise ceiling, positive tau, model beats the
  distance baseline or at least matches it);
- a LAWLESS world where the held-out layout's times are random — the harness must NOT
  report transfer (rho near zero).

A harness that can't tell those apart would pass any model, which is worse than no harness.
"""

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.training.validate import (
    aggregate_layout_table,
    build_cells,
    leave_one_layout_out,
    split_half_ceiling,
    validate,
)

# Four fake "layouts": the same six ngrams live at different positions, so a
# geometry-lawful duration transfers across them while a memorized lookup cannot. Each
# layout's distance multiset shifts up by one (means 4.5 / 5.5 / 6.5 / 7.5), so the TRUE
# layout ranking is layA < layB < layC < layD — what the tau assertions check against.
_D = {  # cross-hand home-row position pair for each integer distance
    2: ((-1, 2), (1, 2)),
    3: ((-1, 2), (2, 2)),
    4: ((-2, 2), (2, 2)),
    5: ((-2, 2), (3, 2)),
    6: ((-3, 2), (3, 2)),
    7: ((-3, 2), (4, 2)),
    8: ((-4, 2), (4, 2)),
    9: ((-4, 2), (5, 2)),
    10: ((-5, 2), (5, 2)),
}
_NGRAMS = ["ab", "cd", "ef", "gh", "ij", "kl"]
_POSITIONS = {
    layout: {ng: _D[base + i] for i, ng in enumerate(_NGRAMS)}
    for base, layout in [(2, "layA"), (3, "layB"), (4, "layC"), (5, "layD")]
}


def _distance(positions):
    (x1, y1), (x2, y2) = positions
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _lawful_rows(seed=0, n_pids=8, samples_per_pid=6, lawless_layout=None):
    """duration = 60 + 25*distance + noise; optionally one layout is pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for layout, ngrams in _POSITIONS.items():
        for ngram, positions in ngrams.items():
            samples = []
            for pid in range(1, n_pids + 1):
                for _ in range(samples_per_pid):
                    wpm = int(rng.integers(65, 95))
                    if layout == lawless_layout:
                        dur = int(rng.integers(60, 260))
                    else:
                        dur = int(60 + 25 * _distance(positions) + rng.normal(0, 4))
                    samples.append((wpm, dur, pid, 50))
            rows.append(
                StrokeRow(
                    layout=layout,
                    positions=positions,
                    ngram=ngram,
                    frequency=100,
                    samples=samples,
                )
            )
    return rows


# --- splits -----------------------------------------------------------------------------


def test_leave_one_layout_out_partitions_rows():
    rows = _lawful_rows()
    train, test = leave_one_layout_out(rows, "layB")
    assert {r.layout for r in test} == {"layB"}
    assert "layB" not in {r.layout for r in train}
    assert len(train) + len(test) == len(rows)


def test_leave_one_layout_out_unknown_layout_raises():
    rows = _lawful_rows()
    with pytest.raises(ValueError, match="no rows"):
        leave_one_layout_out(rows, "colemak")


# --- cells ------------------------------------------------------------------------------


def test_build_cells_respects_wpm_band_and_floor():
    rows = _lawful_rows()
    cells = build_cells(rows, wpm_lo=60, wpm_hi=100, bucket_width=40, min_cell_samples=5)
    assert cells  # non-empty
    for c in cells:
        assert 60 <= c.wpm < 100
        assert c.n >= 5
    # A band that excludes everything yields no cells.
    assert build_cells(rows, wpm_lo=200, wpm_hi=240, bucket_width=40, min_cell_samples=1) == []


def test_build_cells_obs_matches_known_mean():
    # One row, constant duration -> obs must be exactly that duration.
    rows = [
        StrokeRow(
            layout="layA",
            positions=((-1, 2), (1, 2)),
            ngram="ab",
            frequency=10,
            samples=[(70, 150, pid, 50) for pid in range(1, 7)],
        )
    ]
    cells = build_cells(rows, wpm_lo=60, wpm_hi=100, bucket_width=40, min_cell_samples=5)
    assert len(cells) == 1
    assert cells[0].obs == pytest.approx(150.0)
    assert cells[0].layout == "layA"
    assert cells[0].ngram == "ab"


# --- noise ceiling ----------------------------------------------------------------------


def test_split_half_ceiling_high_for_consistent_data():
    # Times depend strongly on the cell and hardly on the participant -> halves agree.
    rows = _lawful_rows(n_pids=12, samples_per_pid=8)
    test = [r for r in rows if r.layout == "layA"]
    ceiling = split_half_ceiling(
        test, wpm_lo=60, wpm_hi=100, bucket_width=40, min_cell_samples=4, n_boot=20, seed=0
    )
    assert ceiling > 0.8


def test_split_half_ceiling_near_zero_for_noise():
    rows = _lawful_rows(n_pids=12, samples_per_pid=8, lawless_layout="layA")
    test = [r for r in rows if r.layout == "layA"]
    ceiling = split_half_ceiling(
        test, wpm_lo=60, wpm_hi=100, bucket_width=40, min_cell_samples=4, n_boot=20, seed=0
    )
    assert abs(ceiling) < 0.6  # pure noise: halves should not agree strongly


# --- layout table -----------------------------------------------------------------------


def test_aggregate_layout_table_weights_by_ngram():
    rows = _lawful_rows()
    cells = build_cells(rows, wpm_lo=60, wpm_hi=100, bucket_width=40, min_cell_samples=5)
    table = aggregate_layout_table(cells)
    assert set(table) == set(_POSITIONS)
    # Every layout aggregates over the same common ngram set here.
    for stats in table.values():
        assert set(stats) == set(_NGRAMS)


# --- end-to-end validate ----------------------------------------------------------------


def _fast_params():
    return {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.3}


def test_validate_reports_transfer_in_a_lawful_world():
    rows = _lawful_rows(n_pids=10, samples_per_pid=8)
    report = validate(
        rows,
        seeds=[0],
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=40,
        min_cell_samples=4,
        n_boot=20,
        train_params=_fast_params(),
    )
    # Every layout is a fold.
    assert set(report["folds"]) == set(_POSITIONS)
    for fold in report["folds"].values():
        m = fold["seeds"][0]
        # Geometry-lawful world: held-out rho should be strongly positive...
        assert m["rho"] > 0.6
        # ...and the fold tau over the 4 layouts must not invert the ranking.
        assert m["tau_all4"] > 0
    # Pooled held-out tau (each layout predicted by the fold that held it out).
    assert report["pooled"][0]["tau_heldout"] > 0


def test_validate_reports_no_transfer_for_a_lawless_holdout():
    rows = _lawful_rows(n_pids=10, samples_per_pid=8, lawless_layout="layD")
    report = validate(
        rows,
        seeds=[0],
        holdouts=["layD"],
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=40,
        min_cell_samples=4,
        n_boot=20,
        train_params=_fast_params(),
    )
    m = report["folds"]["layD"]["seeds"][0]
    # The held-out layout's times are random: no model can predict them.
    assert m["rho"] < 0.5
    # And the harness must say so via the ceiling too (unpredictable data, low ceiling).
    assert report["ceilings"]["layD"] < 0.6


def test_validate_defaults_to_bigram_and_rejects_trigram_rows_without_flag():
    rows = [
        StrokeRow(
            layout="layA",
            positions=((-1, 2), (1, 2), (2, 2)),
            ngram="abc",
            frequency=5,
            samples=[(70, 150, 1, 50)] * 6,
        )
    ]
    with pytest.raises(ValueError, match="length"):
        validate(rows, seeds=[0], train_params=_fast_params())


# --- eval hardening: calibration slope, worst cell, bootstrap CI (backlog E4/E2/E1) ----


def test_calibration_slope_detects_compression():
    from keybo.training.validate import calibration_slope

    rng = np.random.default_rng(0)
    obs = rng.uniform(100, 300, 200)
    # Perfect calibration -> slope ~1; compressed predictions (half the range) -> ~2.
    assert calibration_slope(obs + rng.normal(0, 2, 200), obs) == pytest.approx(1.0, abs=0.05)
    compressed = obs.mean() + (obs - obs.mean()) * 0.5
    assert calibration_slope(compressed, obs) == pytest.approx(2.0, abs=0.1)


def test_validate_reports_slope_worst_cell_and_ci():
    rows = _lawful_rows(n_pids=10, samples_per_pid=8)
    report = validate(
        rows,
        seeds=[0],
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=40,
        min_cell_samples=4,
        n_boot=10,
        train_params=_fast_params(),
    )
    for fold in report["folds"].values():
        m = fold["seeds"][0]
        # slope near 1 in the lawful world (geometry fully explains times)
        assert 0.5 < m["calibration_slope"] < 2.0
        # worst {wpm-bucket} cell rho reported alongside the mean
        assert "worst_bucket" in m and "worst_bucket_rho" in m
        assert m["worst_bucket_rho"] <= m["rho"] + 1e-9
        # participant-bootstrap CI brackets the point estimate
        lo, hi = m["rho_ci95"]
        assert lo <= m["rho"] <= hi


# --- trigram harness support (Phase B keystone enabler) --------------------------------


def _lawful_trigram_rows(seed=0, n_pids=8, samples_per_pid=6):
    """Trigram world: duration = 100 + 20*(d(a,b)+d(b,c)) + noise; same 4-layout shift
    construction as the bigram world so the true ranking is layA < layB < layC < layD."""
    rng = np.random.default_rng(seed)
    rows = []
    tris = ["abc", "def", "ghi", "jkl", "mno", "pqr"]
    for base, layout in [(2, "layA"), (3, "layB"), (4, "layC"), (5, "layD")]:
        for i, tg in enumerate(tris):
            p1 = _D[base + i]
            positions = (p1[0], p1[1], _D[min(base + i + 1, 10)][0])
            dsum = _distance(positions[:2]) + _distance(positions[1:])
            samples = []
            for pid in range(1, n_pids + 1):
                for _ in range(samples_per_pid):
                    wpm = int(rng.integers(65, 95))
                    dur = int(100 + 20 * dsum + rng.normal(0, 5))
                    samples.append((wpm, dur, pid, 50))
            rows.append(
                StrokeRow(
                    layout=layout, positions=positions, ngram=tg, frequency=50, samples=samples
                )
            )
    return rows


def test_validate_supports_trigram_rows():
    rows = _lawful_trigram_rows(n_pids=10, samples_per_pid=8)
    report = validate(
        rows,
        seeds=[0],
        ngram="trigram",
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=40,
        min_cell_samples=4,
        n_boot=10,
        train_params=_fast_params(),
    )
    assert set(report["folds"]) == {"layA", "layB", "layC", "layD"}
    for fold in report["folds"].values():
        assert fold["seeds"][0]["rho"] > 0.5  # lawful world must transfer
    assert report["pooled"][0]["tau_heldout"] > 0


def test_validate_rejects_mismatched_ngram_length():
    rows = _lawful_trigram_rows()
    with pytest.raises(ValueError, match="length"):
        validate(rows, seeds=[0], ngram="bigram", train_params=_fast_params())


# --- C1: tune retargeted at the harness --------------------------------------------------


def test_tune_lolo_prefers_transfer_over_memorization():
    """The LOLO tuner must rank a shallow (transfer-friendly) candidate above a deep
    (memorization-prone) one in the lawful world — the exact preference the CV-MAE tuner
    gets wrong. Small n keeps it fast; candidates passed explicitly for determinism."""
    from keybo.training.tune import tune_lolo

    rows = _lawful_rows(n_pids=8, samples_per_pid=6)
    candidates = [
        {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.3},
        {"n_estimators": 30, "max_depth": 8, "learning_rate": 0.3},
    ]
    best, leaderboard = tune_lolo(
        rows,
        candidates=candidates,
        seeds=[0],
        ngram="bigram",
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=40,
        min_cell_samples=4,
    )
    assert best in candidates
    # Leaderboard is (params, score) sorted best-first, scores finite.
    assert len(leaderboard) == 2
    assert leaderboard[0][1] >= leaderboard[1][1]
    assert all(np.isfinite(s) for _, s in leaderboard)


# --- magnitude metrics (user directive: ordering is not enough) --------------------------


def test_weighted_mae_and_mape_reported_per_fold_and_bucket():
    """Corpus-weighted MAE/MAPE per fold-seed AND per wpm bucket: the optimizer consumes
    magnitudes (fitness is a weighted sum), and only affine miscalibration is harmless —
    rank metrics are blind to nonlinear compression that moves the argmax."""
    rows = _lawful_rows(n_pids=10, samples_per_pid=8)
    report = validate(
        rows,
        seeds=[0],
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=20,
        min_cell_samples=4,
        n_boot=10,
        train_params=_fast_params(),
    )
    for fold in report["folds"].values():
        m = fold["seeds"][0]
        assert m["wmae"] > 0 and np.isfinite(m["wmae"])
        assert 0 < m["wmape"] < 1  # lawful world: errors well under 100%
        # per-bucket magnitude matrix rows: {bucket: {rho, wmae, slope, n}}
        assert m["bucket_matrix"]
        for stats in m["bucket_matrix"].values():
            assert set(stats) >= {"rho", "wmae", "slope", "n"}
            assert stats["n"] >= 5


def test_weighted_mae_weights_by_cell_frequency():
    """A high-frequency cell's error must dominate wmae (weights proxy objective weights)."""
    from keybo.training.validate import weighted_mae

    class C:  # minimal cell stub
        def __init__(self, freq):
            self.frequency = freq

    cells = [C(1000), C(1)]
    pred = np.array([110.0, 200.0])
    obs = np.array([100.0, 100.0])
    # errors: 10 (weight 1000) and 100 (weight 1) -> wmae ~ 10, not ~55
    assert weighted_mae(cells, pred, obs) == pytest.approx((10 * 1000 + 100 * 1) / 1001)
