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

import warnings

import numpy as np
import pytest
from scipy.stats import ConstantInputWarning

from keybo.data.strokes import StrokeRow, iqr_average
from keybo.training.validate import (
    Cell,
    _bootstrap_rho_ci,
    _centered_spearman,
    _predict_cells,
    aggregate_layout_table,
    build_cells,
    leave_one_layout_out,
    split_half_ceiling,
    validate,
    weighted_mape,
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


def test_build_cells_pins_half_open_band_edges_and_bucket_midpoints():
    row = StrokeRow(
        layout="layA",
        positions=((-1, 2), (1, 2)),
        ngram="ab",
        frequency=10,
        samples=[
            (39, 139, 1, 0),
            (40, 140, 2, 0),
            (59, 159, 3, 0),
            (60, 160, 4, 0),
            (139, 239, 5, 0),
            (140, 240, 6, 0),
        ],
    )
    cells = build_cells(
        [row],
        wpm_lo=40,
        wpm_hi=140,
        bucket_width=20,
        min_cell_samples=1,
    )
    assert [(c.bucket, c.wpm, c.n) for c in cells] == [
        (40, 50.0, 2),
        (60, 70.0, 1),
        (120, 130.0, 1),
    ]


def test_weighted_mape_pins_fraction_units_and_frequency_weights():
    cells = [
        Cell("", "", (), 3, 0, 0.0, 0.0, 0, []),
        Cell("", "", (), 1, 0, 0.0, 0.0, 0, []),
    ]
    assert weighted_mape(cells, np.array([110.0, 100.0]), np.array([100.0, 200.0])) == (
        pytest.approx(0.2)
    )


def test_predict_cells_applies_calibration_after_practice_in_target_space():
    from keybo.features.schema import BIGRAM_FEATURE_NAMES, FEATURE_VERSION
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.models.base import ModelMetadata, TypingModel

    class CalibratedLogratModel:
        metadata = ModelMetadata(
            feature_version=FEATURE_VERSION,
            feature_names=list(BIGRAM_FEATURE_NAMES),
            wpm_range=(40, 140),
            ngram="bigram",
            extra={
                "training": {
                    "target_space": "LOGRAT",
                    "calibration": {"deltas_ms": {"pinky_first": 62.0}},
                    "practice_term": {"values": {"qw": float(np.log(1.1))}},
                }
            },
        )
        target_space = TypingModel.target_space
        to_ms = TypingModel.to_ms

        @staticmethod
        def predict(X):
            wpm = X[:, BIGRAM_FEATURE_NAMES.index("wpm")]
            return np.log(138.0 * wpm / 12000.0)

    cells = [
        Cell(
            layout="qwerty",
            ngram="qw",
            positions=((-5, 3), (-4, 3)),
            frequency=3,
            bucket=90,
            wpm=100.0,
            obs=0.0,
            n=1,
            samples=[],
        )
    ]

    # Base 138 ms * practice 1.1 * calibration (138 + 62) / 138 = 220 ms.
    assert _predict_cells(CalibratedLogratModel(), cells, ROW_STAGGERED_30) == pytest.approx(
        [220.0]
    )


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
    # And the harness must say so via the CONCLUSION every consumer reads — the fraction of
    # ceiling attained, not the ceiling itself.
    #
    # This used to assert ``ceilings["layD"] < 0.6`` and broke when split_half_ceiling
    # gained its Spearman-Brown length correction (0.4971 -> 0.6228 on this fixture). That
    # rise is the correction working, not a regression: the raw split-half value is a
    # half-length reliability, and lengthening it is what makes it comparable to the
    # full-sample rho in the numerator. Pinning the intermediate ceiling pinned an
    # artifact of the old scale; pinning the ratio pins the claim ("no transfer"), which
    # gets STRONGER under the fix (frac 0.5172 -> 0.4129) because the denominator grew
    # while the unpredictable rho did not.
    assert m["rho_frac_ceiling"] < 0.5
    # the ceiling stays a correlation, and well under a lawful layout's
    assert 0.0 < report["ceilings"]["layD"] < 0.8


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


# --- participant bootstrap: the CI must be a real interval, not a point mass ------------


def _disagreeing_cells(n_pids=10, samples_per_pid=6, seed=0):
    """Cells whose two participant halves rank the ngrams in OPPOSITE orders.

    Every participant contributes to every cell, so the drawn PARTICIPANTS decide the
    per-cell means: which half a replicate over-samples flips the sign of rho. A bootstrap
    that resamples participants must therefore report a WIDE interval; one that reuses the
    full-sample observations reports a point mass regardless of what it drew.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for i, ngram in enumerate(_NGRAMS):
        samples = []
        for pid in range(1, n_pids + 1):
            ascending = pid <= n_pids // 2
            base = 100 + 20 * i if ascending else 220 - 20 * i
            for _ in range(samples_per_pid):
                samples.append((70, int(base + rng.normal(0, 3)), pid, 50))
        cells.append(
            Cell(
                layout="held",
                ngram=ngram,
                positions=_D[2 + i],
                frequency=100,
                bucket=60,
                wpm=70.0,
                obs=iqr_average([s[1] for s in samples]),
                n=len(samples),
                samples=samples,
            )
        )
    return cells


@pytest.mark.parametrize("trial", range(40))
def test_weighted_iqr_average_matches_literal_replication(trial):
    """The rebuild's aggregation must BE ``iqr_average``, on (value, count) bins.

    A replicate of the real qwerty fold is ~27M samples, so the rebuild works on counts
    instead of an expanded array — which is only legitimate if it agrees exactly with the
    aggregation :func:`build_cells` used, outlier trimming included.
    """
    from keybo.training.validate import _weighted_iqr_average

    rng = np.random.default_rng(trial)
    n = int(rng.integers(1, 12))
    values = np.sort(rng.choice(np.arange(50, 400, dtype=np.float64), size=n, replace=False))
    weights = rng.integers(0, 5, size=n)
    if trial % 7 == 0:  # an extreme value the IQR rule must trim
        values[-1] = 100_000.0
    if trial % 11 == 0:  # all the mass on a single duration
        weights = np.zeros(n, dtype=np.int64)
        weights[int(rng.integers(0, n))] = int(rng.integers(1, 30))
    if not weights.sum():
        pytest.skip("empty draw is covered by the drop-cells test")
    assert _weighted_iqr_average(values, weights) == pytest.approx(
        iqr_average(np.repeat(values, weights).tolist())
    )


def test_bootstrap_ci_is_not_degenerate_under_participant_disagreement():
    """RED at a6da599: the shipped bootstrap returns a ZERO-WIDTH interval.

    It ``set()``-ed the replacement draw (killing multiplicity) and kept whole cells on
    mere pid-set intersection while reusing the full-sample obs, so every replicate scored
    the identical rho on the identical numbers: percentile 2.5 == percentile 97.5.
    """
    cells = _disagreeing_cells()
    obs = np.array([c.obs for c in cells])
    pred = np.arange(len(cells), dtype=float)

    lo, hi = _bootstrap_rho_ci(cells, pred, obs, n_boot=400, seed=7)

    assert np.isfinite(lo) and np.isfinite(hi)
    # The halves disagree completely, so participant-level uncertainty is near-total.
    assert hi - lo > 0.5, f"degenerate CI [{lo}, {hi}] — participant resampling is inert"


def test_bootstrap_resampling_preserves_draw_multiplicity():
    """A participant drawn k times must contribute k copies of its samples.

    Two participants, one slow and one fast, and a draw of (2, 0): the replicate is the
    SLOW participant twice, so the rebuilt cell mean must be the slow value — not the
    two-participant average that ``set()``-ing the draw would leave in place.
    """
    from keybo.training.validate import _prepare_bootstrap, _resample_cell_observations

    cells = [
        Cell(
            layout="held",
            ngram="ab",
            positions=_D[2],
            frequency=10,
            bucket=60,
            wpm=70.0,
            obs=200.0,
            n=2,
            samples=[(70, 300, 1, 50), (70, 100, 2, 50)],
        )
    ]
    keep, boot_obs = _resample_cell_observations(_prepare_bootstrap(cells, {1: 0, 2: 1}), [2, 0])
    assert keep == [0]
    assert boot_obs[0] == pytest.approx(300.0)  # slow pid twice, fast pid absent
    # The mirror draw yields the fast participant twice.
    _, boot_obs = _resample_cell_observations(_prepare_bootstrap(cells, {1: 0, 2: 1}), [0, 2])
    assert boot_obs[0] == pytest.approx(100.0)


def test_bootstrap_rebuild_at_unit_counts_reproduces_cell_obs():
    """Every participant drawn exactly once must reproduce ``Cell.obs`` exactly.

    This pins the rebuild against the SAME aggregation the cells were built with
    (``iqr_average``), so a replicate is a resample of the data rather than a different
    statistic computed on it.
    """
    from keybo.training.validate import _prepare_bootstrap, _resample_cell_observations

    cells = _disagreeing_cells()
    index = {pid: i for i, pid in enumerate(sorted({s[2] for c in cells for s in c.samples}))}
    keep, boot_obs = _resample_cell_observations(_prepare_bootstrap(cells, index), [1] * len(index))
    assert keep == list(range(len(cells)))
    for cell, rebuilt in zip(cells, boot_obs, strict=True):
        assert rebuilt == pytest.approx(cell.obs)


def test_bootstrap_drops_cells_with_no_resampled_samples():
    """A cell whose contributors are all un-drawn is DROPPED, never aggregated at 0.0.

    ``iqr_average([])`` returns 0.0, so a silent rebuild would inject a spurious
    zero-duration cell into the replicate and corrupt its rho.
    """
    from keybo.training.validate import _prepare_bootstrap, _resample_cell_observations

    cells = [
        Cell("held", "ab", _D[2], 10, 60, 70.0, 300.0, 1, [(70, 300, 1, 50)]),
        Cell("held", "cd", _D[3], 10, 60, 70.0, 100.0, 1, [(70, 100, 2, 50)]),
    ]
    keep, boot_obs = _resample_cell_observations(_prepare_bootstrap(cells, {1: 0, 2: 1}), [2, 0])
    assert keep == [0]  # pid 2's cell is gone, not present as obs 0.0
    assert boot_obs.tolist() == [pytest.approx(300.0)]


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    cells = _disagreeing_cells()
    obs = np.array([c.obs for c in cells])
    pred = np.arange(len(cells), dtype=float)
    point = _centered_spearman(cells, pred, obs)

    lo, hi = _bootstrap_rho_ci(cells, pred, obs, n_boot=400, seed=7)
    assert lo <= point <= hi
    # Same seed -> same interval; a different seed still yields a real interval.
    assert (lo, hi) == _bootstrap_rho_ci(cells, pred, obs, n_boot=400, seed=7)
    other_lo, other_hi = _bootstrap_rho_ci(cells, pred, obs, n_boot=400, seed=8)
    assert other_hi - other_lo > 0.5


def test_bootstrap_ci_brackets_rho_of_an_independent_predictor():
    """The documented reading: the interval covers an OUT-OF-SAMPLE prediction's rho.

    Pinned because the honest caveat is easy to lose: a percentile bootstrap does NOT
    bracket the rho of a predictor built from these same observations (resampling breaks
    the shared noise that inflated it). ``validate`` only ever scores models trained on the
    other layouts, which is the case asserted here.
    """
    rng = np.random.default_rng(5)
    n_pids = 24
    truth = {ng: 100.0 + 20 * i for i, ng in enumerate(_NGRAMS)}
    quirk = {(pid, ng): rng.normal(0, 30) for pid in range(1, n_pids + 1) for ng in _NGRAMS}
    cells = []
    for i, ngram in enumerate(_NGRAMS):
        samples = [
            (70, int(truth[ngram] + quirk[(pid, ngram)] + rng.normal(0, 4)), pid, 50)
            for pid in range(1, n_pids + 1)
            for _ in range(8)
        ]
        cells.append(
            Cell(
                "held",
                ngram,
                _D[2 + i],
                100,
                60,
                70.0,
                iqr_average([s[1] for s in samples]),
                len(samples),
                samples,
            )
        )
    obs = np.array([c.obs for c in cells])
    # Independent of the sampled observations: the world's true per-ngram durations.
    pred = np.array([truth[c.ngram] for c in cells])
    point = _centered_spearman(cells, pred, obs)

    lo, hi = _bootstrap_rho_ci(cells, pred, obs, n_boot=400, seed=11)
    assert hi - lo > 0, f"degenerate CI [{lo}, {hi}]"
    assert lo <= point <= hi, f"CI [{lo}, {hi}] excludes independent-predictor rho {point}"


def test_bootstrap_ci_narrow_when_participants_agree():
    """Sanity in the other direction: agreeing participants -> a TIGHT interval.

    Width > 0 must come from real between-participant variance, not from noise the
    estimator manufactures, so the same code must also report near-certainty.
    """
    rng = np.random.default_rng(1)
    cells = []
    for i, ngram in enumerate(_NGRAMS):
        samples = [
            (70, int(100 + 20 * i + rng.normal(0, 2)), pid, 50)
            for pid in range(1, 21)
            for _ in range(8)
        ]
        cells.append(
            Cell(
                "held",
                ngram,
                _D[2 + i],
                100,
                60,
                70.0,
                iqr_average([s[1] for s in samples]),
                len(samples),
                samples,
            )
        )
    obs = np.array([c.obs for c in cells])
    pred = np.arange(len(cells), dtype=float)
    lo, hi = _bootstrap_rho_ci(cells, pred, obs, n_boot=400, seed=3)
    assert lo == pytest.approx(1.0, abs=1e-9) and hi == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_ci_nan_for_single_participant():
    """One participant = no participant-level replication; refuse rather than invent."""
    cells = [
        Cell("held", ng, _D[2 + i], 10, 60, 70.0, 100.0 + i, 2, [(70, 100 + i, 1, 50)] * 2)
        for i, ng in enumerate(_NGRAMS)
    ]
    obs = np.array([c.obs for c in cells])
    lo, hi = _bootstrap_rho_ci(cells, np.arange(len(cells), dtype=float), obs, n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_handles_all_identical_observations():
    """All-constant durations: rho is undefined per replicate, so the CI must be NaN.

    The failure mode to avoid is a confident-looking interval computed from a handful of
    accidentally-finite replicates.
    """
    cells = [
        Cell(
            "held",
            ng,
            _D[2 + i],
            10,
            60,
            70.0,
            150.0,
            20,
            [(70, 150, pid, 50) for pid in range(1, 11) for _ in range(2)],
        )
        for i, ng in enumerate(_NGRAMS)
    ]
    obs = np.array([c.obs for c in cells])
    with warnings.catch_warnings():
        # scipy rightly warns that rho is undefined on constant input — that IS the case
        # under test, so assert the refusal instead of leaking the warning as suite noise.
        warnings.simplefilter("ignore", ConstantInputWarning)
        lo, hi = _bootstrap_rho_ci(
            cells, np.arange(len(cells), dtype=float), obs, n_boot=100, seed=0
        )
    assert np.isnan(lo) and np.isnan(hi)


def _rows_with_participant_variance(seed=0, n_pids=12, samples_per_pid=8):
    """The lawful world plus a per-participant-PER-NGRAM idiosyncratic offset.

    ``_lawful_rows`` draws every participant from an identical distribution, so a correct
    participant bootstrap reports (rightly) almost no uncertainty there. A *uniform*
    per-participant speed offset would not help either: it shifts every cell equally and
    Spearman is rank-invariant to a common shift, so rho would still be exactly 1.0 in
    every replicate. What makes the ranking genuinely uncertain — and what real typists
    have — is participants DISAGREEING about which ngram is faster, i.e. a
    participant x ngram interaction. That is the spread the CI must propagate.
    """
    rng = np.random.default_rng(seed)
    quirk = {(pid, ng): rng.normal(0, 35) for pid in range(1, n_pids + 1) for ng in _NGRAMS}
    rows = []
    for layout, ngrams in _POSITIONS.items():
        for ngram, positions in ngrams.items():
            samples = []
            for pid in range(1, n_pids + 1):
                for _ in range(samples_per_pid):
                    dur = 60 + 25 * _distance(positions) + quirk[(pid, ngram)] + rng.normal(0, 4)
                    samples.append((int(rng.integers(65, 95)), int(dur), pid, 50))
            rows.append(
                StrokeRow(
                    layout=layout, positions=positions, ngram=ngram, frequency=100, samples=samples
                )
            )
    return rows


def test_validate_reports_positive_width_ci_on_real_folds():
    """End-to-end: with real between-participant spread, ``rho_ci95`` is a true interval."""
    report = validate(
        _rows_with_participant_variance(),
        seeds=[0],
        wpm_lo=60,
        wpm_hi=100,
        bucket_width=40,
        min_cell_samples=4,
        n_boot=10,
        train_params=_fast_params(),
    )
    for layout, fold in report["folds"].items():
        m = fold["seeds"][0]
        lo, hi = m["rho_ci95"]
        assert np.isfinite(lo) and np.isfinite(hi), f"{layout}: non-finite CI"
        assert hi > lo, f"{layout}: degenerate CI [{lo}, {hi}]"
        assert lo <= m["rho"] <= hi, f"{layout}: CI {[lo, hi]} excludes rho {m['rho']}"


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


def test_tune_lolo_scores_both_depths_and_this_fixture_CANNOT_separate_them():
    """The LOLO tuner's mechanics, and an honest statement of what this fixture can show.

    RENAMED from ``test_tune_lolo_prefers_transfer_over_memorization`` (2026-07-28), because
    it never demonstrated that preference. Measured: this fixture is geometry-lawful with
    sigma=4 noise, so BOTH depth-2 and depth-8 reach rho == 1.0 against a ceiling of 1.0 on
    every fold — ``rho_frac_ceiling`` saturates at exactly 1.000000 for both, and the gap is
    0.000000. The old ``leaderboard[0][1] >= leaderboard[1][1]`` assertion is satisfied by a
    TIE, so "shallow ranks above deep" was a stable-sort artifact, not a preference — the same
    tie-credit defect found in ``readjudicate.py`` and ``board_iweb_vs_blend.py``.

    So this test now asserts only what it can establish (both candidates score, finitely, and
    the leaderboard is ordered), and asserts the tie EXPLICITLY so a future fixture change
    that creates real separation shows up here rather than passing silently. The
    minimum-margin gate is disabled for the same reason it would otherwise fire: there is no
    margin to resolve.

    A real transfer-over-memorization test needs a fixture where the deep model can actually
    overfit — unlawful per-layout idiosyncrasy the shallow model cannot absorb. That fixture
    does not exist yet; writing it is the open work this rename exposes.
    """
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
        min_margin=0.0,  # nothing to resolve: see the docstring
    )
    assert best in candidates
    # Leaderboard is (params, score) sorted best-first, scores finite.
    assert len(leaderboard) == 2
    assert leaderboard[0][1] >= leaderboard[1][1]
    assert all(np.isfinite(s) for _, s in leaderboard)
    # The tie is the measured fact, pinned so a change in either direction is visible.
    assert leaderboard[0][1] == pytest.approx(leaderboard[1][1], abs=1e-12)
    assert leaderboard[0][1] == pytest.approx(1.0, abs=1e-12), "saturated ceiling"


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


def test_bucket_matrix_reports_umae_and_support_per_bucket():
    """A bucket slice is only readable next to its own support and its rare-cell error.

    A bucket's wmae is corpus-weighted, so in this dataset it is dominated by whichever
    layout contributes the most occurrences; without per-bucket ``umae`` a slice can look
    good while the rare cells inside it are abandoned (the same gap ``uniform_mae`` closes
    globally). And a bucket metric with no sample/participant count cannot be checked
    against a support floor, so a thin high-speed bucket would read as a real result.
    """
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
        stats_by_bucket = fold["seeds"][0]["bucket_matrix"]
        assert stats_by_bucket
        for stats in stats_by_bucket.values():
            assert set(stats) >= {
                "rho",
                "wmae",
                "umae",
                "slope",
                "n",
                "n_raw",
                "n_participants",
            }
            assert stats["umae"] > 0 and np.isfinite(stats["umae"])
            # support counts the actual observations behind the slice, so raw samples
            # must exceed cells and participants must be real.
            assert stats["n_raw"] >= stats["n"]
            assert 1 <= stats["n_participants"] <= 10


def test_bucket_umae_is_unweighted_and_wmae_is_frequency_weighted():
    """The two bucket magnitudes must not be the same number computed twice: ``umae`` gives
    a rare cell equal say, ``wmae`` gives it its corpus share."""
    from keybo.training.validate import _bucket_matrix

    cells = [
        Cell("held", "ab", ((0, 0), (1, 0)), 1000, 60, 70.0, 100.0, 4, [(70, 100, 1, 0)] * 4),
        Cell("held", "cd", ((0, 0), (2, 0)), 1, 60, 70.0, 100.0, 4, [(70, 100, 2, 0)] * 4),
    ] * 3  # 6 cells so the bucket clears the 5-cell floor
    pred = np.array([110.0, 200.0] * 3)
    obs = np.array([100.0, 100.0] * 3)

    stats = _bucket_matrix(cells, pred, obs)["60"]

    assert stats["umae"] == pytest.approx((10 + 100) / 2)
    assert stats["wmae"] == pytest.approx((10 * 1000 + 100 * 1) / 1001)
    assert stats["n"] == 6
    assert stats["n_raw"] == 24
    assert stats["n_participants"] == 2
    # obs is constant here, so spearman is undefined and _per_bucket_rho drops the bucket:
    # pin the NaN fallback explicitly rather than covering it only incidentally.
    assert np.isnan(stats["rho"])


def test_bucket_matrix_rho_floor_matches_the_row_floor():
    """A row must never be emitted with finite magnitudes but a silently-dropped rho: the
    rho floor has to track ``min_bucket_cells``, not a separate hardcoded 5."""
    from keybo.training.validate import _bucket_matrix

    # 3 cells => below the default 5-cell floor, so no row at all by default.
    cells = [
        Cell("held", ng, ((0, 0), (1, 0)), 10, 60, 70.0, 100.0, 4, [(70, 100, 1, 0)] * 4)
        for ng in ("ab", "cd", "ef")
    ]
    pred = np.array([110.0, 130.0, 150.0])
    obs = np.array([100.0, 120.0, 160.0])

    assert _bucket_matrix(cells, pred, obs) == {}

    # Lowering the floor emits the row -- and its rho must be a real number, not NaN,
    # because the rho pass now uses the SAME floor.
    stats = _bucket_matrix(cells, pred, obs, min_bucket_cells=3)["60"]
    assert stats["n"] == 3
    assert np.isfinite(stats["rho"])


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


def test_uniform_mae_and_decile_profile_reported():
    """Rare-ngram guard: wmae alone lets selection abandon rare cells, which are the only
    evidence for position pairs the optimizer explores off the frequency distribution."""
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
        assert m["umae"] > 0 and np.isfinite(m["umae"])
        assert len(m["freq_decile_mae"]) >= 3  # small synthetic world -> few deciles ok
        assert all(v >= 0 for v in m["freq_decile_mae"].values())


def test_validate_evaluates_lograt_models_in_ms():
    """With the LOGRAT default the harness's per-cell predictions must be converted back
    to ms before any metric: rho survives a monotone transform but wmae/umae/slope do
    not (raw log predictions would produce wmae ~ the whole duration scale)."""
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
        # lawful world durations are 60-320ms; a log-space leak would give wmae > 100
        # (every |pred - obs| ~ obs) and a wildly non-unit calibration slope.
        assert m["wmae"] < 60
        assert m["rho"] > 0.6
        assert 0.3 < m["calibration_slope"] < 3.0
