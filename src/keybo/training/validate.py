"""Leave-one-layout-out validation harness (OQ-5) — the cross-layout trust gate.

The optimizer's whole job is judging layouts no human has typed on, so the only evidence
that the model transfers is: hide one layout entirely, train on the rest, predict the
hidden one. This module implements that experiment with the decision rules pre-registered
in ``agent-artifacts/OQ5-generalization-validation.md`` (tightened per the 2026-07-04
fable audit):

- **Noise ceiling first** (:func:`split_half_ceiling`): split the held-out layout's
  *participants* in half and correlate the halves' per-cell mean times. No model can beat
  the data's own agreement with itself; every rho is reported alongside this ceiling.
- **Decisive metric = layout-level ranking** (Kendall's tau): an additive practice effect
  ("frequent bigrams are fast everywhere") inflates per-bigram correlations while being
  ranking-irrelevant, so per-bigram rho alone can reward fit the optimizer can't use.
- **Supplementary:** per-bigram Spearman rho computed on *bucket-centered* values (the
  wpm -> duration axis is an input to the model, so credit for it would be self-praise),
  plus MAE against a distance-only linear baseline (the floor a learned model must beat
  for the learning to have added anything transferable).
- **Seeds:** every conclusion should hold across >= 3 training seeds (single-seed leader
  boards were the original OQ-1 probe's failure mode).

Cells, not raw samples, are the unit of evaluation: a cell is (layout, ngram, wpm bucket)
with an IQR-mean observed duration — the same robust aggregation training targets use.
Cells below the sample floor are refused, not printed with a caveat.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau, spearmanr

from keybo.data.strokes import StrokeRow, iqr_average
from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.verdicts import bucket_regression_report


@dataclass
class Cell:
    """One (layout, ngram, wpm-bucket) evaluation unit."""

    layout: str
    ngram: str
    positions: tuple[tuple[int, int], ...]
    frequency: int  # the source row's occurrence count (bookkeeping; NOT a feature)
    bucket: int  # bucket start wpm
    wpm: float  # bucket midpoint — the wpm fed to the model
    obs: float  # IQR-mean of the bucket's observed durations
    n: int  # samples in the bucket
    samples: list[tuple[int, int, int, int]]  # (wpm, duration, pid, hold)


# --- splits -----------------------------------------------------------------------------


def leave_one_layout_out(
    rows: list[StrokeRow], holdout: str
) -> tuple[list[StrokeRow], list[StrokeRow]]:
    """Partition rows into (train = every other layout, test = the held-out layout)."""
    train = [r for r in rows if r.layout != holdout]
    test = [r for r in rows if r.layout == holdout]
    if not test:
        known = sorted({r.layout for r in rows})
        raise ValueError(f"no rows for holdout layout {holdout!r}; layouts present: {known}")
    return train, test


# --- cells ------------------------------------------------------------------------------


def build_cells(
    rows: list[StrokeRow],
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
) -> list[Cell]:
    """Bucket every row's samples by WPM and aggregate each bucket into a :class:`Cell`.

    Only samples with ``wpm_lo <= wpm < wpm_hi`` participate, and a cell must clear
    ``min_cell_samples`` or it is dropped entirely (a starved cell is noise, and printing
    it would launder that noise into the metrics).
    """
    cells: list[Cell] = []
    for row in rows:
        by_bucket: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for sample in row.samples:
            wpm = sample[0]
            if not wpm_lo <= wpm < wpm_hi:
                continue
            bucket = wpm_lo + ((wpm - wpm_lo) // bucket_width) * bucket_width
            by_bucket[bucket].append(sample)
        for bucket, samples in sorted(by_bucket.items()):
            if len(samples) < min_cell_samples:
                continue
            cells.append(
                Cell(
                    layout=row.layout,
                    ngram=row.ngram,
                    positions=row.positions,
                    frequency=row.frequency,
                    bucket=bucket,
                    wpm=bucket + bucket_width / 2,
                    obs=iqr_average([s[1] for s in samples]),
                    n=len(samples),
                    samples=samples,
                )
            )
    return cells


def _bucket_centered(cells: list[Cell], values: np.ndarray) -> np.ndarray:
    """Subtract each wpm bucket's mean: the wpm->duration axis is a model INPUT, so any
    correlation earned along it is credit for information the model was handed."""
    out = np.asarray(values, dtype=np.float64).copy()
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cells):
        by_bucket[c.bucket].append(i)
    for idx in by_bucket.values():
        out[idx] -= out[idx].mean()
    return out


def _centered_spearman(cells: list[Cell], pred: np.ndarray, obs: np.ndarray) -> float:
    if len(cells) < 3:
        return float("nan")
    rho = spearmanr(_bucket_centered(cells, pred), _bucket_centered(cells, obs)).statistic
    return float(rho)


def calibration_slope(pred: np.ndarray, obs: np.ndarray) -> float:
    """OLS slope of obs on pred. 1 = calibrated; >1 = predictions COMPRESS the true range
    (rank metrics are blind to this, but fitness is a weighted sum — gaps are load-bearing;
    backlog E4)."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    var = ((pred - pred.mean()) ** 2).sum()
    if var <= 0:
        return float("nan")
    return float(((pred - pred.mean()) * (obs - obs.mean())).sum() / var)


def weighted_mae(cells, pred: np.ndarray, obs: np.ndarray) -> float:
    """Corpus-frequency-weighted MAE: the magnitude error the OPTIMIZER actually feels.

    Fitness is sum(freq * t-hat), so a cell's prediction error matters in proportion to its
    corpus weight; rank metrics are invariant to all monotone transforms while the
    optimizer is only invariant to AFFINE ones — this metric covers the gap (user
    directive 2026-07-06: same ordering does not imply same optimal layout)."""
    w = np.array([c.frequency for c in cells], dtype=np.float64)
    if not w.sum():
        return float("nan")
    return float((w * np.abs(np.asarray(pred) - np.asarray(obs))).sum() / w.sum())


def uniform_mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """Unweighted per-cell MAE — the rare-ngram guard (user directive 2026-07-06).

    wmae is objective-aligned but the OPTIMIZER queries position pairs off the frequency
    distribution: rare cells are the only evidence for many pairs the search explores, so
    selection must not abandon them. Caveat: uniform weighting overweights noisy thin
    cells; use beside wmae + the decile profile, not instead of them."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(obs))))


def freq_decile_mae(cells, pred: np.ndarray, obs: np.ndarray) -> dict[int, float]:
    """MAE per cell-frequency decile (1 = rarest cells, 10 = most frequent)."""
    err = np.abs(np.asarray(pred) - np.asarray(obs))
    order = np.argsort([c.frequency for c in cells])
    out: dict[int, float] = {}
    splits = np.array_split(order, 10)
    for d, idx in enumerate(splits, start=1):
        if len(idx):
            out[d] = float(err[idx].mean())
    return out


def weighted_mape(cells, pred: np.ndarray, obs: np.ndarray) -> float:
    """Weighted mean absolute PERCENT error (scale-free companion to weighted_mae)."""
    obs = np.asarray(obs, dtype=np.float64)
    ok = obs > 0
    if not ok.any():
        return float("nan")
    w = np.array([c.frequency for c in cells], dtype=np.float64)[ok]
    if not w.sum():
        return float("nan")
    rel = np.abs(np.asarray(pred)[ok] - obs[ok]) / obs[ok]
    return float((w * rel).sum() / w.sum())


def _per_bucket_rho(
    cells: list[Cell], pred: np.ndarray, obs: np.ndarray, min_bucket_cells: int = 5
) -> dict[int, float]:
    """Plain Spearman within each wpm bucket (already single-bucket => no centering
    needed). Buckets with fewer than ``min_bucket_cells`` cells are skipped (a 3-cell rho is
    noise). The floor is a parameter so a caller that reports bucket rows at a *lower* floor
    cannot emit a row whose other metrics are finite while its rho was silently dropped."""
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cells):
        by_bucket[c.bucket].append(i)
    out: dict[int, float] = {}
    for bucket, idx in sorted(by_bucket.items()):
        if len(idx) < min_bucket_cells:
            continue
        rho = spearmanr(pred[idx], obs[idx]).statistic
        if np.isfinite(rho):
            out[bucket] = float(rho)
    return out


def _bucket_matrix(
    cells: list[Cell],
    pred: np.ndarray,
    obs: np.ndarray,
    min_bucket_cells: int = 5,
    bucket_rhos: dict[int, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-wpm-bucket slice: ``{bucket: {rho, wmae, umae, slope, n, n_raw, n_participants}}``.

    Both magnitudes are reported because they answer different questions inside a slice:
    ``wmae`` is corpus-weighted (what the optimizer feels), ``umae`` gives every cell equal
    say (the rare-ngram guard). Support travels with the numbers — a slice metric read
    without its sample/participant count cannot be checked against a support floor, and a
    thin high-speed bucket would then read as a real result.

    ``bucket_rhos`` may be passed in when the caller already computed it (the validate() path
    does) to avoid recomputing the same Spearman pass; it is computed at the SAME
    ``min_bucket_cells`` floor as the rows, so a row's rho is never silently dropped.
    """
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cells):
        by_bucket[c.bucket].append(i)
    if bucket_rhos is None:
        bucket_rhos = _per_bucket_rho(cells, pred, obs, min_bucket_cells)
    out: dict[str, dict[str, float]] = {}
    for bucket, idx in sorted(by_bucket.items()):
        if len(idx) < min_bucket_cells:
            continue
        sub = [cells[k] for k in idx]
        out[str(bucket)] = {
            "rho": bucket_rhos.get(bucket, float("nan")),
            "wmae": weighted_mae(sub, pred[idx], obs[idx]),
            "umae": uniform_mae(pred[idx], obs[idx]),
            "slope": calibration_slope(pred[idx], obs[idx]),
            "n": len(idx),
            "n_raw": int(sum(c.n for c in sub)),
            "n_participants": len({s[2] for c in sub for s in c.samples}),
        }
    return out


def _weighted_iqr_average(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    """``iqr_average(np.repeat(sorted_values, weights))`` without materializing the repeat.

    ``sorted_values`` must be ascending and ``weights`` the matching integer multiplicities.
    A replicate of the qwerty fold is ~27M samples, so the observation rebuild has to work
    on (value, count) bins rather than on expanded sample lists.
    """
    keep = weights > 0
    values, counts = sorted_values[keep], weights[keep]
    if not len(values):
        return 0.0
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])

    def percentile(q: float) -> float:
        # numpy's default 'linear' rule, evaluated against the implied expanded array.
        rank = (total - 1) * q
        lo_rank, hi_rank = int(np.floor(rank)), int(np.ceil(rank))
        lo = values[np.searchsorted(cumulative, lo_rank, side="right")]
        hi = values[np.searchsorted(cumulative, hi_rank, side="right")]
        return float(lo + (hi - lo) * (rank - lo_rank))

    q1, q3 = percentile(0.25), percentile(0.75)
    iqr = q3 - q1
    inlier = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
    if not inlier.any():  # mirrors iqr_average's fall back to the plain mean
        return float(np.average(values, weights=counts))
    return float(np.average(values[inlier], weights=counts[inlier]))


def _prepare_bootstrap(cells: list[Cell], pid_index: dict[int, int]) -> dict:
    """Pre-bin every cell's durations by (duration, participant) for fast rebuilds.

    Each cell becomes ascending unique durations plus a sparse ``bins x participants``
    count matrix, so one replicate is a single matrix-vector product against the draw
    counts instead of a re-scan of the raw samples.
    """
    values_by_cell: list[np.ndarray] = []
    offsets = [0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    n_pids = len(pid_index)
    for cell in cells:
        durations = np.array([s[1] for s in cell.samples], dtype=np.float64)
        pids = np.array([pid_index[s[2]] for s in cell.samples], dtype=np.int64)
        values, inverse = np.unique(durations, return_inverse=True)
        # One row per (duration, participant) pair actually present in this cell.
        codes, pair_counts = np.unique(inverse * n_pids + pids, return_counts=True)
        rows.append(offsets[-1] + codes // n_pids)
        cols.append(codes % n_pids)
        counts.append(pair_counts)
        values_by_cell.append(values)
        offsets.append(offsets[-1] + len(values))
    matrix = csr_matrix(
        (np.concatenate(counts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(offsets[-1], n_pids),
        dtype=np.int64,
    )
    return {"values": values_by_cell, "offsets": np.array(offsets, dtype=np.int64), "bins": matrix}


def _resample_cell_observations(
    prepared: dict, pid_counts: np.ndarray
) -> tuple[list[int], np.ndarray]:
    """Rebuild each cell's IQR-mean from a participant draw, dropping starved cells.

    ``pid_counts[j]`` is how many times participant ``j`` was drawn, so a participant drawn
    k times contributes k copies of every sample it gave. Cells left with NO samples are
    reported as dropped rather than aggregated: ``iqr_average([])`` is 0.0, and a spurious
    zero-duration cell would silently corrupt the replicate's rho.

    Emptiness is the ONLY drop rule here — ``build_cells``'s ``min_cell_samples`` floor is
    deliberately NOT re-applied per replicate. That floor decides which cells are admissible
    evidence, and re-imposing it on each draw would re-select the cell set from replicate to
    replicate, so the interval would mix "how uncertain is rho on these cells" with "which
    cells survived this draw". Cells are chosen once, on the full sample; the bootstrap then
    varies only the participants behind them.
    """
    weighted = np.asarray(prepared["bins"] @ np.asarray(pid_counts, dtype=np.int64)).ravel()
    offsets = prepared["offsets"]
    keep: list[int] = []
    rebuilt: list[float] = []
    for i, values in enumerate(prepared["values"]):
        weights = weighted[offsets[i] : offsets[i + 1]]
        if weights.any():
            keep.append(i)
            rebuilt.append(_weighted_iqr_average(values, weights))
    return keep, np.array(rebuilt, dtype=np.float64)


def _bootstrap_rho_ci(
    cells: list[Cell],
    pred: np.ndarray,
    obs: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI on the centered rho via PARTICIPANT-CLUSTER bootstrap (backlog E1).

    Participants are the resampling unit because samples within a participant are
    correlated: draw ``n_participants`` of them WITH REPLACEMENT, and a participant drawn
    k times contributes k copies of all its samples. Each cell's observation is then
    REBUILT from that resampled pool with the same ``iqr_average`` aggregation
    :func:`build_cells` used, and rho is recomputed on the rebuilt values.

    Rebuilding is the whole point: reusing the full-sample observations makes every
    replicate score the same number on the same data, which is what made this CI collapse
    to a zero-width point mass (a participant drawn 3x also has to count 3x — deduplicating
    the draw turns the bootstrap into a subsample without replacement).

    Cells whose contributors are all un-drawn are DROPPED from that replicate, and a
    replicate keeping < 3 cells is discarded (rho on two points is not informative).
    Returns ``(nan, nan)`` when there are < 2 participants or fewer than 20 replicates
    yield a finite rho — a refusal, not an interval invented from a handful of draws.

    Read it as an interval on the rho an INDEPENDENT (out-of-sample) prediction earns —
    which is what this harness always evaluates. It is a plain percentile interval, so if
    ``pred`` were instead derived from these same observations, the point rho would sit
    above the interval: resampling breaks the noise the two share, and no percentile
    bootstrap corrects that coupling. Not a defect, but do not read the interval as
    bracketing a rho computed against a predictor fit on the held-out cells themselves.
    """
    if not cells:
        return (float("nan"), float("nan"))
    all_pids = sorted({s[2] for c in cells for s in c.samples})
    if len(all_pids) < 2:
        return (float("nan"), float("nan"))
    prepared = _prepare_bootstrap(cells, {pid: i for i, pid in enumerate(all_pids)})
    rng = np.random.default_rng(seed)
    uniform = np.full(len(all_pids), 1.0 / len(all_pids))
    rhos: list[float] = []
    for _ in range(n_boot):
        # multinomial == drawing len(all_pids) participants with replacement, kept as
        # per-participant multiplicities.
        counts = rng.multinomial(len(all_pids), uniform)
        keep, boot_obs = _resample_cell_observations(prepared, counts)
        if len(keep) < 3:
            continue
        rho = _centered_spearman([cells[i] for i in keep], pred[keep], boot_obs)
        if np.isfinite(rho):
            rhos.append(rho)
    if len(rhos) < 20:
        return (float("nan"), float("nan"))
    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


# --- noise ceiling ----------------------------------------------------------------------


def spearman_brown(half_rho: float) -> float:
    """Lengthen a HALF-sample reliability to the FULL-sample one: ``2r / (1 + r)``.

    A split-half correlation measures how well one half predicts the other, so it is the
    reliability of a HALF-length instrument. The model's rho, by contrast, is scored on
    cells aggregated from ALL participants. Comparing the two directly understates the
    ceiling, and it does so UNEVENLY: ``2r/(1+r) / r`` is decreasing in ``r`` (1.443 at
    r=0.60, 1.008 at r=0.99), so a noisier arm's ratio is inflated more than a cleaner
    arm's and a per-arm ``rho/ceiling`` comparison can invert an ordering.

    Domain: ``r <= 0`` is returned unchanged — Spearman-Brown is derived for a reliability
    (a variance ratio, so non-negative), and lengthening a negative correlation is not
    meaningful. ``nan`` propagates.
    """
    if not np.isfinite(half_rho) or half_rho <= 0.0:
        return float(half_rho)
    return float(2.0 * half_rho / (1.0 + half_rho))


def split_half_ceiling(
    test_rows: list[StrokeRow],
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
    n_boot: int = 50,
    seed: int = 0,
    correct_length: bool = True,
) -> float:
    """Split-half reliability of the held-out layout's per-cell mean times.

    Participants (not samples) are bisected — samples within a participant are correlated,
    so a sample-level split would overstate the ceiling. Each half re-aggregates its own
    cells (floor = half the cell floor, min 2); cells present in both halves are correlated
    with the same bucket-centered Spearman the model metric uses.

    Each bisection's rho is then lengthened by :func:`spearman_brown` (``correct_length``,
    default on) BEFORE averaging, because the raw split-half value is a half-length
    reliability while the model's rho is scored on full-sample cells. Pass
    ``correct_length=False`` to reproduce a pre-2026-07-28 number; it is the wrong
    denominator for a ``rho/ceiling`` ratio and exists only for artifact reconciliation.

    The correction is applied PER BISECTION rather than to the mean: Spearman-Brown is
    non-linear, so ``mean(f(r)) != f(mean(r))``, and the per-bisection form is the one
    whose every term is a full-length reliability estimate.
    """
    per_key: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    pids: set[int] = set()
    for row in test_rows:
        for wpm, duration, pid, _hold in row.samples:
            if not wpm_lo <= wpm < wpm_hi:
                continue
            bucket = wpm_lo + ((wpm - wpm_lo) // bucket_width) * bucket_width
            per_key[(row.ngram, bucket)].append((pid, duration))
            pids.add(pid)

    all_pids = sorted(pids)
    if len(all_pids) < 2:
        return float("nan")
    half_floor = max(2, min_cell_samples // 2)
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(n_boot):
        perm = rng.permutation(all_pids)
        half_a = set(perm[: len(perm) // 2])
        pairs_a: list[float] = []
        pairs_b: list[float] = []
        buckets: list[int] = []
        for (_ngram, bucket), samples in per_key.items():
            a = [d for p, d in samples if p in half_a]
            b = [d for p, d in samples if p not in half_a]
            if len(a) < half_floor or len(b) < half_floor:
                continue
            pairs_a.append(iqr_average(a))
            pairs_b.append(iqr_average(b))
            buckets.append(bucket)
        if len(pairs_a) < 3:
            continue
        fake_cells = [
            Cell("", "", (), 0, bucket, 0.0, 0.0, 0, []) for bucket in buckets
        ]  # only .bucket is read by the centering
        rho = _centered_spearman(fake_cells, np.array(pairs_a), np.array(pairs_b))
        if np.isfinite(rho):
            rhos.append(spearman_brown(rho) if correct_length else rho)
    return float(np.mean(rhos)) if rhos else float("nan")


# --- layout-level ranking ---------------------------------------------------------------


def aggregate_layout_table(
    cells: list[Cell], values: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """layout -> ngram -> n-weighted mean value (observed times by default).

    Passing ``values`` (aligned with ``cells``) aggregates model predictions through the
    identical pipeline, so predicted and observed tables are directly comparable.
    """
    vals = np.array([c.obs for c in cells]) if values is None else np.asarray(values)
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for c, v in zip(cells, vals, strict=True):
        pair = acc[c.layout][c.ngram]
        pair[0] += v * c.n
        pair[1] += c.n
    return {
        layout: {ngram: s / n for ngram, (s, n) in ngrams.items()} for layout, ngrams in acc.items()
    }


def layout_ranking_tau(
    obs_table: dict[str, dict[str, float]], pred_table: dict[str, dict[str, float]]
) -> float:
    """Kendall's tau between observed and predicted layout ordering, on the common
    ngram set (fitness comparisons are only meaningful over material every layout can
    type — the same intersection rule the score CLI applies)."""
    layouts = sorted(set(obs_table) & set(pred_table))
    if len(layouts) < 2:
        return float("nan")
    common: set[str] | None = None
    for layout in layouts:
        ngrams = set(obs_table[layout]) & set(pred_table[layout])
        common = ngrams if common is None else common & ngrams
    if not common:
        return float("nan")
    obs_scores = [np.mean([obs_table[la][ng] for ng in sorted(common)]) for la in layouts]
    pred_scores = [np.mean([pred_table[la][ng] for ng in sorted(common)]) for la in layouts]
    return float(kendalltau(obs_scores, pred_scores).statistic)


# --- prediction + baseline --------------------------------------------------------------


def _predict_cells(model, cells: list[Cell], geometry: Geometry) -> np.ndarray:
    """g(geometry, wpm) + b(ngram) per cell, in MILLISECONDS — the model's full prediction.

    The practice term b (stored in the model's training metadata, when trained with it)
    is added by NGRAM IDENTITY: it is a legitimate, layout-independent part of predicted
    time. It cancels exactly in the layout-ranking tau (verified structurally); the
    per-cell rho does credit it, which is honest — see the OQ-1 artifact's decomposition.

    b lives in the model's target space (it was backfit there), so the order is fixed:
    add b to the raw prediction FIRST, then convert the sum to ms. For a LOGRAT model
    the reverse order would apply a log-space offset to a millisecond value.
    """
    featurize = (
        trigram_features_from_positions
        if len(cells[0].positions) == 3
        else bigram_features_from_positions
    )
    X = np.vstack([featurize(geometry, c.positions, wpm=c.wpm) for c in cells])
    pred = model.predict(X)
    practice = (model.metadata.extra.get("training") or {}).get("practice_term")
    if practice:
        values = practice.get("values", {})
        pred = pred + np.array([values.get(c.ngram, 0.0) for c in cells])
    calibration = (model.metadata.extra.get("training") or {}).get("calibration")
    if calibration and calibration.get("deltas_ms") and len(cells[0].positions) == 2:
        from keybo.training.calibration import delta_log, finger_class

        pred = pred + np.array(
            [
                delta_log(
                    finger_class(geometry, *c.positions),
                    c.wpm,
                    calibration["deltas_ms"],
                )
                for c in cells
            ]
        )
    return model.to_ms(pred, X)


def _distance(positions) -> float:
    """Sum of consecutive-pair euclidean distances (1 term for bigrams, 2 for trigrams)."""
    return float(
        sum(
            np.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(positions, positions[1:], strict=False)
        )
    )


def _baseline_fit(train_cells: list[Cell]) -> np.ndarray:
    """The dumb floor: duration ~ 1 + distance + wpm, least squares."""
    X = np.array([[1.0, _distance(c.positions), c.wpm] for c in train_cells])
    y = np.array([c.obs for c in train_cells])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _baseline_predict(coef: np.ndarray, cells: list[Cell]) -> np.ndarray:
    X = np.array([[1.0, _distance(c.positions), c.wpm] for c in cells])
    return X @ coef


# --- the harness ------------------------------------------------------------------------


def validate(
    rows: list[StrokeRow],
    seeds: list[int],
    ngram: str = "bigram",
    holdouts: list[str] | None = None,
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
    n_boot: int = 50,
    train_params: dict | None = None,
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    baseline_buckets: Mapping[int, float] | None = None,
) -> dict:
    """Run the full leave-one-layout-out experiment; returns the report dict.

    ``baseline_buckets`` (bucket start wpm -> rho, e.g. an incumbent's ``bucket_rhos``) turns on the
    high-wpm non-regression VERDICT: each fold/seed gains a ``high_wpm_gate`` block from
    :func:`keybo.verdicts.bucket_regression_report`. Omitting it leaves ``gated: False`` in the
    artifact rather than nothing at all, so an UNGATED result is never mistaken for a passing one
    (HIGHWPM-1: these per-bucket rhos were computed all along and nothing ever gated on them).

    Report shape::

        {
          "config": {...},
          "ceilings": {layout: split-half rho},
          "folds": {layout: {"n_cells": int, "seeds": [per-seed metrics...]}},
          "pooled": [per-seed {"seed", "tau_heldout"}],
        }

    Per-seed fold metrics: ``rho`` (bucket-centered Spearman on the held-out cells),
    ``rho_frac_ceiling``, ``tau_all4`` (this fold's model ranking every layout, held-out
    included), ``mae_model`` / ``mae_baseline`` / ``beats_baseline``. ``tau_heldout`` in
    ``pooled`` is the strictest number: every layout scored only by the fold that held it
    out, so the ranking is fully out-of-sample.
    """
    from keybo.training.train import train_bigram_model, train_trigram_model

    n_expected = {"bigram": 2, "trigram": 3}[ngram]
    if any(len(r.ngram) != n_expected for r in rows):
        raise ValueError(f"row ngram length does not match ngram={ngram!r} (expected {n_expected})")
    train_fn = train_bigram_model if ngram == "bigram" else train_trigram_model

    all_layouts = sorted({r.layout for r in rows})
    holdouts = list(holdouts) if holdouts is not None else all_layouts
    cell_kw = dict(
        wpm_lo=wpm_lo,
        wpm_hi=wpm_hi,
        bucket_width=bucket_width,
        min_cell_samples=min_cell_samples,
    )

    all_cells = build_cells(rows, **cell_kw)
    obs_table = aggregate_layout_table(all_cells)

    report: dict = {
        "config": {
            "seeds": list(seeds),
            "holdouts": holdouts,
            **cell_kw,
            "n_boot": n_boot,
            "train_params": dict(train_params or {}),
        },
        "ceilings": {},
        "folds": {},
        "pooled": [],
    }

    folds = [(h, s) for h in holdouts for s in seeds]
    iterator = folds
    if progress:
        from tqdm import tqdm

        iterator = tqdm(folds, desc="LOLO folds", unit="fold")

    # pred_heldout[seed][layout] -> that layout's predicted table row, out-of-sample.
    pred_heldout: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)

    for holdout, seed in iterator:
        train_rows, test_rows = leave_one_layout_out(rows, holdout)
        if holdout not in report["ceilings"]:
            report["ceilings"][holdout] = split_half_ceiling(
                test_rows, n_boot=n_boot, seed=0, **cell_kw
            )
        test_cells = build_cells(test_rows, **cell_kw)
        if not test_cells:
            raise ValueError(
                f"holdout {holdout!r} yields no cells at min_cell_samples="
                f"{min_cell_samples}; lower the floor or widen the wpm band"
            )
        fold = report["folds"].setdefault(holdout, {"n_cells": len(test_cells), "seeds": []})

        params = {**(train_params or {}), "random_state": seed, "n_jobs": 1}
        model = train_fn(train_rows, target_wpm=(wpm_lo + wpm_hi) / 2, **params)

        obs = np.array([c.obs for c in test_cells])
        pred = _predict_cells(model, test_cells, geometry)
        rho = _centered_spearman(test_cells, pred, obs)
        ceiling = report["ceilings"][holdout]
        train_cells = build_cells(train_rows, **cell_kw)
        coef = _baseline_fit(train_cells)
        base_pred = _baseline_predict(coef, test_cells)
        mae_model = float(np.mean(np.abs(pred - obs)))
        mae_baseline = float(np.mean(np.abs(base_pred - obs)))

        pred_all = _predict_cells(model, all_cells, geometry)
        tau_all4 = layout_ranking_tau(obs_table, aggregate_layout_table(all_cells, pred_all))

        bucket_rhos = _per_bucket_rho(test_cells, pred, obs)
        bucket_matrix = _bucket_matrix(test_cells, pred, obs, bucket_rhos=bucket_rhos)
        worst_bucket, worst_rho = (
            min(bucket_rhos.items(), key=lambda kv: kv[1]) if bucket_rhos else (None, float("nan"))
        )
        ci_lo, ci_hi = _bootstrap_rho_ci(test_cells, pred, obs, n_boot=max(100, n_boot), seed=seed)
        fold["seeds"].append(
            {
                "seed": seed,
                "rho": rho,
                "rho_ci95": [ci_lo, ci_hi],
                "ceiling": ceiling,
                "rho_frac_ceiling": (
                    rho / ceiling if np.isfinite(ceiling) and abs(ceiling) > 0.05 else None
                ),
                "tau_all4": tau_all4,
                "mae_model": mae_model,
                "mae_baseline": mae_baseline,
                "beats_baseline": mae_model < mae_baseline,
                "calibration_slope": calibration_slope(pred, obs),
                "wmae": weighted_mae(test_cells, pred, obs),
                "wmape": weighted_mape(test_cells, pred, obs),
                "umae": uniform_mae(pred, obs),
                "freq_decile_mae": freq_decile_mae(test_cells, pred, obs),
                "bucket_matrix": bucket_matrix,
                "worst_bucket": worst_bucket,
                "worst_bucket_rho": float(worst_rho),
                "bucket_rhos": {str(k): v for k, v in bucket_rhos.items()},
                # Always present, gated or not: an artifact that merely OMITS a verdict reads the
                # same whether the gate ran and passed or never ran at all (TAUGATE-1).
                "high_wpm_gate": bucket_regression_report(
                    bucket_rhos, baseline_buckets or {}, f"{holdout} seed={seed}"
                ),
            }
        )
        pred_heldout[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]

    for seed in seeds:
        tau = layout_ranking_tau(obs_table, pred_heldout[seed])
        report["pooled"].append({"seed": seed, "tau_heldout": tau})
    return report
