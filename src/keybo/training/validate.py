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
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr

from keybo.data.strokes import StrokeRow, iqr_average
from keybo.features import bigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30, Geometry


@dataclass
class Cell:
    """One (layout, ngram, wpm-bucket) evaluation unit."""

    layout: str
    ngram: str
    positions: tuple[tuple[int, int], ...]
    frequency: int  # the source row's corpus frequency (feature input)
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


# --- noise ceiling ----------------------------------------------------------------------


def split_half_ceiling(
    test_rows: list[StrokeRow],
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
    n_boot: int = 50,
    seed: int = 0,
) -> float:
    """Split-half reliability of the held-out layout's per-cell mean times.

    Participants (not samples) are bisected — samples within a participant are correlated,
    so a sample-level split would overstate the ceiling. Each half re-aggregates its own
    cells (floor = half the cell floor, min 2); cells present in both halves are correlated
    with the same bucket-centered Spearman the model metric uses. The mean over ``n_boot``
    random bisections is the ceiling: the rho a perfect model would score on this data.
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
            rhos.append(rho)
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
    X = np.vstack(
        [
            bigram_features_from_positions(geometry, c.positions, freq=c.frequency, wpm=c.wpm)
            for c in cells
        ]
    )
    return model.predict(X)


def _distance(positions) -> float:
    (x1, y1), (x2, y2) = positions
    return float(np.hypot(x1 - x2, y1 - y2))


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
    holdouts: list[str] | None = None,
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
    n_boot: int = 50,
    train_params: dict | None = None,
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
) -> dict:
    """Run the full leave-one-layout-out experiment; returns the report dict.

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
    from keybo.training.train import train_bigram_model

    if any(len(r.ngram) != 2 for r in rows):
        raise NotImplementedError("the validation harness supports bigram stroke rows only")

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
        model = train_bigram_model(train_rows, target_wpm=(wpm_lo + wpm_hi) / 2, **params)

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

        fold["seeds"].append(
            {
                "seed": seed,
                "rho": rho,
                "ceiling": ceiling,
                "rho_frac_ceiling": (
                    rho / ceiling if np.isfinite(ceiling) and abs(ceiling) > 0.05 else None
                ),
                "tau_all4": tau_all4,
                "mae_model": mae_model,
                "mae_baseline": mae_baseline,
                "beats_baseline": mae_model < mae_baseline,
            }
        )
        pred_heldout[seed][holdout] = aggregate_layout_table(test_cells, pred)[holdout]

    for seed in seeds:
        tau = layout_ranking_tau(obs_table, pred_heldout[seed])
        report["pooled"].append({"seed": seed, "tau_heldout": tau})
    return report
