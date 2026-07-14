"""BAND-1 (rule c1fa917): WPM-banded specialist models vs the global surface.

User hypothesis: one model ingesting all WPM may lose to per-band specialists
(cf. quality injection losing to the direct quantile model). Banding SCHEME is
the experimental variable: hard 20/40-wide bands, equal-mass bands, overlapping
bands with blended predictions, plus a capacity control (global, 5x trees) and
a per-band-affine-recalibration diagnostic ("is it just scale?").

Training banding varies; the EVALUATION frame (20-wpm cells, min 10 samples,
40-140) never does. Participant-pure leave-one-layout-out folds. Bigram scout
at seed 0; decision rules live in the prereg block.

Run:  nice -n 10 /tmp/keybo_venv/bin/python agent-artifacts/experiments/wpm_banding.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from keybo.data.strokes import StrokeRow, load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import (  # noqa: E402
    _predict_cells,
    build_cells,
    calibration_slope,
    leave_one_layout_out,
    weighted_mae,
)

T0 = time.time()
SOURCE = "/local/home/zegertho/keybo-e2e/bistrokes_v5.tsv"
SOURCE_SHA = "d6cb4c81e8915f384787f872af58a6f2acffb7afa97f5d6c6b2a3804488bb1da"
OUT = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/band1_scout.json")
OBJECTIVE = Path(__file__).resolve().parents[2] / "data" / "corpus" / "bigrams.txt"
WPM_LO, WPM_HI, EVAL_WIDTH, MIN_CELL = 40, 140, 20, 10
RANKING_BUCKET = 80  # midpoint 90 = serve wpm
SEED = 0
# byte-identical to the baseline artifact's train_params
TRAIN_PARAMS = dict(
    colsample_bytree=0.7, gamma=0.957, learning_rate=0.05, max_depth=3,
    min_child_weight=4, n_estimators=300, reg_alpha=0.141, reg_lambda=0.011,
    subsample=0.7, verbosity=0, n_jobs=24, random_state=SEED,
)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def band_rows(rows: list[StrokeRow], lo: float, hi: float) -> list[StrokeRow]:
    """Rows with samples restricted to [lo, hi); rows left empty are dropped."""
    out = []
    for r in rows:
        kept = [s for s in r.samples if lo <= s[0] < hi]
        if kept:
            out.append(StrokeRow(r.layout, r.positions, r.ngram, r.frequency, kept))
    return out


def fit_model(rows: list[StrokeRow], **overrides):
    params = {**TRAIN_PARAMS, **overrides}
    return train_bigram_model(
        rows, target_wpm=90.0, geometry=ROW_STAGGERED_30, progress=False, **params
    )


def wlogmae(cells, pred, obs) -> float:
    w = np.array([c.frequency for c in cells], dtype=np.float64)
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 1e-9)
    obs = np.maximum(np.asarray(obs, dtype=np.float64), 1e-9)
    return float((w * np.abs(np.log(pred) - np.log(obs))).sum() / w.sum())


def metrics_block(cells, pred, obs) -> dict:
    by_band: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cells):
        by_band[c.bucket].append(i)
    out = {
        "pooled": {
            "weighted_log_mae": wlogmae(cells, pred, obs),
            "wmae": weighted_mae(cells, pred, obs),
            "calibration_slope": calibration_slope(pred, obs),
            "n_cells": len(cells),
        },
        "by_band": {},
    }
    for b, idx in sorted(by_band.items()):
        sub = [cells[i] for i in idx]
        out["by_band"][str(b)] = {
            "weighted_log_mae": wlogmae(sub, pred[idx], obs[idx]),
            "wmae": weighted_mae(sub, pred[idx], obs[idx]),
            "calibration_slope": calibration_slope(pred[idx], obs[idx]),
            "n_cells": len(idx),
        }
    return out


class HardBands:
    def __init__(self, rows, edges: list[float]):
        self.edges = edges
        self.models = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=False):
            sub = band_rows(rows, lo, hi)
            log(f"    band [{lo:.0f},{hi:.0f}): {len(sub)} rows")
            self.models.append(fit_model(sub))

    def predict(self, cells) -> np.ndarray:
        pred = np.empty(len(cells))
        by_model: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(cells):
            k = sum(1 for e in self.edges[1:-1] if c.wpm >= e)
            by_model[k].append(i)
        for k, idx in by_model.items():
            sub = [cells[i] for i in idx]
            pred[idx] = _predict_cells(self.models[k], sub, ROW_STAGGERED_30)
        return pred


class OverlapBands:
    """Width-40 stride-20 bands; triangular-weight blend of covering models (in ms)."""

    BANDS = [(40, 80), (60, 100), (80, 120), (100, 140)]

    def __init__(self, rows):
        self.models = []
        for lo, hi in self.BANDS:
            sub = band_rows(rows, lo, hi)
            log(f"    band [{lo},{hi}): {len(sub)} rows")
            self.models.append(fit_model(sub))
        self.centers = [(lo + hi) / 2 for lo, hi in self.BANDS]

    def predict(self, cells) -> np.ndarray:
        preds = np.column_stack(
            [_predict_cells(m, cells, ROW_STAGGERED_30) for m in self.models]
        )
        out = np.empty(len(cells))
        half = 20.0  # stride
        for i, c in enumerate(cells):
            ws = []
            for k, (lo, hi) in enumerate(self.BANDS):
                if lo <= c.wpm < hi:
                    ws.append((k, max(1e-9, 1.0 - abs(c.wpm - self.centers[k]) / (2 * half))))
            tot = sum(w for _, w in ws)
            out[i] = sum(preds[i, k] * w for k, w in ws) / tot
        return out


def eqmass_edges(rows) -> list[float]:
    wpms = np.concatenate(
        [[s[0] for s in r.samples if WPM_LO <= s[0] < WPM_HI] for r in rows]
    ).astype(float)
    qs = np.quantile(wpms, [0.2, 0.4, 0.6, 0.8])
    edges = [float(WPM_LO), *[float(q) for q in qs], float(WPM_HI)]
    # guard degenerate duplicates (heavy ties at integer wpm)
    for j in range(1, len(edges)):
        edges[j] = max(edges[j], edges[j - 1] + 1.0)
    return edges


def band_affine_diag(g_model, train_rows, cells, g_pred) -> np.ndarray:
    """Per-band WLS affine (fit on TRAIN-fold cells, ms space) applied to G's test preds."""
    train_cells = build_cells(train_rows, WPM_LO, WPM_HI, EVAL_WIDTH, MIN_CELL)
    tp = _predict_cells(g_model, train_cells, ROW_STAGGERED_30)
    to = np.array([c.obs for c in train_cells])
    tw = np.array([c.frequency for c in train_cells], dtype=np.float64)
    coef = {}
    for b in range(WPM_LO, WPM_HI, EVAL_WIDTH):
        idx = [i for i, c in enumerate(train_cells) if c.bucket == b]
        if len(idx) < 10:
            coef[b] = (0.0, 1.0)
            continue
        p, o, w = tp[idx], to[idx], tw[idx]
        pm = (w * p).sum() / w.sum()
        om = (w * o).sum() / w.sum()
        denom = (w * (p - pm) ** 2).sum()
        slope = float((w * (p - pm) * (o - om)).sum() / denom) if denom > 0 else 1.0
        slope = max(slope, 1e-3)
        coef[b] = (om - slope * pm, slope)
    out = g_pred.copy()
    for i, c in enumerate(cells):
        a, s = coef[c.bucket]
        out[i] = a + s * g_pred[i]
    return out


def main() -> None:
    sha = hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()
    if sha != SOURCE_SHA:
        raise SystemExit(f"source hash mismatch: {sha}")
    git = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    log(f"source verified; git {git[:9]}; loading rows")
    rows = load_strokes(SOURCE, ngram_len=2, wpm_threshold=0, min_samples=1)
    layouts = sorted({r.layout for r in rows})
    log(f"{len(rows)} rows, folds: {layouts}")
    obj_freq = {}
    for line in open(OBJECTIVE):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            obj_freq[p[0]] = int(p[1])

    result = {
        "experiment": "BAND-1 scout (rule c1fa917)",
        "git": git, "source_sha256": sha, "seed": SEED,
        "train_params": {k: v for k, v in TRAIN_PARAMS.items() if k not in ("n_jobs",)},
        "eval_frame": {"wpm_lo": WPM_LO, "wpm_hi": WPM_HI, "width": EVAL_WIDTH,
                       "min_cell_samples": MIN_CELL},
        "status": "running", "folds": {},
    }

    def checkpoint():
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1, default=float))
        os.replace(tmp, OUT)

    checkpoint()
    rank_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)  # arm -> layout -> {pred, obs}

    for holdout in layouts:
        log(f"=== fold {holdout} ===")
        train_rows, test_rows = leave_one_layout_out(rows, holdout)
        cells = build_cells(test_rows, WPM_LO, WPM_HI, EVAL_WIDTH, MIN_CELL)
        obs = np.array([c.obs for c in cells])
        log(f"  {len(cells)} eval cells")
        fold: dict = {"n_cells": len(cells)}

        preds: dict[str, np.ndarray] = {}
        log("  arm G")
        g_model = fit_model(train_rows)
        preds["G"] = _predict_cells(g_model, cells, ROW_STAGGERED_30)
        log("  arm CAP-G (n_estimators x5)")
        cap = fit_model(train_rows, n_estimators=1500)
        preds["CAP-G"] = _predict_cells(cap, cells, ROW_STAGGERED_30)
        del cap
        log("  arm HARD-20")
        preds["HARD-20"] = HardBands(train_rows, [40, 60, 80, 100, 120, 140]).predict(cells)
        log("  arm HARD-40")
        preds["HARD-40"] = HardBands(train_rows, [40, 80, 120, 140]).predict(cells)
        edges = eqmass_edges(train_rows)
        fold["eqmass_edges"] = edges
        log(f"  arm EQMASS-5 (edges {['%.0f' % e for e in edges]})")
        preds["EQMASS-5"] = HardBands(train_rows, edges).predict(cells)
        log("  arm OVL-40/20")
        preds["OVL-40/20"] = OverlapBands(train_rows).predict(cells)
        log("  diagnostic: per-band affine on G")
        preds["G+BANDAFFINE"] = band_affine_diag(g_model, train_rows, cells, preds["G"])
        del g_model

        fold["arms"] = {name: metrics_block(cells, p, obs) for name, p in preds.items()}
        # fixed-wpm-90 layout score inputs (bucket 80-100), objective-weighted
        idx90 = [i for i, c in enumerate(cells) if c.bucket == RANKING_BUCKET]
        for name, p in preds.items():
            num_p = num_o = den = 0.0
            for i in idx90:
                f = obj_freq.get(cells[i].ngram)
                if f:
                    num_p += f * p[i]
                    num_o += f * cells[i].obs
                    den += f
            if den:
                rank_scores[name][holdout] = {"pred": num_p / den, "obs": num_o / den}
        result["folds"][holdout] = fold
        checkpoint()
        for name in ("G", "CAP-G", "HARD-20", "HARD-40", "EQMASS-5", "OVL-40/20",
                     "G+BANDAFFINE"):
            m = fold["arms"][name]["pooled"]
            log(f"  {name:<13} wlogmae {m['weighted_log_mae']:.5f} wmae {m['wmae']:.3f} "
                f"slope {m['calibration_slope']:.3f}")

    # cross-fold: tau over held-out layouts at wpm 90, per arm
    result["ranking_wpm90"] = {}
    for name, per_layout in rank_scores.items():
        ls = sorted(per_layout)
        if len(ls) >= 3:
            tau = kendalltau([per_layout[la]["pred"] for la in ls],
                             [per_layout[la]["obs"] for la in ls]).statistic
            result["ranking_wpm90"][name] = {"tau_heldout": float(tau),
                                             "scores": {la: per_layout[la] for la in ls}}
    result["status"] = "complete"
    result["elapsed_seconds"] = time.time() - T0
    checkpoint()
    log("ALL-DONE")


if __name__ == "__main__":
    main()
