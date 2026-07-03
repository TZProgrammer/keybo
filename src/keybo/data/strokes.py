"""Load the bistroke/tristroke TSV tables that the data pipeline produces and models train on.

Each line is::

    positions<TAB>ngram<TAB>frequency<TAB>(wpm, duration)<TAB>(wpm, duration)...

where ``positions`` is a Python-literal tuple of ``(x, y)`` positions. Rows are filtered by
a WPM threshold on individual samples and a minimum surviving-sample count.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class StrokeRow:
    positions: tuple[tuple[int, int], ...]
    ngram: str
    frequency: int
    samples: list[tuple[int, int]]  # (wpm, duration) pairs clearing the wpm threshold


def iqr_average(values: list[float]) -> float:
    """Mean of ``values`` after discarding 1.5*IQR outliers (falls back to plain mean)."""
    if not values:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [v for v in values if lo <= v <= hi]
    return float(np.mean(kept)) if kept else float(np.mean(values))


def _parse_sample(token: str) -> tuple[int, int] | None:
    try:
        wpm, duration = ast.literal_eval(token)
        return int(wpm), int(duration)
    except (SyntaxError, ValueError, TypeError):
        return None


def load_strokes(
    path: str, ngram_len: int, wpm_threshold: int, min_samples: int
) -> list[StrokeRow]:
    """Load stroke rows of a given n-gram length, keeping only samples at/above the WPM
    threshold and rows with at least ``min_samples`` such samples."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"stroke file not found: {path}")

    rows: list[StrokeRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            pos_str, ngram, freq_str, *sample_tokens = parts
            if len(ngram) != ngram_len:
                continue
            try:
                positions = ast.literal_eval(pos_str)
            except (SyntaxError, ValueError):
                continue

            samples = [
                s for tok in sample_tokens if (s := _parse_sample(tok)) and s[0] >= wpm_threshold
            ]
            if len(samples) < min_samples:
                continue

            try:
                frequency = int(freq_str)
            except ValueError:
                frequency = len(samples)
            rows.append(
                StrokeRow(positions=positions, ngram=ngram, frequency=frequency, samples=samples)
            )
    return rows
