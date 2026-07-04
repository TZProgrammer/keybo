"""Load the bistroke/tristroke TSV tables that the data pipeline produces and models train on.

Each line is::

    layout<TAB>positions<TAB>ngram<TAB>frequency<TAB>(wpm, duration, pid, hold)<TAB>...

where ``layout`` is the source keyboard layout the participant typed on, ``positions`` is a
Python-literal tuple of ``(x, y)`` positions, ``pid`` is the participant id, and ``hold`` is
the first key's press-to-release time in ms (-1 when the release timestamp was unusable).
Rows are filtered by a WPM threshold on individual samples and a minimum surviving-sample
count.

Pre-schema files (no layout column; 2-tuple samples) are detected by their first byte —
they start with ``(`` — and rejected with an error naming the fix, because silently
loading zero rows after an hours-long processing run is how work gets thrown away.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

import numpy as np

#: sample tuple layout, for readers of ``StrokeRow.samples``
SAMPLE_FIELDS = ("wpm", "duration", "pid", "hold")


@dataclass
class StrokeRow:
    layout: str
    positions: tuple[tuple[int, int], ...]
    ngram: str
    frequency: int
    samples: list[tuple[int, int, int, int]]  # (wpm, duration, pid, hold), wpm >= threshold


def iqr_average(values: list[float]) -> float:
    """Mean of ``values`` after discarding 1.5*IQR outliers (falls back to plain mean)."""
    if not values:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [v for v in values if lo <= v <= hi]
    return float(np.mean(kept)) if kept else float(np.mean(values))


def _parse_sample(token: str) -> tuple[int, int, int, int] | None:
    try:
        wpm, duration, pid, hold = ast.literal_eval(token)
        return int(wpm), int(duration), int(pid), int(hold)
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
            if line.startswith("("):
                raise ValueError(
                    f"{path} is in the pre-2026-07 stroke format (no layout column); "
                    "re-run `keybo process-data` to regenerate it"
                )
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            layout, pos_str, ngram, freq_str, *sample_tokens = parts
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
                StrokeRow(
                    layout=layout,
                    positions=positions,
                    ngram=ngram,
                    frequency=frequency,
                    samples=samples,
                )
            )
    return rows
