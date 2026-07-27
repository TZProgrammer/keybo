"""A parallel, on-disk cache for the 14-gauge vector of a layout.

``GaugeContext.vector`` is a PURE function of (layout, corpus) — the shipped ``gauge_matrix``
docstring notes that none of the underlying scorers memoize, and it measures 0.144 s/layout
here. Every arm of this sweep reuses layouts from a small number of banks, so computing the
union ONCE and indexing into it turns a ~40-minute serial cost into ~30 s on 192 cores.

Correctness discipline:

* the cache is keyed on (corpus identity sha256, layout string, LIVE_GAUGES tuple) — never on
  a truncation or any other lossy display form (trap 38);
* a POSITIVE CONTROL asserts the parallel/cached vector is bit-identical to the shipped
  serial ``E.gauge_matrix`` on a sample, because a hand-rolled fast path that loses the
  validated one's guarantees is exactly trap 28's habitat;
* workers rebuild the context per process (it is ``lru_cache``d per process and the tables are
  the slow part), and the corpus is passed by NAME so a worker cannot silently pick up a
  different one.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import keybo.analysis.evidence_scorer as E

_CTX_CORPUS: str | None = None


def _init(corpus: str | None) -> None:
    global _CTX_CORPUS
    _CTX_CORPUS = corpus
    E.gauge_context(corpus)  # warm the per-process lru_cache


def _work(layouts: list[str]) -> list[list[float]]:
    context = E.gauge_context(_CTX_CORPUS)
    return [[v[g] for g in E.LIVE_GAUGES] for v in (context.vector(lay) for lay in layouts)]


def _key(corpus_sha: str) -> str:
    payload = json.dumps(
        {"corpus": corpus_sha, "gauges": list(E.LIVE_GAUGES)}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class GaugeCache:
    """Layout -> 14-vector, computed in parallel and persisted as one ``.npz``."""

    def __init__(self, corpus: str | None, cache_dir: str, workers: int | None = None):
        self.corpus = corpus
        self.context = E.gauge_context(corpus)
        self.corpus_sha = json.dumps(
            dict(self.context.identity.get("sha256", {})), sort_keys=True
        )
        self.workers = workers or min(96, (os.cpu_count() or 8))
        self.path = Path(cache_dir) / f"gauges-{self.context.corpus_name}-{_key(self.corpus_sha)}.npz"
        self.table: dict[str, np.ndarray] = {}
        if self.path.is_file():
            blob = np.load(self.path, allow_pickle=False)
            keys, values = blob["layouts"], blob["gauges"]
            self.table = {str(k): values[i] for i, k in enumerate(keys)}
            print(f"gauge cache: loaded {len(self.table)} layouts from {self.path.name}", flush=True)

    def ensure(self, layouts) -> None:
        """Compute and cache every layout in ``layouts`` not already present."""
        missing = list(dict.fromkeys(lay for lay in layouts if lay not in self.table))
        if not missing:
            return
        chunk = max(8, len(missing) // (self.workers * 4) + 1)
        chunks = [missing[i : i + chunk] for i in range(0, len(missing), chunk)]
        print(
            f"gauge cache: computing {len(missing)} layouts on {self.workers} workers "
            f"({len(chunks)} chunks of ~{chunk})",
            flush=True,
        )
        done = 0
        with ProcessPoolExecutor(
            max_workers=self.workers, initializer=_init, initargs=(self.corpus,)
        ) as pool:
            for block, rows in zip(chunks, pool.map(_work, chunks), strict=True):
                for lay, row in zip(block, rows, strict=True):
                    self.table[lay] = np.asarray(row, dtype=float)
                done += len(block)
                if done % max(1, (len(missing) // 8)) < len(block):
                    print(f"  gauges: {done}/{len(missing)}", flush=True)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.table)
        np.savez_compressed(
            self.path,
            layouts=np.array(keys),
            gauges=np.array([self.table[k] for k in keys]),
        )

    def matrix(self, layouts: list[str]) -> np.ndarray:
        """``(len(layouts), 14)`` in :data:`E.LIVE_GAUGES` order, cache-only."""
        self.ensure(layouts)
        return np.array([self.table[lay] for lay in layouts])

    def positive_control(self, layouts: list[str], n: int = 12) -> dict:
        """Assert the cached matrix equals the SHIPPED serial ``gauge_matrix`` bit-for-bit.

        A partial positive control is not a positive control (trap 3), so this compares the
        full 14-vector for every sampled layout, not a summary of it, and returns the max abs
        difference so a caller can record the number rather than a claim.
        """
        sample = list(layouts)[:n]
        shipped = E.gauge_matrix(sample, self.context)
        mine = self.matrix(sample)
        diff = float(np.abs(shipped - mine).max())
        assert diff == 0.0, f"gauge cache disagrees with shipped gauge_matrix: max diff {diff}"
        return {"n_checked": len(sample), "max_abs_diff_vs_shipped_gauge_matrix": diff}
