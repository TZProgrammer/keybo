"""Shared plumbing for the LOS-1 drivers: env pinning, provenance assertion, per-seed surface.

Every driver imports this FIRST, before anything that reads thread env vars or resolves ``keybo``.
Two measured traps in this campaign motivate the two guards here:

* the shared venv's editable install resolves ``import keybo`` to ``/local/home/zegertho/repos/keybo``,
  which is a live sibling's working tree — a naive import silently measures ANOTHER agent's branch;
* the four thread env vars are inert once xgboost/numpy has been imported, so they must be set before.
"""

from __future__ import annotations

import os

# D6: pin threads BEFORE any import that reads them. Must precede numpy/xgboost.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "48")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

WT = Path("/local/home/zegertho/repos/keybo-wt-los")
SHARED = Path("/local/home/zegertho/repos/keybo")
STATE = Path("/local/home/zegertho/agent/state/los")
ART = STATE / "artifacts"
SEED_TABLES = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/seed-tables")
TOURNAMENT_JSON = Path("/local/home/zegertho/agent/state/tournament/artifacts/tournament.json")

# The n=25 common seed set: 0,1,2 are the SHIPPED models; 3..24 are SEEDTB-1's cached tables.
SHIPPED_SEEDS = (0, 1, 2)
CACHED_SEEDS = tuple(range(3, 25))
ALL_SEEDS = SHIPPED_SEEDS + CACHED_SEEDS


def _branch(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - provenance reporting only
        return f"<unknown: {exc}>"


def assert_provenance() -> dict:
    """D5: prove which checkout ``keybo`` resolved to, and print BOTH checkouts' branches.

    Refuses to run out of the shared checkout: this driver's numbers must come from MY worktree,
    and the shared tree is a sibling's working area whose branch can change under me mid-run.
    """
    import keybo

    resolved = Path(keybo.__file__).resolve()
    info = {
        "keybo__file__": str(resolved),
        "resolved_checkout": str(resolved.parents[2]),
        "my_worktree": str(WT),
        "my_branch": _branch(WT),
        "shared_checkout": str(SHARED),
        "shared_branch": _branch(SHARED),
        "python": sys.executable,
        "numpy": np.__version__,
        "threads": {v: os.environ.get(v) for v in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    print("[provenance] keybo.__file__ =", info["keybo__file__"])
    print("[provenance] resolved checkout =", info["resolved_checkout"])
    print(f"[provenance] MY worktree {WT} on branch {info['my_branch']!r}")
    print(f"[provenance] SHARED checkout {SHARED} on branch {info['shared_branch']!r}")
    if resolved.parents[2] != WT:
        raise SystemExit(
            f"D5 FAILED: keybo resolved to {resolved.parents[2]}, not my worktree {WT}. "
            "Run with PYTHONPATH=<my worktree>/src so the measurement is of MY tree."
        )
    return info


def require_finite(name: str, arr) -> np.ndarray:
    """Guard the operands at the point where numbers become a claim.

    A sibling intersected 27 boards' observed trigram sets, got the EMPTY set, and published
    "fastest = qwerty" for all 27 because every operand was nan. This is that guard.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"{name}: EMPTY array — an empty intersection is the silent nan path")
    if not np.all(np.isfinite(a)):
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"{name}: {bad} of {a.size} values are not finite")
    return a


def load_boards() -> dict[str, str]:
    """The 13-board adoption field, from TOURNAMENT-1's artifact + keybo-lsb+lm from the brief."""
    d = json.loads(TOURNAMENT_JSON.read_text())
    boards = dict(d["boards"])
    # The brief supplies a 14th board not in the tournament field.
    boards["keybo-lsb+lm"] = "pyuo,vgdnmhiea.cstrlkj-z'fwbxq"
    for name, s in boards.items():
        if len(s) != 30 or len(set(s)) != 30:
            raise ValueError(f"board {name!r} is not 30 distinct chars: {s!r}")
    return boards


class PerSeedSurface:
    """Per-seed (T2, Tc) millisecond tables + the corpus, giving per-seed ms/char for any board.

    ``ms/char`` for one seed is ``sum_k f_k * (T2[a,b] + Tc[a,b,c]) / covered_mass``, which is
    exactly what :meth:`keybo.analysis.timecard.TimeSurface.card` accumulates on the seed-MEAN
    tables and what :meth:`seed_totals` accumulates per seed. Vectorized here because the
    per-trigram python loop is ~50 ms per board per seed and this runs 25 seeds x 14 boards x
    3 pricings x thousands of resamples.
    """

    def __init__(self, target_wpm: float = 90.0, corpus: str | None = None) -> None:
        from keybo.data.corpus import load_frequencies, production_corpus_dir
        from keybo.geometry import ROW_STAGGERED_30

        self.geometry = ROW_STAGGERED_30
        self.n = len(self.geometry.slots) + 1  # 30 slots + space
        tri = load_frequencies(str(production_corpus_dir(corpus) / "trigrams.txt"))
        self.tri = {k: v for k, v in tri.items() if len(k) == 3}
        self.total_mass = sum(self.tri.values())
        self.target_wpm = target_wpm
        self._load_seed_tables()
        self._index_corpus()

    def _load_seed_tables(self) -> None:
        """Seeds 0-2 rebuilt from the SHIPPED models; 3-24 from SEEDTB-1's cached npz tables."""
        from keybo.analysis.timecard import TimeSurface

        # The shipped surface with keep_seed_tables=True gives the three shipped seeds' tables,
        # and is also the parity target for NC2.
        self._shipped = TimeSurface(self.tri, target_wpm=self.target_wpm,
                                    geometry=self.geometry, keep_seed_tables=True)
        T2s = list(self._shipped._T2s)
        Tcs = list(self._shipped._Tcs)
        for s in CACHED_SEEDS:
            d = np.load(SEED_TABLES / f"tables_seed{s}.npz")
            T2s.append(require_finite(f"T2 seed{s}", d["T2"]))
            Tcs.append(require_finite(f"Tc seed{s}", d["Tc"]))
        self.T2s = np.stack(T2s)                      # (25, 31, 31)
        self.Tcs = np.stack(Tcs)                      # (25, 31, 31, 31)
        if self.T2s.shape != (len(ALL_SEEDS), self.n, self.n):
            raise ValueError(f"T2s shape {self.T2s.shape} unexpected")
        if self.Tcs.shape != (len(ALL_SEEDS), self.n, self.n, self.n):
            raise ValueError(f"Tcs shape {self.Tcs.shape} unexpected")

    def _index_corpus(self) -> None:
        """Freeze the corpus into (char triple index, freq) arrays over the C30M charset."""
        self._ngrams = list(self.tri.keys())
        self._freqs = np.array([self.tri[g] for g in self._ngrams], dtype=np.float64)

    def _slot_of(self, board: str) -> dict[str, int]:
        if len(board) != 30 or len(set(board)) != 30:
            raise ValueError(f"board must be 30 distinct chars, got {board!r}")
        slot = {c: i for i, c in enumerate(board)}
        slot[" "] = self.n - 1
        return slot

    def rows(self, board: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(a, b, c, f) index/freq arrays for the corpus rows this board can type."""
        slot = self._slot_of(board)
        a, b, c, f = [], [], [], []
        for g, fr in zip(self._ngrams, self._freqs, strict=True):
            try:
                ia, ib, ic = slot[g[0]], slot[g[1]], slot[g[2]]
            except KeyError:
                continue
            a.append(ia); b.append(ib); c.append(ic); f.append(fr)
        return (np.array(a, dtype=np.intp), np.array(b, dtype=np.intp),
                np.array(c, dtype=np.intp), np.array(f, dtype=np.float64))

    def per_seed_ms_per_char(self, board: str, mask: np.ndarray | None = None) -> np.ndarray:
        """(25,) per-seed ms/char. ``mask`` selects a subset of this board's corpus rows.

        The denominator is the COVERED mass of the selected rows — the same denominator
        ``TimeCard.ms_per_char`` uses. Dividing by the corpus total instead scales everything by
        ~1.13 while preserving ranking, so no comparison test would catch it.
        """
        a, b, c, f = self.rows(board)
        if mask is not None:
            a, b, c, f = a[mask], b[mask], c[mask], f[mask]
        require_finite(f"freqs {board}", f)
        covered = float(f.sum())
        if covered <= 0:
            raise ValueError(f"board {board!r}: covered mass is 0 — every row was filtered out")
        t2 = self.T2s[:, a, b]                       # (25, nrows)
        tc = self.Tcs[:, a, b, c]                    # (25, nrows)
        totals = (f[None, :] * (t2 + tc)).sum(axis=1)
        return require_finite(f"ms_per_char {board}", totals / covered)

    def covered_mass(self, board: str, mask: np.ndarray | None = None) -> float:
        _, _, _, f = self.rows(board)
        if mask is not None:
            f = f[mask]
        return float(f.sum())


def dump(name: str, obj: dict) -> Path:
    """Write an artifact JSON (machine-generated; CLOSING-1 rule 2: emit, never transcribe)."""
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=False, default=float))
    print(f"[artifact] {p} ({p.stat().st_size} bytes)")
    return p
