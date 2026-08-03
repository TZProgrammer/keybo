"""Shared plumbing for the LOSVAR-1 drivers. Import FIRST, before anything that reads threads.

Two measured traps this guards (both from the campaign's record, not rediscovered):
 * the shared venv's editable install resolves ``import keybo`` to /local/home/zegertho/repos/keybo,
   a LIVE sibling worktree whose branch moves under you — so a naive import measures another branch;
 * the four thread env vars are inert once numpy/xgboost has been imported, so they must be set first.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "48")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

WT = Path("/local/home/zegertho/repos/keybo-wt-losvar")
SHARED = Path("/local/home/zegertho/repos/keybo")
STATE = Path("/local/home/zegertho/agent/state/losvar")
ART = STATE / "artifacts"
CACHE = Path("/local/home/zegertho/agent/workspaces/losvar/cache")
SEED_TABLES = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/seed-tables")
TOURNAMENT_JSON = Path("/local/home/zegertho/agent/state/tournament/artifacts/tournament.json")
CALIB_K03 = Path("/local/home/zegertho/agent/state/calib/artifacts/k03_reconcile_fix.json")
BI31 = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
TRI31 = Path("/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv")

SHIPPED_SEEDS = (0, 1, 2)
CACHED_SEEDS = tuple(range(3, 25))
ALL_SEEDS = SHIPPED_SEEDS + CACHED_SEEDS

#: the LOLO frame, byte-identical to calib/k03 and to k31_train.py's gate (so NC2 is a real control)
HOLDOUTS = ("azerty", "dvorak", "qwerty", "qwertz")
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
SCORING_WPM = 90.0
#: the bucket whose midpoint IS the production scoring wpm (40 + 2*20 = 80, midpoint 90)
SCORING_BUCKET = 80
BOOT_SEED = 20260803          # registered in the LOSVAR-1 prereg

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


def _branch(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:                                          # noqa: BLE001
        return f"<unknown: {exc}>"


def assert_provenance() -> dict:
    """D5: prove which checkout ``keybo`` resolved to; refuse to measure another agent's branch."""
    import keybo
    resolved = Path(keybo.__file__).resolve()
    mine, shared = _branch(WT), _branch(SHARED)
    info = {"keybo__file__": str(resolved), "resolved_checkout": str(resolved.parents[2]),
            "my_worktree": str(WT), "my_branch": mine,
            "shared_checkout": str(SHARED), "shared_branch": shared,
            "python": sys.executable, "numpy": np.__version__,
            "threads": {v: os.environ.get(v) for v in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    print("[provenance] keybo.__file__ =", info["keybo__file__"])
    print(f"[provenance] MY worktree {WT} on branch {mine!r}")
    print(f"[provenance] SHARED checkout {SHARED} on branch {shared!r} (not mine; must not be measured)")
    if resolved.parents[2] != WT:
        raise SystemExit(f"D5 FAILED: keybo resolved to {resolved.parents[2]}, not {WT}")
    if mine != "losvar":
        raise SystemExit(f"D5 FAILED: my worktree is on branch {mine!r}, expected 'losvar'")
    return info


def require_finite(name: str, arr) -> np.ndarray:
    """Guard operands where numbers become claims (the empty-intersection nan cascade)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"{name}: EMPTY array — an empty intersection is the silent nan path")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: {int((~np.isfinite(a)).sum())} of {a.size} values not finite")
    return a


def load_boards() -> dict[str, str]:
    d = json.loads(TOURNAMENT_JSON.read_text())
    boards = dict(d["boards"])
    for name, s in boards.items():
        if len(s) != 30 or len(set(s)) != 30:
            raise ValueError(f"board {name!r} is not 30 distinct chars: {s!r}")
    return boards


def dump(name: str, obj: dict) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[artifact] {p} ({p.stat().st_size} bytes)")
    return p
