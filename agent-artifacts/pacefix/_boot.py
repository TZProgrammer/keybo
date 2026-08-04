"""Shared bootstrap for every PACEFIX driver (PACEFIX-1). Adapted from gatefolds/_boot.py.

Two jobs, both hard-failing rather than warning:

1. Pin the four thread-count env vars BEFORE xgboost/numpy touch a thread pool.
2. Prove which ``keybo`` is on the path -- the venv resolves ``keybo`` to the SHARED checkout
   silently, so a driver run from this worktree can import the parent's code.
"""

from __future__ import annotations

import os

# MUST precede `import xgboost` / `import numpy` anywhere in the process.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

WORKTREE = "/local/home/zegertho/repos/keybo-wt-pacefix"
ARTIFACTS = f"{WORKTREE}/agent-artifacts/pacefix"
# NOT bare /tmp: it is tmpfs, gets wiped, and a bare-/tmp script shadow-imports platform.py.
SCRATCH = "/tmp/pacefix_wk"

STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"


def assert_tree() -> None:
    """Print the resolved keybo + branch, and refuse to run against another checkout."""
    import subprocess

    import keybo

    path = os.path.realpath(os.path.dirname(keybo.__file__))
    expect = os.path.realpath(f"{WORKTREE}/src/keybo")
    branch = subprocess.run(
        ["git", "-C", WORKTREE, "branch", "--show-current"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", WORKTREE, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"[tree] keybo.__file__ = {keybo.__file__}")
    print(f"[tree] branch = {branch}  HEAD = {head}")
    print(f"[tree] threads = OMP={os.environ.get('OMP_NUM_THREADS')}")
    if path != expect:
        raise SystemExit(f"WRONG TREE: keybo resolved to {path}, expected {expect}")
    if branch != "pacefix":
        raise SystemExit(f"WRONG BRANCH: {branch!r}, expected 'pacefix'")


def require(owner, name: str):
    """Assert a symbol exists on THIS tree before leaning on it (brief-decay defence)."""
    if not hasattr(owner, name):
        raise SystemExit(f"MISSING SYMBOL: {owner!r} has no {name!r} on this tree")
    return getattr(owner, name)


def load_by_path(module_name: str, path: str):
    """Load a sibling's instrument BY PATH, never by adding its dir to sys.path.

    Every ``agent-artifacts/*/_boot.py`` shares the module name ``_boot`` with a DIFFERENT
    ``WORKTREE`` pinned inside it. Loading by path with a UNIQUE module name avoids that.
    PACEFIX-1 invariant 3: ``interpframe/metrics.py`` DISPATCHES ON A NAME SUBSTRING, so
    anything loaded here must have its dispatch VERIFIED against OUR frame.
    """
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {module_name!r} from {path}")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[instr] loaded {module_name} from {path}")
    return mod

def load_rows_cached():
    """Load the stroke rows, memoized to a pickle in SCRATCH.

    ``load_strokes`` takes ~210 s on this file (measured), and every driver needs the SAME rows.
    Caching keys on (path, mtime, size) so a changed input can never be served from a stale cache.
    """
    import hashlib
    import os
    import pickle

    st = os.stat(STROKES)
    key = hashlib.sha256(
        f"{STROKES}|{st.st_mtime_ns}|{st.st_size}|ngram_len=2|wpm_threshold=0|min_samples=1".encode()
    ).hexdigest()[:16]
    os.makedirs(SCRATCH, exist_ok=True)
    cache = f"{SCRATCH}/rows-{key}.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            rows = pickle.load(fh)
        print(f"[rows] CACHE HIT {cache}: {len(rows)} rows")
        return rows
    from keybo.data.strokes import load_strokes

    rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
    tmp = f"{cache}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
    print(f"[rows] loaded + cached {cache}: {len(rows)} rows")
    return rows
