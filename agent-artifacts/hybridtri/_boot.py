"""Shared bootstrap for every HYBRIDTRI driver (HYBRIDB-1 / TRIAXIS-1).

Two jobs, both hard-failing rather than warning:

1. Pin the four thread-count env vars BEFORE xgboost/numpy touch a thread pool. The fleet's
   standing hazard: xgboost + OpenBLAS each spawn nproc(=192) threads and the box thrashes.
2. Prove which ``keybo`` is on the path. The venv resolves ``keybo`` to the SHARED checkout
   (/local/home/zegertho/repos/keybo/src) silently, so a driver run from this worktree can
   import the parent's code and measure the wrong tree.
"""

from __future__ import annotations

import os

# MUST precede `import xgboost` / `import numpy` anywhere in the process.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

WORKTREE = "/local/home/zegertho/repos/keybo-wt-hybridtri"
ARTIFACTS = f"{WORKTREE}/agent-artifacts/hybridtri"
# NOT bare /tmp: it is tmpfs, gets wiped, and a bare-/tmp script shadow-imports platform.py
# and has picked up another agent's stale file. Own subdir, own models.
SCRATCH = "/tmp/hybridtri_wk/models"


def assert_tree() -> None:
    """Print the resolved keybo + branch, and refuse to run against another checkout."""
    import subprocess

    import keybo

    path = os.path.realpath(os.path.dirname(keybo.__file__))
    expect = os.path.realpath(f"{WORKTREE}/src/keybo")
    branch = subprocess.run(
        ["git", "-C", WORKTREE, "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", WORKTREE, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"[tree] keybo.__file__ = {keybo.__file__}")
    print(f"[tree] branch = {branch}  HEAD = {head}")
    print(f"[tree] threads = OMP={os.environ.get('OMP_NUM_THREADS')}")
    if path != expect:
        raise SystemExit(f"WRONG TREE: keybo resolved to {path}, expected {expect}")


def require(owner, name: str):
    """Assert a symbol exists on THIS tree before leaning on it (brief-decay defence)."""
    if not hasattr(owner, name):
        raise SystemExit(f"MISSING SYMBOL: {owner!r} has no {name!r} on this tree")
    return getattr(owner, name)


def load_by_path(module_name: str, path: str):
    """Load a sibling's instrument BY PATH, never by adding its dir to sys.path.

    ``agent-artifacts/interpframe/_boot.py`` SHADOWS this ``_boot`` (same module name, WORKTREE
    pinned elsewhere), so a plain ``import metrics`` after a sys.path.insert makes assert_tree()
    demand the sibling's checkout. Loading by path with a UNIQUE module name avoids both.
    """
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {module_name!r} from {path}")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[instr] loaded {module_name} from {path}")
    return mod
