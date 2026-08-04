"""Shared bootstrap for every INTERPFRAME driver.

Imported FIRST by every driver, before xgboost/numpy touch a thread pool. Two jobs:

1. Pin the four thread-count env vars. The fleet's standing hazard: xgboost + OpenBLAS
   each spawn nproc(=192) threads and the box thrashes.
2. Prove which `keybo` is on the path. The venv resolves `keybo` to the SHARED checkout
   (/local/home/zegertho/repos/keybo/src) silently, so a driver run from this worktree can
   import the parent's code and measure the wrong tree. `assert_tree()` prints keybo.__file__
   plus the git branch and HARD-FAILS if the resolved package is not this worktree.
"""

from __future__ import annotations

import os

# MUST precede `import xgboost` / `import numpy` anywhere in the process.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

WORKTREE = "/local/home/zegertho/repos/keybo-wt-interpframe"
ARTIFACTS = f"{WORKTREE}/agent-artifacts/interpframe"


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
        raise SystemExit(
            f"WRONG TREE: keybo resolved to {path}, expected {expect}. "
            f"Run with PYTHONPATH={WORKTREE}/src"
        )


def require(symbol_owner, name: str):
    """Assert a symbol exists on THIS tree before leaning on it (brief-decay defence)."""
    if not hasattr(symbol_owner, name):
        raise SystemExit(f"MISSING SYMBOL: {symbol_owner!r} has no {name!r} on this tree")
    return getattr(symbol_owner, name)
