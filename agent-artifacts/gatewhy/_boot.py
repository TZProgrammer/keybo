"""Provenance guard for every GATEWHY-1 driver.

The venv resolves `keybo` to the SHARED checkout silently, so every driver asserts which tree it
imported and which branch that tree is on BEFORE producing a number. A measurement from the wrong
tree is not a measurement.
"""

from __future__ import annotations

import os
import subprocess
import sys

TREE = "/local/home/zegertho/repos/keybo-wt-gatewhy"
ARTIFACTS = f"{TREE}/agent-artifacts/gatewhy"
BRANCH = "gatewhy"

# Pin the thread vars BEFORE anything imports xgboost/numpy BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

if f"{TREE}/src" not in sys.path:
    sys.path.insert(0, f"{TREE}/src")


def assert_tree() -> None:
    """Refuse to run unless `keybo` came from MY worktree on MY branch."""
    import keybo

    got = os.path.realpath(keybo.__file__)
    want = os.path.realpath(f"{TREE}/src/keybo/__init__.py")
    print(f"[boot] keybo.__file__ = {got}")
    if got != want:
        raise SystemExit(f"ABORT: keybo resolved to {got}, not {want}")
    branch = subprocess.run(
        ["git", "-C", TREE, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", TREE, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"[boot] branch = {branch} @ {head}")
    if branch != BRANCH:
        raise SystemExit(f"ABORT: branch is {branch}, not {BRANCH}")


def require(module, *names: str) -> None:
    """Assert every symbol exists BEFORE a long run leans on it (brief-decay defence)."""
    for name in names:
        if not hasattr(module, name):
            raise SystemExit(f"ABORT: {module.__name__} has no {name!r}")
        print(f"[boot] {module.__name__}.{name} present")


def require_key(mapping, *keys: str, where: str = "") -> None:
    """Assert every key exists in a dict read from an artifact (rc=0 + all-None is a bug)."""
    for key in keys:
        if key not in mapping:
            raise SystemExit(f"ABORT: {where}: key {key!r} absent (have: {sorted(mapping)[:12]})")
        print(f"[boot] {where}: key {key!r} present")
