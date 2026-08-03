"""Shared preamble: pin threads BEFORE xgboost, force MY worktree, assert D5.

Import this FIRST in every driver, before any keybo/xgboost import. The thread pins are inert
if applied after xgboost loads, and the sys.path insert is load-bearing: the venv's editable
install resolves `import keybo` to the SHARED checkout /local/home/zegertho/repos/keybo, which
other agents move between branches.
"""
import os

# G-THREADS: all four, before xgboost. Applied at import time, so the ordering is structural.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "48")

import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

MY_WT = "/local/home/zegertho/repos/keybo-wt-sfbprice"
SHARED = "/local/home/zegertho/repos/keybo"
E2E = "/local/home/zegertho/keybo-e2e"
BI = f"{E2E}/bistrokes31_v1.tsv"
TRI = f"{E2E}/tristrokes31_cond_v1.tsv"
SHIPPED = f"{SHARED}/data/models/k31"          # READ-ONLY. Never written.
SEEDTABLES = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/seed-tables"
ART = "/local/home/zegertho/agent/state/sfbprice/artifacts"
OUT = "/local/home/zegertho/agent/workspaces/sfbprice/out"

sys.path.insert(0, MY_WT + "/src")
os.makedirs(ART, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def _branch(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:                                    # noqa: BLE001
        return f"<unknown: {e}>"


def assert_d5():
    """G-D5: refuse to measure another agent's branch. Prints BOTH checkouts' branches."""
    import keybo
    resolved = keybo.__file__
    mine, shared = _branch(MY_WT), _branch(SHARED)
    print(f"  keybo.__file__ = {resolved}")
    print(f"  my worktree {MY_WT} @ branch {mine!r}")
    print(f"  SHARED checkout {SHARED} @ branch {shared!r}  (not mine; must not be measured)")
    assert resolved.startswith(MY_WT + "/"), (
        f"D5 FAIL: keybo resolved to {resolved}, NOT my worktree {MY_WT}. The shared checkout is "
        f"live for other agents (currently on {shared!r}); refusing to measure another branch."
    )
    assert mine == "sfbprice", f"D5 FAIL: my worktree is on branch {mine!r}, expected 'sfbprice'"
    return resolved


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------------------------
# THE FIELD. Every string verified against a repo artifact or the pinned registry, NOT
# transcribed from the parent's brief. The 7 tuned + flagship strings are byte-copied from
# `_guard.py` on branch `tournament` (which verified them against repo artifacts); the 5
# community strings come from src/keybo/layouts.py (import-time validated); the 2 extras from
# src/keybo/cli/analyze.py::_EXTRA_NAMED.
# ---------------------------------------------------------------------------------------------
TUNED = {
    "arm-B":       "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1":      "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "F(2.5)":      "flmpg-,uoysntdcireahkxbwv.'jzq",
    "F(2.0)":      "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
    "candidate":   "pyu.,vdfnlhieaocstrmkj'-qgwbzx",
    "keybo-lsb":   "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
}
#: brief-named extras, kept OUT of the 13-board matrix so it stays comparable to TOURNAMENT-1
EXTRAS = {
    "keybo-c30m":   "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}


def build_boards():
    """The 13-board TOURNAMENT-1 field, community strings pulled from the pinned registry."""
    from keybo.cli.analyze import _EXTRA_NAMED
    from keybo.layouts import NAMED_LAYOUTS

    boards = dict(TUNED)
    for name in ("colemak", "colemak-dh", "graphite", "semimak", "dvorak", "qwerty"):
        boards[name] = NAMED_LAYOUTS[name]
    # cross-check the extras against the in-tree source rather than trusting my copy
    for name, want in EXTRAS.items():
        got = _EXTRA_NAMED[name]
        assert got == want, f"{name}: analyze.py says {got!r}, _guard has {want!r}"
    for name, lay in boards.items():
        assert len(lay) == 30 and len(set(lay)) == 30, f"{name} is not a 30-key permutation: {lay!r}"
    return boards


FIELD_ORDER = ("arm-B", "BALL-1", "F(2.5)", "F(2.0)", "candidate", "keybo-lsb", "flagship-c3",
               "colemak", "colemak-dh", "graphite", "semimak", "dvorak", "qwerty")

SERVE = 80      # pick2's serve bucket
MIN_N = 30      # pick2's per-pair support floor
WPM = 90.0      # the production scoring wpm
SEEDS = tuple(range(25))
