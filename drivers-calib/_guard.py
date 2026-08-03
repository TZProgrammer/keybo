"""Shared preamble: pin threads BEFORE xgboost, force MY worktree, assert D5.

Import FIRST in every driver. Thread pins are inert once xgboost has loaded, and the sys.path
insert is load-bearing: the venv's editable install resolves `import keybo` to the SHARED
checkout /local/home/zegertho/repos/keybo, which other agents move between branches.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "48")

import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

MY_WT = "/local/home/zegertho/repos/keybo-wt-calib"
SHARED = "/local/home/zegertho/repos/keybo"
E2E = "/local/home/zegertho/keybo-e2e"
BI = f"{E2E}/bistrokes31_v1.tsv"
SHIPPED = f"{SHARED}/data/models/k31"          # READ-ONLY. Never written.
SFBPRICE_C02 = "/local/home/zegertho/agent/state/sfbprice/artifacts/c02_contrast.json"
ART = "/local/home/zegertho/agent/state/calib/artifacts"
OUT = "/local/home/zegertho/agent/workspaces/calib/out"

sys.path.insert(0, MY_WT + "/src")
os.makedirs(ART, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SERVE = 80      # the parent's / pick2's serve bucket
MIN_N = 30      # per-pair support floor
WPM = 90.0      # production scoring wpm
BOOT_SEED = 20260803   # registered in CALIB-1 PREREG


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
    assert mine == "calib", f"D5 FAIL: my worktree is on branch {mine!r}, expected 'calib'"
    return resolved


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --- the SIX class contrasts, with the definitions RECOVERED and VERIFIED against the parent -----
# Traps, both verified: row classes are EITHER-KEY (row_a==R or row_b==R), not landing-key; and in
# ROW_STAGGERED_31 y=3 is the TOP row, y=1 the BOTTOM (slots[0..9] are y=3 = qwerty q..p).
TOP_ROW, HOME_ROW, BOTTOM_ROW = 3, 2, 1


def class_masks(pairs, same_finger_ids):
    """name -> boolean list over `pairs`. Reproduces all six published ratios exactly."""
    return {
        "same-finger":        [id(r) in same_finger_ids for r in pairs],
        "same-hand (not sf)": [r["same_hand"] and id(r) not in same_finger_ids for r in pairs],
        "two rows apart":     [r["dy"] >= 2 for r in pairs],
        "bottom-row":         [BOTTOM_ROW in (r["row_a"], r["row_b"]) for r in pairs],
        "adjacent-finger":    [bool(r["adjacent"]) for r in pairs],
        "top-row":            [TOP_ROW in (r["row_a"], r["row_b"]) for r in pairs],
    }


CLASS_ORDER = ("same-finger", "same-hand (not sf)", "two rows apart",
               "bottom-row", "adjacent-finger", "top-row")

#: the parent's published table, for the E-control. (n, raw, model, ratio)
PUBLISHED = {
    "same-finger":        (55, 63.00, 41.03, 0.651),
    "same-hand (not sf)": (168, 20.00, 8.53, 0.427),
    "two rows apart":     (111, 16.50, 6.73, 0.408),
    "bottom-row":         (212, 28.75, 8.97, 0.312),
    "adjacent-finger":    (61, 32.00, 8.48, 0.265),
    "top-row":            (326, -6.75, -1.09, 0.162),
}

# --- reuse constants needed by the copied surface.py helper -------------------------------------
SEEDS = tuple(range(25))
SEEDTABLES = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/seed-tables"
