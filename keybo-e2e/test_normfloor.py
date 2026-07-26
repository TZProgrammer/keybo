"""Zero-error verification for the NORMALIZED (ceiling-fraction) floor.

Checks (all must pass before the normalized-floor search is trusted):
  1. The frozen fast_eval.CEILINGS re-derive EXACTLY from the frozen comm-pool board
     (max saved% per surface over the 46-layout reference population) — so the constant
     is auditable, not a bare copy. Uses fast_eval.SixSurface to rescore every board
     layout (which itself is 0-error vs the board, per test_fast_eval).
  2. SixSurface.normfloor reproduces the floor3-board.json ceiling-fraction floor for
     the 5 incumbents to <1e-9 (the same normalization, applied by an independent
     rescore path).
  3. normfloor_batch == normfloor per-layout (vectorized == scalar).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")

import fast_eval as FE

COMM_POOL_BOARD = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
    "comm-pool-board/tri-frequency-layouts.json"
)
FLOOR3 = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/floor3/floor3-board.json"
)
INCUMBENTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}


def test_ceilings_rederive_from_board():
    """Rescore every comm-pool board layout on the 6 surfaces with our own scorer and
    confirm the per-surface max == the frozen fast_eval.CEILINGS. This proves the
    hard-coded ceilings ARE the reference-population maxima (auditable constant)."""
    six = FE.SixSurface()
    board = json.load(open(COMM_POOL_BOARD))["board"]
    saved_by_surface = {s: [] for s in FE.SURFACES}
    for lay in board:
        sv = six.saved(FE.perm_of(lay))
        for i, s in enumerate(FE.SURFACES):
            saved_by_surface[s].append(sv[i])
    max_err = 0.0
    for s in FE.SURFACES:
        derived = max(saved_by_surface[s])
        frozen = FE.CEILINGS[s]
        err = abs(derived - frozen)
        max_err = max(max_err, err)
        assert err < 1e-9, f"{s}: derived ceiling {derived} != frozen {frozen} (err {err})"
    print(
        f"ceilings re-derive from board (n={len(board)}) max|err|={max_err:.3e}  OK"
    )


def test_normfloor_matches_floor3():
    """SixSurface.normfloor reproduces floor3-board.json six_surface ceiling-fraction
    floor for the incumbents. floor3 stored the 3-surface panels, but the 6-surface
    normalized floor = min over all 6 of saved/ceiling — recompute it from floor3's
    stored saved_all7 the same way and compare to our scorer."""
    six = FE.SixSurface()
    f3 = json.load(open(FLOOR3))
    # floor3 name -> our incumbent name
    f3names = {
        "keybo-lsb": "keybo-lsb",
        "lsb-sib": "lsb-sib",
        "archive-1843": "archive-1843",
        "archive-1846": "archive-1846",
        "keybo-lsb+lm": "keybo-lsb+lm",
    }
    max_err = 0.0
    for f3name, ourname in f3names.items():
        lay = INCUMBENTS[ourname]
        got = six.normfloor(FE.perm_of(lay))
        # reference: min over 6 surfaces of floor3 saved_all7 / our ceilings
        sv7 = f3["layouts"][f3name]["saved_all7"]
        ref = min(sv7[s] / FE.CEILINGS[s] for s in FE.SURFACES)
        err = abs(got - ref)
        max_err = max(max_err, err)
        assert err < 1e-9, f"{ourname}: normfloor {got} != floor3-derived {ref} (err {err})"
    print(f"normfloor vs floor3 saved_all7 max|err|={max_err:.3e}  OK")


def test_normfloor_batch_matches_scalar():
    six = FE.SixSurface()
    import random

    rng = random.Random(3)
    chars = list(FE.C30M)
    perms = []
    scal = []
    for _ in range(50):
        rng.shuffle(chars)
        lay = "".join(chars)
        p = FE.perm_of(lay)
        perms.append(p)
        scal.append(six.normfloor(p))
    batch = six.normfloor_batch(np.array(perms))
    max_err = float(np.max(np.abs(batch - np.array(scal))))
    assert max_err < 1e-11, f"normfloor_batch != scalar: max|err|={max_err}"
    print(f"normfloor_batch == scalar max|err|={max_err:.3e}  OK")


if __name__ == "__main__":
    test_ceilings_rederive_from_board()
    test_normfloor_matches_floor3()
    test_normfloor_batch_matches_scalar()
    print("ALL NORMFLOOR VERIFICATION PASSED")
