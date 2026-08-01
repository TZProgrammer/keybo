"""Shared environment guard for every priceband driver.

Import this FIRST, before numpy/xgboost, in every driver. It pins the four thread vars
(inert if set after the OpenMP runtime initializes) and asserts that `keybo` resolves to
THIS worktree, not the shared repos/keybo/src checkout (which follows whatever branch that
checkout is on -- a silent-wrong-answer footgun that has hit 6 agents)."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys

WT = "/local/home/zegertho/agent/workspaces/priceband/wt"
STATE = "/local/home/zegertho/agent/state/priceband"
ART = STATE + "/artifacts"
for _p in (WT + "/src", WT + "/drivers-priceband"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import keybo

assert keybo.__file__.startswith(WT), f"WRONG keybo checkout: {keybo.__file__}"
os.makedirs(ART, exist_ok=True)


def verify_evaluators(layouts):
    """Re-verify both fast evaluators against the shipped card()/KmStats on every run."""
    import fasteval
    import fastsfb

    fs = fasteval.FastSurface()
    w1 = fasteval.verify_against_card(fs, layouts)
    w2 = fastsfb.verify(layouts)
    assert w1 < 1e-6, f"fasteval drift {w1}"
    assert w2 < 1e-6, f"fastsfb drift {w2}"
    return fs, w1, w2
