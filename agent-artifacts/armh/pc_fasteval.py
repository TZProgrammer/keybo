"""POSITIVE CONTROL: does `evobj.FastEval` (the SEARCH path) agree with the SHIPPED
`keybo analyze` (the JUDGE path) on every gauge ARM H optimizes or judges on?

This is the gate on ARM H's whole design: it searches a FastEval objective and adjudicates
on shipped-analyze gauges. If the two paths disagree, every number is a cross-path artifact
(trap 13: two numbers under one gauge name).

ADAPTED from ARM G's `pc_fasteval.py` with the worktree literal repointed /tmp/armg ->
/tmp/armh (trap 35: a harvested driver's hardcoded path silently un-isolates a worktree),
and EXTENDED with:
  * `--mutate` : plants a multiplicative factor on the FastEval side and requires rc=1.
    A control that cannot fail tests nothing.
  * the two extra layouts ARM H's verdicts actually rest on: BALL-1 (the feasible 1-swap
    neighbour of arm B) and the six frozen champions.

Run BEFORE any ARM H search result exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import evobj as EV  # noqa: E402

WORKTREE = Path("/tmp/armh")

import keybo  # noqa: E402

assert str(Path(keybo.__file__).resolve()).startswith("/tmp/armh/"), keybo.__file__

LAYOUTS = {
    "arm-B":        "flmpg-yuo,sntdcireahkxbwv'.jzq",
    #: BALL-1: the ONE layout in arm B's exhaustive 1-swap ball satisfying all 13 hard
    #: axis constraints. ARM H's headline rests on this layout, so it must be in the control.
    "BALL-1":       "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "arm-A":        "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "keybo-lsb":    "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "flagship-c3":  "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "graphite":     "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "qwerty":       "qwertyuiopasdfghjkl'zxcvbnm,.-",
    "six-s907919":  "puy.,vdfnlheioamtsrc'jqk-gwbxz",
    "six-s915838":  "pyou,vdflrghaeictsnmk'j.-wbzxq",
    "six-s923757":  "lcfmk.uoyprnstdiaeghzxwbv-,'qj",
    "six-s931676":  "lnfdg.,yehcrstmaoiupxzbwvk-q'j",
    "six-s939595":  "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
}
GAUGES = ("sfb", "sfs", "sfb-dist", "sfs-dist", "lsb", "lsb-dist", "alt", "roll", "sr-roll",
          "redir", "scissor", "imbalance", "oxey-style", "comfort")


def main() -> int:
    mutate = "--mutate" in sys.argv
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armh/"), fe.corpus_dir
    print(f"FastEval.corpus_dir = {fe.corpus_dir}  (POSITIVE worktree control)")
    print(f"keybo.__file__      = {keybo.__file__}")
    print(f"MUTATION            = {mutate}  (planted factor on the FastEval side)")

    names = list(LAYOUTS)
    perms = np.stack([EV.perm_of(LAYOUTS[n]) for n in names])
    g = {k: v.copy() for k, v in fe.gauges(perms).items()}
    if mutate:
        # plant the factor on the gauge ARM H's headline rests on
        g["oxey-style"] = g["oxey-style"] * 1.000000001

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "8"
    cmd = ["uv", "run", "--no-sync", "keybo", "analyze", "--json"] + [LAYOUTS[n] for n in names]
    p = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    blob = json.loads(p.stdout)
    rows = blob["rows"]
    # trap 38: `analyze` once keyed `rows` on a truncated spec and silently dropped a row, so
    # the row count must be asserted. `analyze` ALWAYS injects its own `qwerty` reference row
    # (verified: 2 requested layouts -> 3 rows, the extra keyed literally 'qwerty'), and my
    # request list already contains qwerty BY STRING, which arrives under its own key. So the
    # expected count is distinct-requested + 1 for the injected reference. Assert the exact
    # identity rather than a >= (a >= is the tie-credit defect wearing a row-count costume).
    expect = set(LAYOUTS.values()) | {"qwerty"}
    assert set(rows) == expect, (
        f"analyze row-key set mismatch: missing {sorted(expect - set(rows))}, "
        f"unexpected {sorted(set(rows) - expect)}")
    for n in LAYOUTS:
        assert LAYOUTS[n] in rows, f"{n} dropped from analyze output"

    out: dict = {"worktree": str(WORKTREE), "keybo_file": keybo.__file__,
                 "corpus_dir": str(fe.corpus_dir), "mutated": mutate,
                 "n_layouts": len(names), "cells": {}, "worst": {}}
    worst_overall = 0.0
    print(f"\n{'gauge':<12} {'worst |rel diff|':>18}  {'worst |abs diff|':>18}  bit-exact")
    n_bitexact = 0
    for gg in GAUGES + ("_ms_per_char",):
        wr = wa = 0.0
        for i, n in enumerate(names):
            row = rows[LAYOUTS[n]]
            ship = row["time"]["ms_per_char"] if gg == "_ms_per_char" else row["gauges"][gg]
            fast = float(g[gg][i])
            a = abs(fast - ship)
            r = a / max(abs(ship), 1e-12)
            out["cells"].setdefault(gg, {})[n] = {"shipped": ship, "fasteval": fast,
                                                 "abs": a, "rel": r}
            wr = max(wr, r)
            wa = max(wa, a)
        out["worst"][gg] = {"rel": wr, "abs": wa, "bit_exact": wa == 0.0}
        n_bitexact += wa == 0.0
        worst_overall = max(worst_overall, wr)
        print(f"{gg:<12} {wr:>18.3e}  {wa:>18.3e}  {'YES' if wa == 0.0 else 'no'}")
    out["worst_rel_overall"] = worst_overall
    out["n_bit_exact_of_15"] = int(n_bitexact)
    out["armB_frozen"] = 253.90057910352604
    out["armB_shipped"] = rows[LAYOUTS["arm-B"]]["time"]["ms_per_char"]
    out["armB_fasteval"] = float(g["_ms_per_char"][names.index("arm-B")])
    out["BALL1_shipped_oxey"] = rows[LAYOUTS["BALL-1"]]["gauges"]["oxey-style"]
    out["BALL1_shipped_ms"] = rows[LAYOUTS["BALL-1"]]["time"]["ms_per_char"]
    print("\narm B frozen   253.90057910352604")
    print(f"arm B shipped  {out['armB_shipped']!r}  "
          f"diff {abs(out['armB_shipped'] - out['armB_frozen']):.3e}")
    print(f"arm B fasteval {out['armB_fasteval']!r}  "
          f"diff {abs(out['armB_fasteval'] - out['armB_frozen']):.3e}")
    print(f"BALL-1 shipped oxey {out['BALL1_shipped_oxey']!r}  ms {out['BALL1_shipped_ms']!r}")
    dest = Path(HERE / ("pc_fasteval_MUTATED.json" if mutate else "pc_fasteval.json"))
    json.dump(out, open(dest, "w"), indent=1)
    rc = 0 if worst_overall < 1e-9 else 1
    print(f"\nWROTE {dest}  worst_rel_overall={worst_overall:.3e}  "
          f"bit_exact={n_bitexact}/15  rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
