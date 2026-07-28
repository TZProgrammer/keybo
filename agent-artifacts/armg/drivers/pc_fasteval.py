"""POSITIVE CONTROL: does evobj.FastEval (the SEARCH path) agree with the SHIPPED
`keybo analyze` (the SCORING path) on every gauge I intend to optimize or judge on?

This is the gate. ARM G's whole design rests on searching a FastEval objective and then
judging on shipped-analyze gauges. If the two paths disagree, every number I report is a
cross-path artifact (trap 13: two numbers under one gauge name).

Run BEFORE any search result exists.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import evobj as EV

WORKTREE = Path("/tmp/armg")

# worktree isolation POSITIVE control (not "no hardcodes found" — trap 35)
import keybo
assert str(Path(keybo.__file__).resolve()).startswith("/tmp/armg/"), keybo.__file__

LAYOUTS = {
    "arm-B":        "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "arm-A":        "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "keybo-lsb":    "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "flagship-c3":  "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "graphite":     "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "qwerty":       "qwertyuiopasdfghjkl'zxcvbnm,.-",
}
GAUGES = ("sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll",
          "redir","scissor","imbalance","oxey-style","comfort")

def main() -> int:
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armg/"), fe.corpus_dir
    print(f"FastEval.corpus_dir = {fe.corpus_dir}  (POSITIVE worktree control)")
    print(f"keybo.__file__      = {keybo.__file__}")

    names = list(LAYOUTS)
    perms = np.stack([EV.perm_of(LAYOUTS[n]) for n in names])
    g = fe.gauges(perms)

    # shipped analyze on the same layouts, one call
    cmd = ["uv","run","--no-sync","keybo","analyze","--json"] + [LAYOUTS[n] for n in names]
    p = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    blob = json.loads(p.stdout)
    rows = blob["rows"]
    assert len(rows) >= len(set(LAYOUTS.values())), f"analyze dropped a row: {len(rows)}"

    out = {"worktree": str(WORKTREE), "keybo_file": keybo.__file__,
           "corpus_dir": str(fe.corpus_dir), "cells": {}, "worst": {}}
    worst_overall = 0.0
    print(f"\n{'gauge':<12} {'worst |rel diff|':>18}  {'worst |abs diff|':>18}")
    for gg in GAUGES + ("_ms_per_char",):
        wr = wa = 0.0
        for i, n in enumerate(names):
            row = rows[LAYOUTS[n]]
            ship = row["time"]["ms_per_char"] if gg == "_ms_per_char" else row["gauges"][gg]
            fast = float(g[gg][i])
            a = abs(fast - ship); r = a / max(abs(ship), 1e-12)
            out["cells"].setdefault(gg, {})[n] = {"shipped": ship, "fasteval": fast,
                                                  "abs": a, "rel": r}
            wr = max(wr, r); wa = max(wa, a)
        out["worst"][gg] = {"rel": wr, "abs": wa}
        worst_overall = max(worst_overall, wr)
        print(f"{gg:<12} {wr:>18.3e}  {wa:>18.3e}")
    out["worst_rel_overall"] = worst_overall
    # arm B frozen value re-check through BOTH paths
    out["armB_frozen"] = 253.90057910352604
    out["armB_shipped"] = rows[LAYOUTS["arm-B"]]["time"]["ms_per_char"]
    out["armB_fasteval"] = float(g["_ms_per_char"][names.index("arm-B")])
    print(f"\narm B frozen   253.90057910352604")
    print(f"arm B shipped  {out['armB_shipped']!r}  diff {abs(out['armB_shipped']-out['armB_frozen']):.3e}")
    print(f"arm B fasteval {out['armB_fasteval']!r}  diff {abs(out['armB_fasteval']-out['armB_frozen']):.3e}")
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE/"pc_fasteval.json"
    json.dump(out, open(dest,"w"), indent=1)
    print(f"\nWROTE {dest}  worst_rel_overall={worst_overall:.3e}")
    return 0 if worst_overall < 1e-9 else 1

if __name__ == "__main__":
    sys.exit(main())
