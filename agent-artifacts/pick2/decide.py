"""pick2 step 6: the decision. Pareto dominance on the resolved axes, then a stated rule.

The logic, fixed before reading the answer:

1. SPEED IS A GATE, NOT A RANKER. Step 2 measured that the leading group is pairwise
   unresolvable on speed, so using speed to order it would be reading estimator noise. A board
   passes the gate if it is NOT RESOLVABLY SLOWER than the fastest board (paired 95% CI on the
   3 seed tables, plus sign-stability over 6 frames). Everything that passes is speed-equivalent.
2. Among gate-passers, rank on the FELT axes -- and only on axes that can actually discriminate
   (invariant / non-rankable / double-counting axes excluded, see board.py).
3. Prefer a board that DOMINATES rather than one that wins a weighted sum: a weighted sum lets
   me pick the winner by picking the weights, which is not a measurement.
4. Tie-break, in order, and each is declared a JUDGEMENT not a measurement:
   (a) raw observational support -- a gain riding on measured position n-grams beats one riding
       on model extrapolation, because the model FAILS its own transfer bar (OQ-5);
   (b) real-world provenance -- a board with users has survived tests my instrument cannot run;
   (c) adoption friction.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T95_DF2 = 4.302653
FRAMES = [(w, c) for w in (90, 110, 120) for c in ("blend-v1", "iweb")]


def main() -> int:
    b = json.loads((HERE / "board.json").read_text())
    rows, counted = b["rows"], b["axes_counted"]
    data = {f: json.loads((HERE / f"speed_wpm{f[0]}_{f[1]}.json").read_text())["rows"] for f in FRAMES}
    names = list(rows)

    def paired(a, x):
        d = np.array(rows[a]["per_seed_ms_per_char"]) - np.array(rows[x]["per_seed_ms_per_char"])
        m, sd = float(d.mean()), float(np.std(d, ddof=1))
        signs = {int(np.sign((np.array(data[f][a]["per_seed_ms_per_char"])
                             - np.array(data[f][x]["per_seed_ms_per_char"])).mean())) for f in FRAMES}
        return m, sd, T95_DF2 * sd / np.sqrt(3), len(signs) == 1

    # ---- 1. the speed gate
    fastest = min(names, key=lambda n: rows[n]["ms_per_char"])
    print(f"1. SPEED GATE  (fastest = {fastest} at {rows[fastest]['ms_per_char']:.4f} ms/char)")
    print(f"   a board passes unless it is RESOLVABLY slower: paired 95% CI (3 seeds) excludes 0")
    print(f"   AND the sign holds across all 6 frames (wpm 90/110/120 x blend-v1/iweb)\n")
    gate = []
    print(f"   {'board':14s} {'ms/char':>9s} {'vs fastest':>11s} {'sd':>7s} {'95%half':>8s} 6frames  verdict")
    for n in sorted(names, key=lambda n: rows[n]["ms_per_char"]):
        m, sd, half, stable = paired(n, fastest)
        slower = (m > half) and stable
        if not slower:
            gate.append(n)
        print(f"   {n:14s} {rows[n]['ms_per_char']:9.4f} {m:+11.4f} {sd:7.4f} {half:8.4f} "
              f"{str(stable):>7s}  {'RESOLVABLY SLOWER' if slower else 'PASSES (speed-equivalent)'}")
    print(f"\n   => {len(gate)} of {len(names)} boards are speed-equivalent to the fastest: {gate}")

    # ---- 2. Pareto on the felt axes, among gate-passers
    ax = list(counted)
    print(f"\n2. PARETO DOMINANCE among the {len(gate)} gate-passers, on {len(ax)} discriminating axes")
    print(f"   axes: {ax}")

    def better(n, o, a):
        lo = counted[a]
        x, y = rows[n]["axes"][a], rows[o]["axes"][a]
        if x == y:
            return 0
        return 1 if ((x < y) == lo) else -1

    dominated_by = {n: [] for n in gate}
    for n, o in itertools.permutations(gate, 2):
        cmps = [better(n, o, a) for a in ax]
        if all(c >= 0 for c in cmps) and any(c > 0 for c in cmps):
            dominated_by[o].append(n)
    nd = [n for n in gate if not dominated_by[n]]
    for n in gate:
        tag = "NONDOMINATED" if not dominated_by[n] else f"dominated by {dominated_by[n]}"
        print(f"   {n:14s} {tag}")
    print(f"\n   => {len(nd)} nondominated: {nd}")

    # ---- 3. per-axis leader board among gate-passers (who is best at what)
    print(f"\n3. PER-AXIS LEADERS among gate-passers (best value, and where each nondominated board ranks)")
    print(f"   {'axis':10s} {'best':14s} {'value':>9s}   " + " ".join(f"{n[:9]:>10s}" for n in nd))
    for a in ax:
        lo = counted[a]
        order = sorted(gate, key=lambda n: rows[n]["axes"][a], reverse=not lo)
        best = order[0]
        rk = {n: order.index(n) + 1 for n in gate}
        print(f"   {a:10s} {best:14s} {rows[best]['axes'][a]:9.3f}   "
              + " ".join(f"{rows[n]['axes'][a]:6.3f}#{rk[n]:<3d}" for n in nd))

    out = {"fastest": fastest, "gate_passers": gate, "nondominated": nd,
           "dominated_by": dominated_by, "axes": ax}
    (HERE / "decision.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE / 'decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
