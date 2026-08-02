"""B08 — reconcile MY 8,555-move 3-opt scan against DEADCODE-1's 20,300, and re-run the
optimality test on DEADCODE-1's EXACT enumeration so the claim is apples-to-apples.

THE DISCREPANCY (I flagged this against my own claim before a verifier did):
  * DEADCODE-1 (`state/deadcode/artifacts/task1a.py:40-46,84-86`) enumerates, for every triple
    (i,j,k), all 5 non-identity reorderings => C(30,3) * 5 = 20,300 evaluations.
  * I enumerated 435 transpositions + 2*C(30,3) three-CYCLES = 8,555 evaluations.

THE CLAIM TO TEST: those 5 reorderings are 2 three-cycles + 3 SINGLE SWAPS (its own
`TRIPLE_MOVES` table says so: ("b","a","c")->[(i,j)], ("a","c","b")->[(j,k)], ("c","b","a")->
[(i,k)]). So DEADCODE-1's 20,300 contains each transposition MANY times over (once per triple
containing that pair) while mine counts each once. If true, the two scans cover the SAME SET of
distinct resulting layouts and "0 of 8,555" == "0 of 20,300" as a mathematical statement.

This driver PROVES it rather than asserting it:
  (1) enumerate both neighbourhoods as SETS of resulting permutations and compare the sets;
  (2) re-run the improvability test using DEADCODE-1's own 20,300 enumeration verbatim on the
      candidate and on arm-B, so the number I publish matches the number the ledger uses.
"""
import itertools
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "drivers-priceband"))

import numpy as np  # noqa: E402

import keybo  # noqa: E402

WT = os.path.abspath(os.path.join(HERE, ".."))
assert keybo.__file__.startswith(WT), f"WRONG KEYBO: {keybo.__file__} not under {WT}"

from fasteval import FastSurface  # noqa: E402

OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b08_neighbourhood.json"
SEED_FLOOR = 0.135

CAND = "pyu.,vdfnlhieaocstrmkj'-qgwbzx"
ARMB = "flmpg-yuo,sntdcireahkxbwv'.jzq"
BOARDS = {"CANDIDATE F(1.75)": CAND, "arm-B": ARMB,
          "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
          "F(2.0)": "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
          "F(2.5)": "flmpg-,uoysntdcireahkxbwv.'jzq"}

# DEADCODE-1's own table, copied verbatim from task1a.py:40-47
TARGETS = [t for t in itertools.permutations(("a", "b", "c")) if t != ("a", "b", "c")]
CYCLES = (("b", "c", "a"), ("c", "a", "b"))


def apply_target(perm, i, j, k, target):
    """DEADCODE-1's apply_target, verbatim: reorder the VALUES at char-indices i,j,k."""
    out = perm.copy()
    src = {"a": perm[i], "b": perm[j], "c": perm[k]}
    out[i], out[j], out[k] = src[target[0]], src[target[1]], src[target[2]]
    return out


def main():
    fs = FastSurface()
    n = 30
    out = {}

    print("=" * 96)
    print("(1) ARE THE TWO NEIGHBOURHOODS THE SAME SET? (proved by enumeration, not asserted)")
    print("=" * 96)
    p0 = fs.perm(ARMB)

    # DEADCODE-1's 20,300
    dead = set()
    ndead = 0
    for i, j, k in itertools.combinations(range(n), 3):
        for t in TARGETS:
            dead.add(tuple(apply_target(p0, i, j, k, t).tolist()))
            ndead += 1

    # mine: 435 transpositions + 2*C(30,3) cycles
    mine = set()
    nmine = 0
    for i, j in itertools.combinations(range(n), 2):
        q = p0.copy()
        q[i], q[j] = q[j], q[i]
        mine.add(tuple(q.tolist()))
        nmine += 1
    for i, j, k in itertools.combinations(range(n), 3):
        for t in CYCLES:
            mine.add(tuple(apply_target(p0, i, j, k, t).tolist()))
            nmine += 1

    print(f"  DEADCODE-1 evaluations : {ndead:6d}   distinct resulting layouts: {len(dead):6d}")
    print(f"  MY evaluations         : {nmine:6d}   distinct resulting layouts: {len(mine):6d}")
    print(f"  sets EQUAL?            : {dead == mine}")
    print(f"  in DEADCODE not in mine: {len(dead - mine)}")
    print(f"  in mine not in DEADCODE: {len(mine - dead)}")
    print(f"\n  => DEADCODE-1's 20,300 is an EVALUATION count with duplicates: each of the 435")
    print(f"     transpositions recurs once per triple containing it (28 times), and 435*28 +")
    print(f"     8120 = {435 * 28 + 8120} = 20,300. Distinct moves = {len(dead)} = 435 + 8120 = 8555.")
    out["neighbourhood"] = {"deadcode_evals": ndead, "deadcode_distinct": len(dead),
                            "mine_evals": nmine, "mine_distinct": len(mine),
                            "sets_equal": dead == mine,
                            "identity_435x28_plus_8120": 435 * 28 + 8120}
    assert dead == mine, "THE TWO NEIGHBOURHOODS DIFFER — my optimality claim would be weaker"

    print("\n" + "=" * 96)
    print("(2) RE-RUN THE OPTIMALITY TEST ON DEADCODE-1'S EXACT 20,300 ENUMERATION")
    print("=" * 96)
    print(f"{'board':22} {'base ms':>11} {'2opt/435':>9} {'3opt/20300':>11} "
          f"{'strict3/8120':>13} {'best delta':>11} {'> floor?':>9}")
    res = {}
    for name, lay in BOARDS.items():
        p = fs.perm(lay)
        base = fs.ms_per_char_perm(p)
        n2 = 0
        best2 = 0.0
        for i, j in itertools.combinations(range(n), 2):
            q = p.copy()
            q[i], q[j] = q[j], q[i]
            d = fs.ms_per_char_perm(q) - base
            if d < -1e-12:
                n2 += 1
                best2 = min(best2, d)
        n3 = n3c = 0
        best3 = best3c = 0.0
        for i, j, k in itertools.combinations(range(n), 3):
            for t in TARGETS:
                d = fs.ms_per_char_perm(apply_target(p, i, j, k, t)) - base
                if d < -1e-12:
                    n3 += 1
                    best3 = min(best3, d)
                    if t in CYCLES:
                        n3c += 1
                        best3c = min(best3c, d)
        res[name] = {"layout": lay, "base_ms": base, "n2_improving": n2, "best2": best2,
                     "n3_improving_of_20300": n3, "best3": best3,
                     "n3cycle_improving_of_8120": n3c, "best3_cycle": best3c,
                     "strict_3opt_optimum": n3 == 0,
                     "best3_clears_seed_floor": abs(best3) > SEED_FLOOR}
        print(f"{name:22} {base:11.6f} {n2:9d} {n3:11d} {n3c:13d} {best3:+11.6f} "
              f"{'YES' if abs(best3) > SEED_FLOOR else 'no':>9}")
    out["optimality"] = res

    print("\n  VERDICT:")
    for name in ("CANDIDATE F(1.75)", "arm-B"):
        r = res[name]
        print(f"    {name:22} strict 3-opt local optimum on DEADCODE-1's OWN enumeration: "
              f"{r['strict_3opt_optimum']}  ({r['n3_improving_of_20300']} of 20,300 improving)")
    print("\n  => the published claim should read '0 of 20,300 evaluated moves (8,555 distinct)',")
    print("     which is EXACTLY the DEADCODE-1 convention for arm-B. My '0 of 8,555' was the")
    print("     same fact in the distinct-move convention, NOT a weaker scan — now proved by set")
    print("     equality rather than asserted.")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
