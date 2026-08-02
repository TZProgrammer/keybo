"""B04 — the SYMMETRIC-POLISH field comparison at 3-opt depth + the convergence test.

Two things the adjudication turns on:

(1) **The 3-opt test on the candidate.** REPOLISH-1/DEADCODE-1 established arm-B is a strict
    3-opt local optimum (0 / 20,300). If the candidate board 3-opt-polishes INTO arm-B, it is
    not a distinct board and the whole argument collapses (the BALL-1 -> arm-B collapse).
    So: run 3-opt to convergence on every board and record the destination.

(2) **The symmetric-polish field**, per prereg SS2b: class OWN boards judged POLISHED,
    class COMMUNITY boards judged AS PUBLISHED. Plus the Hamming distance each polish
    moves, so caveat #3 stays visible in the same table as the score it qualifies.

3-opt neighbourhood = all C(30,3) x 2 = 8,120 3-cycles + the 20,300 - 8,120 covered by
pair moves; implemented as DEADCODE-1 did: all 3-subsets, both cyclic orientations, which
with the 435 transpositions is the full 3-opt class.
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

from fasteval import CHARS, FastSurface  # noqa: E402
from fastsfb import FastGauges  # noqa: E402

TABLE = "/local/home/zegertho/agent/state/bestfinal/artifacts/b02_master_table.json"
OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b04_symmetric_3opt.json"
SEED_FLOOR = 0.135

TRIPLES = list(itertools.combinations(range(30), 3))


def moves_3opt(p):
    """Every 2-swap and 3-cycle neighbour of p, as a (M, 31) array of char->slot perms.

    p maps char-index -> slot-index. A SLOT-level move relabels p's values.
    """
    out = []
    # 435 transpositions
    for i, j in itertools.combinations(range(30), 2):
        q = p.copy()
        q[p == i] = j
        q[p == j] = i
        out.append(q)
    # 2 x C(30,3) 3-cycles
    for i, j, k in TRIPLES:
        for a, b, c in ((j, k, i), (k, i, j)):
            q = p.copy()
            q[p == i] = a
            q[p == j] = b
            q[p == k] = c
            out.append(q)
    return np.array(out)


def to_str(p):
    s2c = {int(s): CHARS[i] for i, s in enumerate(p[:30])}
    return "".join(s2c[i] for i in range(30))


def polish3(fs, lay, max_sweeps=60):
    p = fs.perm(lay)
    cur = fs.ms_per_char_perm(p)
    n_first = None
    best_first = None
    for sweep in range(max_sweeps):
        M = moves_3opt(p)
        vals = np.array([fs.ms_per_char_perm(q) for q in M])
        d = vals - cur
        if sweep == 0:
            n_first = int((d < -1e-12).sum())
            best_first = float(d.min())
        k = int(np.argmin(vals))
        if vals[k] >= cur - 1e-12:
            break
        p, cur = M[k], float(vals[k])
    return {
        "n_moves_scanned": len(M),
        "n_improving_3opt_at_start": n_first,
        "best_3opt_delta_at_start": best_first,
        "at_own_3opt_optimum": n_first == 0,
        "polished3_ms": cur,
        "polished3_layout": to_str(p),
        "gain": fs.ms_per_char(lay) - cur,
        "hamming_moved": sum(1 for a, b in zip(lay, to_str(p)) if a != b),
    }


def main():
    fs = FastSurface()
    fg = FastGauges()
    rows = json.load(open(TABLE))["rows"]

    print("== RECONCILIATION ==")
    m = fs.ms_per_char(rows["arm-B"]["layout"])
    print(f"  arm-B {m:.6f} vs published 253.900579  diff {abs(m - 253.900579):.2e}")
    assert abs(m - 253.900579) < 1e-5

    CAND = "FRONTIER@sfb<=1.75"
    print(f"\n== 3-OPT POLISH TO CONVERGENCE ({len(TRIPLES) * 2 + 435} moves per sweep) ==")
    res = {}
    for name in sorted(rows):
        lay = rows[name]["layout"]
        r = polish3(fs, lay)
        r["layout"] = lay
        r["sfb_before"] = fg.sfb_only(fg.perm(lay))
        r["sfb_after"] = fg.sfb_only(fg.perm(r["polished3_layout"]))
        r["class"] = rows[name]["class"]
        res[name] = r
        print(f"  {name:24s} {rows[name]['ms_per_char']:9.4f} -> {r['polished3_ms']:9.4f} "
              f"(gain {r['gain']:+7.4f}, {r['gain'] / SEED_FLOOR:5.2f} fl) "
              f"n3opt={r['n_improving_3opt_at_start']:5d} hamming={r['hamming_moved']:3d} "
              f"sfb {r['sfb_before']:.4f}->{r['sfb_after']:.4f}")

    print("\n== (1) THE COLLAPSE TEST — where does each board 3-opt-polish TO? ==")
    from collections import defaultdict
    grp = defaultdict(list)
    for k, r in res.items():
        grp[r["polished3_layout"]].append(k)
    for lay, ks in sorted(grp.items(), key=lambda x: res[x[1][0]]["polished3_ms"]):
        r0 = res[ks[0]]
        print(f"  {lay}  ms={r0['polished3_ms']:9.4f} sfb={r0['sfb_after']:.4f}"
              f"  <- {', '.join(sorted(ks))}")

    armb = res["arm-B"]["polished3_layout"]
    cand3 = res[CAND]["polished3_layout"]
    collapsed = cand3 == armb
    print(f"\n  CANDIDATE {CAND}:")
    print(f"    3-opt improving moves at start = {res[CAND]['n_improving_3opt_at_start']}")
    print(f"    3-opt destination == arm-B ?  {collapsed}   "
          f"{'*** COLLAPSES -> argument dead' if collapsed else '*** DISTINCT -> survives'}")
    print(f"    hamming(candidate, arm-B) = "
          f"{sum(1 for a, b in zip(rows[CAND]['layout'], rows['arm-B']['layout']) if a != b)}")

    print("\n== (2) SYMMETRIC-POLISH FIELD (prereg SS2b: OWN polished, COMMUNITY as published) ==")
    print(f"{'board':24} {'class':>9} {'judged ms':>11} {'sfb':>8} {'d vs best':>10} "
          f"{'seed fl':>8} {'hamming':>8}")
    judged = {}
    for name, r in res.items():
        if r["class"] == "COMMUNITY":
            judged[name] = {"ms": rows[name]["ms_per_char"], "sfb": r["sfb_before"],
                            "basis": "as-published", "hamming": 0,
                            "layout": rows[name]["layout"]}
        else:
            judged[name] = {"ms": r["polished3_ms"], "sfb": r["sfb_after"],
                            "basis": "3-opt polished", "hamming": r["hamming_moved"],
                            "layout": r["polished3_layout"]}
    best = min(judged.values(), key=lambda v: v["ms"])["ms"]
    for name in sorted(judged, key=lambda k: judged[k]["ms"]):
        j = judged[name]
        print(f"{name:24} {res[name]['class']:>9} {j['ms']:11.4f} {j['sfb']:8.4f} "
              f"{j['ms'] - best:+10.4f} {(j['ms'] - best) / SEED_FLOOR:8.2f} {j['hamming']:8d}")

    json.dump({"three_opt": res, "judged": judged, "candidate": CAND,
               "candidate_collapses_into_armB": collapsed},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
