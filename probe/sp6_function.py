"""SP6 — the penalty FUNCTION for `scissor`: is one number enough, or does it need a split?

The brief asks for dy1/dy2, per-finger, or narrow/wide. FIRST a structural fact that
reshapes the question (read from the shipped code, not assumed):

    is_scissor(g,a,b) = is_adjacent(g,a,b) AND abs(a[1]-b[1]) == 2      (classify.py:99)

So the term weighted +4.0 is the NARROW, dy==2-ONLY predicate. Consequences:
  * a dy1/dy2 split is NOT a split of this term -- dy1 is not in its support at all. Asking
    for it is asking about `bad_scissor` (BADSCISSOR-1's CROSS-CUT: adds 72 dy=1 descents,
    drops 12 of the 24 narrow pairs) or about wscissor-GRADED. Those are DIFFERENT terms and
    re-pricing them is not re-pricing this weight.
  * narrow/wide is likewise not a split: `is_scissor` IS the narrow one. "Wide" pairs
    (non-adjacent 2-row) are OUTSIDE it -- and trap 12 records that the served OBJECTIVE
    already prices them, but this SCORER does not.
  * per-finger IS a legitimate split of the actual support.

So this probe prices, with THEORY-1's byte-identical matched estimator (positive-controlled at
165 cells diff 0.0 in my own worktree):
  1. the shipped support, decomposed by FINGER PAIR -- the only split that partitions it;
  2. the shipped support vs the two neighbours it excludes (dy==1 adjacent, dy==2
     NON-adjacent), to price what a re-scoped predicate would buy;
  3. a same-support HETEROGENEITY test: is one number enough for the pairs it does cover?
  4. per-pair enumeration so the support is auditable (how many ordered pairs, which fingers).

Every contrast reports its strata count and the disjointness check (trap 16): a contrast whose
member/non-member key sets are disjoint is NOT identified, and the estimator's own
`n_strata`/`frac_pos` are quoted alongside every number.

FRAME: T2 = g(geometry, wpm=90), ROW_STAGGERED_30, space excluded, ordered distinct pairs
(870). MODELLED only -- no realized-speed claim.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/scissorprice/probe")
import matched_prices as M  # noqa: E402  (byte-identical THEORY-1 estimator, md5 38294e1b...)

from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402

A = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1"
OUT = "/local/home/zegertho/agent/state/scissorprice/artifacts"
TABLES = {
    "AALTO": np.load(f"{A}/T2_prod.npy"),
    "COMMUNITY": np.load(f"{A}/T2_comm.npy"),
    "POOL": np.load(f"{A}/T2_pool.npy"),
}
SRCS = ("AALTO", "COMMUNITY", "POOL")
SLOTS = M.SLOTS
PAIRS = [(a, b) for a in SLOTS for b in SLOTS if a != b]
print(f"ordered distinct pairs: {len(PAIRS)}")


def fing(p):
    ax = abs(p[0])
    return "pinky" if ax in (5, 6) else "ring" if ax == 4 else "middle" if ax == 3 else "index"


def fpair(ab):
    return tuple(sorted((fing(ab[0]), fing(ab[1]))))


def rowspan(ab):
    return abs(ab[0][1] - ab[1][1])


def sci(ab):
    return C.is_scissor(G, ab[0], ab[1])


def adj(ab):
    return C.is_adjacent(G, ab[0], ab[1])


def samehand(ab):
    return C.same_hand(G, ab[0], ab[1])


def samefinger(ab):
    return C.same_finger(G, ab[0], ab[1])


res = {}

# ================== 0. AUDIT THE SUPPORT (positive self-check) ============================
supp = [ab for ab in PAIRS if sci(ab)]
print(f"\n{'='*80}\n0. THE SHIPPED SUPPORT: is_scissor fires on {len(supp)} of {len(PAIRS)} ordered pairs")
by_fp, by_dy = {}, {}
for ab in supp:
    by_fp[fpair(ab)] = by_fp.get(fpair(ab), 0) + 1
    by_dy[rowspan(ab)] = by_dy.get(rowspan(ab), 0) + 1
print(f"   rowspan distribution: {by_dy}   (must be {{2: n}} only -- the predicate is dy==2)")
assert set(by_dy) == {2}, f"support is not dy==2 only: {by_dy}"
print("   by finger pair: " + ", ".join(f"{'-'.join(k)}={v}" for k, v in sorted(by_fp.items(), key=lambda kv: -kv[1])))
assert all(samehand(ab) and not samefinger(ab) for ab in supp), "support has cross-hand or same-finger pairs"
print("   self-checks: all same-hand, all distinct fingers, all adjacent, all dy==2 ✓")
res["support"] = {"n_ordered_pairs": len(supp),
                  "by_finger_pair": {"-".join(k): v for k, v in by_fp.items()},
                  "by_rowspan": {str(k): v for k, v in by_dy.items()}}

# ================== 1. HEADLINE, reproduced: scissor vs adjacent-FLAT ====================
LAND = lambda ab: M.land_sig(ab[1])  # noqa: E731
print(f"\n{'='*80}\n1. HEADLINE contrast (reproduces the dossier's scissor row)")
print(f"   {'contrast':44s}" + "".join(f"{s[:4]:>9s}{'nS':>5s}{'%+':>6s}" for s in SRCS))


def show(label, mem, non, strata=LAND, store=None, key=None):
    out = {}
    line = f"   {label:44s}"
    for s in SRCS:
        r = M.matched(TABLES[s], mem, non, strata)
        out[s] = {k: float(r[k]) if isinstance(r[k], (int, float, np.floating)) else r[k]
                  for k in ("delta_ms", "n_strata", "frac_pos")}
        line += f"{r['delta_ms']:+9.2f}{int(r['n_strata']):5d}{100*r['frac_pos']:6.0f}"
    print(line)
    if store is not None:
        store[key or label] = out
    return out


h = {}
show("scissor vs adjacent-flat (dy0)", sci, lambda ab: adj(ab) and rowspan(ab) == 0, store=h)
show("scissor vs adjacent dy1", sci, lambda ab: adj(ab) and rowspan(ab) == 1, store=h)
show("scissor vs NONadj dy2 (the 'wide' pairs)", sci,
     lambda ab: samehand(ab) and not samefinger(ab) and not adj(ab) and rowspan(ab) == 2, store=h)
show("scissor vs all same-hand 2-finger", sci,
     lambda ab: samehand(ab) and not samefinger(ab) and not sci(ab), store=h)
res["headline"] = h

# ================== 2. PER-FINGER-PAIR decomposition — the only true split ===============
print(f"\n{'='*80}\n2. PER-FINGER-PAIR (the only split that partitions the shipped support)")
print("   each row: THAT finger pair's dy2-adjacent pairs vs THAT SAME pair's flat pairs")
print("   (so the finger identity is held FIXED and only the row span varies -- otherwise the")
print("    contrast swaps the key set, trap 16)")
print(f"   {'finger pair':20s}{'nPairs':>7s}" + "".join(f"{s[:4]:>9s}{'nS':>5s}{'%+':>6s}" for s in SRCS))
pf = {}
for k in sorted(by_fp, key=lambda k: -by_fp[k]):
    mem = lambda ab, k=k: sci(ab) and fpair(ab) == k
    non = lambda ab, k=k: adj(ab) and rowspan(ab) == 0 and fpair(ab) == k
    nmem = sum(1 for ab in PAIRS if mem(ab))
    nnon = sum(1 for ab in PAIRS if non(ab))
    # trap 16 disjointness: do the two groups share any KEY at all?
    kmem = {p for ab in PAIRS if mem(ab) for p in ab}
    knon = {p for ab in PAIRS if non(ab) for p in ab}
    line = f"   {'-'.join(k):20s}{nmem:7d}"
    row = {"n_member_pairs": nmem, "n_nonmember_pairs": nnon,
           "key_overlap": len(kmem & knon), "disjoint_keys": len(kmem & knon) == 0}
    for s in SRCS:
        r = M.matched(TABLES[s], mem, non, LAND)
        row[s] = {"delta_ms": float(r["delta_ms"]), "n_strata": int(r["n_strata"]),
                  "frac_pos": float(r["frac_pos"])}
        line += f"{r['delta_ms']:+9.2f}{int(r['n_strata']):5d}{100*r['frac_pos']:6.0f}"
    line += f"   keyoverlap={row['key_overlap']}" + ("  ** DISJOINT **" if row["disjoint_keys"] else "")
    print(line)
    pf["-".join(k)] = row
res["per_finger_pair"] = pf

# ================== 3. HETEROGENEITY: is one number enough for this support? =============
print(f"\n{'='*80}\n3. HETEROGENEITY across finger pairs (does the support need a split?)")
for s in SRCS:
    vals = [pf[k][s]["delta_ms"] for k in pf if pf[k]["n_strata" if False else s]["n_strata"] > 0]
    ok = [k for k in pf if pf[k][s]["n_strata"] > 0]
    if len(vals) >= 2:
        print(f"   {s:10s} n_pairs_priced={len(vals)}  range [{min(vals):+.2f}, {max(vals):+.2f}]"
              f"  spread {max(vals)-min(vals):.2f} ms  sd {np.std(vals):.2f}"
              f"  signs {'ALL +' if min(vals)>0 else 'MIXED'}")
        print(f"              priced: {', '.join(f'{k}={pf[k][s]['delta_ms']:+.1f}' for k in ok)}")
res["heterogeneity"] = {
    s: {"values": {k: pf[k][s]["delta_ms"] for k in pf if pf[k][s]["n_strata"] > 0}} for s in SRCS
}

# ================== 4. WHAT A RE-SCOPED PREDICATE WOULD BUY ==============================
print(f"\n{'='*80}\n4. THE NEIGHBOURS THE SHIPPED PREDICATE EXCLUDES (priced vs adjacent-flat)")
print("   -> is the +4.0 term's SUPPORT wrong, independently of its LEVEL?")
nb = {}
show("adjacent dy1 (excluded)  vs adj flat", lambda ab: adj(ab) and rowspan(ab) == 1,
     lambda ab: adj(ab) and rowspan(ab) == 0, store=nb)
show("NONadj dy2 ('wide', excluded) vs adj flat",
     lambda ab: samehand(ab) and not samefinger(ab) and not adj(ab) and rowspan(ab) == 2,
     lambda ab: adj(ab) and rowspan(ab) == 0, store=nb)
show("NONadj dy1 (excluded) vs adj flat",
     lambda ab: samehand(ab) and not samefinger(ab) and not adj(ab) and rowspan(ab) == 1,
     lambda ab: adj(ab) and rowspan(ab) == 0, store=nb)
res["excluded_neighbours"] = nb

json.dump(res, open(f"{OUT}/sp6_penalty_function.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp6_penalty_function.json")
