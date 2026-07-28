"""FIND-pass probe 9: the bound bounds a LARGER feasible set than the search explores.

The driver pins SPACE at slot 30 (cond_rebuild.py:161,171: `np.append(rng.permutation(N), N)`
with N=30, and TableBigramScorer.permutation():111 pins perm[30]=space_slot).
`gilmore_lawler_bound(F2, T2)` gets bare 31x31 arrays and has NO notion of a pinned index —
it minimizes over ALL 31! permutations.

min over 31! <= min over {31! : space pinned}, so the bound is still VALID for the pinned
problem — but it is bounding a RELAXATION of the actual problem, and every extra pct-pt of
slack lands in the quoted certificate. How much?
"""
import gzip, itertools, json, shutil, tempfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import gilmore_lawler_bound, certificate

ROOT = "/tmp/qapaudit"; QW = NAMED_LAYOUTS["qwerty"]; geom = ROW_STAGGERED_30; N = 30
def load_freq(p):
    o = {}
    for ln in open(p):
        q = ln.rstrip("\n").split("\t")
        if len(q) == 2: o[q[0]] = int(q[1])
    return o
def load_model(stem):
    d = tempfile.mkdtemp()
    for suf in (".json", ".meta.json"):
        with gzip.open(f"{ROOT}/data/models/k31/{stem}{suf}.gz", "rb") as fi, open(f"{d}/{stem}{suf}", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return XGBoostTypingModel.load(f"{d}/{stem}.json")
bi = load_freq(f"{ROOT}/data/corpus/blend-v1/bigrams.txt")
bts = [TableBigramScorer(load_model(f"bigram_reg31_seed{s}"), bi, target_wpm=90.0, chars=QW) for s in (0,1,2)]
T2 = np.mean([s._T for s in bts], axis=0); F2 = bts[0]._F
lb_free = gilmore_lawler_bound(F2, T2)

# --- rebuild the GL cost matrix (positive-controlled) and inspect WHERE space goes -------
def gl_cost(F, T):
    n = F.shape[0]; off = ~np.eye(n, dtype=bool); cost = np.empty((n, n))
    for i in range(n):
        f_out = F[i][off[i]]; f_in = F[:, i][off[:, i]]
        for k in range(n):
            t_out = T[k][off[k]]; t_in = T[:, k][off[:, k]]
            cost[i, k] = F[i, i]*T[k, k] + 0.5*(
                float(np.sort(f_out)[::-1] @ np.sort(t_out)) +
                float(np.sort(f_in)[::-1] @ np.sort(t_in)))
    return cost
C = gl_cost(F2, T2)
r, c = linear_sum_assignment(C)
assert abs(C[r, c].sum() - lb_free) < 1e-6, "control failed: my cost matrix != shipped bound"
print(f"[control] my GL cost matrix reproduces the shipped bound: {C[r,c].sum():.4f} vs {lb_free:.4f}")

SPACE_CHAR_IDX = 30   # TableBigramScorer._index[' '] = len(chars) = 30
SPACE_SLOT     = 30   # _space_slot = len(geometry.slots) = 30
assigned_slot_for_space = int(c[list(r).index(SPACE_CHAR_IDX)])
print(f"\nGL's own LAP assignment puts the SPACE character at slot {assigned_slot_for_space} "
      f"(the search REQUIRES slot {SPACE_SLOT})")
print(f"  -> the unconstrained bound {'DOES' if assigned_slot_for_space == SPACE_SLOT else 'DOES NOT'} respect the pin")

# --- the pinned bound: force space->slot 30 by making every other slot infeasible --------
BIG = 1e30
Cp = C.copy()
Cp[SPACE_CHAR_IDX, :] = BIG; Cp[SPACE_CHAR_IDX, SPACE_SLOT] = C[SPACE_CHAR_IDX, SPACE_SLOT]
Cp[:, SPACE_SLOT] = BIG;     Cp[SPACE_CHAR_IDX, SPACE_SLOT] = C[SPACE_CHAR_IDX, SPACE_SLOT]
rp, cp = linear_sum_assignment(Cp)
lb_pinned = float(Cp[rp, cp].sum())
assert np.isfinite(lb_pinned) and lb_pinned < BIG
print(f"\nGL bound, space FREE   (what the code computes): {lb_free:.4f}")
print(f"GL bound, space PINNED (the actual problem)    : {lb_pinned:.4f}")
print(f"  the pinned bound is {(lb_pinned-lb_free)/lb_free*100:+.4f}% tighter")

best_fit = json.load(open("/tmp/qapaudit/agent-artifacts/qapaudit/probe6.json"))["best_search_fit"]
print(f"\ngap at the search-optimal layout:")
print(f"  vs FREE bound   (as shipped): {(best_fit-lb_free)/lb_free*100:.4f}%   <-- the QUOTED number")
print(f"  vs PINNED bound (correct set): {(best_fit-lb_pinned)/lb_pinned*100:.4f}%")
print(f"  slack attributable to ignoring the pin: "
      f"{(best_fit-lb_free)/lb_free*100 - (best_fit-lb_pinned)/lb_pinned*100:+.4f} pct-pts")

# --- does the certificate at least mention which set it bounds? -------------------------
c_ = certificate(F2, T2, best_fit)
print(f"\nshipped statement string:\n  \"{c_['statement']}\"")
for kw in ("bigram", "component", "quadratic", "space", "pinned"):
    print(f"  mentions '{kw}': {kw in c_['statement'].lower()}")
print(f"  dict keys: {sorted(c_.keys())}")

# --- SMALL-CASE CONTROL: verify the pinned-vs-free claim exhaustively -------------------
print("\n[exhaustive control, n=5, space:=index 4 pinned to slot 4]")
def indep_fitness(F, T, p):
    n = len(p); s = 0.0
    for i in range(n):
        for j in range(n): s += float(F[i][j])*float(T[p[i]][p[j]])
    return s
nv = 0
for rr in range(200):
    g = np.random.default_rng(rr); n = 5
    F = g.uniform(0, 10, (n, n)); T = g.uniform(50, 250, (n, n))
    lbf = gilmore_lawler_bound(F, T)
    opt_pin = min(indep_fitness(F, T, list(p) + [n-1])
                  for p in itertools.permutations(range(n-1)))
    if lbf > opt_pin + 1e-7*abs(opt_pin): nv += 1
print(f"  free bound > pinned-problem optimum in {nv}/200 cases (must be 0: bound stays VALID)")

json.dump({"lb_free": float(lb_free), "lb_pinned": float(lb_pinned),
           "space_slot_from_LAP": assigned_slot_for_space,
           "gap_free_pct": float((best_fit-lb_free)/lb_free*100),
           "gap_pinned_pct": float((best_fit-lb_pinned)/lb_pinned*100),
           "statement": c_["statement"], "pin_violations": nv},
          open("/tmp/qapaudit/agent-artifacts/qapaudit/probe9.json", "w"), indent=2)
print("\nPROBE9-DONE")
