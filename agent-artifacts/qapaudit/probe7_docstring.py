"""FIND-pass probe 7: the module DOCSTRING describes a different algorithm than the CODE.

Docstring (qap_bound.py:6-9) describes ONLY an OUTGOING-row relaxation:
    "sort the i-th row of F descending against the k-th row of T ascending"
The CODE (qap_bound.py:49-57) computes OUTGOING + INCOMING and HALVES the sum, per the
inline comment at :45-48. Both are complete accountings of the objective; they are
DIFFERENT bounds. Which is tighter, and by how much on the real instance?
"""
import gzip, itertools, json, shutil, tempfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from keybo.geometry import ROW_STAGGERED_30
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import gilmore_lawler_bound

def _sdm(f, t): return float(np.sort(f)[::-1] @ np.sort(t))

def bound(F, T, mode):
    """mode='code' reproduces the shipped fn; 'docstring' = outgoing-only (as written)."""
    n = F.shape[0]; off = ~np.eye(n, dtype=bool); cost = np.empty((n, n))
    for i in range(n):
        f_out = F[i][off[i]]; f_in = F[:, i][off[:, i]]
        for k in range(n):
            t_out = T[k][off[k]]; t_in = T[:, k][off[:, k]]
            if mode == "code":
                cost[i, k] = F[i, i]*T[k, k] + 0.5*(_sdm(f_out, t_out) + _sdm(f_in, t_in))
            else:  # docstring: outgoing row only, full weight
                cost[i, k] = F[i, i]*T[k, k] + _sdm(f_out, t_out)
    r, c = linear_sum_assignment(cost); return float(cost[r, c].sum())

def indep_fitness(F, T, p):
    n = len(p); s = 0.0
    for i in range(n):
        for j in range(n): s += float(F[i][j])*float(T[p[i]][p[j]])
    return s
def indep_brute(F, T):
    return min(indep_fitness(F, T, p) for p in itertools.permutations(range(F.shape[0])))

# control
rng = np.random.default_rng(2)
mx = 0.0
for _ in range(30):
    n = int(rng.integers(2, 8)); F = rng.uniform(0, 10, (n, n)); T = rng.uniform(50, 250, (n, n))
    mx = max(mx, abs(bound(F, T, "code") - gilmore_lawler_bound(F, T)))
print(f"[control] bound(mode='code') reproduces shipped exactly: max abs diff {mx:.3e}")
assert mx < 1e-9

# is the DOCSTRING version also valid? and which is tighter?
nv_code = nv_doc = 0; ratios = []
for n in range(2, 7):
    for r in range(120):
        g = np.random.default_rng(hash(("d", n, r)) % (2**31))
        F = g.uniform(0, 10, (n, n)); T = g.uniform(50, 250, (n, n))
        bc, bd, opt = bound(F, T, "code"), bound(F, T, "docstring"), indep_brute(F, T)
        if bc > opt + 1e-7*abs(opt): nv_code += 1
        if bd > opt + 1e-7*abs(opt): nv_doc += 1
        ratios.append((bc - bd) / abs(opt) * 100)
ratios = np.array(ratios)
print(f"\n600 exhaustive cases n=2..6:")
print(f"  violations, CODE version:      {nv_code}")
print(f"  violations, DOCSTRING version: {nv_doc}")
print(f"  (code - docstring)/opt %: median {np.median(ratios):+.4f}  min {ratios.min():+.4f}  max {ratios.max():+.4f}")
print(f"  code tighter (larger) in {(ratios > 0).mean()*100:.1f}% of cases")

# on the REAL instance
ROOT = "/tmp/qapaudit"; QW = NAMED_LAYOUTS["qwerty"]
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
bc_ = load_freq(f"{ROOT}/data/corpus/blend-v1/bigrams.txt")
bts = [TableBigramScorer(load_model(f"bigram_reg31_seed{s}"), bc_, target_wpm=90.0, chars=QW) for s in (0,1,2)]
T2 = np.mean([s._T for s in bts], axis=0); F2 = bts[0]._F
b_code, b_doc = bound(F2, T2, "code"), bound(F2, T2, "docstring")
print(f"\nREAL 31x31 instance:")
print(f"  CODE      bound {b_code:.4f}")
print(f"  DOCSTRING bound {b_doc:.4f}")
print(f"  the code's bound is {(b_code-b_doc)/b_doc*100:+.4f}% relative to the docstring's")
# what gap would each report for the search-optimal layout (from probe6)?
best_fit = json.load(open("/tmp/qapaudit/agent-artifacts/qapaudit/probe6.json"))["best_search_fit"]
print(f"  gap at search-optimum: CODE {(best_fit-b_code)/b_code*100:.4f}%  "
      f"DOCSTRING {(best_fit-b_doc)/b_doc*100:.4f}%")

# F symmetry check: F2 is NOT symmetric (directed bigram counts) -> the halving matters
print(f"\n  F2 symmetric? {np.allclose(F2, F2.T)}   T2 symmetric? {np.allclose(T2, T2.T)}")
print(f"  max|F2 - F2.T| {np.abs(F2-F2.T).max():.4g}  max|T2 - T2.T| {np.abs(T2-T2.T).max():.4g}")

json.dump({"real_code": b_code, "real_docstring": b_doc,
           "viol_code": nv_code, "viol_docstring": nv_doc,
           "median_rel_diff_pct": float(np.median(ratios)),
           "code_tighter_frac": float((ratios > 0).mean())},
          open("/tmp/qapaudit/agent-artifacts/qapaudit/probe7.json", "w"), indent=2)
print("\nPROBE7-DONE")
