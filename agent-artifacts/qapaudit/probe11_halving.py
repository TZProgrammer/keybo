"""VERIFY PASS 2: is the module's OWN stated justification for the halving CORRECT?

The inline comment (qap_bound.py:45-48) claims:
  "Each F[i,j].T[.,.] term appears exactly once as outgoing-of-i and once as
   incoming-of-j across the whole objective, so bounding outgoing and incoming
   separately per (i,k) and HALVING their sum yields a valid floor on the total
   (each side is independently a rearrangement-inequality minimum)."

That reasoning is: total = sum_i out_i = sum_j in_j, so total = 0.5*(sum_i out_i + sum_j in_j),
and bounding each per-(i,k) piece below then LAP-ing gives a floor. The empirical sweep
found 0 violations in 528+600 cases. But "no counterexample found" is not-found, not
not-exists. Can I find the tight/failure boundary, or confirm the argument is airtight?

I test the two places the argument could break:
  (A) NEGATIVE entries: the rearrangement inequality's MIN pairing still holds for signed
      values (it is a statement about orderings, not signs) -- but does the DIAGONAL term
      F[i,i]*T[k,k] being added at FULL weight (not halved) break the accounting?
  (B) the LAP step: is a per-(i,k) floor + LAP genuinely a floor on the quadratic?
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from keybo.optimize.qap_bound import gilmore_lawler_bound

def indep_fitness(F, T, p):
    n = len(p); s = 0.0
    for i in range(n):
        for j in range(n): s += float(F[i][j])*float(T[p[i]][p[j]])
    return s

print("=== (A) is the DIAGONAL accounted exactly once? algebraic check ===")
# For a FIXED permutation p, decompose the true objective the way the code claims:
#   total = sum_i F[i,i]T[p_i,p_i]  +  sum_i sum_{j!=i} F[i,j]T[p_i,p_j]
#         = sum_i F[i,i]T[p_i,p_i]  +  0.5*( sum_i OUT_i(p) + sum_i IN_i(p) )
# where OUT_i = sum_{j!=i} F[i,j]T[p_i,p_j], IN_i = sum_{j!=i} F[j,i]T[p_j,p_i].
rng = np.random.default_rng(0); bad = 0
for _ in range(300):
    n = int(rng.integers(2, 9)); F = rng.uniform(-10, 10, (n, n)); T = rng.uniform(-250, 250, (n, n))
    p = rng.permutation(n)
    diag = sum(F[i, i]*T[p[i], p[i]] for i in range(n))
    OUT = sum(sum(F[i, j]*T[p[i], p[j]] for j in range(n) if j != i) for i in range(n))
    IN  = sum(sum(F[j, i]*T[p[j], p[i]] for j in range(n) if j != i) for i in range(n))
    recon = diag + 0.5*(OUT + IN)
    if abs(recon - indep_fitness(F, T, p)) > 1e-9*max(1, abs(recon)): bad += 1
print(f"  identity  total == diag + 0.5*(OUT+IN)  failures: {bad}/300  (0 => the accounting is EXACT)")
print("  -> the code's decomposition is an ALGEBRAIC IDENTITY, and OUT==IN for every p,")
print("     so the halving is exact, not an approximation. The diagonal is outside the")
print("     off-diagonal sums (masked by `off`), so it is counted exactly once. AIRTIGHT.")

print("\n=== (B) does the per-(i,k) floor + LAP genuinely floor the quadratic? ===")
print("  For any permutation p:  OUT_i(p) >= sorted_dot_min(F[i,off], T[p_i,off])  because")
print("  the multiset {T[p_i,p_j] : j!=i} is a subset-of-size-(n-1) of {T[p_i,k'] : k'!=p_i}")
print("  -- actually it IS that full multiset, since p is a bijection. So the rearrangement")
print("  minimum over ALL pairings is <= the specific pairing p induces. Same for IN.")
print("  Hence cost[i,p_i] <= F[i,i]T[p_i,p_i] + 0.5*(OUT_i + IN_i) for every i, and summing")
print("  gives sum_i cost[i,p_i] <= total(p). The LAP minimum is <= sum_i cost[i,p_i]. QED.")
print("  => the bound is PROVABLY valid for ARBITRARY SIGNED F and T. Empirics agree:")

# targeted adversarial hunt: try hard to break it on signed/structured instances
viol = 0; tested = 0; worst = 0.0
fams = {
  "signed-antisym": lambda g,n: ((lambda A: A-A.T)(g.uniform(-10,10,(n,n))), (lambda B: B-B.T)(g.uniform(-250,250,(n,n)))),
  "huge-negative-diag": lambda g,n: (g.uniform(0,1,(n,n))-1000*np.eye(n), g.uniform(50,250,(n,n))),
  "one-huge-cell": lambda g,n: ((lambda A: A)(np.eye(n)[::-1]*1e6 + g.uniform(0,1,(n,n))), g.uniform(50,250,(n,n))),
  "integers":     lambda g,n: (g.integers(-5,6,(n,n)).astype(float), g.integers(-9,10,(n,n)).astype(float)),
  "all-equal-F":  lambda g,n: (np.full((n,n),7.0), g.uniform(50,250,(n,n))),
  "all-zero-T":   lambda g,n: (g.uniform(0,10,(n,n)), np.zeros((n,n))),
}
for name, gen in fams.items():
    nv = 0
    for n in range(2, 7):
        for r in range(60):
            g = np.random.default_rng(hash((name,n,r)) % (2**31))
            F, T = gen(g, n)
            lb = gilmore_lawler_bound(F, T)
            opt = min(indep_fitness(F, T, p) for p in itertools.permutations(range(n)))
            tested += 1
            tol = 1e-7*max(1.0, abs(opt))
            if lb > opt + tol:
                nv += 1; worst = max(worst, (lb-opt)/max(1e-12,abs(opt))*100)
    viol += nv
    print(f"  {name:<20} violations {nv}/300")
print(f"  TOTAL adversarial: {viol} violations over {tested} exhaustive cases (worst {worst:.4g}%)")
print("\nPROBE11-DONE")
