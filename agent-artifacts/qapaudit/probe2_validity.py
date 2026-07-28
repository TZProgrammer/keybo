"""FIND-pass probe 2: is the SHIPPED GL bound a genuine lower bound?

INDEPENDENCE: the brute-force optimum here is computed with `indep_fitness`, a
hand-written triple loop that shares NO code with `keybo.optimize.qap_bound`. The
shipped test's `brute_force_min` calls the shipped `qap_fitness` (a shared component,
per this campaign's independence rule); this probe does not.
"""
import itertools, json
import numpy as np
from keybo.optimize.qap_bound import gilmore_lawler_bound, qap_fitness, certificate

def indep_fitness(F, T, perm):
    """Hand-written objective. No numpy fancy indexing, no shipped code."""
    n = len(perm)
    tot = 0.0
    for i in range(n):
        for j in range(n):
            tot += float(F[i][j]) * float(T[perm[i]][perm[j]])
    return tot

def indep_brute(F, T):
    n = F.shape[0]
    return min(indep_fitness(F, T, p) for p in itertools.permutations(range(n)))

# ---- 0. positive-control MY fitness against the shipped one (trap 28 / independence) --
rng = np.random.default_rng(1)
mx = 0.0
for _ in range(50):
    n = int(rng.integers(2, 8)); F = rng.uniform(-3, 10, (n, n)); T = rng.uniform(-5, 250, (n, n))
    p = rng.permutation(n)
    mx = max(mx, abs(indep_fitness(F, T, p) - qap_fitness(F, T, p)) / max(1.0, abs(qap_fitness(F, T, p))))
print(f"[control] max rel disagreement indep_fitness vs shipped qap_fitness: {mx:.3e}")
assert mx < 1e-12, "my independent fitness disagrees with the shipped one -> stop"

# ---- 1. closed-form n=2 external anchor -----------------------------------------------
# n=2: perms are (0,1) and (1,0). Hand-derived by algebra, no loops.
F = np.array([[1.0, 5.0], [2.0, 3.0]]); T = np.array([[10.0, 40.0], [70.0, 20.0]])
id_val  = 1*10 + 5*40 + 2*70 + 3*20      # p=(0,1)
swp_val = 1*20 + 5*70 + 2*40 + 3*10      # p=(1,0)
lb2 = gilmore_lawler_bound(F, T)
print(f"[n=2 closed form] id {id_val}  swap {swp_val}  true min {min(id_val, swp_val)}  GL lb {lb2:.6f}")
assert lb2 <= min(id_val, swp_val) + 1e-9, "n=2 CLOSED-FORM VIOLATION"

# ---- 2. exhaustive validity sweep, much wider than the shipped test --------------------
cases, viol = [], []
def sweep(name, gen, n_lo, n_hi, reps):
    for n in range(n_lo, n_hi + 1):
        for r in range(reps):
            F, T = gen(np.random.default_rng(hash((name, n, r)) % (2**31)), n)
            lb = gilmore_lawler_bound(F, T)
            opt = indep_brute(F, T)
            ok = lb <= opt + 1e-7 * max(1.0, abs(opt))
            tight = (opt - lb) / abs(opt) * 100 if opt != 0 else float("nan")
            cases.append((name, n, r, float(lb), float(opt), bool(ok), float(tight)))
            if not ok:
                viol.append((name, n, r, float(lb), float(opt)))

G = {
 # the shipped test's own generator, but over n=2..7 and 12 reps (it does n=6, 5 reps)
 "shipped-like":  lambda g, n: (g.uniform(0, 10, (n, n)), g.uniform(50, 250, (n, n))),
 # ASYMMETRIC extremes / structure the shipped test never generates:
 "F-has-negatives": lambda g, n: (g.uniform(-10, 10, (n, n)), g.uniform(50, 250, (n, n))),
 "T-has-negatives": lambda g, n: (g.uniform(0, 10, (n, n)), g.uniform(-250, 250, (n, n))),
 "both-negative":   lambda g, n: (g.uniform(-10, 10, (n, n)), g.uniform(-250, 250, (n, n))),
 "F-sparse":        lambda g, n: (g.uniform(0, 10, (n, n)) * (g.random((n, n)) < 0.25),
                                  g.uniform(50, 250, (n, n))),
 "F-zero-diag":     lambda g, n: (g.uniform(0, 10, (n, n)) * (1 - np.eye(n)),
                                  g.uniform(50, 250, (n, n))),
 "T-zero-diag":     lambda g, n: (g.uniform(0, 10, (n, n)),
                                  g.uniform(50, 250, (n, n)) * (1 - np.eye(n))),
 "F-symmetric":     lambda g, n: ((lambda A: A + A.T)(g.uniform(0, 10, (n, n))),
                                  g.uniform(50, 250, (n, n))),
 "both-symmetric":  lambda g, n: ((lambda A: A + A.T)(g.uniform(0, 10, (n, n))),
                                  (lambda B: B + B.T)(g.uniform(50, 250, (n, n)))),
 "huge-diag-F":     lambda g, n: (g.uniform(0, 1, (n, n)) + 500 * np.eye(n),
                                  g.uniform(50, 250, (n, n))),
 "keybo-like":      lambda g, n: (g.uniform(0, 2.6e7, (n, n)), g.uniform(114, 249, (n, n))),
}
for name, gen in G.items():
    sweep(name, gen, 2, 7, 8)
print(f"\n[sweep] {len(cases)} exhaustive cases, n=2..7, 11 structure families")
print(f"[sweep] LOWER-BOUND VIOLATIONS: {len(viol)}")
for v in viol[:20]:
    print("   VIOL", v)

# tightness summary per family
print("\n[tightness] (opt-lb)/|opt| %, by family (n=7 only):")
for name in G:
    ts = [c[6] for c in cases if c[0] == name and c[1] == 7]
    print(f"   {name:<18} median {np.median(ts):8.2f}%  max {max(ts):9.2f}%")

json.dump({"violations": viol, "n_cases": len(cases), "cases": cases},
          open("/tmp/qapaudit/agent-artifacts/qapaudit/probe2.json", "w"), indent=2)
print("\nPROBE2-DONE")
