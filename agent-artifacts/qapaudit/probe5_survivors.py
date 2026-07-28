"""FIND-pass probe 5: are the 6 mutation SURVIVORS real defect classes or harmless
equivalences? A survivor only matters if the mutant can produce a WRONG answer.

For each survivor I implement the MUTANT bound inline and search for a violation
(mutant_bound > true_optimum) by exhaustive brute force over small instances.
The true optimum uses indep_fitness (hand-written, no shipped code).
"""
import itertools, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from keybo.optimize.qap_bound import certificate, gilmore_lawler_bound

def indep_fitness(F, T, perm):
    n = len(perm); tot = 0.0
    for i in range(n):
        for j in range(n):
            tot += float(F[i][j]) * float(T[perm[i]][perm[j]])
    return tot

def indep_brute(F, T):
    n = F.shape[0]
    return min(indep_fitness(F, T, p) for p in itertools.permutations(range(n)))

def _sdm(f, t):
    return float(np.sort(f)[::-1] @ np.sort(t))

def bound_variant(F, T, variant):
    """Reimplementation of gilmore_lawler_bound with one mutation switched in.
    POSITIVE CONTROL: variant='shipped' must match the shipped fn bit-for-bit."""
    n = F.shape[0]
    off = ~np.eye(n, dtype=bool)
    cost = np.empty((n, n))
    for i in range(n):
        f_out = F[i][off[i]]
        f_in  = F[i][off[i]] if variant == "f_in_row" else F[:, i][off[:, i]]
        for k in range(n):
            t_out = T[k][off[k]]
            t_in  = T[k][off[k]] if variant == "t_in_row" else T[:, k][off[:, k]]
            if variant == "out_twice":
                pair = _sdm(f_out, t_out) + _sdm(f_out, t_out)
            else:
                pair = _sdm(f_out, t_out) + _sdm(f_in, t_in)
            cost[i, k] = F[i, i] * T[k, k] + 0.5 * pair
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum())

# ---- POSITIVE CONTROL FIRST (trap: never use a control after using its result) --------
rng = np.random.default_rng(11)
mx = 0.0
for _ in range(40):
    n = int(rng.integers(2, 8)); F = rng.uniform(0, 10, (n, n)); T = rng.uniform(50, 250, (n, n))
    a, b = bound_variant(F, T, "shipped"), gilmore_lawler_bound(F, T)
    mx = max(mx, abs(a - b) / max(1.0, abs(b)))
print(f"[control] my bound_variant('shipped') vs SHIPPED gilmore_lawler_bound: max rel diff {mx:.3e}")
assert mx < 1e-12, "my reimplementation does not reproduce the shipped bound -> results below are void"
print("[control] PASSED -> the mutant variants below are trustworthy reimplementations\n")

# ---- hunt for violations per survivor variant ----------------------------------------
out = {}
for variant in ("out_twice", "t_in_row", "f_in_row"):
    viol, worst_ratio, n_cases = [], 0.0, 0
    for n in range(2, 7):
        for r in range(150):
            g = np.random.default_rng(hash((variant, n, r)) % (2**31))
            # asymmetric, nonneg — exactly the keybo shape
            F = g.uniform(0, 10, (n, n)); T = g.uniform(50, 250, (n, n))
            lb = bound_variant(F, T, variant); opt = indep_brute(F, T)
            n_cases += 1
            if lb > opt + 1e-7 * abs(opt):
                viol.append((n, r, lb, opt, (lb - opt) / opt * 100))
                worst_ratio = max(worst_ratio, (lb - opt) / opt * 100)
    out[variant] = dict(n_cases=n_cases, n_viol=len(viol),
                        worst_overshoot_pct=worst_ratio, examples=viol[:3])
    print(f"{variant:<12} {n_cases} cases | VIOLATIONS {len(viol)} "
          f"| worst overshoot {worst_ratio:.3f}%")
    for v in viol[:2]:
        print(f"     n={v[0]} seed={v[1]}: mutant bound {v[2]:.4f} > true opt {v[3]:.4f}  (+{v[4]:.3f}%)")

# ---- and on a SYMMETRIC T (where the transposition is a no-op by construction) -------
print("\n[sanity] on SYMMETRIC T the t_in/f_in transpositions must be exact no-ops:")
g = np.random.default_rng(99); n = 6
Fs = g.uniform(0, 10, (n, n)); Ts = g.uniform(50, 250, (n, n)); Ts = Ts + Ts.T
print(f"   shipped {bound_variant(Fs, Ts, 'shipped'):.6f}  t_in_row {bound_variant(Fs, Ts, 't_in_row'):.6f}")

# ---- certificate GUARD probes: does it silently return a nonsense gap? ----------------
print("\n=== certificate() guard probes ===")
g = np.random.default_rng(5); n = 6
F = g.uniform(0, 10, (n, n)); T = g.uniform(50, 250, (n, n))
lb = gilmore_lawler_bound(F, T)
guards = {}
# 1. found_fitness BELOW the bound (impossible for a real layout => signals a bug)
c = certificate(F, T, found_fitness=lb * 0.5)
guards["found_below_bound"] = c["gap_pct"]
print(f" found = 0.5*lb  -> gap_pct {c['gap_pct']:+.2f}%   statement: {c['statement'][:70]}")
# 2. found_fitness = nan
c = certificate(F, T, found_fitness=float("nan"))
guards["found_nan"] = c["gap_pct"]
print(f" found = nan     -> gap_pct {c['gap_pct']}   statement: {c['statement'][:70]}")
# 3. found_fitness = inf
c = certificate(F, T, found_fitness=float("inf"))
guards["found_inf"] = c["gap_pct"]
print(f" found = inf     -> gap_pct {c['gap_pct']}")
# 4. found_fitness = 0
c = certificate(F, T, found_fitness=0.0)
guards["found_zero"] = c["gap_pct"]
print(f" found = 0.0     -> gap_pct {c['gap_pct']:+.2f}%")
# 5. all-zero F (lb == 0) -> the lb>0 branch
Z = np.zeros((n, n))
c = certificate(Z, T, found_fitness=0.0)
guards["allzero_F"] = c["gap_pct"]
print(f" F all-zero      -> lb {c['lower_bound']}  gap_pct {c['gap_pct']}  (inf branch)")
# 6. does certificate validate F/T finiteness at all?
Fn = F.copy(); Fn[0, 0] = np.nan
try:
    c = certificate(Fn, T, found_fitness=1e6)
    guards["F_has_nan"] = str(c["gap_pct"]); print(f" F has a NaN     -> NO RAISE, gap_pct {c['gap_pct']}  lb {c['lower_bound']}")
except Exception as e:
    guards["F_has_nan"] = f"raised {type(e).__name__}"; print(f" F has a NaN     -> raised {type(e).__name__}")
# 7. non-square / mismatched shape guard (the ONE guard that exists)
try:
    gilmore_lawler_bound(np.zeros((3, 3)), np.zeros((4, 4)))
    guards["shape_mismatch"] = "NO RAISE"; print(" shape mismatch  -> NO RAISE (guard failed)")
except ValueError:
    guards["shape_mismatch"] = "ValueError"; print(" shape mismatch  -> ValueError (guard works)")

json.dump({"survivors": out, "cert_guards": guards},
          open("/tmp/qapaudit/agent-artifacts/qapaudit/probe5.json", "w"), indent=2, default=str)
print("\nPROBE5-DONE")
