"""Calibrate the sequential test's alpha-spending boundary BEFORE any new seed is trained.

Design choice to justify: peeking at every n inflates alpha. Options:
 (A) fixed n, one test  -- no inflation, but n is guessed
 (B) Pocock/OBF group-sequential -- needs a pre-fixed max n and looks at fixed information
     fractions; our "information" is n_seeds which we control, so this fits.
 (C) simulate the ACTUAL rule under H0 and report the realized type-I rate (an empirical
     alpha-spending calibration). Exact for our rule; no asymptotics needed.

We take (C) as primary and cross-check the boundary against (B)'s O'Brien-Fleming shape.
H0 simulation: per-seed margins ~ N(0, sd) i.i.d. -- the null for a paired t on seed
margins. Because the t-statistic is scale-invariant, the realized alpha does NOT depend on
sd, so one simulation calibrates all 10 pairs.
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260802)
NSIM = 200_000
N_MIN, N_MAX = 4, 15          # start peeking at n=4 (first new seed), cap at 15

def realized_alpha(crit_p, n_min=N_MIN, n_max=N_MAX, nsim=NSIM, need_consecutive=1):
    """Fraction of H0 datasets where the rule EVER fires between n_min..n_max."""
    z = RNG.standard_normal((nsim, n_max))
    fired = np.zeros(nsim, dtype=bool)
    run = np.zeros(nsim, dtype=int)
    for n in range(n_min, n_max+1):
        x = z[:, :n]
        m = x.mean(1); s = x.std(1, ddof=1)
        t = m / (s/np.sqrt(n))
        p = 2*stats.t.sf(np.abs(t), df=n-1)
        hit = p < crit_p
        run = np.where(hit, run+1, 0)
        fired |= (run >= need_consecutive)
    return fired.mean()

print(f"H0 sim: {NSIM} datasets, peek at every n in [{N_MIN},{N_MAX}]")
print(f"{'nominal p':>10} {'realized alpha (1 peek-hit)':>28} {'realized (2 consecutive)':>26}")
for crit in (0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005):
    a1 = realized_alpha(crit)
    a2 = realized_alpha(crit, need_consecutive=2)
    print(f"{crit:>10.4f} {a1:>28.4f} {a2:>26.4f}")

# The uncorrected naive rate, for the record
print(f"\nNAIVE p<0.05 at every peek -> realized alpha = {realized_alpha(0.05):.4f} "
      f"(vs nominal 0.05) -- this is the inflation the correction must remove")

# Single-look reference (no peeking) at each n
print("\nSingle-look-only alpha (sanity: should be ~0.05):")
for n in (4, 9, 15):
    print(f"  n={n}: {realized_alpha(0.05, n_min=n, n_max=n):.4f}")

# --- Power: given a TRUE margin of delta with per-seed sd, what n resolves? ------------
print("\n=== POWER at the calibrated boundary (whatever we pick) vs true effect size ===")
def power(crit_p, d_over_sd, n_min=N_MIN, n_max=N_MAX, nsim=40_000, need_consecutive=1):
    z = RNG.standard_normal((nsim, n_max)) + d_over_sd
    fired = np.zeros(nsim, dtype=bool); run=np.zeros(nsim,dtype=int)
    first = np.full(nsim, -1)
    for n in range(n_min, n_max+1):
        x = z[:, :n]; m=x.mean(1); s=x.std(1,ddof=1)
        t = m/(s/np.sqrt(n)); p = 2*stats.t.sf(np.abs(t), df=n-1)
        hit = p < crit_p; run = np.where(hit, run+1, 0)
        nowfired = (run>=need_consecutive) & ~fired
        first[nowfired]=n; fired |= (run>=need_consecutive)
    return fired.mean(), (first[first>0].mean() if (first>0).any() else np.nan)

# candidate-vs-arm-B observed: mean 0.0991, sd 0.1235 -> d/sd = 0.80
for crit in (0.05, 0.015, 0.01):
    print(f" nominal p<{crit}:")
    for dsd in (0.80, 1.0, 1.5, 2.0, 3.0):
        pw, en = power(crit, dsd)
        print(f"   d/sd={dsd:<4} power(by n=15)={pw:.3f}  mean n at fire={en:.1f}")
