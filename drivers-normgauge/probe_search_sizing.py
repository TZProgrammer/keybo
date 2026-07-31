"""SIZING ONLY (not a result): how strong a search do I need to hit a KNOWN optimum?

Free target: MODELNORM's AALTO champion is the 10M-unique-eval memetic-island optimum on
EXACTLY my frame (AALTO .native == .standardized, bit-exact), fit = 223236317224.4177.
So AALTO gives me a calibrated yardstick for search strength before I spend any budget.
"""
import time
import numpy as np
from keybo.analysis import surfaces as S

i, j, k, f = S.trigram_objective(S.default_trigram_path())
i, j, k = i.astype(np.int64), j.astype(np.int64), k.astype(np.int64)
flat = np.ascontiguousarray(S.load_surface("AALTO_TRI_PS_FREQ_PRIOR").ravel()[:, None])
TARGET = 223236317224.4177

def fit(perm):
    return float((np.bincount(perm[i] * 961 + perm[j] * 31 + perm[k], weights=f, minlength=29791) @ flat)[0])

def batch_fits(perms):
    H = np.stack([np.bincount(p[i] * 961 + p[j] * 31 + p[k], weights=f, minlength=29791) for p in perms])
    return (H @ flat).ravel()

rng = np.random.default_rng(1)
p = np.concatenate([rng.permutation(30), [30]])
t = time.perf_counter()
# one full 435-neighbour sweep
pairs = [(a, b) for a in range(30) for b in range(a + 1, 30)]
cands = []
for a, b in pairs:
    q = p.copy(); q[[a, b]] = q[[b, a]]; cands.append(q)
v = batch_fits(cands)
dt = time.perf_counter() - t
print(f"one 435-neighbour sweep: {dt*1000:.1f} ms -> {435/dt:.0f} evals/s")
print(f"a 10M-eval budget = {10e6/ (435/dt) / 60:.1f} min single-threaded")
# greedy steepest-descent from random: how many sweeps to converge, and how close?
best = fit(p); steps = 0
t = time.perf_counter()
while True:
    cands = []
    for a, b in pairs:
        q = p.copy(); q[[a, b]] = q[[b, a]]; cands.append(q)
    v = batch_fits(cands)
    m = int(v.argmin())
    if v[m] >= best - 1e-6:
        break
    best = float(v[m]); p = cands[m]; steps += 1
print(f"greedy: {steps} sweeps, {steps*435:,} evals, {time.perf_counter()-t:.1f}s")
print(f"  best={best:.4f}  target={TARGET:.4f}  gap={100*(best-TARGET)/TARGET:+.4f}%")
