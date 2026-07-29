"""FIND-phase probe: what IS the .standardized frame, and how independent are the three?"""
import numpy as np
from pathlib import Path
from keybo.testkit import assert_module_under
assert_module_under('keybo', '/tmp/normgauge')
from keybo.analysis import surfaces as S

NAT = Path("/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
           "old-new-layout-comparison/tri_frequency_old_new_surfaces")
FAM = "TRI_PS_FREQ_PRIOR"
std = {p: S.load_surface(f"{p}_{FAM}") for p in S.POOLS}
nat = {p: np.load(NAT / f"{p}_{FAM}.native.npy") for p in S.POOLS}

print("=== (1) structure of d_m = std - nat  (c-independent, so a bigram-level term) ===")
for p in S.POOLS:
    d = std[p] - nat[p]
    d2 = d[:, :, 0]                      # the (a,b) bigram-level shift
    print(f"{p:10s} mean={d2.mean():+.6f} sd={d2.std():.6f} min={d2.min():+.4f} max={d2.max():+.4f}")
    # is it rank-1 / separable / constant per row?
    u, s, vt = np.linalg.svd(d2)
    print(f"{'':10s} svd top5 {np.round(s[:5], 4)}  (rank1 frac {s[0]**2/ (s**2).sum():.6f})")
    print(f"{'':10s} row-constant? row-sd of row-means {d2.mean(axis=1).std():.6f}; "
          f"mean within-row sd {d2.std(axis=1).mean():.6f}")

print()
print("=== (2) is std[m] just nat[m] shifted to a common overall level? ===")
for p in S.POOLS:
    print(f"{p:10s} nat.mean={nat[p].mean():.6f} std.mean={std[p].mean():.6f} "
          f"d.mean={(std[p]-nat[p]).mean():+.6f}")

print()
print("=== (3) cell-level linear structure on the SHIPPED frame: POOL ~ AALTO + COMMUNITY ===")
A = std["AALTO"].ravel(); C = std["COMMUNITY"].ravel(); P = std["POOL"].ravel()
X = np.column_stack([np.ones_like(A), A, C])
beta, *_ = np.linalg.lstsq(X, P, rcond=None)
pred = X @ beta
ss_res = ((P - pred) ** 2).sum(); ss_tot = ((P - P.mean()) ** 2).sum()
print(f"POOL = {beta[0]:+.6f} + {beta[1]:+.6f}*AALTO + {beta[2]:+.6f}*COMMUNITY   "
      f"R2={1 - ss_res/ss_tot:.8f}  resid sd={np.sqrt(ss_res/len(P)):.6f} ms")
print(f"coef sum (convexity check) = {beta[1]+beta[2]:.6f}")
print(f"cell sd: AALTO {A.std():.4f}  COMMUNITY {C.std():.4f}  POOL {P.std():.4f}")
print("cell corr matrix:")
print(np.round(np.corrcoef(np.stack([A, C, P])), 6))
