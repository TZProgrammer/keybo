"""FIND-phase: the weight-bearing structure at FIT level (what the optimizer actually sees).

Three questions, all on the SHIPPED .standardized frame, blend-v1:
 Q1 how redundant is POOL given AALTO+COMMUNITY, at fit level (not cell level)?
 Q2 what is COMMUNITY's own seed-to-seed fit reliability, as a fraction of the range it
    must discriminate over? (per-seed arrays exist for the BASE family ONLY — labelled)
 Q3 over a RANDOM pool vs the CANDIDATE field, how correlated are the three fits?
"""
import numpy as np
from pathlib import Path
from keybo.testkit import assert_module_under
assert_module_under("keybo", "/tmp/normgauge")
from keybo.analysis import surfaces as S

NAT = Path("/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
           "old-new-layout-comparison/tri_frequency_old_new_surfaces")
FAM = "TRI_PS_FREQ_PRIOR"
i, j, k, f = S.trigram_objective(S.default_trigram_path())
i, j, k = i.astype(np.int64), j.astype(np.int64), k.astype(np.int64)

def hist(perm):
    return np.bincount(perm[i] * 961 + perm[j] * 31 + perm[k], weights=f, minlength=29791)

def fits(perms, flat):                     # (n,29791) @ (29791,m)
    H = np.stack([hist(p) for p in perms])
    return H @ flat

CAND = {  # the shipped registry + campaign arms that are C30M-scorable
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "arm-A": "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}
flat3 = np.ascontiguousarray(np.stack([S.load_surface(f"{p}_{FAM}").ravel() for p in S.POOLS]).T)
rng = np.random.default_rng(20260728)
rand = [np.concatenate([rng.permutation(30), [30]]) for _ in range(400)]
Frand = fits(rand, flat3)
Fcand = fits([S.layout_permutation(v) for v in CAND.values()], flat3)

def report(F, label):
    print(f"\n--- {label} (n={len(F)}) ---")
    print("  fit sd (ms):   " + "  ".join(f"{p}={F[:, n].std(ddof=1):.4e}" for n, p in enumerate(S.POOLS)))
    C = np.corrcoef(F.T)
    print("  fit corr:\n" + "\n".join("   " + " ".join(f"{v:+.6f}" for v in row) for row in np.round(C, 6)))
    A, Cc, P = F[:, 0], F[:, 1], F[:, 2]
    X = np.column_stack([np.ones_like(A), A, Cc])
    beta, *_ = np.linalg.lstsq(X, P, rcond=None)
    r = P - X @ beta
    print(f"  POOL = {beta[0]:+.4e} + {beta[1]:+.6f}*AALTO + {beta[2]:+.6f}*COMMUNITY"
          f"   R2={1 - (r**2).sum()/((P-P.mean())**2).sum():.8f}")
    print(f"  resid sd = {r.std(ddof=1):.4e} ms = {100*r.std(ddof=1)/P.std(ddof=1):.4f}% of POOL's own fit sd")
    print(f"  coef sum {beta[1]+beta[2]:.6f}   (0.5/0.5 => POOL is a SYMMETRIC blend: no A-vs-C tilt)")
    return beta

report(Frand, "RANDOM pool")
report(Fcand, "CANDIDATE field")

print("\n=== Q2 COMMUNITY seed-to-seed fit reliability (BASE family — the ONLY one with per-seed arrays) ===")
bi = {s: np.load(NAT / f"COMMUNITY_BASE.bigram.seed{s}.npy") for s in (0, 1, 2)}
cd = {s: np.load(NAT / f"COMMUNITY_BASE.conditional.seed{s}.npy") for s in (0, 1, 2)}
per_seed = np.ascontiguousarray(np.stack([(bi[s][:, :, None] + cd[s]).ravel() for s in (0, 1, 2)]).T)
base_nat = np.load(NAT / "COMMUNITY_BASE.native.npy")
print("  control: mean(per-seed) reconstructs COMMUNITY_BASE.native, max|d| =",
      f"{np.abs(np.mean([bi[s][:, :, None] + cd[s] for s in (0, 1, 2)], axis=0) - base_nat).max():.3e}")
for label, perms in (("RANDOM pool", rand), ("CANDIDATE field", [S.layout_permutation(v) for v in CAND.values()])):
    Fs = fits(perms, per_seed)                        # (n, 3 seeds)
    seed_sd = Fs.std(axis=1, ddof=1)                  # per-layout across-seed sd
    across = Fs.mean(axis=1)                          # the seed-mean fit
    print(f"  {label:16s} mean across-seed sd = {seed_sd.mean():.4e} ms | "
          f"layout spread (sd of seed-mean) = {across.std(ddof=1):.4e} ms | "
          f"noise/signal = {seed_sd.mean()/across.std(ddof=1):.4f}")
