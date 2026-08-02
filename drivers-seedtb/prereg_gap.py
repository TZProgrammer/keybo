"""DISCLOSURE: my prereg's gray zone had a crack. Quantify the alpha cost of both readings.

THE GAP: I defined resolution on the HOLM-ADJUSTED p but the gray zone on the RAW p in
[0.05,0.20). A pair whose raw p lands in (holm_thr, 0.05) is therefore neither resolved nor
extended -- it falls through both branches. At stage 1, `arm-B vs F(2.5)` was exactly there
(raw p=0.0377, Holm thr 0.0167). The rule AS WRITTEN does not license extending it, yet the
seeds got trained anyway (the gray-zone pair needed them), so the n=25 number exists.

Two readings, both reported:
  STRICT   -- only `arm-B vs candidate` was licensed for stage 2. arm-B vs F(2.5) stays
              "not resolved at n=15" and its n=25 p is EXPLORATORY.
  AMENDED  -- gray zone = "not Holm-resolved AND raw p < 0.20" (what I should have written).
              This licenses both. Simulate ITS realized type-I so the cost is disclosed.
"""
import sys
MY_WT="/local/home/zegertho/agent/workspaces/seedtb/wt"; sys.path.insert(0,MY_WT+"/src")
import keybo; assert keybo.__file__.startswith(MY_WT)
import numpy as np
from scipy import stats
RNG=np.random.default_rng(80220261)
NSIM=120_000; N1,N2=15,25

def sim(rule, nsim=NSIM, nboards=5, alpha=0.05, alpha2=0.02):
    """Global-null sim over 5 boards -> 4 primary pairs, Holm within family, two-stage."""
    e=RNG.standard_normal((nsim,nboards,N2))
    pri=[(0,j) for j in range(1,nboards)]
    def pvals(n):
        out=[]
        for i,j in pri:
            d=e[:,i,:n]-e[:,j,:n]
            t=d.mean(1)/(d.std(1,ddof=1)/np.sqrt(n))
            out.append(2*stats.t.sf(np.abs(t),df=n-1))
        return np.column_stack(out)
    P1=pvals(N1); P2=pvals(N2)
    def holm_mask(P,a):
        m=P.shape[1]; order=np.argsort(P,1); Ps=np.take_along_axis(P,order,1)
        thr=a/(m-np.arange(m)); ok=Ps<thr
        # step-down: stop at first failure
        keep=np.cumprod(ok,axis=1).astype(bool)
        rej=np.zeros_like(P,dtype=bool)
        np.put_along_axis(rej,order,keep,1)
        return rej
    R1=holm_mask(P1,alpha)
    if rule=="strict":
        gray=(~R1)&(P1>=0.05)&(P1<0.20)
    else:
        gray=(~R1)&(P1<0.20)
    R2=gray&(P2<alpha2)
    anyrej=(R1|R2).any(1)
    perpair=(R1|R2).mean(0)
    return anyrej.mean(), perpair.mean(), gray.any(1).mean()

for rule in ("strict","amended"):
    fwe,pp,ex=sim(rule)
    print(f"{rule:>8}: FWE over 4 primary pairs = {fwe:.4f}   mean per-pair = {pp:.4f}   "
          f"P(any extension|H0) = {ex:.4f}")
print("\n(target: FWE <= ~0.05. Holm at n=15 alone gives FWE ~0.047 per the prereg sim.)")

# What alpha2 keeps the AMENDED rule at FWE<=0.05?
print("\nAMENDED rule, FWE vs alpha2:")
for a2 in (0.05,0.03,0.02,0.01,0.005):
    fwe,_,_=sim("amended",alpha2=a2)
    print(f"  alpha2={a2:<6} FWE={fwe:.4f}")
