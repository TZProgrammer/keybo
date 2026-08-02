"""Pre-registration support: calibrate the EXACT design, incl. multiplicity across pairs."""
import numpy as np
from scipy import stats
RNG = np.random.default_rng(31415926)
NSIM = 200_000

def two_stage(crit1, crit2, n1=15, n2=25, gray=(0.05,0.20), nsim=NSIM):
    """Realized type-I of: test at n1; if p in [gray), extend to n2 and re-test."""
    z = RNG.standard_normal((nsim, n2))
    def pval(n):
        x=z[:, :n]; t=x.mean(1)/(x.std(1,ddof=1)/np.sqrt(n)); return 2*stats.t.sf(np.abs(t), df=n-1)
    p1 = pval(n1); fire1 = p1 < crit1
    extend = (~fire1) & (p1 >= gray[0]) & (p1 < gray[1])
    p2 = pval(n2); fire2 = extend & (p2 < crit2)
    return (fire1|fire2).mean(), fire1.mean(), extend.mean()

print("=== TWO-STAGE design: look at n=15, extend to n=25 only if p in [0.05,0.20) ===")
print(f"{'crit1':>8} {'crit2':>8} {'realized alpha':>15} {'fire@n15':>10} {'P(extend|H0)':>13}")
for c1,c2 in ((0.05,0.05),(0.05,0.03),(0.05,0.02),(0.05,0.01),(0.045,0.02),(0.04,0.03)):
    a,f1,ex = two_stage(c1,c2)
    print(f"{c1:>8.3f} {c2:>8.3f} {a:>15.4f} {f1:>10.4f} {ex:>13.4f}")

print("\n=== FAMILY-WISE across the 4 arm-B pairs (correlated: share arm-B) ===")
def fwe(crit, npair_boards=5, n=15, nsim=60_000, holm=False, primary_only=True):
    """5 boards, independent per-seed noise -> 10 correlated pairwise margins."""
    e = RNG.standard_normal((nsim, npair_boards, n))
    pairs = [(0,j) for j in range(1,5)] if primary_only else [(i,j) for i in range(5) for j in range(i+1,5)]
    ps=[]
    for i,j in pairs:
        d = e[:,i,:]-e[:,j,:]
        t = d.mean(1)/(d.std(1,ddof=1)/np.sqrt(n))
        ps.append(2*stats.t.sf(np.abs(t), df=n-1))
    P = np.column_stack(ps)
    if not holm:
        return (P < crit).any(1).mean()
    m = P.shape[1]; order=np.argsort(P,1); Ps=np.take_along_axis(P,order,1)
    thr = crit/(m-np.arange(m))
    return (Ps<thr).any(1).mean()   # any rejection = FWE event under global null

for crit in (0.05, 0.025, 0.0125):
    print(f" uncorrected per-pair p<{crit}: FWE over 4 arm-B pairs = {fwe(crit):.4f}")
print(f" HOLM at 0.05 over 4 arm-B pairs: FWE = {fwe(0.05, holm=True):.4f}")
print(f" HOLM at 0.05 over all 10 pairs : FWE = {fwe(0.05, holm=True, primary_only=False):.4f}")

print("\n=== POWER of the two-stage design (crit1=0.05, crit2=0.02) vs d/sd ===")
def power2(d, crit1=0.05, crit2=0.02, n1=15, n2=25, gray=(0.05,0.20), nsim=40_000):
    z = RNG.standard_normal((nsim,n2)) + d
    def pv(n):
        x=z[:,:n]; t=x.mean(1)/(x.std(1,ddof=1)/np.sqrt(n)); return 2*stats.t.sf(np.abs(t),df=n-1)
    p1=pv(n1); f1=p1<crit1
    ex=(~f1)&(p1>=gray[0])&(p1<gray[1]); f2=ex&(pv(n2)<crit2)
    return (f1|f2).mean(), f1.mean()
for d in (0.5, 0.6, 0.8, 1.0, 1.2):
    tot,at15 = power2(d)
    print(f"  d/sd={d:<4}: power@n15={at15:.3f}  total(two-stage)={tot:.3f}")

print("\n=== n needed for 80%/90% power, single look, two-sided 0.05 ===")
for d in (0.4,0.5,0.6,0.8,1.0):
    ns=[]
    for target in (0.80,0.90):
        for n in range(3,400):
            nc = stats.t.ppf(1-0.025, n-1)
            # noncentral t power
            pw = stats.nct.sf(nc, n-1, d*np.sqrt(n)) + stats.nct.cdf(-nc, n-1, d*np.sqrt(n))
            if pw>=target: ns.append(n); break
        else: ns.append(None)
    print(f"  d/sd={d:<4}: n80={ns[0]}  n90={ns[1]}")
