"""Trap 52: validate the identification structure IN THE BAND WHERE THE SCORER IS USED.

Random permutations are NOT the operating band. Build a NEAR-OPTIMAL pool (the 15 registry
layouts plus local neighbourhoods of them), re-estimate, and compare. Trap 26 forbids
pooling optimized+random into ONE dof estimate, so the two pools are reported SEPARATELY.
"""
import json, sys, random, numpy as np, importlib.util
spec=importlib.util.spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')
# collin3 runs its control on import; suppress its pool block by importing the fn only
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
ctl=[l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l]
print('(collin3 import) '+ctl[0] if ctl else '(no control line!)')
shares_vec=c3.shares_vec; TERMS=c3.TERMS; W={k:v[0] for k,v in c3.DEFAULT_OXEY_WEIGHTS.items()}
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
from keybo.analysis.surfaces import C30M

REG={**NAMED_LAYOUTS, **_EXTRA_NAMED}
c30m=set(C30M)
usable={n:s for n,s in REG.items() if set(s)==c30m}
print(f'\nregistry: {len(REG)} named; {len(usable)} are exact C30M permutations -> {sorted(usable)}')
skipped=sorted(set(REG)-set(usable)); print(f'skipped (charset mismatch): {skipped}')

rng=random.Random(31337)
def neigh(s,k):
    l=list(s)
    for _ in range(k):
        i,j=rng.randrange(30),rng.randrange(30); l[i],l[j]=l[j],l[i]
    return ''.join(l)
# near-optimal pool: each usable registry layout + 1-3 swap neighbours (still near-optimal)
pool_no=[]
for n,s in usable.items():
    pool_no.append(s)
    for _ in range(30):
        pool_no.append(neigh(s, rng.choice([1,1,2,2,3])))
print(f'near-optimal pool: {len(pool_no)} layouts ({len(usable)} seeds x 31)')
Xn=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool_no])
Xr=np.load('/tmp/penaudit/probe/_X_random.npy')

def summarize(X,label):
    sd=X.std(0); Xs=(X-X.mean(0))/np.where(sd>1e-12,sd,1.0)
    R=np.corrcoef(Xs.T); ev=np.linalg.eigvalsh(R)[::-1]
    eff=ev.sum()**2/(ev**2).sum()
    contrib=X*np.array([W[t] for t in TERMS]); tot=contrib.sum(1)
    cov=[float(np.cov(contrib[:,i],tot)[0,1]/tot.var()) for i in range(len(TERMS))]
    vif={}
    for i,t in enumerate(TERMS):
        A=np.delete(Xs,i,axis=1); y=Xs[:,i]
        co,*_=np.linalg.lstsq(A,y,rcond=None)
        r2=1-((y-A@co)**2).sum()/((y-y.mean())**2).sum(); vif[t]=1/max(1e-12,1-r2)
    print(f'\n=== {label} (n={X.shape[0]}) ===')
    print(f'  effective dof {eff:.2f} of {len(TERMS)}   score mean {tot.mean():.2f} sd {tot.std():.2f}')
    print(f'  {"term":14s}{"mean":>9s}{"sd":>9s}{"w*sd":>9s}{"var-share":>11s}{"VIF":>8s}')
    for i,t in enumerate(TERMS):
        print(f'  {t:14s}{X[:,i].mean():9.4f}{X[:,i].std():9.4f}{abs(W[t])*X[:,i].std():9.3f}{100*cov[i]:10.1f}%{vif[t]:8.2f}')
    return dict(eff_dof=float(eff), corr=R.tolist(), mean=X.mean(0).tolist(), sd=X.std(0).tolist(),
                min=X.min(0).tolist(), max=X.max(0).tolist(),
                var_share=cov, vif={k:float(v) for k,v in vif.items()},
                score_mean=float(tot.mean()), score_sd=float(tot.std()))
Sr=summarize(Xr,'RANDOM-PERMUTATION POOL')
Sn=summarize(Xn,'NEAR-OPTIMAL POOL (registry + 1-3 swap neighbours)')
print('\n=== DOMAIN SHIFT: does the random pool even COVER the near-optimal band? (trap 51) ===')
print(f'  {"term":14s}{"random [min,max]":>26s}{"near-opt [min,max]":>26s}  coverage')
for i,t in enumerate(TERMS):
    rl,rh=Xr[:,i].min(),Xr[:,i].max(); nl,nh=Xn[:,i].min(),Xn[:,i].max()
    inside = nl>=rl and nh<=rh
    frac=float(((Xn[:,i]>=rl)&(Xn[:,i]<=rh)).mean())
    print(f'  {t:14s}[{rl:9.4f},{rh:9.4f}]  [{nl:9.4f},{nh:9.4f}]  '
          f'{"COVERED" if inside else f"OUT ({100*(1-frac):.0f}% of near-opt outside)"}')
print('\n=== per-term rank-correlation of the two pools\' correlation structure ===')
from scipy.stats import spearmanr
iu=np.triu_indices(len(TERMS),1)
rr=np.array(Sr['corr'])[iu]; rn=np.array(Sn['corr'])[iu]
print(f'  spearman(rho_random, rho_nearopt) over {len(rr)} off-diagonal pairs = {spearmanr(rr,rn).statistic:+.4f}')
print(f'  max |rho_random - rho_nearopt| = {np.abs(rr-rn).max():.4f}')
json.dump(dict(random_pool=Sr, near_optimal_pool=Sn, terms=TERMS, weights=W,
   registry_used=sorted(usable), registry_skipped=skipped, near_opt_n=len(pool_no)),
   open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/band_compare.json','w'),indent=1)
print('\nwrote band_compare.json')
