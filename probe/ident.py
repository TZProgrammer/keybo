"""Identification structure over the 11 terms. Estimated WITHIN a homogeneous pool (trap 26:
a mixed optimized+random pool gives a Simpson artifact BELOW both sub-pools)."""
import json, numpy as np
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS
TERMS=list(DEFAULT_OXEY_WEIGHTS); W={k:v[0] for k,v in DEFAULT_OXEY_WEIGHTS.items()}
X=np.load('/tmp/scissorprice/probe/_X_random.npy')
print(f'pool: {X.shape[0]} random C30M permutations, blend-v1  (HOMOGENEOUS -- trap 26)')
sd=X.std(0); Xs=(X-X.mean(0))/np.where(sd>1e-12,sd,1.0)
R=np.corrcoef(Xs.T)
print('\n=== correlation matrix (|rho| >= 0.70 starred) ===')
print(f'{"":>13s}'+''.join(f'{t[:6]:>8s}' for t in TERMS))
for i,t in enumerate(TERMS):
    print(f'{t:>13s}'+''.join((f'{R[i,j]:+8.2f}' if (abs(R[i,j])<0.70 or i==j) else f'*{R[i,j]:+7.2f}') for j in range(len(TERMS))))
ev=np.linalg.eigvalsh(R)[::-1]
eff=ev.sum()**2/(ev**2).sum()
print(f'\neigenvalues: {np.round(ev,3)}')
print(f'PARTICIPATION-RATIO effective dof = {eff:.2f} of {len(TERMS)}')
cum=np.cumsum(ev)/ev.sum()
for k,c in enumerate(cum,1):
    if c>=0.95: print(f'  {k} of {len(TERMS)} PCs explain {100*c:.1f}% of variance'); break
print(f'  PC1 alone: {100*cum[0]:.1f}% ; PC1-3: {100*cum[2]:.1f}%')
print('\n=== VIF: regress each term on the other 10 ===')
print(f'{"term":14s}{"R2_others":>11s}{"VIF":>10s}   verdict')
vifs={}
for i,t in enumerate(TERMS):
    A=np.delete(Xs,i,axis=1); y=Xs[:,i]
    coef,*_=np.linalg.lstsq(A,y,rcond=None)
    r2=1-((y-A@coef)**2).sum()/((y-y.mean())**2).sum()
    v=1/max(1e-12,1-r2); vifs[t]=float(v)
    verdict='NOT separately identified (VIF>10)' if v>10 else ('marginal (VIF 5-10)' if v>5 else 'identified')
    print(f'{t:14s}{r2:11.4f}{v:10.2f}   {verdict}')
print('\n=== correlation clusters, single-linkage ===')
for thr in (0.5,0.6,0.7,0.8):
    par=list(range(len(TERMS)))
    def find(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    for i in range(len(TERMS)):
        for j in range(i+1,len(TERMS)):
            if abs(R[i,j])>=thr: par[find(i)]=find(j)
    cl={}
    for i,t in enumerate(TERMS): cl.setdefault(find(i),[]).append(t)
    groups=[v for v in cl.values()]
    print(f'  |rho|>={thr}: {len(groups)} clusters  {groups}')
print('\n=== the DOMINANT collinear pairs (|rho| >= 0.5) ===')
pairs=sorted(((abs(R[i,j]),R[i,j],TERMS[i],TERMS[j]) for i in range(len(TERMS)) for j in range(i+1,len(TERMS))),reverse=True)
for a,r,ti,tj in pairs:
    if a<0.5: break
    print(f'  rho({ti:13s}, {tj:13s}) = {r:+.4f}')
print('\n=== SCORE-SHARE decomposition: which terms MOVE the total? ===')
contrib=X*np.array([W[t] for t in TERMS])
tot=contrib.sum(1)
print(f'  total oxey score over pool: mean {tot.mean():.2f}  sd {tot.std():.2f}  range [{tot.min():.2f},{tot.max():.2f}]')
print(f'{"term":14s}{"w*sd":>9s}{"share of|w*sd|":>16s}{"corr(term,total)":>18s}{"var-share via cov":>19s}')
wsd=np.array([abs(W[t])*X[:,i].std() for i,t in enumerate(TERMS)])
for i,t in enumerate(TERMS):
    cr=np.corrcoef(contrib[:,i],tot)[0,1]
    covshare=np.cov(contrib[:,i],tot)[0,1]/tot.var()
    print(f'{t:14s}{wsd[i]:9.3f}{100*wsd[i]/wsd.sum():15.1f}%{cr:18.3f}{100*covshare:18.1f}%')
json.dump(dict(terms=TERMS,weights=W,pool_n=int(X.shape[0]),corpus='blend-v1',
  pool_kind='random C30M permutations (homogeneous)',
  share_mean=X.mean(0).tolist(),share_sd=X.std(0).tolist(),
  share_min=X.min(0).tolist(),share_max=X.max(0).tolist(),
  corr=R.tolist(),eigenvalues=ev.tolist(),effective_dof=float(eff),vif=vifs,
  w_times_sd=wsd.tolist(),
  cov_share_of_total_var=[float(np.cov(contrib[:,i],tot)[0,1]/tot.var()) for i in range(len(TERMS))]),
  open('/local/home/zegertho/agent/state/scissorprice/artifacts/identification.json','w'),indent=1)
print('\nwrote identification.json')
