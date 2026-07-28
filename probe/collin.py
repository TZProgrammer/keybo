"""The single biggest threat: are the 11 oxey terms SEPARATELY IDENTIFIED?

Method: build the 11-term share vector for a POOL of layouts (the shares the scorer itself
computes, via OxeyStyleScorer.pattern_shares -- the shipped path, not a reimplementation),
then measure (a) the correlation structure, (b) effective dof, (c) VIF per term, (d) which
terms cluster. Estimated WITHIN a homogeneous pool (trap 26).
"""
import sys, json, numpy as np, itertools, random
from keybo.scoring.oxey import OxeyStyleScorer, DEFAULT_OXEY_WEIGHTS
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.analysis.surfaces import C30M

CD=production_corpus_dir(None)
print('corpus dir:', CD)
bg=load_frequencies(str(CD/'bigrams.txt')); tg=load_frequencies(str(CD/'trigrams.txt'))
import os
skpath = CD/'1-skip.txt'
sg=load_frequencies(str(skpath))
print(f'bigrams {len(bg)}  skipgrams {len(sg)}  trigrams {len(tg)}')
sc=OxeyStyleScorer(bg,sg,tg)
TERMS=list(DEFAULT_OXEY_WEIGHTS)
print(f'\n{len(TERMS)} TERMS (verbatim from the dict):')
for i,(k,(w,why)) in enumerate(DEFAULT_OXEY_WEIGHTS.items()):
    print(f'  {i+1:2d}. {k:14s} {w:+6.1f}')

rng=random.Random(20260728)
def rand_layout():
    ch=list(C30M); rng.shuffle(ch); return ''.join(ch)
POOL_N=400
pool=[rand_layout() for _ in range(POOL_N)]
X=np.array([[sc.pattern_shares(Layout(s,G))[t] for t in TERMS] for s in pool])
print(f'\nshare matrix: {X.shape}  (random-permutation pool, n={POOL_N}, corpus=blend-v1)')
for i,t in enumerate(TERMS):
    print(f'  {t:14s} mean {X[:,i].mean():8.4f}  sd {X[:,i].std():8.5f}  range [{X[:,i].min():.4f},{X[:,i].max():.4f}]')
# trap 23: any term a permutation INVARIANT? test directly by shuffling, not via std>0
print('\n=== trap 23: permutation invariance test (distinct values over the pool) ===')
for i,t in enumerate(TERMS):
    u=len(np.unique(np.round(X[:,i],10)))
    print(f'  {t:14s} distinct values: {u:4d}  sd={X[:,i].std():.3e}  {"<< INVARIANT" if u<=2 else ""}')
Xs=(X-X.mean(0))/np.where(X.std(0)>1e-12,X.std(0),1.0)
R=np.corrcoef(Xs.T)
print('\n=== correlation matrix (|rho| >= 0.7 flagged) ===')
print('      '+''.join(f'{t[:6]:>8s}' for t in TERMS))
for i,t in enumerate(TERMS):
    print(f'{t[:6]:>6s}'+''.join((f'{R[i,j]:+8.2f}' if abs(R[i,j])<0.7 or i==j else f'*{R[i,j]:+7.2f}') for j in range(len(TERMS))))
ev=np.linalg.eigvalsh(R)[::-1]
print(f'\neigenvalues: {np.round(ev,3)}')
print(f'  participation-ratio effective dof = {(ev.sum()**2/(ev**2).sum()):.2f}  of {len(TERMS)}')
cum=np.cumsum(ev)/ev.sum()
for k,c in enumerate(cum,1):
    if c>=0.95: print(f'  {k} components explain {100*c:.1f}% of variance (95% threshold)'); break
print('\n=== VIF per term (1/(1-R2) regressing each term on the other 10) ===')
for i,t in enumerate(TERMS):
    if X[:,i].std()<1e-12: print(f'  {t:14s} VIF undefined (constant)'); continue
    A=np.delete(Xs,i,axis=1); y=Xs[:,i]
    coef,*_=np.linalg.lstsq(A,y,rcond=None)
    r2=1-((y-A@coef)**2).sum()/((y-y.mean())**2).sum()
    vif=1/max(1e-12,1-r2)
    print(f'  {t:14s} R2_other={r2:7.4f}  VIF={vif:9.2f}  {"<< NOT SEPARATELY IDENTIFIED" if vif>10 else ""}')
# correlation clusters at |rho|>=0.7 (single linkage)
print('\n=== correlation clusters (single-linkage, |rho| >= 0.7) ===')
par=list(range(len(TERMS)))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for i in range(len(TERMS)):
    for j in range(i+1,len(TERMS)):
        if abs(R[i,j])>=0.7: par[find(i)]=find(j)
cl={}
for i,t in enumerate(TERMS): cl.setdefault(find(i),[]).append(t)
for k,v in enumerate(cl.values(),1): print(f'  cluster {k}: {v}')
print(f'  => {len(cl)} clusters over {len(TERMS)} terms')
json.dump(dict(terms=TERMS, weights={k:v[0] for k,v in DEFAULT_OXEY_WEIGHTS.items()},
   pool_n=POOL_N, corpus=str(CD), share_mean=X.mean(0).tolist(), share_sd=X.std(0).tolist(),
   corr=R.tolist(), eigenvalues=ev.tolist(),
   eff_dof=float(ev.sum()**2/(ev**2).sum()),
   clusters=[v for v in cl.values()]),
   open('/local/home/zegertho/agent/state/scissorprice/artifacts/collinearity.json','w'), indent=1)
print('\nwrote collinearity.json')
