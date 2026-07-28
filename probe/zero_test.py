"""Explicit ZERO test: for each term, does DROPPING it change the RANKING the scorer produces?
A weight that cannot move the ordering is operationally zero regardless of its value.
Also: the trap-42 discipline -- put a CI on the number that would justify an exclusion."""
import io, contextlib, importlib.util, json, random, numpy as np
from scipy.stats import spearmanr, kendalltau
spec=importlib.util.spec_from_file_location('c3','/tmp/scissorprice/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print([l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
shares_vec=c3.shares_vec; TERMS=c3.TERMS; W={k:v[0] for k,v in c3.DEFAULT_OXEY_WEIGHTS.items()}
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
from keybo.analysis.surfaces import C30M
REG={**NAMED_LAYOUTS,**_EXTRA_NAMED}; usable={n:s for n,s in REG.items() if set(s)==set(C30M)}
rng=random.Random(31337)
def neigh(s,k):
    l=list(s)
    for _ in range(k):
        i,j=rng.randrange(30),rng.randrange(30); l[i],l[j]=l[j],l[i]
    return ''.join(l)
pool=[]
for n,s in usable.items():
    pool.append(s)
    for _ in range(80): pool.append(neigh(s,rng.choice([1,1,2,2,3,3,4,5])))
X=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
w=np.array([W[t] for t in TERMS]); full=X@w
print(f'pool n={len(pool)} near-optimal, blend-v1. Full oxey score: mean {full.mean():.3f} sd {full.std():.3f}')
print(f'\n{"term dropped":14s}{"spearman":>10s}{"kendall":>9s}{"top1 same":>10s}{"top10 overlap":>14s}'
      f'{"max rank move":>14s}  reading')
base_order=np.argsort(full)
for i,t in enumerate(TERMS):
    w2=w.copy(); w2[i]=0.0; alt=X@w2
    sp=spearmanr(full,alt).statistic; kt=kendalltau(full,alt).statistic
    o2=np.argsort(alt)
    top1 = base_order[0]==o2[0]
    ov=len(set(base_order[:10].tolist())&set(o2[:10].tolist()))
    r1=np.empty(len(pool)); r1[base_order]=np.arange(len(pool))
    r2=np.empty(len(pool)); r2[o2]=np.arange(len(pool))
    mx=int(np.abs(r1-r2).max())
    reading = 'operationally ZERO (ranking unchanged)' if sp>0.9999 else ('minor' if sp>0.99 else 'material')
    print(f'  {t:12s}{sp:10.5f}{kt:9.4f}{str(top1):>10s}{ov:>10d}/10{mx:14d}  {reading}')
print('\n=== and the RANKING AGREEMENT of the oxey score with MEASURED time (the real question) ===')
from keybo.analysis import surfaces as SF
NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
obj=SF.trigram_objective(SF.default_trigram_path(None)); mass=obj[3].sum()
for src in ('AALTO','COMMUNITY','POOL'):
    S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
    y=np.array([SF.score_fit(l,S,obj) for l in pool])/mass
    print(f'  {src:10s} spearman(oxey_score, fitted ms/char) = {spearmanr(full,y).statistic:+.4f}')
    # and with the sign-corrected variant
    wc=w.copy()
    for nm in ('inroll','outroll','onehand'): wc[TERMS.index(nm)]=abs(wc[TERMS.index(nm)])
    print(f'  {"":10s} spearman(sign-corrected oxey, fitted)  = {spearmanr(X@wc,y).statistic:+.4f}')
