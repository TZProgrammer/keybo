"""PER-CLUSTER attribution alongside per-term — the brief's explicit requirement.

Because effective dof is 2.50 in the operating band, a per-term slope is not an identified
quantity. So: (1) group the 11 terms into correlation clusters, (2) refit with ONE regressor
per cluster (the cluster's first PC), (3) report how much each CLUSTER explains and how the
per-term slopes redistribute inside it. Also a leave-one-CLUSTER-out (trap 25: leave-one-TERM
-out is anti-conservative when terms are duplicated).
"""
import io, contextlib, importlib.util, json, random, numpy as np
spec=importlib.util.spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print([l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
shares_vec=c3.shares_vec; TERMS=c3.TERMS; W={k:v[0] for k,v in c3.DEFAULT_OXEY_WEIGHTS.items()}
from keybo.analysis import surfaces as SF
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
from keybo.analysis.surfaces import C30M
NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
obj=SF.trigram_objective(SF.default_trigram_path(None)); mass=obj[3].sum()
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
print(f'pool n={len(pool)} (near-optimal, wider); corpus blend-v1')
sd=X.std(0); Xs=(X-X.mean(0))/sd
R=np.corrcoef(Xs.T)
# clusters at |rho|>=0.5 on THIS pool
par=list(range(len(TERMS)))
def find(a):
    while par[a]!=a: par[a]=par[par[a]]; a=par[a]
    return a
for i in range(len(TERMS)):
    for j in range(i+1,len(TERMS)):
        if abs(R[i,j])>=0.5: par[find(i)]=find(j)
cl={}
for i,t in enumerate(TERMS): cl.setdefault(find(i),[]).append(i)
CLUSTERS=[[TERMS[i] for i in v] for v in cl.values()]
ev=np.linalg.eigvalsh(R)[::-1]; eff=ev.sum()**2/(ev**2).sum()
print(f'effective dof on THIS pool = {eff:.2f} of 11')
print(f'{len(CLUSTERS)} clusters at |rho|>=0.5:')
for k,c in enumerate(CLUSTERS,1): print(f'  C{k}: {c}')
OUT={'pool_n':len(pool),'clusters':CLUSTERS,'effective_dof':float(eff),'per_source':{}}
for src in ('AALTO','COMMUNITY','POOL'):
    S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
    y=np.array([SF.score_fit(l,S,obj) for l in pool])/mass
    ys=(y-y.mean())/y.std()
    # full per-term fit
    cf,*_=np.linalg.lstsq(Xs,ys,rcond=None)
    r2f=1-((ys-Xs@cf)**2).sum()/(ys**2).sum()
    # cluster PC regressors
    Z=[]
    for c in cl.values():
        sub=Xs[:,c]
        u,s_,vt=np.linalg.svd(sub-sub.mean(0),full_matrices=False)
        pc=(sub-sub.mean(0))@vt[0]
        # orient so PC correlates positively with the cluster's mean share
        if np.corrcoef(pc,sub.mean(1))[0,1]<0: pc=-pc
        Z.append(pc/pc.std())
    Z=np.column_stack(Z)
    cz,*_=np.linalg.lstsq(Z,ys,rcond=None)
    r2c=1-((ys-Z@cz)**2).sum()/(ys**2).sum()
    print(f'\n=== {src}: per-TERM R2 = {r2f:.4f}   per-CLUSTER(PC1) R2 = {r2c:.4f}  '
          f'(cluster frame uses {Z.shape[1]} regressors vs {Xs.shape[1]})')
    print(f'  {"cluster":42s}{"beta(PC1)":>11s}{"|beta|share":>13s}{"LOCO dR2":>10s}')
    tot=np.abs(cz).sum()
    loco={}
    for k,(c,b) in enumerate(zip(cl.values(),cz),1):
        Zk=np.delete(Z,k-1,axis=1)
        ck,*_=np.linalg.lstsq(Zk,ys,rcond=None)
        r2k=1-((ys-Zk@ck)**2).sum()/(ys**2).sum()
        nm=','.join(TERMS[i] for i in c)
        loco[nm]=float(r2c-r2k)
        print(f'  {nm:42s}{b:+11.4f}{100*abs(b)/tot:12.1f}%{r2c-r2k:10.4f}')
    OUT['per_source'][src]=dict(r2_per_term=float(r2f), r2_per_cluster=float(r2c),
        cluster_beta={','.join(TERMS[i] for i in c):float(b) for c,b in zip(cl.values(),cz)},
        loco_dr2=loco,
        per_term_beta={t:float(cf[i]) for i,t in enumerate(TERMS)})
json.dump(OUT,open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/cluster_attribution.json','w'),indent=1)
print('\nwrote cluster_attribution.json')
