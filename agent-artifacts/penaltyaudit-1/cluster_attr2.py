"""PER-CLUSTER attribution — fixed clustering. Single-linkage CHAINED all 11 into one group
on the wide near-optimal pool (a known single-linkage pathology), which is uninformative.
Use hierarchical AVERAGE linkage on distance 1-|rho| and report a threshold SWEEP, so the
reader sees how the attribution depends on the grouping rather than on one arbitrary cut."""
import io, contextlib, importlib.util, json, random, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
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
Xs=(X-X.mean(0))/X.std(0); R=np.corrcoef(Xs.T)
ev=np.linalg.eigvalsh(R)[::-1]; eff=ev.sum()**2/(ev**2).sum()
D=1-np.abs(R); np.fill_diagonal(D,0.0)
Zl=linkage(squareform(D,checks=False),method='average')
print(f'pool n={len(pool)}; effective dof {eff:.2f} of 11 (AVERAGE-linkage clustering on 1-|rho|)')
SURF={s:np.load(f'{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy') for s in ('AALTO','COMMUNITY','POOL')}
Y={s:(lambda y:(y-y.mean())/y.std())(np.array([SF.score_fit(l,SURF[s],obj) for l in pool])/mass) for s in SURF}
OUT={'pool_n':len(pool),'effective_dof':float(eff),'sweep':{}}
for K in (2,3,4,5,6):
    lab=fcluster(Zl,K,criterion='maxclust')
    groups=[[i for i in range(len(TERMS)) if lab[i]==g] for g in sorted(set(lab))]
    names=[','.join(TERMS[i] for i in g) for g in groups]
    print(f'\n===== K={K} clusters =====')
    for nm in names: print(f'   {nm}')
    ent={}
    for src in SURF:
        ys=Y[src]
        Zc=[]
        for g in groups:
            sub=Xs[:,g]
            _,_,vt=np.linalg.svd(sub-sub.mean(0),full_matrices=False)
            pc=(sub-sub.mean(0))@vt[0]
            if np.corrcoef(pc,sub.mean(1))[0,1]<0: pc=-pc
            Zc.append(pc/pc.std())
        Zc=np.column_stack(Zc)
        cz,*_=np.linalg.lstsq(Zc,ys,rcond=None)
        r2c=1-((ys-Zc@cz)**2).sum()/(ys**2).sum()
        cf,*_=np.linalg.lstsq(Xs,ys,rcond=None)
        r2f=1-((ys-Xs@cf)**2).sum()/(ys**2).sum()
        loco=[]
        for k in range(len(groups)):
            Zk=np.delete(Zc,k,axis=1)
            ck,*_=np.linalg.lstsq(Zk,ys,rcond=None)
            loco.append(float(r2c-(1-((ys-Zk@ck)**2).sum()/(ys**2).sum())))
        print(f'  {src:10s} R2 per-term {r2f:.4f} | per-cluster {r2c:.4f} | '
              +'  '.join(f'{nm.split(",")[0][:9]}..:b={b:+.3f},dR2={l:.4f}' for nm,b,l in zip(names,cz,loco)))
        ent[src]=dict(r2_per_term=float(r2f),r2_per_cluster=float(r2c),
                      betas=dict(zip(names,[float(b) for b in cz])),
                      loco_dr2=dict(zip(names,loco)))
    OUT['sweep'][K]=dict(groups=names, per_source=ent)
json.dump(OUT,open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/cluster_attribution.json','w'),indent=1)
print('\nwrote cluster_attribution.json (overwrote the single-linkage version)')
