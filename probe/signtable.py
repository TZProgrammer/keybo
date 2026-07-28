"""The auditable marginal-vs-conditional table for ALL 11 terms, both bands, 3 sources.
Persisted as JSON so the suppression claim needs no re-run."""
import io, contextlib, importlib.util, random, json, numpy as np
from scipy.stats import spearmanr, pearsonr
spec=importlib.util.spec_from_file_location('c3','/tmp/scissorprice/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
PC=[l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0]; print(PC)
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
POOLS={}
p341=[]
for n,s in usable.items():
    p341.append(s)
    for _ in range(30): p341.append(neigh(s,rng.choice([1,1,2,2,3])))
POOLS['near_optimal_n341']=p341
rng2=random.Random(20260728)
POOLS['random_n400']=[''.join(rng2.sample(list(C30M),30)) for _ in range(400)]
OUT={'note':'marginal r = pearson(share, fitted ms/char), NO other term partialled. conditional beta '
             '= standardized OLS coefficient with all 11 terms. oxey convention: POSITIVE weight = penalty '
             '= expects MORE time, so a term is CORRECTLY signed iff sign(marginal r) == sign(oxey weight).',
     'frame':'g-only, <POOL>_TRI_PS_FREQ_PRIOR.native, baked 90 WPM, corpus blend-v1',
     'positive_control':PC.strip(),'weights':W,'pools':{}}
for pname,pool in POOLS.items():
    X=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
    Xs=(X-X.mean(0))/X.std(0)
    rec={}
    for src in ('AALTO','COMMUNITY','POOL'):
        S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
        y=np.array([SF.score_fit(l,S,obj) for l in pool])/mass
        ys=(y-y.mean())/y.std()
        co,*_=np.linalg.lstsq(Xs,ys,rcond=None)
        r2=1-((ys-Xs@co)**2).sum()/(ys**2).sum()
        rec[src]={'multivariate_r2':float(r2),
          'marginal_r':{t:float(pearsonr(X[:,i],y).statistic) for i,t in enumerate(TERMS)},
          'marginal_rho':{t:float(spearmanr(X[:,i],y).statistic) for i,t in enumerate(TERMS)},
          'conditional_beta':{t:float(co[i]) for i,t in enumerate(TERMS)}}
    OUT['pools'][pname]={'n':len(pool),'per_source':rec}
    if pname=='near_optimal_n341':
        print(f'\n=== {pname} (n={len(pool)}) — THE IN-BAND TABLE ===')
        print(f'{"term":14s}{"oxeyW":>7s}| {"marg r A":>9s}{"C":>8s}{"P":>8s} | {"cond b A":>9s}{"C":>8s}{"P":>8s} | verdict')
        for t in TERMS:
            m=[rec[s]['marginal_r'][t] for s in ('AALTO','COMMUNITY','POOL')]
            b=[rec[s]['conditional_beta'][t] for s in ('AALTO','COMMUNITY','POOL')]
            exp=np.sign(W[t])
            msign = 'marg OK' if all(np.sign(x)==exp for x in m) else ('marg WRONG 3/3' if all(np.sign(x)==-exp for x in m) else 'marg MIXED')
            supp = ' SUPPRESSION' if np.sign(np.mean(m))!=np.sign(np.mean(b)) else ''
            print(f'{t:14s}{W[t]:+7.1f}| {m[0]:+9.3f}{m[1]:+8.3f}{m[2]:+8.3f} | {b[0]:+9.3f}{b[1]:+8.3f}{b[2]:+8.3f} | {msign}{supp}')
json.dump(OUT,open('/local/home/zegertho/agent/state/scissorprice/artifacts/sign_table.json','w'),indent=1)
print('\nwrote sign_table.json')
