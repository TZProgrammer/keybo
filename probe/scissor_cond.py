"""Does the scissor 7.0x survive CONDITIONING? The dossier's ratio is MARGINAL/MARGINAL.
Recompute it from the CONDITIONAL (partial) slopes in raw ms/char/pt units, same anchor."""
import io, contextlib, importlib.util, random, json, numpy as np
spec=importlib.util.spec_from_file_location('c3','/tmp/scissorprice/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print([l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
shares_vec=c3.shares_vec; TERMS=c3.TERMS
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
print(f'pool n={len(pool)}')
out={}
print(f"\n{'src':10s}{'scissor MARG':>14s}{'scissor COND':>14s}{'sfb MARG':>10s}{'sfb COND':>10s}"
      f"{'ratio MARG':>12s}{'ratio COND':>12s}")
for src in ('AALTO','COMMUNITY','POOL'):
    S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
    y=np.array([SF.score_fit(l,S,obj) for l in pool])/mass
    A=np.column_stack([np.ones(len(pool)),X]); co,*_=np.linalg.lstsq(A,y,rcond=None)
    cond={t:co[1+i] for i,t in enumerate(TERMS)}
    marg={}
    for i,t in enumerate(TERMS):
        a=np.column_stack([np.ones(len(pool)),X[:,i]]); c,*_=np.linalg.lstsq(a,y,rcond=None); marg[t]=c[1]
    rm=(marg['scissor']/marg['sfb'])*12.0/4.0
    rc=(cond['scissor']/cond['sfb'])*12.0/4.0
    print(f'{src:10s}{marg["scissor"]:+14.4f}{cond["scissor"]:+14.4f}{marg["sfb"]:+10.4f}{cond["sfb"]:+10.4f}'
          f'{rm:11.3f}x{rc:11.3f}x')
    out[src]=dict(marg_scissor=float(marg['scissor']),cond_scissor=float(cond['scissor']),
                  marg_sfb=float(marg['sfb']),cond_sfb=float(cond['sfb']),
                  ratio_marg=float(rm),ratio_cond=float(rc),
                  cond_implied_scissor=float((cond['scissor']/cond['sfb'])*12.0))
print('\n  ratio_COND is the version that partials out the other ten. If it stays >>1, the 7.0x')
print('  direction survives conditioning; its LEVEL is what moves.')
json.dump(out,open('/local/home/zegertho/agent/state/scissorprice/artifacts/scissor_conditional.json','w'),indent=1)
print('  wrote scissor_conditional.json')
