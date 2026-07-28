"""Verify the cited 'oxey-style is R2=0.9937 on {sfb,lsb,scissor,imbalance,redir,alt}'.
The figure traces only to the parent's brief + a callback log, with no primary artifact, so I
re-derive it rather than cite it (trap 20 / 'a label is not its referent')."""
import io, contextlib, importlib.util, random, numpy as np
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
for label, pool in (('near-optimal', [s for n,s in usable.items() for _ in [0]] +
                    [neigh(s,rng.choice([1,1,2,2,3,3,4,5])) for n,s in usable.items() for _ in range(80)]),):
    X=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
    y=X@np.array([W[t] for t in TERMS])          # the oxey-style score itself
    SIX=['sfb','lsb','scissor','imbalance','redirect','alternate']
    ix=[TERMS.index(t) for t in SIX]
    A=np.column_stack([np.ones(len(pool))]+[X[:,i] for i in ix])
    co,*_=np.linalg.lstsq(A,y,rcond=None)
    r2=1-((y-A@co)**2).sum()/((y-y.mean())**2).sum()
    print(f'{label} pool n={len(pool)}: R2(oxey-style ~ {SIX}) = {r2:.4f}')
    print(f'  cited figure was 0.9937 -> {"REPRODUCED (within 0.01)" if abs(r2-0.9937)<0.01 else "DIFFERS on my frame/pool"}')
    # and on a random pool
    rp=[''.join(random.Random(1000+i).sample(list(C30M),30)) for i in range(400)]
    Xr=np.array([[shares_vec(s)[t] for t in TERMS] for s in rp])
    yr=Xr@np.array([W[t] for t in TERMS])
    Ar=np.column_stack([np.ones(len(rp))]+[Xr[:,i] for i in ix])
    cr,*_=np.linalg.lstsq(Ar,yr,rcond=None)
    r2r=1-((yr-Ar@cr)**2).sum()/((yr-yr.mean())**2).sum()
    print(f'random pool n={len(rp)}:      R2 = {r2r:.4f}')
    print('  NOTE this is near-tautological by construction: the oxey score IS a weighted sum of')
    print('  these 6 PLUS dsfb/inroll/outroll/onehand/bad_redirect, so R2 measures how much the')
    print('  OTHER FIVE add. 1-R2 is their unique contribution.')
    print(f'  -> the other five terms contribute 1-R2 = {1-r2:.4f} (near-opt) / {1-r2r:.4f} (random)')
