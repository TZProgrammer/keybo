"""FUNCTIONAL FORM per term: constant / linear-in-share / saturating / threshold / ZERO.

Method (all three sources, near-optimal band = the operating domain):
  For each term, bin the pool by that term's share and read the MEAN fitted ms/char per bin,
  holding nothing (marginal shape) -- then ALSO partial out the other 10 terms (conditional
  shape). A term whose marginal slope flattens at high share SATURATES; one that is flat
  everywhere is ZERO; one with a knee is THRESHOLD.
  The 'price per share-point' is the slope in ms/char per percentage point of share -- the
  unit `DEFAULT_OXEY_WEIGHTS` is defined in (weight per corpus-share PERCENT).
Bootstrap over LAYOUTS for the slope CI.
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
# a WIDER near-optimal pool for form estimation: more neighbours, more swap depths
pool=[]
for n,s in usable.items():
    pool.append(s)
    for _ in range(80): pool.append(neigh(s,rng.choice([1,1,2,2,3,3,4,5])))
print(f'form pool: {len(pool)} layouts ({len(usable)} seeds); corpus blend-v1')
X=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
OUT={'pool_n':len(pool),'terms':TERMS,'weights':W,'per_source':{}}
for src in ('AALTO','COMMUNITY','POOL'):
    S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
    y=np.array([SF.score_fit(l,S,obj) for l in pool])/mass    # ms/char
    print(f'\n################ {src}_TRI_PS_FREQ_PRIOR.native — ms/char, g-frame, 90 WPM baked ################')
    print(f'  fitted ms/char over pool: mean {y.mean():.3f} sd {y.std():.3f} range [{y.min():.3f},{y.max():.3f}]')
    print(f'\n  {"term":13s}{"valid range (share%)":>22s}{"lin slope ms/char/pt":>22s}{"CI95":>20s}'
          f'{"quad":>9s}{"form":>12s}')
    src_out={}
    for i,t in enumerate(TERMS):
        x=X[:,i]
        lo,hi=np.percentile(x,[1,99])
        # linear + quadratic in the term alone (marginal shape)
        A1=np.column_stack([np.ones_like(x),x]); c1,*_=np.linalg.lstsq(A1,y,rcond=None)
        xc=x-x.mean()
        A2=np.column_stack([np.ones_like(x),xc,xc**2]); c2,*_=np.linalg.lstsq(A2,y,rcond=None)
        r2_1=1-((y-A1@c1)**2).sum()/((y-y.mean())**2).sum()
        r2_2=1-((y-A2@c2)**2).sum()/((y-y.mean())**2).sum()
        # bootstrap the linear slope over LAYOUTS
        bs=np.empty(2000); n=len(x); rr=np.random.default_rng(7)
        for b in range(2000):
            ix=rr.integers(0,n,n); Ab=np.column_stack([np.ones(n),x[ix]])
            cb,*_=np.linalg.lstsq(Ab,y[ix],rcond=None); bs[b]=cb[1]
        ci=(float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
        # form call: quadratic gain + curvature sign
        gain=r2_2-r2_1
        curv=c2[2]
        if ci[0]<=0<=ci[1]: form='ZERO/flat'
        elif gain>0.01 and np.sign(curv)!=np.sign(c1[1]): form='SATURATING'
        elif gain>0.01: form='CONVEX'
        else: form='LINEAR'
        print(f'  {t:13s}[{lo:8.3f},{hi:8.3f}]{c1[1]:22.4f}   [{ci[0]:+7.4f},{ci[1]:+7.4f}]'
              f'{gain:9.4f}{form:>12s}')
        src_out[t]=dict(valid_range=[float(lo),float(hi)], slope_ms_per_char_per_pt=float(c1[1]),
                        slope_ci95=list(ci), r2_linear=float(r2_1), r2_quad=float(r2_2),
                        quad_gain=float(gain), curvature=float(curv), form=form,
                        share_mean=float(x.mean()), share_sd=float(x.std()))
    OUT['per_source'][src]=src_out
json.dump(OUT,open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/functional_form.json','w'),indent=1)
print('\nwrote functional_form.json')
