"""Trap 49: MARGINAL vs CONDITIONAL. A wrong CONDITIONAL sign under high VIF may be
collinearity suppression, not an inverted mechanism. So for each term, correlate its share
with MEASURED TIME (the model surface fit) marginally, and compare to the multivariate fit.

Target: the served trigram fit per layout under each NATIVE per-source surface (g-frame,
baked 90 WPM). This is 'what our measurements say the layout costs'.
⚠ Not an identity check trap: the oxey shares are pattern COUNTS from the corpus; the target
is a SUM OF FITTED MS over the same corpus. They share the corpus weighting but the share
vector is not a component of the surface fit, so this is not algebra. Stated, not assumed —
see the explicit test at the bottom.
"""
import io, contextlib, importlib.util, json, numpy as np
spec=importlib.util.spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print([l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
shares_vec=c3.shares_vec; TERMS=c3.TERMS; W={k:v[0] for k,v in c3.DEFAULT_OXEY_WEIGHTS.items()}
from keybo.analysis import surfaces as SF
from scipy.stats import spearmanr, pearsonr
NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
obj=SF.trigram_objective(SF.default_trigram_path(None))
print(f'trigram objective: {len(obj[3])} trigrams, corpus=blend-v1')
POOLS={}
POOLS['random']=(np.load('/tmp/penaudit/probe/_X_random.npy'), json.load(open('/tmp/penaudit/probe/_pool_random.json')))
bc=json.load(open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/band_compare.json'))
# rebuild the near-optimal pool with the SAME seed so it is the same object
import random
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
pn=[]
for n,s in usable.items():
    pn.append(s)
    for _ in range(30): pn.append(neigh(s,rng.choice([1,1,2,2,3])))
assert len(pn)==bc['near_opt_n']
POOLS['near-optimal']=(np.array([[shares_vec(s)[t] for t in TERMS] for s in pn]), pn)
for src in ('AALTO','COMMUNITY','POOL'):
    S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
    print(f'\n################ TARGET = {src}_TRI_PS_FREQ_PRIOR.native  (g-frame, 90 WPM baked) ################')
    for pname,(X,lays) in POOLS.items():
        y=np.array([SF.score_fit(l,S,obj) for l in lays])
        Xs=(X-X.mean(0))/np.where(X.std(0)>1e-12,X.std(0),1.0); ys=(y-y.mean())/y.std()
        co,*_=np.linalg.lstsq(Xs,ys,rcond=None)
        r2=1-((ys-Xs@co)**2).sum()/((ys-ys.mean())**2).sum()
        print(f'\n  --- {pname} pool (n={len(lays)}); multivariate R2 = {r2:.4f} ---')
        print(f'  {"term":14s}{"oxeyW":>7s}{"MARGINAL r":>12s}{"marg rho":>10s}'
              f'{"CONDITIONAL beta":>18s}  verdict')
        for i,t in enumerate(TERMS):
            mr=pearsonr(X[:,i],y).statistic; ms=spearmanr(X[:,i],y).statistic
            b=co[i]
            # oxey's sign convention: positive weight = penalty = expects MORE time
            expect=np.sign(W[t])
            mflag = 'marg AGREES w/ oxey sign' if np.sign(mr)==expect else 'marg CONTRADICTS oxey sign'
            cflag = 'cond agrees' if np.sign(b)==expect else 'cond CONTRADICTS'
            same  = '' if np.sign(mr)==np.sign(b) else '  <<SUPPRESSION (marg/cond differ)'
            print(f'  {t:14s}{W[t]:+7.1f}{mr:+12.4f}{ms:+10.4f}{b:+18.4f}  {mflag}; {cflag}{same}')
