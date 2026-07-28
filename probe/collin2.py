"""Identification analysis of the 11 oxey terms — VECTORIZED, positive-controlled.

Trap 28 applies: I must not lose the shipped scorer's validation. So the design is
(1) precompute per-(slot-pair)/(slot-triple) class masks ONCE from the SAME predicates the
scorer uses, (2) accumulate corpus mass into slot space with numpy, (3) POSITIVE-CONTROL the
whole vectorized path against `OxeyStyleScorer.pattern_shares` on several layouts, and only
then use it on a large pool.
"""
import sys, json, numpy as np, random, time
from keybo.scoring.oxey import OxeyStyleScorer, DEFAULT_OXEY_WEIGHTS
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.features import classify as C
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.analysis.surfaces import C30M

CD=production_corpus_dir(None)
bg=load_frequencies(str(CD/'bigrams.txt')); tg=load_frequencies(str(CD/'trigrams.txt'))
sg=load_frequencies(str(CD/'1-skip.txt'))
TERMS=list(DEFAULT_OXEY_WEIGHTS)
S=list(G.slots); NS=len(S)

# ---------- slot-space masks, from the scorer's OWN predicates -------------------------
t0=time.time()
M_sfb=np.zeros((NS,NS),bool); M_alt=np.zeros((NS,NS),bool); M_lsb=np.zeros((NS,NS),bool)
M_sci=np.zeros((NS,NS),bool); M_in=np.zeros((NS,NS),bool);  M_out=np.zeros((NS,NS),bool)
M_dsfb=np.zeros((NS,NS),bool); HAND=np.array([G.hand(p[0]) for p in S])
for i,a in enumerate(S):
    for j,b in enumerate(S):
        cls=C.classify_positions(G,a,b)
        if cls is C.BigramClass.SAME_FINGER and a!=b: M_sfb[i,j]=True
        elif cls is C.BigramClass.ALTERNATE: M_alt[i,j]=True
        if C.is_lsb(G,a,b): M_lsb[i,j]=True
        if C.is_scissor(G,a,b): M_sci[i,j]=True
        if C.is_inwards(G,a,b): M_in[i,j]=True
        if C.is_outwards(G,a,b): M_out[i,j]=True
        if G.same_finger(a[0],b[0]) and a!=b: M_dsfb[i,j]=True
M_oh=np.zeros((NS,NS,NS),bool); M_rd=np.zeros((NS,NS,NS),bool); M_br=np.zeros((NS,NS,NS),bool)
for i,a in enumerate(S):
    for j,b in enumerate(S):
        if HAND[i]==0 or HAND[i]!=HAND[j]: continue
        d1=abs(b[0])-abs(a[0])
        if d1==0: continue
        for k,c in enumerate(S):
            if HAND[k]!=HAND[i]: continue
            d2=abs(c[0])-abs(b[0])
            if d2==0: continue
            if (d1>0)==(d2>0): M_oh[i,j,k]=True
            else:
                M_rd[i,j,k]=True
                if not any(abs(p[0]) in (1,2) for p in (a,b,c)): M_br[i,j,k]=True
print(f'masks built in {time.time()-t0:.1f}s')

# ---------- corpus mass in CHARACTER space (layout maps char->slot) --------------------
CI={c:i for i,c in enumerate(C30M)}
def pack2(freqs):
    ii,jj,ff=[],[],[]
    for g,f in freqs.items():
        if len(g)==2 and g[0] in CI and g[1] in CI:
            ii.append(CI[g[0]]); jj.append(CI[g[1]]); ff.append(float(f))
    return np.array(ii),np.array(jj),np.array(ff)
def pack3(freqs):
    ii,jj,kk,ff=[],[],[],[]
    for g,f in freqs.items():
        if len(g)==3 and all(ch in CI for ch in g):
            ii.append(CI[g[0]]); jj.append(CI[g[1]]); kk.append(CI[g[2]]); ff.append(float(f))
    return np.array(ii),np.array(jj),np.array(kk),np.array(ff)
B2=pack2(bg); S2=pack2(sg); T3=pack3(tg)
print(f'packed: bigrams {len(B2[2])}  skipgrams {len(S2[2])}  trigrams {len(T3[2])}')

def shares_vec(lay):
    """lay: 30-char string. Returns the 11 shares, vectorized."""
    p=np.array([lay.index(c) for c in C30M])   # char-index -> slot index
    bi,bj,bf=B2; si,sj,sf=S2; ti,tj,tk,tf=T3
    a,b=p[bi],p[bj]
    tot2=bf.sum()
    out={}
    out['sfb']=bf[M_sfb[a,b]].sum(); out['alternate']=bf[M_alt[a,b]].sum()
    out['lsb']=bf[M_lsb[a,b]].sum(); out['scissor']=bf[M_sci[a,b]].sum()
    out['inroll']=bf[M_in[a,b]].sum(); out['outroll']=bf[M_out[a,b]].sum()
    hl=np.zeros(3)
    for pos in (a,b):
        h=HAND[pos]
        hl[0]+=bf[h==-1].sum()/2; hl[2]+=bf[h==1].sum()/2
    sa,sb=p[si],p[sj]
    out['dsfb']=100.0*sf[M_dsfb[sa,sb]].sum()/sf.sum()
    ta,tb,tc=p[ti],p[tj],p[tk]; tot3=tf.sum()
    out['onehand']=100.0*tf[M_oh[ta,tb,tc]].sum()/tot3
    out['redirect']=100.0*tf[M_rd[ta,tb,tc]].sum()/tot3
    out['bad_redirect']=100.0*tf[M_br[ta,tb,tc]].sum()/tot3
    for k in ('sfb','alternate','lsb','scissor','inroll','outroll'): out[k]=100.0*out[k]/tot2
    out['imbalance']=100.0*abs(hl[0]-hl[2])/(hl[0]+hl[2])
    return out

# ---------- POSITIVE CONTROL vs the shipped scorer ------------------------------------
sc=OxeyStyleScorer(bg,sg,tg)
rng=random.Random(20260728)
def rl():
    ch=list(C30M); rng.shuffle(ch); return ''.join(ch)
CTRL=['qwerty' and ''.join(C30M)]+[rl() for _ in range(4)]
worst=0.0
for lay in CTRL:
    ref=sc.pattern_shares(Layout(lay,G)); mine=shares_vec(lay)
    for t in TERMS:
        d=abs(ref[t]-mine[t]); worst=max(worst,d)
        if d>1e-9: print(f'  MISMATCH {t}: shipped={ref[t]:.10f} mine={mine[t]:.10f} d={d:.3g}')
print(f'POSITIVE CONTROL vectorized shares vs OxeyStyleScorer.pattern_shares: '
      f'{len(CTRL)} layouts x {len(TERMS)} terms, max abs diff = {worst:.3g}')
assert worst<1e-9, 'vectorized path does NOT reproduce the shipped scorer'
np.save('/tmp/scissorprice/probe/_ctrl_ok.npy', np.array([worst]))
POOL_N=400
pool=[rl() for _ in range(POOL_N)]
t0=time.time()
X=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
print(f'share matrix {X.shape} in {time.time()-t0:.1f}s  (random-permutation pool, corpus={CD.name})')
np.save('/tmp/scissorprice/probe/_X_random.npy', X)
with open('/tmp/scissorprice/probe/_pool_random.json','w') as f: json.dump(pool,f)
print('saved X_random.npy + pool_random.json')
