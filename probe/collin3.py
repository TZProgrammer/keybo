"""Identification analysis of the 11 oxey terms — VECTORIZED, POSITIVE-CONTROLLED.

Trap 28: a hand-rolled reimplementation loses the validation, so the vectorized path is
gated on reproducing `OxeyStyleScorer.pattern_shares` EXACTLY before it is used.

The bug the control caught on the first attempt: SPACE. `Layout.has_key(' ')` is True and
`pattern_shares` therefore counts space-touching n-grams in every DENOMINATOR (trap 9: a
wrong denominator is invisible to a numerator check). Space sits at (0,0) with hand()==0,
so it fires NO geometric predicate but DOES classify as ALTERNATE. Excluding it moved
`alternate` by 17.9 share-points.
"""
import sys, json, numpy as np, random, time
from keybo.scoring.oxey import OxeyStyleScorer, DEFAULT_OXEY_WEIGHTS
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.features import classify as C
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.analysis.surfaces import C30M

CD = production_corpus_dir(None)
bg = load_frequencies(str(CD/'bigrams.txt'))
sg = load_frequencies(str(CD/'1-skip.txt'))
tg = load_frequencies(str(CD/'trigrams.txt'))
TERMS = list(DEFAULT_OXEY_WEIGHTS)

# 31 slots: 0..29 the letter slots, 30 = SPACE at the geometry's space_position.
SLOTS = [*G.slots, G.space_position]
NS = len(SLOTS)                     # 31
SPACE = NS - 1
HAND = np.array([G.hand(p[0]) for p in SLOTS])
assert HAND[SPACE] == 0

t0 = time.time()
M_sfb=np.zeros((NS,NS),bool); M_alt=np.zeros((NS,NS),bool); M_lsb=np.zeros((NS,NS),bool)
M_sci=np.zeros((NS,NS),bool); M_in =np.zeros((NS,NS),bool); M_out=np.zeros((NS,NS),bool)
M_dsfb=np.zeros((NS,NS),bool)
for i,a in enumerate(SLOTS):
    for j,b in enumerate(SLOTS):
        cls = C.classify_positions(G,a,b)
        if cls is C.BigramClass.SAME_FINGER and a!=b: M_sfb[i,j]=True
        elif cls is C.BigramClass.ALTERNATE:          M_alt[i,j]=True
        if C.is_lsb(G,a,b):     M_lsb[i,j]=True
        if C.is_scissor(G,a,b): M_sci[i,j]=True
        if C.is_inwards(G,a,b): M_in[i,j]=True
        if C.is_outwards(G,a,b):M_out[i,j]=True
        if G.same_finger(a[0],b[0]) and a!=b: M_dsfb[i,j]=True   # oxey.py's dsfb test
M_oh=np.zeros((NS,NS,NS),bool); M_rd=np.zeros((NS,NS,NS),bool); M_br=np.zeros((NS,NS,NS),bool)
for i,a in enumerate(SLOTS):
    for j,b in enumerate(SLOTS):
        if HAND[i]==0 or HAND[i]!=HAND[j]: continue
        d1 = abs(b[0])-abs(a[0])
        for k,c in enumerate(SLOTS):
            if HAND[k]!=HAND[i]: continue
            d2 = abs(c[0])-abs(b[0])
            if not (d1 and d2): continue
            if (d1>0)==(d2>0): M_oh[i,j,k]=True
            else:
                M_rd[i,j,k]=True
                if not any(abs(p[0]) in (1,2) for p in (a,b,c)): M_br[i,j,k]=True
print(f'masks over {NS} slots built in {time.time()-t0:.1f}s')

# corpus packed in CHARACTER space; space maps to the SPACE slot for every layout.
CI = {c:i for i,c in enumerate(C30M)}; CI[' '] = SPACE
def pack(freqs, n):
    cols=[[] for _ in range(n)]; f=[]
    for g,v in freqs.items():
        if len(g)!=n or not all(ch in CI for ch in g): continue
        for d in range(n): cols[d].append(CI[g[d]])
        f.append(float(v))
    return [np.array(c) for c in cols]+[np.array(f)]
B2=pack(bg,2); S2=pack(sg,2); T3=pack(tg,3)
print(f'packed bigrams {len(B2[-1])}  skipgrams {len(S2[-1])}  trigrams {len(T3[-1])}')

def shares_vec(lay):
    """The 11 shares for a 30-char layout, vectorized. `p` maps a C30M-space index to a
    slot; the SPACE index maps to the SPACE slot (identity across layouts)."""
    p = np.empty(NS, dtype=int)
    for c in C30M: p[CI[c]] = lay.index(c)
    p[SPACE] = SPACE
    bi,bj,bf = B2; a,b = p[bi],p[bj]; tot2 = bf.sum()
    out = {}
    out['sfb']      = bf[M_sfb[a,b]].sum()
    out['alternate']= bf[M_alt[a,b]].sum()
    out['lsb']      = bf[M_lsb[a,b]].sum()
    out['scissor']  = bf[M_sci[a,b]].sum()
    out['inroll']   = bf[M_in [a,b]].sum()
    out['outroll']  = bf[M_out[a,b]].sum()
    # hand load: HALF the bigram mass per key, and ONLY for keys with hand != 0 (space skipped)
    hL = bf[HAND[a]==-1].sum()/2 + bf[HAND[b]==-1].sum()/2
    hR = bf[HAND[a]== 1].sum()/2 + bf[HAND[b]== 1].sum()/2
    si,sj,sf = S2; sa,sb = p[si],p[sj]
    out['dsfb'] = 100.0*sf[M_dsfb[sa,sb]].sum()/sf.sum()
    ti,tj,tk,tf = T3; ta,tb,tc = p[ti],p[tj],p[tk]; tot3 = tf.sum()
    out['onehand']      = 100.0*tf[M_oh[ta,tb,tc]].sum()/tot3
    out['redirect']     = 100.0*tf[M_rd[ta,tb,tc]].sum()/tot3
    out['bad_redirect'] = 100.0*tf[M_br[ta,tb,tc]].sum()/tot3
    for k in ('sfb','alternate','lsb','scissor','inroll','outroll'):
        out[k] = 100.0*out[k]/tot2
    out['imbalance'] = 100.0*abs(hL-hR)/(hL+hR)
    return out

# ---------------- POSITIVE CONTROL against the shipped scorer -------------------------
sc = OxeyStyleScorer(bg,sg,tg)
rng = random.Random(20260728)
def rl():
    ch=list(C30M); rng.shuffle(ch); return ''.join(ch)
CTRL = [''.join(C30M)] + [rl() for _ in range(6)]
worst=0.0; wt=None
for lay in CTRL:
    ref=sc.pattern_shares(Layout(lay,G)); mine=shares_vec(lay)
    for t in TERMS:
        d=abs(ref[t]-mine[t])
        if d>worst: worst, wt = d, (t, ref[t], mine[t])
print(f'POSITIVE CONTROL vs OxeyStyleScorer.pattern_shares: {len(CTRL)} layouts x {len(TERMS)} '
      f'terms, max abs diff = {worst:.4g}' + (f'  worst={wt}' if worst>1e-9 else ''))
assert worst < 1e-9, f'vectorized path does NOT reproduce the shipped scorer: {wt}'
print('  -> vectorized path IS the shipped scorer. Safe to use on a pool.')

# ---------------- quantify the SPACE effect (the bug the control caught) --------------
sp2 = sum(v for k,v in bg.items() if ' ' in k)/sum(bg.values())
sp3 = sum(v for k,v in tg.items() if ' ' in k)/sum(tg.values())
sps = sum(v for k,v in sg.items() if ' ' in k)/sum(sg.values())
print(f'\nSPACE MASS: bigrams {100*sp2:.2f}%  skipgrams {100*sps:.2f}%  trigrams {100*sp3:.2f}%')
q=shares_vec(''.join(C30M))
print(f'  qwerty-order alternate share = {q["alternate"]:.3f}%  '
      f'of which space-touching is structurally ALL of the {100*sp2:.2f}% space mass')

# ---------------- how much of each term's share is LAYOUT-INVARIANT? ------------------
# A term whose share barely moves across layouts cannot influence an optimizer's choice no
# matter what weight it carries. This is the `sfr`-permutation-invariant check (trap 23)
# generalized to a CONTINUOUS version, tested by SHUFFLING (never via a std>0 filter).
POOL_N = 400
pool = [rl() for _ in range(POOL_N)]
t0=time.time()
X = np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
print(f'\nshare matrix {X.shape} in {time.time()-t0:.1f}s  (random-permutation pool, corpus={CD.name})')
np.save('/tmp/scissorprice/probe/_X_random.npy', X)
json.dump(pool, open('/tmp/scissorprice/probe/_pool_random.json','w'))

print(f'\n{"term":14s}{"weight":>8s}{"mean":>9s}{"sd":>9s}{"min":>9s}{"max":>9s}{"distinct":>9s}'
      f'{"sd/mean":>9s}  contribution to score (w*share)')
W = {k:v[0] for k,v in DEFAULT_OXEY_WEIGHTS.items()}
for i,t in enumerate(TERMS):
    col=X[:,i]; u=len(np.unique(np.round(col,10)))
    cv = col.std()/abs(col.mean()) if col.mean() else float('nan')
    print(f'{t:14s}{W[t]:+8.1f}{col.mean():9.4f}{col.std():9.5f}{col.min():9.4f}{col.max():9.4f}'
          f'{u:9d}{cv:9.4f}   {W[t]*col.mean():+8.3f} +- {abs(W[t])*col.std():.3f}'
          + ('   << INVARIANT' if u<=2 else ''))
