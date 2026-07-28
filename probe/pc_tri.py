"""Reconcile: THEORY-1 priced _v1_pattern (oxeylyzer-1 port) classes on T2+Tcond.
`DEFAULT_OXEY_WEIGHTS` is consumed by OxeyStyleScorer, which uses a DIFFERENT classifier.
Quantify the disagreement, then positive-control THEORY-1's number with ITS classifier."""
import sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0,'/tmp/scissorprice/probe')
import matched_prices as M
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.analysis.community import _v1_pattern, FINGERS, SLOT2DOF
TH='/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1'
POS=list(G.slots); IX=M.IX
_L={5:0,4:1,3:2,2:3,1:3}; _R={1:6,2:6,3:7,4:8,5:9}
def fenum(p): return _L[abs(p[0])] if p[0]<0 else _R[abs(p[0])]
# positive control on the enum map (THEORY-1's own assert)
for slot,p in enumerate(G.slots):
    assert fenum(p)==FINGERS[SLOT2DOF[slot]], (slot,p)
print('enum-map positive control: PASS (matches community.FINGERS[SLOT2DOF])')

def oxey_py_class(a,b,c):
    """scoring/oxey.py's OWN trigram branch -- the classifier that consumes DEFAULT_OXEY_WEIGHTS."""
    ha,hb,hc=G.hand(a[0]),G.hand(b[0]),G.hand(c[0])
    if not(ha==hb==hc and ha!=0): return None
    d1=abs(b[0])-abs(a[0]); d2=abs(c[0])-abs(b[0])
    if d1 and d2 and (d1>0)==(d2>0): return 'onehand'
    if d1 and d2: return 'bad_redirect' if not any(abs(p[0]) in (1,2) for p in (a,b,c)) else 'redirect'
    return None
def v1_class(a,b,c):
    r=_v1_pattern(fenum(a),fenum(b),fenum(c))
    if r is None: return None
    if r.startswith('onehand'): return 'onehand'
    if r.startswith('bad_redirect'): return 'bad_redirect'
    if r.startswith('redirect'): return 'redirect'
    if r.startswith('alternate'): return 'alternate'
    if r.startswith('inroll'): return 'inroll'
    if r.startswith('outroll'): return 'outroll'
    return r
triples=[(a,b,c) for a in POS for b in POS for c in POS]
print(f'\n=== CLASSIFIER DISAGREEMENT (all {len(triples)} ordered triples incl. repeats) ===')
cm=Counter((oxey_py_class(*t), v1_class(*t)) for t in triples)
for cls in ('onehand','redirect','bad_redirect'):
    n_ox=sum(v for (o,v_),v in cm.items() if o==cls)
    n_v1=sum(v for (o,v_),v in cm.items() if v_==cls)
    agree=cm.get((cls,cls),0)
    print(f'  {cls:14s} oxey.py={n_ox:6d}  _v1_pattern={n_v1:6d}  BOTH={agree:6d}  '
          f'-> oxey.py-only={n_ox-agree}, v1-only={n_v1-agree}')
print('  where oxey.py says onehand, _v1_pattern says:',
      Counter(v1_class(*t) for t in triples if oxey_py_class(*t)=='onehand').most_common())
print('  where _v1_pattern says onehands, oxey.py says:',
      Counter(oxey_py_class(*t) for t in triples if v1_class(*t)=='onehand').most_common())

print('\n=== POSITIVE CONTROL: THEORY-1 onehand-vs-* with ITS classifier on T2+Tcond ===')
T2={'AALTO':np.load(f'{TH}/T2_prod.npy'),'COMMUNITY':np.load(f'{TH}/T2_comm.npy'),'POOL':np.load(f'{TH}/T2_pool.npy')}
Tc={'AALTO':np.load(f'{TH}/Tc_aalto.npy'),'COMMUNITY':np.load(f'{TH}/Tc_comm.npy'),'POOL':np.load(f'{TH}/Tc_pool.npy')}
def matched3(S, mem, non, strata, tri):
    cells=defaultdict(lambda:([],[]))
    for t in tri:
        k=strata(t)
        v=S[IX[t[0]],IX[t[1]],IX[t[2]]]
        if mem(*t): cells[k][0].append(v)
        elif non(*t): cells[k][1].append(v)
    num=den=0.0; ds=[]
    for m_,n_ in cells.values():
        if not m_ or not n_: continue
        d=float(np.mean(m_)-np.mean(n_)); w=float(min(len(m_),len(n_)))
        num+=w*d; den+=w; ds.append(d)
    if den==0: return None
    return dict(delta_ms=num/den,n_strata=len(ds),frac_pos=float(np.mean(np.array(ds)>0)))
LANDBC=lambda t:(M.land_sig(t[1]),M.land_sig(t[2]))
tri_nodup=[(a,b,c) for a in POS for b in POS for c in POS]
for label,(mem,non) in {
  'onehand vs alternate':(lambda a,b,c:v1_class(a,b,c)=='onehand', lambda a,b,c:v1_class(a,b,c)=='alternate'),
  'onehand vs redirect(any)':(lambda a,b,c:v1_class(a,b,c)=='onehand', lambda a,b,c:v1_class(a,b,c) in('redirect','bad_redirect')),
  'onehand vs redirect(plain)':(lambda a,b,c:v1_class(a,b,c)=='onehand', lambda a,b,c:v1_class(a,b,c)=='redirect'),
  'redirect(any) vs alternate':(lambda a,b,c:v1_class(a,b,c) in('redirect','bad_redirect'), lambda a,b,c:v1_class(a,b,c)=='alternate'),
  'bad_redirect vs redirect':(lambda a,b,c:v1_class(a,b,c)=='bad_redirect', lambda a,b,c:v1_class(a,b,c)=='redirect'),
}.items():
    out=[]
    for src in ('AALTO','COMMUNITY','POOL'):
        S=T2[src][:,:,None]+Tc[src]
        r=matched3(S,mem,non,LANDBC,tri_nodup)
        out.append(f"{src[:4]}={r['delta_ms']:+7.2f}({r['n_strata']}str,{100*r['frac_pos']:.0f}%)" if r else f'{src[:4]}=DISJOINT')
    print(f'  {label:28s} '+'  '.join(out))
print("\n  THEORY-1 report says: onehand vs alternating +37.2/+89.5/+52.6 (93/95/89% strata)")
print("                       onehand vs redirect    +5.8/+3.2/+7.3")
print("                       bad_redirect vs redir  +22.2/+5.9/+11.1")
print("                       redirect vs rolls(class) +21.7/+41.1/+23.5 (90% strata)")
