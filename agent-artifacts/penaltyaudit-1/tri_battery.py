"""TRIGRAM-term matched prices: onehand, redirect, bad_redirect, dsfb (+ alternate at tri level).

FRAME (label every number):
  * ROW_STAGGERED_30's 30 letter slots; SPACE EXCLUDED (thumb, hand()==0 pollutes ALTERNATE).
  * ordered triples with a!=b and b!=c (a==c IS allowed -- that is the sfs/skipgram case,
    which is exactly what `dsfb` prices; it is flagged separately).
  * ms entries of the NATIVE per-source 31x31x31 surface S[a,b,c] (g-frame only, no b(ngram),
    baked 90 WPM). For the CONDITIONAL-only view we also price Tc = S - S.mean(axis=2,keepdims)
    is NOT used; instead THEORY-1's Tc_* tensors are priced directly where relevant.
  * MATCHING: the strongest control available is the pair of LANDING SIGNATURES of b and c
    (row,finger,lateral each) PLUS the a->b bigram class. THEORY-1 showed the (b,c)-exact
    control is the one that flips the redirect sign, so both levels are reported.
"""
import json, os, sys, numpy as np
from collections import defaultdict
sys.path.insert(0,'/tmp/penaudit/probe')
import matched_prices as M
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.analysis.community import _v1_pattern

NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
POS=list(G.slots); IX=M.IX

# ---- oxey's OWN trigram predicates, lifted verbatim from scoring/oxey.py -------------
def oxey_tri_class(a,b,c):
    """EXACTLY scoring/oxey.py's trigram branch (lines ~136-148)."""
    ha,hb,hc = G.hand(a[0]),G.hand(b[0]),G.hand(c[0])
    if not (ha==hb==hc and ha!=0): return None
    d1=abs(b[0])-abs(a[0]); d2=abs(c[0])-abs(b[0])
    if d1 and d2 and (d1>0)==(d2>0): return "onehand"
    if d1 and d2:
        if not any(abs(p[0]) in (1,2) for p in (a,b,c)): return "bad_redirect"
        return "redirect"
    return None

def is_alt_tri(a,b,c):
    """oxey counts `alternate` at the BIGRAM level (a,b). Tri-level alternation for contrast."""
    ha,hb,hc=G.hand(a[0]),G.hand(b[0]),G.hand(c[0])
    return ha!=0 and hb!=0 and hc!=0 and ha!=hb and hb!=hc

def is_dsfb(a,b,c):
    """oxey's `dsfb`: same finger on the SKIPGRAM (a,c), distinct keys. (oxey.py:126-129
    applies g.same_finger(a,b) to the SKIPGRAM table, i.e. the (1st,3rd) of a trigram.)"""
    return G.same_finger(a[0],c[0]) and a!=c

def land(p): return M.land_sig(p)
def abclass(a,b): return C.classify_positions(G,a,b).value

# strata definitions, weakest -> strongest
STRATA={
 'land(b),land(c)':            lambda t: (land(t[1]), land(t[2])),
 'land(b),land(c),class(a,b)': lambda t: (land(t[1]), land(t[2]), abclass(t[0],t[1])),
 '(b,c) EXACT':                lambda t: (t[1], t[2]),
 '(b,c) EXACT,class(a,b)':     lambda t: (t[1], t[2], abclass(t[0],t[1])),
}

def matched_boot3(S, member, nonmember, strata, triples, nboot=3000, seed=0):
    cells=defaultdict(lambda:([],[]))
    for t in triples:
        k=strata(t)
        if k is None: continue
        v=S[IX[t[0]],IX[t[1]],IX[t[2]]]
        if member(*t): cells[k][0].append(v)
        elif nonmember(*t): cells[k][1].append(v)
    ds,ws=[],[]
    for mem,non in cells.values():
        if not mem or not non: continue
        ds.append(float(np.mean(mem)-np.mean(non))); ws.append(float(min(len(mem),len(non))))
    if not ds: return None
    ds=np.array(ds); ws=np.array(ws)
    rng=np.random.default_rng(seed); n=len(ds); bs=np.empty(nboot)
    for i in range(nboot):
        ix=rng.integers(0,n,n); w=ws[ix]; bs[i]=(w*ds[ix]).sum()/w.sum()
    return dict(delta_ms=float((ws*ds).sum()/ws.sum()),
                ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
                n_strata=n, frac_pos=float(np.average(ds>0,weights=ws)),
                p10=float(np.percentile(ds,10)), p90=float(np.percentile(ds,90)))

# ---- triple universe -----------------------------------------------------------------
triples=[(a,b,c) for a in POS for b in POS for c in POS if a!=b and b!=c]
print(f'triple universe (a!=b, b!=c, space excluded): {len(triples)}')
n_oh=sum(1 for t in triples if oxey_tri_class(*t)=='onehand')
n_rd=sum(1 for t in triples if oxey_tri_class(*t)=='redirect')
n_br=sum(1 for t in triples if oxey_tri_class(*t)=='bad_redirect')
n_al=sum(1 for t in triples if is_alt_tri(*t))
n_ds=sum(1 for t in triples if is_dsfb(*t))
print(f'  onehand {n_oh}  redirect {n_rd}  bad_redirect {n_br}  alt(tri) {n_al}  dsfb(skip a==c-finger) {n_ds}')
# trap 16 detector on the headline contrast
kOH={p for t in triples if oxey_tri_class(*t)=='onehand' for p in t}
kAL={p for t in triples if is_alt_tri(*t) for p in t}
print(f'  key-set overlap onehand vs alt(tri): {len(kOH & kAL)} (empty => NOT identified)')
# THEORY-1 D7: can onehand and alternate EVER share a stratum on (b,c)?
sOH={(t[1],t[2]) for t in triples if oxey_tri_class(*t)=='onehand'}
sAL={(t[1],t[2]) for t in triples if is_alt_tri(*t)}
print(f'  (b,c) strata shared by onehand & alt(tri): {len(sOH & sAL)}  <-- THEORY-1 D7 says 0 STRUCTURAL')

TESTS={
 'onehand vs alternate(tri)':      (lambda a,b,c: oxey_tri_class(a,b,c)=='onehand', is_alt_tri),
 'onehand vs redirect':            (lambda a,b,c: oxey_tri_class(a,b,c)=='onehand',
                                    lambda a,b,c: oxey_tri_class(a,b,c) in ('redirect','bad_redirect')),
 'redirect(any) vs alternate(tri)':(lambda a,b,c: oxey_tri_class(a,b,c) in ('redirect','bad_redirect'), is_alt_tri),
 'bad_redirect vs redirect':       (lambda a,b,c: oxey_tri_class(a,b,c)=='bad_redirect',
                                    lambda a,b,c: oxey_tri_class(a,b,c)=='redirect'),
 'dsfb vs non-dsfb (skipgram)':    (is_dsfb, lambda a,b,c: not is_dsfb(a,b,c)),
}
SURF={}
for pool in ('AALTO','COMMUNITY','POOL'):
    for fam in ('BASE','TRI_PS_FREQ_PRIOR'):
        p=f'{NAT}/{pool}_{fam}.native.npy'
        if os.path.exists(p): SURF[f'{pool}_{fam}']=np.load(p)
print('surfaces:',list(SURF))
res={}
for tname,(mem,non) in TESTS.items():
    res[tname]={}
    for sname,strat in STRATA.items():
        res[tname][sname]={}
        for src,S in SURF.items():
            res[tname][sname][src]=matched_boot3(S,mem,non,strat,triples)
    print(f'\n=== {tname} ===')
    print(f"  {'strata':28s}"+''.join(f'{s[:15]:>17s}' for s in SURF))
    for sname in STRATA:
        line=f'  {sname:28s}'
        for src in SURF:
            r=res[tname][sname][src]
            line+=f"{r['delta_ms']:+9.2f}/{r['n_strata']:<7d}" if r else f"{'-- DISJOINT':>17s}"
        sg={np.sign(res[tname][sname][s]['delta_ms']) for s in SURF if res[tname][sname][s]}
        line+='  AGREE' if len(sg)==1 else '  **SPLIT**'
        print(line)
json.dump({'counts':dict(onehand=n_oh,redirect=n_rd,bad_redirect=n_br,alt_tri=n_al,dsfb=n_ds,
           triples=len(triples), bc_strata_shared_onehand_alt=len(sOH & sAL)),
           'results':res}, open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/tri_battery.json','w'), indent=1, default=float)
print('\nwrote tri_battery.json')
