"""Trap 43: reconcile the onehand-vs-redirect gap by READING what varies, not tolerating it."""
import sys, numpy as np
from collections import defaultdict
sys.path.insert(0,'/tmp/scissorprice/probe')
import matched_prices as M
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.analysis.community import _v1_pattern
TH='/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1'
POS=list(G.slots); IX=M.IX
_L={5:0,4:1,3:2,2:3,1:3}; _R={1:6,2:6,3:7,4:8,5:9}
def fenum(p): return _L[abs(p[0])] if p[0]<0 else _R[abs(p[0])]
def v1(a,b,c): return _v1_pattern(fenum(a),fenum(b),fenum(c))
T2={'AALTO':np.load(f'{TH}/T2_prod.npy'),'COMMUNITY':np.load(f'{TH}/T2_comm.npy'),'POOL':np.load(f'{TH}/T2_pool.npy')}
Tc={'AALTO':np.load(f'{TH}/Tc_aalto.npy'),'COMMUNITY':np.load(f'{TH}/Tc_comm.npy'),'POOL':np.load(f'{TH}/Tc_pool.npy')}
SURF={k:T2[k][:,:,None]+Tc[k] for k in T2}
def matched3(S,mem,non,strata,tri):
    cells=defaultdict(lambda:([],[]))
    for t in tri:
        k=strata(t)
        if k is None: continue
        v=S[IX[t[0]],IX[t[1]],IX[t[2]]]
        if mem(t): cells[k][0].append(v)
        elif non(t): cells[k][1].append(v)
    num=den=0.0; ds=[]; ws=[]
    for m_,n_ in cells.values():
        if not m_ or not n_: continue
        d=float(np.mean(m_)-np.mean(n_)); w=float(min(len(m_),len(n_)))
        num+=w*d; den+=w; ds.append(d); ws.append(w)
    if den==0: return None
    ds=np.array(ds); ws=np.array(ws)
    return dict(delta_ms=num/den,n_strata=len(ds),frac_pos=float(np.average(ds>0,weights=ws)))
TRI_ND=[(a,b,c) for a in POS for b in POS for c in POS]
TRI_D =[(a,b,c) for a in POS for b in POS for c in POS if a!=b and b!=c]
STR={'land(b),land(c)':lambda t:(M.land_sig(t[1]),M.land_sig(t[2])),
     'land(c) only':   lambda t: M.land_sig(t[2]),
     'land(b),land(c),land(a)':lambda t:(M.land_sig(t[0]),M.land_sig(t[1]),M.land_sig(t[2]))}
REDVARIANTS={
 'redirects+bad+both_sfs (ALL)': lambda t: (v1(*t) or '').startswith(('redirect','bad_redirect')),
 'redirects only (no sfs, no bad)': lambda t: v1(*t)=='redirects',
 'redirects + redirects_sfs':    lambda t: (v1(*t) or '') in ('redirects','redirects_sfs'),
 'redirects + bad_redirects (no sfs)': lambda t: (v1(*t) or '') in ('redirects','bad_redirects'),
}
OH=lambda t: (v1(*t) or '')=='onehands'
print(f"{'redirect def':36s} {'strata':26s} {'triples':8s}  AALTO    COMM     POOL    verdict")
for rn,rp in REDVARIANTS.items():
    for sn,sf in STR.items():
        for tn,tri in (('nodup',TRI_ND),('a!=b,b!=c',TRI_D)):
            vals=[]
            for src in ('AALTO','COMMUNITY','POOL'):
                r=matched3(SURF[src],OH,rp,sf,tri)
                vals.append(r['delta_ms'] if r else None)
            if any(v is None for v in vals): continue
            hit = all(v>0 for v in vals)
            near = abs(vals[0]-5.8)<1.0 and abs(vals[1]-3.2)<1.0 and abs(vals[2]-7.3)<1.0
            mark = '  <<< MATCHES PUBLISHED' if near else ('  all-positive' if hit else '')
            print(f'{rn:36s} {sn:26s} {tn:9s} {vals[0]:+7.2f} {vals[1]:+7.2f} {vals[2]:+7.2f}{mark}')
print('\nPUBLISHED (THEORY-1 report.md): onehand vs redirect = +5.8 / +3.2 / +7.3')
