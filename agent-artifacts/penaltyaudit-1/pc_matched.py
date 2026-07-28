"""Positive control: reproduce THEORY-1's frozen matched_prices.json with the copied estimator."""
import json, numpy as np, sys
sys.path.insert(0, '/tmp/penaudit/probe')
import matched_prices as M
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G
A='/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1'
TABLES={'AALTO':np.load(f'{A}/T2_prod.npy'),'COMMUNITY':np.load(f'{A}/T2_comm.npy'),'POOL':np.load(f'{A}/T2_pool.npy')}
LAND     = lambda ab: M.land_sig(ab[1])
LAND_ROW = lambda ab: (M.land_sig(ab[1]), M.rowspan(ab))
def shb2(ab): return M.shb(ab)
TESTS={
 "same_finger (SFB) vs same-hand 2-finger": (M.sfb, shb2, LAND_ROW),
 "same-hand 2-finger vs alternate hands":   (shb2, M.alt, LAND_ROW),
 "same_finger (SFB) vs alternate hands":    (M.sfb, M.alt, LAND_ROW),
 "row span 1 vs row span 0 (same hand)": (lambda ab: shb2(ab) and M.rowspan(ab)==1, lambda ab: shb2(ab) and M.rowspan(ab)==0, LAND),
 "row span 2 vs row span 0 (same hand)": (lambda ab: shb2(ab) and M.rowspan(ab)==2, lambda ab: shb2(ab) and M.rowspan(ab)==0, LAND),
 "row span 2 vs row span 1 (same hand)": (lambda ab: shb2(ab) and M.rowspan(ab)==2, lambda ab: shb2(ab) and M.rowspan(ab)==1, LAND),
 "row span 1 vs 0 (SFB only)": (lambda ab: M.sfb(ab) and M.rowspan(ab)==1, lambda ab: M.sfb(ab) and M.rowspan(ab)==0, LAND),
 "row span 2 vs 1 (SFB only)": (lambda ab: M.sfb(ab) and M.rowspan(ab)==2, lambda ab: M.sfb(ab) and M.rowspan(ab)==1, LAND),
 "column gap 2 vs 1 (same hand, same row)": (lambda ab: shb2(ab) and M.rowspan(ab)==0 and M.colgap(ab)==2, lambda ab: shb2(ab) and M.rowspan(ab)==0 and M.colgap(ab)==1, LAND),
 "lsb (index/middle stretch) vs same-hand non-lsb": (M.lsb, lambda ab: shb2(ab) and not M.lsb(ab), LAND_ROW),
 "scissor (adj finger, 2 rows) vs adj finger flat": (M.scissor, lambda ab: M.adjacent(ab) and M.rowspan(ab)==0, LAND),
}
frozen=json.load(open(f'{A}/matched_prices.json'))
worst=0.0; n=0; miss=[]
for name,(mem,non,st) in TESTS.items():
    if name not in frozen: miss.append(name); continue
    for src,T in TABLES.items():
        r=M.matched(T,mem,non,st)
        f=frozen[name][src]
        for k in ('delta_ms','n_strata','frac_pos','p10','p90'):
            d=abs(float(r[k])-float(f[k])); worst=max(worst,d); n+=1
            if d>1e-9: print(f'  MISMATCH {name} / {src} / {k}: mine={r[k]} frozen={f[k]} d={d:.3g}')
print(f'POSITIVE CONTROL matched estimator: {n} cells, max abs diff = {worst:.6g}')
if miss: print('names absent from frozen (not compared):', miss)
