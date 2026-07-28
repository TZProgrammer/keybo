"""Per-term matched prices with STRATUM-clustered bootstrap CIs, all sources, NATIVE frame.

Routes through THEORY-1's validated `matched_prices.matched` for the point estimate
(positive-controlled bit-exact, 165 cells) and adds a bootstrap CI by resampling STRATA
with the same min-count weighting.
"""
import json, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, '/tmp/penaudit/probe')
import matched_prices as M
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G

NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
TH ="/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1"

def bigram_tables():
    """Per-source BIGRAM tables. Two provenances, both labelled:
    (a) THEORY-1's T2_* (the fitted bigram tensor per source) -- 31x31, the object THEORY-1
        priced. This is the NATIVE per-source bigram tensor.
    (b) the c-marginal of the NATIVE trigram surface, S.mean(2) -- 31x31, includes the
        conditional term's average. Different quantity; reported separately."""
    out={}
    for src,f in (('AALTO','T2_prod'),('COMMUNITY','T2_comm'),('POOL','T2_pool')):
        out[f'{src}|T2native']=np.load(f'{TH}/{f}.npy')
    for src in ('AALTO','COMMUNITY','POOL'):
        for fam in ('BASE','TRI_PS_FREQ_PRIOR'):
            import os
            p=f'{NAT}/{src}_{fam}.native.npy'
            if os.path.exists(p):
                out[f'{src}|{fam}nativeMarg']=np.load(p).mean(axis=2)
    return out

def matched_boot(T, member, nonmember, strata, nboot=4000, seed=0):
    """THEORY-1's estimator + a stratum bootstrap. Returns point estimate identical to
    M.matched (asserted) plus a 95% CI over resampled strata."""
    pairs=[(a,b) for a in M.SLOTS for b in M.SLOTS if a!=b]
    cells=defaultdict(lambda: ([],[]))
    for a,b in pairs:
        k=strata((a,b))
        if k is None: continue
        t=T[M.IX[a],M.IX[b]]
        if member((a,b)): cells[k][0].append(t)
        elif nonmember((a,b)): cells[k][1].append(t)
    ds,ws=[],[]
    for mem,non in cells.values():
        if not mem or not non: continue
        ds.append(float(np.mean(mem)-np.mean(non))); ws.append(float(min(len(mem),len(non))))
    if not ds: return None
    ds=np.array(ds); ws=np.array(ws)
    point=float((ws*ds).sum()/ws.sum())
    ref=M.matched(T,member,nonmember,strata)
    assert abs(point-ref['delta_ms'])<1e-12, (point, ref['delta_ms'])
    rng=np.random.default_rng(seed); n=len(ds); bs=np.empty(nboot)
    for i in range(nboot):
        ix=rng.integers(0,n,n); w=ws[ix]
        bs[i]=(w*ds[ix]).sum()/w.sum()
    return dict(delta_ms=point, ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
                n_strata=n, frac_pos=float(np.average(ds>0,weights=ws)),
                p10=float(np.percentile(ds,10)), p90=float(np.percentile(ds,90)))

LAND     = lambda ab: M.land_sig(ab[1])
LAND_ROW = lambda ab: (M.land_sig(ab[1]), M.rowspan(ab))
LAND_ORIGIN = lambda ab: (M.land_sig(ab[1]), M.origin_sig(ab[0]))
def shb2(ab): return M.shb(ab)
def inr(ab): return C.is_inwards(G,*ab)
def outr(ab): return C.is_outwards(G,*ab)
def roll(ab): return inr(ab) or outr(ab)

# ---- the term battery. Each row names the oxey TERM it prices. -----------------------
BATTERY = {
 # sfb (+12.0)
 "sfb: SFB vs same-hand 2-finger [matched land+rowspan]": ("sfb", M.sfb, shb2, LAND_ROW),
 "sfb: SFB vs alternate [matched land+rowspan]":          ("sfb", M.sfb, M.alt, LAND_ROW),
 # lsb (+3.0)
 "lsb: LSB vs same-hand non-LSB [matched land+rowspan]":  ("lsb", M.lsb, lambda ab: shb2(ab) and not M.lsb(ab), LAND_ROW),
 # scissor (+4.0)
 "scissor: adj-2row vs adj-FLAT [matched land]":          ("scissor", M.scissor, lambda ab: M.adjacent(ab) and M.rowspan(ab)==0, LAND),
 "scissor: adj-2row vs NONadj-2row [matched land]":       ("scissor", M.scissor, lambda ab: shb2(ab) and M.rowspan(ab)==2 and not M.adjacent(ab), LAND),
 # inroll (-2.0) / outroll (-1.0)  -- BIGRAM level
 "inroll: outer_high vs outer_low [matched land]":        ("inroll_vs_outroll", inr, outr, LAND),
 "inroll: outer_high vs outer_low [matched land+rowspan]":("inroll_vs_outroll", inr, outr, LAND_ROW),
 "inroll: outer_high vs same-hand non-roll [land+rows]":  ("inroll", inr, lambda ab: shb2(ab) and not roll(ab), LAND_ROW),
 "outroll: outer_low vs same-hand non-roll [land+rows]":  ("outroll", outr, lambda ab: shb2(ab) and not roll(ab), LAND_ROW),
 "roll(any) vs same-hand non-roll [land+rows]":           ("roll", roll, lambda ab: shb2(ab) and not roll(ab), LAND_ROW),
 # alternate (-0.5)
 "alternate: ALT vs same-hand(any) [matched land]":       ("alternate", M.alt, lambda ab: shb2(ab) or M.sfb(ab), LAND),
 "alternate: ALT vs same-hand 2-finger [matched land]":   ("alternate", M.alt, shb2, LAND),
 "alternate: ALT vs same-hand 2f [matched land+rowspan]": ("alternate", M.alt, shb2, LAND_ROW),
 # imbalance (+1.5) -- the row/finger legs THEORY-1 identified
 "imbalance-leg: land BOTTOM vs land HOME [origin+finger fixed]": ("imbalance_row", None, None, None),
 "imbalance-leg: land PINKY vs land MIDDLE [origin+row fixed]":   ("imbalance_finger", None, None, None),
}

def row_finger_contrast(T, kind):
    """THEORY-1's row/finger legs, re-derived: hold ORIGIN key and the other landing
    coordinate fixed, vary the one under test."""
    cells=defaultdict(lambda: ([],[]))
    for a in M.SLOTS:
        for b in M.SLOTS:
            if a==b: continue
            t=T[M.IX[a],M.IX[b]]
            lr,lf,_=M.land_sig(b)
            if kind=='row':      # bottom(1) vs home(2), origin + landing FINGER fixed
                k=(a,lf)
                if lr==1: cells[k][0].append(t)
                elif lr==2: cells[k][1].append(t)
            else:                # pinky vs middle, origin + landing ROW fixed
                k=(a,lr)
                if lf=='pinky': cells[k][0].append(t)
                elif lf=='middle': cells[k][1].append(t)
    ds,ws=[],[]
    for mem,non in cells.values():
        if not mem or not non: continue
        ds.append(float(np.mean(mem)-np.mean(non))); ws.append(float(min(len(mem),len(non))))
    ds=np.array(ds); ws=np.array(ws)
    rng=np.random.default_rng(0); n=len(ds); bs=np.empty(4000)
    for i in range(4000):
        ix=rng.integers(0,n,n); w=ws[ix]; bs[i]=(w*ds[ix]).sum()/w.sum()
    return dict(delta_ms=float((ws*ds).sum()/ws.sum()),
                ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
                n_strata=n, frac_pos=float(np.average(ds>0,weights=ws)),
                p10=float(np.percentile(ds,10)), p90=float(np.percentile(ds,90)))

TABLES=bigram_tables()
res={}
for name,(term,mem,non,st) in BATTERY.items():
    res[name]={'_term':term}
    for src,T in TABLES.items():
        if mem is None:
            kind='row' if 'BOTTOM' in name else 'finger'
            res[name][src]=row_finger_contrast(T,kind)
        else:
            res[name][src]=matched_boot(T,mem,non,st)

hdr=f"{'contrast':52s}"
KEYS=[k for k in TABLES if k.endswith('T2native')]+[k for k in TABLES if not k.endswith('T2native')]
for src in KEYS: hdr+=f"{src.split('|')[0][:5]+'/'+src.split('|')[1][:9]:>18s}"
print(hdr); print('-'*len(hdr))
for name in BATTERY:
    line=f"{name:52s}"
    for src in KEYS:
        r=res[name][src]
        line+=f"{r['delta_ms']:+9.2f}" if r else f"{'--':>9s}"
        line+=f"{r['n_strata']:>4d}/{100*r['frac_pos']:3.0f}" if r else f"{'':>8s}"
    signs={np.sign(res[name][s]['delta_ms']) for s in KEYS if res[name][s]}
    line+="  AGREE" if len(signs)==1 else "  **SPLIT**"
    print(line)
print("\ncolumns: delta_ms  n_strata/frac_pos%   (native frame, g only, baked 90 WPM)")
json.dump(res, open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/term_battery.json','w'), indent=1, default=float)
print('wrote term_battery.json')
