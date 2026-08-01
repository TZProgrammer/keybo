"""Task 3 systematic character (hand/row/finger + sg_dist) and Task 4 reproduction."""
import sys, json, statistics as st
sys.path.insert(0,"/tmp/normopt/src")
import numpy as np
from keybo.geometry import ROW_STAGGERED_30, Finger
from keybo.analysis import surfaces as S
from keybo.data.corpus import load_frequencies, production_corpus_dir

V=json.load(open("/tmp/normopt/runs/verdict.json")); AN=json.load(open("/tmp/normopt/runs/analyze-all.json"))
rows=AN["rows"]
def row(lay):
    if lay in rows: return rows[lay]
    return next(v for v in rows.values() if v.get("layout")==lay)
P,F,W = V["produced"],V["field"],V["winners"]
G=ROW_STAGGERED_30
slots=G.slots
FING=[G.finger_of(i) if hasattr(G,'finger_of') else None for i in range(30)]
print("geometry attrs:", [a for a in dir(G) if not a.startswith('_')][:25])

# --- monogram mass per key from the bigram table (first-char marginal) ---
bg=load_frequencies(str(production_corpus_dir(None)/"bigrams.txt"))
mono={}
for k,v in bg.items():
    if len(k)==2:
        for ch in k[0]: mono[ch]=mono.get(ch,0)+v
TOT=sum(mono.values())

def profile(lay):
    """hand / row / finger mass shares for one 30-char row-major layout."""
    hand={"L":0.0,"R":0.0}; rowsh=[0.0,0.0,0.0]; fing={}
    for i,ch in enumerate(lay):
        m=mono.get(ch,0)/TOT*100
        s=slots[i]; f=G.finger_at(s) if hasattr(G,'finger_at') else None
        col = i%10
        hand["L" if col<5 else "R"]+=m
        rowsh[i//10]+=m
        if f is not None: fing[f.name if hasattr(f,'name') else str(f)]=fing.get(f.name if hasattr(f,'name') else str(f),0.0)+m
    return hand,rowsh,fing

print("\n"+"="*98); print("E) SYSTEMATIC CHARACTER — monogram mass share by hand / row (corpus-weighted)"); print("="*98)
print(f"{'board':16}{'L%':>8}{'R%':>8}{'|L-R|':>8}{'top%':>8}{'home%':>8}{'bot%':>8}")
def show(nm,lay):
    h,r,f=profile(lay)
    print(f"{nm:16}{h['L']:8.3f}{h['R']:8.3f}{abs(h['L']-h['R']):8.3f}{r[0]:8.3f}{r[1]:8.3f}{r[2]:8.3f}")
    return h,r,f
per_arm={}
for a in "ABC":
    ks=sorted([k for k in P if P[k]['arm']==a],key=lambda k:P[k]['seed'])
    accL=[];accH=[];accIMB=[]
    for k in ks:
        h,r,f=profile(P[k]["layout"]); accL.append(h["L"]); accH.append(r[1]); accIMB.append(abs(h["L"]-h["R"]))
    per_arm[a]=(accL,accH,accIMB)
for nm,lay in [("A winner",W["A"]),("B winner",W["B"]),("C winner",W["C"]),
               ("keybo-lsb",F["keybo-lsb"]["layout"]),("keybo-c30m",F["keybo-c30m"]["layout"]),
               ("arm-B field",F["arm-B"]["layout"]),("qwerty30m",F["qwerty30m"]["layout"])]:
    show(nm,lay)
print("\n  ARM AGGREGATES over 10 seeds (mean +- sd):")
for a in "ABC":
    L,H,I=per_arm[a]
    print(f"   arm {a}: left-hand% {st.mean(L):6.3f}+-{st.stdev(L):.3f} | home-row% {st.mean(H):6.3f}+-{st.stdev(H):.3f} | |L-R| {st.mean(I):6.3f}+-{st.stdev(I):.3f}")

print("\n"+"="*98); print("F) GAUGES: arm means +- sd over 10 seeds (is sfb/roll/alt materially different?)"); print("="*98)
GA=["sfb","sfs","sfb-dist","sfs-dist","lsb","alt","roll","sr-roll","redir","scissor","imbalance","oxey-style","comfort"]
print(f"{'gauge':12}"+"".join(f"{'arm '+a:>18}" for a in "ABC")+"   B-A/sdA   C-A/sdA")
for g in GA:
    v={a:[row(P[k]["layout"])["gauges"][g] for k in P if P[k]['arm']==a] for a in "ABC"}
    line=f"{g:12}"+"".join(f"{st.mean(v[a]):11.4f}+-{st.stdev(v[a]):5.3f}" for a in "ABC")
    sdA=st.stdev(v["A"])
    line+=f"  {(st.mean(v['B'])-st.mean(v['A']))/sdA:+8.2f}  {(st.mean(v['C'])-st.mean(v['A']))/sdA:+8.2f}"
    print(line)

print("\n"+"="*98); print("G) TASK 4 — REPRODUCTION: min Hamming from each produced layout to the field"); print("="*98)
def ham(a,b): return sum(1 for x,y in zip(a,b) if x!=y)
fl={k:v["layout"] for k,v in F.items()}
print(f"{'run':7}{'nearest field board':24}{'Hamming':>9}   {'exact?':>7}")
best_by_arm={a:[] for a in "ABC"}
for a in "ABC":
    for k in sorted([k for k in P if P[k]['arm']==a],key=lambda k:P[k]['seed']):
        lay=P[k]["layout"]
        d={n:ham(lay,f) for n,f in fl.items()}
        n0=min(d,key=d.get)
        best_by_arm[a].append(d[n0])
        print(f"{k:7}{n0:24}{d[n0]:9d}   {'YES' if d[n0]==0 else 'no':>7}")
print("\n  min-Hamming-to-field, per arm (lower = closer to a known board):")
for a in "ABC":
    v=best_by_arm[a]; print(f"   arm {a}: min {min(v)}  median {st.median(v):.1f}  max {max(v)}   (30 = shares no key position)")
print(f"\n  EXACT reproductions of a field board: {sum(1 for a in 'ABC' for v in best_by_arm[a] if v==0)} of 30")
