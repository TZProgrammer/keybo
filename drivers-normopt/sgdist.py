"""sg_dist: the corpus-weighted mean geometry.distance(a,c) over trigrams — the served
`sg_distance` feature per SGDIST-SHIP-1. The GAUGE code is on unmerged branch sgdist-ship,
so I compute the same quantity directly from the shipped geometry + corpus."""
import sys, json, statistics as st
sys.path.insert(0,"/tmp/normopt/src")
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.analysis import surfaces as S

tri=load_frequencies(str(production_corpus_dir(None)/"trigrams.txt"))
tri={k:v for k,v in tri.items() if len(k)==3}
POS=(*G.slots, G.space_position)
def sg_dist(lay30):
    slot={ch:i for i,ch in enumerate(lay30)}; slot[" "]=len(POS)-1
    num=den=0.0
    for ng,f in tri.items():
        try: a,c = slot[ng[0]], slot[ng[2]]
        except KeyError: continue
        num += f*G.distance(POS[a],POS[c]); den += f
    return num/den, 100.0*den/sum(tri.values())

V=json.load(open("/tmp/normopt/runs/verdict.json"))
P,F,W=V["produced"],V["field"],V["winners"]
print("="*80); print("sg_dist (corpus-weighted mean distance(a,c) over trigrams; lower = tighter skip)"); print("="*80)
out={}
for a in "ABC":
    ks=sorted([k for k in P if P[k]['arm']==a],key=lambda k:P[k]['seed'])
    vals=[]
    for k in ks:
        d,cov=sg_dist(P[k]["layout"]); vals.append(d); out[k]=d
    print(f"  arm {a}: mean {st.mean(vals):.6f} +- {st.stdev(vals):.6f}   min {min(vals):.6f} max {max(vals):.6f}")
print()
for nm,lay in [("A winner",W["A"]),("B winner",W["B"]),("C winner",W["C"])]:
    d,cov=sg_dist(lay); print(f"  {nm:12} sg_dist {d:.6f}  (trigram coverage {cov:.4f}%)")
for n in ["keybo-lsb","keybo-c30m","arm-B","graphite","semimak","qwerty30m"]:
    d,cov=sg_dist(F[n]["layout"]); print(f"  {n:12} sg_dist {d:.6f}")
sdA=st.stdev([out[k] for k in out if k[0]=="A"])
mA=st.mean([out[k] for k in out if k[0]=="A"])
for a in "BC":
    m=st.mean([out[k] for k in out if k[0]==a])
    print(f"\n  arm {a} minus arm A (mean sg_dist) = {m-mA:+.6f}  = {(m-mA)/sdA:+.2f} x sd(arm A)")
json.dump(out, open("/tmp/normopt/runs/sgdist.json","w"), indent=1)
