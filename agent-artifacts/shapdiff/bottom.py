"""WHY does `bottom` dominate? It is a LANDING-KEY row one-hot. Test the mechanism:
which characters moved rows between the two boards, and does the bottom-row corpus mass differ?"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="2"
import keybo, numpy as np
print("keybo:",keybo.__file__)
from keybo.analysis.shap_diff import shap_diff
from keybo.geometry import ROW_STAGGERED_30
from keybo.data.corpus import load_frequencies, production_corpus_dir
F="pyou'vgdnmheai.cstrlkjz,-wfbxq"; G="bldwz'foujnrtsgyhaeixqmcvkp,.-"
geom=ROW_STAGGERED_30
# row of each slot: y==3 top, 2 home, 1 bottom (per ngram.py)
rows=[geom.slots[i][1] for i in range(30)]
ROW={3:"top",2:"home",1:"bottom"}
print("row layout of the 30 slots:", [ROW[r] for r in rows][:10],"...")
fr={c:ROW[rows[i]] for i,c in enumerate(F)}
gr={c:ROW[rows[i]] for i,c in enumerate(G)}
moved=[(c,fr[c],gr[c]) for c in sorted(set(F)) if fr[c]!=gr[c]]
print(f"\n{len(moved)} of 30 chars change ROW between the boards:")
for c,a,b in moved: print(f"   {c!r}: flagship {a:<7} -> graphite {b}")
# corpus mass landing on each row (weight = trigram marginal on the SECOND char, matching the frame)
tri=load_frequencies(str(production_corpus_dir(None)/"trigrams.txt"))
for tag,rmap,lay in (("flagship-c3",fr,F),("graphite",gr,G)):
    on=set(lay)|{" "}
    mass={"top":0.0,"home":0.0,"bottom":0.0,"space":0.0}; tot=0.0
    for ng,f in tri.items():
        if len(ng)!=3 or any(ch not in on for ch in ng): continue
        tot+=f
        second=ng[1]
        mass["space" if second==" " else rmap[second]]+=f
    print(f"\n{tag}: corpus mass by LANDING-KEY row (T2's second key), share of covered:")
    for k in ("top","home","bottom","space"):
        print(f"   {k:<7} {100*mass[k]/tot:7.3f}%")
d=shap_diff(F,G,name_a="flagship-c3",name_b="graphite")
print(f"\nbottom contribution: {[c.ms_per_char for c in d.contributions if c.feature=='bottom'][0]:+.4f} ms/char")
print("ALL bottom top-bigrams by |ms/char| (top 14):")
for bg,v in d.top_bigrams("bottom",14): print(f"   {bg!r} {v:+.5f}")
# how much of `bottom` is SPACE-initiated (i.e. ' x' bigrams)?
allb=d.top_bigrams("bottom",10**6)
sp=sum(v for bg,v in allb if bg.startswith("␣")); rest=sum(v for bg,v in allb if not bg.startswith("␣"))
print(f"\nbottom: space-initiated bigrams sum {sp:+.5f} | all others {rest:+.5f}")
