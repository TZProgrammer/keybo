"""Which conclusions survive a correction so conservative my prereg gap CANNOT matter?

Bonferroni over EVERY test I ever computed: 10 pairs x 2 stages (n=15, n=25) = 20 tests.
alpha_bonf = 0.05/20 = 0.0025. This dominates Holm, dominates the two-stage inflation, and
needs no assumption about my gray-zone definition. Anything surviving here is safe under any
reading of my preregistration.
"""
import json, sys, itertools
MY_WT="/local/home/zegertho/agent/workspaces/seedtb/wt"; sys.path.insert(0,MY_WT+"/src")
import keybo; assert keybo.__file__.startswith(MY_WT)
import numpy as np
from scipy import stats
ART="/local/home/zegertho/agent/state/seedtb/artifacts"
d=json.load(open(f"{ART}/margins_n25.json")); mspc=d["mspc"]
NAMES=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate"]
ALPHA=0.05/20
print(f"Bonferroni over 20 tests (10 pairs x 2 stages): alpha = {ALPHA:.5f}\n")
print(f"{'pair':<24}{'mean':>10}{'p (n=25)':>11}{'survives?':>11}{'signs':>9}")
surv=[]
for x,y in itertools.combinations(NAMES,2):
    dd=np.array(mspc[x])-np.array(mspc[y]); m=dd.mean(); s=dd.std(ddof=1); nn=len(dd)
    t=m/(s/np.sqrt(nn)); p=float(2*stats.t.sf(abs(t),nn-1))
    ok=p<ALPHA
    if ok: surv.append((x,y,m,p))
    print(f"{x+' vs '+y:<24}{m:>+10.4f}{p:>11.2e}{('YES' if ok else 'no'):>11}"
          f"{str((int((dd>0).sum()),int((dd<0).sum()))):>9}")
print(f"\n{len(surv)} of 10 pairs survive the most conservative possible correction.")
print("\nThe surviving ORDER (only these edges are established):")
for x,y,m,p in sorted(surv,key=lambda r:r[3]):
    fast,slow=(x,y) if m<0 else (y,x)
    print(f"  {fast} FASTER than {slow}   (|Δ| {abs(m):.4f} ms/char, p {p:.2e})")

# Is the surviving edge set a total order? Build it.
print("\n=== TRANSITIVE CLOSURE of the established edges ===")
edges={(x if m<0 else y,y if m<0 else x) for x,y,m,p in surv}
import collections
better=collections.defaultdict(set)
for a,b in edges: better[a].add(b)
for _ in range(5):
    for a in list(better):
        for b in list(better[a]): better[a] |= better[b]
for n in NAMES:
    print(f"  {n:<11} established FASTER than: {sorted(better[n]) if better[n] else '(nothing)'}")
undet=[(x,y) for x,y in itertools.combinations(NAMES,2)
       if y not in better[x] and x not in better[y]]
print(f"\nSTILL UNDETERMINED pairs ({len(undet)}): {[f'{a} vs {b}' for a,b in undet]}")
print("\n=> arm-B is established faster than: " + str(sorted(better["arm-B"])))
print("=> F(2.0)  is established faster than: " + str(sorted(better["F(2.0)"])))
print("=> NEITHER arm-B nor F(2.0) is established faster than the other.")
json.dump({"alpha_bonf":ALPHA,"survivors":[{"a":x,"b":y,"mean":m,"p":p} for x,y,m,p in surv],
           "better_than":{k:sorted(v) for k,v in better.items()},
           "undetermined":[f"{a} vs {b}" for a,b in undet]},
          open(f"{ART}/conservative.json","w"),indent=1)
