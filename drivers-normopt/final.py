"""FINAL comparison with the CORRECTED control (A2 = control on the reported gauge)."""
import sys, json, statistics as st
sys.path.insert(0,"/tmp/normopt/src")
import numpy as np
from keybo.scoring import model_norm as MN
V=json.load(open("/tmp/normopt/runs/verdict.json"))
A2=json.load(open("/tmp/normopt/runs/armA2.json"))
P,F=V["produced"],V["field"]
FLOOR=0.135; SEARCH_FLOOR=0.883   # SEARCHPARAMS-1: design-matched SEARCH-seed scale
anchors=MN.Anchors.read("/tmp/normopt/drivers-normopt/../drivers-normgauge/anchors.json")
fits=MN.SurfaceFits()
SB=MN.BlendSpec(weights={"AALTO":.5411,"COMMUNITY":.3977,"POOL":.0612})
SC=MN.BlendSpec(weights={"AALTO":.5,"COMMUNITY":.5})
for k,v in A2.items():
    n=anchors.normalize_many(fits.fit_of(v["layout"]))
    v.update({"ms":v["ms_per_char"],"bl_c":SB.blend(n),"bl_50":SC.blend(n),
              "aalto_n":n["AALTO"],"comm_n":n["COMMUNITY"],"pool_n":n["POOL"],"arm":"A2"})
ALL={**{k:v for k,v in P.items()}, **A2}
def arm(a): return sorted([k for k in ALL if ALL[k]["arm"]==a], key=lambda k:ALL[k]["seed"])

print("="*106)
print("FINAL — FOUR arms. A2 is the CONTROL ON THE RULER IT IS REPORTED ON (parity 1.2e-14).")
print("A  = shipped `optimize --ngram bigram` (SEARCHPARAMS-1: this ruler is INVERTED vs the gauge)")
print("A2 = same shipped annealer + 2-opt, same defaults, objective = the REPORTED gauge")
print("="*106)
print(f"{'arm':4}{'best ms/char':>14}{'median':>11}{'sd':>9}{'range':>9}   best layout")
for a in ("A","A2","B","C"):
    ks=arm(a); ms=[ALL[k]["ms"] for k in ks]
    bk=min(ks,key=lambda k:ALL[k]["ms"])
    print(f"{a:4}{min(ms):14.6f}{st.median(ms):11.6f}{st.stdev(ms):9.6f}{max(ms)-min(ms):9.6f}   {ALL[bk]['layout']!r}")

# winners on their OWN objective (preregistered)
OBJ={"A":lambda d:-d["ms"],"A2":lambda d:-d["ms"],"B":lambda d:d["bl_c"],"C":lambda d:d["bl_50"]}
W={a:max([ALL[k] for k in arm(a)],key=OBJ[a]) for a in ("A","A2","B","C")}
print("\n--- WINNERS (best-of-10 on its OWN objective) ---")
for a in ("A","A2","B","C"):
    d=W[a]; print(f"  {a:3} seed {d['seed']}  ms/char {d['ms']:.6f}  blend(c) {d['bl_c']:.6f}  blend50 {d['bl_50']:.6f}  {d['layout']!r}")

print("\n--- BOTH DIRECTIONS vs the CORRECTED control A2 ---")
sdp=st.mean([st.stdev([ALL[k]["ms"] for k in arm(a)]) for a in ("A2","B","C")])
for a in ("B","C"):
    d=W[a]["ms"]-W["A2"]["ms"]
    print(f"  ms/char: {a} winner MINUS A2 winner = {d:+.6f}  ({'WORSE' if d>0 else 'BETTER'})  "
          f"{abs(d)/FLOOR:.2f}x model-seed floor(0.135) | {abs(d)/SEARCH_FLOOR:.2f}x search-seed floor(0.883) | {abs(d)/sdp:.2f}x within-sd")
for a,key in (("B","bl_c"),("C","bl_50")):
    d=W["A2"][key]-W[a][key]
    print(f"  {'blend(c)' if a=='B' else 'blend50 '}: A2 winner MINUS {a} winner = {d:+.6f}  (A2 is {'WORSE' if d<0 else 'BETTER'} on it)")
# and A2 on the blend vs the OLD A
print(f"\n  A (old, inverted-ruler control) winner blend(c) {W['A']['bl_c']:.6f}  |  A2 winner blend(c) {W['A2']['bl_c']:.6f}")
print(f"  A2 winner ms/char {W['A2']['ms']:.6f} vs A winner {W['A']['ms']:.6f} = {W['A2']['ms']-W['A']['ms']:+.6f} "
      f"({abs(W['A2']['ms']-W['A']['ms'])/FLOOR:.1f}x floor) -- fixing the ruler is worth this much")

print("\n--- BOOTSTRAP best-of-10 (10k), ms/char ---")
rng=np.random.default_rng(20260801); B=10000
boot={}
for a in ("A","A2","B","C"):
    v=np.array([ALL[k]["ms"] for k in arm(a)])
    boot[a]=np.array([v[rng.integers(0,10,10)].min() for _ in range(B)])
    print(f"  {a:3} point {v.min():.6f}  boot mean {boot[a].mean():.6f} sd {boot[a].std():.6f} "
          f"95%CI [{np.percentile(boot[a],2.5):.6f}, {np.percentile(boot[a],97.5):.6f}]")
for x,y in (("B","A2"),("C","A2"),("A2","A")):
    d=boot[x]-boot[y]; pt=np.array([ALL[k]["ms"] for k in arm(x)]).min()-np.array([ALL[k]["ms"] for k in arm(y)]).min()
    print(f"  {x}-{y}: point {pt:+.6f} boot mean {d.mean():+.6f} 95%CI [{np.percentile(d,2.5):+.6f}, {np.percentile(d,97.5):+.6f}] "
          f"P({x} better) {float((d<0).mean()):.3f}  CI excl 0? {'YES' if np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0 else 'NO'}")

def ham(a,b): return sum(1 for x,y in zip(a,b) if x!=y)
print("\n--- HAMMING between winners ---")
for x in ("A","A2","B","C"):
    print("  "+"  ".join(f"{x}-{y}:{ham(W[x]['layout'],W[y]['layout']):2d}" for y in ("A","A2","B","C")))
print("\n--- TASK 4: min Hamming to the field, per arm ---")
fl={k:v["layout"] for k,v in F.items()}
for a in ("A","A2","B","C"):
    hs=[min(ham(ALL[k]["layout"],f) for f in fl.values()) for k in arm(a)]
    near=[(k,min(fl,key=lambda n:ham(ALL[k]["layout"],fl[n])),min(ham(ALL[k]["layout"],f) for f in fl.values())) for k in arm(a)]
    b=min(near,key=lambda t:t[2])
    print(f"  arm {a:3} min {min(hs):2d}  median {st.median(hs):4.1f}  -> closest: {b[0]} is {b[2]} keys from {b[1]}")
json.dump({"all":{k:{kk:vv for kk,vv in v.items()} for k,v in ALL.items()},
           "winners":{a:W[a]["layout"] for a in ("A","A2","B","C")}},
          open("/tmp/normopt/runs/final.json","w"), indent=1, sort_keys=True, default=str)
print("\nwrote final.json")
