"""Bootstrap the best-of-10 statistic + paired deltas + gauge frame + Task 4."""
import sys, json, statistics as st, itertools
sys.path.insert(0,"/tmp/normopt/src")
import numpy as np
V  = json.load(open("/tmp/normopt/runs/verdict.json"))
XS = json.load(open("/tmp/normopt/runs/crossscore.json"))
AN = json.load(open("/tmp/normopt/runs/analyze-all.json"))
NM = json.load(open("/tmp/normopt/runs/names.json"))
P, F = V["produced"], V["field"]
FLOOR = 0.135
r1 = {k: XS["produced"][k]["ms_per_char"] for k in P}
r2 = {k: P[k]["ms"] for k in P}
def arm(a): return sorted([k for k in P if P[k]["arm"]==a], key=lambda k:P[k]["seed"])

print("="*100); print("A) PAIRED PER-SEED DELTAS (same seed 0-9 in every arm — the paired test)"); print("="*100)
for name,R,lo in (("ruler2 analyze ms/char (FLOOR scale, lower better)",r2,True),
                  ("ruler1 bigram-table ms/char (arm A's search ruler, lower better)",r1,True)):
    print(f"\n{name}")
    for x,y in (("B","A"),("C","A"),("C","B")):
        d=[R[f"{x}-s{s}"]-R[f"{y}-s{s}"] for s in range(10)]
        nneg=sum(1 for v in d if v<0)
        print(f"  {x} minus {y}: mean {st.mean(d):+.6f}  median {st.median(d):+.6f}  sd {st.stdev(d):.6f}  "
              f"sign {nneg}/10 favour {x}   [per-seed { ' '.join(f'{v:+.3f}' for v in d) }]")

print("\n"+"="*100)
print("B) BOOTSTRAP of the BEST-OF-10 statistic (10k resamples of the 10 seeds, with replacement)")
print("   The right null for 'does the objective choice beat search noise?' — comparing MINIMA,")
print("   so the yardstick must be the sampling sd of the MINIMUM, not of the raw draws.")
print("="*100)
rng=np.random.default_rng(20260801)
B=10000
for name,R in (("ruler2 analyze ms/char",r2),("ruler1 bigram-table ms/char",r1)):
    vals={a:np.array([R[k] for k in arm(a)]) for a in "ABC"}
    boot={a:np.array([vals[a][rng.integers(0,10,10)].min() for _ in range(B)]) for a in "ABC"}
    print(f"\n{name}")
    for a in "ABC":
        print(f"  arm {a} best-of-10: point {vals[a].min():.6f}  boot mean {boot[a].mean():.6f}  "
              f"sd {boot[a].std():.6f}  95%CI [{np.percentile(boot[a],2.5):.6f}, {np.percentile(boot[a],97.5):.6f}]")
    for x,y in (("B","A"),("C","A"),("C","B")):
        d=boot[x]-boot[y]
        pt=vals[x].min()-vals[y].min()
        frac=float((d<0).mean())
        print(f"  {x}-{y}: point {pt:+.6f} | boot mean {d.mean():+.6f} sd {d.std():.6f} "
              f"95%CI [{np.percentile(d,2.5):+.6f}, {np.percentile(d,97.5):+.6f}] "
              f"| P({x} better) {frac:.3f} | CI excludes 0? {'YES' if np.percentile(d,2.5)>0 or np.percentile(d,97.5)<0 else 'NO'}"
              + (f" | |pt|/FLOOR {abs(pt)/FLOOR:.2f}x" if R is r2 else ""))

print("\n"+"="*100); print("C) THE 15-GAUGE + sg_dist FRAME — arm winners vs the field"); print("="*100)
rows=AN["rows"]
def row(lay):
    if lay in rows: return rows[lay]
    return next(v for v in rows.values() if v.get("layout")==lay)
W=V["winners"]
GA=["sfr","sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll","redir","scissor","imbalance","oxey-style","comfort"]
def sgd(lay):
    r=row(lay)
    for k in ("sg_dist","sg-dist","sgdist"):
        if k in r: return r[k]
        if k in r.get("gauges",{}): return r["gauges"][k]
    return r.get("sg_distance")
show=[("A win",W["A"]),("B win",W["B"]),("C win",W["C"]),
      ("keybo-lsb",F["keybo-lsb"]["layout"]),("keybo-c30m",F["keybo-c30m"]["layout"]),
      ("arm-B(field)",F["arm-B"]["layout"]),("graphite",F["graphite"]["layout"]),("semimak",F["semimak"]["layout"])]
hdr=f"{'board':13}"+"".join(f"{g:>10}" for g in GA)+f"{'ms/char':>11}"
print(hdr)
for nm,lay in show:
    r=row(lay); g=r["gauges"]
    print(f"{nm:13}"+"".join(f"{g[k]:10.4f}" for k in GA)+f"{r['time']['ms_per_char']:11.4f}")
print("\nsg_dist probe:", {nm: sgd(lay) for nm,lay in show[:3]})
print("available row keys:", sorted(row(W['A']).keys()))
