"""The preregistered comparison. ms/char from the SHIPPED analyze (the 0.135-floor scale)."""
import sys, json, statistics as st
sys.path.insert(0, "/tmp/normopt/src")
from keybo.scoring import model_norm as MN

FLOOR = 0.135
AN   = json.load(open("/tmp/normopt/runs/analyze-all.json"))
NM   = json.load(open("/tmp/normopt/runs/names.json"))
XS   = json.load(open("/tmp/normopt/runs/crossscore.json"))
rows = AN["rows"]
# rows are keyed by layout string (or registry name for the ref)
def row_of(lay):
    if lay in rows: return rows[lay]
    for k,v in rows.items():
        if v.get("layout")==lay: return v
    raise KeyError(lay)

anchors = MN.Anchors.read("/tmp/normopt/drivers-normgauge/anchors.json")
fits    = MN.SurfaceFits()
SB = MN.BlendSpec(weights={"AALTO":.5411,"COMMUNITY":.3977,"POOL":.0612})
SC = MN.BlendSpec(weights={"AALTO":.5,"COMMUNITY":.5})

def rec(lay):
    r = row_of(lay); n = anchors.normalize_many(fits.fit_of(lay))
    return {"layout":lay, "ms":r["time"]["ms_per_char"], "cov":r["time"]["coverage_pct"],
            "aalto_n":n["AALTO"],"comm_n":n["COMMUNITY"],"pool_n":n["POOL"],
            "bl_c":SB.blend(n),"bl_50":SC.blend(n), "gauges":r["gauges"], "row":r}

P = {k: rec(v) for k,v in NM["produced"].items()}
F = {k: rec(v) for k,v in NM["field"].items()}
for k,v in P.items(): v["arm"]=k[0]; v["seed"]=int(k.split("s")[1])

OBJ = {"A":lambda d:-d["ms"], "B":lambda d:d["bl_c"], "C":lambda d:d["bl_50"]}   # higher=better
ARMS = "ABC"
print("="*100)
print("TASK 2/3 — 30 runs, shipped keybo optimize, hyperparams at DEFAULTS, seeds 0-9")
print("ms/char = shipped `keybo analyze` trigram TimeSurface (the scale the 0.135 floor was measured on)")
print("="*100)
for a in ARMS:
    rs = sorted([v for v in P.values() if v["arm"]==a], key=lambda d:d["seed"])
    print(f"\n--- ARM {a} " + {"A":"ms/char (CONTROL)","B":"normgauge registered (c)","C":"normgauge 50/50"}[a])
    print(f"{'seed':>4} {'layout':32s} {'ms/char':>11} {'blend(c)':>9} {'blend50':>9} {'aalto-n':>8} {'comm-n':>8} {'pool-n':>8}")
    for d in rs:
        print(f"{d['seed']:>4} {d['layout']:32s} {d['ms']:11.6f} {d['bl_c']:9.6f} {d['bl_50']:9.6f} {d['aalto_n']:8.6f} {d['comm_n']:8.6f} {d['pool_n']:8.6f}")
    ms=[d["ms"] for d in rs]
    print(f"     ms/char  min {min(ms):.6f}  med {st.median(ms):.6f}  max {max(ms):.6f}  sd {st.stdev(ms):.6f}  range {max(ms)-min(ms):.6f}")
    for lab,key in (("blend(c)","bl_c"),("blend50","bl_50")):
        v=[d[key] for d in rs]
        print(f"     {lab:8s} min {min(v):.6f}  med {st.median(v):.6f}  max {max(v):.6f}  sd {st.stdev(v):.6f}")

# --- winners, per the PREREGISTERED definition: best on its OWN objective -----
W={}
for a in ARMS:
    rs=[v for v in P.values() if v["arm"]==a]
    W[a]=max(rs, key=OBJ[a])
print("\n"+"="*100); print("WINNERS (preregistered: best-of-10 on its OWN objective)"); print("="*100)
for a in ARMS:
    d=W[a]; print(f"arm {a} seed {d['seed']:>2}  {d['layout']!r}  ms/char {d['ms']:.6f}  blend(c) {d['bl_c']:.6f}  blend50 {d['bl_50']:.6f}")

def ham(a,b): return sum(1 for x,y in zip(a,b) if x!=y)
sd_pool = st.mean([st.stdev([d["ms"] for d in P.values() if d["arm"]==a]) for a in ARMS])
print(f"\npooled within-arm sd(ms/char) = {sd_pool:.6f}   FLOOR = {FLOOR}")
print("\n--- BOTH DIRECTIONS (the honest cost, symmetric) ---")
for a in ("B","C"):
    dms = W[a]["ms"]-W["A"]["ms"]
    print(f"  ms/char : arm {a} winner MINUS arm A winner = {dms:+.6f}  "
          f"({'WORSE' if dms>0 else 'BETTER'} by {abs(dms):.6f}; {abs(dms)/FLOOR:.2f}x floor, {abs(dms)/sd_pool:.2f}x within-sd)")
for a in ("B","C"):
    key = "bl_c" if a=="B" else "bl_50"
    db = W["A"][key]-W[a][key]
    print(f"  {'blend(c)' if a=='B' else 'blend50 '}: arm A winner MINUS arm {a} winner = {db:+.6f}  "
          f"({'A is WORSE on it' if db<0 else 'A is BETTER on it'} by {abs(db):.6f})")
dbc = W["B"]["ms"]-W["C"]["ms"]
print(f"\n  P1 test  B vs C on ms/char = {dbc:+.6f}  ({abs(dbc)/FLOOR:.2f}x floor)")
print("\n--- HAMMING (identity, NOT materiality) ---")
for x,y in (("A","B"),("A","C"),("B","C")):
    print(f"  winner({x}) vs winner({y}): {ham(W[x]['layout'],W[y]['layout'])}/30")
for a in ARMS:
    rs=[v for v in P.values() if v["arm"]==a]
    hs=[ham(rs[i]['layout'],rs[j]['layout']) for i in range(len(rs)) for j in range(i+1,len(rs))]
    print(f"  within arm {a}: median {st.median(hs):.1f}/30  min {min(hs)}  max {max(hs)}")

json.dump({"produced":{k:{kk:vv for kk,vv in v.items() if kk!='row'} for k,v in P.items()},
           "field":{k:{kk:vv for kk,vv in v.items() if kk!='row'} for k,v in F.items()},
           "winners":{a:W[a]["layout"] for a in ARMS},
           "pooled_within_arm_sd_ms":sd_pool,"floor":FLOOR},
          open("/tmp/normopt/runs/verdict.json","w"), indent=1, sort_keys=True)
print("\nwrote verdict.json")
