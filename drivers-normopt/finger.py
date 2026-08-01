"""Per-finger TIME distribution shift (from analyze --attribution)."""
import sys, json, statistics as st
sys.path.insert(0,"/tmp/normopt/src")
V=json.load(open("/tmp/normopt/runs/verdict.json")); AN=json.load(open("/tmp/normopt/runs/analyze-all.json"))
rows=AN["rows"]
def row(lay):
    if lay in rows: return rows[lay]
    return next(v for v in rows.values() if v.get("layout")==lay)
P,F,W=V["produced"],V["field"],V["winners"]
print("attribution keys:", list(row(W["A"])["attribution"].keys()))
att=row(W["A"])["attribution"]
pf=att.get("finger_time_pct")
print("per-finger sample:", json.dumps(pf, indent=1)[:600] if pf else None)
FING=list(pf.keys()) if isinstance(pf,dict) else []
def pfof(lay):
    a=row(lay)["attribution"]
    d=a.get("finger_time_pct")
    tot=sum(d.values())
    return {k: 100.0*v/tot for k,v in d.items()}
print("\n"+"="*104); print("H) PER-FINGER TIME SHARE (% of total predicted ms) — arm means over 10 seeds"); print("="*104)
order=[f for f in FING]
print(f"{'':10}"+"".join(f"{f:>9}" for f in order))
arms={}
for a in "ABC":
    ks=[k for k in P if P[k]['arm']==a]
    vals={f:[pfof(P[k]["layout"])[f] for k in ks] for f in order}
    arms[a]=vals
    print(f"arm {a:6}"+"".join(f"{st.mean(vals[f]):9.3f}" for f in order))
    print(f"   (sd)  "+"".join(f"{st.stdev(vals[f]):9.3f}" for f in order))
print("\n  shift in sd(arm A) units:")
for a in "BC":
    print(f"   {a}-A  "+"".join(f"{(st.mean(arms[a][f])-st.mean(arms['A'][f]))/max(st.stdev(arms['A'][f]),1e-9):+9.2f}" for f in order))
print("\n  field reference:")
for n in ("keybo-lsb","keybo-c30m","qwerty30m"):
    d=pfof(F[n]["layout"]); print(f"   {n:12}"+"".join(f"{d[f]:9.3f}" for f in order))
