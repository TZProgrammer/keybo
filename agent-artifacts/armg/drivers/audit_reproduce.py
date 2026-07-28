"""(a) WRONG-CONSTANT-BEHIND-A-TRUE-CONCLUSION SWEEP.
Re-derive every number the ARM G verdict rests on FROM THE RAW RUN JSONS, independently of
armg-judgement.json (which is the artifact under audit). If a number disagrees, the artifact
is wrong even if the conclusion is right."""
import json, glob, subprocess, sys, itertools
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, search as S, evobj as EV
RUNS=Path("/local/home/zegertho/agent/state/armg/artifacts/runs")
J=json.load(open("/local/home/zegertho/agent/state/armg/artifacts/armg-judgement.json"))
A=json.load(open("/local/home/zegertho/agent/state/armg/artifacts/armg-archive-analysis.json"))
G=tuple(S.ARMG_DIR)
ARMB="flmpg-yuo,sntdcireahkxbwv'.jzq"

# --- rebuild from the RAW per-run JSONs, not the summary, not the judgement ---
raw={}
for f in sorted(glob.glob(str(RUNS/"*-r?.json"))):
    b=json.load(open(f)); nm=Path(f).stem
    raw[nm]={"layout":b["champion"]["layout"],"uniq":b["unique_evals"],"top50":[e["layout"] for e in b["top50"]],
             "arm":b["arm"],"seed":b["seed"]}
print(f"raw per-run JSONs read: {len(raw)}")

# score champions through the SHIPPED CLI independently
lays=sorted({v["layout"] for v in raw.values()} | {ARMB})
p=subprocess.run(["uv","run","--no-sync","keybo","analyze","--json",*lays],cwd="/tmp/armg",
                 capture_output=True,text=True)
assert p.returncode==0, p.stderr[-2000:]
rows=json.loads(p.stdout)["rows"]
ms={l:rows[l]["time"]["ms_per_char"] for l in lays}
def D_of(l):
    g=rows[l]["gauges"]
    return sum(max(0.0,S.ARMG_DIR[k]*(g[k]-S.ARMG_REF[k])/S.ARMG_SCALE[k]) for k in G)

fails=[]
def chk(name, mine, published, tol):
    ok=abs(mine-published)<=tol
    print(f"  {name:<34} re-derived {mine!r:<24} published {published!r:<24} {'OK' if ok else '*** MISMATCH ***'}")
    if not ok: fails.append((name,mine,published))

print("\n--- sd_G and the band ---")
base=[v["layout"] for k,v in raw.items() if v["arm"]=="baseline"]
bms=np.array([ms[l] for l in base])
sd=float(bms.std(ddof=1))
chk("sd_G (ddof=1, n=5, shipped CLI)", sd, J["ruler_MEASURED_NOT_BORROWED"]["sd_G"], 1e-9)
chk("2*sd_G", 2*sd, J["ruler_MEASURED_NOT_BORROWED"]["band_2sd"], 1e-9)
chk("verdict band edge armB+2sd", S.ARMG_REF_MS+2*sd, 253.998921, 1e-5)
chk("search band edge armB+EPS", S.ARMG_REF_MS+S.ARMG_EPS, 254.023979, 1e-5)
chk("BAND GAP (EPS - 2sd)", S.ARMG_EPS-2*sd, 0.0251, 5e-5)
print(f"  [ddof SENSITIVITY] sd with ddof=0 would be {float(bms.std(ddof=0)):.6f} -> 2sd {2*float(bms.std(ddof=0)):.6f}")
print(f"  [range] {float(bms.max()-bms.min()):.6f} vs published {J['ruler_MEASURED_NOT_BORROWED']['range']:.6f}")

print("\n--- best armg champion ms, and F1 ---")
ag=[v["layout"] for k,v in raw.items() if v["arm"]=="armg"]
chk("min armg ms", min(ms[l] for l in ag), 254.0137, 1e-4)
print(f"  F1 fires iff min armg ms > verdict edge: {min(ms[l] for l in ag):.6f} > {S.ARMG_REF_MS+2*sd:.6f} -> {min(ms[l] for l in ag) > S.ARMG_REF_MS+2*sd}")
chk("fastest of all 10 (control)", min(ms[l] for l in base), 253.9381, 1e-4)

print("\n--- archive sweep: 273 / min D / 7 in band ---")
pool=set()
for v in raw.values(): pool |= set(v["top50"])
chk("n distinct archive layouts", len(pool), A["n_archive"], 0.5)
# rescore the WHOLE pool via FastEval (positive-controlled path) -- 273 layouts
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
pl=sorted(pool); g=fe.gauges(np.stack([EV.perm_of(x) for x in pl]))
Dv=S.armg_deficit(g); mv=g["_ms_per_char"]
chk("global min D over archive", float(Dv.min()), A["global_min_D"], 1e-9)
edge=S.ARMG_REF_MS+2*sd
chk("n archive layouts inside MEASURED band", int((mv<=edge).sum()), A["n_inband"], 0.5)
zero=[pl[i] for i in range(len(pl)) if Dv[i]==0.0]
print(f"  D==0 layouts in archive: {zero}  (published {A['zero_D_layouts_excl_armB']})")
print(f"  is arm B in the archive pool at all? {ARMB in pool}")

print("\n--- placebo: mean/min D by arm ---")
agD=np.array([D_of(v['layout']) for k,v in raw.items() if v['arm']=='armg'])
bsD=np.array([D_of(v['layout']) for k,v in raw.items() if v['arm']=='baseline'])
chk("armg mean D", float(agD.mean()), 2.4973, 5e-4)
chk("control mean D", float(bsD.mean()), 2.5771, 5e-4)
chk("armg min D", float(agD.min()), 1.0594, 5e-4)
chk("control min D", float(bsD.min()), 1.2415, 5e-4)

print("\n--- axis-win count 7.80 vs 7.80 ---")
def wins(l):
    a=rows[l]["gauges"]; b=rows[ARMB]["gauges"]
    return sum(1 for k in G if S.ARMG_DIR[k]*(a[k]-b[k])<0)
wa=[wins(v['layout']) for k,v in raw.items() if v['arm']=='armg']
wb=[wins(v['layout']) for k,v in raw.items() if v['arm']=='baseline']
chk("armg mean axis-wins", float(np.mean(wa)), 7.80, 1e-9)
chk("control mean axis-wins", float(np.mean(wb)), 7.80, 1e-9)
print(f"  per-seed armg {wa}  control {wb}")

print("\n--- the headline self-kill numbers ---")
sel=J["selection"]["winner_layout"]
chk("selected champion oxey-style", rows[sel]["gauges"]["oxey-style"], 11.3958, 5e-4)
chk("arm B oxey-style", rows[ARMB]["gauges"]["oxey-style"], 8.6110, 5e-4)
chk("selected champion ms", ms[sel], 254.0170, 5e-4)
chk("selected champion D", D_of(sel), 1.0594, 5e-4)
# oxey share of MY deficit
worse=[k for k in G if S.ARMG_DIR[k]*(rows[sel]["gauges"][k]-rows[ARMB]["gauges"][k])>0]
tot=sum(abs(rows[sel]["gauges"][k]-rows[ARMB]["gauges"][k])/S.ARMG_SCALE[k] for k in worse)
oshare=abs(rows[sel]["gauges"]["oxey-style"]-rows[ARMB]["gauges"]["oxey-style"])/S.ARMG_SCALE["oxey-style"]/tot*100
chk("oxey share of my own deficit (%)", oshare, 20.0, 0.1)
chk("hamming(selected, armB)", sum(1 for x,y in zip(sel,ARMB,strict=True) if x!=y), 16, 0.5)

print("\n--- unique_evals achieved, from raw ---")
for k in sorted(raw): print(f"  {k:<14} {raw[k]['uniq']:>9,} ({raw[k]['uniq']/1e6:.1%})")
print(f"  all >= 80% floor: {all(v['uniq']>=800000 for v in raw.values())}")

print(f"\n{'='*70}\nMISMATCHES: {len(fails)}")
for f in fails: print("  ***", f)
