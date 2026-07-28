"""(a) continued -- hunting the THIRD wrong-constant-behind-a-true-conclusion.
Candidate found by the sweep: sd_G's ddof/statistic choice. Was it PRE-REGISTERED, and does
the verdict survive every defensible alternative? A number that only works at one arbitrary
choice is the shape I am looking for."""
import json, glob, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, search as S
RUNS=Path("/local/home/zegertho/agent/state/armg/artifacts/runs")
raw={}
for f in sorted(glob.glob(str(RUNS/"*-r?.json"))):
    b=json.load(open(f)); raw[Path(f).stem]={"layout":b["champion"]["layout"],"arm":b["arm"]}
J=json.load(open("/local/home/zegertho/agent/state/armg/artifacts/armg-judgement.json"))
prof={c["name"]:c for c in J["champions"]}
bms=np.array([c["ms_per_char"] for c in J["champions"] if c["arm"]=="baseline"])
ams=np.array([c["ms_per_char"] for c in J["champions"] if c["arm"]=="armg"])
ARMB=S.ARMG_REF_MS
print("Q: does F1 (verdict=FAILURE) survive EVERY defensible ruler choice, or only ddof=1?")
print(f"   min armg ms = {ams.min():.6f};  arm B = {ARMB:.6f}\n")
print(f"{'ruler variant':<46} {'value':>10} {'2x':>10} {'edge':>12} {'F1 fires?':>10}")
variants={
 "sd ddof=1, n=5 baseline champs (REGISTERED)": float(bms.std(ddof=1)),
 "sd ddof=0, n=5 baseline champs":              float(bms.std(ddof=0)),
 "sd ddof=1 over ALL 10 champions":             float(np.concatenate([bms,ams]).std(ddof=1)),
 "sd ddof=1 over the 5 ARMG champions":         float(ams.std(ddof=1)),
 "half-RANGE of baseline champs":               float((bms.max()-bms.min())/2),
 "full RANGE of baseline champs":               float(bms.max()-bms.min()),
 "BORROWED SPEEDTIE-1 sd (0.0617)":             0.06171827216711913,
 "BORROWED SPEEDTIE-1 range (0.1760)":          0.17597221728999557,
}
sur=[]
for k,v in variants.items():
    edge=ARMB+2*v; fires=ams.min()>edge
    sur.append(fires)
    print(f"{k:<46} {v:>10.6f} {2*v:>10.6f} {edge:>12.6f} {str(fires):>10}")
print(f"\n=> F1 fires under {sum(sur)} of {len(sur)} defensible rulers.")
print("   The ONLY rulers under which F1 would NOT fire are the two BORROWED SPEEDTIE-1 ones")
print("   -- i.e. exactly the rulers the standing quadruple rule forbids me from using.")
print("   So the verdict is NOT an artifact of my ddof choice; it is robust across every")
print("   ruler computed from MY OWN replicate structure.")
print()
print("BUT THE HONEST CAVEAT I DID NOT STATE:", flush=True)
print("   My prereg section 4 fixed 'sd, ddof=1' -- I checked. Let me verify that claim.")
pre=open("/tmp/armg/agent-artifacts/armg/PREREGISTRATION.md").read()
import re
for m in re.finditer(r'[^\n]*ddof[^\n]*', pre): print("   PREREG SAYS:", m.group(0).strip())
print()
print("   And is the 'FASTER' cell also ruler-robust? A layout is FASTER iff ms < armB - 2sd:")
for k,v in variants.items():
    lo=ARMB-2*v
    n=int((np.concatenate([bms,ams])<lo).sum())
    print(f"     {k:<46} needs ms < {lo:.6f} -> {n} champions qualify")
