import json, subprocess, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, evobj as EV
import keybo
assert str(Path(keybo.__file__).resolve()).startswith('/tmp/armg/')
P = json.load(open('/local/home/zegertho/agent/state/optevidence/artifacts/search-noise-placebo.json'))
six = {f"s{r['seed']}": r['layout'] for r in P['runs']['baseline']}
frozen_ms = {f"s{r['seed']}": r['ms_per_char'] for r in P['runs']['baseline']}
INC = {"arm-B":"flmpg-yuo,sntdcireahkxbwv'.jzq","arm-A":"udy.,fgpmliheaocsntr-k'qjwzbvx",
 "keybo-lsb":"pyuo,vgdnlhiea.cstrmkj-z'fwbxq","keybo-lsb+lm":"pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
 "flagship-c3":"pyou'vgdnmheai.cstrlkjz,-wfbxq","graphite":"bldwz'foujnrtsgyhaeixqmcvkp,.-"}
allL = {**six, **INC}
names=list(allL)
# DEDUPE: seed900000 champion IS arm B (SPEEDTIE-1 headline), so the spec list has a
# genuine duplicate. The shipped analyze REFUSES a dropped row (trap 38 fix) -- correct
# behaviour, so send unique specs and map back.
specs=sorted(set(allL[n] for n in names))
print(f"{len(names)} names -> {len(specs)} UNIQUE specs (duplicates: "
      f"{[n for n in names if sum(1 for m in names if allL[m]==allL[n])>1]})")
p=subprocess.run(["uv","run","--no-sync","keybo","analyze","--json"]+specs,cwd="/tmp/armg",capture_output=True,text=True)
assert p.returncode==0, p.stderr[-3000:]
rows=json.loads(p.stdout)["rows"]
assert len(rows)>=len(specs), f"DROPPED ROW: {len(rows)} vs {len(specs)}"
G=("sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll","redir","scissor","imbalance","oxey-style","comfort")
out={}
for n in names:
    r=rows[allL[n]]
    out[n]={"layout":allL[n],"ms_per_char":r["time"]["ms_per_char"],**{g:r["gauges"][g] for g in G},"sfr":r["gauges"]["sfr"]}
# verify frozen ms reproduce
print("=== FROZEN ms/char REPRODUCTION (six 1M champions) ===")
worst=0.0
for n in six:
    d=abs(out[n]["ms_per_char"]-frozen_ms[n]); worst=max(worst,d)
    print(f"  {n:<10} mine {out[n]['ms_per_char']:.10f} frozen {frozen_ms[n]:.10f} diff {d:.3e}")
print(f"  WORST DIFF = {worst:.3e}")
print()
print(f"{'name':<13} {'ms/char':>12} {'oxey-style':>11} {'imbalance':>10} {'scissor':>8} {'sfb':>7} {'redir':>7}")
for n in sorted(out, key=lambda k: out[k]["ms_per_char"]):
    o=out[n]; print(f"{n:<13} {o['ms_per_char']:>12.4f} {o['oxey-style']:>11.4f} {o['imbalance']:>10.4f} {o['scissor']:>8.4f} {o['sfb']:>7.4f} {o['redir']:>7.4f}")
json.dump({"reproduction_worst_abs_diff":worst,"profiles":out,"frozen_ms":frozen_ms},open('/tmp/armg/agent-artifacts/armg/prereg-inputs.json','w'),indent=1)
print("\nWROTE prereg-inputs.json")
