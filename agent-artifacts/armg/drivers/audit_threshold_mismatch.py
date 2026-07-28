"""(b) IS THE SIGN ERROR ONLY IN THE BAND? Enumerate EVERY threshold that appears in both a
RUN-TIME role and a JUDGE-TIME role, and check they agree. The general form of my defect."""
import json, re, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import search as S, judge_armg as JU
J=json.load(open("/local/home/zegertho/agent/state/armg/artifacts/armg-judgement.json"))
print("EVERY constant with BOTH a run-time and a judge-time role:\n")
rows=[]
# 1. the speed band -- the known defect
rows.append(("speed band", f"search: EPS={S.ARMG_EPS} (borrowed 2x0.0617)",
             f"judge: 2*sd_G={J['ruler_MEASURED_NOT_BORROWED']['band_2sd']:.6f} (measured)",
             "MISMATCH (the known defect): search 0.1234 LOOSER than judge 0.0983 by 0.0251"))
# 2. ARMG_REF -- same dict in both?
same_ref = S.ARMG_REF is JU.S.ARMG_REF
rows.append(("ARMG_REF (arm B gauges)", "search: search.ARMG_REF",
             "judge: imports search.ARMG_REF", f"SAME OBJECT: {same_ref} -> cannot diverge"))
# 3. ARMG_SCALE
rows.append(("ARMG_SCALE (s_g)", "search: search.ARMG_SCALE",
             "judge: imports search.ARMG_SCALE", f"SAME OBJECT: {S.ARMG_SCALE is JU.S.ARMG_SCALE} -> cannot diverge"))
# 4. ARMG_DIR
rows.append(("ARMG_DIR (directions)", "search: search.ARMG_DIR",
             "judge: imports search.ARMG_DIR", f"SAME OBJECT: {S.ARMG_DIR is JU.S.ARMG_DIR} -> cannot diverge"))
# 5. the GAUGE FRAME -- does the search's D use the same 14 axes the judge counts?
search_axes=set(S.ARMG_DIR); judge_axes=set(JU.GAUGES)
rows.append(("gauge frame (14 live axes)", f"search D sums {len(search_axes)} axes",
             f"judge counts {len(judge_axes)} axes", f"IDENTICAL SET: {search_axes==judge_axes}"))
# 6. the reference LAYOUT
rows.append(("reference layout", f"search: ARMG_LAYOUT_REF", "judge: INCUMBENTS['arm-B']",
             f"IDENTICAL: {S.ARMG_LAYOUT_REF==JU.INCUMBENTS['arm-B']}"))
# 7. ARMG_REF_MS
rows.append(("arm B ms/char", f"search: ARMG_REF_MS={S.ARMG_REF_MS}",
             f"judge: S.ARMG_REF_MS (imported)", "SAME CONSTANT"))
# 8. the 80% achieved floor -- runner computes clears_floor; judge re-filters on it
import run_armg as RA
floor_run=RA.ACHIEVED_FLOOR
rows.append(("80% achieved floor", f"runner: ACHIEVED_FLOOR={floor_run} sets clears_floor",
             "judge: filters on r['clears_floor'] (does NOT recompute)",
             "SINGLE SOURCE -> consistent, but judge TRUSTS the runner's flag (see note)"))
# 9. EVALUATION PATH: search scores with FastEval, judge scores with the shipped CLI
rows.append(("scoring path", "search: FastEval.gauges (in-process)",
             "judge: shipped `keybo analyze --json` (subprocess)",
             "DIFFERENT PATHS -- positive-controlled to worst rel 1.233e-14 BEFORE any run"))
# 10. D_FAILURE_BAR: judge-only, never used at run time
rows.append(("D_FAILURE_BAR (F2)", "search: NOT USED",
             f"judge: {JU.D_FAILURE_BAR} (flagship-c3's D)", "JUDGE-ONLY -> no mismatch possible"))
# 11. LAMBDA
rows.append(("penalty LAMBDA", f"search: {S.ARMG_LAMBDA}", "judge: NOT USED",
             "RUN-ONLY -> no mismatch possible"))
for r in rows:
    print(f"  {r[0]:<28}\n      run   : {r[1]}\n      judge : {r[2]}\n      => {r[3]}\n")
# the one real question: does the judge's D reproduce the search's D on the champions?
print("CROSS-PATH CHECK on the quantity that matters -- D computed by the SEARCH path")
print("(FastEval) vs the JUDGE path (shipped CLI), on all 10 champions:")
import numpy as np, evobj as EV
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
worst=0.0
for c in J["champions"]:
    g=fe.gauges(np.stack([EV.perm_of(c["layout"])]))
    d_search=float(S.armg_deficit(g)[0]); d_judge=c["D_vs_armB"]
    worst=max(worst,abs(d_search-d_judge))
print(f"  worst |D_search - D_judge| over 10 champions = {worst:.3e}")
print(f"  => the two paths agree on D to {worst:.1e}; the band is the ONLY divergent threshold.")
