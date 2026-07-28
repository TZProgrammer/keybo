"""SELF-SEPARATION: re-read ARM G's own results as a hostile stranger.
For each claim: what refutes it, does my check SHARE A COMPONENT with the target, and did any
control run only AFTER I used its result?"""
import json
import sys
from pathlib import Path

sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import evobj as EV
import numpy as np
import search as S

ART=Path("/local/home/zegertho/agent/state/armg/artifacts")
J=json.load(open(ART/"armg-judgement.json")); A=json.load(open(ART/"armg-archive-analysis.json"))
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
ARMB="flmpg-yuo,sntdcireahkxbwv'.jzq"; ARMB_MS=S.ARMG_REF_MS
out={}

print("=== A1. Is F1 an artifact of my EPS, or would a CORRECT EPS have passed? ===")
# counterfactual: had I set EPS = 2*sd_G (0.0983), the penalty would have bound tighter.
# I CANNOT claim it would have found a D=0 layout -- but I CAN bound what it would have had
# to find, from the archive.
print(f"  global min D over the 273-layout archive = {A['global_min_D']:.4f}")
print(f"  best D inside the MEASURED band          = {A['inband_sorted_by_D'][0]['D']:.4f}"
      f" (found by {A['inband_sorted_by_D'][0]['found_by']})")
print(f"  D=0 layouts anywhere in the archive      = {A['zero_D_layouts_excl_armB']}")
print("  => even with a perfectly-set EPS, the search would have had to find something the")
print("     273-layout archive does not contain. F1 is NOT merely a band-setting artifact;")
print("     the TIED-AND-STRICTLY-BETTER cell was unreachable at this budget either way.")
out["A1"]={"claim":"F1 is not merely an EPS artifact","min_D_archive":A["global_min_D"],
  "zero_D":A["zero_D_layouts_excl_armB"],
  "refuted_by":"a D=0 layout appearing in the archive, or at a tighter EPS"}

print("\n=== A2. SHARED COMPONENT: does my D-based selection share a component with D? ===")
# YES and it is fatal to any 'ARM G won on D' reading. D is BOTH the search objective AND
# the selection statistic. So a D advantage for armg over the control would be partly
# tautological. Test with a statistic that does NOT contain D: per-axis win count vs arm B.
champs={c["name"]:c for c in J["champions"]}
def wins_vs_armB(c):
    p=fe.gauges(np.stack([EV.perm_of(c["layout"])]))
    b=fe.gauges(np.stack([EV.perm_of(ARMB)]))
    w=sum(1 for g in S.ARMG_DIR if S.ARMG_DIR[g]*(float(p[g][0])-float(b[g][0]))<0)
    return w
ag=[(n,c) for n,c in champs.items() if c["arm"]=="armg"]
bs=[(n,c) for n,c in champs.items() if c["arm"]=="baseline"]
wa=[wins_vs_armB(c) for _,c in ag]; wb=[wins_vs_armB(c) for _,c in bs]
print(f"  D (SHARES the objective):        armg mean {np.mean([c['D_vs_armB'] for _,c in ag]):.4f}  vs control {np.mean([c['D_vs_armB'] for _,c in bs]):.4f}")
print(f"  axis-win count vs arm B (does NOT share it): armg mean {np.mean(wa):.2f} vs control {np.mean(wb):.2f}")
print(f"  armg per-seed wins {wa}  control {wb}")
print("  => on the NON-shared statistic the arms are also indistinguishable. The null is not")
print("     an artifact of using D to judge D.")
out["A2"]={"D_armg":float(np.mean([c['D_vs_armB'] for _,c in ag])),
 "D_ctrl":float(np.mean([c['D_vs_armB'] for _,c in bs])),
 "wins_armg":wa,"wins_ctrl":wb,
 "shared_component":"D is BOTH objective and selection statistic -- so the axis-win count is the honest test"}

print("\n=== A3. Did any control run only AFTER I used its result? ===")
print("  sd_G: the baseline CONTROL arm was launched in the SAME command as the armg arm,")
print("        BEFORE any result existed, and the judge that consumes it was committed")
print("        (ceb85cd) WHILE the runs were still executing. So no.")
print("  BUT: EPS=0.1234 was set from the BORROWED 0.0617 because my own sd could not exist")
print("        before my runs. That is a genuine ordering constraint, not a violation --")
print("        and it is precisely what produced the F1 failure.")
out["A3"]={"post_hoc_controls":"none","ordering_constraint":"EPS necessarily preceded sd_G"}

print("\n=== A4. Is the selected champion's '8 better / 4 worse' vs arm B robust? ===")
sel=J["pairwise_vs_selected"]["arm-B"]
print(f"  contested {sel['n_contested']}, better {sel['better']}, worse {sel['worse']}, tie {sel['tie']}")
print(f"  tied axes (BY CONSTRUCTION): {sel['tied_axes']}")
print(f"  worse axes: {sel['worse_axes']}")
print(f"  cluster-corrected: {sel['cluster_corrected']['clusters_better']} better / "
      f"{sel['cluster_corrected']['clusters_worse']} worse of {sel['cluster_corrected']['n_clusters']}")
print(f"  per-cluster: {sel['cluster_corrected']['per_cluster']}")
print("  => 8/4 shrinks to 5/3 once the oxey/lsb/sfs duplication is collapsed. NOT a dominator")
print("     (4 worse axes), so the TIED-AND-STRICTLY-BETTER cell is genuinely not reached.")
out["A4"]=sel

print("\n=== A5. HOSTILE: is my champion just arm B with a few keys moved? ===")
def ham(a,b): return sum(1 for x,y in zip(a,b,strict=True) if x!=y)
selL=J["selection"]["winner_layout"]
print(f"  selected: {selL}")
print(f"  arm B   : {ARMB}")
print(f"  Hamming(selected, arm B) = {ham(selL,ARMB)} of 30")
hs=J["hamming_armg"]; hb=J["hamming_baseline"]
print(f"  armg    : n_runs {hs['n_runs']} n_distinct {hs['n_distinct']} mean_over_runs "
      f"{hs['mean_over_runs']:.2f} mean_over_distinct {hs['mean_over_distinct']:.2f} zero_pairs {hs['n_zero_pairs']}")
print(f"  control : n_runs {hb['n_runs']} n_distinct {hb['n_distinct']} mean_over_runs "
      f"{hb['mean_over_runs']:.2f} mean_over_distinct {hb['mean_over_distinct']:.2f} zero_pairs {hb['n_zero_pairs']}")
print(f"  armg champions' Hamming to arm B: {hs['vs_armB']}")
out["A5"]={"hamming_sel_vs_armB":ham(selL,ARMB),"armg":hs,"baseline":hb}

print("\n=== A6. Cross-arm per-seed pairing: did the objective move the champion AT ALL? ===")
for r in range(5):
    a=[c for n,c in champs.items() if c["arm"]=="armg" and c["seed"]==20260728+7919*r][0]
    b=[c for n,c in champs.items() if c["arm"]=="baseline" and c["seed"]==20260728+7919*r][0]
    h=ham(a["layout"],b["layout"])
    print(f"  seed {a['seed']}: hamming(armg,control)={h:>2}  D {b['D_vs_armB']:.4f}->{a['D_vs_armB']:+.4f}"
          f"  ms {b['ms_per_char']:.4f}->{a['ms_per_char']:.4f}"
          f"{'   <-- IDENTICAL CHAMPION' if h==0 else ''}")
out["A6"]=[{"seed":20260728+7919*r,
  "hamming":ham([c for n,c in champs.items() if c["arm"]=="armg" and c["seed"]==20260728+7919*r][0]["layout"],
                [c for n,c in champs.items() if c["arm"]=="baseline" and c["seed"]==20260728+7919*r][0]["layout"])}
  for r in range(5)]
json.dump(out,open(ART/"armg-self-separation.json","w"),indent=1,default=str)
print(f"\nWROTE {ART/'armg-self-separation.json'}")
