"""Render the report tables from tournament.json + secondary.json. No new numbers computed."""
import json, sys, itertools
sys.path.insert(0,"/local/home/zegertho/agent/workspaces/tournament/wt/drivers-tournament")
from _guard import BOARDS, ART
import numpy as np

T=json.load(open(f"{ART}/tournament.json"))
NAMES=list(BOARDS); PR=("all","observed","common")
PRN={"all":"ALL-CELLS","observed":"OBSERVED-ONLY","common":"COMMON-SUPPORT"}
out=[]
def w(s=""): out.append(s)

w("## Board means (ms/char, seed-mean over 25 model seeds), all three pricings")
w()
w("| board | ALL-CELLS | rank | OBSERVED-ONLY | rank | COMMON-SUPPORT | rank |")
w("|---|---|---|---|---|---|---|")
M={pr:{n:float(np.mean(T["mspc"][pr][n])) for n in NAMES} for pr in PR}
R={pr:{n:i+1 for i,n in enumerate(sorted(NAMES,key=lambda x:M[pr][x]))} for pr in PR}
for n in sorted(NAMES,key=lambda x:M["all"][x]):
    w(f"| {n} | {M['all'][n]:.4f} | {R['all'][n]} | {M['observed'][n]:.4f} | {R['observed'][n]} "
      f"| {M['common'][n]:.4f} | {R['common'][n]} |")
w()
w("**Measured resolution floor (FLOOR-A, split-half same-board placebo — truth is 0 by construction):**")
w()
w("| pricing | p50 | **p90 (the floor)** | p99 | max |")
w("|---|---|---|---|---|")
for pr in PR:
    f=T["floor_A"][pr]
    w(f"| {PRN[pr]} | {f['p50']:.4f} | **{f['p90']:.4f}** | {f['p99']:.4f} | {f['max']:.4f} |")
w()

for pr in PR:
    fl=T["floor_A"][pr]["p90"]
    rows=T["pairs"][pr]
    nw=sum(1 for r in rows if r["verdict_kind"]=="WIN")
    nt=sum(1 for r in rows if r["verdict_kind"]=="TIED")
    nu=sum(1 for r in rows if r["verdict_kind"]=="UNRESOLVED")
    w(f"## Full pairwise matrix — {PRN[pr]} pricing (floor = {fl:.4f} ms/char)")
    w()
    w(f"**{nw} WIN · {nt} TIED · {nu} UNRESOLVED** of {len(rows)} pairs.")
    w()
    w("| pair | class | H | mean Δ | sd | signs +/− | p (paired-t) | perm p | 95% CI | verdict |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows,key=lambda x:x["p_raw"]):
        v=r["verdict"]
        vd=("**"+v+"**" if r["verdict_kind"]=="WIN" else v)
        w(f"| {r['pair']} | {r['pair_class'][:4]} | {r['hamming']} | {r['mean']:+.4f} | {r['sd']:.4f} "
          f"| {r['signs_pos_neg'][0]}/{r['signs_pos_neg'][1]} | {r['p_raw']:.2e} | {r['perm_p']:.3f} "
          f"| [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {vd} |")
    w()

w("## FLIPPED verdicts across pricings (INVARIANT 5 — a flipped pair is NOT DECIDED)")
w()
if T["flips"]:
    w("| pair | H | ALL-CELLS | OBSERVED-ONLY | COMMON-SUPPORT |")
    w("|---|---|---|---|---|")
    for f in T["flips"]:
        v=f["verdicts"]
        w(f"| {f['pair']} | {f['hamming']} | {v['all']} | {v['observed']} | {v['common']} |")
else:
    w("**NONE** — every pair returns the same verdict under all three pricings.")
w()
w("## Transitivity / Condorcet")
w()
w("| pricing | WIN edges | 3-cycles | antisymmetric (D8) | consistent with total order by mean (D7) |")
w("|---|---|---|---|---|")
for pr in PR:
    c=T["condorcet"][pr]
    w(f"| {PRN[pr]} | {c['n_wins']} | **{c['n_3cycles']}** | {c['antisymmetric_D8']} "
      f"| {c['consistent_with_total_order_D7']} |")
w()
w("## FLOOR-C — verdict stability across DISJOINT seed halves (seeds 0-11 vs 12-23)")
w()
w("| pricing | pairs disagreeing | CONTRADICTORY (each half names a different winner) |")
w("|---|---|---|")
for pr in PR:
    f=T["floor_C"][pr]
    w(f"| {PRN[pr]} | {f['n_disagree']} / {T['n_pairs']} | **{f['n_contradictory']}** |")
w()
try:
    S=json.load(open(f"{ART}/secondary.json"))
    w("## Secondary axes (each labelled MEASURED vs OPINION)")
    w()
    w("| board | speed ms/char (M) | sfb % (M) | lat-span (M) | lsb % (M, uninformative) | comfort (**OPINION**) |")
    w("|---|---|---|---|---|---|")
    for n in sorted(NAMES,key=lambda x:S["rows"][x]["speed"]):
        r=S["rows"][n]
        w(f"| {n} | {r['speed']:.4f} | {r['sfb']:.4f} | {r['lat_span']:.4f} | {r['lsb']:.4f} "
          f"| {r['comfort_OPINION']:.6g} |")
    w()
    mc=S["multicriterion"]
    w(f"**Multi-criterion Condorcet test** over {{speed, sfb, lat-span}} (per-pair majority, "
      f"{mc['n_triples']} triples): **{mc['n_distinct_3cycles']} distinct 3-cycles**; "
      f"Condorcet winner: **{mc['condorcet_winner'] or 'NONE'}**.")
    w()
    if mc["cycles"]:
        w("Cycles (unordered triples):")
        for c in mc["cycles"][:20]: w(f"- {' / '.join(c)}")
        w()
    w("Copeland (3-axis majority): " + ", ".join(f"`{k}`={v:+d}"
      for k,v in sorted(mc["copeland"].items(), key=lambda kv:-kv[1])))
    w()
except FileNotFoundError:
    w("_secondary.json not yet generated_")
open(f"{ART}/report_tables.md","w").write("\n".join(out))
print("\n".join(out)[:3000])
print(f"\n...wrote {ART}/report_tables.md ({len(out)} lines)")
