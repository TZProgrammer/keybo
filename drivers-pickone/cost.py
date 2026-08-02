"""Span-vs-floor arithmetic + real-typing cost, from MY measured ms/char. No model refit."""
import os, sys, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="48"
WT="/local/home/zegertho/agent/workspaces/pickone/wt"
sys.path.insert(0,WT+"/src")
import keybo; assert keybo.__file__.startswith(WT), keybo.__file__

A="/local/home/zegertho/agent/state/pickone/artifacts/analyze_16boards_blendv1.json"
d=json.load(open(A)); rows=d["rows"]
NAME={"flmpg-yuo,sntdcireahkxbwv'.jzq":"arm-B","pyu.,gdfnlhieaocstrmkj'-qbwzvx":"F(2.0)",
 "flmpg-,uoysntdcireahkxbwv.'jzq":"F(2.5)","pyu.,vdfnlhieaocstrmkj'-qgwbzx":"candidate",
 "flmpg-yuo,sntcdireahkxbwv'.jzq":"BALL-1","qwfpbjluy;arstgmneiozxcdvkh,./":"colemak-dh",
 "wlypbzfou;crstgmneiaqjvdkxh/,.":"canary","vmlcpxfouj;strdy.naeizkqgwbh',":"sturdy",
 "frdpvqjuoysntcb.heaizxkgwml,;/":"recurva",'ypoujkdlcwinea,mhtsrq";.:bfgvx':"mtgap"}
ms={NAME.get(k,k):v["time"]["ms_per_char"] for k,v in rows.items()}

# typing-volume scenarios. wpm counts 5 chars per "word" (incl. space) -- the standard.
SCEN=[("light 60wpm / 2h/day",60,2.0),("heavy 100wpm / 6h/day",100,6.0)]
def chars_per_day(wpm,h): return h*3600.0*(wpm*5.0/60.0)

CLUSTER=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate"]
cl=[ms[b] for b in CLUSTER]
span=max(cl)-min(cl)
FLOOR=0.1350   # campaign resolution floor, quoted by the parent
CROSSFAM_SD=0.1196; SEARCH_SPREAD=0.8830

out={"provenance":{"source":A,"keybo":keybo.__file__,
     "floor_and_sd_are_PARENT_SUPPLIED_not_remeasured_here":True},
     "ms_per_char":ms,
     "cluster":{"boards":CLUSTER,"min":min(cl),"max":max(cl),"span_ms_per_char":span,
      "span_pct_of_mean":100*span/(sum(cl)/len(cl)),
      "span_vs_floor_pct":100*span/FLOOR,"span_vs_crossfam_sd_pct":100*span/CROSSFAM_SD,
      "span_vs_search_spread_pct":100*span/SEARCH_SPREAD}}

BEST=min(cl); best_board=CLUSTER[cl.index(BEST)]
GAPS={"intra-cluster span (worst-best of 5)":span,
      "keybo-lsb vs fastest cluster":ms["keybo-lsb"]-BEST,
      "flagship-c3 vs fastest cluster":ms["flagship-c3"]-BEST,
      "colemak-dh vs fastest cluster":ms["colemak-dh"]-BEST,
      "colemak-dh vs keybo-lsb":ms["colemak-dh"]-ms["keybo-lsb"],
      "semimak vs keybo-lsb":ms["semimak"]-ms["keybo-lsb"],
      "graphite vs keybo-lsb":ms["graphite"]-ms["keybo-lsb"],
      "qwerty vs fastest cluster":ms["qwerty"]-BEST,
      "qwerty vs colemak-dh":ms["qwerty"]-ms["colemak-dh"]}
cost={}
for label,gap in GAPS.items():
    e={"gap_ms_per_char":gap,"gap_pct_of_qwerty":100*gap/ms["qwerty"],
       "multiple_of_intracluster_span":gap/span if span else None,
       "multiple_of_floor":gap/FLOOR}
    for sn,wpm,h in SCEN:
        cpd=chars_per_day(wpm,h); sec=gap*cpd/1000.0
        e[sn]={"chars_per_day":cpd,"sec_per_day":sec,"min_per_year":sec*365/60.0,
               "hours_per_year":sec*365/3600.0}
    cost[label]=e
out["fastest_cluster_board"]=best_board
out["cost_of_each_gap"]=cost
json.dump(out,open("/local/home/zegertho/agent/state/pickone/artifacts/cost_arithmetic.json","w"),indent=1)

print(f"cluster span = {span:.4f} ms/char = {100*span/(sum(cl)/len(cl)):.4f}% of mean")
print(f"  = {100*span/FLOOR:.0f}% of the {FLOOR} floor, {100*span/CROSSFAM_SD:.0f}% of cross-fam sd, {100*span/SEARCH_SPREAD:.0f}% of search spread")
print(f"fastest cluster board: {best_board} @ {BEST:.4f}\n")
print("%-34s %8s %6s %6s | %8s %9s | %8s %9s"%("gap","ms/char","xspan","xfloor","s/day L","h/yr L","s/day H","h/yr H"))
for label,e in cost.items():
    L=e[SCEN[0][0]]; H=e[SCEN[1][0]]
    print("%-34s %8.4f %6.1f %6.1f | %8.1f %9.2f | %8.1f %9.2f"%(
      label,e["gap_ms_per_char"],e["multiple_of_intracluster_span"],e["multiple_of_floor"],
      L["sec_per_day"],L["hours_per_year"],H["sec_per_day"],H["hours_per_year"]))
