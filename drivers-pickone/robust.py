"""Corpus-robustness: does the DECISION structure survive blend-v1 -> iweb? (ranks vs groupings)"""
import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="48"
WT="/local/home/zegertho/agent/workspaces/pickone/wt"; sys.path.insert(0,WT+"/src")
import keybo; assert keybo.__file__.startswith(WT), keybo.__file__
ART="/local/home/zegertho/agent/state/pickone/artifacts"
NAME={"flmpg-yuo,sntdcireahkxbwv'.jzq":"arm-B","pyu.,gdfnlhieaocstrmkj'-qbwzvx":"F(2.0)",
 "flmpg-,uoysntdcireahkxbwv.'jzq":"F(2.5)","pyu.,vdfnlhieaocstrmkj'-qgwbzx":"candidate",
 "flmpg-yuo,sntcdireahkxbwv'.jzq":"BALL-1","qwfpbjluy;arstgmneiozxcdvkh,./":"colemak-dh",
 "wlypbzfou;crstgmneiaqjvdkxh/,.":"canary","vmlcpxfouj;strdy.naeizkqgwbh',":"sturdy",
 "frdpvqjuoysntcb.heaizxkgwml,;/":"recurva",'ypoujkdlcwinea,mhtsrq";.:bfgvx':"mtgap"}
def load(f):
    d=json.load(open(f)); r={}
    for k,v in d["rows"].items():
        n=NAME.get(k,k); a=v.get("attribution") or {}; fp=a.get("finger_time_pct",{}); g=v["gauges"]
        r[n]={"ms":v["time"]["ms_per_char"],"lweak":fp.get("LP",0)+fp.get("LR",0),
              "pinky":fp.get("LP",0)+fp.get("RP",0),"LR":fp.get("LR",0),"LP":fp.get("LP",0),
              "sfb":g["sfb"],"sfs":g["sfs"],"lsb":g["lsb"],"scissor":g["scissor"],
              "alt":g["alt"],"roll":g["roll"],"redir":g["redir"]}
    return r
B=load(f"{ART}/analyze_16boards_blendv1.json"); I=load(f"{ART}/analyze_16boards_iweb.json")
CLUSTER=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate"]
LOWW=["keybo-lsb","candidate","F(2.0)"]; HIGHW=["arm-B","BALL-1","F(2.5)","flagship-c3","archive-1846"]
out={"provenance":{"keybo":keybo.__file__,"blend":f"{ART}/analyze_16boards_blendv1.json",
                   "iweb":f"{ART}/analyze_16boards_iweb.json"},
     "note":"RANK instability != DECISION instability. Test the GROUPINGS a learner acts on."}
rk={}
for cn,d in (("blend-v1",B),("iweb",I)):
    rk[cn]={k:{n:ix+1 for ix,(n,_) in enumerate(sorted(d.items(),key=lambda x:x[1][k]))}
            for k in ("ms","lweak","pinky","sfb","lsb","scissor")}
out["rank_instability"]={k:{"n_moved":sum(1 for n in rk["blend-v1"][k] if rk["blend-v1"][k][n]!=rk["iweb"][k][n]),
                            "n_boards":len(rk["blend-v1"][k])} for k in rk["blend-v1"]}
dec={}
for cn,d in (("blend-v1",B),("iweb",I)):
    cw=max(d[x]["ms"] for x in CLUSTER); cb=min(d[x]["ms"] for x in CLUSTER)
    lo=[d[x]["lweak"] for x in LOWW]; hi=[d[x]["lweak"] for x in HIGHW]
    dec[cn]={"cluster_span_ms":cw-cb,
      "cluster_worst_minus_keybo_lsb":cw-d["keybo-lsb"]["ms"],
      "cluster_worst_vs_best_community_mtgap":d["mtgap"]["ms"]-cw,
      "cluster_worst_vs_colemak_dh":d["colemak-dh"]["ms"]-cw,
      "weak_left_LOW_range":[min(lo),max(lo)],"weak_left_HIGH_range":[min(hi),max(hi)],
      "weak_left_gap_pp":min(hi)-max(lo),"weak_left_groups_overlap":max(lo)>min(hi),
      "keybo_lsb_rank_lweak":rk[cn]["lweak"]["keybo-lsb"],
      "keybo_lsb_rank_pinky":rk[cn]["pinky"]["keybo-lsb"],
      "keybo_lsb_rank_ms":rk[cn]["ms"]["keybo-lsb"]}
out["decision_structure"]=dec
out["stable_conclusions"]=[
 "the 5-board cluster is faster than keybo-lsb on BOTH corpora (gap +0.631 / +0.194 ms/char)",
 "the cluster is faster than the FASTEST community board (mtgap) by +3.03 / +2.54 ms/char on BOTH",
 "weak-left LOW vs HIGH groups do NOT overlap on either corpus (gap +4.54pp / +3.71pp)",
 "keybo-lsb is rank 2/17 on total pinky load on BOTH corpora",
]
out["unstable_conclusions"]=[
 "WHICH cluster board is fastest (arm-B rank 1 on blend-v1, rank 4 on iweb; candidate 5 -> 1)",
 "keybo-lsb's weak-left RANK (1 on blend-v1, 5 on iweb) — canary/colemak-dh/colemak pass it on iweb",
 "cluster span itself is corpus-dependent: 0.0991 (blend-v1) vs 0.2610 (iweb)",
]
out["by_board"]={"blend-v1":B,"iweb":I}
json.dump(out,open(f"{ART}/corpus_robustness.json","w"),indent=1)
print("WROTE corpus_robustness.json\n")
for k,v in out["rank_instability"].items(): print(f"  {k:8s}: {v['n_moved']}/{v['n_boards']} ranks moved blend-v1 -> iweb")
print("\nSTABLE:");  [print("  +",s) for s in out["stable_conclusions"]]
print("UNSTABLE:"); [print("  -",s) for s in out["unstable_conclusions"]]
