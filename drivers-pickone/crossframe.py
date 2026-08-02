"""Cross-frame check: do OUR boards keep their lead on the COMMUNITY's own tools? (robustness)"""
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
d=json.load(open(f"{ART}/analyze_16boards_blendv1.json"))
OURS=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate","keybo-lsb","flagship-c3","archive-1846"]
COMM=["colemak-dh","canary","sturdy","recurva","mtgap","graphite","semimak","colemak"]
board={}
for k,v in d["rows"].items():
    n=NAME.get(k,k)
    board[n]={"ms":v["time"]["ms_per_char"],"genkey":v["community"]["genkey"],
              "oxey1":v["community"]["oxeylyzer1"],"oxey2":v["community"]["oxeylyzer2"],
              "gk_primed":(v.get("community_primed") or {}).get("genkey_primed")}
def rank(metric,lower_better):
    valid={n:board[n][metric] for n in board if board[n][metric] is not None}
    return {n:ix+1 for ix,(n,_) in enumerate(sorted(valid.items(),key=lambda x:x[1],reverse=not lower_better))}
frames={"our_model_ms":rank("ms",True),"genkey":rank("genkey",True),
        "oxeylyzer1":rank("oxey1",False),"oxeylyzer2":rank("oxey2",False),
        "genkey_primed":rank("gk_primed",True)}
out={"provenance":{"keybo":keybo.__file__,"source":f"{ART}/analyze_16boards_blendv1.json"},
     "finding":"our cluster leads ONLY on our own fitted surface; community tools reorder it",
     "per_frame_rank":{f:{n:r.get(n) for n in ["arm-B","keybo-lsb","flagship-c3","recurva","semimak","graphite","colemak-dh"]} for f,r in frames.items()},
     "best_of_ours_per_frame":{}, "best_of_community_per_frame":{}, "community_beats_all_ours":{}}
for f,(metric,lb) in {"our_model_ms":("ms",True),"genkey":("genkey",True),
    "oxeylyzer1":("oxey1",False),"oxeylyzer2":("oxey2",False)}.items():
    ov={n:board[n][metric] for n in OURS if board[n][metric] is not None}
    cv={n:board[n][metric] for n in COMM if board[n][metric] is not None}
    bo=(min if lb else max)(ov,key=ov.get); bc=(min if lb else max)(cv,key=cv.get)
    beat = (cv[bc]<ov[bo]) if lb else (cv[bc]>ov[bo])
    out["best_of_ours_per_frame"][f]={ "board":bo,"val":ov[bo]}
    out["best_of_community_per_frame"][f]={"board":bc,"val":cv[bc]}
    out["community_beats_all_ours"][f]=bool(beat)
json.dump(out,open(f"{ART}/crossframe.json","w"),indent=1)
print("WROTE crossframe.json\n")
print("%-14s %6s %6s %6s %6s"%("board","ms","genky","oxey1","oxey2"))
for n in ["arm-B","F(2.0)","keybo-lsb","flagship-c3","recurva","semimak","graphite","colemak-dh"]:
    r={f:frames[f].get(n,'-') for f in ("our_model_ms","genkey","oxeylyzer1","oxeylyzer2")}
    print("%-14s %6s %6s %6s %6s"%(n,r["our_model_ms"],r["genkey"],r["oxeylyzer1"],r["oxeylyzer2"]))
print("\ncommunity board beats ALL of ours on:", [f for f,b in out["community_beats_all_ours"].items() if b])
