"""THE DECISIVE TEST: does our speed lead survive on the COMMUNITY-fitted surface?
AALTO = 98.7% qwerty (the B3-CORPUS-1 blind spot). COMMUNITY = fit on real optimized-layout typists.
If a board's rank inverts between them, its lead is an artifact of the qwerty-heavy fold."""
import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="48"
WT="/local/home/zegertho/agent/workspaces/pickone/wt"; sys.path.insert(0,WT+"/src")
import keybo; assert keybo.__file__.startswith(WT), keybo.__file__
ART="/local/home/zegertho/agent/state/pickone/artifacts"
N={"flmpg-yuo,sntdcireahkxbwv'.jzq":"arm-B","flmpg-,uoysntdcireahkxbwv.'jzq":"F(2.5)",
 "flmpg-yuo,sntcdireahkxbwv'.jzq":"BALL-1","pyu.,gdfnlhieaocstrmkj'-qbwzvx":"F(2.0)",
 "pyu.,vdfnlhieaocstrmkj'-qgwbzx":"candidate","qwfpbjluy;arstgmneiozxcdvkh,./":"colemak-dh",
 "wlypbzfou;crstgmneiaqjvdkxh/,.":"canary","vmlcpxfouj;strdy.naeizkqgwbh',":"sturdy",
 "frdpvqjuoysntcb.heaizxkgwml,;/":"recurva",'ypoujkdlcwinea,mhtsrq";.:bfgvx':"mtgap"}
out={"provenance":{"keybo":keybo.__file__},
 "why":"AALTO fold is 98.7% qwerty w/ unexplained dvorak provenance (B3-CORPUS-1). COMMUNITY is fit on "
       "real optimized-layout typists. A rank inversion between them = the lead is a qwerty-fold artifact.",
 "caveat":"COMM-CROSS rules the community surface a DESCRIPTIVE gauge; an inversion does NOT prove the "
          "community board is faster. It proves our lead is NOT frame-robust. That is enough to sink a pick.",
 "per_corpus":{}}
for corpus,f in (("blend-v1","analyze_16boards_blendv1.json"),("iweb","analyze_16boards_iweb.json")):
    d=json.load(open(f"{ART}/{f}")); R={}
    for k,v in d["rows"].items():
        n=N.get(k,k); s=(v.get("model_scores") or {}).get("surfaces") or {}
        if not s: continue
        e={"ms":v["time"]["ms_per_char"]}
        for kk,vv in s.items():
            if isinstance(vv,dict) and vv.get("fit") is not None: e[kk.split("_")[0]]=vv["fit"]
        if len(e)>1: R[n]=e
    rk={}
    for tag in ("AALTO","COMMUNITY","POOL"):
        vals={n:R[n][tag] for n in R if R[n].get(tag) is not None}
        rk[tag]={n:i+1 for i,(n,_) in enumerate(sorted(vals.items(),key=lambda x:x[1]))}
    inv={n:{"AALTO":rk["AALTO"].get(n),"COMMUNITY":rk["COMMUNITY"].get(n),"POOL":rk["POOL"].get(n),
            "ms":R[n]["ms"],
            "aalto_to_community_move":(rk["COMMUNITY"].get(n) or 0)-(rk["AALTO"].get(n) or 0)}
         for n in R}
    out["per_corpus"][corpus]={"n_boards_indexable":len(R),"ranks":inv}
json.dump(out,open(f"{ART}/surfaces_inversion.json","w"),indent=1)
print("WROTE surfaces_inversion.json\n")
for corpus,blk in out["per_corpus"].items():
    print(f"=== {corpus} ({blk['n_boards_indexable']} C30M-indexable boards) ===")
    print("%-14s %9s | %5s %5s %5s | %s"%("board","ms/char","AALT","COMM","POOL","A->C move"))
    for n,e in sorted(blk["ranks"].items(),key=lambda x:x[1]["ms"]):
        flag=" <<< COLLAPSES" if e["aalto_to_community_move"]>=5 else (" <<< RISES" if e["aalto_to_community_move"]<=-3 else "")
        print("%-14s %9.3f | %5s %5s %5s | %+3d%s"%(n[:14],e["ms"],e["AALTO"],e["COMMUNITY"],e["POOL"],e["aalto_to_community_move"],flag))
    print()
