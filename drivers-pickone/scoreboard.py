"""Master scoreboard for the report: every decision axis, one row per board."""
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
blend=json.load(open(f"{ART}/analyze_16boards_blendv1.json"))["rows"]
row=json.load(open(f"{ART}/rowload.json"))
sk=json.load(open(f"{ART}/strain_keystroke.json"))["keystroke_strain"]
sb={}
for k,v in blend.items():
    n=NAME.get(k,k); a=v.get("attribution") or {}; fp=a.get("finger_time_pct",{}); g=v["gauges"]
    sb[n]={"ms":v["time"]["ms_per_char"],"saved%":v["time"]["saved_vs_ref_pct"],
           "sfb":g["sfb"],"sfs":g["sfs"],"lsb":g["lsb"],"scissor":g["scissor"],
           "alt":g["alt"],"roll":g["roll"],"redir":g["redir"],
           "weak_left_time":fp.get("LP",0)+fp.get("LR",0),"pinky_time":fp.get("LP",0)+fp.get("RP",0),
           "weak_left_ks":sk.get(n,{}).get("weak_left_pct"),"pinky_ks":sk.get(n,{}).get("pinky_total_pct"),
           "bottom%":row.get(n,{}).get("bottom%"),"offhome%":row.get(n,{}).get("off_home%"),
           "genkey":v["community"]["genkey"]}
json.dump(sb,open(f"{ART}/scoreboard.json","w"),indent=1)
ORDER=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate","keybo-lsb","flagship-c3","archive-1846",
       "mtgap","canary","semimak","recurva","graphite","colemak","colemak-dh","qwerty","sturdy"]
print("%-13s %8s %6s | %5s %5s %5s %5s | %6s %6s %6s | %6s"%(
  "board","ms/char","sav%","sfb","sfs","lsb","scis","wkL_t","pnk_t","bot%","genky"))
for n in ORDER:
    if n not in sb: continue
    v=sb[n]
    print("%-13s %8.2f %6.2f | %5.3f %5.3f %5.3f %5.3f | %6.2f %6.2f %6.2f | %6.1f"%(
      n,v["ms"],v["saved%"],v["sfb"],v["sfs"],v["lsb"],v["scissor"],
      v["weak_left_time"],v["pinky_time"],v["bottom%"],v["genkey"]))
print("\nWROTE scoreboard.json")
