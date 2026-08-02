"""Off-home + bottom-row + top-row keystroke load per board (corpus-weighted). Model-free."""
import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="48"
WT="/local/home/zegertho/agent/workspaces/pickone/wt"; sys.path.insert(0,WT+"/src")
import keybo; assert keybo.__file__.startswith(WT), keybo.__file__
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.data.corpus import production_corpus_dir
BOARDS={"arm-B":"flmpg-yuo,sntdcireahkxbwv'.jzq","F(2.5)":"flmpg-,uoysntdcireahkxbwv.'jzq",
 "BALL-1":"flmpg-yuo,sntcdireahkxbwv'.jzq","F(2.0)":"pyu.,gdfnlhieaocstrmkj'-qbwzvx",
 "candidate":"pyu.,vdfnlhieaocstrmkj'-qgwbzx","keybo-lsb":_EXTRA_NAMED["keybo-lsb"],
 "flagship-c3":_EXTRA_NAMED["flagship-c3"],"archive-1846":_EXTRA_NAMED["archive-1846"],
 "graphite":NAMED_LAYOUTS["graphite"],"semimak":NAMED_LAYOUTS["semimak"],
 "colemak":NAMED_LAYOUTS["colemak"],"colemak-dh":"qwfpbjluy;arstgmneiozxcdvkh,./",
 "canary":"wlypbzfou;crstgmneiaqjvdkxh/,.","recurva":"frdpvqjuoysntcb.heaizxkgwml,;/",
 "mtgap":'ypoujkdlcwinea,mhtsrq";.:bfgvx',"qwerty":NAMED_LAYOUTS["qwerty"]}
cdir=production_corpus_dir(None)
uni={}
with open(cdir/"bigrams.txt") as fh:
    for line in fh:
        p=line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(p)<2: continue
        a,b=p[0],p[1]; ng,cnt=(a,b) if not a.isdigit() else (b,a)
        try: c=int(cnt)
        except ValueError: continue
        if ng: uni[ng[0]]=uni.get(ng[0],0)+c
# geometry row convention (verified against qwerty): y=3 TOP, y=2 HOME, y=1 BOTTOM, y=0 space/thumb.
lay=Layout(NAMED_LAYOUTS["qwerty"],ROW_STAGGERED_30)
rows_of={ch:lay.pos(ch)[1] for ch in "qazwsx"}  # top/home/bottom sample
print("qwerty q/a/z rows:",{c:rows_of[c] for c in "qaz"})  # expect q=3,a=2,z=1
res={}
for n,s in BOARDS.items():
    lay=Layout(s,ROW_STAGGERED_30); tot=0; off=0; bot=0; top=0; home=0
    for ch,c in uni.items():
        if ch==" " or not lay.has_key(ch): continue
        y=lay.pos(ch)[1]; tot+=c
        if y==2: home+=c
        elif y==1: bot+=c
        elif y==3: top+=c
    res[n]={"home%":100*home/tot,"top%":100*top/tot,"bottom%":100*bot/tot,
            "off_home%":100*(top+bot)/tot}
json.dump(res,open("/local/home/zegertho/agent/state/pickone/artifacts/rowload.json","w"),indent=1)
print("\n%-13s %7s %7s %7s %9s"%("board","home%","top%","bot%","offhome%"))
for n,v in sorted(res.items(),key=lambda x:x[1]["off_home%"]):
    print("%-13s %7.2f %7.2f %7.2f %9.2f"%(n,v["home%"],v["top%"],v["bottom%"],v["off_home%"]))
