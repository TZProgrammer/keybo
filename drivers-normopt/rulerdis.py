"""How much do the two ms rulers disagree? If they rank boards differently, arm A's
'control' identity depends on which ruler you call 'ms/char' — a reportable finding."""
import sys, json, statistics as st
sys.path.insert(0,"/tmp/normopt/src")
import numpy as np
from scipy.stats import spearmanr
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30
from keybo.analysis import surfaces as S

V=json.load(open("/tmp/normopt/runs/verdict.json")); XS=json.load(open("/tmp/normopt/runs/crossscore.json"))
model=XGBoostTypingModel.load("/tmp/normopt-scratch/models/bigram_reg31_seed0.json")
freqs=load_frequencies(str(production_corpus_dir(None)/"bigrams.txt"))
tbl=TableBigramScorer(model,freqs,target_wpm=90.0,chars=S.C30M)
MASS=float(sum(freqs.values()))
def r1(lay): return float(tbl.fitness(Layout(lay,ROW_STAGGERED_30)))/MASS

names=list(V["produced"])+list(V["field"])
def lay(n): return (V["produced"] if n in V["produced"] else V["field"])[n]["layout"]
def r2(n):  return (V["produced"] if n in V["produced"] else V["field"])[n]["ms"]
ok=[n for n in names if S.is_c30m(lay(n))]
a=np.array([r1(lay(n)) for n in ok]); b=np.array([r2(n) for n in ok])
rho,p=spearmanr(a,b)
print("="*96)
print("D) DO THE TWO ms RULERS AGREE? (n=%d C30M boards: my 30 + the field)" % len(ok))
print("="*96)
print(f"  spearman(ruler1 bigram-table, ruler2 analyze-trigram) = {rho:+.6f}  (p={p:.3g})")
print(f"  pearson  = {np.corrcoef(a,b)[0,1]:+.6f}")
disc=sum(1 for i in range(len(ok)) for j in range(i+1,len(ok))
         if (a[i]-a[j])*(b[i]-b[j])<0)
tot=len(ok)*(len(ok)-1)//2
print(f"  DISCORDANT PAIRS: {disc} of {tot} ({100*disc/tot:.1f}%) — the two rulers disagree on this many orderings")
print("\n  Where the campaign's OWN frozen boards sit on each ruler (they beat all 30 of mine on ruler2):")
print(f"  {'board':24}{'ruler1':>12}{'ruler2':>12}")
for n in ["keybo-lsb","keybo-c30m","arm-B","BALL-1","arm-A","graphite","semimak",
          "ng:registered-best","ng:droppool-best","ng:anchor-AALTO","ng:10M-AALTO-champ","qwerty30m"]:
    if n in V["field"]: print(f"  {n:24}{r1(lay(n)):12.6f}{r2(n):12.6f}")
print(f"\n  my best on ruler1: {min(ok,key=lambda n:r1(lay(n)))} = {min(r1(lay(n)) for n in ok):.6f}")
print(f"  my best on ruler2: {min(ok,key=lambda n:r2(n))} = {min(r2(n) for n in ok):.6f}")
mine=[n for n in ok if n in V["produced"]]
print(f"\n  BEST OF MY 30 on ruler2 = {min(r2(n) for n in mine):.6f} ({min(mine,key=r2)})")
print(f"  keybo-lsb (frozen field)  = {r2('keybo-lsb'):.6f}  -> the field board is "
      f"{r2('keybo-lsb')-min(r2(n) for n in mine):+.6f} vs my best "
      f"({abs(r2('keybo-lsb')-min(r2(n) for n in mine))/0.135:.1f}x floor)")
