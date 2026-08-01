import sys, numpy as np
sys.path.insert(0,"/tmp/quadgram-wt/src")
from keybo.data.strokes import load_strokes
from keybo.training.train import train_quadgram_model
from keybo.features.schema import QUADGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_31
CAND4=dict(n_estimators=427,max_depth=5,learning_rate=0.10903767015375725,min_child_weight=6,subsample=0.6086566147198375,colsample_bytree=0.9893815206317236,gamma=0.0,reg_alpha=0.0,reg_lambda=1.0,n_jobs=8)
rows=load_strokes("/tmp/quadstrokes31_cond_v1.tsv",ngram_len=4,wpm_threshold=0,min_samples=1)
print(f"{len(rows)} rows",flush=True)
# train full quadgram on ALL data (importance is about "is the leading key used", not transfer)
m=train_quadgram_model(rows,target_wpm=90.0,geometry=ROW_STAGGERED_31,random_state=0,**CAND4)
booster=m._regressor.get_booster()
score=booster.get_score(importance_type="gain")  # {f0:g,...}
# map f-index -> name
names=QUADGRAM_FEATURE_NAMES
imp=np.zeros(len(names))
for k,v in score.items():
    idx=int(k[1:]); imp[idx]=v
total=imp.sum()
# group by prefix
groups={"tg1_ (LEADING key trigram-level)":0.0,"tg2_ (last-3 trigram-level)":0.0,"bg1_ (LEADING bigram a,b)":0.0,"bg2_ (b,c)":0.0,"bg3_ (c,d)":0.0,"wpm":0.0}
for nm,g in zip(names,imp):
    if nm.startswith("tg1_"): groups["tg1_ (LEADING key trigram-level)"]+=g
    elif nm.startswith("tg2_"): groups["tg2_ (last-3 trigram-level)"]+=g
    elif nm.startswith("bg1_"): groups["bg1_ (LEADING bigram a,b)"]+=g
    elif nm.startswith("bg2_"): groups["bg2_ (b,c)"]+=g
    elif nm.startswith("bg3_"): groups["bg3_ (c,d)"]+=g
    elif nm=="wpm": groups["wpm"]+=g
print(f"total gain={total:.1f}")
print("=== gain share by block ===")
for k,v in groups.items():
    print(f"  {k:38s} {100*v/total:6.2f}%")
lead = groups["tg1_ (LEADING key trigram-level)"]+groups["bg1_ (LEADING bigram a,b)"]
print(f"\n  LEADING-KEY blocks (tg1_ + bg1_) TOTAL: {100*lead/total:.2f}%  <-- the 4th key's contribution")
# top 8 individual columns
order=np.argsort(imp)[::-1][:10]
print("\n=== top 10 columns by gain ===")
for i in order:
    print(f"  {names[i]:28s} {100*imp[i]/total:5.2f}%")
