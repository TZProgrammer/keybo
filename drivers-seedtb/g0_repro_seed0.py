"""GATE 0 commensurability test: does the RECOVERED recipe bit-reproduce shipped seed 0?

Recipe recovered from /local/home/zegertho/keybo-e2e/k31_train.py (the actual producer of
data/models/k31/*, per PREREGISTRATIONS.md 'K31 stages D-F OUTCOMES' + runs/k31_train.log).
Only the BIGRAM model is retrained here (fast, ~15s in the original log) -- if the bigram
does not reproduce, nothing else matters.

Thread vars are pinned by the WRAPPER before python starts (they are inert post-import).
"""
import hashlib
import json
import os
import sys
import time

MY_WT = "/local/home/zegertho/agent/workspaces/seedtb/wt"
sys.path.insert(0, MY_WT + "/src")
import keybo
assert keybo.__file__.startswith(MY_WT), f"WRONG keybo: {keybo.__file__}"
import xgboost
print("keybo:", keybo.__file__)
print("xgboost:", xgboost.__version__)
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    print(f"  {v}={os.environ.get(v)}")

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_31
from keybo.training.train import train_bigram_model

t0 = time.time()
E2E = "/local/home/zegertho/keybo-e2e"
rows = load_strokes(f"{E2E}/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
print(f"[{time.time()-t0:7.1f}s] {len(rows)} bigram rows (expect 2202)", flush=True)

OUT = "/local/home/zegertho/agent/state/seedtb/artifacts/g0"
os.makedirs(OUT, exist_ok=True)
# n_jobs=48 as in the original recipe
m = train_bigram_model(rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                       random_state=0, n_jobs=48)
m.save(f"{OUT}/bigram_reg31_seed0.json")
print(f"[{time.time()-t0:7.1f}s] trained+saved", flush=True)

import gzip
shipped = gzip.open("/local/home/zegertho/repos/keybo/data/models/k31/bigram_reg31_seed0.json.gz","rb").read()
mine = open(f"{OUT}/bigram_reg31_seed0.json","rb").read()
h_s, h_m = hashlib.sha256(shipped).hexdigest(), hashlib.sha256(mine).hexdigest()
print("shipped sha256:", h_s)
print("mine    sha256:", h_m)
print("BYTE-IDENTICAL:", h_s == h_m)
# structural compare if not byte-identical
ds, dm = json.loads(shipped), json.loads(mine)
ts = ds["learner"]["gradient_booster"]["model"]["trees"]
tm = dm["learner"]["gradient_booster"]["model"]["trees"]
print("n_trees shipped/mine:", len(ts), len(tm))
print("base_score shipped/mine:", ds["learner"]["learner_model_param"]["base_score"],
      dm["learner"]["learner_model_param"]["base_score"])
import numpy as np
if len(ts)==len(tm):
    difs=[]
    for a,b in zip(ts,tm):
        la, lb = np.array(a["base_weights"],dtype=float), np.array(b["base_weights"],dtype=float)
        if la.shape!=lb.shape:
            difs.append(np.inf); continue
        difs.append(float(np.max(np.abs(la-lb))))
    print("max |base_weight| delta over trees:", max(difs))
    print("n trees with identical structure:", sum(1 for d in difs if d==0.0), "/", len(difs))
# practice term compare
msh = json.loads(gzip.open("/local/home/zegertho/repos/keybo/data/models/k31/bigram_reg31_seed0.meta.json.gz","rb").read())
mym = json.load(open(f"{OUT}/bigram_reg31_seed0.meta.json"))
ps = msh["extra"]["training"]["practice_term"]["values"]; pm = mym["extra"]["training"]["practice_term"]["values"]
print("practice n_ngrams shipped/mine:", len(ps), len(pm))
if set(ps)==set(pm):
    dd=max(abs(ps[k]-pm[k]) for k in ps)
    print("max |practice delta|:", dd)
json.dump({"shipped_sha256":h_s,"mine_sha256":h_m,"byte_identical":h_s==h_m,
           "n_trees":[len(ts),len(tm)],"wall_s":time.time()-t0},
          open(f"{OUT}/g0_bigram_repro.json","w"), indent=1)
print(f"[{time.time()-t0:7.1f}s] ALL-DONE")
