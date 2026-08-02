"""Train N model seeds (bigram + trigram PAIR per seed) with the EXACT recovered k31 recipe.

Usage: train_seeds.py <seed_start> <seed_end_inclusive>
Writes to OUT (my own dir) -- NEVER data/models/k31/.
Also DEGENERACY-CHECKS: sha256 of each artifact, so identical seeds are impossible to miss.
"""
import hashlib, json, os, sys, time
MY_WT = "/local/home/zegertho/agent/workspaces/seedtb/wt"
sys.path.insert(0, MY_WT + "/src")
import keybo
assert keybo.__file__.startswith(MY_WT), f"WRONG keybo: {keybo.__file__}"
import xgboost

OUT = "/local/home/zegertho/agent/workspaces/seedtb/models"
os.makedirs(OUT, exist_ok=True)
E2E = "/local/home/zegertho/keybo-e2e"
S0, S1 = int(sys.argv[1]), int(sys.argv[2])

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_31
from keybo.training.train import train_bigram_model, train_trigram_model

# EXACT CAND4 from /local/home/zegertho/keybo-e2e/k31_train.py
CAND4 = dict(
    n_estimators=427, max_depth=5, learning_rate=0.10903767015375725,
    min_child_weight=6, subsample=0.6086566147198375,
    colsample_bytree=0.9893815206317236,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
)
NJOBS = 48
t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)
log(f"xgboost {xgboost.__version__}; seeds {S0}..{S1}; n_jobs={NJOBS}; "
    f"threads={os.environ.get('OMP_NUM_THREADS')}")

def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
timings = {}

log("loading bistrokes31_v1.tsv")
rows = load_strokes(f"{E2E}/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} bigram rows")
assert len(rows) == 2202, f"bigram row count {len(rows)} != 2202 (frame drift!)"
for s in range(S0, S1+1):
    ts = time.time()
    m = train_bigram_model(rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                           random_state=s, n_jobs=NJOBS)
    m.save(f"{OUT}/bigram_reg31_seed{s}.json")
    timings[f"bigram_seed{s}"] = time.time()-ts
    log(f"saved bigram_reg31_seed{s}  ({time.time()-ts:.1f}s)  sha={sha(f'{OUT}/bigram_reg31_seed{s}.json')[:16]}")
del rows

log("loading tristrokes31_cond_v1.tsv")
tri_rows = load_strokes(f"{E2E}/tristrokes31_cond_v1.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)
log(f"{len(tri_rows)} cond-trigram rows")
assert len(tri_rows) == 16643, f"trigram row count {len(tri_rows)} != 16643 (frame drift!)"
for s in range(S0, S1+1):
    ts = time.time()
    m = train_trigram_model(tri_rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                            random_state=s, n_jobs=NJOBS, **CAND4)
    m.save(f"{OUT}/trigram_cond31_seed{s}.json")
    timings[f"trigram_seed{s}"] = time.time()-ts
    log(f"saved trigram_cond31_seed{s} ({time.time()-ts:.1f}s) sha={sha(f'{OUT}/trigram_cond31_seed{s}.json')[:16]}")

# --- D7: trigram determinism check (registered caveat) — reuse the loaded data ---
if os.environ.get("SEEDTB_D7") == "1":
    d7 = f"{OUT}/d7"; os.makedirs(d7, exist_ok=True)
    for s0 in (0, 1, 2):
        ts = time.time()
        m = train_trigram_model(tri_rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                                random_state=s0, n_jobs=NJOBS, **CAND4)
        m.save(f"{d7}/trigram_cond31_seed{s0}.json")
        import gzip as _gz
        ship = hashlib.sha256(_gz.open(
            f"/local/home/zegertho/repos/keybo/data/models/k31/trigram_cond31_seed{s0}.json.gz","rb").read()).hexdigest()
        mine_h = sha(f"{d7}/trigram_cond31_seed{s0}.json")
        timings[f"d7_trigram_seed{s0}"] = time.time()-ts
        log(f"D7 trigram seed{s0}: shipped={ship[:16]} mine={mine_h[:16]} "
            f"BYTE-IDENTICAL={ship==mine_h}")
        timings[f"d7_match_seed{s0}"] = (ship == mine_h)

json.dump({"timings_s": timings, "total_s": time.time()-t0, "n_jobs": NJOBS,
           "omp": os.environ.get("OMP_NUM_THREADS"), "xgboost": xgboost.__version__,
           "seeds": list(range(S0,S1+1)),
           "sha": {f"{k}_seed{s}": sha(f"{OUT}/{k}_seed{s}.json")
                   for k in ("bigram_reg31","trigram_cond31") for s in range(S0,S1+1)}},
          open(f"/local/home/zegertho/agent/state/seedtb/artifacts/train_{S0}_{S1}.json","w"), indent=1)
log("ALL-DONE")
