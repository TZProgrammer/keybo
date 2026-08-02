"""Train seeds 3..24 with the EXACT SEEDTB-1-recovered k31 recipe, and SAVE THE PER-SEED TABLES.

Saving the tables is the improvement over SEEDTB-1: (T2, Tc) are position-indexed and
LAYOUT-INDEPENDENT, so with them cached ANY future board comparison at n=25 needs zero retraining.
SEEDTB-1 retrained, scored 5 boards, then let its workspace (and the models) be destroyed.

Writes to my own dirs. data/models/k31/ is NEVER written.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/local/home/zegertho/agent/workspaces/tournament/wt/drivers-tournament")
from _guard import assert_d5, sha, E2E, OUT_MODELS, ART

S0, S1 = int(sys.argv[1]), int(sys.argv[2])
TABLES = "/local/home/zegertho/agent/workspaces/tournament/tables"
os.makedirs(OUT_MODELS, exist_ok=True); os.makedirs(TABLES, exist_ok=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

import xgboost
log(f"D5 OK keybo={assert_d5()}")
log(f"xgboost {xgboost.__version__} seeds {S0}..{S1} n_jobs=48 OMP={os.environ.get('OMP_NUM_THREADS')}")

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_31, ROW_STAGGERED_30 as G
from keybo.training.train import train_bigram_model, train_trigram_model
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.features import trigram_features_from_positions

# EXACT CAND4 from /local/home/zegertho/keybo-e2e/k31_train.py. NOT _DEFAULT_PARAMS --
# SEEDTB-1's load-bearing CORRECTION 2: the shipped trigram is 427 trees at depth 5.
CAND4 = dict(n_estimators=427, max_depth=5, learning_rate=0.10903767015375725,
             min_child_weight=6, subsample=0.6086566147198375,
             colsample_bytree=0.9893815206317236, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0)
NJOBS, WPM = 48, 90.0
POS = [*G.slots, G.space_position]; N = len(POS)
timings, shas = {}, {}

log("loading bistrokes31_v1.tsv")
rows = load_strokes(f"{E2E}/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} bigram rows")
assert len(rows) == 2202, f"D6 FAIL bigram rows {len(rows)} != 2202 (frame drift)"
bigrams = {}
for s in range(S0, S1 + 1):
    ts = time.time()
    m = train_bigram_model(rows, target_wpm=WPM, geometry=ROW_STAGGERED_31,
                           random_state=s, n_jobs=NJOBS)
    p = f"{OUT_MODELS}/bigram_reg31_seed{s}.json"; m.save(p)
    timings[f"bigram_seed{s}"] = time.time() - ts; shas[f"bigram_seed{s}"] = sha(p)
    # build the bigram half of the table NOW while the model is in hand
    placeholder = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(G.slots)]
    bigrams[s] = np.asarray(TableBigramScorer(m, {}, target_wpm=WPM, chars=placeholder,
                                              geometry=G)._T, dtype=float)
    log(f"saved bigram_reg31_seed{s} ({time.time()-ts:.1f}s) sha={shas[f'bigram_seed{s}'][:16]}")
del rows

log("loading tristrokes31_cond_v1.tsv")
tri_rows = load_strokes(f"{E2E}/tristrokes31_cond_v1.tsv", ngram_len=3, wpm_threshold=0, min_samples=1)
log(f"{len(tri_rows)} cond-trigram rows")
assert len(tri_rows) == 16643, f"D6 FAIL trigram rows {len(tri_rows)} != 16643 (frame drift)"
vecs = np.vstack([trigram_features_from_positions(G, (a, b, c), wpm=WPM)
                  for a in POS for b in POS for c in POS])
for s in range(S0, S1 + 1):
    ts = time.time()
    m = train_trigram_model(tri_rows, target_wpm=WPM, geometry=ROW_STAGGERED_31,
                            random_state=s, n_jobs=NJOBS, **CAND4)
    p = f"{OUT_MODELS}/trigram_cond31_seed{s}.json"; m.save(p)
    timings[f"trigram_seed{s}"] = time.time() - ts; shas[f"trigram_seed{s}"] = sha(p)
    Tc = np.asarray(m.predict_ms(vecs).reshape(N, N, N), dtype=float)
    np.savez_compressed(f"{TABLES}/tables_seed{s}.npz", T2=bigrams[s], Tc=Tc)
    log(f"saved trigram_cond31_seed{s} ({time.time()-ts:.1f}s) sha={shas[f'trigram_seed{s}'][:16]} "
        f"+ tables_seed{s}.npz")

json.dump({"timings_s": timings, "total_s": time.time() - t0, "n_jobs": NJOBS,
           "omp": os.environ.get("OMP_NUM_THREADS"), "xgboost": xgboost.__version__,
           "seeds": list(range(S0, S1 + 1)), "sha": shas,
           "bigram_rows": 2202, "trigram_rows": 16643},
          open(f"{ART}/train_{S0}_{S1}.json", "w"), indent=1)
log("ALL-DONE")
