"""K31 stage D (rule 2542bc4): production retrain on the K31 tables + LOLO gate.

Gate: LOLO on bistrokes31_v1 (4 folds x 2 seeds, production recipe) must hold
tau = 1.0 and rho/ceiling within 3% of the v5 baseline 1.0236. Pass => train + save
bigram_reg31_seed{0,1,2} and trigram_cond31_seed{0,1,2}.

Trigram params are the production CAND4 (join_lograt.py) — the trigram surface never
adopted the REG-LOLO bigram params. FRAME NOTE (recorded deviation): production
trigram models were trained on the JOIN frame (old tristrokes_v1 x tristrokes_last,
pre-BUF2); K31 uses the one-pass BUF2 conditioned frame (the charter's registered
choice — the old frames cannot carry the quote slot). The trigram LOLO is reported
against the direct-LOGRAT baseline 0.9928 as sanity, not a hard gate.
"""

import json
import os
import sys
import time

import numpy as np

# Only needed when keybo is not installed (no `pip install -e .`); harmless otherwise.
if os.environ.get("KEYBO_SRC"):
    sys.path.insert(0, os.environ["KEYBO_SRC"])

from keybo.data.strokes import load_strokes
from keybo.geometry import ROW_STAGGERED_31
from keybo.training.train import train_bigram_model, train_trigram_model
from keybo.training.validate import validate

SEEDS = [0, 1, 2]
BASELINE_RHO = 1.0236
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


log("loading bistrokes31_v1.tsv")
rows = load_strokes("bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} bigram rows")

# ---- LOLO gate ------------------------------------------------------------------------------
report = validate(rows, seeds=[0, 1], ngram="bigram", n_boot=10,
                  geometry=ROW_STAGGERED_31)
fracs = [m["rho_frac_ceiling"] for fold in report["folds"].values()
         for m in fold["seeds"] if m["rho_frac_ceiling"] is not None]
taus = [p["tau_heldout"] for p in report["pooled"]]
rho = float(np.mean(fracs))
gate = bool(min(taus) >= 1.0 - 1e-9 and rho >= BASELINE_RHO * 0.97)
log(f"LOLO gate: taus {taus} rho/ceiling {rho:.4f} (baseline {BASELINE_RHO}) "
    f"=> {'PASS' if gate else 'FAIL'}")
json.dump({"taus": taus, "rho_frac": rho, "gate_pass": gate},
          open("runs/k31_lolo_gate.json", "w"), indent=1)
if not gate:
    log("GATE FAIL — stopping before training (registered consequence: report to user)")
    log("ALL-DONE")
    sys.exit(0)

# ---- production retrain ----------------------------------------------------------------------
for seed in SEEDS:
    m = train_bigram_model(rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                           random_state=seed, n_jobs=48)
    m.save(f"models/bigram_reg31_seed{seed}.json")
    log(f"saved bigram_reg31_seed{seed}")
del rows

CAND4 = dict(
    n_estimators=427, max_depth=5, learning_rate=0.10903767015375725,
    min_child_weight=6, subsample=0.6086566147198375,
    colsample_bytree=0.9893815206317236,
    # CAND4 predates REG-LOLO; explicitly neutralize the adopted bigram regularization
    # defaults so the trigram params match production exactly.
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
)

log("loading tristrokes31_cond_v1.tsv")
tri_rows = load_strokes("tristrokes31_cond_v1.tsv", ngram_len=3, wpm_threshold=0,
                        min_samples=1)
log(f"{len(tri_rows)} cond-trigram rows")

tri_report = validate(tri_rows, seeds=[0, 1], ngram="trigram", n_boot=10,
                      geometry=ROW_STAGGERED_31, train_params=CAND4)
tri_fracs = [m["rho_frac_ceiling"] for fold in tri_report["folds"].values()
             for m in fold["seeds"] if m["rho_frac_ceiling"] is not None]
tri_taus = [p["tau_heldout"] for p in tri_report["pooled"]]
log(f"trigram LOLO sanity: taus {tri_taus} rho/ceiling {float(np.mean(tri_fracs)):.4f} "
    f"(direct-LOGRAT baseline 0.9928)")
json.dump({"taus": tri_taus, "rho_frac": float(np.mean(tri_fracs))},
          open("runs/k31_tri_sanity.json", "w"), indent=1)

for seed in SEEDS:
    m = train_trigram_model(tri_rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                            random_state=seed, n_jobs=48, **CAND4)
    m.save(f"models/trigram_cond31_seed{seed}.json")
    log(f"saved trigram_cond31_seed{seed}")
log("ALL-DONE")
