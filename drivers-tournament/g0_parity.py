"""GATE 0 — reproduce SEEDTB-1's published per-seed values from MY independent path (D4),
resolve the colemak-dh ambiguity, and prove per-seed tables are layout-independent.

Runs on the 3 SHIPPED seeds only (no training). Must pass before I spend 30-60 min retraining.
"""
import json, sys, gzip, shutil, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/local/home/zegertho/agent/workspaces/tournament/wt/drivers-tournament")
from _guard import assert_d5, BOARDS, COLEMAK_DH_VARIANTS, ART, SHIPPED

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"D5 OK keybo={assert_d5()}")

from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.analysis.timecard import TimeSurface

WPM = 90.0

def load_shipped(stem):
    """Inflate a vendored .gz model pair and load it (mirrors timecard._load_gz_model)."""
    with tempfile.TemporaryDirectory() as td:
        for suf in (".json", ".meta.json"):
            with gzip.open(f"{SHIPPED}/{stem}{suf}.gz", "rb") as s, open(Path(td)/f"{stem}{suf}", "wb") as d:
                shutil.copyfileobj(s, d)
        return XGBoostTypingModel.load(str(Path(td)/f"{stem}.json"))

POS = [*G.slots, G.space_position]
N = len(POS)
log(f"geometry ROW_STAGGERED_30: {len(G.slots)} slots, N={N} (with space)")

def build_tables(bi, tri):
    """Per-seed (T2, Tc) position-indexed tables. LAYOUT-INDEPENDENT -- the whole point."""
    placeholder = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(G.slots)]
    T2 = TableBigramScorer(bi, {}, target_wpm=WPM, chars=placeholder, geometry=G)._T
    vecs = np.vstack([trigram_features_from_positions(G, (a, b, c), wpm=WPM)
                      for a in POS for b in POS for c in POS])
    Tc = tri.predict_ms(vecs).reshape(N, N, N)
    return np.asarray(T2, dtype=float), np.asarray(Tc, dtype=float)

tables = []
for s in (0, 1, 2):
    bi, tri = load_shipped(f"bigram_reg31_seed{s}"), load_shipped(f"trigram_cond31_seed{s}")
    tables.append(build_tables(bi, tri))
    log(f"built tables seed{s}")

tri_freq = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
tri_freq = {k: v for k, v in tri_freq.items() if len(k) == 3}
log(f"corpus blend-v1: {len(tri_freq)} trigrams, mass {sum(tri_freq.values())}")

def score(lay, T2, Tc):
    slot = {ch: i for i, ch in enumerate(lay)}; slot[" "] = N - 1
    tot = 0.0; cov = 0
    for ng, f in tri_freq.items():
        try: a, b, c = slot[ng[0]], slot[ng[1]], slot[ng[2]]
        except KeyError: continue
        cov += f; tot += (T2[a, b] + Tc[a, b, c]) * f
    return tot / max(cov, 1), tot, cov

# ---- D4: reproduce SEEDTB-1's published per-seed ms/char for the 5 cluster boards -------------
seedtb = json.load(open("/local/home/zegertho/agent/state/seedtb/artifacts/margins_n25.json"))
worst = 0.0; d4 = {}
for nm in ("arm-B", "F(2.5)", "BALL-1", "F(2.0)", "candidate"):
    pub = seedtb["mspc"][nm][:3]
    mine = [score(BOARDS[nm], *tables[s])[0] for s in (0, 1, 2)]
    dif = [abs(a - b) for a, b in zip(pub, mine)]
    worst = max(worst, max(dif)); d4[nm] = {"published": pub, "mine": mine, "absdiff": dif}
    log(f"D4 {nm:12s} worst |diff| = {max(dif):.3e}")
log(f"D4 WORST ABSOLUTE DELTA ACROSS 15 PUBLISHED VALUES = {worst:.3e}")

# ---- positive control vs the SHIPPED code path (seed-mean tables) -----------------------------
surf = TimeSurface(tri_freq, target_wpm=WPM, keep_seed_tables=True)
pc = {}; worst_rel = 0.0
for nm in ("arm-B", "candidate", "qwerty"):
    shipped_card = surf.card(BOARDS[nm]).ms_per_char
    T2m = np.mean([t[0] for t in tables], axis=0); Tcm = np.mean([t[1] for t in tables], axis=0)
    mine = score(BOARDS[nm], T2m, Tcm)[0]
    rel = abs(mine - shipped_card) / shipped_card
    worst_rel = max(worst_rel, rel); pc[nm] = {"shipped": shipped_card, "mine": mine, "rel": rel}
    log(f"PC {nm:12s} shipped={shipped_card:.9f} mine={mine:.9f} rel={rel:.3e}")
    st_pub = surf.seed_totals(BOARDS[nm])
    st_mine = [score(BOARDS[nm], *tables[s])[1] for s in (0, 1, 2)]
    r2 = max(abs(a-b)/a for a, b in zip(st_pub, st_mine))
    log(f"   seed_totals rel dev = {r2:.3e}")
    pc[nm]["seed_totals_rel"] = r2
log(f"POSITIVE CONTROL worst rel dev vs shipped card() = {worst_rel:.3e}")

# ---- resolve the registered colemak-dh ambiguity ----------------------------------------------
TARGET = 258.75802535209823  # ship2 covonly.json
T2m = np.mean([t[0] for t in tables], axis=0); Tcm = np.mean([t[1] for t in tables], axis=0)
dh = {}
for nm, lay in COLEMAK_DH_VARIANTS.items():
    v = score(lay, T2m, Tcm)[0]
    dh[nm] = {"string": lay, "ms_per_char": v, "absdiff_vs_prior": abs(v - TARGET)}
    log(f"colemak-dh {nm:32s} {v:.9f}  |diff vs prior {TARGET}| = {abs(v-TARGET):.3e}")
winner = min(dh, key=lambda k: dh[k]["absdiff_vs_prior"])
log(f"=> colemak-dh RECONCILING VARIANT: {winner}")

# ---- prove the tables are layout-independent (so ONE retrain scores ALL boards) ---------------
li = {"T2_shape": list(tables[0][0].shape), "Tc_shape": list(tables[0][1].shape),
      "note": "position-indexed; no layout enters build_tables()"}
log(f"table shapes T2{li['T2_shape']} Tc{li['Tc_shape']} -- layout-independent by construction")

json.dump({"d4": d4, "d4_worst_absdiff": worst, "positive_control": pc,
           "pc_worst_rel": worst_rel, "colemak_dh": dh, "colemak_dh_winner": winner,
           "layout_independence": li, "wall_s": time.time() - t0},
          open(f"{ART}/g0_parity.json", "w"), indent=1)
log("ALL-DONE")
