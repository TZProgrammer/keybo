"""POSITIVE CONTROL: my own per-seed table path must reproduce the SHIPPED path exactly.

Two independent reproductions on the ORIGINAL 3 shipped seeds:
  (1) my hand-built per-seed tables vs the shipped TimeSurface.seed_totals()
  (2) both vs BESTFINAL-1's published b06 seed_ms_per_char for all 5 cluster boards
If either fails, the tie-break is void.
"""
import json, sys, time
MY_WT = "/local/home/zegertho/agent/workspaces/seedtb/wt"
sys.path.insert(0, MY_WT + "/src")
import keybo
assert keybo.__file__.startswith(MY_WT), f"WRONG keybo: {keybo.__file__}"
import numpy as np
from keybo.analysis.timecard import TimeSurface, _load_gz_model
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.scoring.table_scorer import TableBigramScorer

t0=time.time()
BOARDS = {
 "arm-B":     "flmpg-yuo,sntdcireahkxbwv'.jzq",
 "F(2.5)":    "flmpg-,uoysntdcireahkxbwv.'jzq",
 "BALL-1":    "flmpg-yuo,sntcdireahkxbwv'.jzq",
 "F(2.0)":    "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
 "candidate": "pyu.,vdfnlhieaocstrmkj'-qgwbzx",
}
corpus_dir = production_corpus_dir(None)
print("corpus dir:", corpus_dir)
tri = load_frequencies(str(corpus_dir / "trigrams.txt"))
print(f"[{time.time()-t0:6.1f}s] corpus trigrams: {len(tri)} mass {sum(tri.values())}")

# ---- shipped path -------------------------------------------------------------------
surf = TimeSurface(tri, target_wpm=90.0, keep_seed_tables=True)
print(f"[{time.time()-t0:6.1f}s] shipped TimeSurface built (keep_seed_tables)")
shipped_totals = {n: surf.seed_totals(L) for n, L in BOARDS.items()}
cards = {n: surf.card(L) for n, L in BOARDS.items()}
covered = {n: c.total_ms / c.ms_per_char for n, c in cards.items()}
print("covered mass per board (must be equal across the 5):",
      {n: round(v,1) for n,v in covered.items()})

# ---- MY OWN path: build each seed's tables independently -----------------------------
GEOM = ROW_STAGGERED_30
positions = [*GEOM.slots, GEOM.space_position]
N = len(positions)
placeholder = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(GEOM.slots)]
vecs = np.vstack([trigram_features_from_positions(GEOM, (a,b,c), wpm=90.0)
                  for a in positions for b in positions for c in positions])

def seed_tables(bi_model, tri_model):
    T2 = TableBigramScorer(bi_model, {}, target_wpm=90.0, chars=placeholder, geometry=GEOM)._T
    Tc = tri_model.predict_ms(vecs).reshape(N, N, N)
    return T2, Tc

def board_total(T2, Tc, lay30):
    slot = {ch: i for i, ch in enumerate(lay30)}; slot[" "] = N-1
    tot = 0.0; cov = 0
    for ng, f in tri.items():
        if len(ng) != 3: continue
        try: a,b,c = slot[ng[0]], slot[ng[1]], slot[ng[2]]
        except KeyError: continue
        tot += (T2[a,b] + Tc[a,b,c]) * f; cov += f
    return tot, cov

mine = {n: [] for n in BOARDS}; mine_cov = {}
for s in (0,1,2):
    T2, Tc = seed_tables(_load_gz_model(f"bigram_reg31_seed{s}"),
                         _load_gz_model(f"trigram_cond31_seed{s}"))
    for n, L in BOARDS.items():
        tot, cov = board_total(T2, Tc, L)
        mine[n].append(tot); mine_cov[n] = cov
    print(f"[{time.time()-t0:6.1f}s] my seed {s} done")

# ---- (1) my totals vs shipped seed_totals -------------------------------------------
print("\n=== CHECK 1: my per-seed totals vs shipped TimeSurface.seed_totals() ===")
worst1 = 0.0
for n in BOARDS:
    for s in range(3):
        rel = abs(mine[n][s]-shipped_totals[n][s]) / shipped_totals[n][s]
        worst1 = max(worst1, rel)
print(f"worst relative deviation: {worst1:.3e}   PASS={worst1 < 1e-12}")

# ---- (2) ms_per_char vs BESTFINAL b06 ------------------------------------------------
B06_RAW = json.load(open("/local/home/zegertho/agent/state/bestfinal/artifacts/b06_r3_seed_stability.json"))["rows"]
B06 = {("candidate" if k.startswith("CANDIDATE") else k): v for k, v in B06_RAW.items()}
print("\n=== CHECK 2: my ms_per_char vs BESTFINAL-1 b06 published per-seed values ===")
print(f"{'board':<11} {'seed':>4} {'mine':>16} {'BESTFINAL b06':>16} {'abs delta':>12}")
worst2 = 0.0; mypc = {}
for n, L in BOARDS.items():
    mypc[n] = [t/mine_cov[n] for t in mine[n]]
    for s in range(3):
        ref = B06[n]["seed_ms_per_char"][s]
        d = abs(mypc[n][s]-ref); worst2 = max(worst2, d)
        print(f"{n:<11} {s:>4} {mypc[n][s]:>16.10f} {ref:>16.10f} {d:>12.2e}")
    print(f"{n:<11} {'mean':>4} {np.mean(mypc[n]):>16.10f} {B06[n]['mean_ms_per_char']:>16.10f} "
          f"{abs(np.mean(mypc[n])-B06[n]['mean_ms_per_char']):>12.2e}")
print(f"\nworst abs delta vs b06: {worst2:.3e}   PASS={worst2 < 1e-8}")
print(f"arm-B mean ms/char (target 253.900579): {np.mean(mypc['arm-B']):.6f}")
json.dump({"worst_rel_vs_seed_totals": worst1, "worst_abs_vs_b06": worst2,
           "my_ms_per_char": mypc, "covered": mine_cov,
           "arm_B_mean": float(np.mean(mypc["arm-B"]))},
          open("/local/home/zegertho/agent/state/seedtb/artifacts/pc_parity.json","w"), indent=1)
print(f"[{time.time()-t0:6.1f}s] ALL-DONE")
