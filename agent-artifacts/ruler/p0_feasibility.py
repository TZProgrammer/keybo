"""Prototype: can the gauge be expressed as a TableTrigramScorer-shaped (31,31,31) table,
and does the NAIVE 'wire TableTrigramScorer + TableBigramScorer' reading of the brief
actually reconcile to analyze? Measures parity + eval speed."""
import os, sys, time
MY = "/local/home/zegertho/repos/keybo-wt-both"
import keybo
assert keybo.__file__.startswith(MY + "/"), keybo.__file__
import numpy as np
from keybo.analysis.timecard import default_surface, _load_gz_model
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring import model_norm as MN
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.scoring.table_trigram import TableTrigramScorer

WPM = 90.0
surf = default_surface(WPM)
tri = surf.tri
C30M = MN.S.C30M
BOARDS = {"C30M": C30M, **{k: v for k, v in NAMED_LAYOUTS.items()}}
# only 30-char boards whose charset is a permutation of a 30-char set
print("=== boards ===")
for k, v in BOARDS.items():
    print(" ", k, repr(v), len(v), len(set(v)))

# ---- the combined gauge table: T3g[a,b,c] = T2[a,b] + Tc[a,b,c] ----
T3g = surf._T2[:, :, None] + surf._Tc
print("T3g", T3g.shape)

class GaugeProto(TableTrigramScorer):
    def __init__(self, chars, tri_freqs, T3):
        # bypass model predict; replicate the corpus indexing
        self._chars = tuple(chars); self._geometry = ROW_STAGGERED_30
        self._T3 = T3
        charset = set(self._chars) | {" "}
        ci = {c: i for i, c in enumerate(self._chars)}; ci[" "] = len(self._chars)
        I, J, L, F = [], [], [], []
        for tg, f in tri_freqs.items():
            if len(tg) == 3 and all(c in charset for c in tg):
                I.append(ci[tg[0]]); J.append(ci[tg[1]]); L.append(ci[tg[2]]); F.append(f)
        self._i = np.array(I, dtype=np.intp); self._j = np.array(J, dtype=np.intp)
        self._l = np.array(L, dtype=np.intp); self._f = np.array(F, dtype=np.float64)
        self._slot_index = {p: i for i, p in enumerate(ROW_STAGGERED_30.slots)}
        self._space_slot = len(ROW_STAGGERED_30.slots)
        self._covered = self._f.sum()
    def ms_per_char(self, layout): return self.fitness(layout) / self._covered

print("\n=== PARITY: combined-table scorer vs surf.card() ===")
worst_total = worst_mpc = 0.0
for name, lay in BOARDS.items():
    gp = GaugeProto(lay, tri, T3g)
    L = Layout(lay, ROW_STAGGERED_30)
    mine_t, mine_m = gp.fitness(L), gp.ms_per_char(L)
    card = surf.card(lay)
    rt = abs(mine_t - card.total_ms) / card.total_ms
    rm = abs(mine_m - card.ms_per_char) / card.ms_per_char
    worst_total = max(worst_total, rt); worst_mpc = max(worst_mpc, rm)
    print(f"  {name:9s} total rel {rt:.3e}  ms/char rel {rm:.3e}  (mine {mine_m:.10f} card {card.ms_per_char:.10f}) cov {card.coverage_pct:.4f}")
print(f"  WORST total rel {worst_total:.3e}   WORST ms/char rel {worst_mpc:.3e}")

print("\n=== THE BRIEF'S NAIVE READING: TableBigramScorer(bigram_reg31_seed0)+TableTrigramScorer(trigram_cond31_seed0) ===")
bg = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
bm = _load_gz_model("bigram_reg31_seed0"); tm = _load_gz_model("trigram_cond31_seed0")
for name, lay in list(BOARDS.items())[:3]:
    tb = TableBigramScorer(bm, bg, target_wpm=WPM, chars=lay)
    tt = TableTrigramScorer(tm, tri, target_wpm=WPM, chars=lay)
    L = Layout(lay, ROW_STAGGERED_30)
    naive = tb.fitness(L) + tt.fitness(L)
    card = surf.card(lay)
    print(f"  {name:9s} naive {naive:.6e}  card.total {card.total_ms:.6e}  rel dev {abs(naive-card.total_ms)/card.total_ms:.4e}")

print("\n=== eval speed (the thing that makes multi-restart search feasible) ===")
gp = GaugeProto(C30M, tri, T3g)
L = Layout(C30M, ROW_STAGGERED_30)
p = gp.permutation(L)
n = 300
t = time.perf_counter()
for _ in range(n): gp.fitness_of_permutation(p)
per = (time.perf_counter() - t) / n
print(f"  rows kept {len(gp._f)} / {len(tri)}   fitness_of_permutation {per*1e3:.3f} ms")
t = time.perf_counter()
for _ in range(20): gp.fitness(L)
print(f"  fitness(layout) {(time.perf_counter()-t)/20*1e3:.3f} ms   card() ~47 ms")
t = time.perf_counter(); surf.card(C30M); print(f"  card() measured {(time.perf_counter()-t)*1e3:.1f} ms")
