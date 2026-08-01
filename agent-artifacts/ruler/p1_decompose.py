"""WHY the naive 'TableBigramScorer + TableTrigramScorer' sum is NOT the gauge (1.5-1.8% off).
Three candidate causes, isolated one at a time."""
MY = "/local/home/zegertho/repos/keybo-wt-ruler"
import keybo; assert keybo.__file__.startswith(MY + "/")
import numpy as np
from keybo.analysis.timecard import default_surface, _load_gz_model, TimeSurface
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring import model_norm as MN
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.scoring.table_trigram import TableTrigramScorer

WPM=90.0; C30M=MN.S.C30M
surf = default_surface(WPM); tri = surf.tri
bg = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
L = Layout(C30M, ROW_STAGGERED_30)
card = surf.card(C30M)
print("gauge total_ms      =", card.total_ms)

# --- cause 1: the BIGRAM TERM's weighting. gauge weights T2[a,b] by the TRIGRAM corpus
#     marginal over the first two chars, NOT by bigrams.txt.
ci = {c:i for i,c in enumerate(C30M)}; ci[" "]=30
marg = {}
for tg,f in tri.items():
    if len(tg)==3 and all(c in set(C30M)|{" "} for c in tg):
        marg[tg[:2]] = marg.get(tg[:2],0)+f
print("\n-- bigram-term weighting --")
print("  bigrams.txt kept mass  :", sum(v for k,v in bg.items() if len(k)==2 and all(c in set(C30M)|{' '} for c in k)))
print("  trigram first-2 marginal mass:", sum(marg.values()))
tb_bgtxt = TableBigramScorer(_load_gz_model("bigram_reg31_seed0"), bg,   target_wpm=WPM, chars=C30M)
tb_marg  = TableBigramScorer(_load_gz_model("bigram_reg31_seed0"), marg, target_wpm=WPM, chars=C30M)
print("  T2 term via bigrams.txt      :", tb_bgtxt.fitness(L))
print("  T2 term via trigram marginal :", tb_marg.fitness(L))
# gauge's own T2 term (seed-mean), computed from the surface directly
p = np.arange(31)
slot={c:i for i,c in enumerate(C30M)}; slot[" "]=30
t2_gauge=0.0; tc_gauge=0.0
for tg,f in tri.items():
    try: a,b,c = slot[tg[0]],slot[tg[1]],slot[tg[2]]
    except KeyError: continue
    t2_gauge += surf._T2[a,b]*f; tc_gauge += surf._Tc[a,b,c]*f
print("  gauge T2 term (seed-MEAN, tri-marginal weights):", t2_gauge)
print("  gauge Tc term                                   :", tc_gauge)
print("  T2+Tc =", t2_gauge+tc_gauge, " (card.total_ms =", card.total_ms, ")")

# --- cause 2: SEED AVERAGING (gauge = mean of seeds 0,1,2; --model gives ONE) ---
print("\n-- seed averaging --")
tt0 = TableTrigramScorer(_load_gz_model("trigram_cond31_seed0"), tri, target_wpm=WPM, chars=C30M)
print("  Tc term seed0 only:", tt0.fitness(L), " seed-MEAN:", tc_gauge,
      " rel dev %.4e" % (abs(tt0.fitness(L)-tc_gauge)/tc_gauge))
print("  T2 term seed0 only:", tb_marg.fitness(L), " seed-MEAN:", t2_gauge,
      " rel dev %.4e" % (abs(tb_marg.fitness(L)-t2_gauge)/t2_gauge))

# --- attribute the naive 1.5% ---
naive = tb_bgtxt.fitness(L) + tt0.fitness(L)
print("\n-- attribution of the naive error (C30M) --")
print("  naive (bigrams.txt + seed0 Tc)      : %.6e  rel %.4e" % (naive, abs(naive-card.total_ms)/card.total_ms))
fix_w = tb_marg.fitness(L) + tt0.fitness(L)
print("  fix weighting only (seed0)          : %.6e  rel %.4e" % (fix_w, abs(fix_w-card.total_ms)/card.total_ms))
print("  fix both (seed-mean + tri weights)  : %.6e  rel %.4e" % (t2_gauge+tc_gauge, abs(t2_gauge+tc_gauge-card.total_ms)/card.total_ms))

# --- cause 3: does TableTrigramScorer == TrigramModelScorer (the fast-path parity claim)? ---
print("\n-- TableTrigramScorer vs TrigramModelScorer (fast-path parity) --")
from keybo.scoring.model_scorer import TrigramModelScorer
small = dict(list(tri.items())[:400])
m = _load_gz_model("trigram_cond31_seed0")
tt = TableTrigramScorer(m, small, target_wpm=WPM, chars=C30M)
tm = TrigramModelScorer(m, small, target_wpm=WPM)
for name,lay in (("C30M",C30M),("dvorak","',.pyfgcrlaoeuidhtns;qjkxbmwvz")):
    if set(lay)!=set(C30M): print("  skip",name,"(different charset)"); continue
    LL=Layout(lay,ROW_STAGGERED_30)
    a,b = tt.fitness(LL), tm.fitness(LL)
    print(f"  {name:7s} table {a:.10e}  model {b:.10e}  rel {abs(a-b)/abs(b):.3e}")
