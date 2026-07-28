"""FIND-pass probe 3: THE COMPONENT QUESTION.

What the CERTIFICATE bounds vs what the SEARCH minimizes, measured on the real
shipped instance.

Certificate (cond_rebuild.py:256): certificate(F2, T2, qap_fitness(F2, T2, best_perm))
  -> bounds  fit_bi(p) = sum F2[i,j] * T2[p[i],p[j]]         (cond_rebuild.py:147-148)
Search    (cond_rebuild.py:220-221): fit_fn = fit_tri_corrected  (when simplify) ...
  -> minimizes fit_tri_corrected(p) = sum F3 * T3c[...]      (cond_rebuild.py:143-144)
  ... where T3c = T2[:,:,None] + Tcond                        (cond_rebuild.py:130)

So the question is: how big is the objective the search actually minimized, relative
to the piece the certificate bounds? All F/T objects come from SHIPPED constructors.
"""
import gzip, json, shutil, tempfile
import numpy as np

from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import certificate, gilmore_lawler_bound, qap_fitness

ROOT = "/tmp/qapaudit"
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30
N = 30

def load_freq(path):
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            out[p[0]] = int(p[1])
    return out

def load_model(stem):
    d = tempfile.mkdtemp()
    for suf in (".json", ".meta.json"):
        with gzip.open(f"{ROOT}/data/models/k31/{stem}{suf}.gz", "rb") as fi, open(f"{d}/{stem}{suf}", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return XGBoostTypingModel.load(f"{d}/{stem}.json")

bi_corpus  = load_freq(f"{ROOT}/data/corpus/blend-v1/bigrams.txt")
tri_corpus = load_freq(f"{ROOT}/data/corpus/blend-v1/trigrams.txt")

# --- bigram side: EXACTLY the driver's construction (cond_rebuild.py:115-118) ---------
bi_models = [load_model(f"bigram_reg31_seed{s}") for s in (0, 1, 2)]
bts = [TableBigramScorer(m, bi_corpus, target_wpm=90.0, chars=QWERTY) for m in bi_models]
T2 = np.mean([sc._T for sc in bts], axis=0)
F2 = bts[0]._F

# --- trigram side: EXACTLY the driver's construction (cond_rebuild.py:120-140) --------
from keybo.scoring.model_scorer import predict_ms
cond_models = [load_model(f"trigram_cond31_seed{s}") for s in (0, 1, 2)]
positions31 = [*geom.slots, geom.space_position]
n31 = len(positions31)
vec_all = np.vstack([trigram_features_from_positions(geom, (a, b, c), wpm=90.0)
                     for a in positions31 for b in positions31 for c in positions31])
Tcond = np.mean([predict_ms(m, vec_all).reshape(n31, n31, n31) for m in cond_models], axis=0)
T3c = T2[:, :, None] + Tcond
assert np.isfinite(T3c).all() and np.isfinite(Tcond).all()

char_idx = {c: i for i, c in enumerate(QWERTY)}; char_idx[" "] = N
charset = set(QWERTY) | {" "}
ks = [(char_idx[t[0]], char_idx[t[1]], char_idx[t[2]], f)
      for t, f in tri_corpus.items() if len(t) == 3 and all(c in charset for c in t)]
I3 = np.array([k[0] for k in ks]); J3 = np.array([k[1] for k in ks])
L3 = np.array([k[2] for k in ks]); F3 = np.array([k[3] for k in ks], dtype=float)
print(f"corpus: {len(ks)} trigram cells, total freq {F3.sum():.4g}")

def fit_bi(p):            return float((F2 * T2[np.ix_(p, p)]).sum())
def fit_tri_corrected(p): return float((F3 * T3c[p[I3], p[J3], p[L3]]).sum())
def fit_combined(p):      return fit_bi(p) + fit_tri_corrected(p)
# The T2-only part *inside* the trigram objective (the first-bigram physics term):
def tri_T2part(p):        return float((F3 * T2[p[I3], p[J3]]).sum())
def tri_condpart(p):      return float((F3 * Tcond[p[I3], p[J3], p[L3]]).sum())

# --- the bound + certificate: SHIPPED path, on (F2, T2) -------------------------------
lb = gilmore_lawler_bound(F2, T2)
assert np.isfinite(lb) and lb > 0

sc = bts[0]
print("\n=== MAGNITUDES on real layouts (all finite-asserted) ===")
rows = {}
for name in ("qwerty", "colemak"):
    p = sc.permutation(Layout(NAMED_LAYOUTS[name], geom))
    b, t, c = fit_bi(p), fit_tri_corrected(p), fit_combined(p)
    t2p, cp = tri_T2part(p), tri_condpart(p)
    for v in (b, t, c, t2p, cp): assert np.isfinite(v)
    rows[name] = dict(bi=b, tri=t, comb=c, tri_T2part=t2p, tri_condpart=cp,
                      cert_gap_pct=(b - lb) / lb * 100,
                      bi_share_of_comb=b / c * 100, tri_share_of_comb=t / c * 100)
    print(f"\n{name}:")
    print(f"  fit_bi              (CERTIFIED)   {b:>18.2f}   = {b/c*100:5.2f}% of combined")
    print(f"  fit_tri_corrected   (UNCERTIFIED) {t:>18.2f}   = {t/c*100:5.2f}% of combined")
    print(f"     ...of which T2 physics part    {t2p:>18.2f}")
    print(f"     ...of which Tcond increment    {cp:>18.2f}")
    print(f"  fit_combined                      {c:>18.2f}")
    print(f"  certificate gap on the bi part:   {(b-lb)/lb*100:.3f}%")

print(f"\nGL lower bound on (F2,T2): {lb:.2f}")
print(f"ratio  uncertified/certified (qwerty): {rows['qwerty']['tri']/rows['qwerty']['bi']:.3f}x")

# --- DOES THE BOUND TRANSFER? A bound on fit_bi is NOT a bound on fit_combined ---------
# fit_combined = fit_bi + fit_tri >= lb + min_p fit_tri(p).  We have no bound on fit_tri,
# so the only *valid* statement about combined is lb + 0 (if fit_tri >= 0), i.e.:
print("\n=== WHAT A READER WOULD INFER vs WHAT IS PROVEN ===")
p_q = sc.permutation(Layout(NAMED_LAYOUTS["qwerty"], geom))
comb_q = fit_combined(p_q)
naive_gap_if_read_as_combined = (comb_q - lb) / lb * 100
print(f"If a reader applies the certified bound {lb:.2f} to the COMBINED fitness of qwerty")
print(f"  ({comb_q:.2f}), the apparent gap is {naive_gap_if_read_as_combined:.1f}% — not {rows['qwerty']['cert_gap_pct']:.2f}%.")
print(f"  i.e. reading the certificate as covering the searched objective understates the")
print(f"  uncertified mass by a factor of ~{rows['qwerty']['tri']/rows['qwerty']['bi']:.1f}.")

# --- Is the certificate's own quantity even the search's argmax criterion? -------------
# Rank named + random layouts by fit_bi vs by fit_tri_corrected: if the orderings differ,
# the certified component is not a monotone proxy for the searched one.
rng = np.random.default_rng(4242)
cand = []
for _ in range(400):
    p = np.append(rng.permutation(N), N)
    cand.append((fit_bi(p), fit_tri_corrected(p), fit_combined(p)))
A = np.array(cand)
from scipy.stats import spearmanr
r_bi_tri  = spearmanr(A[:, 0], A[:, 1]).statistic
r_bi_comb = spearmanr(A[:, 0], A[:, 2]).statistic
print(f"\nover 400 random perms: spearman(fit_bi, fit_tri_corrected) = {r_bi_tri:.4f}")
print(f"                       spearman(fit_bi, fit_combined)      = {r_bi_comb:.4f}")

json.dump({"lb": float(lb), "rows": rows,
           "naive_gap_if_combined_qwerty": float(naive_gap_if_read_as_combined),
           "spearman_bi_tri": float(r_bi_tri), "spearman_bi_comb": float(r_bi_comb),
           "n_tri_cells": len(ks)},
          open("/tmp/qapaudit/agent-artifacts/qapaudit/probe3.json", "w"), indent=2)
print("\nPROBE3-DONE")
