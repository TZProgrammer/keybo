"""WHICH mis-specification causes the inversion: BIGRAM-vs-TRIGRAM, or 1-SEED-vs-3-SEED-MEAN?

My t4 comparison changed BOTH at once (default objective = bigram + one model seed; gauge =
bigram+trigram + 3-seed mean), so "the objective is mis-specified" was right but UNRESOLVED as to
WHY. That distinction decides which recommendation matters: wiring TableTrigramScorer into the
search (fixes the term) vs averaging model seeds (fixes the seed). Cheap to separate, so separate it.

Four scorers over the SAME 256 C30M layouts, ranked against the gauge:
  A  bigram,  1 seed   <- the shipped default objective
  B  bigram,  3-seed mean
  C  bi+tri,  1 seed
  D  bi+tri,  3-seed mean  <- IS the gauge (spearman must be 1.0; a self-check)
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import numpy as np
from scipy.stats import spearmanr
import _harness as H
from keybo.analysis.timecard import _load_gz_model, default_surface
from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.scoring import model_norm as MN
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.data.corpus import load_frequencies, production_corpus_dir

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t6_disentangle.json"
C30M = MN.S.C30M
pos = [*ROW_STAGGERED_30.slots, ROW_STAGGERED_30.space_position]
surf = default_surface(90.0)

# --- per-seed bigram T2 tables (same construction TimeSurface uses) ---
bigfreq = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
T2s = [TableBigramScorer(_load_gz_model("bigram_reg31_seed%d" % s), {}, target_wpm=90.0,
                         chars=C30M)._T for s in (0, 1, 2)]
# --- per-seed trigram Tc tables ---
vecs = np.vstack([trigram_features_from_positions(ROW_STAGGERED_30, (a, b, c), wpm=90.0)
                  for a in pos for b in pos for c in pos])
Tcs = [_load_gz_model("trigram_cond31_seed%d" % s).predict_ms(vecs).reshape(31, 31, 31)
       for s in (0, 1, 2)]

idx = {c: i for i, c in enumerate(C30M)}; idx[" "] = 30
I, J, L, F = [], [], [], []
for ng, f in surf.tri.items():
    if len(ng) != 3: continue
    try: a, b, c = idx[ng[0]], idx[ng[1]], idx[ng[2]]
    except KeyError: continue
    I.append(a); J.append(b); L.append(c); F.append(f)
I, J, L = map(np.array, (I, J, L)); F = np.array(F, float); COV = F.sum()
slot_of = {p: i for i, p in enumerate(ROW_STAGGERED_30.slots)}

def perm(lay30):
    p = np.empty(31, dtype=np.intp)
    for i, ch in enumerate(C30M): p[i] = lay30.index(ch)
    p[30] = 30
    return p

def score(lay30, T2list, Tclist, use_tri):
    p = perm(lay30); a, b, c = p[I], p[J], p[L]
    T2 = np.mean(T2list, axis=0); tot = T2[a, b]
    if use_tri: tot = tot + np.mean(Tclist, axis=0)[a, b, c]
    return float(F @ tot) / COV

d = json.load(open("/local/home/zegertho/agent/state/searchparams/artifacts/t1b_c30m.json"))
lays = [r["layout"] for r in d["runs"]]
gauge = np.array([r["ms_per_char"] for r in d["runs"]])

variants = {
 "A bigram, 1 seed (SHIPPED DEFAULT)": (T2s[:1], Tcs[:1], False),
 "B bigram, 3-seed mean":              (T2s,     Tcs,     False),
 "C bigram+trigram, 1 seed":           (T2s[:1], Tcs[:1], True),
 "D bigram+trigram, 3-seed mean (== the gauge)": (T2s, Tcs, True),
}
res = {"n_layouts": len(lays), "note": "spearman of each candidate objective against the campaign "
       "ms/char gauge, over 256 layouts produced by 256 restarts on the C30M charset", "variants": {}}
for lab, (t2, tc, tri) in variants.items():
    v = np.array([score(l, t2, tc, tri) for l in lays])
    res["variants"][lab] = {"spearman_vs_gauge": float(spearmanr(v, gauge).statistic),
        "argmin_matches_gauge_argmin": bool(int(v.argmin()) == int(gauge.argmin())),
        "gauge_of_this_objectives_pick": float(gauge[int(v.argmin())]),
        "selection_tax_vs_gauge_best": float(gauge[int(v.argmin())] - gauge.min()),
        "tax_in_floor_units": float((gauge[int(v.argmin())] - gauge.min()) / 0.135)}
a = res["variants"]["A bigram, 1 seed (SHIPPED DEFAULT)"]["spearman_vs_gauge"]
b = res["variants"]["B bigram, 3-seed mean"]["spearman_vs_gauge"]
c = res["variants"]["C bigram+trigram, 1 seed"]["spearman_vs_gauge"]
res["attribution"] = {
    "gain_from_averaging_model_seeds_only (A->B)": float(b - a),
    "gain_from_adding_the_trigram_term_only (A->C)": float(c - a),
    "verdict": ("THE TRIGRAM TERM is the dominant fix" if (c - a) > (b - a)
                else "MODEL-SEED AVERAGING is the dominant fix"),
    "reading": "spearman(objective, gauge) closer to 1 == the search would rank layouts the way "
               "the campaign reports them. D must be ~1.0; it is the self-check."}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
