"""FIND-pass probe 1: build the REAL 31x31 QAP instance from SHIPPED data + SHIPPED
models via the SHIPPED TableBigramScorer, then interrogate the SHIPPED GL bound.

No hand-rolled reimplementation of F, T, or the bound (trap 28): every object comes
from the shipped constructor. The only new code is the interrogation.
"""
import gzip, json, os, shutil, tempfile
import numpy as np

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import certificate, gilmore_lawler_bound, qap_fitness

ROOT = "/tmp/qapaudit"
QWERTY = NAMED_LAYOUTS["qwerty"]
geom = ROW_STAGGERED_30


def load_freq(path):
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            out[p[0]] = int(p[1])
    return out


def load_model(stem):
    """Ungzip the shipped model pair into a temp dir and load via the SHIPPED loader."""
    d = tempfile.mkdtemp()
    for suf in (".json", ".meta.json"):
        src = f"{ROOT}/data/models/k31/{stem}{suf}.gz"
        with gzip.open(src, "rb") as fi, open(f"{d}/{stem}{suf}", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return XGBoostTypingModel.load(f"{d}/{stem}.json")


bi_corpus = load_freq(f"{ROOT}/data/corpus/blend-v1/bigrams.txt")
models = [load_model(f"bigram_reg31_seed{s}") for s in (0, 1, 2)]
scorers = [TableBigramScorer(m, bi_corpus, target_wpm=90.0, chars=QWERTY) for m in models]

# The DRIVER's exact construction (cond_rebuild.py:117-118): seed-mean T, seed-0 F.
T = np.mean([sc._T for sc in scorers], axis=0)
F = scorers[0]._F

print(f"instance: F {F.shape} T {T.shape}")
print(f"F finite: {np.isfinite(F).all()}  T finite: {np.isfinite(T).all()}")
print(f"F range [{F.min():.4g}, {F.max():.4g}]  T range [{T.min():.4g}, {T.max():.4g}]")
print(f"F symmetric: {np.allclose(F, F.T)}   T symmetric: {np.allclose(T, T.T)}")
print(f"F diag sum {np.diag(F).sum():.6g}   F offdiag sum {(F.sum()-np.diag(F).sum()):.6g}")
# CRITICAL: is F all-nonneg? GL's rearrangement step needs nonneg to be a *minimum*.
print(f"F min {F.min():.6g} (nonneg={F.min()>=0})   T min {T.min():.6g} (nonneg={T.min()>=0})")

# ---- the bound on the REAL instance -------------------------------------------------
lb = gilmore_lawler_bound(F, T)
print(f"\nGL lower bound on the real 31x31 instance: {lb:.6f}")
assert np.isfinite(lb), "bound is not finite"

# ---- named layouts: bound <= actual? ------------------------------------------------
sc = scorers[0]
print("\n--- named layouts: fitness vs bound ---")
rows = []
for name, s in NAMED_LAYOUTS.items():
    try:
        perm = sc.permutation(Layout(s, geom))
    except (ValueError, KeyError):
        continue
    fit = qap_fitness(F, T, perm)
    assert np.isfinite(fit)
    gap = (fit - lb) / lb * 100
    rows.append((name, fit, gap))
    print(f"  {name:<24} fit {fit:>16.4f}  gap over bound {gap:>8.3f}%  bound<=fit: {lb <= fit}")

# ---- random permutations: does the bound ever invert? --------------------------------
rng = np.random.default_rng(20260728)
NPOS = F.shape[0]
worst = np.inf
n_inv = 0
best_rand = np.inf
for _ in range(200_000):
    p = np.append(rng.permutation(30), 30)   # space PINNED at slot 30, as the driver does
    f = qap_fitness(F, T, p)
    best_rand = min(best_rand, f)
    if f < lb:
        n_inv += 1
    worst = min(worst, f - lb)
print(f"\n200k random perms (space pinned): min fitness {best_rand:.4f}")
print(f"  inversions (fitness < bound): {n_inv}")
print(f"  min (fitness - bound): {worst:.4f}")

# ---- FREE-space permutations (bound does NOT know space is pinned) -------------------
worst_free = np.inf; n_inv_free = 0; best_free = np.inf
for _ in range(200_000):
    p = rng.permutation(NPOS)
    f = qap_fitness(F, T, p)
    best_free = min(best_free, f)
    if f < lb:
        n_inv_free += 1
    worst_free = min(worst_free, f - lb)
print(f"\n200k FREE perms (space unpinned): min fitness {best_free:.4f}")
print(f"  inversions: {n_inv_free}   min (fitness - bound): {worst_free:.4f}")

json.dump({
    "lb": float(lb), "named": [(n, float(f), float(g)) for n, f, g in rows],
    "best_rand_pinned": float(best_rand), "n_inv_pinned": n_inv,
    "best_rand_free": float(best_free), "n_inv_free": n_inv_free,
    "F_min": float(F.min()), "T_min": float(T.min()),
    "F_diag_sum": float(np.diag(F).sum()),
}, open("/tmp/qapaudit/agent-artifacts/qapaudit/probe1.json", "w"), indent=2)
print("\nPROBE1-DONE")
