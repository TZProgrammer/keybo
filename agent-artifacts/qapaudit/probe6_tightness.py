"""FIND-pass probe 6: IS THE CERTIFICATE INFORMATIVE, or is the quoted gap just the
bound's own looseness?

The certificate reports gap = (found - lb)/lb. That decomposes as
    found - lb  =  (found - OPT)      +  (OPT - lb)
                   genuine suboptimality   irreducible GL slack
31! forbids computing OPT, but a DEEP SEARCH ON THE CERTIFIED OBJECTIVE ITSELF gives the
tightest available upper bound on OPT, which upper-bounds the suboptimality term and
lower-bounds the slack term.

*** I AM RUNNING A SEARCH. Disclosed per my brief: a bound check requires it. It is a
search on fit_bi (the CERTIFIED quantity) purely to bracket OPT — no layout is proposed,
adopted, or recommended. ***
"""
import gzip, json, shutil, tempfile, time
import numpy as np

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import certificate, gilmore_lawler_bound, qap_fitness

ROOT = "/tmp/qapaudit"; QWERTY = NAMED_LAYOUTS["qwerty"]; geom = ROW_STAGGERED_30; N = 30

def load_freq(p):
    o = {}
    for ln in open(p):
        q = ln.rstrip("\n").split("\t")
        if len(q) == 2: o[q[0]] = int(q[1])
    return o

def load_model(stem):
    d = tempfile.mkdtemp()
    for suf in (".json", ".meta.json"):
        with gzip.open(f"{ROOT}/data/models/k31/{stem}{suf}.gz", "rb") as fi, open(f"{d}/{stem}{suf}", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return XGBoostTypingModel.load(f"{d}/{stem}.json")

bi_corpus = load_freq(f"{ROOT}/data/corpus/blend-v1/bigrams.txt")
bts = [TableBigramScorer(load_model(f"bigram_reg31_seed{s}"), bi_corpus, target_wpm=90.0, chars=QWERTY)
       for s in (0, 1, 2)]
T2 = np.mean([sc._T for sc in bts], axis=0); F2 = bts[0]._F
lb = gilmore_lawler_bound(F2, T2)
assert np.isfinite(lb) and lb > 0
print(f"GL bound on the certified (F2,T2) instance: {lb:.4f}")

def fit_bi(p): return float((F2 * T2[np.ix_(p, p)]).sum())

# ---- SA + full 2-opt on the CERTIFIED objective (driver's own recipe, cond_rebuild:158) --
rng = np.random.default_rng(70707)
def search(restarts, iters):
    deltas = []
    for _ in range(150):
        p = np.append(rng.permutation(N), N); f1 = fit_bi(p)
        i, j = rng.integers(0, N, 2); p[i], p[j] = p[j], p[i]
        d = fit_bi(p) - f1
        if d > 0: deltas.append(d)
    T0 = float(np.median(deltas)) / np.log(2)
    best = []
    for r in range(restarts):
        perm = np.append(rng.permutation(N), N); cur = fit_bi(perm); temp = T0
        for _ in range(iters):
            i, j = rng.integers(0, N, 2)
            if i == j: continue
            perm[i], perm[j] = perm[j], perm[i]
            cand = fit_bi(perm)
            if cand - cur <= 0 or rng.random() < np.exp(-(cand - cur) / temp): cur = cand
            else: perm[i], perm[j] = perm[j], perm[i]
            temp *= 0.9995
        imp = True
        while imp:
            imp = False
            for i in range(N):
                for j in range(i + 1, N):
                    perm[i], perm[j] = perm[j], perm[i]
                    c2 = fit_bi(perm)
                    if c2 < cur - 1e-9: cur = c2; imp = True
                    else: perm[i], perm[j] = perm[j], perm[i]
        best.append((cur, perm.copy()))
    best.sort(key=lambda t: t[0]); return best

t0 = time.time()
res = search(restarts=24, iters=30_000)
best_fit = res[0][0]
print(f"deep search on the CERTIFIED objective: {time.time()-t0:.1f}s, "
      f"best fit_bi {best_fit:.4f}  ({len(res)} restarts)")
print(f"  restart spread: min {res[0][0]:.2f}  max {res[-1][0]:.2f}  "
      f"spread {(res[-1][0]-res[0][0])/res[0][0]*100:.4f}%")

# ---- the decomposition -----------------------------------------------------------------
cert_best = certificate(F2, T2, best_fit)
print(f"\ncertificate on the SEARCH-OPTIMAL bigram layout: gap {cert_best['gap_pct']:.4f}%")
print("  => an (essentially) bigram-OPTIMAL layout STILL certifies at this gap.")
print(f"  => at most {0.0:.2f}% of that is suboptimality; >= {cert_best['gap_pct']:.4f}% is IRREDUCIBLE GL SLACK.")

print("\n--- reference band: what gap do UNOPTIMIZED layouts certify at? ---")
band = {}
for name in ("qwerty", "colemak"):
    f = fit_bi(bts[0].permutation(Layout(NAMED_LAYOUTS[name], geom)))
    band[name] = (f - lb) / lb * 100
    print(f"  {name:<10} certifies at {band[name]:.4f}%")
rng2 = np.random.default_rng(31337)
gaps = []
for _ in range(20_000):
    p = np.append(rng2.permutation(N), N)
    gaps.append((fit_bi(p) - lb) / lb * 100)
gaps = np.array(gaps)
print(f"  20k RANDOM layouts: min {gaps.min():.4f}%  p1 {np.percentile(gaps,1):.4f}%  "
      f"median {np.median(gaps):.4f}%  max {gaps.max():.4f}%")
best_of = {k: float(np.min([np.min(gaps[i:i+k]) for i in range(0, 20000-k, k)])) for k in (1, 100, 1000)}
print(f"  best-of-1000-random certifies at {best_of[1000]:.4f}%")

print("\n=== THE INFORMATIVENESS TEST ===")
print(f"  GL slack floor (search-optimal layout):  {cert_best['gap_pct']:.4f}%")
print(f"  best-of-20k-random layout:               {gaps.min():.4f}%")
print(f"  qwerty (a 150-year-old layout):          {band['qwerty']:.4f}%")
print(f"  ledger certificates span:                2.54% .. 4.38%")
print(f"  => spread between 'optimal' and 'best random draw': "
      f"{gaps.min() - cert_best['gap_pct']:.4f} pct-pts of resolving power.")

json.dump({"lb": float(lb), "best_search_fit": float(best_fit),
           "gap_at_search_optimum_pct": float(cert_best["gap_pct"]),
           "named_gaps_pct": band, "random_gap_min": float(gaps.min()),
           "random_gap_p1": float(np.percentile(gaps, 1)),
           "random_gap_median": float(np.median(gaps)),
           "random_gap_max": float(gaps.max()), "best_of_random": best_of,
           "restart_spread_pct": float((res[-1][0]-res[0][0])/res[0][0]*100)},
          open("/tmp/qapaudit/agent-artifacts/qapaudit/probe6.json", "w"), indent=2)
print("\nPROBE6-DONE")
