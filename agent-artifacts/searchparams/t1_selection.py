"""TASK 1 correction: the SELECTION-CONSISTENT best-of-N curve.

The naive curve takes min(gauge) over the subset. That is NOT what a user gets: `--attempts N`
selects the argmin of the SEARCH OBJECTIVE (fitness) and reports THAT layout. Since the two
rulers disagree on the pool argmin (verified), selecting on the gauge would be an oracle the
user does not have. This driver reports the honest, selection-consistent curve:

    argmin over subset by FITNESS  ->  read that layout's ms/char

and quantifies the oracle-vs-honest gap (the "ruler disagreement tax").
"""
from __future__ import annotations
import json
import numpy as np

POOL = "/local/home/zegertho/agent/state/searchparams/artifacts/t1_pool.json"
OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t1_curve_selection_consistent.json"
FLOOR = 0.135

d = json.load(open(POOL)); runs = d["runs"]
fit = np.array([r["fitness"] for r in runs]); mpc = np.array([r["ms_per_char"] for r in runs])
n = len(runs); rng = np.random.default_rng(2026)
ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def curve(draws=4000):
    honest, oracle = {}, {}
    for N in ladder:
        if N > n: continue
        if N == n:
            idx = np.arange(n)[None, :]
        else:
            idx = np.array([rng.choice(n, size=N, replace=False) for _ in range(draws)])
        pick = idx[np.arange(idx.shape[0]), fit[idx].argmin(axis=1)]   # select on FITNESS
        got = mpc[pick]                                               # report its gauge value
        orc = mpc[idx].min(axis=1)                                    # oracle: select on gauge
        honest[N] = {"mean": float(got.mean()), "median": float(np.median(got)),
                     "p10": float(np.percentile(got, 10)), "p90": float(np.percentile(got, 90)),
                     "sd": float(got.std(ddof=0)), "max": float(got.max()), "min": float(got.min())}
        oracle[N] = {"mean": float(orc.mean())}
    return honest, oracle

honest, oracle = curve()
res = {"n_pool": n, "floor_ms_per_char": FLOOR,
       "selection_rule": "argmin of the SEARCH OBJECTIVE within the subset, then read that "
                         "layout's ms/char gauge -- exactly what `keybo optimize --attempts N` returns",
       "curve_honest_ms_per_char": honest,
       "curve_oracle_select_on_gauge": oracle,
       "ruler_disagreement_tax_ms_per_char": {str(N): honest[N]["mean"] - oracle[N]["mean"] for N in honest}}
doub = []
for i, N in enumerate(ladder[:-1]):
    N2 = ladder[i+1]
    if N2 not in honest: break
    delta = honest[N]["mean"] - honest[N2]["mean"]
    doub.append({"N": N, "2N": N2, "E_N": honest[N]["mean"], "E_2N": honest[N2]["mean"],
                 "delta_ms_per_char": float(delta), "below_floor": bool(delta < FLOOR)})
res["doubling_deltas"] = doub
nstar = next((r["N"] for r in doub if r["below_floor"]), None)
res["saturation_N_star_honest"] = nstar
if nstar:
    gap = honest[1]["mean"] - honest[nstar]["mean"]
    res["default_verdict_honest"] = {
        "shipped_default_attempts": 1, "E_best_of_1": honest[1]["mean"],
        "E_best_of_Nstar": honest[nstar]["mean"], "gap_ms_per_char": float(gap),
        "gap_in_floor_units": float(gap / FLOOR),
        "verdict": "UNDER-POWERED" if gap >= FLOOR else "DEFAULT IS FINE"}
    res["default_verdict_honest"]["E_1_minus_pool_floor"] = float(honest[1]["mean"] - honest[256]["mean"])
    res["default_verdict_honest"]["E_1_minus_pool_floor_in_floor_units"] = float(
        (honest[1]["mean"] - honest[256]["mean"]) / FLOOR)
# how badly do the rulers disagree? rank correlation over the pool
from scipy.stats import spearmanr, pearsonr
res["ruler_agreement"] = {
    "spearman_fitness_vs_ms_per_char": float(spearmanr(fit, mpc).statistic),
    "pearson": float(pearsonr(fit, mpc).statistic),
    "note": "search objective = bigram single-seed table; gauge = K31 trigram 3-seed-mean surface. "
            "A correlation < 1 means selecting on the objective does NOT always pick the "
            "gauge-best layout -- an irreducible tax the restart budget cannot remove.",
    "pool_argmin_fitness_ms_per_char": float(mpc[int(fit.argmin())]),
    "pool_argmin_gauge_ms_per_char": float(mpc.min()),
    "tax_at_pool_scale": float(mpc[int(fit.argmin())] - mpc.min()),
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
