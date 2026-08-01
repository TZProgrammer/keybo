"""TASK 1 analysis: best-of-N curve by without-replacement subsampling, on BOTH rulers."""
from __future__ import annotations
import json, sys
import numpy as np

POOL = "/local/home/zegertho/agent/state/searchparams/artifacts/t1_pool.json"
OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t1_curve.json"
FLOOR = 0.135  # campaign resolution floor, ms/char (PREREG:10405)

d = json.load(open(POOL)); runs = d["runs"]
fit = np.array([r["fitness"] for r in runs])
mpc = np.array([r["ms_per_char"] for r in runs])
sec = np.array([r["sec"] for r in runs])
outer = np.array([r["outer_count"] for r in runs])
lays = [r["layout"] for r in runs]
n = len(runs)
rng = np.random.default_rng(2026)

def curve(vals, ladder, draws=2000):
    """best-of-N WITHOUT replacement (what a user running --attempts N actually gets)."""
    out = {}
    for N in ladder:
        if N > n: continue
        if N == n:
            mins = np.array([vals.min()])
        else:
            idx = np.array([rng.choice(n, size=N, replace=False) for _ in range(draws)])
            mins = vals[idx].min(axis=1)
        out[N] = {"mean": float(mins.mean()), "median": float(np.median(mins)),
                  "p10": float(np.percentile(mins, 10)), "p90": float(np.percentile(mins, 90)),
                  "sd": float(mins.std(ddof=0)), "min": float(mins.min()), "max": float(mins.max())}
    return out

ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256]
res = {"meta": d["meta"], "floor_ms_per_char": FLOOR, "n_pool": n}

# --- single-attempt spread (Task 1b) ---
best_mpc, best_fit = mpc.min(), fit.min()
res["single_attempt_spread"] = {
    "ms_per_char": {"mean": float(mpc.mean()), "sd": float(mpc.std(ddof=1)), "min": float(mpc.min()),
                    "max": float(mpc.max()), "p10": float(np.percentile(mpc, 10)),
                    "p90": float(np.percentile(mpc, 90)), "range": float(mpc.max() - mpc.min()),
                    "p90_minus_p10": float(np.percentile(mpc, 90) - np.percentile(mpc, 10))},
    "fitness": {"mean": float(fit.mean()), "sd": float(fit.std(ddof=1)), "min": float(fit.min()),
                "max": float(fit.max()), "range_pct_of_best": float((fit.max()-fit.min())/fit.min()*100)},
    "P_single_worse_than_best_by_floor": float((mpc >= best_mpc + FLOOR).mean()),
    "P_single_worse_than_best_by_half_floor": float((mpc >= best_mpc + FLOOR/2).mean()),
    "wall_sec_per_attempt": {"mean": float(sec.mean()), "sd": float(sec.std(ddof=1))},
    "outer_iterations": {"mean": float(outer.mean()), "min": int(outer.min()), "max": int(outer.max()),
                         "T_end_over_T0_mean": float(np.mean(0.999**outer))},
}
# design-matched noise scale: the SEARCH-SEED spread of ms/char at fixed model (secondary ruler)
pairs = np.abs(mpc[:, None] - mpc[None, :])[np.triu_indices(n, 1)]
res["design_matched_noise_scale"] = {
    "definition": "median |d ms/char| over all C(n,2) SEARCH-SEED pairs at fixed model "
                  "(replicate structure matched to THIS question; the 0.135 campaign floor is a "
                  "MODEL-SEED floor over 91 board pairs -- PREREG:8610 says the quadruple must match)",
    "median": float(np.median(pairs)), "mean": float(pairs.mean()),
    "p90": float(np.percentile(pairs, 90)), "n_pairs": int(len(pairs)),
    "vs_campaign_floor": "LARGER than 0.135" if np.median(pairs) > FLOOR else "SMALLER than 0.135",
}
# --- best-of-N curves ---
res["curve_ms_per_char"] = curve(mpc, ladder)
res["curve_fitness"] = curve(fit, ladder)

# --- doubling deltas + saturation N* (pre-registered rule) ---
cm = res["curve_ms_per_char"]
doub = []
for i, N in enumerate(ladder[:-1]):
    N2 = ladder[i+1]
    if N2 not in cm: break
    delta = cm[N]["mean"] - cm[N2]["mean"]
    doub.append({"N": N, "2N": N2, "E_best_of_N": cm[N]["mean"], "E_best_of_2N": cm[N2]["mean"],
                 "delta_ms_per_char": float(delta), "below_floor": bool(delta < FLOOR)})
res["doubling_deltas_ms_per_char"] = doub
nstar = next((r["N"] for r in doub if r["below_floor"]), None)
res["saturation_N_star"] = nstar
res["saturation_rule"] = ("smallest N with E[best-of-N] - E[best-of-2N] < %.3f ms/char" % FLOOR)
if nstar is not None:
    gap1 = cm[1]["mean"] - cm[nstar]["mean"]
    res["default_verdict"] = {
        "shipped_default_attempts": 1,
        "E_best_of_1": cm[1]["mean"], "E_best_of_Nstar": cm[nstar]["mean"],
        "gap_ms_per_char": float(gap1), "floor": FLOOR,
        "verdict": "UNDER-POWERED (gap >= floor)" if gap1 >= FLOOR else "DEFAULT IS FINE (gap < floor)",
        "gap_in_floor_units": float(gap1 / FLOOR),
    }
    # also: how far is best-of-1 from the pool best (the strongest form of the question)?
    res["default_verdict"]["E_best_of_1_minus_pool_best"] = float(cm[1]["mean"] - best_mpc)
    res["default_verdict"]["E_best_of_1_minus_pool_best_in_floor_units"] = float((cm[1]["mean"] - best_mpc)/FLOOR)
    # fitness-ruler version
    cf = res["curve_fitness"]
    res["default_verdict"]["fitness_gap_pct_best_of_1_vs_pool_best"] = float(
        (cf[1]["mean"] - float(fit.min())) / float(fit.min()) * 100)

# --- degenerate-pool falsifier (pre-registered) ---
uniq = {}
for l, f in zip(lays, fit): uniq.setdefault(l, 0); uniq[l] += 1
best_layout = lays[int(np.argmin(fit))]
res["degenerate_pool_check"] = {
    "n_distinct_layouts": len(uniq), "n_runs": n,
    "seed_is_varying": bool(len(uniq) > 1),
    "best_layout": best_layout, "best_layout_hit_count": uniq[best_layout],
    "best_fitness": float(fit.min()), "best_ms_per_char": float(mpc[int(np.argmin(fit))]),
    "top5_most_repeated": sorted(uniq.items(), key=lambda kv: -kv[1])[:5],
    "argmin_on_two_rulers_agrees": bool(int(np.argmin(fit)) == int(np.argmin(mpc))),
    "best_on_gauge_layout": lays[int(np.argmin(mpc))], "best_on_gauge": float(mpc.min()),
}
json.dump(res, open(OUT, "w"), indent=1)
for k in ("single_attempt_spread","design_matched_noise_scale","doubling_deltas_ms_per_char",
          "saturation_N_star","default_verdict","degenerate_pool_check"):
    print("==", k, "=="); print(json.dumps(res[k], indent=1))
print("== curve ms/char =="); print(json.dumps(res["curve_ms_per_char"], indent=1))
