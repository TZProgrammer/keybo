"""TASK 2 confirmation: FRESH-SEED head-to-head, alpha=0.98 vs the shipped alpha=0.999.

Why this arm exists. My pre-registered rule ("some alpha beats the default by >= floor at >= 2 of
3 budgets") returned SCHEDULE MATTERS on the 48-seed sweep, but "some alpha" is a min over 9 arms
on those same seeds -- a winner's curse. My split-sample check was itself noisy (24 train seeds
picked alpha=0.995, a poor arm, at long budgets). So the clean test:

  - the DIRECTION ("faster cooling helps") was pre-registered from the pilot, BEFORE the sweep;
  - alpha=0.98 was the sweep's winner at short budgets;
  - so re-test that ONE pre-specified pair on 128 seeds NEVER USED IN THE ALPHA CHOICE (100..227).

This is a two-arm confirmation, no selection, matched seeds, matched wall clock.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import numpy as np
import _harness as H
from scipy.stats import binomtest, wilcoxon

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t2_confirm.json"
FLOOR = 0.135
SEEDS = list(range(100, 228))            # fresh: not used to choose alpha
PAIR = [0.999, 0.98]

sc = H.build_search_scorer()
data, t0 = {}, time.perf_counter()
for a in PAIR:
    runs = []
    for s in SEEDS:
        r = H.one_attempt(sc, seed=s, alpha=a)
        r["ms_per_char"] = H.ms_per_char(r["layout"])
        runs.append(r)
    data[str(a)] = runs
    print("  alpha=%g done (%.0fs)" % (a, time.perf_counter() - t0), flush=True)

rng = np.random.default_rng(31337)
def A(a, k): return np.array([r[k] for r in data[str(a)]])
res = {"seeds": [SEEDS[0], SEEDS[-1]], "n": len(SEEDS), "floor": FLOOR,
       "design": "two pre-specified arms, matched seeds, fresh seeds not used to choose alpha",
       "arms": {}}
for a in PAIR:
    fit, mpc, sec = A(a, "fitness"), A(a, "ms_per_char"), A(a, "sec")
    res["arms"][str(a)] = {"fitness_mean": float(fit.mean()), "ms_per_char_mean": float(mpc.mean()),
        "ms_per_char_sd": float(mpc.std(ddof=1)), "sec_per_attempt": float(sec.mean()),
        "outer_mean": float(A(a, "outer_count").mean()),
        "T_end_over_T0": float(np.mean(a ** A(a, "outer_count"))),
        "sec_sa": float(A(a, "sec_sa").mean()), "sec_2opt": float(A(a, "sec_2opt").mean())}

# paired, per-seed (single attempt)
f9, f8 = A(0.999, "fitness"), A(0.98, "fitness")
m9, m8 = A(0.999, "ms_per_char"), A(0.98, "ms_per_char")
wins = int((f8 < f9).sum()); n = len(SEEDS)
res["paired_single_attempt"] = {
    "alpha098_win_rate_on_objective": wins / n, "wins": wins, "n": n,
    "sign_test_p": float(binomtest(wins, n, 0.5).pvalue),
    "wilcoxon_p": float(wilcoxon(f8, f9).pvalue),
    "mean_d_ms_per_char_098_minus_999": float((m8 - m9).mean()),
    "in_floor_units": float((m8 - m9).mean() / FLOOR),
    "material_by_floor": bool(abs((m8 - m9).mean()) >= FLOOR)}

# EQUAL WALL CLOCK: best-of-N where N = floor(budget / that arm's own sec/attempt)
def at_budget(a, B, draws=6000):
    fit, mpc, sec = A(a, "fitness"), A(a, "ms_per_char"), A(a, "sec")
    N = max(1, int(B // sec.mean()))
    dr = rng.choice(len(fit), size=(draws, N), replace=True)
    pick = dr[np.arange(draws), fit[dr].argmin(axis=1)]
    return {"N": N, "E_ms_per_char": float(mpc[pick].mean()),
            "sd": float(mpc[pick].std(ddof=0)),
            "p90": float(np.percentile(mpc[pick], 90))}
res["equal_wall_clock"] = {}
clears = 0
for B in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    d9, d8 = at_budget(0.999, B), at_budget(0.98, B)
    gain = d9["E_ms_per_char"] - d8["E_ms_per_char"]
    ok = bool(gain >= FLOOR); clears += ok
    res["equal_wall_clock"][str(B)] = {"alpha0.999": d9, "alpha0.98": d8,
        "gain_ms_per_char": float(gain), "gain_in_floor_units": float(gain / FLOOR),
        "clears_floor": ok}
res["n_budgets_clearing"] = clears
res["verdict"] = ("SCHEDULE MATTERS: alpha=0.98 beats the shipped alpha=0.999 by >= 1 floor at %d of 8 "
                  "equal-wall-clock budgets, on fresh seeds" % clears) if clears >= 2 else \
                 ("SHIPPED ALPHA IS FINE: alpha=0.98's advantage is below the floor at "
                  "equal wall clock on fresh seeds (%d of 8 clear)" % clears)
json.dump({"meta": res, "runs": data}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
