"""TASK 2, corrected: two defects in my OWN first t2 verdict, fixed.

DEFECT 1 -- BUDGET CEILING. At the 16s budget every arm could afford N=48, i.e. the whole seed
pool, so N was clipped by POOL SIZE not by the budget: the 16s column compares "all 48" vs
"all 48" and is not a wall-clock comparison at all. Fix: only report a budget as valid for an arm
if N_affordable < n_seeds (headroom), and sample WITH replacement for budgets that need N > pool
(labelled), so the curve does not silently saturate.

DEFECT 2 -- WINNER'S CURSE. "best alpha" was a min over 9 arms on the same 48 seeds, so the
reported gain is upward-biased by selection. Fix: SPLIT-SAMPLE. Pick the best alpha on seeds
0..23, then quote its gain on the HELD-OUT seeds 24..47. That is an honest out-of-sample gain.
"""
from __future__ import annotations
import json
import numpy as np

SWEEP = "/local/home/zegertho/agent/state/searchparams/artifacts/t2_sweep.json"
OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t2_fair.json"
FLOOR = 0.135
d = json.load(open(SWEEP))
by = {a["label"]: a for a in d["arms"]}
ALPHA_ARMS = [l for l in by if l.startswith("alpha=") and "NO-2opt" not in l and "max_outer" not in l]
rng = np.random.default_rng(777)
def arr(a, k): return np.array([r[k] for r in a["runs"]])
NS = len(d["seeds"])

def best_of_at_budget(fit, mpc, sec_mean, B, draws=4000, pool_idx=None):
    """E[ms/char of argmin-fitness] over N = floor(B/sec) draws WITH replacement.

    With-replacement is the right model once N approaches the pool: it treats the pool as an
    estimate of the run distribution rather than clipping N at the pool size (defect 1).
    """
    idx_pool = np.arange(len(fit)) if pool_idx is None else pool_idx
    N = max(1, int(B // sec_mean))
    draw = rng.choice(idx_pool, size=(draws, N), replace=True)
    pick = draw[np.arange(draws), fit[draw].argmin(axis=1)]
    return N, float(mpc[pick].mean()), bool(N <= len(idx_pool))

res = {"floor": FLOOR, "n_seeds": NS, "defects_fixed": [
    "budget ceiling: with-replacement draws so N is set by the BUDGET, never clipped by pool size",
    "winner's curse: alpha chosen on seeds 0-23, gain quoted on held-out seeds 24-47"]}

BUDGETS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
res["curve"] = {}
for lab in ALPHA_ARMS:
    a = by[lab]; fit, mpc, sec = arr(a, "fitness"), arr(a, "ms_per_char"), arr(a, "sec")
    row = {"sec_per_attempt": float(sec.mean()), "at_budget": {}}
    for B in BUDGETS:
        N, E, within = best_of_at_budget(fit, mpc, sec.mean(), B)
        row["at_budget"][str(B)] = {"N": N, "E_ms_per_char": E, "N_within_pool": within}
    res["curve"][lab] = row

# ---- split-sample, winner's-curse-free verdict ----
tr, te = np.arange(0, NS // 2), np.arange(NS // 2, NS)
split = {"train_seeds": [0, NS//2-1], "test_seeds": [NS//2, NS-1], "per_budget": {}}
clears = 0
for B in BUDGETS:
    picks = {}
    for lab in ALPHA_ARMS:
        a = by[lab]; fit, mpc, sec = arr(a, "fitness"), arr(a, "ms_per_char"), arr(a, "sec")
        _, E_tr, _ = best_of_at_budget(fit, mpc, sec[tr].mean(), B, pool_idx=tr)
        picks[lab] = E_tr
    winner = min(picks, key=picks.get)
    out = {}
    for lab in (winner, "alpha=0.999"):
        a = by[lab]; fit, mpc, sec = arr(a, "fitness"), arr(a, "ms_per_char"), arr(a, "sec")
        N, E_te, _ = best_of_at_budget(fit, mpc, sec[te].mean(), B, pool_idx=te)
        out[lab] = {"N": N, "E_ms_per_char_heldout": E_te}
    gain = out["alpha=0.999"]["E_ms_per_char_heldout"] - out[winner]["E_ms_per_char_heldout"]
    ok = bool(gain >= FLOOR and winner != "alpha=0.999")
    clears += ok
    split["per_budget"][str(B)] = {"winner_on_train": winner, "heldout": out,
        "gain_vs_default_heldout_ms_per_char": float(gain),
        "gain_in_floor_units": float(gain / FLOOR), "clears_floor": ok}
split["budgets_clearing_floor"] = int(clears)
split["n_budgets"] = len(BUDGETS)
split["verdict"] = ("THE SCHEDULE MATTERS (out-of-sample, at matched wall clock)" if clears >= 2
                    else "THE SHIPPED ALPHA IS FINE at matched wall clock (out-of-sample)")
res["split_sample_verdict"] = split

# ---- the SIMPLEST fair statement: same wall clock, default alpha vs best alpha ----
# equal-time comparison at the DEFAULT's own cost, i.e. "if I have T seconds, which alpha?"
res["headline_equal_time"] = {}
for B in (1.0, 4.0, 16.0):
    rows = [(l, res["curve"][l]["at_budget"][str(B)]["E_ms_per_char"],
             res["curve"][l]["at_budget"][str(B)]["N"]) for l in ALPHA_ARMS]
    rows.sort(key=lambda t: t[1])
    res["headline_equal_time"][str(B)] = {
        "ranking": [{"alpha": l, "E_ms_per_char": v, "N_affordable": n} for l, v, n in rows],
        "default_rank": [l for l, _, _ in rows].index("alpha=0.999") + 1,
        "spread_best_to_worst": float(rows[-1][1] - rows[0][1])}
json.dump(res, open(OUT, "w"), indent=1)
print("== equal-wall-clock ranking (with-replacement, N set by budget) ==")
for B in ("1.0", "4.0", "16.0"):
    h = res["headline_equal_time"][B]
    print(" budget %ss  (default alpha ranks %d/%d, spread %.4f ms/char)" % (B, h["default_rank"], len(ALPHA_ARMS), h["spread_spread"] if False else h["spread_best_to_worst"]))
    for r in h["ranking"]: print("    %-14s N=%-4d E=%.4f" % (r["alpha"], r["N_affordable"], r["E_ms_per_char"]))
print("\n== SPLIT-SAMPLE (winner's-curse-free) verdict ==")
print(json.dumps(split, indent=1))
