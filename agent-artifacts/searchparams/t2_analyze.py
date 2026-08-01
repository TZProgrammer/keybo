"""TASK 2 analysis: cost/quality curve at MATCHED WALL CLOCK + matched-seed paired tests."""
from __future__ import annotations
import json
import numpy as np

SWEEP = "/local/home/zegertho/agent/state/searchparams/artifacts/t2_sweep.json"
OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t2_curve.json"
FLOOR = 0.135
BUDGETS = [1.0, 4.0, 16.0]

d = json.load(open(SWEEP)); arms = d["arms"]
by = {a["label"]: a for a in arms}
rng = np.random.default_rng(4242)

def arr(a, k): return np.array([r[k] for r in a["runs"]])

res = {"floor": FLOOR, "n_seeds": len(d["seeds"]), "budgets_sec": BUDGETS,
       "shipped_default_arm": "alpha=0.999", "arms": {}}

for lab, a in by.items():
    fit, mpc, sec = arr(a, "fitness"), arr(a, "ms_per_char"), arr(a, "sec")
    outer = arr(a, "outer_count")
    n = len(fit)
    row = {"kw": a["kw"], "fitness_mean": float(fit.mean()), "fitness_sd": float(fit.std(ddof=1)),
           "ms_per_char_mean": float(mpc.mean()), "ms_per_char_sd": float(mpc.std(ddof=1)),
           "sec_per_attempt": float(sec.mean()), "outer_mean": float(outer.mean()),
           "T_end_over_T0_mean": float(np.mean(np.asarray(a["kw"].get("alpha", 0.999)) ** outer)),
           "sa_only_ms_gap": None}
    # best-of-N at a fixed TIME budget: N = floor(budget / sec_per_attempt), selection on FITNESS
    row["at_budget"] = {}
    for B in BUDGETS:
        N = max(1, int(B // sec.mean()))
        N = min(N, n)
        idx = np.arange(n)[None, :] if N == n else np.array(
            [rng.choice(n, size=N, replace=False) for _ in range(4000)])
        pick = idx[np.arange(idx.shape[0]), fit[idx].argmin(axis=1)]
        row["at_budget"][str(B)] = {"N_affordable": int(N),
            "E_ms_per_char": float(mpc[pick].mean()),
            "E_fitness": float(fit[idx].min(axis=1).mean())}
    res["arms"][lab] = row

# --- matched-seed paired head-to-head vs the shipped alpha=0.999 ---
base = by["alpha=0.999"]
bfit, bmpc = arr(base, "fitness"), arr(base, "ms_per_char")
from scipy.stats import binomtest, wilcoxon
for lab, a in by.items():
    if lab == "alpha=0.999": continue
    fit, mpc = arr(a, "fitness"), arr(a, "ms_per_char")
    wins = int((fit < bfit).sum()); n = len(fit)
    d_mpc = mpc - bmpc
    res["arms"][lab]["vs_default_paired"] = {
        "win_rate_on_objective": wins / n, "wins": wins, "n": n,
        "sign_test_p": float(binomtest(wins, n, 0.5).pvalue),
        "wilcoxon_p_on_objective": float(wilcoxon(fit, bfit).pvalue) if n > 5 else None,
        "mean_d_ms_per_char": float(d_mpc.mean()),
        "mean_d_ms_per_char_in_floor_units": float(d_mpc.mean() / FLOOR),
        "material_by_floor": bool(abs(d_mpc.mean()) >= FLOOR),
        "note": ("directionally consistent, MAGNITUDE BELOW THE FLOOR"
                 if (binomtest(wins, n, 0.5).pvalue < 0.05 and abs(d_mpc.mean()) < FLOOR) else ""),
    }

# --- 2-opt's worth, matched alpha + matched seeds ---
res["two_opt_worth"] = {}
for a_ in (0.999, 0.99, 0.95, 0.9):
    on, off = by.get("alpha=%g" % a_), by.get("alpha=%g NO-2opt" % a_)
    if not (on and off): continue
    m_on, m_off = arr(on, "ms_per_char"), arr(off, "ms_per_char")
    f_on, f_off = arr(on, "fitness"), arr(off, "fitness")
    res["two_opt_worth"]["alpha=%g" % a_] = {
        "ms_per_char_gain_from_2opt": float(m_off.mean() - m_on.mean()),
        "gain_in_floor_units": float((m_off.mean() - m_on.mean()) / FLOOR),
        "fitness_gain_pct": float((f_off.mean() - f_on.mean()) / f_on.mean() * 100),
        "sec_cost_of_2opt": float(arr(on, "sec_2opt").mean()),
        "sec_sa": float(arr(on, "sec_sa").mean()),
        "2opt_share_of_wall_clock": float(arr(on, "sec_2opt").mean() / arr(on, "sec").mean()),
    }

# --- SCHEDULE-MATTERS verdict (pre-registered rule) ---
def best_alpha_at(B):
    alphas = {l: r for l, r in res["arms"].items() if l.startswith("alpha=") and "NO-2opt" not in l and "max_outer" not in l}
    return min(alphas.items(), key=lambda kv: kv[1]["at_budget"][str(B)]["E_ms_per_char"])
verdict = {"rule": "some alpha beats shipped alpha=0.999 by >= %.3f ms/char on E[best-of-N(BUDGET)] "
                   "at >= 2 of 3 budgets" % FLOOR, "per_budget": {}}
clears = 0
for B in BUDGETS:
    lab, row = best_alpha_at(B)
    base_v = res["arms"]["alpha=0.999"]["at_budget"][str(B)]["E_ms_per_char"]
    gain = base_v - row["at_budget"][str(B)]["E_ms_per_char"]
    ok = gain >= FLOOR
    clears += ok
    verdict["per_budget"][str(B)] = {"best_alpha": lab, "E_best": row["at_budget"][str(B)]["E_ms_per_char"],
        "E_default": base_v, "gain_vs_default_ms_per_char": float(gain),
        "gain_in_floor_units": float(gain / FLOOR), "clears_floor": bool(ok),
        "N_affordable_best": row["at_budget"][str(B)]["N_affordable"],
        "N_affordable_default": res["arms"]["alpha=0.999"]["at_budget"][str(B)]["N_affordable"]}
verdict["budgets_clearing_floor"] = int(clears)
verdict["verdict"] = ("THE SCHEDULE MATTERS" if clears >= 2 else "THE SHIPPED ALPHA IS FINE at matched wall clock")
res["schedule_verdict"] = verdict
json.dump(res, open(OUT, "w"), indent=1)

print("== per-arm cost/quality ==")
print("%-30s %10s %9s %8s %7s %8s" % ("arm","fit_mean","ms/char","sec/att","outer","T/T0"))
for l, r in res["arms"].items():
    print("%-30s %.4e %9.4f %8.3f %7.0f %8.3f" % (l, r["fitness_mean"], r["ms_per_char_mean"],
          r["sec_per_attempt"], r["outer_mean"], r["T_end_over_T0_mean"]))
print("\n== at matched wall clock (E[ms/char] of best-of-N affordable) ==")
print("%-30s %s" % ("arm", "  ".join("%5.0fs:N/E" % B for B in BUDGETS)))
for l, r in res["arms"].items():
    print("%-30s %s" % (l, "  ".join("%3d/%.4f" % (r["at_budget"][str(B)]["N_affordable"],
          r["at_budget"][str(B)]["E_ms_per_char"]) for B in BUDGETS)))
print("\n== 2-opt worth =="); print(json.dumps(res["two_opt_worth"], indent=1))
print("\n== schedule verdict =="); print(json.dumps(verdict, indent=1))
