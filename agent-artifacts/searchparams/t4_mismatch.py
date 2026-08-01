"""THE OBJECTIVE/GAUGE MISMATCH -- the finding T1b forced out, and it dominates everything.

T1b: 256 properly-powered searches on the SHIPPED default objective produced layouts that are
BETTER on that objective than BALL-1 (1.1727e11 vs 1.1840e11, -0.95%) yet WORSE on the campaign's
reporting gauge (255.63 vs 253.97 ms/char, +1.67). 0 of 256 beat BALL-1 on the gauge.

That is only possible if the shipped `keybo optimize` default objective is NOT the quantity the
campaign reports and ranks layouts by. This driver nails down exactly what the two are and
measures how badly they disagree, because it decides whether "the search is under-powered" is even
the right diagnosis: you cannot under-power your way to a layout that is good on a gauge you are
not optimizing.
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import numpy as np
import _harness as H
from keybo.analysis.timecard import default_surface
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring import model_norm as MN

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t4_mismatch.json"
FLOOR = 0.135
C30M = MN.S.C30M
res = {}

# ---- 1. what IS each quantity? ----
res["definitions"] = {
    "shipped_search_objective": {
        "path": "cli/optimize.py::run -> TableBigramScorer(model=--model, load_freqs, target_wpm, chars=--start)",
        "kind": "BIGRAM only (quadratic), ONE model artifact (whatever --model points at), "
                "bigram corpus weighting",
        "note": "this is what `keybo optimize` MINIMIZES on its default flags"},
    "campaign_reporting_gauge": {
        "path": "analysis/timecard.py::TimeSurface.card().ms_per_char via default_surface(90)",
        "kind": "BIGRAM T2 + TRIGRAM Tc (cubic), 3-SEED MEAN of bigram AND trigram models, "
                "trigram corpus weighting",
        "note": "this is what analyze prints, what the ledger's ms/char tables quote, and what "
                "the 0.135 resolution floor is defined on"},
}

# ---- 2. the cubic term is half the mass, and it is NOT in the search objective ----
surf = default_surface(90.0)
T2, Tc = surf._T2, surf._Tc
def split(lay30):
    slot = {c: i for i, c in enumerate(lay30)}; slot[" "] = 30
    t2 = t3 = 0.0; cov = 0
    for ng, f in surf.tri.items():
        try: a, b, c = slot[ng[0]], slot[ng[1]], slot[ng[2]]
        except KeyError: continue
        cov += f; t2 += T2[a, b] * f; t3 += Tc[a, b, c] * f
    return t2 / cov, t3 / cov

# ---- 3. measure the disagreement over the two 256-pools + the campaign boards ----
pools = {}
for tag, path in (("qwerty-charset", "t1_pool.json"), ("C30M-charset", "t1b_c30m.json")):
    d = json.load(open("/local/home/zegertho/agent/state/searchparams/artifacts/" + path))
    runs = d["runs"] if "runs" in d else d["runs"]
    fit = np.array([r["fitness"] for r in runs]); mpc = np.array([r["ms_per_char"] for r in runs])
    from scipy.stats import spearmanr, pearsonr
    pools[tag] = {"n": len(runs),
        "spearman_objective_vs_gauge": float(spearmanr(fit, mpc).statistic),
        "pearson": float(pearsonr(fit, mpc).statistic),
        "argmin_agrees": bool(int(fit.argmin()) == int(mpc.argmin())),
        "gauge_of_objective_best": float(mpc[int(fit.argmin())]),
        "gauge_best": float(mpc.min()),
        "selection_tax_ms_per_char": float(mpc[int(fit.argmin())] - mpc.min()),
        "selection_tax_in_floor_units": float((mpc[int(fit.argmin())] - mpc.min()) / FLOOR)}
res["pool_disagreement"] = pools

REF = {"BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq", "arm B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
       "MID": "flmpg.yuo,sntcdireahkxbwv'-jzq", "HEADLINE": "flmpg-,uoysntcdireahkxvwb.'jzq"}
d = json.load(open("/local/home/zegertho/agent/state/searchparams/artifacts/t1b_c30m.json"))
runs = d["runs"]; fit = np.array([r["fitness"] for r in runs]); mpc = np.array([r["ms_per_char"] for r in runs])
sc = H.build_search_scorer(start=C30M)
best_i = int(fit.argmin())
rows = []
for lab, l in list(REF.items()) + [("OUR best-on-objective (256 restarts)", runs[best_i]["layout"]),
                                   ("OUR best-on-gauge (oracle over 256)", runs[int(mpc.argmin())]["layout"])]:
    f = sc.fitness(Layout(l, ROW_STAGGERED_30)); g = H.ms_per_char(l)
    q, c = split(l)
    rows.append({"label": lab, "layout": l, "search_objective_fitness": f, "gauge_ms_per_char": g,
                 "gauge_quadratic_part": q, "gauge_cubic_part": c})
rows.sort(key=lambda r: r["search_objective_fitness"])
res["head_to_head"] = {"rows": rows,
    "ranking_by_objective": [r["label"] for r in rows],
    "ranking_by_gauge": [r["label"] for r in sorted(rows, key=lambda r: r["gauge_ms_per_char"])]}
ours = [r for r in rows if r["label"].startswith("OUR best-on-objective")][0]
ball = [r for r in rows if r["label"] == "BALL-1"][0]
res["the_inversion"] = {
    "our_objective_advantage_pct": float((ball["search_objective_fitness"] - ours["search_objective_fitness"])
                                          / ball["search_objective_fitness"] * 100),
    "our_gauge_deficit_ms_per_char": float(ours["gauge_ms_per_char"] - ball["gauge_ms_per_char"]),
    "our_gauge_deficit_in_floor_units": float((ours["gauge_ms_per_char"] - ball["gauge_ms_per_char"]) / FLOOR),
    "quadratic_part_delta": float(ours["gauge_quadratic_part"] - ball["gauge_quadratic_part"]),
    "cubic_part_delta": float(ours["gauge_cubic_part"] - ball["gauge_cubic_part"]),
    "reading": "we WIN the quantity we optimize and LOSE the quantity the campaign reports; the "
               "deficit is carried by the CUBIC (trigram) term the default objective omits",
}
# how much of the gauge does the default objective's own quadratic proxy explain?
q_all = np.array([split(r["layout"])[0] for r in runs[:128]])
c_all = np.array([split(r["layout"])[1] for r in runs[:128]])
from scipy.stats import spearmanr
res["term_structure"] = {
    "n_layouts": 128,
    "spearman_quadratic_vs_total_gauge": float(spearmanr(q_all, q_all + c_all).statistic),
    "spearman_cubic_vs_total_gauge": float(spearmanr(c_all, q_all + c_all).statistic),
    "spearman_quadratic_vs_cubic": float(spearmanr(q_all, c_all).statistic),
    "sd_quadratic": float(q_all.std(ddof=1)), "sd_cubic": float(c_all.std(ddof=1)),
    "reading": "if the cubic term dominates the VARIANCE of the gauge across optimized layouts, a "
               "bigram-only objective cannot rank them, no matter how many restarts it gets"}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
