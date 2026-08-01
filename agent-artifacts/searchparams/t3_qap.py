"""TASK 3: is the Gilmore-Lawler bound VALID for the objectives in play, and what is the gap?

Pre-registered order: validity FIRST (numerically, not by reading), gap SECOND.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import numpy as np
import _harness as H
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.geometry import ROW_STAGGERED_30
from keybo.optimize.qap_bound import qap_fitness, gilmore_lawler_bound, certificate, CertificateScopeError

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t3_qap.json"
res = {}
sc = H.build_search_scorer()
F, T = sc._F, sc._T
rng = np.random.default_rng(12345)

# ---- VALIDITY TEST A: is the SEARCH objective literally qap_fitness(F,T,.)? ----
# TableBigramScorer.fitness_of_permutation == (F * T[ix_(p,p)]).sum() == qap_fitness. Test it.
devs = []
for _ in range(200):
    p = np.empty(31, dtype=np.intp)
    p[:30] = rng.permutation(30)
    p[30] = 30                      # space pinned (as the scorer does)
    a = sc.fitness_of_permutation(p)
    b = qap_fitness(F, T, p)
    devs.append(abs(a - b))
res["validity_search_objective"] = {
    "test": "TableBigramScorer.fitness_of_permutation(p) vs qap_fitness(F,T,p), 200 random perms",
    "max_abs_dev": float(max(devs)),
    "max_rel_dev": float(max(devs) / abs(qap_fitness(F, T, p))),
    "bit_exact": bool(max(devs) == 0.0),
    "verdict": "VALID: the search objective IS qap_fitness(F,T,.) for these very (F,T)"
               if max(devs) == 0.0 else "CHECK: not bit-exact",
}

# ---- VALIDITY TEST B: is the CAMPAIGN GAUGE (ms/char) a quadratic assignment at all? ----
# ms/char = (sum_ngram (T2[a,b] + Tc[a,b,c]) * f) / covered.  The Tc term is CUBIC in perm,
# so no (F,T) pair reproduces it. Demonstrate that the cubic term is non-negligible, i.e. the
# invalidity is material rather than a technicality.
from keybo.analysis.timecard import default_surface
surf = default_surface(90.0)
T2, Tc = surf._T2, surf._Tc
def split(lay30):
    slot = {c: i for i, c in enumerate(lay30)}; slot[" "] = 30
    t2 = t3 = 0.0; cov = 0
    for ng, f in surf.tri.items():
        try: a, b, c = slot[ng[0]], slot[ng[1]], slot[ng[2]]
        except KeyError: continue
        cov += f; t2 += T2[a, b] * f; t3 += Tc[a, b, c] * f
    return t2, t3, cov
t2q, t3q, covq = split(NAMED_LAYOUTS["qwerty"])
res["validity_campaign_gauge"] = {
    "test": "decompose ms/char into its quadratic (T2) and CUBIC (Tc) parts on qwerty",
    "quadratic_ms_per_char": t2q / covq, "cubic_ms_per_char": t3q / covq,
    "cubic_share_of_mass": t3q / (t2q + t3q),
    "verdict": "INVALID for the campaign ms/char gauge: a GL bound is quadratic machinery and "
               "the gauge carries a genuinely cubic trigram term; min(A+B) >= min(A)+min(B) with "
               "min(B) uncertified. The bound cannot be transferred to ms/char.",
}

# ---- THE GAP, on the objective the bound is valid for ----
def cert_for(lay30, label):
    # The table fixes the charset (permutations of --start only). A named layout with a
    # DIFFERENT charset (dvorak "\'", graphite/semimak "-") is not in the certified space at
    # all -- report that rather than mis-scoring it.
    if set(lay30) != set(H.START):
        return {"label": label, "layout": lay30,
                "NOT_CERTIFIABLE": "charset differs from the table's (%s); the bound is over "
                                   "permutations of --start's charset only"
                                   % "".join(sorted(set(lay30) ^ set(H.START)))}
    lay = Layout(lay30, ROW_STAGGERED_30)
    p = sc.permutation(lay)
    fit = sc.fitness_of_permutation(p)
    try:
        c = certificate(F, T, fit, scope="the shipped bigram-table search objective (bigram_reg31_seed0, wpm 90)")
    except CertificateScopeError as e:
        return {"label": label, "layout": lay30, "MISMATCH_ALARM": str(e)}
    return {"label": label, "layout": lay30, "fitness": fit,
            "lower_bound": c["lower_bound"], "gap_pct": c["gap_pct"],
            "ms_per_char_gauge": H.ms_per_char(lay30)}

t0 = time.perf_counter()
res["lower_bound"] = float(gilmore_lawler_bound(F, T))
res["bound_sec"] = time.perf_counter() - t0

targets = [(NAMED_LAYOUTS[n], n) for n in ("qwerty", "colemak", "dvorak", "graphite", "semimak")]
# a random layout, for the bound's dynamic range
rp = list(NAMED_LAYOUTS["qwerty"]); rng2 = np.random.default_rng(7); rng2.shuffle(rp)
targets.append(("".join(rp), "random"))
best_path = "/local/home/zegertho/agent/state/searchparams/artifacts/t1_pool.json"
try:
    pool = json.load(open(best_path))
    runs = sorted(pool["runs"], key=lambda r: r["fitness"])
    targets.append((runs[0]["layout"], "best-of-%d (searchparams T1 pool)" % len(runs)))
    targets.append((runs[len(runs) // 2]["layout"], "median-of-pool"))
    targets.append((runs[-1]["layout"], "worst-of-pool"))
except FileNotFoundError:
    pass
res["certificates"] = [cert_for(l, n) for l, n in targets]

# ---- the bound's OWN resolution floor ON THIS INSTANCE ----
# QAPBOUND-1 measured ~2.3410% on its instance. Do not quote it blind: the floor is the gap the
# bound STILL reports on a layout that is (as near as we can get) optimal. Our best-of-256 IS
# that layout, so its gap IS this instance's floor estimate, and every other certificate must be
# read against it.
cs = [c for c in res["certificates"] if "gap_pct" in c]
best = min(cs, key=lambda c: c["fitness"])
res["bound_resolution_floor_this_instance"] = {
    "definition": "the gap the GL bound still reports on the best layout we can find (best-of-256 "
                  "at shipped defaults) -- i.e. pure bound looseness, not search error",
    "floor_pct": best["gap_pct"], "from_label": best["label"],
    "qapbound1_reported_floor_pct": 2.3410,
    "dynamic_range": {c["label"]: {"gap_pct": c["gap_pct"],
                                   "gap_above_this_instance_floor_pp": c["gap_pct"] - best["gap_pct"]}
                      for c in cs},
}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1)[:4000])
