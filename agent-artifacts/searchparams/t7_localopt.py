"""Are the campaign's boards LOCAL OPTIMA on the gauge? (cheap; complements the slow t5)

t5 asks "can a powered search on the right ruler REACH BALL-1?" -- expensive. This asks the
cheaper, sharper question: is BALL-1 (etc.) 2-opt-stable ON THE GAUGE ITSELF, and are MY layouts?
If the campaign boards are gauge-local-optima and my 256-restart layouts are NOT, that localizes
the whole deficit to the objective rather than to search power -- and it directly tests the
ledger's "BALL-1 is a local optimum" claim on the ruler the ledger reports.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
from itertools import combinations
import numpy as np
import _harness as H
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring import model_norm as MN
# NOTE: `from t5_matched import GaugeScorer` would EXECUTE that module's body, which runs its
# own 128-restart pool (I hit exactly this and had to kill it). Extract just the class source.
import ast, textwrap
_src = open("/tmp/searchparams/agent-artifacts/searchparams/t5_matched.py").read()
_tree = ast.parse(_src)
_cls = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == "GaugeScorer")
_ns = {}
exec(compile(ast.Module(body=[_cls], type_ignores=[]), "<GaugeScorer>", "exec"),
     {"np": np, "default_surface": __import__("keybo.analysis.timecard", fromlist=["x"]).default_surface,
      "ROW_STAGGERED_30": ROW_STAGGERED_30, "Layout": Layout, "IScorer":
      __import__("keybo.scoring.base", fromlist=["x"]).IScorer}, _ns)
GaugeScorer = _ns["GaugeScorer"]

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t7_localopt.json"
FLOOR = 0.135
C30M = MN.S.C30M
gs = GaugeScorer(C30M)
REF = {"BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq", "arm B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
       "MID": "flmpg.yuo,sntcdireahkxbwv'-jzq", "HEADLINE": "flmpg-,uoysntcdireahkxvwb.'jzq"}

# parity re-gate (this file must not silently use a different ruler than t5/analyze)
dev = max(abs(gs.ms_per_char(Layout(l, ROW_STAGGERED_30)) - H.ms_per_char(l)) for l in REF.values())
assert dev < 1e-9, dev
print("parity ok: %.2e" % dev, flush=True)


def probe(lay30):
    """All 435 single transpositions on the GAUGE: how many improve, and by how much?"""
    lay = Layout(lay30, ROW_STAGGERED_30)
    base = gs.ms_per_char(lay)
    chars = list(lay.chars)
    deltas = []
    for i, j in combinations(range(len(chars)), 2):
        lay.swap(chars[i], chars[j])
        deltas.append(gs.ms_per_char(lay) - base)
        lay.undo()
    d = np.array(deltas)
    return {"ms_per_char": base, "n_swaps": len(d),
            "n_improving": int((d < 0).sum()),
            "n_improving_by_floor": int((d < -FLOOR).sum()),
            "best_improvement": float(-d.min()) if d.min() < 0 else 0.0,
            "is_2opt_local_optimum_on_gauge": bool((d >= 0).all()),
            "median_abs_delta": float(np.median(np.abs(d)))}

res = {"ruler": "the campaign ms/char gauge itself (parity-gated to analyze, dev %.1e)" % dev,
       "floor": FLOOR, "boards": {}}
t0 = time.perf_counter()
for lab, l in REF.items():
    res["boards"][lab] = probe(l); res["boards"][lab]["source"] = "campaign (PREREG:9423)"
    print(" %-9s %s" % (lab, json.dumps(res["boards"][lab])), flush=True)

# my C30M pool: the objective-best and the gauge-best of 256 restarts
d = json.load(open("/local/home/zegertho/agent/state/searchparams/artifacts/t1b_c30m.json"))
runs = d["runs"]
fit = np.array([r["fitness"] for r in runs]); mpc = np.array([r["ms_per_char"] for r in runs])
for lab, l in (("OURS best-on-objective", runs[int(fit.argmin())]["layout"]),
               ("OURS best-on-gauge (oracle/256)", runs[int(mpc.argmin())]["layout"])):
    res["boards"][lab] = probe(l); res["boards"][lab]["source"] = "searchparams 256-restart C30M pool"
    print(" %-9s %s" % (lab, json.dumps(res["boards"][lab])), flush=True)

# how far can a pure gauge 2-opt polish carry OUR layouts? (cheap upper bound on the fix's value)
from keybo.optimize.local_search import two_opt
polished = []
order = np.argsort(mpc)[:12]      # 12 best-on-gauge from the pool
for k in order:
    lay = two_opt(Layout(runs[int(k)]["layout"], ROW_STAGGERED_30), gs)
    polished.append({"before": float(mpc[int(k)]), "after": gs.ms_per_char(lay),
                     "layout_after": "".join(lay.chars)})
best_after = min(p["after"] for p in polished)
res["gauge_2opt_polish_of_our_pool"] = {
    "n_polished": len(polished), "rows": polished,
    "best_before": float(mpc.min()), "best_after": best_after,
    "gain": float(mpc.min() - best_after),
    "vs_BALL1": float(best_after - res["boards"]["BALL-1"]["ms_per_char"]),
    "vs_armB": float(best_after - res["boards"]["arm B"]["ms_per_char"]),
    "beats_BALL1": bool(best_after < res["boards"]["BALL-1"]["ms_per_char"]),
    "reading": "a 2-opt polish ON THE GAUGE applied to layouts found by the WRONG objective -- the "
               "cheapest possible version of 'fix the objective'. If this alone closes the gap to "
               "BALL-1, the deficit was objective mis-specification, not search power."}
res["wall_sec"] = time.perf_counter() - t0
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["gauge_2opt_polish_of_our_pool"], indent=1))
