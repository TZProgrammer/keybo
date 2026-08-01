"""TASK 2: cooling-schedule sweep at MATCHED SEEDS, reported as a cost/quality curve."""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import _harness as H

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t2_sweep.json"
SEEDS = list(range(48))
ALPHAS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
MAX_OUTER = [25, 50, 100, 200, 400, None]

sc = H.build_search_scorer()
arms, t0 = [], time.perf_counter()

def run_arm(label, **kw):
    recs = []
    for s in SEEDS:
        r = H.one_attempt(sc, seed=s, **kw)
        r["ms_per_char"] = H.ms_per_char(r["layout"])
        r["ms_per_char_sa_only"] = None
        recs.append(r)
    arms.append({"label": label, "kw": {k: v for k, v in kw.items()}, "runs": recs})
    import statistics as st
    print("  %-34s fit_mean=%.6e  ms/char_mean=%.4f  sec/att=%.3f  outer_mean=%.0f  (%.0fs)" % (
        label, st.mean(r["fitness"] for r in recs), st.mean(r["ms_per_char"] for r in recs),
        st.mean(r["sec"] for r in recs), st.mean(r["outer_count"] for r in recs),
        time.perf_counter() - t0), flush=True)

print("ALPHA SWEEP (2-opt ON, max_outer=None) -- %d seeds each" % len(SEEDS), flush=True)
for a in ALPHAS:
    run_arm("alpha=%g" % a, alpha=a)

print("CONTROL: --no-local-search (what is the 2-opt polish worth?)", flush=True)
for a in (0.999, 0.99, 0.95, 0.9):
    run_arm("alpha=%g NO-2opt" % a, alpha=a, local_search=False)

print("MAX-OUTER LADDER (alpha=0.999, 2-opt ON)", flush=True)
for mo in MAX_OUTER:
    run_arm("alpha=0.999 max_outer=%s" % mo, alpha=0.999, max_outer=mo)
print("MAX-OUTER LADDER (alpha=0.9, 2-opt ON) -- where cooling actually reaches cold", flush=True)
for mo in (100, 200, 400, None):
    run_arm("alpha=0.9 max_outer=%s" % mo, alpha=0.9, max_outer=mo)

json.dump({"seeds": SEEDS, "arms": arms, "wall_sec": time.perf_counter() - t0,
           "gauge": "K31 trigram 3-seed-mean ms/char", "objective": "bigram table seed0 wpm90",
           "shipped_default_arm": "alpha=0.999"}, open(OUT, "w"), indent=1)
print("wrote", OUT, "in %.0fs" % (time.perf_counter() - t0))
