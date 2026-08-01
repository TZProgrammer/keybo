"""TASK 1: restart saturation at the SHIPPED defaults. 256 independent attempts."""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import _harness as H

N_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 256
OUT = sys.argv[2] if len(sys.argv) > 2 else "/local/home/zegertho/agent/state/searchparams/artifacts/t1_pool.json"

sc = H.build_search_scorer()
recs = []
t0 = time.perf_counter()
for seed in range(N_POOL):
    r = H.one_attempt(sc, seed=seed)          # shipped defaults: alpha .999, max_outer None, 2opt ON
    r["ms_per_char"] = H.ms_per_char(r["layout"])
    r["ms_per_char_sa_only"] = None
    recs.append(r)
    if (seed + 1) % 16 == 0:
        print("  %3d/%d  %.0fs elapsed" % (seed + 1, N_POOL, time.perf_counter() - t0), flush=True)
meta = {
    "n_pool": N_POOL, "defaults": {"start": "qwerty", "alpha": 0.999, "max_outer": None,
    "local_search": True, "target_wpm": 90.0, "objective": "bigram table (TableBigramScorer, bigram_reg31_seed0)"},
    "gauge": "analysis.timecard.default_surface(90).card().ms_per_char (K31 trigram, 3-seed mean)",
    "budget": H.stop_budget(), "wall_sec_total": time.perf_counter() - t0,
}
json.dump({"meta": meta, "runs": recs}, open(OUT, "w"), indent=1)
print("wrote", OUT, "in %.0fs" % meta["wall_sec_total"])
