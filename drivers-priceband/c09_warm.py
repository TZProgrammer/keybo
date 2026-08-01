"""C09 -- F5 WARM-START STABILITY (the conservative-direction falsifier).

The named residual threat to the frontier design is DIFFERENTIAL SEARCH QUALITY: a tighter cap
has a smaller feasible set, so the search may converge worse there and manufacture a rise.

The test: cross-seed EVERY cap from EVERY other cap's best boards (repair into feasibility, then
re-descend under the cap). This can only LOWER F-hat at tight caps, hence only SHRINK the estimated
price. If the price survives cross-seeding, the rise is not tight-cap search failure.

It also seeds from the 13 FIELD boards + the unconstrained optimum, so every cap gets the benefit of
the best boards anyone (including the human campaign) has found."""
import json
import os
import time

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

TAG = os.environ.get("PB_TAG", "c07")
d = json.load(open(_env.ART + f"/{TAG}_frontier.json"))
runs = d["runs"] if "runs" in d else d
PRICED = d["priced"]
INERT = d["inert"]
CAPS = PRICED + INERT

fs, w1, w2 = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
print(f"evaluators {w1:.2e}/{w2:.2e}", flush=True)

# ---- the donor pool: every board the frontier run kept, + the 13 field boards ----
donors = []
for k, v in runs.items():
    for q in (v.get("top_perms") or []):
        donors.append(np.array(q, dtype=np.intp))
    if v.get("perm_at_best"):
        donors.append(np.array(v["perm_at_best"], dtype=np.intp))
for b in OPTIMIZED:
    donors.append(fs.perm(FIELD[b]))
# dedupe
seen = set()
uniq = []
for q in donors:
    t = tuple(q.tolist())
    if t not in seen:
        seen.add(t)
        uniq.append(q)
donors = uniq
print(f"donor pool: {len(donors)} distinct boards "
      f"(sfb {min(obj.sfb(q[:30]) for q in donors):.3f}..{max(obj.sfb(q[:30]) for q in donors):.3f})", flush=True)

res = {}
t0 = time.perf_counter()
for cap in CAPS:
    ts = time.perf_counter()
    best = (np.inf, None)
    vals = []
    for q in donors:
        p, feas = S.drive_under_cap(obj, q.copy(), cap)      # repair into feasibility
        if not feas:
            continue
        p, m = S.cap_three_opt(obj, p, cap)                  # re-descend (2-opt + 3-cycles)
        if obj.sfb(p[:30]) > cap + 1e-9:
            continue
        vals.append(float(m))
        if m < best[0]:
            best = (m, p.copy())
    lab = f"{cap:.2f}" if cap < 1e8 else "inf"
    res[str(cap)] = dict(cap=float(cap), n=len(vals), best=float(best[0]) if vals else None,
                         sfb_at_best=float(obj.sfb(best[1][:30])) if best[1] is not None else None,
                         perm=best[1][:31].tolist() if best[1] is not None else None,
                         vals=vals, sec=time.perf_counter() - ts)
    print(f"  cap {lab:>7}: warm best {best[0] if not vals else round(best[0],4)}  "
          f"sfb {round(float(obj.sfb(best[1][:30])),4) if best[1] is not None else None}  "
          f"n {len(vals)}/{len(donors)}  {time.perf_counter()-ts:.0f}s", flush=True)
    json.dump(res, open(_env.ART + f"/{TAG}_warm.json", "w"), indent=1)

print(f"\ntotal {time.perf_counter()-t0:.0f}s")
json.dump(res, open(_env.ART + f"/{TAG}_warm.json", "w"), indent=1)
print(f"wrote {TAG}_warm.json")
