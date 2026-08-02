"""C03 -- SCOPING part 3: does the INEQUALITY-CAP form converge to near-optimal boards,
and how does best-of-N behave? This is what sets my restart count N (pre-registration input)."""
import json
import time

import _env
import numpy as np
from boards import FIELD

import fastsfb
import search as S

fs, _, _ = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
rng = np.random.default_rng(7)

print("== unconstrained reference (cap=inf), 24 random 2-opt descents ==")
t0 = time.perf_counter()
un = []
for _ in range(24):
    _, m = S.two_opt_ms(obj, S.random_perm(rng))
    un.append(m)
un = np.array(un)
print(f"  min {un.min():.4f}  med {np.median(un):.4f}  sd {un.std(ddof=1):.4f}   {(time.perf_counter()-t0)/24:.2f} s/restart")
print(f"  best-of-N: " + "  ".join(f"N={n}:{np.min(un[:n]):.4f}" for n in (1, 2, 4, 8, 16, 24)))

print("\n== cap form: min ms s.t. sfb <= cap, 12 random restarts each ==")
res = {}
for cap in (1.0, 1.5, 2.0, 2.5, 3.0, 6.0):
    t0 = time.perf_counter()
    m, p, vals = S.cap_min_ms(obj, rng, cap, restarts=12)
    dt = time.perf_counter() - t0
    sfb = obj.sfb(p[:30]) if p is not None else None
    res[str(cap)] = dict(best=float(m), n=len(vals), sd=float(vals.std(ddof=1)) if len(vals) > 1 else None,
                         sfb_at_best=float(sfb) if sfb is not None else None, sec=dt, vals=vals.tolist())
    print(f"  cap {cap:>4.1f}: best {m:>10.4f}  sfb@best {sfb if sfb is None else round(float(sfb),4):>7}"
          f"  feas {len(vals):>2}/12  sd {vals.std(ddof=1) if len(vals)>1 else float('nan'):.4f}  {dt/12:.2f}s/restart")
    print(f"           best-of-N: " + "  ".join(f"N={n}:{np.min(vals[:n]):.4f}" for n in (1, 2, 4, 8, 12) if len(vals) >= n))
json.dump(dict(unconstrained=un.tolist(), caps=res), open(_env.ART + "/c03_cap.json", "w"), indent=1)
print("\nwrote c03_cap.json")
