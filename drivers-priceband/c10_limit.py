"""C10 -- the TWO questions the brief asks that the frontier answers directly.

(A) THE LIMIT QUESTION (the brief's stated deliverable): "as headroom shrinks toward zero, does
    the price converge to a stable positive value or diverge/flip?" On the frontier this is
    price(c) as c -> the field's own sfb and beyond, read off the curve.

(B) IS THE TWO-SIDED PRICE ILL-POSED AT THE FIELD'S LOCATION? The frontier gives a ONE-SIDED
    (shadow) price. If the model's unconstrained speed optimum sits at sfb* ~ 2.26 and the field
    sits at 1.07-2.54, then near sfb* BOTH raising and lowering sfb must cost -- so 'the price per
    pp' (a signed two-sided slope) is not a well-defined quantity there, which would explain the
    prior arm's UP +1.40 / DOWN +3.33 without any confound. Test: the LOWER frontier (min ms s.t.
    sfb <= c) vs the UPPER-constrained problem (min ms s.t. sfb >= c). A V-shape around sfb* is
    the signature of ill-posedness; a monotone rise would refute it."""
import json
import os
import time

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

fs, w1, w2 = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
rng = np.random.default_rng(9091)
N = int(os.environ.get("PB_N10", "24"))

# The FLOOR-side price curve: does price(c) converge as c -> sfb_floor?
prem = json.load(open(_env.ART + "/c05_premise.json"))
floor = prem["sfb_floor_global"]
print(f"sfb floor (global, by descent) = {floor:.4f}", flush=True)

# ---- (B) the UPPER-constrained problem: min ms s.t. sfb >= c ----
# implemented by negating the gauge: gauge'(L) = -sfb(L) <= -c  <=>  sfb(L) >= c
def neg_sfb(q):
    return -obj.sfb(q[:30])


print(f"\n== (B) V-SHAPE TEST: min ms/char s.t. sfb >= c  (N={N} restarts/cap) ==", flush=True)
print(f"{'sfb >= c':>10}{'best ms':>11}{'sfb@best':>10}{'n':>5}{'sec':>7}")
upper = {}
for c in (1.0, 1.5, 2.0, 2.26, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
    t0 = time.perf_counter()
    m, p, vals = S.cap_min_ms(obj, rng, -c, restarts=N, gauge=neg_sfb)
    dt = time.perf_counter() - t0
    upper[str(c)] = dict(best=float(m) if np.isfinite(m) else None, n=len(vals),
                         sfb_at_best=float(obj.sfb(p[:30])) if p is not None else None,
                         vals=vals.tolist(), sec=dt)
    print(f"{c:>10.2f}{m:>11.4f}{obj.sfb(p[:30]) if p is not None else float('nan'):>10.4f}{len(vals):>5}{dt:>7.0f}",
          flush=True)
    json.dump(dict(floor=floor, upper=upper), open(_env.ART + "/c10_limit.json", "w"), indent=1)

# ---- (A) tight-cap refinement: price(c) as c -> floor, with MORE restarts where it starved ----
print(f"\n== (A) LIMIT: refine the tightest caps (where F4 flagged starvation), N={4*N} ==", flush=True)
tight = {}
for c in (0.85, 0.9, 0.95, 1.0, 1.1, 1.25):
    t0 = time.perf_counter()
    m, p, vals = S.cap_min_ms(obj, rng, c, restarts=4 * N)
    # 3-opt polish the best few
    best = float(m)
    if p is not None:
        p3, m3 = S.cap_three_opt(obj, p.copy(), c)
        if obj.sfb(p3[:30]) <= c + 1e-9:
            best = min(best, float(m3))
    dt = time.perf_counter() - t0
    tight[str(c)] = dict(best=best, n=len(vals), sec=dt,
                         sd=float(vals.std(ddof=1)) if len(vals) > 1 else None,
                         sfb_at_best=float(obj.sfb(p[:30])) if p is not None else None)
    print(f"   cap {c:>5.2f}: best {best:>10.4f}  n {len(vals):>3}/{4*N}  sd "
          f"{vals.std(ddof=1) if len(vals)>1 else float('nan'):.4f}  {dt:.0f}s", flush=True)
    json.dump(dict(floor=floor, upper=upper, tight=tight), open(_env.ART + "/c10_limit.json", "w"), indent=1)

json.dump(dict(floor=floor, upper=upper, tight=tight, N=N),
          open(_env.ART + "/c10_limit.json", "w"), indent=1)
print("\nwrote c10_limit.json")
