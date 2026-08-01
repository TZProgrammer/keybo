"""C07 -- THE PRIMARY ESTIMATOR: the constrained speed frontier F(c) = min ms s.t. sfb <= c,
replicated R times with disjoint seed blocks, polished with 3-opt, INCLUDING the inert-cap
placebo caps. Emits everything the six pre-registered criteria need.

Detached-friendly: writes JSON incrementally and a DONE marker file at the end."""
import json
import os
import sys
import time

import _env
import numpy as np
from boards import FIELD

import fastsfb
import search as S

N = int(os.environ.get("PB_N", "32"))
R = int(os.environ.get("PB_R", "4"))
TOP3 = int(os.environ.get("PB_TOP3", "8"))
TAG = os.environ.get("PB_TAG", "c07")

PRICED = [0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
INERT = [3.0, 3.5, 5.0, 8.0, 1e9]          # 1e9 == unconstrained
CAPS = PRICED + INERT

fs, w1, w2 = _env.verify_evaluators({k: FIELD[k] for k in ("BALL-1", "qwerty30m", "semimak")})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
print(f"evaluators {w1:.2e}/{w2:.2e}   N={N} restarts/cap  R={R} replicates  3-opt top-{TOP3}", flush=True)

res = {}
t_start = time.perf_counter()
for r in range(R):
    for cap in CAPS:
        # disjoint seed block per (replicate, cap): reproducible and independent
        rng = np.random.default_rng([20260801, r, int(cap * 1000)])
        t0 = time.perf_counter()
        vals = []
        boards = []
        for _ in range(N):
            p = S.random_perm(rng)
            p, feas = S.drive_under_cap(obj, p, cap)
            if not feas:
                continue
            p, m = S.cap_two_opt(obj, p, cap)
            if obj.sfb(p[:30]) > cap + 1e-9:
                continue
            vals.append(float(m))
            boards.append(p.copy())
        # 3-opt polish the best TOP3 (richer class used as SEARCH)
        pol = []
        if vals:
            order = np.argsort(vals)[:TOP3]
            for k in order:
                p3, m3 = S.cap_three_opt(obj, boards[k].copy(), cap)
                if obj.sfb(p3[:30]) <= cap + 1e-9:
                    pol.append((float(m3), p3))
        best_raw = float(min(vals)) if vals else None
        best_pol = float(min(m for m, _ in pol)) if pol else None
        bestp = min(pol, key=lambda t: t[0])[1] if pol else None
        dt = time.perf_counter() - t0
        key = f"r{r}_c{cap}"
        res[key] = dict(
            replicate=r, cap=float(cap), n_feasible=len(vals), vals=vals,
            best_raw=best_raw, best_polished=best_pol,
            polished_vals=[m for m, _ in pol],
            sfb_at_best=float(obj.sfb(bestp[:30])) if bestp is not None else None,
            perm_at_best=bestp[:31].tolist() if bestp is not None else None,
            top_perms=[q[:31].tolist() for _, q in sorted(pol, key=lambda t: t[0])[:3]],
            sec=dt)
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        print(f"  r{r} cap {cap:>7.2f}: feas {len(vals):>3}/{N}  raw {best_raw if best_raw is None else round(best_raw,4)}"
              f"  3opt {best_pol if best_pol is None else round(best_pol,4)}  sd {sd:.4f}  {dt:.0f}s", flush=True)
        json.dump(res, open(_env.ART + f"/{TAG}_frontier.json", "w"), indent=1)
    print(f" -- replicate {r} done, {time.perf_counter()-t_start:.0f}s elapsed", flush=True)

json.dump(dict(N=N, R=R, TOP3=TOP3, priced=PRICED, inert=INERT, runs=res,
               total_sec=time.perf_counter() - t_start),
          open(_env.ART + f"/{TAG}_frontier.json", "w"), indent=1)
print(f"\nwrote {TAG}_frontier.json  ({time.perf_counter()-t_start:.0f}s total)")
