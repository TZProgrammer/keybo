"""C04 -- SCOPING part 4 (last before pre-registration). The facts the design needs:
  (a) what IS the minimum achievable sfb? (tests the register's 'boards sit AT the sfb floor')
  (b) where in sfb does the UNCONSTRAINED speed optimum sit?
  (c) cost of a 3-opt polish sweep (budget input)
  (d) is the cap frontier monotone-poolable? (a board with sfb<=c is feasible at every cap>=c)
"""
import json
import time

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

fs, _, _ = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
rng = np.random.default_rng(31337)

# (a) minimize sfb itself (steepest descent on sfb from random starts)
print("== (a) MINIMUM ACHIEVABLE sfb (steepest sfb descent, 10 random starts) ==")
mins = []
for _ in range(10):
    p, _ = S.drive_under_cap(obj, S.random_perm(rng), -1.0)   # cap unreachable => pure sfb descent
    mins.append(obj.sfb(p[:30]))
mins = np.array(mins)
print(f"   sfb-descent minima: min {mins.min():.4f}  med {np.median(mins):.4f}  max {mins.max():.4f}")
print(f"   field sfb range: {min(fg.sfb_only(fg.perm(FIELD[b])) for b in OPTIMIZED):.4f}"
      f" .. {max(fg.sfb_only(fg.perm(FIELD[b])) for b in OPTIMIZED):.4f}")

# (b) unconstrained speed optimum: where does it sit in sfb?
print("\n== (b) UNCONSTRAINED speed optimum: its sfb ==")
best = (np.inf, None)
uns = []
for _ in range(24):
    p, m = S.two_opt_ms(obj, S.random_perm(rng))
    uns.append((m, float(obj.sfb(p[:30]))))
    if m < best[0]:
        best = (m, p.copy())
uns.sort()
print(f"   top-8 (ms, sfb): " + "  ".join(f"({m:.3f},{s:.3f})" for m, s in uns[:8]))
sfb_star = float(obj.sfb(best[1][:30]))
print(f"   best board: ms {best[0]:.4f}  sfb {sfb_star:.4f}")
top = [s for m, s in uns[:8]]
print(f"   sfb of the 8 best random-2opt boards: med {np.median(top):.4f}  range {min(top):.4f}..{max(top):.4f}")
# field
fb = min(OPTIMIZED, key=lambda b: fs.ms_per_char(FIELD[b]))
print(f"   FIELD best {fb}: ms {fs.ms_per_char(FIELD[fb]):.4f}  sfb {fg.sfb_only(fg.perm(FIELD[fb])):.4f}")

# (c) 3-opt cost
print("\n== (c) 3-opt (3-cycle) polish cost ==")
p = fs.perm(FIELD["BALL-1"])
t0 = time.perf_counter(); C = S.cycle_perms(p); t_gen = time.perf_counter() - t0
print(f"   n 3-cycles = {len(C)}   generate {t_gen*1e3:.0f} ms")
t0 = time.perf_counter()
_ = np.array([obj.sfb(q[:30]) for q in C[:2000]])
t_sfb = (time.perf_counter() - t0) / 2000
print(f"   sfb per cycle {t_sfb*1e6:.0f} us => full sfb sweep {t_sfb*len(C):.2f} s")
t0 = time.perf_counter(); pp, mm = S.cap_three_opt(obj, p, 3.0); t3 = time.perf_counter() - t0
print(f"   one cap_three_opt from BALL-1 (cap 3.0): {t3:.1f} s   ms {fs.ms_per_char(FIELD['BALL-1']):.4f} -> {mm:.4f}")

# (d) verify a 3-cycle is a genuine permutation & matches literal string 3-cycle
lay = FIELD["BALL-1"]; p0 = fs.perm(lay)
CY = S.cycle_perms(p0)
i, j, k = S.CYC[100]
L = list(lay); L[j], L[k], L[i] = L[i], L[j], L[k]
e = abs(obj.ms(CY[100]) - fs.surf.card("".join(L)).ms_per_char)
print(f"\n== (d) cycle_perms correctness: |ms(cycle) - card(literal 3-cycle)| = {e:.3e} ==")
assert e < 1e-6, e
assert all(len(set(q[:30].tolist())) == 30 for q in CY[:500])
print("   all sampled cycle perms are valid bijections")

json.dump(dict(sfb_descent_minima=mins.tolist(), unconstrained=[(float(a), float(b)) for a, b in uns],
               sfb_at_unconstrained_best=sfb_star, n_cycles=int(len(C)),
               sec_per_3opt=t3, sfb_us_per_cycle=t_sfb * 1e6,
               field_best=fb, field_best_ms=float(fs.ms_per_char(FIELD[fb])),
               field_best_sfb=float(fg.sfb_only(fg.perm(FIELD[fb])))),
          open(_env.ART + "/c04_geom.json", "w"), indent=1)
print("\nwrote c04_geom.json")
