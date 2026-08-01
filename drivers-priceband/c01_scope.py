"""C01 -- SCOPING (before pre-registration). Three questions:
  (1) is `swap_perms` correct? (verify vs direct string swaps)
  (2) how good/expensive is 2-opt-from-random vs the field? (does the frontier reach the band?)
  (3) what is the SEARCH noise of my own estimator (so I can derive my own N)?
"""
import json
import time

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

fs, w1, w2 = _env.verify_evaluators(FIELD)
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
rng = np.random.default_rng(20260801)

# ---------- (1) swap_perms correctness vs literal string transposition ----------
lay = FIELD["BALL-1"]
p = fs.perm(lay)
P = S.swap_perms(p)
bad = 0
for k, (i, j) in enumerate(S.IJ):
    L = list(lay)
    L[i], L[j] = L[j], L[i]
    if not np.array_equal(P[k], fs.perm("".join(L))):
        bad += 1
print(f"(1) swap_perms: {len(S.IJ)} transpositions, mismatches vs literal string swap = {bad}")
assert bad == 0
# and the ms/char of a swept neighbour must equal card() of the swapped string
L = list(lay); L[3], L[17] = L[17], L[3]
k = int(np.where((S.IJ[:, 0] == 3) & (S.IJ[:, 1] == 17))[0][0])
e = abs(obj.ms(P[k]) - fs.surf.card("".join(L)).ms_per_char)
print(f"    neighbour ms/char vs shipped card(swapped string): abs err {e:.3e}")
assert e < 1e-6

# ---------- (2) does the field sit at a 2-opt optimum? does random+2opt reach the band? ----------
print("\n(2) field convergence + reachability")
print(f"    {'board':<14}{'ms':>11}{'sfb':>8}{'improving swaps':>17}{'best gain':>11}")
conv = {}
for b in OPTIMIZED:
    p0 = fs.perm(FIELD[b])
    m0 = obj.ms(p0)
    _, _, mss = obj.sweep(p0)
    n_imp = int((mss < m0 - 1e-12).sum())
    conv[b] = dict(ms=float(m0), n_improving=n_imp, best_gain=float(m0 - mss.min()))
    print(f"    {b:<14}{m0:>11.4f}{obj.sfb(p0[:30]):>8.4f}{n_imp:>17}{m0-mss.min():>11.4f}")

t0 = time.perf_counter()
rand_ms = []
for _ in range(12):
    q = S.random_perm(rng)
    _, m = S.two_opt_ms(obj, q)
    rand_ms.append(m)
t_per = (time.perf_counter() - t0) / 12
rand_ms = np.array(rand_ms)
print(f"\n    2-opt from RANDOM x12: min {rand_ms.min():.4f}  med {np.median(rand_ms):.4f}  max {rand_ms.max():.4f}")
print(f"      sd {rand_ms.std(ddof=1):.4f}   cost {t_per:.2f} s/descent")
fieldbest = min(conv[b]["ms"] for b in OPTIMIZED)
print(f"    FIELD BEST = {fieldbest:.4f}; gap of random-2opt best to field best = {rand_ms.min()-fieldbest:+.4f}")

# polish the field boards to a true 2-opt optimum, for a fair 'in-band' altitude reference
pol = {}
for b in OPTIMIZED:
    _, m = S.two_opt_ms(obj, fs.perm(FIELD[b]))
    pol[b] = float(m)
print(f"    field boards POLISHED to 2-opt: best {min(pol.values()):.4f}  (raw best {fieldbest:.4f})")

json.dump(dict(swap_perms_mismatches=bad, field=conv, polished=pol,
               rand2opt=dict(vals=rand_ms.tolist(), sec_per_descent=t_per),
               field_best_raw=fieldbest, field_best_polished=min(pol.values())),
          open(_env.ART + "/c01_scope.json", "w"), indent=1)
print("\nwrote c01_scope.json")
