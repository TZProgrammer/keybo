"""C00 -- environment + evaluator verification gate. Must pass before anything else runs."""
import json

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

fs, w1, w2 = _env.verify_evaluators(FIELD)
print(f"\nfasteval worst {w1:.3e} / fastsfb worst {w2:.3e}  -- BOTH VERIFIED (<1e-6)")

import time

import fastsfb

fg = fastsfb.FastGauges()
# timing: how expensive is one (ms/char, sfb) evaluation?
p = fs.perm(FIELD["BALL-1"])
pg = fg.perm(FIELD["BALL-1"])
t0 = time.perf_counter()
for _ in range(200):
    fs.ms_per_char_perm(p)
t_ms = (time.perf_counter() - t0) / 200
t0 = time.perf_counter()
for _ in range(200):
    fg.sfb_only(pg)
t_sfb = (time.perf_counter() - t0) / 200
print(f"\nper-eval cost: ms/char {t_ms*1e6:.0f} us   sfb {t_sfb*1e6:.0f} us   both {(t_ms+t_sfb)*1e6:.0f} us")
print(f"  => 1e5 evals = {(t_ms+t_sfb)*1e5:.1f} s")

print(f"\n{'board':<14}{'ms/char':>12}{'sfb':>9}")
rows = {}
for b in sorted(FIELD):
    s = FIELD[b]
    m = fs.ms_per_char(s)
    sfb = fg.sfb_only(fg.perm(s))
    rows[b] = dict(ms=float(m), sfb=float(sfb))
    print(f"{b:<14}{m:>12.6f}{sfb:>9.4f}")
best = min(OPTIMIZED, key=lambda b: rows[b]["ms"])
print(f"\nFIELD BEST (13 optimized) = {best} at {rows[best]['ms']:.6f} ms/char, sfb {rows[best]['sfb']:.4f}")
json.dump(
    dict(fasteval_worst=w1, fastsfb_worst=w2, per_eval_us=dict(ms=t_ms * 1e6, sfb=t_sfb * 1e6),
         rows=rows, field_best=best, field_best_ms=rows[best]["ms"]),
    open(_env.ART + "/c00_env.json", "w"), indent=1)
print("wrote c00_env.json")
