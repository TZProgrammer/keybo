"""C02 -- SCOPING part 2 (still before pre-registration): is the CONSTRAINED frontier
feasible and affordable? For a few sfb targets, run a handful of restarts and measure
(a) cost per restart, (b) the spread of feasible local minima, (c) whether the constrained
optimum at the FIELD's own sfb (~1.5-2.5) is competitive with the field."""
import json
import time

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

fs, _, _ = _env.verify_evaluators({k: FIELD[k] for k in ("BALL-1", "qwerty30m")})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
rng = np.random.default_rng(11)

TOL = 0.05
out = {}
for target in (1.0, 1.5, 2.5, 4.0, 6.0):
    t0 = time.perf_counter()
    m, p, s, nf, allms = S.constrained_min_ms(obj, rng, target, TOL, restarts=6)
    dt = time.perf_counter() - t0
    out[str(target)] = dict(best_ms=float(m), sfb=float(s) if s is not None else None,
                            n_feasible=int(nf), sec=dt, all_ms=allms.tolist())
    sd = allms.std(ddof=1) if len(allms) > 1 else float("nan")
    print(f"sfb target {target:>4.1f} (tol {TOL}): best {m:>10.4f}  sfb {s if s is None else round(float(s),4)}"
          f"  feasible {nf}/6  sd {sd:.4f}  {dt:.1f}s  ({dt/6:.1f}s/restart)")
json.dump(dict(tol=TOL, targets=out), open(_env.ART + "/c02_feas.json", "w"), indent=1)
print("\nwrote c02_feas.json")
