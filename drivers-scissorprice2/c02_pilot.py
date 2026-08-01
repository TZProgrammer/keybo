"""Gate 2: PILOT the cap-constrained search -- feasibility, cost, and the SHAPE hint.

Sizes N and the cap grid before pre-registering. Small N on purpose: this is scoping, and it
is DISCLOSED in the pre-registration so no criterion can be accused of hindsight."""
import _env  # noqa: F401
import json
import time

import boards
import fastgauge
import numpy as np
import search
from _env import ART


class Obj(search.Objective):
    def scissor(self, p):
        return self.fg.scissor_only(p[:30])


def main():
    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    obj = Obj(fs, fg)
    gauge = lambda q: fg.scissor_only(q[:30])  # noqa: E731

    caps = [0.02, 0.05, 0.09, 0.14, 0.20, 0.26, 0.35, 0.52, 0.80, 1.60, float("inf")]
    N = 8
    res = {}
    print(f"== pilot: N={N} random restarts per cap, 2-opt only (no 3-opt polish) ==")
    print(f"{'cap':>7}{'best ms':>11}{'scissor@best':>14}{'n_feas':>8}{'sd':>9}{'sec':>8}")
    for c in caps:
        rng = np.random.default_rng(777)
        t0 = time.time()
        best, bp, vals = search.cap_min_ms(obj, rng, c, N, gauge=gauge)
        dt = time.time() - t0
        sc = float(gauge(bp)) if bp is not None else None
        res[str(c)] = {
            "best_ms": None if not np.isfinite(best) else float(best),
            "scissor_at_best": sc,
            "n_feasible": int(len(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "sec": dt,
        }
        bs = f"{best:>11.4f}" if np.isfinite(best) else f"{'INFEAS':>11}"
        ss = f"{sc:>14.4f}" if sc is not None else f"{'--':>14}"
        sd = res[str(c)]["sd"]
        sds = f"{sd:>9.4f}" if sd is not None else f"{'--':>9}"
        print(f"{c:>7}{bs}{ss}{len(vals):>8}{sds}{dt:>8.1f}")

    out = {"N": N, "caps": [str(c) for c in caps], "pilot": res,
           "fasteval_worst": w1, "fastgauge_worst": w2}
    with open(ART + "/s02_pilot.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote s02_pilot.json")


if __name__ == "__main__":
    main()
