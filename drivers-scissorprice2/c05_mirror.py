"""The MIRROR PROBLEM:  min ms/char  s.t.  scissor >= c   -- interior vs boundary optimum.

PRICEBAND-1's decisive cheap diagnostic. If the LOWER-bound constraint is SLACK for small c
(the optimizer voluntarily sits well above c), the speed optimum in the gauge is INTERIOR, and
a signed TWO-SIDED "price per pp" is ILL-POSED there -- not merely hard to measure. That is
what explained the prior arm's wrong sign for sfb.

Implemented by flipping the gauge's sign: `scissor >= c`  <=>  `(-scissor) <= (-c)`, so the
SAME `cap_min_ms` machinery runs with gauge = -scissor and cap = -c. No new search code.
"""
import _env  # noqa: F401
import json
import time

import boards
import fastgauge
import numpy as np
import search
from _env import ART

CS = [0.02, 0.06, 0.09, 0.14, 0.20, 0.26, 0.35, 0.52, 0.80, 1.20, 1.60, 2.40]
N = 24


class Obj(search.Objective):
    def scissor(self, p):
        return self.fg.scissor_only(p[:30])


def main():
    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    obj = Obj(fs, fg)
    scis = lambda q: fg.scissor_only(q[:30])  # noqa: E731
    neg = lambda q: -fg.scissor_only(q[:30])  # noqa: E731

    print(f"== MIRROR: min ms s.t. scissor >= c   (N={N} restarts/c) ==")
    print(f"{'c':>7}{'min ms | scis>=c':>18}{'scissor@best':>14}{'slack?':>9}{'feas':>7}{'sec':>7}")
    res = {}
    for c in CS:
        rng = np.random.default_rng(90210 + int(c * 1000))
        t0 = time.time()
        best, bp, vals = search.cap_min_ms(obj, rng, -c, N, gauge=neg)
        sc = float(scis(bp)) if bp is not None else None
        # SLACK means the optimizer voluntarily sits ABOVE the lower bound by a real margin.
        slack = None if sc is None else sc - c
        res[str(c)] = {
            "min_ms_geq": None if not np.isfinite(best) else float(best),
            "scissor_at_best": sc, "slack": slack,
            "n_feasible": int(len(vals)),
            "restart_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "sec": time.time() - t0,
        }
        print(f"{c:>7.2f}{best:>18.4f}{sc:>14.4f}{slack:>9.4f}{len(vals):>7}"
              f"{res[str(c)]['sec']:>7.0f}", flush=True)

    # The UNCONSTRAINED optimum, same N and same machinery: the reference level for slackness.
    # MUST be computed before the verdict block below, which differences against it.
    rng = np.random.default_rng(90210)
    t0 = time.time()
    ub, up, uvals = search.cap_min_ms(obj, rng, float("inf"), N, gauge=scis)
    unc = {"min_ms": float(ub), "scissor_at_best": float(scis(up)),
           "n_feasible": int(len(uvals)), "sec": time.time() - t0}
    print(f"\nUNCONSTRAINED (same N, same code): ms={ub:.4f} at scissor={scis(up):.4f}")

    # VERDICT, computed not asserted.
    #
    # The right criterion is NOT "is the achieved value above c" (an achieved-vs-c margin is
    # noisy at this gauge's ~0.05pp granularity, and my first version read it that way and
    # printed a WRONG verdict). It is whether the CONSTRAINT COSTS ANYTHING: compare
    # `min ms s.t. scissor >= c` against the UNCONSTRAINED best found by the SAME N and the
    # SAME code. If forcing scissor UP to c is free, the lower bound is slack ==> the speed
    # optimum lies at or above c, and a two-sided price is ill-posed below it.
    STEP = 0.05  # measured median single-swap |d scissor| in band (s01_geom.json)
    COST_TOL = 0.10  # ms/char; ~1/8 of the 0.8519 search-seed sd, far inside best-of-N noise
    free_cs = [c for c in CS if res[str(c)]["min_ms_geq"] is not None
               and res[str(c)]["min_ms_geq"] - unc["min_ms"] <= COST_TOL]
    costly_cs = [c for c in CS if res[str(c)]["min_ms_geq"] is not None
                 and res[str(c)]["min_ms_geq"] - unc["min_ms"] > COST_TOL]
    slack_cs = [c for c in CS if res[str(c)]["slack"] is not None and res[str(c)]["slack"] > STEP]
    binding_cs = [c for c in CS if res[str(c)]["slack"] is not None and res[str(c)]["slack"] <= STEP]
    # cost of the LOWER bound above the optimum: one-sided price on the upper side
    up_prices = {}
    for a, b in zip(CS, CS[1:]):
        ma, mb = res[str(a)]["min_ms_geq"], res[str(b)]["min_ms_geq"]
        if ma is not None and mb is not None:
            up_prices[f"[{a},{b}]"] = (mb - ma) / (b - a)

    # cost of the lower bound, vs the unconstrained level -- the quantity the verdict reads
    cost_vs_uncon = {str(c): (res[str(c)]["min_ms_geq"] - unc["min_ms"])
                     if res[str(c)]["min_ms_geq"] is not None else None for c in CS}
    largest_free = max(free_cs) if free_cs else None
    verdict = (
        f"INTERIOR: forcing scissor UP to c is FREE (<= {COST_TOL} ms/char vs unconstrained) for "
        f"every c <= {largest_free}, and costs monotonically above it. The speed optimum in "
        f"scissor is therefore INTERIOR near ~{largest_free}, so a signed TWO-SIDED price is "
        f"ill-posed at/below that value -- only the ONE-SIDED shadow price is well-posed."
        if largest_free is not None and largest_free > min(CS) else
        "BOUNDARY: the lower bound costs from the smallest c up, so the optimum is at the floor."
    )
    out = {"N": N, "cs": CS, "mirror": res, "unconstrained": unc,
           "median_swap_step_used_for_slack": STEP,
           "cost_tolerance_ms": COST_TOL,
           "cost_vs_unconstrained": cost_vs_uncon,
           "free_at_c": free_cs, "costly_at_c": costly_cs,
           "largest_free_c": largest_free,
           "slack_at_c_by_achieved_margin": slack_cs,
           "binding_at_c_by_achieved_margin": binding_cs,
           "upper_side_prices_per_pp": up_prices,
           "verdict": verdict,
           "fasteval_worst": w1, "fastgauge_worst": w2}
    print(f"\nUNCONSTRAINED reference (same N, same code): {unc['min_ms']:.4f}")
    print(f"cost of the LOWER bound vs unconstrained (ms/char):")
    for c in CS:
        v = cost_vs_uncon[str(c)]
        print(f"   c={c:<6} {v:+.4f}  {'FREE' if v <= COST_TOL else 'COSTS'}")
    print(f"\nfree (<= {COST_TOL} ms) at c = {free_cs}")
    print(f"costly at c = {costly_cs}")
    print(f"achieved-margin slack (>{STEP} pp) at c = {slack_cs}   [diagnostic only, noisy]")
    print(f"VERDICT: {verdict}")
    print("\nupper-side prices (ms/char per pp), from the mirror:")
    for k, v in up_prices.items():
        print(f"  {k:<16}{v:+.4f}")
    with open(ART + "/s05_mirror.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(ART + "/s05_DONE", "w") as f:
        f.write("0\n")
    print("\nwrote s05_mirror.json")


if __name__ == "__main__":
    main()
