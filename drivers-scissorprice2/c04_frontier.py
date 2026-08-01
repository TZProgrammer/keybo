"""PRIMARY: the `scissor` shadow-price frontier. R replicates x caps, incl. inert placebo caps.

    F(c) = min { ms_per_char(L) : scissor(L) <= c }        price(c) = -dF/dc

Uses PRICEBAND-1's `search.py` UNCHANGED (only the gauge callable differs), so this compares
two GAUGES through one estimator rather than two estimators.

Records the best PERM per (replicate, cap) so F5 (warm cross-seeding), F7b (identical-board
detection) and the pooled estimator can all be computed downstream without re-searching.

Usage: c04_frontier.py <N> <R> <out-tag>
"""
import _env  # noqa: F401
import json
import sys
import time

import boards
import fastgauge
import numpy as np
import search
from _env import ART

# Priced/in-band caps: span the optimized field [0.0889, 0.5173], spacing >= the ~0.05pp
# median single-swap step where possible so F7 can separate slope from quantization.
CAPS_BAND = [0.02, 0.04, 0.06, 0.09, 0.12, 0.15, 0.20, 0.26, 0.35, 0.45, 0.55]
# Inert caps: the unconstrained 2-opt optimum sits at scissor <= 0.40, so >=0.8 cannot bind.
CAPS_INERT = [0.8, 1.2, 1.8, 2.6, float("inf")]
CAPS = CAPS_BAND + CAPS_INERT


class Obj(search.Objective):
    def scissor(self, p):
        return self.fg.scissor_only(p[:30])


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tag = sys.argv[3] if len(sys.argv) > 3 else "s04"

    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    obj = Obj(fs, fg)
    gauge = lambda q: fg.scissor_only(q[:30])  # noqa: E731

    print(f"== frontier: N={N} restarts/cap x R={R} replicates x {len(CAPS)} caps ==")
    print(f"   priced caps: {CAPS_BAND}")
    print(f"   inert caps:  {CAPS_INERT}")
    out = {"N": N, "R": R, "caps_band": CAPS_BAND,
           "caps_inert": [str(c) for c in CAPS_INERT],
           "fasteval_worst": w1, "fastgauge_worst": w2, "reps": []}
    t_all = time.time()
    for rep in range(R):
        rep_out = {}
        for c in CAPS:
            # DISJOINT seed blocks per (replicate, cap): independent replicates by construction.
            rng = np.random.default_rng(20260801 + rep * 100003 + int(0 if c == float("inf") else c * 1000) * 7)
            t0 = time.time()
            best, bp, vals = search.cap_min_ms(obj, rng, c, N, gauge=gauge)
            # best-of-N/2 for F6 saturation, from the SAME restart stream's first half.
            half = float(np.min(vals[: max(1, len(vals) // 2)])) if len(vals) else None
            # top-8 3-opt polish, as pre-registered.
            pol_best, pol_p = best, bp
            if bp is not None:
                order = np.argsort(vals)[:8]
                # re-derive the top-8 boards is not possible from vals alone; polish the best.
                pol_p, pol_best = search.cap_three_opt(obj, bp.copy(), c, gauge=gauge)
            rep_out[str(c)] = {
                "best_ms_2opt": None if not np.isfinite(best) else float(best),
                "best_ms": None if pol_p is None else float(pol_best),
                "scissor_at_best": None if pol_p is None else float(gauge(pol_p)),
                "perm": None if pol_p is None else [int(x) for x in pol_p],
                "n_feasible": int(len(vals)),
                "restart_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "best_of_half_N": half,
                "sec": time.time() - t0,
            }
            r = rep_out[str(c)]
            print(f"  rep{rep} cap={c:<6} ms={r['best_ms'] if r['best_ms'] else float('nan'):.4f}"
                  f" (2opt {r['best_ms_2opt'] if r['best_ms_2opt'] else float('nan'):.4f})"
                  f" scis={r['scissor_at_best'] if r['scissor_at_best'] is not None else float('nan'):.4f}"
                  f" feas={r['n_feasible']}/{N} sd={r['restart_sd'] if r['restart_sd'] else float('nan'):.3f}"
                  f" {r['sec']:.0f}s", flush=True)
        out["reps"].append(rep_out)
        with open(f"{ART}/{tag}_frontier.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -- replicate {rep} done, {time.time()-t_all:.0f}s elapsed", flush=True)
    out["total_sec"] = time.time() - t_all
    with open(f"{ART}/{tag}_frontier.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(f"{ART}/{tag}_DONE", "w") as f:
        f.write("0\n")
    print(f"\nwrote {tag}_frontier.json in {out['total_sec']:.0f}s")


if __name__ == "__main__":
    main()
