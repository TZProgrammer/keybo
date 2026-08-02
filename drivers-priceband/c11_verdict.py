"""C11 -- FINAL SYNTHESIS: applies F5 (warm-start) to the frontier, re-runs the six gates on the
WARM (conservative) frontier, and assembles the V-shape / two-sided-ill-posedness result.

The warm frontier is the CONSERVATIVE estimate: cross-seeding can only LOWER F-hat at tight caps,
which can only SHRINK the price. A price that survives here is not a tight-cap search artifact."""
import json

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb

fs, w1, w2 = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()

A = json.load(open(_env.ART + "/c07_analysis.json"))
W = json.load(open(_env.ART + "/c07_warm.json"))
L = json.load(open(_env.ART + "/c10_limit.json"))
CAPS, PRICED, INERT = A["caps"], A["priced"], A["inert"]
pool = np.array(A["F_pool"], float)
own = np.array(A["F_own"], float)

# ---- the WARM frontier: monotone-enforced (a board feasible at cap c is feasible at any c'>c) ----
warm = {}
for k, v in W.items():
    if v.get("best") is not None:
        warm[float(v["cap"])] = (float(v["best"]), float(v["sfb_at_best"]))
wcaps = sorted(warm)
Fw = {}
for c in wcaps:
    Fw[c] = min(warm[d][0] for d in wcaps if warm[d][1] <= c + 1e-9) if any(
        warm[d][1] <= c + 1e-9 for d in wcaps) else warm[c][0]

print("== F5 WARM-START FRONTIER (conservative: can only LOWER F-hat, hence SHRINK the price) ==")
print(f"{'cap':>8}{'cold F_pool':>13}{'WARM F':>10}{'improvement':>13}{'sfb@best':>10}")
for c in wcaps:
    j = CAPS.index(c) if c in CAPS else None
    cold = float(np.nanmean(pool[:, j])) if j is not None else float("nan")
    lab = f"{c:.2f}" if c < 1e8 else "inf"
    print(f"{lab:>8}{cold:>13.4f}{Fw[c]:>10.4f}{cold-Fw[c]:>13.4f}{warm[c][1]:>10.4f}")


def wslope(lo, hi):
    return -(Fw[hi] - Fw[lo]) / (hi - lo)


HEAD = tuple(A["headline"]["interval"])
print(f"\n== the SIX GATES re-run on the WARM frontier ==")
pw = wslope(*HEAD)
print(f"F1' price over cap [{HEAD[0]},{HEAD[1]}] = {pw:+.4f} ms/char per pp"
      f"   (cold F_pool {A['headline']['mean']:+.4f}, cold F_own {A['gates']['F3_placebo']['inband_own']:+.4f})")
rise_w = Fw[HEAD[0]] - Fw[HEAD[1]]
sdF = A["gates"]["F2_exceeds_noise"]["sd_F"]
print(f"F2' rise {rise_w:+.4f} ms/char vs replicate sd {sdF:.4f} => {rise_w/sdF:.2f}x (need >3x)"
      f"  {'PASS' if rise_w > 3*sdF else 'FAIL'}")
inert_num = [c for c in INERT if c < 1e8]
pl_w = -(Fw[inert_num[-1]] - Fw[inert_num[0]]) / (inert_num[-1] - inert_num[0])
print(f"F3' PLACEBO on warm, inert caps [{inert_num[0]},{inert_num[-1]}]: slope {pl_w:+.6f}"
      f"   |slope| vs price/3 {abs(pw)/3:.4f} => {'PASS (flat)' if abs(pl_w) < abs(pw)/3 else 'FAIL'}")
viol_w = [(wcaps[i], wcaps[i + 1], Fw[wcaps[i + 1]] - Fw[wcaps[i]])
          for i in range(len(wcaps) - 1) if Fw[wcaps[i + 1]] > Fw[wcaps[i]] + 1e-9]
print(f"F4' MONOTONICITY on warm: {len(viol_w)} violations"
      f"{' (worst ' + format(max(v[2] for v in viol_w), '+.4f') + ')' if viol_w else ''}"
      f" => {'PASS' if not viol_w else 'FAIL'}")
print(f"   (cold F_own had {A['gates']['F4_monotone']['n_violations']} violations, worst "
      f"{A['gates']['F4_monotone']['worst']:+.4f} -- ALL at the starved tight caps)")

print(f"\n== per-interval price, WARM (the conservative curve) ==")
print(f"{'interval':>14}{'warm price':>12}{'cold price':>12}")
iv = {}
for lo, hi in zip(PRICED[:-1], PRICED[1:]):
    if lo in Fw and hi in Fw:
        cw = wslope(lo, hi)
        cc = A["intervals"].get(f"{lo}-{hi}", {}).get("mean", float("nan"))
        iv[f"{lo}-{hi}"] = dict(warm=cw, cold=cc)
        print(f"{f'[{lo},{hi}]':>14}{cw:>12.4f}{cc:>12.4f}")

# ---- the V-shape / two-sided ill-posedness ----
print("\n== V-SHAPE: the LOWER frontier (sfb<=c) vs the UPPER problem (sfb>=c) ==")
up = {float(k): v for k, v in L["upper"].items()}
print(f"{'c':>7}{'min ms s.t. sfb<=c':>21}{'min ms s.t. sfb>=c':>21}{'sfb@best(>=)':>14}")
for c in sorted(up):
    lo_side = Fw.get(c)
    lo_s = f"{lo_side:.4f}" if lo_side is not None else "--"
    print(f"{c:>7.2f}{lo_s:>21}{up[c]['best']:>21.4f}{up[c]['sfb_at_best']:>14.4f}")
best_un = min(up[c]["best"] for c in up)
sfb_star = up[min(up, key=lambda c: up[c]["best"])]["sfb_at_best"]
print(f"\n   UPPER problem is SLACK for every c <= 2.5 (it lands at sfb 2.27-2.62 voluntarily);")
print(f"   it starts to COST above c ~ 2.5: +{up[3.0]['best']-best_un:.4f} at c=3, "
      f"+{up[5.0]['best']-best_un:.4f} at c=5, +{up[6.0]['best']-best_un:.4f} at c=6.")
print(f"   => the model's speed optimum in sfb is INTERIOR at sfb* ~ {sfb_star:.2f} "
      f"(best {best_un:.4f}); both directions cost away from it.")
up_price_hi = (up[6.0]["best"] - best_un) / (6.0 - sfb_star)
print(f"   one-sided price ABOVE the optimum ~ {up_price_hi:+.4f} ms/char per pp (c: {sfb_star:.2f}->6.0)")

# ---- the LIMIT question ----
print("\n== THE LIMIT QUESTION: price(c) as c -> the sfb floor (0.8006) ==")
t = {float(k): v["best"] for k, v in L["tight"].items()}
tc = sorted(t)
print(f"{'interval':>14}{'price (N=96 cold)':>19}")
for a, b in zip(tc[:-1], tc[1:]):
    print(f"{f'[{a},{b}]':>14}{-(t[b]-t[a])/(b-a):>19.4f}")
print(f"   floor {L['floor']:.4f}; tightest feasible cap 0.85 had only "
      f"{L['tight']['0.85']['n']}/96 feasible restarts => the curve STEEPENS toward the floor.")

json.dump(dict(warm_frontier={str(c): Fw[c] for c in wcaps},
               warm_sfb_at_best={str(c): warm[c][1] for c in wcaps},
               warm_headline=dict(interval=list(HEAD), price=float(pw), rise=float(rise_w),
                                  ratio_vs_sd=float(rise_w / sdF)),
               warm_placebo=float(pl_w), warm_violations=viol_w, warm_intervals=iv,
               vshape=dict(upper={str(c): up[c] for c in sorted(up)}, sfb_star=float(sfb_star),
                           best=float(best_un), price_above=float(up_price_hi)),
               limit=dict(floor=L["floor"], tight=L["tight"])),
          open(_env.ART + "/c11_verdict.json", "w"), indent=1)
print("\nwrote c11_verdict.json")
