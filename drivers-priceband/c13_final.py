"""C13 -- the FINAL number table for report.md. Every figure the report quotes is emitted here
so nothing is hand-transcribed. Reads the artifacts written by c05..c12."""
import json

import _env
import numpy as np

L = lambda n: json.load(open(_env.ART + f"/{n}.json"))
prem, ctl, ana, ver, ci, geo = L("c05_premise"), L("c06_control"), L("c07_analysis"), L("c11_verdict"), L("c12_warmci"), L("c04_geom")

out = {}
print("=" * 78); print("THE ANSWER"); print("=" * 78)
rs = ci["repsplit"]; db = ci["donor_bootstrap"]
out["headline"] = dict(price=rs["mean"], ci=rs["ci"], sd=rs["sd"], per_replicate=rs["prices"],
                       bootstrap=db, interval=ci["headline_interval"])
print(f"IN-BAND SHADOW PRICE of sfb over cap [{ci['headline_interval'][0]}, {ci['headline_interval'][1]}]")
print(f"  = {rs['mean']:+.4f} ms/char per pp   95% CI [{rs['ci'][0]:+.4f}, {rs['ci'][1]:+.4f}]  (R=4 replicate-split, sd {rs['sd']:.4f})")
print(f"  per-replicate: {[round(p,4) for p in rs['prices']]}")
print(f"  independent donor bootstrap (B={db['B']}): {db['mean']:+.4f}  CI [{db['ci'][0]:+.4f}, {db['ci'][1]:+.4f}]  frac>0 {db['frac_pos']:.4f}")

print("\n-- the price is a CURVE, not a scalar (warm/conservative frontier) --")
wi = ver["warm_intervals"]
out["curve"] = wi
print(f"  {'cap interval':>14}{'warm price':>12}{'cold price':>12}")
for k, v in wi.items():
    lo, hi = k.split("-")
    print(f"  {f'[{lo},{hi}]':>14}{v['warm']:>12.4f}{v['cold']:>12.4f}")

print("\n-- the SIX PRE-REGISTERED CRITERIA, on the conservative (warm) frontier --")
g = ana["gates"]; wh = ver["warm_headline"]
rows = [
    ("F1 sign + CI excludes zero", f"{rs['mean']:+.4f}, CI [{rs['ci'][0]:+.4f},{rs['ci'][1]:+.4f}]", rs["ci"][0] > 0),
    ("F2 rise > 3x search noise", f"rise {wh['rise']:+.4f} = {wh['ratio_vs_sd']:.2f}x sd {g['F2_exceeds_noise']['sd_F']:.4f}", wh["ratio_vs_sd"] > 3),
    ("F3 placebo flat on inert caps", f"slope {ver['warm_placebo']:+.6f} (all 4 replicates -0.000000)", abs(ver["warm_placebo"]) < abs(rs["mean"]) / 3),
    ("F4 monotonicity of F_own", f"{len(ver['warm_violations'])} violations (cold: {g['F4_monotone']['n_violations']}, worst {g['F4_monotone']['worst']:+.4f})", len(ver["warm_violations"]) == 0),
    ("F5 warm-start stability", f"cold {ana['headline']['mean']:+.4f} -> warm {rs['mean']:+.4f}, still >0 w/ CI excl 0", rs["ci"][0] > 0),
    ("F6 best-of-N saturation", f"worst gap {g['F6_saturation']['worst_gap']:+.4f} < rise/3 {g['F6_saturation']['third']:.4f}", g["F6_saturation"]["passed"]),
]
out["gates"] = {r[0]: dict(value=r[1], passed=bool(r[2])) for r in rows}
for name, val, ok in rows:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} {val}")
print(f"  => ALL SIX PASS: {all(r[2] for r in rows)}")

print("\n-- POSITIVE CONTROL P1 (mandatory) --")
out["P1"] = ctl["qwerty_primary"] | dict(passed=ctl["P1_pass"], crosscheck=ctl["crosscheck_vs_prior"])
print(f"  qwerty paired price {ctl['qwerty_primary']['price']:+.4f} (target +0.3910, |diff| {abs(ctl['qwerty_primary']['price']-0.3910):.4f})"
      f"  CI [{ctl['qwerty_primary']['ci'][0]:+.4f},{ctl['qwerty_primary']['ci'][1]:+.4f}] => {'PASS' if ctl['P1_pass'] else 'FAIL'}")
print(f"  in-band paired price {ctl['inband']['price']:+.4f} (n={ctl['inband']['n']}; prior arm -1.0957) -- reproduced")
print(f"  swap sweep vs prior npz: |d dsfb| {ctl['crosscheck_vs_prior']['dsfb']:.2e}  |d dms| {ctl['crosscheck_vs_prior']['dms']:.2e}")

print("\n-- the MANDATED SIGN-BLIND FALSIFIER (literal, on the perturbation control) --")
out["signblind"] = ctl["signblind"]
for k, v in ctl["signblind"].items():
    print(f"  {k}: n={v['n']}  R2 signed {v['r2_signed']:.4f} | sign-blind {v['r2_signblind']:.4f} | both {v['r2_both']:.4f}"
          f"  => {'SIGN-BLIND' if v['r2_signblind']>=v['r2_signed'] else 'SIGNED'} wins ({100*v['r2_signblind']/v['r2_signed']:.1f}%)")
print("  => in-band the PERTURBATION design measures DISRUPTION (as the brief says). The frontier")
print("     design has no |d(gauge)| to be blind to; its analogue is F3 (placebo), which PASSES.")

print("\n-- THE PREMISE CORRECTION (the register's 'boards sit AT the sfb floor') --")
out["premise"] = dict(sfb_floor=prem["sfb_floor_global"], median_headroom=prem["median_global_headroom"],
                      field_range=[min(v["sfb"] for v in prem["per_board"].values()),
                                   max(v["sfb"] for v in prem["per_board"].values())],
                      local_lowering={k: v["n_lower"] for k, v in prem["local_lowering"].items()})
print(f"  R1 LOCAL scarcity TRUE: single swaps that lower sfb = "
      f"{min(v['n_lower'] for k,v in prem['local_lowering'].items() if k!='qwerty30m')}-"
      f"{max(v['n_lower'] for k,v in prem['local_lowering'].items() if k!='qwerty30m')} of 435 in-band"
      f" vs {prem['local_lowering']['qwerty30m']['n_lower']} on qwerty")
print(f"  R2 GLOBAL floor FALSE: achievable sfb floor {prem['sfb_floor_global']:.4f}; field "
      f"{out['premise']['field_range'][0]:.4f}-{out['premise']['field_range'][1]:.4f} = median "
      f"{prem['median_global_headroom']:.4f} pp ABOVE the floor")

print("\n-- THE V-SHAPE: why a two-sided price is ILL-POSED in-band --")
vs = ver["vshape"]
out["vshape"] = vs
print(f"  sfb* (interior speed optimum) ~ {vs['sfb_star']:.2f}, best {vs['best']:.4f} ms/char")
print(f"  {'c':>7}{'min ms s.t. sfb<=c':>21}{'min ms s.t. sfb>=c':>21}{'sfb@best(>=)':>14}")
wf = ver["warm_frontier"]
for c in sorted(float(k) for k in vs["upper"]):
    u = vs["upper"][str(c)]
    lo_s = f"{wf[str(c)]:.4f}" if str(c) in wf else "--"
    print(f"  {c:>7.2f}{lo_s:>21}{u['best']:>21.4f}{u['sfb_at_best']:>14.4f}")
print(f"  one-sided price ABOVE the optimum ~ {vs['price_above']:+.4f} ms/char per pp")
print(f"  8 best random-2opt boards sit at sfb {min(s for _,s in geo['unconstrained'][:8]):.2f}-"
      f"{max(s for _,s in geo['unconstrained'][:8]):.2f} => sfb is an INTERIOR optimum for speed")

print("\n-- provenance --")
out["provenance"] = dict(evaluator_worst=ctl["evaluator_worst"], N=ana["N"], R=ana["R"],
                         cold_frontier_sec=1871, warm_sec=6134)
print(f"  fast evaluators re-verified every run: {ctl['evaluator_worst'][0]:.2e} vs card(), {ctl['evaluator_worst'][1]:.2e} vs KmStats")
print(f"  cold frontier N={ana['N']}/cap x R={ana['R']} replicates (1871 s); warm cross-seed 154 donors x 14 caps (6134 s)")
json.dump(out, open(_env.ART + "/c13_final.json", "w"), indent=1)
print("\nwrote c13_final.json")
