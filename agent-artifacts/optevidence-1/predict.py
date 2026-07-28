"""PREDICTION pass — run BEFORE any search. Derives, from the fitted curves ALONE,
which direction a maximizer of the evidence objective will push each gauge."""
import json, sys
import numpy as np

ARM = sys.argv[1] if len(sys.argv) > 1 else \
    "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
d = json.load(open(ARM))
W = d["weights"]
EXP = {"sfb":1,"sfs":1,"sfb-dist":1,"sfs-dist":1,"lsb":1,"lsb-dist":1,"scissor":1,
       "redir":1,"imbalance":1,"alt":-1,"roll":-1,"sr-roll":-1,"comfort":1,"oxey-style":1}

def price(g, x):
    c = g["coeffs"]; k = g["knot"]
    if g["form"] == "linear":   return c[0] + c[1]*x
    if g["form"] == "quadratic":return c[0] + c[1]*x + c[2]*x*x
    return c[0] + c[1]*x + c[2]*np.maximum(x-k, 0.0)

print("%-11s %-9s %9s %9s | %-21s %-21s | %-24s %-22s %s" % (
    "gauge","form","slope_lo","slope_hi","valid_domain","observed_range",
    "argmin over observed","search pushes","mechanism"))
rows = []
for g in W["weights"]:
    lo, hi = g["observed_range"]; vlo, vhi = g["valid_domain"]
    c = g["coeffs"]; k = g["knot"]
    if g["form"] == "hinge":     s_lo, s_hi = c[1], c[1]+c[2]
    elif g["form"] == "linear":  s_lo = s_hi = c[1]
    else:                        s_lo, s_hi = c[1]+2*c[2]*lo, c[1]+2*c[2]*hi
    xs = np.linspace(lo, hi, 20001); ys = price(g, xs)
    am = float(xs[int(np.argmin(ys))])
    span = hi - lo
    ext_hi = float(price(g, np.array([hi + 3*span]))[0])
    ext_lo = float(price(g, np.array([lo - 3*span]))[0])
    if   abs(am-hi) < span*1e-4: push = "UP to max"    + (" & BEYOND" if s_hi < 0 else "")
    elif abs(am-lo) < span*1e-4: push = "DOWN to min"  + (" & BEYOND" if s_lo > 0 else "")
    else:                        push = "INTERIOR x*=%.3f" % am
    good = "penalty" if EXP[g["metric"]] > 0 else "reward"
    print("%-11s %-9s %9.4f %9.4f | [%7.4f,%8.4f] [%7.4f,%8.4f] | %-24s %-22s %s" % (
        g["metric"], g["form"], s_lo, s_hi, vlo, vhi, lo, hi,
        "x*=%.4f p=%.4f" % (am, float(ys.min())), push, good))
    rows.append(dict(metric=g["metric"], good=good, push=push, argmin=am, lo=lo, hi=hi,
                     vlo=vlo, vhi=vhi, s_lo=s_lo, s_hi=s_hi, ext_hi=ext_hi, ext_lo=ext_lo,
                     shap=g["shap_share_pct"], w=g["weight_ms_per_unit"]))
print()
print("=== PATHOLOGY: gauges where the objective PAYS to make a penalty worse ===")
bad = []
for r in rows:
    if r["good"] == "penalty" and "UP" in r["push"]:
        bad.append(r["metric"])
        print("  %-11s PENALTY, argmin at TOP of observed [%.4f,%.4f]; slope_hi=%+.4f; "
              "price 3-spans beyond hi = %+.4f  (valid_domain hi = %.4f)"
              % (r["metric"], r["lo"], r["hi"], r["s_hi"], r["ext_hi"], r["vhi"]))
    if r["good"] == "reward" and "DOWN" in r["push"]:
        bad.append(r["metric"])
        print("  %-11s REWARD, argmin at BOTTOM of observed [%.4f,%.4f]; slope_lo=%+.4f"
              % (r["metric"], r["lo"], r["hi"], r["s_lo"]))
print()
print("=== SHAP-share ordering (what the objective is mostly ABOUT) ===")
for r in sorted(rows, key=lambda r: -r["shap"]):
    print("  %-11s %5.2f%%  %s" % (r["metric"], r["shap"], r["push"]))
json.dump(rows, open("/local/home/zegertho/agent/state/optevidence/artifacts/prediction-curve-analysis.json","w"), indent=1)
print()
print("pathological_gauges =", bad)
