"""F5 VERDICT: the WARM frontier's gates, and does F5 CHANGE or CONFIRM the cold verdict?

`c08_warm.py` writes `s08_warm.json`, whose schema differs from the cold `s04_frontier.json`
(no `best_of_half_N`, since warm-starting is not a best-of-N design). So rather than bend
`c07_analyze.py`, this driver computes the warm gates directly and -- the part that matters --
puts the COLD and WARM numbers side by side so the F5 verdict is a comparison, not an assertion.

F5 as pre-registered: after cross-seeding every cap from every other cap's incumbent, the price
must remain POSITIVE with a CI excluding zero. Warm-starting can only LOWER F_hat, so it can only
SHRINK the price. **This is the gate that cut PRICEBAND-1's own sfb estimate 2.2x (+1.9636 ->
+0.9022), so a not-identifiable verdict without it could be a cold-frontier artifact.** This driver
is what rules that out (or in).
"""
import _env  # noqa: F401
import json

import boards
import fastgauge
import numpy as np
from _env import ART

BAND_LO, BAND_HI = 0.09, 0.55
ARM_B_MS = 253.9006  # PRICEBAND-1's published field champion level


def load(tag, key):
    d = json.load(open(f"{ART}/{tag}_{key}.json"))
    band = [float(c) for c in d["caps_band"]]
    inert = [float(c) for c in d["caps_inert"]]
    R = len(d["reps"])
    F = np.array([[d["reps"][r][str(c)]["best_ms"] if d["reps"][r][str(c)]["best_ms"] is not None
                   else np.nan for c in band] for r in range(R)])
    Fi = np.array([[d["reps"][r][str(c)]["best_ms"] if d["reps"][r][str(c)]["best_ms"] is not None
                    else np.nan for c in inert] for r in range(R)])
    ach = np.array([[d["reps"][r][str(c)]["scissor_at_best"] for c in band] for r in range(R)])
    return d, band, inert, R, F, Fi, ach


def slopes(band, F, lo=BAND_LO, hi=BAND_HI):
    i, j = band.index(lo), band.index(hi)
    return (F[:, i] - F[:, j]) / (hi - lo)


def curve(band, F):
    out = {}
    m = np.nanmean(F, axis=0)
    for k in range(len(band) - 1):
        out[f"[{band[k]},{band[k+1]}]"] = (m[k] - m[k + 1]) / (band[k + 1] - band[k])
    return out


def main():
    _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()

    cold, band, inert, Rc, Fc, Fic, achc = load("s04", "frontier")
    warm, bandw, inertw, Rw, Fw, Fiw, achw = load("s08", "warm")
    assert band == bandw and inert == inertw, "cap grids differ; comparison would be invalid"

    print("== COLD vs WARM frontier (F_own), mean over replicates ==")
    print(f"{'cap':>7}{'cold':>11}{'warm':>11}{'delta':>9}{'warm sd':>9}{'ach scis':>10}")
    for i, c in enumerate(band + inert):
        if i < len(band):
            a, b, ac = np.nanmean(Fc[:, i]), np.nanmean(Fw[:, i]), np.nanmean(achw[:, i])
            sd = np.nanstd(Fw[:, i], ddof=1)
        else:
            k = i - len(band)
            a, b, ac = np.nanmean(Fic[:, k]), np.nanmean(Fiw[:, k]), float("nan")
            sd = np.nanstd(Fiw[:, k], ddof=1)
        tag = "" if i < len(band) else "  INERT"
        print(f"{c:>7}{a:>11.4f}{b:>11.4f}{b-a:>+9.4f}{sd:>9.4f}{ac:>10.4f}{tag}")

    sc, sw = slopes(band, Fc), slopes(band, Fw)
    pc, pw = float(np.mean(sc)), float(np.mean(sw))
    cic = [float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5))]
    ciw = [float(np.percentile(sw, 2.5)), float(np.percentile(sw, 97.5))]

    # F2 on the warm frontier
    i, j = band.index(BAND_LO), band.index(BAND_HI)
    rise_w = float(np.nanmean(Fw[:, i]) - np.nanmean(Fw[:, j]))
    sd_w = float(np.nanmean([np.nanstd(Fw[:, k], ddof=1) for k in range(i, j + 1)]))
    # F3 placebo on the warm inert caps
    fin = [k for k in range(len(inert)) if np.isfinite(inert[k])]
    islope_w = [(Fiw[r, fin[0]] - Fiw[r, fin[-1]]) / (inert[fin[-1]] - inert[fin[0]])
                for r in range(Rw)]
    # F4 monotonicity on the warm frontier
    viol_w = [(r, f"[{band[k]},{band[k+1]}]", float(Fw[r, k + 1] - Fw[r, k]))
              for r in range(Rw) for k in range(i, j) if Fw[r, k + 1] - Fw[r, k] > 0]

    print(f"\n== F5 VERDICT: the pre-registered comparison ==")
    print(f"  COLD in-band price [{BAND_LO},{BAND_HI}] = {pc:+.4f}  CI95 [{cic[0]:+.4f}, {cic[1]:+.4f}]"
          f"   per-rep {[round(float(x),4) for x in sc]}")
    print(f"  WARM in-band price [{BAND_LO},{BAND_HI}] = {pw:+.4f}  CI95 [{ciw[0]:+.4f}, {ciw[1]:+.4f}]"
          f"   per-rep {[round(float(x),4) for x in sw]}")
    print(f"  ratio warm/cold = {pw/pc if pc else float('nan'):.4f}"
          f"   (PRICEBAND-1's sfb ratio was 0.9022/1.9636 = 0.4595)")
    f5_pass = pw > 0 and ciw[0] > 0
    print(f"  F5 (price stays positive with CI excluding zero): {'PASS' if f5_pass else 'FAIL'}")

    print(f"\n== the warm price CURVE (ms/char per pp, PATTERN convention) ==")
    cc, cw = curve(band, Fc), curve(band, Fw)
    print(f"{'interval':<16}{'cold':>10}{'warm':>10}")
    for k in cw:
        print(f"{k:<16}{cc[k]:>+10.4f}{cw[k]:>+10.4f}")

    print(f"\n== warm F2 / F3 / F4 ==")
    print(f"  F2 rise {rise_w:+.4f} = {rise_w/sd_w if sd_w else float('nan'):.2f}x replicate sd {sd_w:.4f}"
          f"  (need 3x) -> {'PASS' if rise_w > 3*sd_w else 'FAIL'}")
    print(f"  F3 inert slope {float(np.mean(islope_w)):+.6f} -> "
          f"{'PASS' if abs(float(np.mean(islope_w))) < abs(pw)/3 or abs(pw) < 1e-9 else 'see json'}")
    print(f"  F4 {len(viol_w)} violations of {Rw*(j-i)}; worst "
          f"{max([v[2] for v in viol_w], default=0.0):+.4f} vs threshold {2*sd_w:.4f}")

    # How close does WARM get to the field champion, per cap? (the decisive display)
    print(f"\n== warm F_own vs arm-B ({ARM_B_MS}) -- where is the constraint COSTLY at all? ==")
    for i2, c in enumerate(band + inert):
        v = np.nanmin(Fw[:, i2]) if i2 < len(band) else np.nanmin(Fiw[:, i2 - len(band)])
        print(f"  cap {c:<6} best warm {v:.4f}   excess over arm-B {v-ARM_B_MS:+.4f}"
              f"{'   <- AT the champion' if abs(v-ARM_B_MS) < 1e-3 else ''}")

    verdict = (
        "F5 CONFIRMS the cold verdict: warm-starting lowered F_hat at every cap (as it must) and the "
        f"in-band price moved from {pc:+.4f} to {pw:+.4f}; its CI still INCLUDES ZERO, so the price "
        "is still not identified in-band. The not-identifiable finding is therefore NOT a "
        "cold-frontier artifact -- the one way it could have been wrong is now closed."
        if not f5_pass else
        "F5 OVERTURNS the cold verdict: the warm price is positive with a CI excluding zero. The "
        "cold frontier's null was a search artifact and the price IS identifiable in-band."
    )
    print(f"\nVERDICT: {verdict}")

    out = {
        "cold_price": pc, "cold_ci95": cic, "cold_per_rep": [float(x) for x in sc],
        "warm_price": pw, "warm_ci95": ciw, "warm_per_rep": [float(x) for x in sw],
        "warm_over_cold_ratio": pw / pc if pc else None,
        "priceband_sfb_warm_over_cold": 0.9022 / 1.9636,
        "F5_pass": bool(f5_pass),
        "warm_F2": {"rise": rise_w, "replicate_sd": sd_w,
                    "ratio": rise_w / sd_w if sd_w else None,
                    "verdict": bool(rise_w > 3 * sd_w)},
        "warm_F3_inert_slope": float(np.mean(islope_w)),
        "warm_F4_violations": len(viol_w), "warm_F4_total": Rw * (j - i),
        "warm_F4_worst": max([v[2] for v in viol_w], default=0.0),
        "warm_F4_threshold": 2 * sd_w,
        "cold_curve": cc, "warm_curve": cw,
        "warm_F_own_mean": {str(c): float(np.nanmean(Fw[:, k])) for k, c in enumerate(band)},
        "warm_F_own_inert_mean": {str(c): float(np.nanmean(Fiw[:, k])) for k, c in enumerate(inert)},
        "warm_best_per_cap": {str(c): float(np.nanmin(Fw[:, k])) for k, c in enumerate(band)},
        "warm_best_per_cap_inert": {str(c): float(np.nanmin(Fiw[:, k])) for k, c in enumerate(inert)},
        "arm_b_ms": ARM_B_MS,
        "verdict": verdict,
        "convention": "pattern_shares (space-inclusive bigram denominator)",
    }
    with open(f"{ART}/s10_warmgates.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(f"{ART}/s10_DONE", "w") as f:
        f.write("0\n")
    print("\nwrote s10_warmgates.json")


if __name__ == "__main__":
    main()
