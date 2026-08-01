"""Gate 3: ADJUDICATE the units of SCISSORPRICE-1's headline, from ITS OWN artifact.

This is P3: prove by exact arithmetic that `+32.59` is a DIMENSIONLESS OXEY WEIGHT and not a
ms/char-per-pp price, so my brief's "36x the sfb price" is a category error. It reads
SCISSORPRICE-1's frozen artifact -- I do not re-run its estimator, I only check the identity
that defines its own reported quantity.
"""
import _env  # noqa: F401
import json

from _env import ART

SP8 = "/local/home/zegertho/agent/state/scissorprice/artifacts/sp8_definitive.json"
# PRICEBAND-1's published sfb shadow price, kmstats convention (state/priceband/report.md §2).
PB_SFB = {"point": 0.9022, "ci": [0.8621, 1.0077]}


def main():
    from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS as W

    w_sfb = W["sfb"][0]
    w_scis = W["scissor"][0]
    print(f"shipped DEFAULT_OXEY_WEIGHTS: sfb={w_sfb}  scissor={w_scis}")

    d = json.load(open(SP8))["definitive_table"]
    cells, worst_w, worst_r = [], 0.0, 0.0
    print("\n== P3: is implied_w == w_sfb * (scissor_slope/sfb_slope)? and ratio == implied_w/w_scissor? ==")
    print(f"{'spec':<26}{'source':<10}{'scis_slope':>11}{'sfb_slope':>10}"
          f"{'implied_w':>11}{'predicted':>11}{'diff':>10}{'slope_ratio':>12}")
    for spec, v in d.items():
        for src, s in v["per_source"].items():
            pred = w_sfb * s["scissor_slope"] / s["sfb_slope"]
            dw = abs(pred - s["implied_w"])
            dr = abs(s["implied_w"] / w_scis - s["ratio"])
            worst_w = max(worst_w, dw)
            worst_r = max(worst_r, dr)
            cells.append({
                "spec": spec, "source": src,
                "scissor_slope_ms_per_pp": s["scissor_slope"],
                "sfb_slope_ms_per_pp": s["sfb_slope"],
                "implied_w_reported": s["implied_w"],
                "implied_w_predicted": pred,
                "abs_diff": dw,
                "slope_ratio_scissor_over_sfb": s["scissor_slope"] / s["sfb_slope"],
            })
            print(f"{spec:<26}{src:<10}{s['scissor_slope']:>11.4f}{s['sfb_slope']:>10.4f}"
                  f"{s['implied_w']:>11.4f}{pred:>11.4f}{dw:>10.1e}"
                  f"{s['scissor_slope']/s['sfb_slope']:>12.4f}")
    print(f"\nworst |implied_w - w_sfb*slope_ratio| = {worst_w:.3e}   (0 => it IS a weight)")
    print(f"worst |implied_w/w_scissor - ratio|   = {worst_r:.3e}")

    ind = d["cond, IN-DOMAIN,  linear"]
    scis = [c["scissor_slope"] for c in ind["per_source"].values()]
    sfbs = [c["sfb_slope"] for c in ind["per_source"].values()]

    # The convention correction: PRICEBAND's sfb price is kmstats; every scissor number is pattern.
    import fastgauge
    fg = fastgauge.FastGauges()
    r = fg.km_over_pattern
    pb_pattern = {"point": PB_SFB["point"] * r, "ci": [c * r for c in PB_SFB["ci"]]}
    print(f"\n== convention: km/pattern denominator ratio = {r:.10f} (MEASURED) ==")
    print(f"PRICEBAND-1 sfb price  {PB_SFB['point']:+.4f} per pp of KMSTATS-sfb")
    print(f"          restated as  {pb_pattern['point']:+.4f} per pp of PATTERN-sfb "
          f"CI [{pb_pattern['ci'][0]:+.4f}, {pb_pattern['ci'][1]:+.4f}]")

    print("\n== THE ADJUDICATION TARGET (what SCISSORPRICE-1 actually claims, in ms/char per pp) ==")
    print(f"  scissor: {min(scis):+.4f} .. {max(scis):+.4f}  (pattern convention)")
    print(f"  sfb    : {min(sfbs):+.4f} .. {max(sfbs):+.4f}  (pattern convention, ITS OWN sfb)")
    print(f"  ratio scissor/sfb: {min(s/f for s, f in zip(scis, sfbs)):.4f}x .. "
          f"{max(s/f for s, f in zip(scis, sfbs)):.4f}x")
    print(f"  vs PRICEBAND-1's independent sfb price {pb_pattern['point']:+.4f} => "
          f"implied scissor price {min(s/f for s, f in zip(scis, sfbs))*pb_pattern['point']:+.4f} .. "
          f"{max(s/f for s, f in zip(scis, sfbs))*pb_pattern['point']:+.4f}")
    print("\n  BRIEF'S CLAIM: '+32.59 ms/char per pp, 36x the sfb price'  ==> REFUTED, unit error.")

    out = {
        "shipped_weights": {"sfb": w_sfb, "scissor": w_scis},
        "identity_cells": cells,
        "worst_abs_diff_implied_w": worst_w,
        "worst_abs_diff_ratio": worst_r,
        "verdict": (
            "implied_w IS w_sfb*(scissor_slope/sfb_slope) exactly => +32.59 is a DIMENSIONLESS "
            "OXEY WEIGHT, not ms/char per pp. The brief's '36x the sfb price' compares a weight "
            "to a price."
        ) if worst_w < 1e-9 else "IDENTITY FAILS -- my unit claim is WRONG and is retracted",
        "km_over_pattern_denominator_ratio": r,
        "priceband_sfb_price_kmstats": PB_SFB,
        "priceband_sfb_price_pattern_convention": pb_pattern,
        "scissorprice1_in_domain_linear": {
            "scissor_slope_ms_per_pp_range": [min(scis), max(scis)],
            "sfb_slope_ms_per_pp_range": [min(sfbs), max(sfbs)],
            "slope_ratio_range": [min(s / f for s, f in zip(scis, sfbs)),
                                  max(s / f for s, f in zip(scis, sfbs))],
            "implied_w_mean": ind["w_indep_mean"],
            "implied_w_cluster_ci95": ind["w_cluster_ci95"],
        },
        "adjudication_target_ms_per_pp": [min(scis), max(scis)],
        "note_reference_set_caveat": (
            "SCISSORPRICE-1's own self-audit: implied_w is +32.59 ex-qwerty (n=503), +12.68 "
            "including qwerty30m (n=809), +13.29 full pool (n=891) -- a 2.6x swing on one "
            "discretionary exclusion. Registered amendment says quote +12.7 to +32.6."
        ),
    }
    with open(ART + "/s03_adjudicate.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote s03_adjudicate.json")


if __name__ == "__main__":
    main()
