"""ARM H REFLECTION Q(a) + Q(d). POST-HOC; changes no registered verdict, tests two.

Q(a) IS 30.0% THE RIGHT RATIO? Numerator 4.1646 is the HEADLINE's oxey improvement: WARM
     (arm-B-seeded), FEASIBLE, 13/13 caps held. Denominator 13.8841 is armh-cold-r2's: COLD,
     INFEASIBLE (5 caps violated), and OUT of band by +0.304. So the two differ in seed
     regime, feasibility status AND band status. Interrogate whether that makes it a
     mixed-frame quantity, and derive the defensible forms.

Q(d) IS 2.024x DISTINGUISHABLE FROM n=5 SAMPLING NOISE? An F-ratio on the two variances plus a
     bootstrap. If not, "the seed family is a fifth quadruple leg" is UNSUPPORTED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import armh_constants as AH  # noqa: E402

STATE = Path("/local/home/zegertho/agent/state/armh/artifacts")
ARMG = Path("/local/home/zegertho/agent/state/armg/artifacts")


def main() -> int:
    out: dict = {"POST_HOC": "computed for the reflection pass, after result commit c85623d."}
    summ = json.load(open(STATE / "runs" / "armh-summary.json"))
    j = json.load(open(STATE / "judgement.json"))
    ARMB_OX = AH.ARMH_REF["oxey-style"]

    # ================= Q(a) THE RATIO =================
    gate = json.load(open(STATE / "gate-armh.json"))["results"]
    cold = {t: v for t, v in gate.items() if t.startswith("armh-cold")}
    warm = {t: v for t, v in gate.items() if t.startswith("armh-warm")}
    best_cold = min(cold.items(), key=lambda kv: kv[1]["oxey"])
    best_warm = min(warm.items(), key=lambda kv: kv[1]["oxey"])

    num = ARMB_OX - best_warm[1]["oxey"]           # 4.1646, FEASIBLE + in-band
    den_asreg = ARMB_OX - best_cold[1]["oxey"]     # 13.8841, INFEASIBLE + out of band
    out["Qa_as_registered"] = {
        "numerator": num, "denominator": den_asreg, "ratio": num / den_asreg,
        "numerator_provenance": {
            "tag": best_warm[0], "layout": best_warm[1]["layout"],
            "seed_regime": "WARM (arm B injected into all 20 islands)",
            "feasible": best_warm[1]["FEASIBLE"],
            "n_caps_violated": best_warm[1]["n_axes_violated"],
            "ms_minus_armB": best_warm[1]["ms_minus_armB"], "in_band": True},
        "denominator_provenance": {
            "tag": best_cold[0], "layout": best_cold[1]["layout"],
            "seed_regime": "COLD (uniform random, no injection)",
            "feasible": best_cold[1]["FEASIBLE"],
            "n_caps_violated": best_cold[1]["n_axes_violated"],
            "ms_minus_armB": best_cold[1]["ms_minus_armB"], "in_band": False},
    }
    same_conv = True          # both nested-bad_redirect, both blend-v1
    same_seedfam = True       # both 31337+104729r
    same_budget = True        # both 1M requested, both cleared the 80% floor
    out["Qa_footing_audit"] = {
        "same_oxey_convention": same_conv,
        "same_corpus_and_frame": True,
        "same_seed_family": same_seedfam,
        "same_budget_and_floor": same_budget,
        "same_SEED_REGIME": False,
        "same_FEASIBILITY_STATUS": False,
        "same_BAND_STATUS": False,
        "VERDICT": ("MIXED-FRAME on three legs: the numerator is WARM/FEASIBLE/IN-BAND and the "
                    "denominator is COLD/INFEASIBLE/OUT-OF-BAND. Convention, corpus, seed "
                    "family and budget DO match, so it is not a convention or corpus artifact "
                    "-- but 'collects 30.0% of the available headroom' reads as a fraction of a "
                    "COMPARABLE total, and the denominator is not comparable: it is not a "
                    "quantity any feasible layout could ever have reached."),
    }

    # -- the defensible forms --
    forms = {}
    # F1: within the SAME feasibility+band frame -- what is achievable while feasible?
    #     denominator = the best oxey among ALL layouts in my archive that are FEASIBLE.
    coll = j["archive_sweep"]["collected_layouts"]
    feas_ox = [v["oxey"] for t, v in gate.items() if v["FEASIBLE"]]
    ship = json.load(open(STATE / "verify-headline.json"))
    forms["F1_within_feasible_frame"] = {
        "definition": ("headline's gain / the best gain achieved by ANY feasible layout found. "
                       "Same frame on all three legs. Answers 'did the constrained search get "
                       "the most it could?'"),
        "value": 1.0,
        "note": ("degenerate BY CONSTRUCTION: the headline IS the best feasible layout found, so "
                 "this ratio is 1.0 and carries no information about the LEVER. It is the right "
                 "frame for a different question than the one the 30.0% was answering."),
    }
    # F2: the SPEED-only relaxation -- hold feasibility, drop only the band.
    axes_ok_out_of_band = [(t, v) for t, v in gate.items()
                           if v["axes_ok"] and not v["speed_ok"]]
    forms["F2_band_relaxation_only"] = {
        "definition": "extra oxey obtainable by keeping all 13 caps but leaving the band",
        "n_such_layouts_in_my_archive": len(axes_ok_out_of_band),
        "value": None,
        "note": ("EMPTY in my archive -- 0 layouts hold all 13 caps while out of band, so this "
                 "decomposition is not estimable from my data. Registering the absence rather "
                 "than substituting a different denominator."),
    }
    # F3: the honest two-number statement -- no ratio at all.
    forms["F3_two_numbers_no_ratio"] = {
        "definition": ("state both quantities and what each one IS, and let the reader see they "
                       "are different objects. This is the form I now recommend."),
        "collected_under_full_non_inferiority": num,
        "reachable_when_13_caps_are_ABANDONED": den_asreg,
        "cost_of_abandoning": {"n_caps_violated": best_cold[1]["n_axes_violated"],
                              "ms_out_of_band_by": best_cold[1]["ms_minus_armB"]},
    }
    # F4: the ratio as an UPPER BOUND on the constrained/unconstrained gap, explicitly labelled
    forms["F4_ratio_as_labelled_bound"] = {
        "definition": ("keep 4.1646/13.8841 = 30.0% but label it exactly: 'the constrained "
                       "collect is 30.0% of what an UNCONSTRAINED, INFEASIBLE, out-of-band cold "
                       "champion reached' -- a cross-frame COMPARISON, not a fraction of "
                       "available headroom."),
        "value": num / den_asreg,
        "why_still_useful": ("the DIRECTION and ORDER OF MAGNITUDE are the finding: most of the "
                            "oxey range is on the far side of the caps. That survives the "
                            "reframing; only the word 'of' does not."),
    }
    out["Qa_defensible_forms"] = forms
    out["Qa_RECOMMENDED"] = (
        "Do NOT register 30.0% as 'the constrained arm collects 30.0% of the lever'. Register "
        "F3 (two numbers, no ratio) as the headline and F4 (the ratio with its cross-frame "
        "label) as the secondary. The 30.0% is arithmetically correct and its CONCLUSION is "
        "right -- most of the oxey range sits beyond the caps -- but as written it is a "
        "fraction whose denominator no feasible layout could attain, which is exactly the "
        "'number wrong while the conclusion is right' class.")

    # ================= Q(d) IS 2.024x DISTINGUISHABLE FROM n=5 NOISE? =================
    mine = np.array([r["search_fitness"] for r in summ["phase1_baseline"]
                     if r["tag"].startswith("baseline-r") and r["clears_floor"]])
    sd_H = float(np.std(mine, ddof=1))
    # arm G's own 5 baseline champions, read from ITS authoritative summary (its per-run
    # record). ⚠ my first version globbed runs/baseline-r*.json and picked up the .ckpt.json
    # sidecars too -- caught by a KeyError, not by reading.
    gsum = json.load(open(ARMG / "runs" / "armg-summary.json"))
    grows = [r for r in gsum["runs"] if r["arm"] == "baseline" and r.get("clears_floor")]
    assert len(grows) == 5, [r.get("tag", r.get("r")) for r in grows]
    gms = np.array([r["search_fitness"] for r in grows])
    # cross-check: arm G registered sd_G = 0.049171
    assert abs(float(np.std(gms, ddof=1)) - 0.049171) < 5e-6, float(np.std(gms, ddof=1))
    sd_G = float(np.std(gms, ddof=1))
    ratio = sd_H / sd_G
    var_ratio = (sd_H ** 2) / (sd_G ** 2)

    from scipy import stats as st
    n1 = len(mine)
    n2 = len(gms)
    df1, df2 = n1 - 1, n2 - 1
    # two-sided F test on the variance ratio
    p_two = 2 * min(st.f.cdf(var_ratio, df1, df2), 1 - st.f.cdf(var_ratio, df1, df2))
    # the critical sd ratio at alpha=0.05, two-sided -- how big must it be to be detectable?
    f_hi = st.f.ppf(0.975, df1, df2)
    f_lo = st.f.ppf(0.025, df1, df2)
    # 95% CI for the variance ratio, hence for the sd ratio
    ci_var = (var_ratio / f_hi, var_ratio / f_lo)
    ci_sd = (float(np.sqrt(ci_var[0])), float(np.sqrt(ci_var[1])))

    # bootstrap: resample both sets of 5, distribution of the sd ratio
    rng = np.random.default_rng(31337)
    boots = []
    for _ in range(20000):
        a = rng.choice(mine, size=n1, replace=True)
        b = rng.choice(gms, size=n2, replace=True)
        sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
        if sb > 0:
            boots.append(sa / sb)
    boots = np.array(boots)
    # NULL simulation: two samples of 5 from ONE normal -- how often does the sd ratio reach 2x?
    pooled = float(np.sqrt((np.var(mine, ddof=1) * df1 + np.var(gms, ddof=1) * df2)
                           / (df1 + df2)))
    null = []
    for _ in range(200000):
        a = rng.normal(0, pooled, n1)
        b = rng.normal(0, pooled, n2)
        sb = np.std(b, ddof=1)
        if sb > 0:
            null.append(np.std(a, ddof=1) / sb)
    null = np.array(null)
    p_null_ge = float((null >= ratio).mean())
    p_null_two = float(((null >= ratio) | (null <= 1 / ratio)).mean())

    out["Qd_sd_ratio_significance"] = {
        "sd_H": sd_H, "n_H": n1, "sd_G": sd_G, "n_G": n2,
        "sd_G_source": "arm G's own runs/baseline-r*.json champion fitnesses, re-read by me",
        "sd_ratio": ratio, "variance_ratio": var_ratio,
        "F_test_two_sided_p": float(p_two),
        "F_crit_hi_at_alpha0.05": float(f_hi),
        "sd_ratio_needed_for_significance_at_alpha0.05": float(np.sqrt(f_hi)),
        "CI95_sd_ratio": ci_sd,
        "bootstrap_20k_sd_ratio_p2.5_p50_p97.5": [float(np.percentile(boots, 2.5)),
                                                  float(np.percentile(boots, 50)),
                                                  float(np.percentile(boots, 97.5))],
        "null_sim_200k_P(ratio>=observed)": p_null_ge,
        "null_sim_200k_P(two_sided)": p_null_two,
        "null_sim_P(ratio>=2.0)_from_one_population": float((null >= 2.0).mean()),
        "VERDICT": None,
    }
    sig = p_two < 0.05
    out["Qd_sd_ratio_significance"]["VERDICT"] = (
        f"NOT DISTINGUISHABLE FROM n=5 SAMPLING NOISE. Observed sd ratio {ratio:.3f} "
        f"(variance ratio {var_ratio:.3f}); two-sided F test p = {p_two:.3f}; 95% CI on the sd "
        f"ratio [{ci_sd[0]:.3f}, {ci_sd[1]:.3f}] CONTAINS 1.0. At n=5 vs n=5 an sd ratio must "
        f"exceed {np.sqrt(f_hi):.3f} to reach alpha=0.05, and two samples drawn from ONE "
        f"population reach a ratio >= 2.0 in {(null >= 2.0).mean() * 100:.1f}% of 200,000 "
        f"simulated draws. => 'the seed family is a fifth quadruple leg' is UNSUPPORTED by this "
        f"evidence and should be WITHDRAWN."
        if not sig else
        f"DISTINGUISHABLE: p = {p_two:.4f}, CI {ci_sd}.")
    out["Qd_what_survives"] = (
        "What survives is WEAKER and still worth registering: sd_H and sd_G are two estimates "
        "of the SAME unknown quantity that happen to differ 2x, and at n=5 that is unremarkable "
        "-- which means neither is a usable ruler on its own. That is an argument for the floor "
        "rule from the PRECISION side rather than the seed-family side: a 2x-uncertain ruler "
        "cannot adjudicate a 1.4x band question, which is exactly what my headline turned on. "
        "The bit-exact repro control still stands as an ENGINE-identity control; it simply "
        "cannot carry an attribution claim it was never powered to make.")

    json.dump(out, open(STATE / "reflect-ratio-and-sd.json", "w"), indent=1, default=str)

    print("=" * 96)
    print("Q(a) THE 30.0% RATIO")
    print("=" * 96)
    a = out["Qa_as_registered"]
    print(f"  numerator   {a['numerator']:.4f}  from {a['numerator_provenance']['tag']}  "
          f"WARM feasible={a['numerator_provenance']['feasible']} "
          f"caps={a['numerator_provenance']['n_caps_violated']} "
          f"dms={a['numerator_provenance']['ms_minus_armB']:+.4f}")
    print(f"  denominator {a['denominator']:.4f}  from {a['denominator_provenance']['tag']}  "
          f"COLD feasible={a['denominator_provenance']['feasible']} "
          f"caps={a['denominator_provenance']['n_caps_violated']} "
          f"dms={a['denominator_provenance']['ms_minus_armB']:+.4f}")
    print(f"  ratio {a['ratio']:.4f}")
    print(f"  footing: {out['Qa_footing_audit']['VERDICT']}")
    print(f"\n  RECOMMENDED: {out['Qa_RECOMMENDED']}")
    print("\n" + "=" * 96)
    print("Q(d) IS 2.024x REAL?")
    print("=" * 96)
    d = out["Qd_sd_ratio_significance"]
    print(f"  sd_H {d['sd_H']:.6f} (n={d['n_H']})   sd_G {d['sd_G']:.6f} (n={d['n_G']})   "
          f"ratio {d['sd_ratio']:.4f}")
    print(f"  F two-sided p = {d['F_test_two_sided_p']:.4f}   "
          f"95% CI on sd ratio [{d['CI95_sd_ratio'][0]:.3f}, {d['CI95_sd_ratio'][1]:.3f}]")
    print(f"  sd ratio needed for alpha=0.05 at n=5 vs n=5: "
          f"{d['sd_ratio_needed_for_significance_at_alpha0.05']:.3f}")
    print(f"  from ONE population, P(sd ratio >= 2.0) = "
          f"{d['null_sim_P(ratio>=2.0)_from_one_population']:.4f}")
    print(f"\n  {d['VERDICT']}")
    print(f"\n  SURVIVES: {out['Qd_what_survives']}")
    print(f"\nWROTE {STATE / 'reflect-ratio-and-sd.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
