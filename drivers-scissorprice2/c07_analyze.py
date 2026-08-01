"""Evaluate the SEVEN pre-registered criteria and emit the price curve + CIs.

Two estimators, as pre-registered:
  F_own(c)  = best over that cap's OWN N restarts       (effort-symmetric; monotonicity and the
                                                         PLACEBO are computed on THIS one, because
                                                         a pooled estimator is monotone BY
                                                         CONSTRUCTION so its inert slope is FORCED
                                                         to ~0 and CANNOT FAIL -- the degenerate
                                                         placebo PRICEBAND-1 found in itself)
  F_pool(c) = best over EVERY board found anywhere with gauge <= c   (better point estimate)

Usage: c07_analyze.py <frontier-tag> <out-tag>
"""
import _env  # noqa: F401
import json
import sys

import boards
import fastgauge
import numpy as np
from _env import ART

# The headline priced interval: spans the optimized field's own scissor range [0.0889, 0.5173].
BAND_LO, BAND_HI = 0.09, 0.55


def caps_of(fr):
    band = [float(c) for c in fr["caps_band"]]
    inert = [float(c) for c in fr["caps_inert"]]
    return band, inert


def f_own(fr, rep, caps):
    return np.array([fr["reps"][rep][k]["best_ms"] if fr["reps"][rep][k]["best_ms"] is not None
                     else np.nan for k in [str(c) for c in caps]])


def f_pool_from(perms_ms, caps, fg, fs):
    """Pooled: for each cap, the best ms among ALL boards found anywhere with gauge <= cap."""
    out = []
    for c in caps:
        vals = [m for g, m in perms_ms if g <= c + 1e-9]
        out.append(min(vals) if vals else np.nan)
    return np.array(out)


def secant(caps, F):
    """Per-interval price = -dF/dc, i.e. (F[i]-F[i+1]) / (c[i+1]-c[i])."""
    out = {}
    for i in range(len(caps) - 1):
        a, b = caps[i], caps[i + 1]
        if np.isfinite(F[i]) and np.isfinite(F[i + 1]):
            out[f"[{a},{b}]"] = (F[i] - F[i + 1]) / (b - a)
    return out


def band_slope(caps, F, lo=BAND_LO, hi=BAND_HI):
    """Overall price across the priced band: (F(lo) - F(hi)) / (hi - lo)."""
    ci = {c: i for i, c in enumerate(caps)}
    if lo not in ci or hi not in ci:
        return None
    a, b = F[ci[lo]], F[ci[hi]]
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return (a - b) / (hi - lo)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "s04"
    out_tag = sys.argv[2] if len(sys.argv) > 2 else "s07"
    fr = json.load(open(f"{ART}/{tag}_frontier.json"))
    band, inert = caps_of(fr)
    R, N = len(fr["reps"]), fr["N"]
    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    scis = lambda p: fg.scissor_only(np.array(p)[:30])  # noqa: E731

    print(f"== analysing {tag}: N={N}, R={R}, band caps {band}, inert {inert} ==")

    # ---- F_own per replicate, and the pooled estimator ----
    own = np.array([f_own(fr, r, band) for r in range(R)])
    own_inert = np.array([f_own(fr, r, inert) for r in range(R)])
    allpts = []           # (gauge, ms) of every best board found, for the pooled estimator
    for r in range(R):
        for k, v in fr["reps"][r].items():
            if v["perm"] is not None:
                allpts.append((float(v["scissor_at_best"]), float(v["best_ms"])))
    pool_band = f_pool_from(allpts, band, fg, fs)
    pool_inert = f_pool_from(allpts, inert, fg, fs)

    print("\n== the frontier ==")
    print(f"{'cap':>7}{'F_own mean':>12}{'F_own sd':>10}{'F_pool':>10}{'scis@best':>11}"
          f"{'feas(min)':>10}{'restart sd':>11}")
    for i, c in enumerate(band + inert):
        col = own[:, i] if i < len(band) else own_inert[:, i - len(band)]
        pl = pool_band[i] if i < len(band) else pool_inert[i - len(band)]
        sc = [fr["reps"][r][str(c)]["scissor_at_best"] for r in range(R)]
        fe = [fr["reps"][r][str(c)]["n_feasible"] for r in range(R)]
        rsd = [fr["reps"][r][str(c)]["restart_sd"] for r in range(R)]
        rsd = [x for x in rsd if x is not None]
        mark = "" if i < len(band) else "  <- INERT"
        print(f"{c:>7}{np.nanmean(col):>12.4f}{np.nanstd(col, ddof=1):>10.4f}{pl:>10.4f}"
              f"{np.nanmean([x for x in sc if x is not None]):>11.4f}{min(fe):>10}"
              f"{np.mean(rsd) if rsd else float('nan'):>11.3f}{mark}")

    # ---- prices ----
    price_own = {k: np.mean([secant(band, own[r]).get(k, np.nan) for r in range(R)])
                 for k in secant(band, own[0])}
    price_pool = secant(band, pool_band)
    slopes = [band_slope(band, own[r]) for r in range(R)]
    slopes = [s for s in slopes if s is not None]
    slope_pool = band_slope(band, pool_band)

    print(f"\n== the price CURVE (ms/char per pp of scissor, PATTERN convention) ==")
    print(f"{'cap interval':<18}{'F_own price':>13}{'F_pool price':>14}")
    for k in price_pool:
        print(f"{k:<18}{price_own.get(k, float('nan')):>13.4f}{price_pool[k]:>14.4f}")

    # ---- F1: sign + CI (percentile over replicates) ----
    ci = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))] if len(slopes) > 1 else None
    pt = float(np.mean(slopes)) if slopes else None
    F1 = pt is not None and pt > 0 and ci is not None and ci[0] > 0

    # ---- F2: rise > 3x replicate sd of F at the priced caps ----
    ixlo, ixhi = band.index(BAND_LO), band.index(BAND_HI)
    rise = float(np.nanmean(own[:, ixlo]) - np.nanmean(own[:, ixhi]))
    rep_sd = float(np.nanmean([np.nanstd(own[:, i], ddof=1) for i in range(ixlo, ixhi + 1)]))
    F2 = rise > 3 * rep_sd

    # ---- F3: PLACEBO on F_own over the inert caps ----
    inert_slopes = []
    for r in range(R):
        v = own_inert[r]
        fin = np.isfinite(v)
        cs = [inert[i] for i in range(len(inert)) if fin[i] and np.isfinite(inert[i])]
        vs = [v[i] for i in range(len(inert)) if fin[i] and np.isfinite(inert[i])]
        if len(cs) > 1:
            inert_slopes.append((vs[0] - vs[-1]) / (cs[-1] - cs[0]))
    inert_pt = float(np.mean(inert_slopes)) if inert_slopes else None
    inert_ci = ([float(np.percentile(inert_slopes, 2.5)), float(np.percentile(inert_slopes, 97.5))]
                if len(inert_slopes) > 1 else None)
    F3 = (inert_pt is not None and pt is not None and abs(inert_pt) < abs(pt) / 3.0
          and inert_ci is not None and inert_ci[0] <= 0 <= inert_ci[1])
    # ALSO: are the inert LEVELS all equal (P4: the constraint is slack so they must agree)?
    inert_levels_spread = float(np.nanmax(own_inert) - np.nanmin(own_inert))
    inert_pool_spread = float(np.nanmax(pool_inert) - np.nanmin(pool_inert))

    # ---- F4: monotonicity of F_own over the priced range ----
    viol = []
    for r in range(R):
        for i in range(ixlo, ixhi):
            d = own[r, i + 1] - own[r, i]     # F must be non-increasing in c
            if np.isfinite(d) and d > 0:
                viol.append({"rep": r, "interval": f"[{band[i]},{band[i+1]}]", "rise": float(d)})
    percap_sd = float(np.nanmean([np.nanstd(own[:, i], ddof=1) for i in range(ixlo, ixhi + 1)]))
    worst_viol = max([v["rise"] for v in viol], default=0.0)
    F4 = worst_viol <= 2 * percap_sd

    # ---- F6: saturation, best-of-N vs best-of-N/2 ----
    gaps = []
    for r in range(R):
        for c in band[ixlo:ixhi + 1]:
            v = fr["reps"][r][str(c)]
            if v["best_of_half_N"] is not None and v["best_ms_2opt"] is not None:
                gaps.append(v["best_of_half_N"] - v["best_ms_2opt"])
    worst_gap = float(max(gaps)) if gaps else None
    F6 = worst_gap is not None and worst_gap < rise / 3.0

    # ---- F7: QUANTIZATION / NON-DEGENERACY (my added gate) ----
    from scipy.stats import spearmanr
    sp_caps, sp_ach = [], []
    for r in range(R):
        for c in band:
            v = fr["reps"][r][str(c)]
            if v["scissor_at_best"] is not None:
                sp_caps.append(c); sp_ach.append(v["scissor_at_best"])
    rho = float(spearmanr(sp_caps, sp_ach).statistic)
    # F7b: how many adjacent priced caps share a byte-identical best board?
    ident = 0
    tot = 0
    for r in range(R):
        for i in range(len(band) - 1):
            a = fr["reps"][r][str(band[i])]["perm"]
            b = fr["reps"][r][str(band[i + 1])]["perm"]
            if a is not None and b is not None:
                tot += 1
                if a == b:
                    ident += 1
    F7a = rho > 0.5
    F7b = tot > 0 and ident < tot / 2
    F7 = F7a and F7b
    # how near-binding is the constraint at each cap? (diagnostic, reported either way)
    bind = {}
    for c in band:
        ach = [fr["reps"][r][str(c)]["scissor_at_best"] for r in range(R)]
        ach = [x for x in ach if x is not None]
        bind[str(c)] = {"mean_achieved": float(np.mean(ach)) if ach else None,
                        "mean_slack_below_cap": float(c - np.mean(ach)) if ach else None}

    gates = {
        "F1_sign_and_CI": {"price": pt, "ci95": ci, "verdict": bool(F1)},
        "F2_exceeds_search_noise": {"rise": rise, "replicate_sd": rep_sd,
                                    "ratio": rise / rep_sd if rep_sd else None,
                                    "threshold": 3.0, "verdict": bool(F2)},
        "F3_placebo_inert_on_F_own": {"inert_slope": inert_pt, "inert_ci95": inert_ci,
                                      "in_band_price": pt,
                                      "inert_levels_spread_F_own": inert_levels_spread,
                                      "inert_levels_spread_F_pool": inert_pool_spread,
                                      "verdict": bool(F3)},
        "F4_monotonicity_F_own": {"n_violations": len(viol), "worst_rise": worst_viol,
                                  "per_cap_sd": percap_sd, "threshold": 2 * percap_sd,
                                  "violations": viol[:20], "verdict": bool(F4)},
        "F6_saturation": {"worst_N_vs_halfN_gap": worst_gap, "threshold": rise / 3.0 if rise else None,
                          "verdict": bool(F6)},
        "F7_quantization": {"F7a_spearman_cap_vs_achieved": rho, "F7a_threshold": 0.5,
                            "F7a": bool(F7a),
                            "F7b_identical_adjacent_best_boards": ident,
                            "F7b_total_adjacent_pairs": tot, "F7b": bool(F7b),
                            "binding": bind, "verdict": bool(F7)},
    }
    print("\n== GATES ==")
    for k, v in gates.items():
        print(f"  {k:<32} {'PASS' if v['verdict'] else 'FAIL'}")
    print(f"\n  in-band price [{BAND_LO},{BAND_HI}] = {pt:+.4f}  CI95 {ci}")
    print(f"  per-replicate slopes: {[round(s,4) for s in slopes]}")
    print(f"  pooled-estimator band slope = {slope_pool:+.4f}")
    print(f"  F2: rise {rise:+.4f} = {rise/rep_sd:.2f}x replicate sd {rep_sd:.4f}")
    print(f"  F3: inert slope {inert_pt:+.6f} CI {inert_ci}; "
          f"inert LEVEL spread F_own {inert_levels_spread:.4f} / F_pool {inert_pool_spread:.4f}")
    print(f"  F4: {len(viol)} violations, worst {worst_viol:+.4f} vs threshold {2*percap_sd:.4f}")
    print(f"  F7a: spearman(cap, achieved scissor) = {rho:+.4f}")
    print(f"  F7b: {ident}/{tot} adjacent cap pairs share a byte-identical best board")

    out = {"tag": tag, "N": N, "R": R, "band_caps": band, "inert_caps": [str(c) for c in inert],
           "priced_interval": [BAND_LO, BAND_HI],
           "convention": "pattern_shares (space-inclusive bigram denominator)",
           "F_own_per_rep": own.tolist(), "F_own_inert_per_rep": own_inert.tolist(),
           "F_pool_band": pool_band.tolist(), "F_pool_inert": pool_inert.tolist(),
           "price_curve_F_own": price_own, "price_curve_F_pool": price_pool,
           "band_price_point": pt, "band_price_ci95": ci, "band_price_per_rep": slopes,
           "band_price_pooled": slope_pool,
           "gates": gates,
           "all_gates_pass": all(v["verdict"] for v in gates.values()),
           "fasteval_worst": w1, "fastgauge_worst": w2}
    with open(f"{ART}/{out_tag}_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nALL GATES PASS: {out['all_gates_pass']}")
    print(f"wrote {out_tag}_analysis.json")


if __name__ == "__main__":
    main()
