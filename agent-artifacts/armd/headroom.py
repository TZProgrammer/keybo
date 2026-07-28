"""Where can a CLAMPED search still gain, and does gaining there make a layout FASTER?

This is the pre-run analysis that makes arm D's prediction reasoned rather than guessed, and it
needs no search at all. Under CLAMP the objective is separable: `sum_g price_g(clip(level_g))`.
So from arm A's champion I can compute, per gauge:

  * HEADROOM = price at arm A's clamped level  -  the curve's best attainable in-domain price.
    That is exactly what the clamped search has left to win on that gauge.
  * DIRECTION = does the gauge have to go UP or DOWN to collect it.
  * MECHANISM = does moving that way make a layout faster or slower in reality.

The third column is the one that matters, and trap 53 is why it has to be read off the CURVE
rather than the summary weight: a linearized sign does not tell you which way a hinge's minimum
lies. Splitting the headroom into mechanism-correct and mechanism-wrong halves predicts arm D's
direction before a single evaluation is spent.

Also measures the DEGENERACY the clamp introduces, because that is the (iii) outcome's signature:
how many distinct near-optimal layouts does the clamped objective map to the SAME score?

MODELLED ONLY. Corpus: blend-v1 (production default). Frame: .native.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")

import armd_obj as AD  # noqa: E402
import evobj as EV  # noqa: E402
from armd_load import load_curves  # noqa: E402
from keybo.analysis.evidence_scorer import CLAMP, EXPECTED_SIGN, LIVE_GAUGES  # noqa: E402

ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
PRE = Path("/local/home/zegertho/agent/state/armd/artifacts/pre-run-analysis.json")
OUT = Path("/local/home/zegertho/agent/state/armd/artifacts/headroom.json")


def main() -> int:
    curves = load_curves(ARM_JSON)
    pre = json.load(open(PRE))
    rows = {r["label"]: r for r in pre["board"]}
    ext = pre["curve_extremes"]

    # ---- headroom from arm A's champion, split by mechanism ------------------------------
    # EXPECTED_SIGN[g] = +1 means "a HIGHER level costs MORE time" (so lower is faster).
    # A gauge whose clamped minimum sits ABOVE arm A's level therefore asks the search to make
    # the layout mechanistically WORSE; below, mechanistically better.
    a = rows["armA"]
    print(f"arm A champion {a['layout']}  ms/char {a['ms_per_char']:.4f}  "
          f"ev_clamp {a['ev_clamp']:.4f}  n_ood {a['n_out_of_domain']}/14")
    print(f"\n{'gauge':<12} {'level':>9} {'clamped':>9} {'wants':>9} {'headroom':>9} "
          f"{'dir':>5} {'exp_sign':>9} {'mechanism':>11} {'shap%':>7}")
    detail = {}
    right_sum = wrong_sum = 0.0
    for name in LIVE_GAUGES:
        c = curves[name]
        lo, hi = c.domain
        level = a["gauges"][name]
        clamped_level = min(max(level, lo), hi)
        here = c.price(level, policy=CLAMP)
        best = ext[name]["min_price"]
        wants = ext[name]["min_at_level"]
        head = here - best  # >= 0 by construction
        # Which way must the CLAMPED level move to collect it?
        if head <= 1e-12:
            direction, mech = "none", "attained"
        else:
            direction = "up" if wants > clamped_level else "down"
            # EXPECTED_SIGN +1 => higher costs more time => "up" is mechanistically WORSE.
            sign = EXPECTED_SIGN[name]
            if sign > 0:
                mech = "WRONG" if direction == "up" else "right"
            else:
                mech = "right" if direction == "up" else "WRONG"
        if mech == "WRONG":
            wrong_sum += head
        elif mech == "right":
            right_sum += head
        detail[name] = {
            "level": level, "clamped_level": clamped_level, "domain": [lo, hi],
            "price_here": here, "best_in_domain_price": best, "wants_level": wants,
            "headroom": head, "direction": direction, "expected_sign": EXPECTED_SIGN[name],
            "mechanism": mech, "shap_share_pct": c.shap_share_pct,
            "in_domain": c.in_domain(level),
        }
        print(f"{name:<12} {level:9.4f} {clamped_level:9.4f} {wants:9.4f} {head:9.4f} "
              f"{direction:>5} {EXPECTED_SIGN[name]:+9.1f} {mech:>11} {c.shap_share_pct:7.2f}")

    total = right_sum + wrong_sum
    print(f"\nremaining clamped headroom from arm A's champion = {total:.4f}")
    print(f"  mechanism-RIGHT (would also make it faster): {right_sum:8.4f}  "
          f"({100 * right_sum / total:5.1f}%)")
    print(f"  mechanism-WRONG (would make it slower):      {wrong_sum:8.4f}  "
          f"({100 * wrong_sum / total:5.1f}%)")
    print(f"  cross-check: ev_clamp - lower_bound = "
          f"{a['ev_clamp'] - pre['clamped_lower_bound']:.4f}")

    # ---- how much attribution the clamp turns into a CONSTANT on the near-optimal band ----
    band = ["armA", "armB", "armC", "keybo-lsb", "keybo-lsb+lm", "lsb-sib",
            "archive-1843", "archive-1846", "flagship-c3"]
    print(f"\nover the near-optimal band ({len(band)} layouts): which gauges does CLAMP freeze?")
    print(f"{'gauge':<12} {'clamped levels':>28} {'frozen?':>9} {'shap%':>7}")
    frozen_share = 0.0
    frozen = {}
    for name in LIVE_GAUGES:
        c = curves[name]
        lo, hi = c.domain
        vals = [min(max(rows[b]["gauges"][name], lo), hi) for b in band]
        is_frozen = (max(vals) - min(vals)) < 1e-9
        if is_frozen:
            frozen_share += c.shap_share_pct
        frozen[name] = {"clamped_min": min(vals), "clamped_max": max(vals),
                        "frozen": is_frozen, "shap_share_pct": c.shap_share_pct}
        print(f"{name:<12} [{min(vals):11.4f},{max(vals):11.4f}] {str(is_frozen):>9} "
              f"{c.shap_share_pct:7.2f}")
    print(f"\nCLAMP freezes {frozen_share:.2f}% of the fitted attribution into a CONSTANT "
          f"across the near-optimal band")

    # ---- degeneracy: distinct layouts, identical clamped score ---------------------------
    print(f"\ndegeneracy check over the {len(band)}-layout band:")
    ties = []
    for x, y in itertools.combinations(band, 2):
        if rows[x]["layout"] == rows[y]["layout"]:
            continue
        d_cl = abs(rows[x]["ev_clamp"] - rows[y]["ev_clamp"])
        d_ex = abs(rows[x]["ev_extrapolate"] - rows[y]["ev_extrapolate"])
        d_ms = abs(rows[x]["ms_per_char"] - rows[y]["ms_per_char"])
        if d_cl < 1e-9:
            ties.append({"a": x, "b": y, "d_clamp": d_cl, "d_extrap": d_ex, "d_ms": d_ms})
            print(f"  TIE  {x} vs {y}: |dev_clamp| = {d_cl:.3e} (EXACT) while "
                  f"|dev_extrap| = {d_ex:.4f} and |dms/char| = {d_ms:.4f}")
    if not ties:
        print("  no exact ties")
    spread_cl = max(rows[b]["ev_clamp"] for b in band) - min(rows[b]["ev_clamp"] for b in band)
    spread_ex = (max(rows[b]["ev_extrapolate"] for b in band)
                 - min(rows[b]["ev_extrapolate"] for b in band))
    print(f"\n  band spread: ev_CLAMP {spread_cl:.4f} vs ev_EXTRAPOLATE {spread_ex:.4f} "
          f"({spread_cl / spread_ex:.3f}x) — total in-domain signal is "
          f"{pre['total_in_domain_signal']:.4f}")

    # ---- rank agreement in the band ONLY (trap 52: validate in the band of use) -----------
    from scipy.stats import spearmanr
    ms = np.array([rows[b]["ms_per_char"] for b in band])
    r_cl = spearmanr(np.array([rows[b]["ev_clamp"] for b in band]), ms).statistic
    r_ex = spearmanr(np.array([rows[b]["ev_extrapolate"] for b in band]), ms).statistic
    print(f"\n  spearman(ev, ms/char) IN THE BAND (n={len(band)}): "
          f"CLAMP {r_cl:+.4f} | EXTRAPOLATE {r_ex:+.4f}   (want +1)")

    # ---- a random pool, for the wide-pool contrast the weights were fitted on ------------
    fe = EV.FastEval(corpus=None, weights_json=ARM_JSON, with_surface=True)
    ce = AD.ClampedEval(fe, policy=CLAMP)
    rng = np.random.default_rng(20260728)
    perms = np.stack([np.concatenate([rng.permutation(30).astype(np.int32),
                                      np.array([30], dtype=np.int32)]) for _ in range(400)])
    g = fe.gauges(perms)
    r_rand_cl = spearmanr(ce.evidence_score(g), g["_ms_per_char"]).statistic
    r_rand_ex = spearmanr(fe.evidence_score(g), g["_ms_per_char"]).statistic
    print(f"  spearman(ev, ms/char) on 400 RANDOM perms: "
          f"CLAMP {r_rand_cl:+.4f} | EXTRAPOLATE {r_rand_ex:+.4f}")
    print("  (the weights were FITTED on a 400-random-permutation pool, so a high number here")
    print("   is in-sample and — per trap 52 — says nothing about the band a search operates in)")

    json.dump({
        "corpus": "blend-v1 (production default)", "frame": "native",
        "armA_headroom": detail,
        "headroom_total": total,
        "headroom_mechanism_right": right_sum,
        "headroom_mechanism_wrong": wrong_sum,
        "headroom_wrong_pct": 100 * wrong_sum / total,
        "band": band,
        "frozen_by_clamp": frozen,
        "frozen_attribution_pct": frozen_share,
        "exact_ties": ties,
        "band_spread_clamp": spread_cl,
        "band_spread_extrapolate": spread_ex,
        "spearman_band": {"clamp": float(r_cl), "extrapolate": float(r_ex)},
        "spearman_random400": {"clamp": float(r_rand_cl), "extrapolate": float(r_rand_ex)},
        "modelled_only": "MODELLED ONLY: fitted-surface attribution, not measured typing speed.",
    }, open(OUT, "w"), indent=1)
    print(f"\nWROTE {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
