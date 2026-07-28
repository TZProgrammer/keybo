"""ARM E pre-run analysis — predict the outcome from the curves alone, before spending 10M evals.

Arm D's most useful artifact was this: its headroom analysis measured that **92.5% of the clamped
headroom remaining from arm A's champion was MECHANISM-WRONG** (collectable only by making the
layout slower), and all five of those gauges then moved in the predicted direction. That is what
made arm D's (iii) verdict a *prediction confirmed* rather than a surprise.

Arm E asks the same three questions of the ARCHIVE curves:

  1. WHERE IS EACH CLAMPED ARCHIVE CURVE MINIMIZED, and is that direction mechanism-RIGHT? Under
     CLAMP the objective is separable, so each term's best attainable value is `min over [lo, hi]`.
     `EXPECTED_SIGN[g] = +1` means a HIGHER level costs MORE time, so a curve minimized ABOVE the
     incumbent's level is asking the search to make the layout mechanistically WORSE.
     ⚠ Trap 53: read this off the CURVE (piecewise coeffs + knot), never off the linearized
     `weight` — under random400 only 3 of 5 wrong-signed weights were actually exploitable.
  2. HOW MUCH IN-DOMAIN SIGNAL IS THERE? If the total range is tiny, a flat objective is plausible
     and the search wanders. Arm D refuted flatness for random400 (1730/1730 distinct); do not
     assume that repeats — measure it here (trap: do not inherit a sibling's finding as a premise).
  3. DOES THE CLAMP EVEN BIND IN THE BAND? This is arm E's premise: under ARCHIVE weights the
     incumbents are in-domain on 12-14 of 14 gauges, so unlike arm D the clamp should be nearly
     inert at the starting point. Quantify: how much attribution does CLAMP freeze into a constant
     across the near-optimal band, and how many exact ties does it create?

Also prices the whole frozen board under the ARCHIVE curves, so "does a clamped ARCHIVE objective
rank the near-optimal band correctly?" is answered BEFORE the search — and if it ranks arm B (the
fastest layout) top, that is a materially different prior than if it does not.

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

import arme_obj as AE  # noqa: E402
import evobj as EV  # noqa: E402
from arme_load import load_curves, load_meta  # noqa: E402
from keybo.analysis.evidence_scorer import (  # noqa: E402
    CLAMP, EXPECTED_SIGN, EXTRAPOLATE, LIVE_GAUGES,
)

ARCHIVE_JSON = ("/local/home/zegertho/agent/state/evidence-scorer/artifacts/"
                "arm-archive400-native.json")
RANDOM_JSON = ("/local/home/zegertho/agent/state/evidence-scorer/artifacts/"
               "arm-random400-native.json")
OPTEV = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMD = Path("/local/home/zegertho/agent/state/armd/artifacts")
OUT = Path("/local/home/zegertho/agent/state/arme/artifacts/prerun-arme.json")


def curve_extremes(curve, n: int = 20001) -> dict:
    """min/max of the fitted curve ON its domain, and where the min sits."""
    lo, hi = curve.domain
    xs = np.linspace(lo, hi, n)
    ys = curve.price_many(xs, policy=CLAMP)
    i_min, i_max = int(np.argmin(ys)), int(np.argmax(ys))
    at = float(xs[i_min])
    edge = ("lo" if abs(at - lo) < (hi - lo) * 1e-6
            else "hi" if abs(at - hi) < (hi - lo) * 1e-6 else "interior")
    return {
        "domain": [lo, hi], "knot": curve.knot, "form": curve.form,
        "min_price": float(ys[i_min]), "min_at_level": at, "min_at": edge,
        "max_price": float(ys[i_max]), "in_domain_range": float(ys[i_max] - ys[i_min]),
        "shap_share_pct": curve.shap_share_pct, "weight_ms_per_unit": curve.weight,
    }


def main() -> int:
    curves = load_curves(ARCHIVE_JSON)
    meta = load_meta(ARCHIVE_JSON)
    rand_curves = load_curves(RANDOM_JSON)
    assert meta.get("surface_frame") == "native", meta.get("surface_frame")
    assert meta.get("pool") == "archive-400", meta.get("pool")
    print(f"weights: source={meta.get('source')} frame={meta.get('surface_frame')} "
          f"corpus={meta.get('corpus')} pool={meta.get('pool')} n={meta.get('n_layouts')}")
    print(f"  surrogate R2: in-sample {meta.get('surrogate_r2_in_sample'):.4f}  "
          f"holdout {meta.get('surrogate_r2_holdout'):.4f}  "
          f"effective dof {meta.get('effective_dof'):.4f}")
    rmeta = load_meta(RANDOM_JSON)
    print(f"  (arm D's random400 fit for contrast: in-sample "
          f"{rmeta.get('surrogate_r2_in_sample'):.4f}  holdout "
          f"{rmeta.get('surrogate_r2_holdout'):.4f}  dof {rmeta.get('effective_dof'):.4f})")

    # ---- how disjoint are the two fits' domains? (the arm's whole premise) ----------------
    print(f"\n{'gauge':<12} {'ARCHIVE domain':<26} {'RANDOM400 domain':<26} {'overlap':>8}")
    n_disjoint = 0
    domain_cmp = {}
    for name in LIVE_GAUGES:
        a, r = curves[name].domain, rand_curves[name].domain
        overlap = not (a[1] < r[0] or r[1] < a[0])
        n_disjoint += (not overlap)
        domain_cmp[name] = {"archive": list(a), "random400": list(r), "overlap": overlap}
        print(f"{name:<12} [{a[0]:11.4f},{a[1]:10.4f}] [{r[0]:11.4f},{r[1]:10.4f}] "
              f"{str(overlap):>8}")
    print(f"\n{n_disjoint} of 14 domains are FULLY DISJOINT between the two fits")

    # ---- 1 & 2: the clamped objective's attainable floor and its in-domain signal ---------
    print(f"\n{'gauge':<12} {'form':<10} {'min@':<9} {'min lvl':>9} {'min price':>10} "
          f"{'range':>8} {'shap%':>7} {'expsign':>8}")
    ext = {}
    for name in LIVE_GAUGES:
        e = curve_extremes(curves[name])
        ext[name] = e
        print(f"{name:<12} {e['form']:<10} {e['min_at']:<9} {e['min_at_level']:9.4f} "
              f"{e['min_price']:10.4f} {e['in_domain_range']:8.4f} "
              f"{e['shap_share_pct']:7.2f} {EXPECTED_SIGN[name]:+8.1f}")
    bound = sum(e["min_price"] for e in ext.values())
    worst = sum(e["max_price"] for e in ext.values())
    span = sum(e["in_domain_range"] for e in ext.values())
    print(f"\nCLAMPED LOWER BOUND (sum of per-gauge in-domain minima) = {bound:.4f}")
    print(f"CLAMPED UPPER BOUND                                     = {worst:.4f}")
    print(f"total in-domain signal (sum of per-gauge ranges)         = {span:.4f}")
    armd_pre = json.load(open(ARMD / "pre-run-analysis.json"))
    print(f"  (arm D's random400 in-domain signal was "
          f"{armd_pre['total_in_domain_signal']:.4f}, lower bound "
          f"{armd_pre['clamped_lower_bound']:.4f})")
    print("  The bound is NOT necessarily attainable: one permutation must produce all 14 levels")
    print("  at once, and the gauges are correlated.")

    # ---- price the frozen board under the ARCHIVE curves ---------------------------------
    fe = EV.FastEval(corpus=None, weights_json=ARCHIVE_JSON, with_surface=True)
    ve = AE.ValidatedClampedEval(fe, curves, policy=CLAMP)
    print(f"\ncorpus dir: {fe.corpus_dir}")
    assert str(fe.corpus_dir).endswith("blend-v1"), fe.corpus_dir

    from keybo.cli.score_evidence import _EXTRA_NAMED
    inc = json.load(open(OPTEV / "incumbent-reference.json"))
    board: dict[str, str] = {}
    for arm, label in (("evidence", "armA"), ("baseline", "armB"), ("constrained", "armC")):
        board[label] = json.load(open(OPTEV / f"runs/arm-{arm}.json"))["champion"]["layout"]
    board["armD"] = json.load(open(ARMD / "runs/arm-domain.json"))["champion"]["layout"]
    board.update(inc["incumbents"])
    board.update(inc["reference"])
    # trap 13: two layouts under one name. Cross-check the registry against the frozen reference.
    for name, lay in inc["incumbents"].items():
        if name in _EXTRA_NAMED and _EXTRA_NAMED[name] != lay:
            raise AssertionError(f"{name}: incumbent-reference.json has {lay!r} but the CLI "
                                 f"registry has {_EXTRA_NAMED[name]!r} (trap 13)")
    board["flagship-c3"] = _EXTRA_NAMED["flagship-c3"]

    perms = np.stack([EV.perm_of(lay) for lay in board.values()])
    g = fe.gauges(perms)
    ev_clamp = ve.evidence_score(g)
    ev_ext = ve.evidence_score_extrapolating(g)
    ood = fe.out_of_domain(g)
    n_ood = np.sum(np.stack([ood[m] for m in LIVE_GAUGES]), axis=0)

    print(f"\n{'layout':<16} {'ms/char':>10} {'ev CLAMP':>10} {'ev EXTRAP':>11} "
          f"{'clamp-ext':>10} {'n_ood':>6}")
    rows = []
    for i, (label, lay) in enumerate(board.items()):
        row = {"label": label, "layout": lay,
               "ms_per_char": float(g["_ms_per_char"][i]),
               "ev_clamp": float(ev_clamp[i]), "ev_extrapolate": float(ev_ext[i]),
               "n_out_of_domain": int(n_ood[i]),
               "gauges": {m: float(g[m][i]) for m in LIVE_GAUGES},
               "out_of_domain": [m for m in LIVE_GAUGES if bool(ood[m][i])]}
        rows.append(row)
        print(f"{label:<16} {row['ms_per_char']:10.4f} {row['ev_clamp']:10.4f} "
              f"{row['ev_extrapolate']:11.4f} "
              f"{row['ev_clamp'] - row['ev_extrapolate']:10.4f} {row['n_out_of_domain']:6d}")

    by_label = {r["label"]: r for r in rows}

    # ---- 3: HEADROOM from the best incumbent, split by mechanism -------------------------
    # Arm D measured this from arm A's champion. Arm E's natural anchor is the best INCUMBENT,
    # because under archive weights that is the in-domain point a search starts near — and it is
    # the layout arm E has to beat (keybo-lsb, 254.6307). Report from armB too, the fastest.
    headroom = {}
    for anchor in ("keybo-lsb", "armB"):
        a = by_label[anchor]
        print(f"\nHEADROOM from {anchor} (ms/char {a['ms_per_char']:.4f}, "
              f"n_ood {a['n_out_of_domain']}/14):")
        print(f"{'gauge':<12} {'level':>9} {'clamped':>9} {'wants':>9} {'headroom':>9} "
              f"{'dir':>5} {'expsign':>8} {'mechanism':>10} {'shap%':>7}")
        detail = {}
        right_sum = wrong_sum = 0.0
        for name in LIVE_GAUGES:
            c = curves[name]
            lo, hi = c.domain
            level = a["gauges"][name]
            clamped = min(max(level, lo), hi)
            here = float(c.price_many(np.array([level]), policy=CLAMP)[0])
            best = ext[name]["min_price"]
            wants = ext[name]["min_at_level"]
            head = here - best  # >= 0 by construction
            if head <= 1e-12:
                direction, mech = "none", "attained"
            else:
                direction = "up" if wants > clamped else "down"
                sign = EXPECTED_SIGN[name]
                # EXPECTED_SIGN +1 => a higher level costs MORE time => "up" is mechanism-WRONG
                if sign > 0:
                    mech = "WRONG" if direction == "up" else "right"
                else:
                    mech = "right" if direction == "up" else "WRONG"
            if mech == "WRONG":
                wrong_sum += head
            elif mech == "right":
                right_sum += head
            detail[name] = {"level": level, "clamped_level": clamped, "domain": [lo, hi],
                            "price_here": here, "best_in_domain_price": best,
                            "wants_level": wants, "headroom": head, "direction": direction,
                            "expected_sign": EXPECTED_SIGN[name], "mechanism": mech,
                            "shap_share_pct": c.shap_share_pct,
                            "in_domain": c.in_domain(level)}
            print(f"{name:<12} {level:9.4f} {clamped:9.4f} {wants:9.4f} {head:9.4f} "
                  f"{direction:>5} {EXPECTED_SIGN[name]:+8.1f} {mech:>10} "
                  f"{c.shap_share_pct:7.2f}")
        total = right_sum + wrong_sum
        print(f"  remaining clamped headroom = {total:.4f}")
        print(f"    mechanism-RIGHT (also makes it faster): {right_sum:8.4f} "
              f"({100 * right_sum / total:5.1f}%)")
        print(f"    mechanism-WRONG (makes it slower):      {wrong_sum:8.4f} "
              f"({100 * wrong_sum / total:5.1f}%)")
        print(f"    [arm D's figure from arm A's champion was 92.5% WRONG]")
        headroom[anchor] = {"per_gauge": detail, "total": total,
                            "mechanism_right": right_sum, "mechanism_wrong": wrong_sum,
                            "wrong_pct": 100 * wrong_sum / total}

    # ---- does CLAMP bind in the band at all? --------------------------------------------
    band = ["armA", "armB", "armC", "armD", "keybo-lsb", "keybo-lsb+lm", "lsb-sib",
            "archive-1843", "archive-1846", "flagship-c3"]
    print(f"\nover the near-optimal band ({len(band)} layouts): which gauges does CLAMP freeze?")
    print(f"{'gauge':<12} {'clamped levels':>28} {'frozen?':>8} {'n outside':>10} {'shap%':>7}")
    frozen_share = 0.0
    frozen = {}
    for name in LIVE_GAUGES:
        c = curves[name]
        lo, hi = c.domain
        raw = [by_label[b]["gauges"][name] for b in band]
        vals = [min(max(v, lo), hi) for v in raw]
        n_out = sum(1 for v in raw if not (lo <= v <= hi))
        is_frozen = (max(vals) - min(vals)) < 1e-9
        if is_frozen:
            frozen_share += c.shap_share_pct
        frozen[name] = {"clamped_min": min(vals), "clamped_max": max(vals),
                        "frozen": is_frozen, "n_outside": n_out,
                        "shap_share_pct": c.shap_share_pct}
        print(f"{name:<12} [{min(vals):11.4f},{max(vals):11.4f}] {str(is_frozen):>8} "
              f"{n_out:10d} {c.shap_share_pct:7.2f}")
    print(f"\nCLAMP freezes {frozen_share:.2f}% of the fitted attribution into a CONSTANT across "
          f"the band")
    armd_head = json.load(open(ARMD / "headroom.json"))
    print(f"  (under random400 it froze {armd_head['frozen_attribution_pct']:.2f}%)")

    # ---- degeneracy: distinct layouts, identical clamped score ---------------------------
    ties = []
    for x, y in itertools.combinations(band, 2):
        if by_label[x]["layout"] == by_label[y]["layout"]:
            continue
        if abs(by_label[x]["ev_clamp"] - by_label[y]["ev_clamp"]) < 1e-9:
            ties.append({"a": x, "b": y,
                         "d_clamp": abs(by_label[x]["ev_clamp"] - by_label[y]["ev_clamp"]),
                         "d_ms": abs(by_label[x]["ms_per_char"] - by_label[y]["ms_per_char"])})
    print(f"\ndegeneracy over the band: {len(ties)} exact ties")
    spread_cl = max(by_label[b]["ev_clamp"] for b in band) - min(by_label[b]["ev_clamp"]
                                                                for b in band)
    spread_ex = max(by_label[b]["ev_extrapolate"] for b in band) - min(
        by_label[b]["ev_extrapolate"] for b in band)
    print(f"  band spread: ev_CLAMP {spread_cl:.4f} vs ev_EXTRAPOLATE {spread_ex:.4f} "
          f"({spread_cl / spread_ex:.3f}x); total in-domain signal {span:.4f}")

    # ---- rank agreement, in the band and on a wide pool (trap 52) ------------------------
    from scipy.stats import spearmanr
    ms_band = np.array([by_label[b]["ms_per_char"] for b in band])
    r_cl = float(spearmanr(np.array([by_label[b]["ev_clamp"] for b in band]), ms_band).statistic)
    r_ex = float(spearmanr(np.array([by_label[b]["ev_extrapolate"] for b in band]),
                           ms_band).statistic)
    print(f"\nspearman(ev, ms/char) IN THE BAND (n={len(band)}): CLAMP {r_cl:+.4f} | "
          f"EXTRAPOLATE {r_ex:+.4f}   (want +1)")
    # incumbents only — the six the arm has to beat, no diagnostic champions
    inc_only = ["keybo-lsb", "keybo-lsb+lm", "lsb-sib", "archive-1843", "archive-1846",
                "flagship-c3"]
    r_inc = float(spearmanr(np.array([by_label[b]["ev_clamp"] for b in inc_only]),
                            np.array([by_label[b]["ms_per_char"] for b in inc_only])).statistic)
    print(f"spearman over the SIX INCUMBENTS only (n=6): CLAMP {r_inc:+.4f}")

    rng = np.random.default_rng(20260728)
    rperms = np.stack([np.concatenate([rng.permutation(30).astype(np.int32),
                                       np.array([30], dtype=np.int32)]) for _ in range(400)])
    rg = fe.gauges(rperms)
    r_rand_cl = float(spearmanr(ve.evidence_score(rg), rg["_ms_per_char"]).statistic)
    r_rand_ex = float(spearmanr(ve.evidence_score_extrapolating(rg),
                                rg["_ms_per_char"]).statistic)
    print(f"spearman(ev, ms/char) on 400 RANDOM perms: CLAMP {r_rand_cl:+.4f} | "
          f"EXTRAPOLATE {r_rand_ex:+.4f}")
    print("  ⚠ trap 52: the archive fit's pool is the ARCHIVE, so neither of these is the")
    print("    in-sample number — but a wide-pool rho still says nothing about the narrow band.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "corpus": "blend-v1 (production default)", "corpus_dir": str(fe.corpus_dir),
        "frame": meta.get("surface_frame"), "weights_json": ARCHIVE_JSON, "weights_meta": meta,
        "domain_comparison_vs_random400": domain_cmp, "n_domains_fully_disjoint": n_disjoint,
        "curve_extremes": ext,
        "clamped_lower_bound": bound, "clamped_upper_bound": worst,
        "total_in_domain_signal": span,
        "armD_random400_in_domain_signal": armd_pre["total_in_domain_signal"],
        "board": rows,
        "headroom": headroom,
        "band": band, "frozen_by_clamp": frozen, "frozen_attribution_pct": frozen_share,
        "armD_frozen_attribution_pct": armd_head["frozen_attribution_pct"],
        "exact_ties": ties,
        "band_spread_clamp": spread_cl, "band_spread_extrapolate": spread_ex,
        "spearman_band": {"clamp": r_cl, "extrapolate": r_ex},
        "spearman_incumbents_only": r_inc,
        "spearman_random400": {"clamp": r_rand_cl, "extrapolate": r_rand_ex},
        "modelled_only": ("MODELLED ONLY: fitted-surface attribution, not measured typing speed. "
                          "No layout here is promoted or adopted."),
    }, open(OUT, "w"), indent=1)
    print(f"\nWROTE {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
