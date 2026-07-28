"""Pre-run analysis that MUST happen before arm D, so the prediction is reasoned not guessed.

Three questions, all answerable from the curves alone — no search needed:

  1. WHERE IS EACH CLAMPED CURVE MINIMIZED? Under CLAMP the objective is
     `sum_g price_g(clip(level_g))`, so each term's best attainable value is
     `min over [lo, hi]` of the fitted curve. For a hinge that is an endpoint or the knot.
     The clamped objective therefore has a computable LOWER BOUND (`sum_g min_g`) — a bound the
     search cannot beat no matter what it does, and which no C30M permutation need attain.
  2. HOW MUCH SIGNAL IS LEFT IN-DOMAIN? `max - min` over the domain, per gauge. If the total
     in-domain range is tiny next to arm A's -45.44, outcome (iii) becomes plausible: a nearly
     flat objective means the search wanders.
  3. WHAT DID ARM A ACTUALLY EXPLOIT, and how much of that survives the clamp? Re-price arm A's
     champion under both policies, per gauge, and show which terms lose their reward.

Also prices the incumbents + arm B/C champions under CLAMP, because the question "does a clamped
evidence objective rank the near-optimal band correctly?" is answerable BEFORE spending 9.4M
evals — and if it ranks arm B above arm A's champion, that is a materially different prior than
if it does not.

MODELLED ONLY. Corpus: blend-v1 (production default). Frame: .native.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")

import armd_obj as AD  # noqa: E402
import evobj as EV  # noqa: E402
from armd_load import load_curves, load_meta  # noqa: E402
from keybo.analysis.evidence_scorer import CLAMP, LIVE_GAUGES  # noqa: E402

ARM_JSON = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
OPTEV = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
OUT = Path("/local/home/zegertho/agent/state/armd/artifacts/pre-run-analysis.json")


def curve_extremes(curve, n: int = 20001) -> dict:
    """min/max of the fitted curve ON its domain, and where the min sits."""
    lo, hi = curve.domain
    xs = np.linspace(lo, hi, n)
    ys = np.array([curve.price(float(x)) for x in xs])
    i_min, i_max = int(np.argmin(ys)), int(np.argmax(ys))
    at = xs[i_min]
    edge = "lo" if abs(at - lo) < (hi - lo) * 1e-6 else ("hi" if abs(at - hi) < (hi - lo) * 1e-6
                                                        else "interior")
    return {
        "domain": [lo, hi],
        "knot": curve.knot,
        "form": curve.form,
        "min_price": float(ys[i_min]),
        "min_at_level": float(at),
        "min_at": edge,
        "max_price": float(ys[i_max]),
        "in_domain_range": float(ys[i_max] - ys[i_min]),
        "shap_share_pct": curve.shap_share_pct,
        "weight_ms_per_unit": curve.weight,
    }


def main() -> int:
    curves = load_curves(ARM_JSON)
    meta = load_meta(ARM_JSON)
    print(f"weights: source={meta.get('source')} frame={meta.get('surface_frame')} "
          f"corpus={meta.get('corpus')} pool={meta.get('pool')} n={meta.get('n_layouts')}")
    assert meta.get("surface_frame") == "native", (
        f"ARM D requires the .native frame; weights say {meta.get('surface_frame')!r}"
    )

    # ---- 1 & 2: the clamped objective's attainable floor and its in-domain signal --------
    print(f"\n{'gauge':<12} {'form':<10} {'min@':<9} {'min level':>11} {'min price':>11} "
          f"{'in-dom range':>13} {'shap%':>7}")
    ext = {}
    for name in LIVE_GAUGES:
        e = curve_extremes(curves[name])
        ext[name] = e
        print(f"{name:<12} {e['form']:<10} {e['min_at']:<9} {e['min_at_level']:11.4f} "
              f"{e['min_price']:11.4f} {e['in_domain_range']:13.4f} {e['shap_share_pct']:7.2f}")
    bound = sum(e["min_price"] for e in ext.values())
    span = sum(e["in_domain_range"] for e in ext.values())
    worst = sum(e["max_price"] for e in ext.values())
    print(f"\nCLAMPED LOWER BOUND (sum of per-gauge in-domain minima) = {bound:.4f}")
    print(f"CLAMPED UPPER BOUND (sum of per-gauge in-domain maxima) = {worst:.4f}")
    print(f"total in-domain signal (sum of per-gauge ranges)        = {span:.4f}")
    print("  (the bound is NOT necessarily attainable: it ignores that one permutation must")
    print("   produce all 14 levels at once, and gauges are correlated.)")

    # ---- 3: re-price the frozen champions + incumbents under CLAMP -----------------------
    fe = EV.FastEval(corpus=None, weights_json=ARM_JSON, with_surface=True)
    ce = AD.ClampedEval(fe, policy=CLAMP)
    print(f"\ncorpus dir: {fe.corpus_dir}")

    board: dict[str, str] = {}
    for arm, label in (("evidence", "armA"), ("baseline", "armB"), ("constrained", "armC")):
        blob = json.load(open(OPTEV / f"runs/arm-{arm}.json"))
        board[label] = blob["champion"]["layout"]
    inc = json.load(open(OPTEV / "incumbent-reference.json"))
    board.update(inc["incumbents"])
    board.update(inc["reference"])
    # `flagship-c3` is on the brief's incumbent list (254.9761) but was NOT in OPTEVIDENCE-1's
    # incumbent-reference.json, so it has to come from the CLI registry — the authoritative
    # source both `analyze` and `score-evidence` name their boards from. Cross-check the five
    # that appear in BOTH against the registry rather than trusting either alone.
    from keybo.cli.score_evidence import _EXTRA_NAMED
    for name, lay in inc["incumbents"].items():
        if name in _EXTRA_NAMED and _EXTRA_NAMED[name] != lay:
            raise AssertionError(
                f"{name}: incumbent-reference.json has {lay!r} but the CLI registry has "
                f"{_EXTRA_NAMED[name]!r} — two layouts under one name (trap 13)"
            )
    board["flagship-c3"] = _EXTRA_NAMED["flagship-c3"]

    perms = np.stack([EV.perm_of(lay) for lay in board.values()])
    g = fe.gauges(perms)
    ev_extrap = fe.evidence_score(g)
    ev_clamp = ce.evidence_score(g)
    ood = fe.out_of_domain(g)
    n_ood = np.sum(np.stack([ood[m] for m in LIVE_GAUGES]), axis=0)

    print(f"\n{'layout':<16} {'ms/char':>10} {'ev EXTRAP':>11} {'ev CLAMP':>11} "
          f"{'clamp-extrap':>13} {'n_ood':>6}")
    rows = []
    for i, (label, lay) in enumerate(board.items()):
        row = {
            "label": label, "layout": lay,
            "ms_per_char": float(g["_ms_per_char"][i]),
            "ev_extrapolate": float(ev_extrap[i]),
            "ev_clamp": float(ev_clamp[i]),
            "n_out_of_domain": int(n_ood[i]),
            "gauges": {m: float(g[m][i]) for m in LIVE_GAUGES},
            "out_of_domain": [m for m in LIVE_GAUGES if bool(ood[m][i])],
        }
        rows.append(row)
        print(f"{label:<16} {row['ms_per_char']:10.4f} {row['ev_extrapolate']:11.4f} "
              f"{row['ev_clamp']:11.4f} {row['ev_clamp'] - row['ev_extrapolate']:13.4f} "
              f"{row['n_out_of_domain']:6d}")

    # rank correlation of each objective against ms/char over this near-optimal pool
    from scipy.stats import spearmanr
    ms = np.array([r["ms_per_char"] for r in rows])
    rho_ex = float(spearmanr(np.array([r["ev_extrapolate"] for r in rows]), ms).statistic)
    rho_cl = float(spearmanr(np.array([r["ev_clamp"] for r in rows]), ms).statistic)
    print(f"\nspearman(ev, ms/char) over this pool of {len(rows)}: "
          f"EXTRAPOLATE {rho_ex:+.4f} | CLAMP {rho_cl:+.4f}   (want +1 if the weights inform)")

    # ---- what arm A's exploit is worth under the clamp -----------------------------------
    a_idx = [r["label"] for r in rows].index("armA")
    print(f"\narm A champion, per-gauge price EXTRAPOLATE vs CLAMP:")
    print(f"{'gauge':<12} {'level':>10} {'domain':<24} {'extrap':>10} {'clamp':>10} {'lost':>10}")
    per_gauge = {}
    for name in LIVE_GAUGES:
        c = curves[name]
        lev = float(g[name][a_idx])
        pe = c.price(lev)
        pc = c.price(lev, policy=CLAMP)
        per_gauge[name] = {"level": lev, "extrapolate": pe, "clamp": pc, "lost": pc - pe,
                           "in_domain": c.in_domain(lev)}
        flag = "" if c.in_domain(lev) else "  <-- OUT"
        print(f"{name:<12} {lev:10.4f} [{c.domain[0]:8.4f},{c.domain[1]:9.4f}] "
              f"{pe:10.4f} {pc:10.4f} {pc - pe:10.4f}{flag}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "corpus": "blend-v1 (production default)",
        "corpus_dir": str(fe.corpus_dir),
        "frame": meta.get("surface_frame"),
        "weights_json": ARM_JSON,
        "weights_meta": meta,
        "curve_extremes": ext,
        "clamped_lower_bound": bound,
        "clamped_upper_bound": worst,
        "total_in_domain_signal": span,
        "board": rows,
        "spearman_ev_vs_ms": {"extrapolate": rho_ex, "clamp": rho_cl},
        "armA_per_gauge_repricing": per_gauge,
        "modelled_only": ("MODELLED ONLY: fitted-surface attribution, not measured typing "
                          "speed. No layout here is promoted or adopted."),
    }, open(OUT, "w"), indent=1)
    print(f"\nWROTE {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
