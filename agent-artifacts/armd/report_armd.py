"""Assemble the five mandated judgements for every arm's champion into one JSON + table.

Run AFTER the three arms complete. Reads `runs/arm-<arm>.json`, judges each champion
against the five incumbents and three reference layouts, and writes
`judgement.json` + a printable table.
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
import judge_armd as J  # noqa: E402
from armd_load import load_curves  # noqa: E402
from keybo.analysis.evidence_scorer import CLAMP, EXPECTED_SIGN  # noqa: E402

from keybo.analysis.evidence_scorer import LIVE_GAUGES  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

STATE = Path("/local/home/zegertho/agent/state/armd/artifacts")
OPTEV = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARM_JSON = ("/local/home/zegertho/agent/state/evidence-scorer/artifacts/"
            "arm-random400-native.json")
#: Arm D is judged SIDE BY SIDE with the three frozen arms, which are read from OPTEVIDENCE-1's
#: run directory rather than recomputed — they are frozen, and re-deriving them would risk
#: quoting a different number under the same name.
ARMS = ("domain", "evidence", "baseline", "constrained")
ARM_DIR = {"domain": STATE / "runs", "evidence": OPTEV / "runs",
           "baseline": OPTEV / "runs", "constrained": OPTEV / "runs"}
#: The near-optimal pool the PAIRED resolution must be computed over (trap 37 + "name the pool"):
#: the four champions plus the six incumbents. qwerty/graphite/semimak are deliberately EXCLUDED
#: — qwerty sits +9.5 ms/char away, which inflates the layout main effect and shrinks the floor.
NEAR_OPTIMAL_POOL_EXTRA = ()  # champions + incumbents are added programmatically
WRONG_SIGNED = ("scissor", "sfb", "sfb-dist", "lsb-dist", "sfs")


def main() -> int:
    out: dict = {"arms": {}, "modelled_only": (
        "MODELLED ONLY: every number is attribution of a FITTED timing surface, not a "
        "measurement of realized typing speed. tau saturated at 1.0 and Phase-D was "
        "cancelled. No layout here is promoted or adopted.")}

    # ---- champions ----
    champs: dict[str, str] = {}
    for arm in ARMS:
        path = ARM_DIR[arm] / f"arm-{arm}.json"
        if not path.exists():
            print(f"MISSING {path} — arm not finished", file=sys.stderr)
            continue
        blob = json.load(open(path))
        champs[f"champ-{arm}"] = blob["champion"]["layout"]
        out["arms"][arm] = {
            "champion": blob["champion"], "unique_evals": blob["unique_evals"],
            "islands": blob["islands"], "epochs_run": blob["epochs_run"],
            "seed": blob["seed"], "bounds": blob.get("bounds"),
            "objective_unit": blob["objective_unit"], "elapsed_s": blob["elapsed_s"],
            "domain_policy": blob.get("domain_policy", "extrapolate"),
            "budget_requested": blob["budget_requested"],
            "top10": blob["top50"][:10],
            "top50": blob["top50"],
        }
    assert champs, "no arm produced a champion"

    everyone = {**champs, **J.INCUMBENTS, **J.REFERENCE}
    names = list(everyone)
    print(f"judging {len(names)} layouts: {names}")

    corpus_dir = production_corpus_dir(None)
    out["corpus"] = str(corpus_dir)
    from keybo.data.corpus import corpus_identity
    out["corpus_identity"] = corpus_identity(corpus_dir)

    # ---- J1: trained-objective score + J2 ms/char + out-of-domain, via the fast path ----
    fe = EV.FastEval(corpus=None, weights_json=ARM_JSON, with_surface=True)
    ce = AD.ClampedEval(fe, policy=CLAMP)
    assert fe.weights_meta.get("surface_frame") == "native", (
        f"ARM D requires the .native frame; weights say "
        f"{fe.weights_meta.get('surface_frame')!r}"
    )
    perms = np.stack([EV.perm_of(everyone[n]) for n in names])
    g = fe.gauges(perms)
    # BOTH totals for every layout. Arm D is scored on the CLAMPED objective it optimized and
    # arms A/C on the EXTRAPOLATING one they optimized, so a single "evidence_score" column
    # would silently mix two different rulers — which is the confusion the parent's
    # `domain_policy` field exists to prevent.
    ev = fe.evidence_score(g)
    ev_clamped = ce.evidence_score(g)
    ood = fe.out_of_domain(g)
    curve_domain = {c.metric: list(c.domain) for c in fe.curves}
    curves = load_curves(ARM_JSON)

    out["per_layout"] = {}
    for i, n in enumerate(names):
        ood_set = sorted([k for k in LIVE_GAUGES if bool(ood[k][i])])
        out["per_layout"][n] = {
            "layout": everyone[n],
            "evidence_score": float(ev[i]),            # EXTRAPOLATE (arms A/C's ruler)
            "evidence_score_clamped": float(ev_clamped[i]),  # CLAMP (arm D's ruler)
            "ms_per_char": float(g["_ms_per_char"][i]),
            "total_ms": float(g["_total_ms"][i]),
            "gauges14": {k: float(g[k][i]) for k in LIVE_GAUGES},
            "out_of_domain": ood_set,
            "n_out_of_domain": len(ood_set),
            "out_of_domain_detail": {
                k: {"level": float(g[k][i]), "valid_domain": curve_domain[k],
                    "distance_outside": float(max(curve_domain[k][0] - g[k][i],
                                                  g[k][i] - curve_domain[k][1], 0.0))}
                for k in ood_set},
        }
    out["valid_domains"] = curve_domain

    # ---- J2 paired: per-seed ms/char + the paired resolution ----
    trigrams = load_frequencies(str(corpus_dir / "trigrams.txt"))
    print("computing paired resolution (loads 6 models, keeps seed tables) ...", flush=True)
    near_optimal = [n for n in names if n not in J.REFERENCE]
    out["paired"] = J.paired_resolution(everyone, trigrams, pool=near_optimal)
    # The full pool too, ONLY so the two figures OPTEVIDENCE-1 published are reconcilable in one
    # artifact rather than looking like a contradiction (trap 41: same name, different unit of
    # aggregation). The verdicts below use the near-optimal one.
    out["paired_full_pool"] = J.paired_resolution(everyone, trigrams, pool=list(names))

    # ---- J4: normalized six-surface floor (corpus-matched ceilings) ----
    print("deriving corpus-matched ceilings + normalized floors ...", flush=True)
    six = J.SixSurface(corpus_dir / "trigrams.txt")
    six_iweb = J.SixSurface(Path("/tmp/optev/data/corpus/trigrams.txt"))
    pc = max(abs(six_iweb.ceiling_map[s] - J.FROZEN_IWEB_CEILINGS[s]) for s in J.SIX)
    out["ceilings"] = {"corpus_derived": six.ceiling_map, "frozen_iweb": J.FROZEN_IWEB_CEILINGS,
                       "iweb_rederivation_worst_abs_diff": pc,
                       "positive_control": "PASS" if pc < 1e-9 else "FAIL"}
    assert pc < 1e-9, f"ceiling re-derivation failed the iWeb positive control: {pc:.3e}"
    for n in names:
        p = EV.perm_of(everyone[n])
        out["per_layout"][n]["normfloor"] = six.normfloor(p)
        out["per_layout"][n]["mean_saved_pct"] = six.mean_saved(p)
        out["per_layout"][n]["saved_per_surface"] = six.saved(p).tolist()

    # ---- J3: the 19-gauge frame via the shipped CLI ----
    print("running keybo analyze --json on all layouts ...", flush=True)
    an = J.analyze_json(everyone)
    out["analyze_frame"] = {"gauge_frame": an["gauge_frame"], "corpus": an["corpus"],
                            "skipgram_table": an["skipgram_table"],
                            "target_wpm": an["target_wpm"]}
    out["analyze_extra_rows"] = an["_extra_rows"]  # the --ref row, not one of ours
    spec_to_name = {everyone[n]: n for n in names}
    rows19: dict[str, dict] = {}
    for row in an["rows"].values():
        name = spec_to_name.get(row["layout"])
        if name is not None:  # skip analyze's own --ref row
            rows19[name] = row
    assert set(rows19) == set(names), f"missing rows for {sorted(set(names) - set(rows19))}"
    for n in names:
        out["per_layout"][n]["gauges19"] = {
            **rows19[n]["gauges"],
            "genkey": rows19[n]["community"]["genkey"],
            "oxeylyzer1": rows19[n]["community"]["oxeylyzer1"],
            "oxeylyzer2": rows19[n]["community"]["oxeylyzer2"],
            "wfd": rows19[n]["community"]["wfd"],
        }
        out["per_layout"][n]["community_primed"] = rows19[n]["community_primed"]
        out["per_layout"][n]["analyze_ms_per_char"] = rows19[n]["time"]["ms_per_char"]

    # ---- J5: dominance on the 10-axis frame ----
    print("building dominance axes ...", flush=True)
    dom_axes: dict[str, dict] = {}
    for n in names:
        p = EV.perm_of(everyone[n])
        r = rows19[n]
        dom_axes[n] = {
            "floor": six.normfloor(p), "mean": six.mean_saved(p),
            "wfd": r["community"]["wfd"],
            "genkey": r["community_primed"]["genkey_primed"],
            "oxey1": r["community_primed"]["oxey1_primed"],
            "oxey2": r["community_primed"]["oxey2_primed"],
            "lsb": r["kmstats"]["lsb"], "sfb": r["kmstats"]["sfb"], "sfs": r["kmstats"]["sfs"],
            "scissor": r["gauges"]["scissor"],
        }
    out["dominance_axes"] = dom_axes
    out["dominance_frame"] = {"axes": list(J.DOM_AXES), "sign": J.DOM_SIGN,
                              "predicate": "n_ge == 10 AND n_strict >= 1 (trap 33)"}
    out["dominance"] = {}
    for cn in champs:
        res = {}
        for inn in J.INCUMBENTS:
            is_dom, n_ge, n_gt = J.dominates(dom_axes[cn], dom_axes[inn])
            res[inn] = {"dominates": is_dom, "n_ge": n_ge, "n_strict": n_gt}
        out["dominance"][cn] = {
            "vs_incumbents": res,
            "dominator_exists": any(v["dominates"] for v in res.values()),
            "best_n_ge": max(v["n_ge"] for v in res.values()),
        }

    # ---- J4 continued: optimizing-the-ruler, win counts on the independent gauges ----
    #: LOWER-better direction for each of the 19 gauges. sfr is a PERMUTATION INVARIANT
    #: (trap 23) so it is excluded from win counts — it is a tie by construction.
    LOWER_BETTER = {
        "sfb": True, "sfs": True, "sfb-dist": True, "sfs-dist": True, "lsb": True,
        "lsb-dist": True, "redir": True, "scissor": True, "imbalance": True,
        "comfort": True, "oxey-style": True,
        "alt": False, "roll": False, "sr-roll": False,
        "genkey": True, "oxeylyzer1": False, "oxeylyzer2": False, "wfd": False,
    }
    out["gauge_direction"] = LOWER_BETTER
    out["invariant_excluded"] = ["sfr"]
    out["ruler_check"] = {}
    for cn in champs:
        per_inc = {}
        for inn in J.INCUMBENTS:
            wins = []
            losses = []
            for gauge, lower in LOWER_BETTER.items():
                a = out["per_layout"][cn]["gauges19"][gauge]
                b = out["per_layout"][inn]["gauges19"][gauge]
                if a == b:
                    continue
                better = (a < b) if lower else (a > b)
                (wins if better else losses).append(gauge)
            per_inc[inn] = {"n_gauges_scored": len(LOWER_BETTER), "wins": sorted(wins),
                            "losses": sorted(losses), "n_win": len(wins), "n_loss": len(losses)}
        out["ruler_check"][cn] = {
            "per_incumbent": per_inc,
            "normfloor": out["per_layout"][cn]["normfloor"],
            "normfloor_negative": out["per_layout"][cn]["normfloor"] < 0,
            "note": ("effective dof over the 19 gauges is ~4-5, so a raw win count "
                     "over-counts independent evidence ~4x (trap 39); sfr excluded as a "
                     "permutation invariant (trap 23)"),
        }


    # ---- ARM D ①: CLAMP BINDING — the abort condition, measured on the champion ----------
    # This is P6 and it is a GATE, not a finding: if pushing a gauge further out of its domain
    # still buys reward, the clamp is not wired into the objective and every arm D number is
    # void. Measured on arm D's own champion, through the same `ClampedEval` the search used.
    print("checking the clamp BINDS on arm D's champion ...", flush=True)
    binding = {}
    if "champ-domain" in champs:
        d_perm = EV.perm_of(champs["champ-domain"])[None]
        g_d = fe.gauges(d_perm)
        base_clamped = float(ce.evidence_score(g_d)[0])
        worst_gain = 0.0
        for name, curve in curves.items():
            lo, hi = curve.domain
            width = hi - lo
            probes = {}
            for tag, level in (("ceiling", hi), ("ceiling+1w", hi + width),
                               ("ceiling+50w", hi + 50 * width), ("floor", lo),
                               ("floor-1w", lo - width), ("floor-50w", lo - 50 * width)):
                g2 = {k: v.copy() for k, v in g_d.items()}
                g2[name] = np.array([float(level)])
                probes[tag] = float(ce.evidence_score(g2)[0])
            # "buys nothing" = going 50 widths PAST an edge scores EXACTLY the same as AT it.
            gain_hi = probes["ceiling"] - probes["ceiling+50w"]   # >0 would mean it still pays
            gain_lo = probes["floor"] - probes["floor-50w"]
            worst_gain = max(worst_gain, abs(gain_hi), abs(gain_lo))
            binding[name] = {
                "probes": probes,
                "reward_for_50_widths_past_ceiling": gain_hi,
                "reward_for_50_widths_below_floor": gain_lo,
                "binds": (probes["ceiling"] == probes["ceiling+50w"]
                          and probes["floor"] == probes["floor-50w"]),
            }
        out["clamp_binding"] = {
            "champion": champs["champ-domain"],
            "base_clamped_score": base_clamped,
            "per_gauge": binding,
            "all_bind": all(v["binds"] for v in binding.values()),
            "worst_abs_reward_outside": worst_gain,
            "note": ("P6 / abort condition: under CLAMP, pushing any gauge 50 domain-widths "
                     "past either edge must change the total by EXACTLY 0.0. If this fails the "
                     "clamp is not in the objective and arm D is void, not a result."),
        }
        assert out["clamp_binding"]["all_bind"], (
            f"THE CLAMP DOES NOT BIND (worst {worst_gain:.3e}) — arm D is VOID, not a result"
        )
        print(f"  clamp binds on all {len(binding)} gauges "
              f"(worst |reward outside| = {worst_gain:.3e})")

    # ---- ARM D ②: does the champion still LEAVE the domain, and is that a bug? -----------
    # Under CLAMP the search has no incentive to leave the domain, so a still-far-outside
    # champion would suggest broken wiring — EXCEPT that a curve minimized AT an edge puts the
    # optimum ON the boundary, and a boundary point is `in_domain` while one step beyond is not.
    # So the count alone cannot distinguish "clamp not binding" from "optimum is on the
    # boundary". `clamp_binding` above is what separates them; this records the count and, for
    # each out-of-domain gauge, whether that gauge's own clamped optimum is unreachable.
    if "champ-domain" in champs:
        d = out["per_layout"]["champ-domain"]
        detail = {}
        for name in d["out_of_domain"]:
            curve = curves[name]
            lo, hi = curve.domain
            level = d["gauges14"][name]
            xs = np.linspace(lo, hi, 20001)
            ys = np.array([curve.price(float(x)) for x in xs])
            argmin_level = float(xs[int(np.argmin(ys))])
            detail[name] = {
                "level": level, "valid_domain": [lo, hi],
                "outside_by": float(max(lo - level, level - hi, 0.0)),
                "clamped_to": float(min(max(level, lo), hi)),
                "curve_argmin_in_domain": argmin_level,
                "argmin_is_an_edge": bool(abs(argmin_level - lo) < (hi - lo) * 1e-6
                                          or abs(argmin_level - hi) < (hi - lo) * 1e-6),
                "clamped_to_its_own_optimum": bool(
                    abs(min(max(level, lo), hi) - argmin_level) < (hi - lo) * 1e-6),
                "expected_sign": EXPECTED_SIGN[name],
            }
        out["armd_out_of_domain"] = {
            "n_out_of_domain": d["n_out_of_domain"],
            "n_gauges": len(curves),
            "armA_n_out_of_domain": out["per_layout"]["champ-evidence"]["n_out_of_domain"]
            if "champ-evidence" in out["per_layout"] else None,
            "per_gauge": detail,
            "n_clamped_to_own_optimum": sum(
                1 for v in detail.values() if v["clamped_to_its_own_optimum"]),
            "interpretation": (
                "An out-of-domain count > 0 under CLAMP is NOT evidence of broken wiring: 8 of "
                "the 14 curves are minimized AT a domain edge, so the clamped optimum sits ON "
                "the boundary by construction and any layout at that optimum reads as outside. "
                "`clamp_binding.all_bind` is the wiring check; this is the description."),
        }

    # ---- ARM D ③: pairwise gap resolution against the PAIRED floor -----------------------
    floor = out["paired"]["paired_floor_max"]  # the conservative threshold (0.2222 for n=8)
    med = out["paired"]["paired_floor_median"]
    seed_means = out["paired"]["seed_mean_ms_per_char"]
    pairs = {}
    order = [n for n in champs] + list(J.INCUMBENTS)
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            gap = seed_means[a] - seed_means[b]
            pairs[f"{a} vs {b}"] = {
                "d_ms_per_char": gap, "abs": abs(gap),
                "resolves_at_max_floor": abs(gap) > floor,
                "resolves_at_median_floor": abs(gap) > med,
                "multiples_of_max_floor": abs(gap) / floor if floor else None,
                "faster": a if gap < 0 else b,
            }
    out["pairwise"] = {
        "paired_floor_max_used": floor, "paired_floor_median": med,
        "pool": out["paired"]["pool"], "n_pool": out["paired"]["n_pool"],
        "pairs": pairs,
        "n_resolving_at_max": sum(1 for v in pairs.values() if v["resolves_at_max_floor"]),
        "n_pairs": len(pairs),
    }

    # ---- ARM D ④: deficit recovery vs arm A, and the degeneracy signature ---------------
    if {"champ-domain", "champ-evidence", "champ-baseline"} <= set(out["per_layout"]):
        a_ms = out["per_layout"]["champ-evidence"]["ms_per_char"]
        b_ms = out["per_layout"]["champ-baseline"]["ms_per_char"]
        c_ms = out["per_layout"].get("champ-constrained", {}).get("ms_per_char")
        d_ms = out["per_layout"]["champ-domain"]["ms_per_char"]
        deficit = a_ms - b_ms
        out["recovery"] = {
            "armA_deficit_vs_armB": deficit,
            "armD_ms_per_char": d_ms, "armA_ms_per_char": a_ms, "armB_ms_per_char": b_ms,
            "armC_ms_per_char": c_ms,
            "armD_recovered_pct": 100.0 * (a_ms - d_ms) / deficit,
            "armC_recovered_pct": (100.0 * (a_ms - c_ms) / deficit) if c_ms else None,
            "armD_residual_vs_armB": d_ms - b_ms,
            "armD_residual_in_floor_multiples": (d_ms - b_ms) / floor if floor else None,
            "note": ("recovery = how much of arm A's ms/char deficit vs arm B the clamp bought "
                     "back. Arm C (bounding the 5 wrong-signed gauges) recovered 28%."),
        }

    # degeneracy: distinct layouts with EQUAL clamped fitness in arm D's own archive
    if "domain" in out["arms"]:
        top = out["arms"]["domain"]["top50"]
        buckets: dict[str, list[str]] = {}
        for row in top:
            buckets.setdefault(f"{row['fitness']:.12g}", []).append(row["layout"])
        plateaus = {k: v for k, v in buckets.items() if len(v) > 1}
        out["degeneracy"] = {
            "n_top50": len(top),
            "n_distinct_fitness": len(buckets),
            "plateaus": plateaus,
            "largest_plateau": max((len(v) for v in plateaus.values()), default=1),
            "fitness_spread_top50": (top[-1]["fitness"] - top[0]["fitness"]) if top else None,
            "note": ("distinct layouts sharing a clamped fitness to 12 significant figures. A "
                     "large plateau is the (iii) signature: the clamped objective cannot "
                     "distinguish them, so the search has nothing to climb."),
        }


    # ---- ARM D ⑤: PLATEAU CENSUS over the FINAL POPULATION (sibling warning 2 / P14) ------
    # The top50 archive (④) is a truncated view. The sibling's point is sharper and needs the
    # WHOLE final population: clamping removes the gradient outside every domain, so the
    # objective is flat out there and the search can sit on tie plateaus. A champion drawn
    # arbitrarily from a plateau is NOT an optimum. The checkpoint holds all 40 islands x 64.
    ckpt_path = ARM_DIR["domain"] / "arm-domain.ckpt.json"
    if ckpt_path.exists():
        ck = json.load(open(ckpt_path))
        all_fit, all_lay = [], []
        for isl in ck["islands"]:
            for p30, f in zip(isl["pop"], isl["fit"], strict=False):
                all_lay.append(EV.layout_of(np.concatenate(
                    [np.asarray(p30, dtype=np.int32), np.array([30], dtype=np.int32)])))
                all_fit.append(float(f))
        uniq_lay = set(all_lay)
        # Bucket at 12 significant figures: a tie at that precision is an exact tie in practice
        # and cannot be a rounding artifact of the objective's own arithmetic.
        buckets: dict[str, set[str]] = {}
        for lay, f in zip(all_lay, all_fit, strict=True):
            buckets.setdefault(f"{f:.12g}", set()).add(lay)
        multi = {k: sorted(v) for k, v in buckets.items() if len(v) > 1}
        champ = ck["best_layout"]
        champ_key = None
        for k, v in buckets.items():
            if champ in v:
                champ_key = k
                break
        champ_tied_with = sorted(set(buckets.get(champ_key, set())) - {champ}) if champ_key else []
        out["plateau_census"] = {
            "source": str(ckpt_path),
            "epoch": ck["epoch"],
            "n_population_slots": len(all_lay),
            "n_distinct_layouts": len(uniq_lay),
            "n_distinct_objective_values": len(buckets),
            "distinct_values_per_distinct_layout": len(buckets) / max(len(uniq_lay), 1),
            "n_plateaus": len(multi),
            "largest_plateau": max((len(v) for v in multi.values()), default=1),
            "n_layouts_on_some_plateau": sum(len(v) for v in multi.values()),
            "champion": champ,
            "champion_is_tied": bool(champ_tied_with),
            "champion_n_exact_ties": len(champ_tied_with),
            "champion_tied_with": champ_tied_with[:20],
            "note": ("distinct objective values vs distinct LAYOUTS in the final population. "
                     "A ratio well below 1 means the clamped objective cannot tell those "
                     "layouts apart, so the champion is a plateau member, not an optimum. This "
                     "is outcome (iii) evidence — the curves carry little in-domain signal — "
                     "NOT a failed run."),
        }
        # Do the tied layouts differ in SPEED? That is what makes a plateau consequential.
        if champ_tied_with:
            tied = [champ] + champ_tied_with[:20]
            tp = np.stack([EV.perm_of(x) for x in tied])
            tg = fe.gauges(tp)
            tms = tg["_ms_per_char"]
            out["plateau_census"]["champion_plateau_ms_per_char"] = {
                x: float(tms[i]) for i, x in enumerate(tied)}
            out["plateau_census"]["champion_plateau_ms_spread"] = float(tms.max() - tms.min())
            out["plateau_census"]["champion_plateau_note"] = (
                "these layouts are INDISTINGUISHABLE to the clamped objective (identical to 12 "
                "sig figs) yet differ in predicted ms/char by the spread shown — so which one "
                "the search returns as 'champion' is arbitrary with respect to speed")

    # ---- ARM D ⑥: IN-BAND RANK TEST, tightening (sibling warning 1 / P13) -----------------
    # The sibling measured rho decaying monotonically to NEGATIVE as the band tightens, on
    # 36,005 incumbent perturbations. I cannot re-use their pool (different workspace), but I can
    # reproduce the SHAPE on my own pool: every layout I have judged, plus arm D's whole final
    # population, sliced by ms/char ceiling. A pool selected by NEITHER objective is the point,
    # so the perturbation pool is built by random swaps off the incumbents — not by either search.
    from scipy.stats import spearmanr
    print("building an independent perturbation pool for the in-band rank test ...", flush=True)
    rng = np.random.default_rng(20260728)
    pool_perms = []
    for lay in J.INCUMBENTS.values():
        base = EV.perm_of(lay)
        for _ in range(600):
            q = base.copy()
            for _ in range(int(rng.integers(1, 5))):
                i, j = rng.choice(30, 2, replace=False)
                q[i], q[j] = q[j], q[i]
            pool_perms.append(q)
    PP = np.stack(pool_perms)
    pg = fe.gauges(PP)
    p_ms = pg["_ms_per_char"]
    p_raw = fe.evidence_score(pg)
    p_cl = ce.evidence_score(pg)
    bands = {}
    for ceiling in (None, 257.0, 256.0, 255.5, 255.0):
        m = np.ones(len(p_ms), dtype=bool) if ceiling is None else (p_ms <= ceiling)
        if m.sum() < 30:
            continue
        bands["all" if ceiling is None else f"<={ceiling}"] = {
            "n": int(m.sum()),
            "rho_raw": float(spearmanr(p_raw[m], p_ms[m]).statistic),
            "rho_clamped": float(spearmanr(p_cl[m], p_ms[m]).statistic),
        }
    out["in_band_rank_test"] = {
        "pool": ("random 1-4 swap perturbations of the six incumbents, 600 each — selected by "
                 "NEITHER objective"),
        "n_pool": int(len(p_ms)),
        "instrument_positive_control_rho": float(spearmanr(p_ms, p_ms).statistic),
        "bands": bands,
        "note": ("rho between an evidence objective and predicted ms/char, as the band tightens. "
                 "Independent reproduction of the sibling's banded-rank finding on my own pool: "
                 "if the CLAMPED rho decays toward/below 0 in the near-optimal band, a good "
                 "clamped score is NOT evidence of a fast layout, and arm D's champion must be "
                 "read on ms/char alone."),
    }

    path = STATE / "judgement.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nWROTE {path}")

    # ---- printable table ----
    print(f"\n{'=' * 126}")
    print(f"ARM D board — corpus blend-v1 (production default), .native frame, 90 WPM. "
          f"MODELLED ONLY.")
    hdr = (f"{'layout':<16}{'ev CLAMP':>10}{'ev EXTRAP':>11}{'ms/char':>10}"
           f"{'vs best inc':>13}{'normfloor':>11}{'mean_sav':>9}{'n_ood':>6}{'dom?':>6}")
    print(hdr)
    print("-" * 126)
    best_inc_ms = min(out["per_layout"][n]["ms_per_char"] for n in J.INCUMBENTS)
    best_inc = min(J.INCUMBENTS, key=lambda n: out["per_layout"][n]["ms_per_char"])
    order = list(champs) + list(J.INCUMBENTS) + list(J.REFERENCE)
    for n in order:
        r = out["per_layout"][n]
        d = r["ms_per_char"] - best_inc_ms
        dom = ""
        if n in out.get("dominance", {}):
            dom = "YES" if out["dominance"][n]["dominator_exists"] else "no"
        print(f"{n:<16}{r['evidence_score_clamped']:>10.4f}{r['evidence_score']:>11.4f}"
              f"{r['ms_per_char']:>10.4f}{d:>+13.4f}"
              f"{r['normfloor']:>11.6f}{r['mean_saved_pct']:>9.4f}"
              f"{r['n_out_of_domain']:>6}{dom:>6}")
    print("-" * 126)
    print("⚠ each arm optimized ONE of these two evidence columns: arm D the CLAMP column, arms "
          "A/C the EXTRAP one.")
    print(f"best incumbent on ms/char: {best_inc} at {best_inc_ms:.4f}")
    p_ = out["paired"]
    print(f"\nPAIRED resolution over the NEAR-OPTIMAL pool (n={p_['n_pool']}: "
          f"{', '.join(p_['pool'])}):")
    print(f"  median {p_['paired_floor_median']:.4f} | max (conservative) "
          f"{p_['paired_floor_max']:.4f} | p95 {p_['paired_floor_p95']:.4f} ms/char")
    print(f"  UNPAIRED floor {p_['unpaired_floor_ms_per_char']:.4f} is the WRONG ruler (trap 37)")
    print(f"  SS shares: layout {p_['ss_share_pct']['layout']:.2f}% | "
          f"seed {p_['ss_share_pct']['seed']:.2f}% | "
          f"residual {p_['ss_share_pct']['residual']:.2f}%")
    pf = out["paired_full_pool"]
    print(f"  (full pool n={pf['n_pool']} incl. qwerty/graphite/semimak: median "
          f"{pf['paired_floor_median']:.4f}, seed {pf['ss_share_pct']['seed']:.2f}% of SS — "
          f"a DIFFERENT pool, hence a different floor)")

    if "recovery" in out:
        r = out["recovery"]
        print(f"\nDEFICIT RECOVERY (arm A's {r['armA_deficit_vs_armB']:+.4f} ms/char deficit "
              f"vs arm B):")
        print(f"  arm C recovered {r['armC_recovered_pct']:.1f}%  |  "
              f"arm D recovered {r['armD_recovered_pct']:.1f}%")
        print(f"  arm D still behind arm B by {r['armD_residual_vs_armB']:+.4f} ms/char "
              f"= {r['armD_residual_in_floor_multiples']:.2f}x the conservative paired floor")
    if "clamp_binding" in out:
        cb = out["clamp_binding"]
        print(f"\nCLAMP BINDING (P6): all_bind={cb['all_bind']}, worst |reward outside| = "
              f"{cb['worst_abs_reward_outside']:.3e}")
    if "armd_out_of_domain" in out:
        o = out["armd_out_of_domain"]
        print(f"OUT-OF-DOMAIN: arm D {o['n_out_of_domain']}/{o['n_gauges']} "
              f"(arm A was {o['armA_n_out_of_domain']}/{o['n_gauges']}); "
              f"{o['n_clamped_to_own_optimum']} of those clamp to that gauge's OWN optimum")
    if "degeneracy" in out:
        dg = out["degeneracy"]
        print(f"DEGENERACY in arm D's top50: {dg['n_distinct_fitness']} distinct fitness values, "
              f"largest plateau {dg['largest_plateau']} layouts, "
              f"spread {dg['fitness_spread_top50']:.6f}")
    pw = out["pairwise"]
    print(f"\nPAIRWISE: {pw['n_resolving_at_max']} of {pw['n_pairs']} champion/incumbent pairs "
          f"resolve at the conservative floor {pw['paired_floor_max_used']:.4f}")
    if "plateau_census" in out:
        pc = out["plateau_census"]
        print(f"\nPLATEAU CENSUS (final population, epoch {pc['epoch']}):")
        print(f"  {pc['n_population_slots']} slots -> {pc['n_distinct_layouts']} distinct layouts "
              f"-> {pc['n_distinct_objective_values']} distinct objective values "
              f"({pc['distinct_values_per_distinct_layout']:.4f} per layout)")
        print(f"  {pc['n_plateaus']} plateaus, largest {pc['largest_plateau']} layouts; "
              f"champion tied with {pc['champion_n_exact_ties']} others")
        if "champion_plateau_ms_spread" in pc:
            print(f"  the champion's OWN plateau spans {pc['champion_plateau_ms_spread']:.4f} "
                  f"ms/char while being IDENTICAL to the clamped objective")
    if "in_band_rank_test" in out:
        ib = out["in_band_rank_test"]
        print(f"\nIN-BAND RANK TEST (n={ib['n_pool']} incumbent perturbations, chosen by NEITHER "
              f"objective; instrument control rho={ib['instrument_positive_control_rho']:.4f}):")
        print(f"  {'band':<10}{'n':>8}{'rho_raw':>11}{'rho_CLAMPED':>13}")
        for k, v in ib["bands"].items():
            print(f"  {k:<10}{v['n']:>8}{v['rho_raw']:>+11.4f}{v['rho_clamped']:>+13.4f}")
        print("  (want +1 if the weights inform about speed; <=0 means a good clamped score is")
        print("   NOT evidence of a fast layout, so the champion must be read on ms/char alone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
