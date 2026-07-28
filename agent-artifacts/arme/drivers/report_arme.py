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
sys.path.append("/local/home/zegertho/agent/state/armd/artifacts/drivers")
import armd_obj as AD  # noqa: E402
import arme_obj as AE  # noqa: E402
import evobj as EV  # noqa: E402
import judge_arme as J  # noqa: E402
from arme_load import load_curves  # noqa: E402
from keybo.analysis.evidence_scorer import CLAMP, EXPECTED_SIGN  # noqa: E402

from keybo.analysis.evidence_scorer import LIVE_GAUGES  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

STATE = Path("/local/home/zegertho/agent/state/arme/artifacts")
ARMD = Path("/local/home/zegertho/agent/state/armd/artifacts")
OPTEV = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARM_JSON = ("/local/home/zegertho/agent/state/evidence-scorer/artifacts/"
            "arm-archive400-native.json")
RANDOM_JSON = ("/local/home/zegertho/agent/state/evidence-scorer/artifacts/"
               "arm-random400-native.json")
#: Arm D is judged SIDE BY SIDE with the three frozen arms, which are read from OPTEVIDENCE-1's
#: run directory rather than recomputed — they are frozen, and re-deriving them would risk
#: quoting a different number under the same name.
ARMS = ("archive", "domain", "evidence", "baseline", "constrained")
ARM_DIR = {"archive": STATE / "runs", "domain": ARMD / "runs", "evidence": OPTEV / "runs",
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
    # ⚠ THREE RULERS, NEVER MIXED. Arm E optimized the CLAMPED ARCHIVE total; arm D optimized
    # the CLAMPED RANDOM400 total; arms A/C optimized the EXTRAPOLATING RANDOM400 total. Quoting
    # one column across arms is exactly the error the `domain_policy` field exists to prevent, and
    # here there is a second axis (which POOL the curves came from) on top of the policy axis. So
    # every layout gets all four columns and each arm's verdict is read on its OWN.
    curves = load_curves(ARM_JSON)                    # ARCHIVE curves — arm E's ruler
    fe = EV.FastEval(corpus=None, weights_json=ARM_JSON, with_surface=True)
    ae = AE.ValidatedClampedEval(fe, curves, policy=CLAMP)     # arm E's own objective
    fe_rand = EV.FastEval(corpus=None, weights_json=RANDOM_JSON, with_surface=True)
    ce_rand = AD.ClampedEval(fe_rand, policy=CLAMP)            # arm D's ruler
    for label, meta in (("archive", fe.weights_meta), ("random400", fe_rand.weights_meta)):
        assert meta.get("surface_frame") == "native", (
            f"ARM E requires the .native frame; {label} weights say "
            f"{meta.get('surface_frame')!r}")
    assert fe.weights_meta.get("pool") == "archive-400", fe.weights_meta.get("pool")
    assert fe_rand.weights_meta.get("pool") == "random-c30m-400", fe_rand.weights_meta.get("pool")
    out["rulers"] = {
        "armE": {"weights": ARM_JSON, "pool": "archive-400", "policy": "clamp",
                 "column": "ev_archive_clamp"},
        "armD": {"weights": RANDOM_JSON, "pool": "random-c30m-400", "policy": "clamp",
                 "column": "evidence_score_clamped"},
        "armA_armC": {"weights": RANDOM_JSON, "pool": "random-c30m-400",
                      "policy": "extrapolate", "column": "evidence_score"},
        "warning": ("these are FOUR DIFFERENT rulers on two axes (fit pool x domain policy). "
                    "Each arm is judged on the one it optimized; a cross-arm comparison of any "
                    "single evidence column is meaningless."),
    }
    perms = np.stack([EV.perm_of(everyone[n]) for n in names])
    g = fe.gauges(perms)
    # BOTH totals for every layout. Arm D is scored on the CLAMPED objective it optimized and
    # arms A/C on the EXTRAPOLATING one they optimized, so a single "evidence_score" column
    # would silently mix two different rulers — which is the confusion the parent's
    # `domain_policy` field exists to prevent.
    g_rand = fe_rand.gauges(perms)
    # sanity: the two evaluators differ ONLY in curves, so the gauge values must be bit-identical
    assert all(np.array_equal(g[m], g_rand[m]) for m in LIVE_GAUGES), (
        "the archive and random400 evaluators disagree on a GAUGE value — they must differ only "
        "in the curves")
    ev_archive_clamp = ae.evidence_score(g)                    # arm E's ruler
    ev_archive_extrap = ae.evidence_score_extrapolating(g)
    ev = fe_rand.evidence_score(g_rand)                        # arms A/C's ruler
    ev_clamped = ce_rand.evidence_score(g_rand)                # arm D's ruler
    ood = fe.out_of_domain(g)                                  # ARCHIVE domains (arm E's)
    ood_rand = fe_rand.out_of_domain(g_rand)                   # random400 domains
    curve_domain = {c.metric: list(c.domain) for c in fe.curves}
    curve_domain_rand = {c.metric: list(c.domain) for c in fe_rand.curves}

    out["per_layout"] = {}
    for i, n in enumerate(names):
        ood_set = sorted([k for k in LIVE_GAUGES if bool(ood[k][i])])
        ood_set_rand = sorted([k for k in LIVE_GAUGES if bool(ood_rand[k][i])])
        out["per_layout"][n] = {
            "layout": everyone[n],
            "ev_archive_clamp": float(ev_archive_clamp[i]),      # ARM E's ruler
            "ev_archive_extrapolate": float(ev_archive_extrap[i]),
            "evidence_score": float(ev[i]),            # EXTRAPOLATE (arms A/C's ruler)
            "evidence_score_clamped": float(ev_clamped[i]),  # CLAMP (arm D's ruler)
            "out_of_domain_random400": ood_set_rand,
            "n_out_of_domain_random400": len(ood_set_rand),
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
    out["valid_domains"] = curve_domain           # ARCHIVE (arm E)
    out["valid_domains_random400"] = curve_domain_rand

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
    # The iWeb positive control reads iWeb from MY OWN worktree, not a sibling's (trap 35: a
    # path into another agent's tree is not isolation). Identity asserted by md5, not by
    # filename — `data/corpus/trigrams.txt` IS iWeb (production), md5
    # 50cab38b6859b6e6520ba5d6ec6553b1, which is the traps file's reference value.
    iweb_path = Path("/tmp/arme/data/corpus/trigrams.txt")
    import hashlib
    iweb_md5 = hashlib.md5(iweb_path.read_bytes()).hexdigest()
    assert iweb_md5 == "50cab38b6859b6e6520ba5d6ec6553b1", (
        f"the iWeb positive control needs iWeb; {iweb_path} has md5 {iweb_md5}")
    six_iweb = J.SixSurface(iweb_path)
    pc = max(abs(six_iweb.ceiling_map[s] - J.FROZEN_IWEB_CEILINGS[s]) for s in J.SIX)
    out["ceilings"] = {"corpus_derived": six.ceiling_map, "frozen_iweb": J.FROZEN_IWEB_CEILINGS,
                       "iweb_source": str(iweb_path), "iweb_md5": iweb_md5,
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
    print("checking the clamp BINDS on arm E's champion ...", flush=True)
    binding = {}
    if "champ-archive" in champs:
        d_perm = EV.perm_of(champs["champ-archive"])[None]
        g_d = fe.gauges(d_perm)
        base_clamped = float(ae.evidence_score(g_d)[0])
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
                probes[tag] = float(ae.evidence_score(g2)[0])
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
            "champion": champs["champ-archive"],
            "base_clamped_score": base_clamped,
            "per_gauge": binding,
            "all_bind": all(v["binds"] for v in binding.values()),
            "worst_abs_reward_outside": worst_gain,
            "note": ("P6 / abort condition: under CLAMP, pushing any gauge 50 domain-widths "
                     "past either edge must change the total by EXACTLY 0.0. If this fails the "
                     "clamp is not in the objective and arm E is void, not a result. Measured "
                     "through `ValidatedClampedEval`, the SAME object the search used."),
        }
        assert out["clamp_binding"]["all_bind"], (
            f"THE CLAMP DOES NOT BIND (worst {worst_gain:.3e}) — arm E is VOID, not a result"
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
    if "champ-archive" in champs:
        d = out["per_layout"]["champ-archive"]
        detail = {}
        for name in d["out_of_domain"]:
            curve = curves[name]
            lo, hi = curve.domain
            level = d["gauges14"][name]
            xs = np.linspace(lo, hi, 20001)
            ys = curve.price_many(xs, policy=CLAMP)
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
        n_edge_min = sum(1 for name in curves
                         if (lambda c: (lambda xs, ys: bool(
                             abs(float(xs[int(np.argmin(ys))]) - c.domain[0])
                             < (c.domain[1] - c.domain[0]) * 1e-6
                             or abs(float(xs[int(np.argmin(ys))]) - c.domain[1])
                             < (c.domain[1] - c.domain[0]) * 1e-6))(
                                 np.linspace(c.domain[0], c.domain[1], 20001),
                                 c.price_many(np.linspace(c.domain[0], c.domain[1], 20001),
                                              policy=CLAMP)))(curves[name]))
        out["arme_out_of_domain"] = {
            "n_out_of_domain": d["n_out_of_domain"],
            "n_gauges": len(curves),
            "armA_n_out_of_domain": out["per_layout"]["champ-evidence"]["n_out_of_domain"]
            if "champ-evidence" in out["per_layout"] else None,
            "armD_n_out_of_domain_on_archive_domains": (
                out["per_layout"]["champ-domain"]["n_out_of_domain"]
                if "champ-domain" in out["per_layout"] else None),
            "n_archive_curves_minimized_at_an_edge": n_edge_min,
            "per_gauge": detail,
            "n_clamped_to_own_optimum": sum(
                1 for v in detail.values() if v["clamped_to_its_own_optimum"]),
            "interpretation": (
                f"An out-of-domain count > 0 under CLAMP is NOT evidence of broken wiring: "
                f"{n_edge_min} of the {len(curves)} ARCHIVE curves are minimized AT a domain "
                f"edge, so the clamped optimum sits ON the boundary by construction and a layout "
                f"at that optimum reads as outside. `clamp_binding.all_bind` is the wiring check; "
                f"this is the description. ⚠ The count is measured against the ARCHIVE domains — "
                f"the same layout has a DIFFERENT count against random400's domains, which is "
                f"the whole premise of arm E, so the two are reported separately."),
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
    if {"champ-archive", "champ-evidence", "champ-baseline"} <= set(out["per_layout"]):
        a_ms = out["per_layout"]["champ-evidence"]["ms_per_char"]
        b_ms = out["per_layout"]["champ-baseline"]["ms_per_char"]
        c_ms = out["per_layout"].get("champ-constrained", {}).get("ms_per_char")
        d_ms = out["per_layout"].get("champ-domain", {}).get("ms_per_char")
        e_ms = out["per_layout"]["champ-archive"]["ms_per_char"]
        best_inc_name = min(J.INCUMBENTS, key=lambda n: out["per_layout"][n]["ms_per_char"])
        inc_ms = out["per_layout"][best_inc_name]["ms_per_char"]
        deficit = a_ms - b_ms
        out["recovery"] = {
            "armA_deficit_vs_armB": deficit,
            "armE_ms_per_char": e_ms, "armD_ms_per_char": d_ms, "armA_ms_per_char": a_ms,
            "armB_ms_per_char": b_ms, "armC_ms_per_char": c_ms,
            "best_incumbent": best_inc_name, "best_incumbent_ms_per_char": inc_ms,
            "armE_recovered_pct": 100.0 * (a_ms - e_ms) / deficit,
            "armD_recovered_pct": (100.0 * (a_ms - d_ms) / deficit) if d_ms else None,
            "armC_recovered_pct": (100.0 * (a_ms - c_ms) / deficit) if c_ms else None,
            "armE_residual_vs_armB": e_ms - b_ms,
            "armE_residual_vs_best_incumbent": e_ms - inc_ms,
            "armE_residual_in_floor_multiples": (e_ms - b_ms) / floor if floor else None,
            "armE_vs_armD": (e_ms - d_ms) if d_ms else None,
            "armE_vs_qwerty30m": e_ms - out["per_layout"]["qwerty30m"]["ms_per_char"],
            "note": ("recovery = how much of arm A's ms/char deficit vs arm B a change bought "
                     "back. Arm C (bounding the 5 wrong-signed gauges) recovered 28%; arm D "
                     "recovered -421.9% (it multiplied the deficit 5.2x). Arm E changes the fit "
                     "POOL rather than the policy, so it tests whether arm D's result was a "
                     "property of the curves or of that fit."),
            "outcome_bands": {
                "E1_inside_incumbent_band": e_ms <= 254.63,
                "E2_between_incumbents_and_armA": 254.63 < e_ms < 256.9,
                "E3_at_or_worse_than_armA": e_ms >= 256.9,
                "worse_than_qwerty30m": e_ms > out["per_layout"]["qwerty30m"]["ms_per_char"],
                "worse_than_qwerty_classic_263_7141": e_ms > 263.7141,
            },
        }

    # degeneracy: distinct layouts with EQUAL clamped fitness in arm D's own archive
    if "archive" in out["arms"]:
        top = out["arms"]["archive"]["top50"]
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
    ckpt_path = ARM_DIR["archive"] / "arm-archive.ckpt.json"
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
    p_arch_cl = ae.evidence_score(pg)                       # ARM E's ruler
    p_arch_ex = ae.evidence_score_extrapolating(pg)
    pg_rand = fe_rand.gauges(PP)
    p_raw = fe_rand.evidence_score(pg_rand)                 # arms A/C's ruler
    p_cl = ce_rand.evidence_score(pg_rand)                  # arm D's ruler
    bands = {}
    for ceiling in (None, 257.0, 256.0, 255.5, 255.0):
        m = np.ones(len(p_ms), dtype=bool) if ceiling is None else (p_ms <= ceiling)
        if m.sum() < 30:
            continue
        bands["all" if ceiling is None else f"<={ceiling}"] = {
            "n": int(m.sum()),
            "rho_archive_clamped": float(spearmanr(p_arch_cl[m], p_ms[m]).statistic),
            "rho_archive_extrapolate": float(spearmanr(p_arch_ex[m], p_ms[m]).statistic),
            "rho_raw": float(spearmanr(p_raw[m], p_ms[m]).statistic),
            "rho_clamped": float(spearmanr(p_cl[m], p_ms[m]).statistic),
        }
    out["in_band_rank_test"] = {
        "pool": ("random 1-4 swap perturbations of the six incumbents, 600 each — selected by "
                 "NEITHER objective"),
        "n_pool": int(len(p_ms)),
        "instrument_positive_control_rho": float(spearmanr(p_ms, p_ms).statistic),
        "bands": bands,
        "columns": {"rho_archive_clamped": "ARM E's ruler (archive curves, CLAMP)",
                    "rho_archive_extrapolate": "archive curves, unbounded",
                    "rho_clamped": "arm D's ruler (random400, CLAMP)",
                    "rho_raw": "arms A/C's ruler (random400, unbounded)"},
        "note": ("rho between an evidence objective and predicted ms/char, as the band tightens. "
                 "The ARCHIVE column is the one arm E rests on: if it decays toward/below 0 in "
                 "the near-optimal band, a good archive score is NOT evidence of a fast layout, "
                 "and arm E's champion must be read on ms/char alone. Reported next to arm D's "
                 "column so 'does the better-covered fit rank the band better?' is answerable "
                 "on ONE pool — which is the question arm E exists to settle."),
    }
    # ⚠ The pool is 1-4-swap perturbations of the incumbents, chosen by NEITHER objective. But a
    # perturbation pool is not the same object as a SEARCH's final population, so also report the
    # rank correlation over arm E's own final population — the band the search actually ended in.
    if ckpt_path.exists():
        ck2 = json.load(open(ckpt_path))
        fin = []
        for isl in ck2["islands"]:
            for p30 in isl["pop"]:
                fin.append(np.concatenate([np.asarray(p30, dtype=np.int32),
                                           np.array([30], dtype=np.int32)]))
        FP = np.stack(fin)
        fgz = fe.gauges(FP)
        f_ms, f_ev = fgz["_ms_per_char"], ae.evidence_score(fgz)
        out["in_band_rank_test"]["final_population"] = {
            "n": int(len(f_ms)),
            "rho_archive_clamped_vs_ms": float(spearmanr(f_ev, f_ms).statistic),
            "ms_per_char_min": float(f_ms.min()), "ms_per_char_max": float(f_ms.max()),
            "ev_min": float(f_ev.min()), "ev_max": float(f_ev.max()),
            "ms_of_best_ev": float(f_ms[int(np.argmin(f_ev))]),
            "ev_of_fastest": float(f_ev[int(np.argmin(f_ms))]),
            "note": ("P16: rho between arm E's own objective and ms/char INSIDE the population "
                     "the search converged to. A value at/below ~0 means the objective could not "
                     "tell fast from slow among the very layouts it was choosing between."),
        }

    # ⚠ The JSON is written at the very END of this function, not here. Arm D's driver dumped at
    # this point and then kept adding sections; for arm E that silently left `champion_drivers`
    # (P14) OUT of the artifact while still PRINTING it to stdout — trap 19 exactly: "a metric
    # absent from a published JSON was never computed — check keys, not prose". Anything computed
    # below would have been quotable from the log but unreconcilable from the artifact.

    # ---- printable table ----
    print(f"\n{'=' * 126}")
    print(f"ARM E board — corpus blend-v1 (production default), .native frame, 90 WPM. "
          f"MODELLED ONLY.")
    hdr = (f"{'layout':<16}{'ARCH clamp':>11}{'r400 clamp':>11}{'r400 extr':>10}"
           f"{'ms/char':>10}{'vs best inc':>12}{'normfloor':>11}{'ood_A':>6}{'ood_R':>6}"
           f"{'dom?':>6}")
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
        print(f"{n:<16}{r['ev_archive_clamp']:>11.4f}{r['evidence_score_clamped']:>11.4f}"
              f"{r['evidence_score']:>10.4f}"
              f"{r['ms_per_char']:>10.4f}{d:>+12.4f}"
              f"{r['normfloor']:>11.6f}"
              f"{r['n_out_of_domain']:>6}{r['n_out_of_domain_random400']:>6}{dom:>6}")
    print("-" * 126)
    print("⚠ FOUR DIFFERENT RULERS. arm E optimized ARCH-clamp; arm D r400-clamp; arms A/C")
    print("  r400-extrap. Cross-arm comparison of any single evidence column is meaningless.")
    print("  ood_A = out-of-domain count vs the ARCHIVE domains; ood_R = vs random400's.")
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
              f"arm D recovered {r['armD_recovered_pct']:.1f}%  |  "
              f"arm E recovered {r['armE_recovered_pct']:.1f}%")
        print(f"  arm E vs arm B      {r['armE_residual_vs_armB']:+.4f} ms/char "
              f"= {r['armE_residual_in_floor_multiples']:.2f}x the conservative paired floor")
        print(f"  arm E vs {r['best_incumbent']:<12} {r['armE_residual_vs_best_incumbent']:+.4f}")
        print(f"  arm E vs arm D      "
              f"{r['armE_vs_armD']:+.4f}" if r['armE_vs_armD'] is not None else "")
        print(f"  arm E vs qwerty30m  {r['armE_vs_qwerty30m']:+.4f}")
        ob = r["outcome_bands"]
        held = [k for k, v in ob.items() if v and k.startswith("E")]
        print(f"  PRE-REGISTERED OUTCOME: {held if held else 'none of E1/E2/E3?!'}")
        print(f"    E1 (<=254.63, inside incumbent band): {ob['E1_inside_incumbent_band']}")
        print(f"    E2 (254.63-256.9):                    "
              f"{ob['E2_between_incumbents_and_armA']}")
        print(f"    E3 (>=256.9):                         {ob['E3_at_or_worse_than_armA']}")
        print(f"    worse than qwerty30m:                 {ob['worse_than_qwerty30m']}")
    if "clamp_binding" in out:
        cb = out["clamp_binding"]
        print(f"\nCLAMP BINDING (P6): all_bind={cb['all_bind']}, worst |reward outside| = "
              f"{cb['worst_abs_reward_outside']:.3e}")
    if "arme_out_of_domain" in out:
        o = out["arme_out_of_domain"]
        print(f"OUT-OF-DOMAIN (vs the ARCHIVE domains): arm E {o['n_out_of_domain']}"
              f"/{o['n_gauges']}; {o['n_clamped_to_own_optimum']} of those clamp to that gauge's "
              f"OWN optimum; {o['n_archive_curves_minimized_at_an_edge']} of "
              f"{o['n_gauges']} archive curves are minimized AT an edge")
    if "degeneracy" in out:
        dg = out["degeneracy"]
        print(f"DEGENERACY in arm E's top50: {dg['n_distinct_fitness']} distinct fitness values, "
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
        print(f"  {'band':<10}{'n':>8}{'ARCH clamp':>12}{'ARCH extr':>11}"
              f"{'r400 clamp':>12}{'r400 extr':>11}")
        for k, v in ib["bands"].items():
            print(f"  {k:<10}{v['n']:>8}{v['rho_archive_clamped']:>+12.4f}"
                  f"{v['rho_archive_extrapolate']:>+11.4f}"
                  f"{v['rho_clamped']:>+12.4f}{v['rho_raw']:>+11.4f}")
        print("  (want +1 if the weights inform about speed; <=0 means a good score is NOT")
        print("   evidence of a fast layout, so the champion must be read on ms/char alone.")
        print("   ARCH clamp is ARM E's own ruler — that is the column arm E rests on.)")
        if "final_population" in ib:
            fp = ib["final_population"]
            print(f"  P16, arm E's OWN final population (n={fp['n']}): "
                  f"rho(ARCH clamp, ms/char) = {fp['rho_archive_clamped_vs_ms']:+.4f}")
            print(f"    ms/char {fp['ms_per_char_min']:.4f}..{fp['ms_per_char_max']:.4f}; "
                  f"best-ev layout is {fp['ms_of_best_ev']:.4f} ms/char while the FASTEST scores "
                  f"{fp['ev_of_fastest']:.4f} (min {fp['ev_min']:.4f})")

    # ---- P14: is the champion COMFORT-DRIVEN, and what carries the attribution? -----------
    # Arm D's champion was: `comfort` held 43.55% of the fitted attribution and sat pinned at its
    # clamped floor. The ARCHIVE fit is a different shape — `comfort` is 11.16% and is minimized
    # at its **hi** edge, not its lo — so this must be measured, not inherited.
    if "champ-archive" in champs:
        e = out["per_layout"]["champ-archive"]
        drive = {}
        for name in LIVE_GAUGES:
            c = curves[name]
            lo, hi = c.domain
            lev = e["gauges14"][name]
            clamped = min(max(lev, lo), hi)
            xs = np.linspace(lo, hi, 20001)
            ys = c.price_many(xs, policy=CLAMP)
            argmin = float(xs[int(np.argmin(ys))])
            width = hi - lo
            here = float(c.price_many(np.array([lev]), policy=CLAMP)[0])
            drive[name] = {
                "level": lev, "clamped_level": clamped, "domain": [lo, hi],
                "price_here": here, "best_in_domain_price": float(ys.min()),
                "headroom_left": here - float(ys.min()),
                "curve_argmin": argmin,
                "at_its_own_optimum": bool(abs(clamped - argmin) < width * 1e-4),
                "pinned_at_lo": bool(abs(clamped - lo) < width * 1e-4),
                "pinned_at_hi": bool(abs(clamped - hi) < width * 1e-4),
                "shap_share_pct": c.shap_share_pct,
                "share_of_total_price": None,   # filled below
            }
        tot_abs = sum(abs(v["price_here"]) for v in drive.values())
        for v in drive.values():
            v["share_of_total_price"] = 100.0 * abs(v["price_here"]) / tot_abs if tot_abs else None
        top_driver = max(drive, key=lambda k: drive[k]["shap_share_pct"])
        out["champion_drivers"] = {
            "champion": champs["champ-archive"],
            "per_gauge": drive,
            "largest_attribution_gauge": top_driver,
            "largest_attribution_pct": drive[top_driver]["shap_share_pct"],
            "comfort_level": drive["comfort"]["level"],
            "comfort_domain": drive["comfort"]["domain"],
            "comfort_pinned_at_hi": drive["comfort"]["pinned_at_hi"],
            "comfort_pinned_at_lo": drive["comfort"]["pinned_at_lo"],
            "comfort_at_its_own_optimum": drive["comfort"]["at_its_own_optimum"],
            "comfort_shap_share_pct": drive["comfort"]["shap_share_pct"],
            "n_gauges_at_their_own_optimum": sum(1 for v in drive.values()
                                                 if v["at_its_own_optimum"]),
            "note": ("P14 / 'is the champion comfort-driven'. ⚠ The ARCHIVE `comfort` curve is "
                     "minimized at its **hi** edge (3.8371), the OPPOSITE of random400's lo edge "
                     "(6.5236) — so arm D's 'pinned against its floor' shape cannot be inherited "
                     "and had to be measured. `comfort` is a hand-chosen taste table "
                     "(DEFAULT_COMFORT, no fitted parameter — trap 48), so attribution resting "
                     "on it is a rival's taste, not evidence."),
        }
        cd = out["champion_drivers"]
        print(f"\nCHAMPION DRIVERS (P14): largest attribution = "
              f"{cd['largest_attribution_gauge']} at {cd['largest_attribution_pct']:.2f}%")
        print(f"  comfort {cd['comfort_level']:.4f} in {cd['comfort_domain']}: "
              f"pinned_hi={cd['comfort_pinned_at_hi']} pinned_lo={cd['comfort_pinned_at_lo']} "
              f"at_own_optimum={cd['comfort_at_its_own_optimum']} "
              f"({cd['comfort_shap_share_pct']:.2f}% of attribution)")
        print(f"  {cd['n_gauges_at_their_own_optimum']} of 14 gauges sit at their own clamped "
              f"optimum")

    # ---- P14b: which way did the champion move each gauge, and was that mechanism-right? ------
    # The headroom analysis predicted this BEFORE the run (79.3% of keybo-lsb's headroom was
    # mechanism-RIGHT). This is the post-hoc counterpart: what the search actually did.
    if "champ-archive" in champs:
        best_inc_name = min(J.INCUMBENTS, key=lambda n: out["per_layout"][n]["ms_per_char"])
        e_g = out["per_layout"]["champ-archive"]["gauges14"]
        i_g = out["per_layout"][best_inc_name]["gauges14"]
        moves, n_right, n_wrong = {}, 0, 0
        for name in LIVE_GAUGES:
            a, b = e_g[name], i_g[name]
            if abs(a - b) < 1e-12:
                continue
            up = a > b
            sign = EXPECTED_SIGN[name]
            # EXPECTED_SIGN +1 => a higher level costs MORE time => moving UP is mechanism-WRONG
            mech = ("WRONG" if up else "right") if sign > 0 else ("right" if up else "WRONG")
            n_wrong += mech == "WRONG"
            n_right += mech == "right"
            moves[name] = {"from": b, "to": a, "direction": "up" if up else "down",
                           "expected_sign": sign, "mechanism": mech,
                           "shap_share_pct": curves[name].shap_share_pct}
        out["champion_moves_vs_best_incumbent"] = {
            "reference": best_inc_name, "per_gauge": moves,
            "n_mechanism_right": n_right, "n_mechanism_wrong": n_wrong,
            "wrong_attribution_pct": sum(v["shap_share_pct"] for v in moves.values()
                                         if v["mechanism"] == "WRONG"),
            "note": ("what the search DID, against what the pre-run headroom analysis predicted "
                     "it would do. A gauge moved mechanism-WRONG is one the objective paid for "
                     "making the layout slower."),
        }
        mv = out["champion_moves_vs_best_incumbent"]
        print(f"\nCHAMPION MOVES vs {best_inc_name}: {mv['n_mechanism_right']} mechanism-right, "
              f"{mv['n_mechanism_wrong']} mechanism-WRONG "
              f"({mv['wrong_attribution_pct']:.2f}% of attribution moved the wrong way)")
        for name, v in sorted(moves.items(), key=lambda kv: -kv[1]["shap_share_pct"]):
            print(f"  {name:<12} {v['from']:9.4f} -> {v['to']:9.4f}  {v['direction']:<4} "
                  f"sgn{v['expected_sign']:+.0f}  {v['mechanism']:<5} "
                  f"({v['shap_share_pct']:.2f}%)")

    # ---- write the artifact LAST, so nothing computed above is missing from it (trap 19) ------
    path = STATE / "judgement.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nWROTE {path}")
    # Enumerate the keys a reader will cite, and fail if one is absent (trap 19: a metric absent
    # from a published JSON was never computed — check KEYS, not prose).
    required = ["per_layout", "paired", "ceilings", "dominance", "ruler_check", "clamp_binding",
                "arme_out_of_domain", "recovery", "degeneracy", "plateau_census",
                "in_band_rank_test", "champion_drivers", "champion_moves_vs_best_incumbent",
                "rulers", "analyze_frame"]
    missing_keys = [k for k in required if k not in out]
    assert not missing_keys, f"judgement.json is missing {missing_keys} — do not cite them"
    print(f"key check: all {len(required)} cited top-level keys present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
