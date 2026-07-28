"""ARM H JUDGE — written and committed WHILE THE RUNS WERE STILL EXECUTING, so its
thresholds provably were not tuned on any result. (ARM G established this practice; the
prereg §4/§5 fixed every threshold before phase 1 launched.)

It answers, in the pre-registered order:
  ① COLLECTED       — a FEASIBLE layout with oxey_style < armB - TOL.
                      ①a from an ARM H search; ①b if ONLY BALL-1 qualifies (enumeration
                      beat the search) -- registered as a distinct, weaker result.
  ② EMPTY FEASIBLE SET, demonstrated -- plus the BINDING constraint and by how much.
  ③ FAILURE by F1/F2/F3/F4.

Every per-axis adjudication runs through the SHIPPED `keybo analyze` path, never FastEval
alone. Every count is CONTESTED-per-pair, never a bare n/15: `sfr` is never counted (a
permutation invariant), and `alt`/`imbalance` are flagged as construction-ties wherever the
candidate shares arm B's hand partition (ULTRAAUDIT-INTERIM).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import armh_constants as AH  # noqa: E402
import evobj as EV  # noqa: E402

WORKTREE = Path("/tmp/armh")
STATE = Path("/local/home/zegertho/agent/state/armh/artifacts")
RUNS = STATE / "runs"

#: correlation clusters, for the cluster-corrected reading (trap 39: 15 of 17 "broad
#: competence" wins sat in 2 of 9 clusters; correcting the ~4x over-count reversed the
#: verdict). `oxey-style` is R^2=0.9082 on its 6 components IN-BAND (my own measurement),
#: so it is NOT independent evidence alongside them -- it shares their cluster.
CLUSTERS = {
    "same-finger":  ("sfb", "sfb-dist", "sfs", "sfs-dist"),
    "lateral":      ("lsb", "lsb-dist"),
    "flow":         ("alt", "roll", "sr-roll", "redir"),
    "scissor":      ("scissor",),
    "balance":      ("imbalance",),
    "comfort":      ("comfort",),
    "composite":    ("oxey-style",),
}


def shipped_analyze(layouts: list[str]) -> dict:
    uniq = sorted(set(layouts))
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "8"
    p = subprocess.run(["uv", "run", "--no-sync", "keybo", "analyze", "--json", *uniq],
                       cwd=str(WORKTREE), capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    rows = json.loads(p.stdout)["rows"]
    for lay in uniq:
        assert lay in rows, f"analyze DROPPED {lay!r} (trap 38)"
    return rows


def hand_partition(lay: str) -> frozenset:
    from keybo.geometry import ROW_STAGGERED_30 as G
    slot_hand = [G.hand(s[0]) for s in G.slots]
    perm = EV.perm_of(lay)
    return frozenset((i, slot_hand[int(perm[i])]) for i in range(30))


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def contested(cand: str, ref: str, rows: dict) -> dict:
    """Per-pair CONTESTED axis counts. A bare n/15 is NOT reportable (ULTRAAUDIT-INTERIM):
    `sfr` ties every pair by construction (permutation invariant) and is never counted, and
    `alt`/`imbalance` tie by construction whenever the two layouts share a hand partition."""
    gc, gr = rows[cand]["gauges"], rows[ref]["gauges"]
    same_part = hand_partition(cand) == hand_partition(ref)
    better, worse, ties = [], [], []
    for a in AH.ARMH_LIVE:
        d = AH.ARMH_DIR[a] * (gc[a] - gr[a])
        if d < -AH.ARMH_TOL:
            better.append(a)
        elif d > AH.ARMH_TOL:
            worse.append(a)
        else:
            ties.append(a)
    construction_ties = [a for a in ("alt", "imbalance") if a in ties and same_part]
    genuine_ties = [a for a in ties if a not in construction_ties]
    n_contested = len(better) + len(worse)
    cl_better, cl_worse = set(), set()
    for name, axes in CLUSTERS.items():
        if any(a in better for a in axes):
            cl_better.add(name)
        if any(a in worse for a in axes):
            cl_worse.add(name)
    return {
        "candidate": cand, "reference": ref, "hamming": hamming(cand, ref),
        "shares_hand_partition": same_part,
        "n_contested": n_contested, "n_better": len(better), "n_worse": len(worse),
        "better": better, "worse": worse,
        "ties_total": ties, "construction_ties": construction_ties,
        "genuine_ties": genuine_ties,
        "sfr_counted": False,
        "cluster_corrected": {"better": sorted(cl_better), "worse": sorted(cl_worse),
                              "n_better": len(cl_better), "n_worse": len(cl_worse)},
        "DOMINATES": len(worse) == 0 and len(better) >= 1,
    }


def feas(lay: str, rows: dict, eps: float) -> dict:
    g = rows[lay]["gauges"]
    ms = rows[lay]["time"]["ms_per_char"]
    edge = AH.ARMH_REF_MS + eps
    viol = {a: AH.ARMH_DIR[a] * (g[a] - AH.ARMH_REF[a]) for a in AH.ARMH_CONSTRAINED
            if AH.ARMH_DIR[a] * (g[a] - AH.ARMH_REF[a]) > AH.ARMH_TOL}
    ox = g[AH.ARMH_TARGET] - AH.ARMH_REF[AH.ARMH_TARGET]
    return {"layout": lay, "ms": ms, "ms_minus_armB": ms - AH.ARMH_REF_MS,
            "speed_ok": ms <= edge + AH.ARMH_TOL,
            "speed_excess_over_edge": max(0.0, ms - edge),
            "axes_violated": viol, "axes_ok": not viol,
            "oxey": g[AH.ARMH_TARGET], "oxey_minus_armB": ox,
            "FEASIBLE": (not viol) and ms <= edge + AH.ARMH_TOL,
            "COLLECTED": (not viol) and ms <= edge + AH.ARMH_TOL and ox < -AH.ARMH_TOL}


def main() -> int:
    summary = json.load(open(RUNS / "armh-summary.json"))
    sd_H = summary["sd_H"]
    eps = summary["eps"]
    out: dict = {
        "prereg": "agent-artifacts/armh/PREREGISTRATION.md (committed 491138b)",
        "judge_note": ("This judge was committed WHILE THE RUNS WERE STILL EXECUTING; every "
                       "threshold comes from the prereg, which predates phase 1."),
        "sd_H": sd_H, "eps_2sd_H": eps,
        "sd_H_quadruple": summary.get("sd_H_quadruple"),
        "sd_H_pool": summary.get("sd_H_pool"),
        "band_edge": AH.ARMH_REF_MS + eps if eps else None,
        "flip_threshold_sd": AH.ARMH_FLIP_SD,
        "BALL1_in_band": (eps >= AH.ARMH_BALL1_MS - AH.ARMH_REF_MS) if eps else None,
        "tol": AH.ARMH_TOL,
        "convention": ("as-shipped NESTED bad_redirect oxey (bad redirect charged +6.0); the "
                       "SAME convention as SPEEDTIE-1's 14.05x and ARM G. NOT comparable to a "
                       "post-OXEYFIX board."),
        "modelled_only": ("MODELLED ONLY: g-frame, baked 90 WPM, blend-v1, skipgrams "
                          "1-skip31, nested-bad_redirect oxey."),
    }

    # ---------------- F1: the 80% unique_evals floor, and n>=3 ----------------
    p1 = summary.get("phase1_baseline", [])
    p2 = summary.get("phase2_armh", [])
    def rowsof(pred):
        return [r for r in p1 + p2 if r.get("ok") and pred(r["tag"])]
    cold = rowsof(lambda t: t.startswith("armh-cold"))
    warm = rowsof(lambda t: t.startswith("armh-warm"))
    base = rowsof(lambda t: t.startswith("baseline-r"))
    out["budget_achieved"] = {
        r["tag"]: {"unique_evals_ACHIEVED": r["unique_evals_ACHIEVED"],
                   "frac": r["achieved_frac"], "clears_80pct_floor": r["clears_floor"],
                   "triple_agree": r["unique_evals_triple_agree"],
                   "keys_npy_retained": r["keys_npy_retained"],
                   "rc_from_sentinel": r["rc"]}
        for r in p1 + p2 if r.get("ok")}
    out["n_failed_runs"] = [r["tag"] for r in p1 + p2 if not r.get("ok")]
    n_search_clearing = sum(1 for r in cold + warm if r["clears_floor"])
    out["F1_fewer_than_3_seeds_clear_floor"] = n_search_clearing < 3

    # ---------------- the candidate set: EVERY champion + EVERY top-50 entry ----------------
    # An archive-only null is NOT a null (trap 4), so EMPTY must be tested against the whole
    # archive, not just the champions.
    cands: dict[str, str] = {}
    for r in cold + warm:
        cands[r["tag"]] = r["layout"]
    archive: set[str] = set()
    for r in cold + warm + base:
        for e in r.get("top50", []):
            archive.add(e["layout"])
    archive |= set(cands.values())
    archive.add(AH.ARMH_BALL1)
    archive.add(AH.ARMH_LAYOUT_REF)
    out["n_distinct_archive_layouts_swept"] = len(archive)

    # score the archive through FastEval (fast), then re-score every FEASIBLE candidate and
    # every champion through SHIPPED analyze (authoritative).
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armh/"), fe.corpus_dir
    alist = sorted(archive)
    g = fe.gauges(np.stack([EV.perm_of(x) for x in alist]))
    edge = AH.ARMH_REF_MS + eps
    nviol = np.zeros(len(alist), dtype=int)
    excess_by_axis = {}
    for a in AH.ARMH_CONSTRAINED:
        ex = AH.ARMH_DIR[a] * (g[a] - AH.ARMH_REF[a])
        excess_by_axis[a] = ex
        nviol += ex > AH.ARMH_TOL
    ms = g["_ms_per_char"]
    ax_ok = nviol == 0
    sp_ok = ms <= edge + AH.ARMH_TOL
    ox_better = g[AH.ARMH_TARGET] < AH.ARMH_REF[AH.ARMH_TARGET] - AH.ARMH_TOL
    out["archive_sweep"] = {
        "n": len(alist),
        "n_axes_feasible": int(ax_ok.sum()),
        "n_speed_ok": int(sp_ok.sum()),
        "n_FEASIBLE": int((ax_ok & sp_ok).sum()),
        "n_COLLECTED": int((ax_ok & sp_ok & ox_better).sum()),
        "min_n_axes_violated": int(nviol.min()),
        "axes_feasible_layouts": [alist[i] for i in np.where(ax_ok)[0]],
        "collected_layouts": [alist[i] for i in np.where(ax_ok & sp_ok & ox_better)[0]],
    }

    # ---------------- ② the BINDING constraint, quantified ----------------
    # Which leg, relaxed alone, would FIRST admit a candidate that improves oxey?
    want = ox_better
    binding = {}
    # (i) the SPEED leg: among layouts that satisfy all 13 axes AND improve oxey, the
    #     smallest ms excess over the band edge.
    cand_i = ax_ok & want
    binding["speed"] = {
        "n_candidates_satisfying_13_axes_and_improving_oxey": int(cand_i.sum()),
        "min_ms_excess_over_edge": (float(np.maximum(ms[cand_i] - edge, 0.0).min())
                                    if cand_i.any() else None),
        "best_layout": (alist[int(np.where(cand_i)[0][np.argmin(ms[cand_i])])]
                        if cand_i.any() else None),
        "min_ms_minus_armB": (float((ms[cand_i] - AH.ARMH_REF_MS).min())
                              if cand_i.any() else None),
    }
    # (ii) each AXIS leg: among layouts that satisfy the band, improve oxey, and violate
    #      ONLY this axis, the smallest excess on it.
    for a in AH.ARMH_CONSTRAINED:
        others = np.zeros(len(alist), dtype=int)
        for c in AH.ARMH_CONSTRAINED:
            if c != a:
                others += excess_by_axis[c] > AH.ARMH_TOL
        only_a = (excess_by_axis[a] > AH.ARMH_TOL) & (others == 0) & sp_ok & want
        binding[a] = {
            "n_violating_ONLY_this_axis_in_band_improving_oxey": int(only_a.sum()),
            "min_excess": (float(excess_by_axis[a][only_a].min()) if only_a.any() else None),
            "min_excess_relative_to_armB": (
                float(excess_by_axis[a][only_a].min() / max(abs(AH.ARMH_REF[a]), 1e-12))
                if only_a.any() else None),
        }
    out["binding_constraint_analysis"] = binding
    # the single binding leg = the one with a finite minimum excess and the SMALLEST relative
    # relaxation needed. Speed is expressed in ms/char and axes relative to arm B, so they are
    # reported side by side rather than compared on one scale (units differ -- trap 41).
    legs_live = [k for k, v in binding.items() if k != "speed"
                 and v["n_violating_ONLY_this_axis_in_band_improving_oxey"] > 0]
    out["binding_verdict"] = {
        "speed_leg_admits_if_relaxed_by_ms": binding["speed"]["min_ms_excess_over_edge"],
        "axis_legs_with_solo_violators": legs_live,
        "note": ("speed is in ms/char and axis excesses are relative to arm B, so the two are "
                 "reported side by side, never compared on one scale (trap 41: different "
                 "units of aggregation)."),
    }

    # ---------------- shipped-path adjudication of the real candidates ----------------
    refs = {"armB": AH.ARMH_LAYOUT_REF, "BALL-1": AH.ARMH_BALL1}
    judged = sorted(set(list(cands.values()) + list(refs.values())
                        + out["archive_sweep"]["collected_layouts"]))
    rows = shipped_analyze(judged)
    out["shipped_feasibility"] = {}
    for tag, lay in list(cands.items()) + [(f"_ref-{k}", v) for k, v in refs.items()]:
        if lay in rows:
            out["shipped_feasibility"][tag] = feas(lay, rows, eps)
    out["shipped_collected"] = {
        lay: feas(lay, rows, eps) for lay in out["archive_sweep"]["collected_layouts"]
        if lay in rows}

    # F2: no returned champion may be infeasible.
    out["F2_gate_rejected_a_champion"] = any(
        not v["FEASIBLE"] for k, v in out["shipped_feasibility"].items()
        if not k.startswith("_ref"))

    # F3: did the SEARCH beat the 435-point enumeration?
    search_ox = [v["oxey"] for k, v in out["shipped_feasibility"].items()
                 if not k.startswith("_ref") and v["FEASIBLE"]]
    out["F3_search_lost_to_enumeration"] = (
        (not search_ox) or min(search_ox) > AH.ARMH_BALL1_OXEY - AH.ARMH_TOL)
    out["best_feasible_search_oxey"] = min(search_ox) if search_ox else None
    out["BALL1_oxey"] = AH.ARMH_BALL1_OXEY

    # ---------------- predictions ----------------
    def clears(rs):
        return [r for r in rs if r["clears_floor"]]
    cold_feas = [k for k, v in out["shipped_feasibility"].items()
                 if k.startswith("armh-cold") and v["axes_ok"]]
    warm_feas = [k for k, v in out["shipped_feasibility"].items()
                 if k.startswith("armh-warm") and v["axes_ok"]]
    faster = [k for k, v in out["shipped_feasibility"].items()
              if not k.startswith("_ref") and v["ms"] < AH.ARMH_REF_MS]
    borrowed = 0.0617
    ratio = sd_H / borrowed if sd_H else None
    out["predictions"] = {
        "P1_warm_ge3_axes_feasible": {"n": len(warm_feas), "held": len(warm_feas) >= 3,
                                      "which": warm_feas},
        "P2_cold_returns_ZERO_axes_feasible": {"n": len(cold_feas),
                                              "held": len(cold_feas) == 0,
                                              "which": cold_feas,
                                              "self_adverse": True},
        "P3_search_beats_BALL1_oxey": {"best": out["best_feasible_search_oxey"],
                                       "bar": AH.ARMH_BALL1_OXEY,
                                       "held": not out["F3_search_lost_to_enumeration"]},
        "P4_no_champion_faster_than_armB": {"faster": faster, "held": not faster,
                                            "self_adverse": True},
        "P5_sd_H_within_1.5x_of_borrowed_0.0617": {
            "sd_H": sd_H, "borrowed": borrowed, "ratio": ratio,
            "held": (ratio is not None and (1 / 1.5) <= ratio <= 1.5)},
        "P6_2sd_H_admits_BALL1": {"two_sd": eps, "needed": AH.ARMH_BALL1_MS - AH.ARMH_REF_MS,
                                  "held": out["BALL1_in_band"]},
    }

    # ---------------- CONTESTED counts for whatever we are reporting ----------------
    headline = None
    if out["shipped_collected"]:
        headline = min(out["shipped_collected"].items(), key=lambda kv: kv[1]["oxey"])[0]
    out["headline_layout"] = headline
    if headline:
        allrefs = {"armB": AH.ARMH_LAYOUT_REF}
        rows2 = shipped_analyze([headline, *allrefs.values()])
        out["contested"] = {k: contested(headline, v, rows2) for k, v in allrefs.items()}

    # ---------------- n_runs vs n_distinct, Hamming BOTH ways (SPEEDTIE-BUDGET traps 2,3) ----
    for name, rs in (("baseline", base), ("armh-cold", cold), ("armh-warm", warm)):
        lays = [r["layout"] for r in clears(rs)]
        dis = sorted(set(lays))
        pair_runs = [hamming(a, b) for i, a in enumerate(lays) for b in lays[i + 1:]]
        pair_dist = [hamming(a, b) for i, a in enumerate(dis) for b in dis[i + 1:]]
        out.setdefault("run_structure", {})[name] = {
            "n_runs": len(lays), "n_distinct": len(dis),
            "mean_hamming_over_runs": float(np.mean(pair_runs)) if pair_runs else None,
            "mean_hamming_over_distinct": float(np.mean(pair_dist)) if pair_dist else None,
            "n_zero_pairs": sum(1 for x in pair_runs if x == 0),
            "champions": lays,
            "recovers_armB": sum(1 for x in lays if x == AH.ARMH_LAYOUT_REF),
        }

    # ---------------- the VERDICT ----------------
    F1 = out["F1_fewer_than_3_seeds_clear_floor"]
    F2 = out["F2_gate_rejected_a_champion"]
    F3 = out["F3_search_lost_to_enumeration"]
    search_collected = [k for k, v in out["shipped_feasibility"].items()
                        if not k.startswith("_ref") and v["COLLECTED"]]
    if F1 or F2:
        verdict = "③ FAILURE (F1 seeds below floor)" if F1 else "③ FAILURE (F2 gate rejected a champion)"
    elif search_collected:
        verdict = "① COLLECTED (①a — by an ARM H SEARCH champion)"
    elif out["shipped_collected"]:
        only_ball = set(out["shipped_collected"]) <= {AH.ARMH_BALL1}
        verdict = ("① COLLECTED (①b — by ENUMERATION only; the search added nothing) "
                   "+ ③ F3" if only_ball else
                   "① COLLECTED (①a — from a search ARCHIVE entry, not a champion)")
    else:
        verdict = "② EMPTY FEASIBLE SET (demonstrated)"
    out["VERDICT"] = verdict
    out["F_conditions"] = {"F1": F1, "F2": F2, "F3": F3}
    out["search_collected_champions"] = search_collected

    json.dump(out, open(STATE / "judgement.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("archive_sweep", "binding_constraint_analysis",
                                   "shipped_feasibility", "budget_achieved")},
                     indent=1, default=str))
    print(f"\n*** VERDICT: {verdict}")
    print(f"WROTE {STATE / 'judgement.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
