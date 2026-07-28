"""ARM H — SELF-SEPARATION: re-read my own results as a hostile stranger.

For each claim: what would refute it, does my check share a component with its target, did
any control run only AFTER I used its result. Report what I KILLED of my own.

The four things I go after, in descending order of how much they would hurt:

  K1  "12 better / 0 worse" — a dominance count is the most defect-prone statistic in this
      campaign (tie-credit defects at FOUR independent sites). Even granting every win is
      numerically strict, HOW BIG is each margin relative to that axis's own spread? A
      strict win of 0.001 on an axis spanning 4 units is a tie wearing a win's clothes.

  K2  the RULER. My headline is in-band under MY sd_H and OUT of band under both other
      rulers ever measured on this objective. That is the anti-conservative direction and it
      is the single biggest threat to my own headline.

  K3  F2. My prereg attaches "the construction is broken" to F2, and the construction is
      measurably NOT broken. Rather than reinterpret F2 to escape a FAILURE verdict (which
      would be moving the line after seeing the data), report the outcome under BOTH
      readings and show whether the substantive finding depends on the choice.

  K4  is the headline a SEARCH result at all? It is warm-only. The cold arm found nothing.
      So it is a neighbourhood search around the incumbent, which I declared in the prereg
      and must restate rather than quietly drop.
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
HEADLINE = "flmpg-,uoysntcdireahkxvwb.'jzq"


def shipped(layouts: list[str]) -> dict:
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "8"
    p = subprocess.run(["uv", "run", "--no-sync", "keybo", "analyze", "--json",
                        *sorted(set(layouts))], cwd=str(WORKTREE),
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout)["rows"]


def main() -> int:
    out: dict = {}
    summary = json.load(open(STATE / "runs" / "armh-summary.json"))
    ver = json.load(open(STATE / "verify-headline.json"))
    sd_H, eps = summary["sd_H"], summary["eps"]
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armh/"), fe.corpus_dir

    # =============== K1: HOW BIG IS EACH "WIN"? =================================
    # Scale each margin against TWO independent yardsticks:
    #  (a) the range of that axis over the SIX FROZEN 1M CHAMPIONS -- the pool-matched,
    #      near-optimal scale my prereg used (trap 26: a random pool would be a Simpson
    #      artifact and would flatter every margin);
    #  (b) the range over arm B's exhaustive 435-member 1-swap ball -- a LOCAL scale that
    #      does not share the six-champion pool with (a).
    six = fe.gauges(np.stack([EV.perm_of(x) for x in AH.ARMH_SIX]))
    b = EV.perm_of(AH.ARMH_LAYOUT_REF)
    pairs = [(i, j) for i in range(30) for j in range(i + 1, 30)]
    nb = np.repeat(b[None, :], len(pairs), axis=0)
    for r, (i, j) in enumerate(pairs):
        nb[r, i], nb[r, j] = b[j], b[i]
    ball = fe.gauges(nb)

    per_axis = ver["headline_per_axis"]
    k1 = {}
    for a in AH.ARMH_LIVE:
        m = -float(per_axis[a]["signed_excess_vs_armB"])   # +ve = improvement
        r6 = float(six[a].max() - six[a].min())
        rb = float(ball[a].max() - ball[a].min())
        k1[a] = {
            "improvement": m,
            "range_over_six_frozen": r6,
            "range_over_1swap_ball": rb,
            "frac_of_six_range": (m / r6 if r6 > 0 else None),
            "frac_of_ball_range": (m / rb if rb > 0 else None),
            "verdict": per_axis[a]["verdict"],
        }
    # A win is "MARGINAL" if it is a strict win but under 1% of BOTH yardsticks. The 1%
    # threshold is a judgement I am making now and labelling as such -- it is not derived,
    # so I report the raw fractions too and no verdict rests on the threshold alone.
    strict = [a for a in AH.ARMH_LIVE if k1[a]["verdict"] == "BETTER"]
    marginal = [a for a in strict
                if (k1[a]["frac_of_six_range"] or 0) < 0.01
                and (k1[a]["frac_of_ball_range"] or 0) < 0.01]
    substantive = [a for a in strict if a not in marginal]
    out["K1_margin_quality"] = {
        "per_axis": k1,
        "n_strict_wins": len(strict),
        "n_MARGINAL_wins_under_1pct_of_both_yardsticks": len(marginal),
        "marginal_axes": marginal,
        "substantive_axes": substantive,
        "threshold_note": ("the 1% cut is MY judgement, made now and labelled -- not derived. "
                           "Raw fractions are published so a reader can set their own."),
        "KILL": (f"the '12 better / 0 worse' is arithmetically correct and every margin "
                 f"exceeds the numerical floor (7.1e-15) by orders -- but {len(marginal)} of "
                 f"{len(strict)} wins are under 1% of BOTH the six-champion range AND the "
                 f"1-swap-ball range: {marginal}. Honest restatement: "
                 f"{len(substantive)} substantive improvements + {len(marginal)} "
                 f"technically-strict-but-negligible + 2 construction-ties."),
    }

    # =============== K2: THE RULER (the biggest threat to my own headline) ==========
    sens = ver["sensitivity_to_ruler"]
    out["K2_ruler_dependence"] = {
        "sensitivity": sens,
        "headline_ms_over_armB": ver["headline_counts"]["ms_minus_armB"],
        "BALL1_ms_over_armB": AH.ARMH_BALL1_MS - AH.ARMH_REF_MS,
        "KILL": ("MY HEADLINE'S 'SPEED-TIED' STATUS IS NOT RULER-ROBUST. At +0.139048 it is "
                 "in-band ONLY under my own sd_H (2sd = 0.199051) -- the LARGEST of the three "
                 "rulers ever measured on this objective -- and OUT of band under both ARM G's "
                 "sd_G (0.098342) and the borrowed 0.0617 (0.1234). My prereg argued any "
                 "residual bias would be TIGHTER (conservative); it came out LOOSER, the same "
                 "anti-conservative direction ARM G was caught in. What my structural fix DOES "
                 "buy is that search band == verdict band, so a looser sd cannot open a gap "
                 "between them -- but it cannot make the headline ruler-independent."),
        "SURVIVES": ("BALL-1 at +0.065847 is IN-BAND under ALL THREE rulers. So the "
                     "ruler-robust collected result is BALL-1 (oxey 7.577429, 13/13 axes "
                     "satisfied), NOT the headline. I report BALL-1 as the claim that does not "
                     "depend on whose ruler you use, and the headline as strictly stronger but "
                     "ruler-dependent."),
    }

    # =============== K3: F2, both readings ==========================================
    f2 = ver["F2_diagnostic"]
    gate = json.load(open(STATE / "gate-armh.json"))["results"]
    warm_only_feasible = all(gate[t]["FEASIBLE"] for t in gate
                             if t.startswith("armh-warm"))
    out["K3_F2_both_readings"] = {
        "n_cross_path_disagreements": f2["n_cross_path_DISAGREEMENTS"],
        "strict_reading": ("F2 as I WROTE it FIRED: the gate rejected 5 returned champions "
                           "(all armh-cold). Verdict under the strict reading: ③ FAILURE."),
        "warrant_reading": ("F2's registered WARRANT -- 'my hardness construction is broken' "
                            "-- is FALSE BY MEASUREMENT: 0 of 10 cross-path disagreements. "
                            "Every rejected champion was ALSO in the objective's infeasible "
                            "branch (fitness >= 1e6), so both paths AGREE. The cold arm simply "
                            "found nothing feasible and returned its least-infeasible archive "
                            "entry, which is correct engine behaviour and exactly what my own "
                            "self-adverse P2 PREDICTED."),
        "MY_OWN_DEFECT": ("F2 was MIS-SPECIFIED IN MY PREREG. It conflated 'an infeasible "
                          "champion was returned' with 'the construction is broken'. A run that "
                          "finds nothing feasible MUST return something, and the engine's "
                          "archive-best is by construction the least-infeasible layout -- so "
                          "F2 as written fires on the EXPECTED output of a correctly-working "
                          "arm. It should have been scoped to a CROSS-PATH DISAGREEMENT "
                          "(objective says feasible, gate says not). I am registering this as "
                          "my defect rather than quietly reading F2 the way that suits me."),
        "does_the_finding_depend_on_the_reading": not warm_only_feasible,
        "RESOLUTION": ("I report BOTH and show the substantive finding does NOT depend on the "
                       "choice: gating the 5 warm champions ALONE returns rc=0 (all FEASIBLE by "
                       "both paths), so 'a feasible strictly-better layout exists' stands under "
                       "either reading. What the strict reading costs me is the LABEL, not the "
                       "result -- and I let the strict reading stand on the label."),
    }

    # =============== K4: is it a SEARCH result? ======================================
    cold = [r for r in summary["phase2_armh"] if r["tag"].startswith("armh-cold") and r["ok"]]
    warm = [r for r in summary["phase2_armh"] if r["tag"].startswith("armh-warm") and r["ok"]]
    coldrows = shipped([r["layout"] for r in cold])
    cold_ox = {r["tag"]: coldrows[r["layout"]]["gauges"][AH.ARMH_TARGET] for r in cold}
    out["K4_warm_only"] = {
        "cold_n_feasible": 0,
        "cold_champion_oxey": cold_ox,
        "cold_best_oxey": min(cold_ox.values()),
        "warm_champions": {r["tag"]: r["layout"] for r in warm},
        "warm_n_distinct": len(set(r["layout"] for r in warm)),
        "KILL": ("THE HEADLINE IS WARM-ONLY. All 5 cold seeds returned INFEASIBLE champions "
                 "(0 feasible), so nothing here is a cold-start discovery. armh-warm injects "
                 "arm B into every island, so it is a NEIGHBOURHOOD SEARCH AROUND THE "
                 "INCUMBENT -- declared in prereg §3 and restated here rather than dropped. "
                 "The honest description is 'a constrained local improvement on arm B', not "
                 "'a search found a better layout'."),
        "AND_THE_LEVER_IS_REAL_BUT_NOT_FREE": (
            f"the cold arm reached oxey {min(cold_ox.values()):.6f} -- FAR below arm B's "
            f"8.611046 and below my headline's 4.446491 -- while violating 4-5 hard axes and "
            f"sitting +0.24 to +0.30 ms/char out. So SPEEDTIE-1's oxey headroom is REAL and "
            f"LARGE, and the reason it is not collectable is that reaching it requires paying "
            f"on the other axes. That is the measured content of the lever."),
    }

    # =============== K5: did any control run only AFTER I used its result? ===========
    out["K5_control_ordering"] = {
        "C1_C7_isolation_reproduction_crosspath_mutation_directions_floor": (
            "ALL ran BEFORE PREREGISTRATION.md was written (commit 491138b), which itself "
            "predates every run."),
        "planted_infeasible_gate_test": (
            "ran BEFORE phase 1 launched, at eps=0.0983 (a placeholder band explicitly NOT "
            "used in any verdict). rc=1 planted / rc=0 clean / rc=1 tight-band."),
        "1swap_ball_enumeration": (
            "ran BEFORE the prereg and is quoted IN it, including the flip threshold "
            "0.032924 -- so the BALL-1 fork was registered before sd_H existed."),
        "judge": "committed 2b90b47 WHILE phase 1 was executing (138 workers live, 0 rc sentinels).",
        "sd_H": ("measured in phase 1 and used to set phase 2's band. It is independent of "
                 "every ARM H result by construction (baseline objective, different arm)."),
        "VIOLATION_FOUND": ("NONE for the controls. But note the ordering hazard I DID create: "
                            "the 1-swap-ball enumeration told me BALL-1 existed BEFORE I "
                            "registered the verdict rules, so my prereg was written knowing a "
                            "feasible layout existed. I disclosed that in prereg §2 and "
                            "registered ①b (enumeration-only) as a distinct weaker outcome "
                            "precisely so that knowledge could not be laundered into ①a."),
    }

    # =============== K6: shared components between check and target ==================
    out["K6_shared_components"] = {
        "objective_vs_verdict": ("BOTH would have used FastEval. Mitigated: every published "
                                 "number and the gate go through SHIPPED analyze, pinned to "
                                 "FastEval at 1.233e-14 with a MUTATION control proving the pin "
                                 "bites. This is a genuine independent path, not the same check "
                                 "twice."),
        "oxey_restates_its_constraints": ("oxey-style is R^2=0.9082 on {sfb,lsb,scissor,"
                                          "imbalance,redir,alt} IN-BAND (my measurement). So "
                                          "minimizing oxey while holding those six is partly "
                                          "self-cancelling, and oxey is NOT independent evidence "
                                          "alongside them. The cluster-corrected count (6 of 6 "
                                          "clusters better) puts oxey in its own cluster for "
                                          "exactly this reason."),
        "lsb_vs_lsb_dist": ("near-duplicates (spearman 1.0000, sibling-measured). The headline "
                            "improves BOTH, so its 12 wins are at most 11 independent axes -- "
                            "and after K1's marginality cut, fewer still."),
        "K1_yardsticks": ("the two yardsticks in K1 do NOT share a pool: the six frozen "
                          "champions vs arm B's 1-swap ball. A margin judged small by both is "
                          "small on two independent local scales."),
    }

    json.dump(out, open(STATE / "self-separation.json", "w"), indent=1, default=str)
    for k in ("K1_margin_quality", "K2_ruler_dependence", "K3_F2_both_readings",
              "K4_warm_only", "K5_control_ordering", "K6_shared_components"):
        print("=" * 100)
        print(k)
        v = out[k]
        for kk in ("KILL", "SURVIVES", "MY_OWN_DEFECT", "RESOLUTION", "VIOLATION_FOUND",
                   "AND_THE_LEVER_IS_REAL_BUT_NOT_FREE", "strict_reading", "warrant_reading"):
            if kk in v:
                print(f"  [{kk}] {v[kk]}")
    print("=" * 100)
    print(f"WROTE {STATE / 'self-separation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
