"""ARM H — HOSTILE VERIFICATION of my own headline, and the F2 diagnostic.

Two claims need refuting-or-confirming by a route that does NOT share a component with the
claim (trap 45; and the SELF-AUDIT SWEEP found two "independent" controls that shared the
component under test):

  (1) F2 fired: the gate rejected all 5 armh-cold champions. My prereg attaches the
      conclusion "=> my hardness construction is broken" to F2. That conclusion is TESTABLE
      and I test it here rather than reinterpreting F2 to suit me. The discriminator is the
      champion's SEARCH FITNESS: if fitness >= BIG then the OBJECTIVE also labelled it
      infeasible, so the two paths AGREE and there is no cross-path defect -- the cold arm
      simply found nothing feasible, which is what my own P2 PREDICTED. If fitness < BIG
      while the gate says infeasible, the paths DISAGREE and my construction really is broken.

  (2) the headline's "12 better / 0 worse / 2 construction-ties" -- verified by reading the
      SHIPPED analyze table cell by cell, printing every number, and re-deriving the hand
      partition from live Geometry. A dominance count is the single most defect-prone
      statistic in this campaign (tie-credit defects at FOUR independent sites).

And the one that cuts against my own headline, which is why it is here rather than omitted:

  (3) the SENSITIVITY TABLE I pre-committed to publishing regardless of outcome -- is the
      headline in-band under the OTHER TWO rulers (sd_G = 0.049171, borrowed = 0.0617)?
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
BIG = AH.ARMH_BIG
SD_G = 0.049171        # ARM G's, quoted from ARMG-1 -- SENSITIVITY ONLY, no verdict
SD_BORROWED = 0.0617   # SPEEDTIE-1's, from the artifact -- SENSITIVITY ONLY, no verdict


def shipped(layouts: list[str]) -> dict:
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "8"
    p = subprocess.run(["uv", "run", "--no-sync", "keybo", "analyze", "--json",
                        *sorted(set(layouts))],
                       cwd=str(WORKTREE), capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    rows = json.loads(p.stdout)["rows"]
    for lay in layouts:
        assert lay in rows, f"analyze DROPPED {lay!r}"
    return rows


def partition(lay: str) -> frozenset:
    from keybo.geometry import ROW_STAGGERED_30 as G
    sh = [G.hand(s[0]) for s in G.slots]
    perm = EV.perm_of(lay)
    return frozenset((i, sh[int(perm[i])]) for i in range(30))


def main() -> int:
    summary = json.load(open(STATE / "runs" / "armh-summary.json"))
    sd_H = summary["sd_H"]
    eps = summary["eps"]
    out: dict = {"sd_H": sd_H, "eps": eps, "BIG": BIG}

    # ================= (1) THE F2 DIAGNOSTIC =================
    # Does the OBJECTIVE agree with the GATE about every champion's feasibility?
    gate = json.load(open(STATE / "gate-armh.json"))["results"]
    rowsf = []
    for r in summary["phase2_armh"]:
        if not r.get("ok"):
            continue
        tag = r["tag"]
        fit = r["search_fitness"]
        obj_says_feasible = fit < BIG
        g = gate[tag]
        gate_says_feasible = g["FEASIBLE"]
        rowsf.append({
            "tag": tag, "search_fitness": fit,
            "objective_branch": "FEASIBLE" if obj_says_feasible else "INFEASIBLE",
            "gate_says": "FEASIBLE" if gate_says_feasible else "INFEASIBLE",
            "AGREE": obj_says_feasible == gate_says_feasible,
            "n_axes_violated_per_gate": g["n_axes_violated"],
            "ms_minus_armB": g["ms_minus_armB"],
        })
    n_dis = sum(1 for x in rowsf if not x["AGREE"])
    out["F2_diagnostic"] = {
        "per_champion": rowsf,
        "n_champions": len(rowsf),
        "n_cross_path_DISAGREEMENTS": n_dis,
        "construction_broken": n_dis > 0,
        "reading": (
            "F2's registered CONCLUSION is 'my hardness construction is broken'. That is "
            "FALSE by measurement iff n_cross_path_DISAGREEMENTS == 0: every cold champion "
            "the gate rejected was ALSO scored in the objective's infeasible branch "
            "(fitness >= 1e6), so both paths agree and the cold arm simply found nothing "
            "feasible -- which is exactly what my own self-adverse P2 predicted."),
    }

    # ================= (2) THE HEADLINE, CELL BY CELL =================
    cands = {
        "armB": AH.ARMH_LAYOUT_REF,
        "BALL-1 (enumerated)": AH.ARMH_BALL1,
        "HEADLINE (armh-warm x4)": "flmpg-,uoysntcdireahkxvwb.'jzq",
    }
    # every COLLECTED archive layout too
    for lay in json.load(open(STATE / "judgement.json"))["archive_sweep"]["collected_layouts"]:
        cands.setdefault(f"collected {lay}", lay)
    for name, lay in cands.items():
        assert len(lay) == 30 and set(lay) == set(EV.C30M), f"{name} is not a C30M perm"
    rows = shipped(list(cands.values()))

    print("=" * 108)
    print("PER-AXIS TABLE, read from the SHIPPED analyze output (arm B is the constraint level)")
    print("=" * 108)
    hl = cands["HEADLINE (armh-warm x4)"]
    b1 = cands["BALL-1 (enumerated)"]
    ab = cands["armB"]
    print(f"{'axis':<12}{'dir':>4}{'armB':>13}{'BALL-1':>13}{'HEADLINE':>13}"
          f"{'HL-armB':>12}  verdict-vs-armB")
    per_axis = {}
    nb = nw = nt = 0
    for a in AH.ARMH_LIVE:
        va, vb, vh = rows[ab]["gauges"][a], rows[b1]["gauges"][a], rows[hl]["gauges"][a]
        d = AH.ARMH_DIR[a] * (vh - va)
        verdict = "BETTER" if d < -AH.ARMH_TOL else ("WORSE" if d > AH.ARMH_TOL else "tie")
        nb += verdict == "BETTER"
        nw += verdict == "WORSE"
        nt += verdict == "tie"
        per_axis[a] = {"dir": AH.ARMH_DIR[a], "armB": va, "BALL1": vb, "headline": vh,
                       "signed_excess_vs_armB": d, "verdict": verdict}
        print(f"{a:<12}{AH.ARMH_DIR[a]:>+4.0f}{va:>13.6f}{vb:>13.6f}{vh:>13.6f}"
              f"{vh - va:>+12.6f}  {verdict}")
    msa, msh = rows[ab]["time"]["ms_per_char"], rows[hl]["time"]["ms_per_char"]
    print(f"{'ms/char':<12}{1:>+4.0f}{msa:>13.6f}"
          f"{rows[b1]['time']['ms_per_char']:>13.6f}{msh:>13.6f}{msh - msa:>+12.6f}")
    same_part = partition(hl) == partition(ab)
    ctie = [a for a in ("alt", "imbalance") if per_axis[a]["verdict"] == "tie" and same_part]
    out["headline_per_axis"] = per_axis
    out["headline_counts"] = {
        "layout": hl, "hamming_from_armB": sum(1 for x, y in zip(hl, ab, strict=True) if x != y),
        "n_better": nb, "n_worse": nw, "n_tie": nt, "n_contested": nb + nw,
        "shares_armB_hand_partition": same_part,
        "construction_ties": ctie,
        "genuine_ties": [a for a in AH.ARMH_LIVE
                         if per_axis[a]["verdict"] == "tie" and a not in ctie],
        "sfr_EXCLUDED_never_counted": True,
        "reportable_as": f"{nb} better / {nw} worse of {nb + nw} CONTESTED "
                         f"(+{len(ctie)} construction-ties: {ctie}; sfr never counted)",
        "ms_minus_armB": msh - msa,
        "oxey_minus_armB": per_axis[AH.ARMH_TARGET]["signed_excess_vs_armB"],
    }

    # ================= (3) THE SENSITIVITY TABLE — pre-committed, and it cuts against me ====
    sens = {}
    for name, sd in (("sd_H (PRIMARY, mine)", sd_H), ("sd_G (ARM G)", SD_G),
                     ("borrowed (SPEEDTIE-1)", SD_BORROWED)):
        band = 2 * sd
        sens[name] = {"sd": sd, "two_sd": band,
                      "HEADLINE_in_band": (msh - msa) <= band + AH.ARMH_TOL,
                      "BALL1_in_band": (AH.ARMH_BALL1_MS - AH.ARMH_REF_MS) <= band + AH.ARMH_TOL,
                      "headline_excess_over_band": max(0.0, (msh - msa) - band)}
    out["sensitivity_to_ruler"] = sens
    print("\n" + "=" * 108)
    print("SENSITIVITY TO THE RULER (pre-committed in prereg §5 — published regardless of outcome)")
    print("=" * 108)
    print(f"{'ruler':<24}{'sd':>11}{'2sd':>11}   HEADLINE(+%.6f)  BALL-1(+%.6f)"
          % (msh - msa, AH.ARMH_BALL1_MS - AH.ARMH_REF_MS))
    for name, v in sens.items():
        print(f"{name:<24}{v['sd']:>11.6f}{v['two_sd']:>11.6f}   "
              f"{'IN-BAND ' if v['HEADLINE_in_band'] else 'OUT     '}          "
              f"{'IN-BAND' if v['BALL1_in_band'] else 'OUT'}")
    out["ruler_robustness"] = {
        "HEADLINE_in_band_under_n_of_3_rulers": sum(1 for v in sens.values()
                                                    if v["HEADLINE_in_band"]),
        "BALL1_in_band_under_n_of_3_rulers": sum(1 for v in sens.values()
                                                 if v["BALL1_in_band"]),
    }

    # ================= (4) is the headline a genuine SEARCH find? =================
    # warm-r2 returned BALL-1 itself. So: did any warm seed find something BETTER than the
    # enumeration, and how far is the headline from arm B and from BALL-1?
    warm = [r for r in summary["phase2_armh"] if r["tag"].startswith("armh-warm") and r["ok"]]
    out["warm_champions"] = {r["tag"]: {"layout": r["layout"], "fitness": r["search_fitness"],
                                       "is_BALL1": r["layout"] == AH.ARMH_BALL1,
                                       "clears_floor": r["clears_floor"]} for r in warm}
    out["headline_vs_BALL1"] = {
        "hamming": sum(1 for x, y in zip(hl, b1, strict=True) if x != y),
        "oxey_headline": rows[hl]["gauges"][AH.ARMH_TARGET],
        "oxey_BALL1": rows[b1]["gauges"][AH.ARMH_TARGET],
        "search_beat_enumeration_by": rows[b1]["gauges"][AH.ARMH_TARGET]
                                      - rows[hl]["gauges"][AH.ARMH_TARGET],
    }
    # is the headline reachable by ONE swap from arm B? (i.e. was it in my enumerated ball?)
    out["headline_in_1swap_ball_of_armB"] = out["headline_counts"]["hamming_from_armB"] == 2

    json.dump(out, open(STATE / "verify-headline.json", "w"), indent=1, default=str)
    print("\n" + "=" * 108)
    print(f"F2 cross-path disagreements: {n_dis} of {len(rowsf)}  => construction broken: "
          f"{n_dis > 0}")
    print(f"headline CONTESTED: {out['headline_counts']['reportable_as']}")
    print(f"headline Hamming from arm B: {out['headline_counts']['hamming_from_armB']}  "
          f"(in 1-swap ball: {out['headline_in_1swap_ball_of_armB']})")
    print(f"search beat the enumeration on oxey by "
          f"{out['headline_vs_BALL1']['search_beat_enumeration_by']:.6f}")
    print(f"WROTE {STATE / 'verify-headline.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
