"""FLAGSHIP-1 — STEELMAN flagship-c3, the layout my recommendation rejects.

Lead with the counterargument. flagship-c3 has a serious case:
  * every surviving aggregate in GEOMEAN-1 ranks it FIRST;
  * SELECT-MAXIMIN-1's MEAN ranking puts it first with AND without the qwerty anchor, and it
    leads every per-corpus column;
  * my own robustness board gives it the best mean (0.9905) AND the best worst-case (0.8832)
    of the six, and it is field-worst on only 2 of 42 (corpus x gauge) cells;
  * CLOSURE-3 produced it as a layout that fully dominates archive-1846 and lsb-sib.

So the honest question is not "is keybo-lsb faster" (it is, on the model) but **is the speed
difference large enough to override a broad gauge-quality advantage?** This driver tries to
answer that on flagship-c3's own terms, three ways:

1. **WHAT DOES THE SPEED GAP BUY, IN UNITS A USER CARES ABOUT?** Convert the ms/char gap to
   words-per-minute-equivalent and to seconds-per-1000-words. If the answer is "unmeasurable",
   the gauge advantage should win. Requires no new model: WPM = 60000 / (5 * ms_per_char)
   under the standard 5-chars-per-word convention, which is exactly the convention the
   surface's own target_wpm uses.

2. **DOES flagship-c3'S ADVANTAGE SURVIVE THE REDUNDANCY CORRECTION?** Its "wins 17 of 45
   cells" (and my 17 of 42) is computed over axes with effective dof ~4. Recount its wins per
   CORRELATION CLUSTER instead of per axis, and check whether the advantage is carried by one
   cluster (in which case it is one fact repeated) or spread across clusters (in which case it
   is broad competence). This is the decisive test of the steelman.

3. **IS THERE A PATHOLOGY ASYMMETRY?** For each candidate, its worst normalized cell and
   whether any cell is catastrophically bad (below the field's 5th percentile over all cells).
   A "never worst anywhere" argument is only as good as the worst case it avoids.

MODELED ONLY. No claim about realized typing speed; the WPM-equivalent conversion in part 1 is
a UNIT CHANGE on a model output, not a prediction about a person.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CAND = [
    "keybo-lsb",
    "keybo-lsb+lm",
    "lsb-sib",
    "archive-1843",
    "archive-1846",
    "flagship-c3",
]
CORPORA = ["iweb", "blend-v1", "blend-v1-no-anchor"]


def main() -> None:
    rob = json.loads(Path("/tmp/flagship-work/robustness.json").read_text())
    boards = {c: json.loads(Path(f"/tmp/flagship-work/b2-{c}.json").read_text()) for c in CORPORA}
    out: dict = {
        "what": "steelman of flagship-c3 against the keybo-lsb recommendation",
        "modeled_only": "unit conversions on model outputs; not predictions about a person",
    }

    # ---- 1: what does the gap buy -------------------------------------------------------
    gap = {}
    for c in CORPORA:
        rs = boards[c]["rows"]
        lsb = rs["keybo-lsb"]["time"]["ms_per_char"]
        c3 = rs["flagship-c3"]["time"]["ms_per_char"]
        # the surface's own convention: target_wpm 90 <-> these ms/char values.
        # equivalent WPM under 5 chars/word: wpm = 60000 / (5 * ms_per_char)
        wpm_lsb = 60000.0 / (5 * lsb)
        wpm_c3 = 60000.0 / (5 * c3)
        gap[c] = {
            "keybo_lsb_ms_per_char": lsb,
            "flagship_c3_ms_per_char": c3,
            "delta_ms_per_char": c3 - lsb,
            "delta_pct": 100.0 * (c3 - lsb) / c3,
            "equiv_wpm_keybo_lsb": wpm_lsb,
            "equiv_wpm_flagship_c3": wpm_c3,
            "equiv_wpm_gain": wpm_lsb - wpm_c3,
            "seconds_saved_per_1000_words": (c3 - lsb) * 5 * 1000 / 1000.0,
            "minutes_saved_per_100k_words": (c3 - lsb) * 5 * 100_000 / 1000.0 / 60.0,
        }
    out["speed_gap_in_user_units"] = gap
    out["speed_gap_reading"] = (
        "the gap is ~0.13% of total typing time. Expressed as WPM-equivalent it is ~0.02 WPM "
        "on a ~47 WPM-equivalent surface. This is NOT a user-perceptible difference and must "
        "not be sold as one; it is a MODEL ORDERING, which is a different thing."
    )

    # ---- 2: wins per cluster, not per axis ----------------------------------------------
    clusters = rob["axis_redundancy"]["clusters"]
    live = [g for cl in clusters for g in cl]
    direction = rob["direction_derived_from_qwerty_is_worst"]
    per_axis_wins = dict.fromkeys(CAND, 0)
    per_cluster_wins = {n: dict.fromkeys(range(len(clusters)), 0) for n in CAND}
    axis_detail = {}
    for c in CORPORA:
        for g in live:
            vals = {n: boards[c]["rows"][n]["gauges"][g] for n in CAND}
            best = (
                min(vals, key=vals.get)
                if direction[g] == "higher_is_worse"
                else max(vals, key=vals.get)
            )
            per_axis_wins[best] += 1
            ci = next(i for i, cl in enumerate(clusters) if g in cl)
            per_cluster_wins[best][ci] += 1
            axis_detail[f"{c}|{g}"] = best
    # a cluster is "won" by whoever wins the most of its cells
    cluster_owner = {}
    for ci, cl in enumerate(clusters):
        tally = {n: per_cluster_wins[n][ci] for n in CAND}
        cluster_owner[f"cluster{ci}:{'+'.join(cl)}"] = {
            "tally": tally,
            "owner": max(tally, key=tally.get),
            "n_cells": len(cl) * len(CORPORA),
        }
    owner_counts = dict.fromkeys(CAND, 0)
    for v in cluster_owner.values():
        owner_counts[v["owner"]] += 1
    out["wins_per_axis_vs_per_cluster"] = {
        "n_live_axes": len(live),
        "n_clusters": len(clusters),
        "effective_dof": rob["axis_redundancy"]["effective_dof_participation_ratio"],
        "per_axis_wins_of_42": per_axis_wins,
        "per_cluster_cell_wins": {n: per_cluster_wins[n] for n in CAND},
        "cluster_owner": cluster_owner,
        "clusters_owned": owner_counts,
        "reading": (
            "per-axis win counts over-weight whichever cluster has the most member axes. "
            "clusters_owned is the redundancy-corrected version of the same statement."
        ),
    }

    # ---- 3: pathology asymmetry ---------------------------------------------------------
    r = rob["robustness_with_anchor"]["per_layout"]
    allv = []
    norm_cells: dict[str, dict[str, float]] = {}
    for c in CORPORA:
        for g in live:
            vals = {n: boards[c]["rows"][n]["gauges"][g] for n in CAND + ["qwerty30m"]}
            if direction[g] == "higher_is_worse":
                best, worst = min(vals.values()), max(vals.values())
            else:
                best, worst = max(vals.values()), min(vals.values())

            rng = worst - best
            cell = {n: (1.0 if rng == 0 else (worst - v) / rng) for n, v in vals.items()}
            norm_cells[f"{c}|{g}"] = cell
            allv.extend(cell[n] for n in CAND)
    p5, p10 = float(np.percentile(allv, 5)), float(np.percentile(allv, 10))
    path = {}
    for n in CAND:
        vs = np.array([norm_cells[k][n] for k in norm_cells])
        path[n] = {
            "worst_norm": float(vs.min()),
            "n_below_p5": int((vs < p5).sum()),
            "n_below_p10": int((vs < p10).sum()),
            "cells_below_p5": [k for k in norm_cells if norm_cells[k][n] < p5],
            "mean_norm": float(vs.mean()),
        }
    out["pathology"] = {
        "field_p5": p5,
        "field_p10": p10,
        "per_layout": path,
        "definition": (
            "a 'pathological axis' = a (corpus,gauge) cell where the layout's "
            "ceiling-fraction position falls below the 5th percentile of the whole "
            "6-layout x 42-cell field"
        ),
    }

    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"wrote {sys.argv[1]}\n")

    print("=== 1. what the speed gap buys (keybo-lsb vs flagship-c3) ===")
    for c, g in gap.items():
        print(
            f"  {c:20s} Δ {g['delta_ms_per_char']:.4f} ms/char = {g['delta_pct']:.4f}% of total; "
            f"equiv WPM {g['equiv_wpm_flagship_c3']:.3f} -> {g['equiv_wpm_keybo_lsb']:.3f} "
            f"(+{g['equiv_wpm_gain']:.4f}); {g['minutes_saved_per_100k_words']:.2f} min per 100k words"
        )
    print(f"\n=== 2. wins per axis vs per cluster (eff dof {out['wins_per_axis_vs_per_cluster']['effective_dof']:.2f}) ===")
    print(f"  per-axis wins of 42: {per_axis_wins}")
    print(f"  clusters owned of {len(clusters)}: {owner_counts}")
    for k, v in cluster_owner.items():
        print(f"    {k:52s} owner={v['owner']:14s} tally={v['tally']}")
    print(f"\n=== 3. pathology (field p5={p5:.4f}, p10={p10:.4f}) ===")
    for n in sorted(CAND, key=lambda m: path[m]["worst_norm"]):
        print(
            f"  {n:14s} worst {path[n]['worst_norm']:.4f}  #<p5 {path[n]['n_below_p5']:2d}  "
            f"#<p10 {path[n]['n_below_p10']:2d}  {path[n]['cells_below_p5']}"
        )


if __name__ == "__main__":
    main()
