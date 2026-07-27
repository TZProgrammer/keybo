"""FLAGSHIP-1 — robustness: worst rank and worst normalized position per candidate.

Question 4 of the brief: "a layout that is never WORST anywhere may be a better flagship
than one that is best somewhere and bad elsewhere". Quantify it, using ceiling-fraction
normalization (FLOOR-METHODOLOGY-1: a raw min() over incompatible scales silently discards
whole sources).

Three things this driver is careful about, each because getting it wrong has cost the
campaign a number:

* **DIRECTION is derived, never assumed.** For every gauge, the direction is fixed by
  qwerty-is-worst: whichever end qwerty30m sits on is the BAD end. That is the campaign's
  registered method (NO-ANCHOR-1 self-correction: `lower_better` omits the 4 invariant
  axes, and `oxey-style` in the 15-gauge frame is LOWER-better while the four COMMUNITY
  gauges oxey1/oxey2/wfd are HIGHER-better — conflating them flips a sign).
* **sfr is dropped, having been PROVEN invariant here, not filtered by variance.** It is a
  permutation invariant (counts doubled letters); numpy gives it std 1.9e-14 so a `std>0`
  filter KEEPS it and then rank-correlates noise (trap 23). This driver asserts
  spread == 0.0 exactly over the field and drops it, leaving 14 LIVE corpus-sensitive axes.
* **EFFECTIVE DOF, not axis count.** `oxey-style` is R2=0.9937 on {sfb,lsb,scissor,
  imbalance,redir,alt}; lsb|lsb-dist rho=1.00; `redir` equals the oxeylyzer redirect family.
  So "wins N of 15" overcounts independent evidence ~4x (traps 11/27). This driver reports
  results BOTH per-axis and per-CORRELATION-CLUSTER, and the cluster version is the one to
  read. Clusters come from the observed correlation matrix, not from a hand-list.

Normalization: ceiling-fraction = (worst - x) / (worst - best) per (corpus, gauge) cell,
over a reference population that INCLUDES qwerty30m as the anchor (so 0 = qwerty and 1 =
field-best), because SELECT-MAXIMIN-1 showed that dropping the anchor collapses every
maximin score to exactly 0.0 — every one of the six is field-worst on at least one cell, so
an anchor-free worst-case does not discriminate. Both variants are computed; the degeneracy
is REPORTED, not hidden.

MODELED/gauge only. No claim about realized typing speed.
"""

from __future__ import annotations

import itertools
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
ANCHOR = "qwerty30m"
EXTERNALS = ["graphite", "semimak"]
CORPORA = ["iweb", "blend-v1", "blend-v1-no-anchor"]
BOARD = "/tmp/flagship-work/b2-{}.json"


def load() -> dict:
    return {c: json.loads(Path(BOARD.format(c)).read_text()) for c in CORPORA}


def main() -> None:
    boards = load()
    out: dict = {
        "what": "FLAGSHIP-1 robustness board: worst rank / worst normalized position",
        "normalization": "ceiling-fraction per (corpus,gauge) cell; 0=field-worst, 1=field-best",
        "anchor_note": (
            "SELECT-MAXIMIN-1: with the qwerty anchor DROPPED, every one of the six is "
            "field-worst on >=1 cell, so anchor-free maximin is identically 0.0 for all six "
            "and does not discriminate. Both variants reported."
        ),
        "modeled_only": "gauge/model outputs; tau saturated at 1.0; Phase-D cancelled",
    }

    # ---- live axes: prove sfr invariant rather than variance-filtering it -----------------
    all_gauges = list(boards["iweb"]["rows"]["keybo-lsb"]["gauges"])
    field = CAND + [ANCHOR] + EXTERNALS
    invariance = {}
    for g in all_gauges:
        spreads = []
        for c in CORPORA:
            vals = [boards[c]["rows"][n]["gauges"][g] for n in field]
            spreads.append(max(vals) - min(vals))
        invariance[g] = {"max_spread_over_field": max(spreads)}
    assert invariance["sfr"]["max_spread_over_field"] == 0.0, "sfr not exactly invariant here"
    live = [g for g in all_gauges if invariance[g]["max_spread_over_field"] != 0.0]
    out["gauge_invariance_test"] = {
        "method": "direct spread over the 9-layout field per corpus (NOT a std>0 filter)",
        "spreads": invariance,
        "dropped_as_invariant": [g for g in all_gauges if g not in live],
        "n_live_axes": len(live),
    }
    assert len(live) == 14, f"expected 14 live axes, got {len(live)}"

    # ---- direction from qwerty-is-worst ---------------------------------------------------
    direction = {}
    for g in live:
        # sign such that HIGHER is worse; qwerty must be the max under it
        q = np.mean([boards[c]["rows"][ANCHOR]["gauges"][g] for c in CORPORA])
        others = np.mean(
            [[boards[c]["rows"][n]["gauges"][g] for c in CORPORA] for n in CAND + EXTERNALS]
        )
        direction[g] = "higher_is_worse" if q > others else "lower_is_worse"
    out["direction_derived_from_qwerty_is_worst"] = direction

    # ---- ceiling-fraction normalization ---------------------------------------------------
    def normalize(pop: list[str]) -> dict:
        """cell -> {layout: ceiling fraction in [0,1], 1 = best in pop}."""
        norm: dict[str, dict[str, float]] = {}
        for c in CORPORA:
            for g in live:
                vals = {n: boards[c]["rows"][n]["gauges"][g] for n in pop}
                if direction[g] == "higher_is_worse":
                    best, worst = min(vals.values()), max(vals.values())
                else:
                    best, worst = max(vals.values()), min(vals.values())
                rng = worst - best
                norm[f"{c}|{g}"] = {
                    n: (1.0 if rng == 0 else (worst - v) / rng) for n, v in vals.items()
                }
        return norm

    for tag, pop in (("with_anchor", CAND + [ANCHOR]), ("anchor_free", CAND)):
        norm = normalize(pop)
        cells = sorted(norm)
        per_layout = {}
        for n in CAND:
            vs = np.array([norm[c][n] for c in cells])
            ranks = []
            for c in cells:
                # rank among the SIX candidates only (1 = best), regardless of pop
                order = sorted(CAND, key=lambda m: -norm[c][m])
                ranks.append(order.index(n) + 1)
            ranks = np.array(ranks)
            worst_cell = cells[int(vs.argmin())]
            per_layout[n] = {
                "mean_norm": float(vs.mean()),
                "worst_norm": float(vs.min()),
                "worst_norm_cell": worst_cell,
                "median_norm": float(np.median(vs)),
                "worst_rank_among_6": int(ranks.max()),
                "n_cells_worst_of_6": int((ranks == 6).sum()),
                "n_cells_best_of_6": int((ranks == 1).sum()),
                "mean_rank": float(ranks.mean()),
                "n_cells": len(cells),
                "cells_where_worst": [c for c, r in zip(cells, ranks, strict=True) if r == 6],
            }
        out[f"robustness_{tag}"] = {
            "reference_population": pop,
            "per_layout": per_layout,
            "maximin_winner": max(per_layout, key=lambda n: per_layout[n]["worst_norm"]),
            "mean_winner": max(per_layout, key=lambda n: per_layout[n]["mean_norm"]),
            "degenerate_maximin": bool(
                len({round(per_layout[n]["worst_norm"], 12) for n in CAND}) == 1
            ),
        }

    # ---- correlation clustering: effective dof, not 14 -----------------------------------
    # build the 14 x (6 candidates x 3 corpora) matrix of normalized values and cluster axes
    norm = normalize(CAND + [ANCHOR])
    A = np.array(
        [[norm[f"{c}|{g}"][n] for c in CORPORA for n in CAND] for g in live]
    )  # 14 x 18
    C = np.corrcoef(A)
    # greedy single-link clustering at |rho| >= 0.9
    THRESH = 0.9
    unassigned = set(range(len(live)))
    clusters: list[list[int]] = []
    while unassigned:
        seed = min(unassigned)
        comp = {seed}
        changed = True
        while changed:
            changed = False
            for i in list(unassigned - comp):
                if any(abs(C[i, j]) >= THRESH for j in comp):
                    comp.add(i)
                    changed = True
        clusters.append(sorted(comp))
        unassigned -= comp
    named_clusters = [[live[i] for i in cl] for cl in clusters]
    # effective dof from the eigenvalues of the correlation matrix (participation ratio)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0, None)
    eff_dof = float(ev.sum() ** 2 / (ev**2).sum())
    out["axis_redundancy"] = {
        "threshold_abs_rho": THRESH,
        "clusters": named_clusters,
        "n_clusters": len(named_clusters),
        "n_axes": len(live),
        "effective_dof_participation_ratio": eff_dof,
        "reading": (
            "cluster count (and effective dof) is the honest denominator for a 'wins N of "
            "15' statement; the raw 14/15 overcounts (traps 11/27)"
        ),
        "pairwise_rho_ge_threshold": [
            [live[i], live[j], float(C[i, j])]
            for i, j in itertools.combinations(range(len(live)), 2)
            if abs(C[i, j]) >= THRESH
        ],
    }

    # cluster-level robustness: one score per cluster = mean of its axes' normalized values
    per_layout_cl = {}
    cluster_cells = [(c, ci) for c in CORPORA for ci in range(len(named_clusters))]
    for n in CAND:
        vs = []
        ranks = []
        for c, ci in cluster_cells:
            score = {
                m: float(np.mean([norm[f"{c}|{g}"][m] for g in named_clusters[ci]]))
                for m in CAND
            }
            vs.append(score[n])
            ranks.append(sorted(CAND, key=lambda m: -score[m]).index(n) + 1)
        vs, ranks = np.array(vs), np.array(ranks)
        per_layout_cl[n] = {
            "mean_norm": float(vs.mean()),
            "worst_norm": float(vs.min()),
            "worst_rank_among_6": int(ranks.max()),
            "n_cells_worst_of_6": int((ranks == 6).sum()),
            "n_cells_best_of_6": int((ranks == 1).sum()),
            "mean_rank": float(ranks.mean()),
            "n_cells": len(cluster_cells),
        }
    out["robustness_by_cluster_with_anchor"] = {
        "cells": [f"{c}|cluster{ci}" for c, ci in cluster_cells],
        "clusters": named_clusters,
        "per_layout": per_layout_cl,
        "maximin_winner": max(per_layout_cl, key=lambda n: per_layout_cl[n]["worst_norm"]),
        "mean_winner": max(per_layout_cl, key=lambda n: per_layout_cl[n]["mean_norm"]),
    }

    dest = Path(sys.argv[1])
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}\n")

    print(f"live axes: {len(live)} (dropped {out['gauge_invariance_test']['dropped_as_invariant']})")
    print(
        f"clusters at |rho|>=0.9: {len(named_clusters)}  effective dof {eff_dof:.2f}\n"
        f"  {named_clusters}\n"
    )
    for tag in ("with_anchor", "anchor_free"):
        r = out[f"robustness_{tag}"]
        print(f"=== {tag} (degenerate_maximin={r['degenerate_maximin']}) ===")
        print(
            f"{'layout':14s} {'mean':>7s} {'worst':>7s} {'wrank':>6s} {'#worst':>7s} "
            f"{'#best':>6s} {'meanrank':>9s}  worst cell"
        )
        for n in sorted(CAND, key=lambda m: -r["per_layout"][m]["mean_norm"]):
            p = r["per_layout"][n]
            print(
                f"{n:14s} {p['mean_norm']:7.4f} {p['worst_norm']:7.4f} "
                f"{p['worst_rank_among_6']:6d} {p['n_cells_worst_of_6']:7d} "
                f"{p['n_cells_best_of_6']:6d} {p['mean_rank']:9.3f}  {p['worst_norm_cell']}"
            )
        print()
    print("=== by CLUSTER (the honest denominator), with anchor ===")
    print(f"{'layout':14s} {'mean':>7s} {'worst':>7s} {'wrank':>6s} {'#worst':>7s} {'#best':>6s} {'meanrank':>9s}")
    r = out["robustness_by_cluster_with_anchor"]
    for n in sorted(CAND, key=lambda m: -r["per_layout"][m]["mean_norm"]):
        p = r["per_layout"][n]
        print(
            f"{n:14s} {p['mean_norm']:7.4f} {p['worst_norm']:7.4f} {p['worst_rank_among_6']:6d} "
            f"{p['n_cells_worst_of_6']:7d} {p['n_cells_best_of_6']:6d} {p['mean_rank']:9.3f}"
        )


if __name__ == "__main__":
    main()
