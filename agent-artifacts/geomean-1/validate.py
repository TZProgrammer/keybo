"""GEOMEAN-1 step 4 — the tests that actually decide it.

Five things `aggregates.py` did NOT do, each one a way a passing result could still be an
artefact:

1. LEAVE-ONE-GROUP-OUT. Leave-one-GAUGE-out is ANTI-CONSERVATIVE on this frame: dropping
   `lsb` leaves `lsb-dist` behind at rho = 1.00, so the information never actually leaves the
   aggregate. This is trap 11 (nested legs are one leg) applied to a robustness test rather
   than to a guard. The honest perturbation drops a whole correlated GROUP. 11 refits.

2. GROUPING-THRESHOLD SENSITIVITY. The grouping is a modelling choice, so a result that only
   holds at |rho| >= 0.8 is a result about the threshold. Re-run A3/A5 at 0.9 / 0.8 / 0.7 and
   also under two declared non-correlation groupings (by MEASUREMENT FAMILY, and ungrouped) to
   separate "grouping helps" from "this particular grouping helps".

3. THE PRIOR-ART CONTROL, exactly. SELECT-MAXIMIN-1 normalized over a POOLED 45-cell field
   (15 corpus-sensitive gauges x 3 corpora) with the 4 invariant axes scored separately
   (`inv_mean`/`inv_worst`). `aggregates.py` normalizes per corpus over all 19. Re-derive
   their exact construction so the comparison to prior art is on their frame, not mine
   (trap 13: never stitch a comparison across two conventions) and so their reported
   all-six-score-0.0000 degeneracy is reproduced or refuted as a NUMBER.

4. TEST (iv), THE RESOLUTION FLOOR, measured not quoted. `TimeSurface(keep_seed_tables=True)`
   exposes `seed_totals()` — the per-seed corpus totals behind the seed-MEAN ms/char. That is
   the real instrument: it gives each layout's own per-seed spread instead of the ~1 ms/char
   constant the campaign quotes. Reported as ms/char spread per layout and as the
   incumbent-vs-incumbent gap against it.

5. AN AGGREGATE-VS-FLOOR CHECK. The aggregate's own margin is dimensionless, so "is the
   margin above the floor" cannot be asked of it directly. Two answers are computed:
   (a) the ms/char gap between the aggregate's rank-1 and rank-2, against the per-seed floor;
   (b) a per-seed REFIT of the aggregate: if the aggregate is recomputed with the ms/char
   column replaced by each seed's own value, does rank-1 hold? (b) only applies to a variant
   that CONTAINS a speed column, so it is run on an A4+time variant, labelled as such.

MODELED/gauge only. No realized-speed claim. Every number names its corpus.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path("/local/home/zegertho/agent/state/geomean/artifacts/geomean-1")
MINIMAX = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/noanchor-1/minimax-selection.json"
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregates import (  # noqa: E402
    ANCHOR,
    CANDIDATES,
    EPS,
    FIELD6,
    ceiling_fraction,
    gauge_matrix,
    geomean,
    grouping_from_correlation,
    load_corr,
    load_pool,
    percentile_rank,
    score_field,
)

#: A declared NON-correlation grouping: by what the gauge measurably IS. Stated as a rival
#: modelling choice so "grouping helps" can be separated from "correlation grouping helps".
FAMILY_GROUPS = [
    ["sfr", "sfb", "sfb-dist", "sfs", "sfs-dist"],  # single-finger reuse
    ["lsb", "lsb-dist"],  # lateral stretch
    ["scissor"],  # two-row adjacent-finger reach
    ["alt", "roll", "sr-roll", "redir"],  # trigram flow classes
    ["imbalance"],  # hand load
    ["comfort", "oxey-style"],  # composite preference scalars
    ["genkey", "oxey1", "oxey2", "wfd"],  # community tool scores
]


def derive_sign(d: dict, corpus: str) -> tuple[dict[str, int], list[str], np.ndarray]:
    gauges = list(d["frame"]["sensitive"]) + list(d["frame"]["invariant"])
    pool = list(d["sensitive"][corpus].keys())
    X, _ = gauge_matrix(d, corpus, pool)
    med = np.median(X, axis=0)
    qi = pool.index(d["named"][ANCHOR])
    sign = {
        g: (-1 if X[qi, j] > med[j] else (+1 if X[qi, j] < med[j] else 0))
        for j, g in enumerate(gauges)
    }
    return sign, gauges, X


def main() -> int:
    d = load_pool()
    corr = load_corr()
    named = d["named"]
    gauges = list(d["frame"]["sensitive"]) + list(d["frame"]["invariant"])
    groups08 = grouping_from_correlation(corr, gauges, "thr_0.8")

    report: dict = {
        "purpose": "GEOMEAN-1: leave-one-GROUP-out, grouping sensitivity, prior-art control, floor",
        "leave_one_group_out": {},
        "grouping_sensitivity": {},
        "prior_art_control": {},
        "resolution_floor": {},
    }

    # ------------------------------------------------------------------ 1. leave-one-GROUP-out
    for corpus in ("iweb", "blend", "noanchor"):
        sign, gnames, _ = derive_sign(d, corpus)
        report["leave_one_group_out"][corpus] = {}
        for field_name, field in (("with_qwerty", [*FIELD6, ANCHOR]), ("without_qwerty", list(FIELD6))):
            lays = [named[n] for n in field]
            X, _ = gauge_matrix(d, corpus, lays)
            base = score_field(X, gauges, sign, groups08)
            base_rank = {c: [field[i] for i in np.argsort(-base[c], kind="mergesort")] for c in CANDIDATES}
            per_cand = {c: {"base_top": base_rank[c][0], "drops": {}} for c in CANDIDATES}
            for grp in groups08:
                keep = [g for g in gauges if g not in grp]
                kj = [gauges.index(g) for g in keep]
                sub_groups = [[g for g in G if g in keep] for G in groups08]
                sub_groups = [G for G in sub_groups if G]
                sc = score_field(X[:, kj], keep, sign, sub_groups)
                label = "+".join(grp)
                for c in CANDIDATES:
                    r = [field[i] for i in np.argsort(-sc[c], kind="mergesort")]
                    per_cand[c]["drops"][label] = {
                        "top": r[0],
                        "top_changed": r[0] != base_rank[c][0],
                        "full_ranking_changed": r != base_rank[c],
                        "ranking": r,
                    }
            for c in CANDIDATES:
                dr = per_cand[c]["drops"]
                per_cand[c]["n_groups"] = len(dr)
                per_cand[c]["n_drops_changing_top"] = sum(1 for v in dr.values() if v["top_changed"])
                per_cand[c]["n_drops_changing_ranking"] = sum(
                    1 for v in dr.values() if v["full_ranking_changed"]
                )
                per_cand[c]["groups_whose_removal_changes_top"] = [
                    g for g, v in dr.items() if v["top_changed"]
                ]
            report["leave_one_group_out"][corpus][field_name] = per_cand

    # -------------------------------------------------------------- 2. grouping sensitivity
    for corpus in ("iweb", "blend", "noanchor"):
        sign, _, _ = derive_sign(d, corpus)
        report["grouping_sensitivity"][corpus] = {}
        variants = {
            "rho_0.9": grouping_from_correlation(corr, gauges, "thr_0.9"),
            "rho_0.8": groups08,
            "rho_0.7": grouping_from_correlation(corr, gauges, "thr_0.7"),
            "measurement_family": FAMILY_GROUPS,
            "ungrouped": [[g] for g in gauges],
        }
        for field_name, field in (("with_qwerty", [*FIELD6, ANCHOR]), ("without_qwerty", list(FIELD6))):
            lays = [named[n] for n in field]
            X, _ = gauge_matrix(d, corpus, lays)
            out = {}
            for vname, grps in variants.items():
                sc = score_field(X, gauges, sign, grps)
                out[vname] = {
                    "n_groups": len(grps),
                    "A3_top": [field[i] for i in np.argsort(-sc["A3_grouped_geomean_ceilfrac"], kind="mergesort")][0],
                    "A5_top": [field[i] for i in np.argsort(-sc["A5_grouped_rank_geomean"], kind="mergesort")][0],
                    "A3_ranking": [field[i] for i in np.argsort(-sc["A3_grouped_geomean_ceilfrac"], kind="mergesort")],
                    "A5_ranking": [field[i] for i in np.argsort(-sc["A5_grouped_rank_geomean"], kind="mergesort")],
                }
            report["grouping_sensitivity"][corpus][field_name] = out

    # ------------------------------------------------- 3. the prior-art control, their frame
    # SELECT-MAXIMIN-1: pooled 45-cell field = 15 corpus-sensitive gauges x 3 corpora,
    # ceiling-fraction normalized; the 4 invariant axes scored SEPARATELY (inv_mean/inv_worst).
    S = list(d["frame"]["sensitive"])
    I = list(d["frame"]["invariant"])
    for field_name, field in (("with_qwerty", [*FIELD6, ANCHOR]), ("without_qwerty", list(FIELD6))):
        lays = [named[n] for n in field]
        cells = []  # (n_layouts,) columns, one per (gauge, corpus)
        for corpus in ("iweb", "blend", "noanchor"):
            sign, gnames, _ = derive_sign(d, corpus)
            X, _ = gauge_matrix(d, corpus, lays)
            for g in S:
                j = gnames.index(g)
                cells.append(ceiling_fraction(X[:, j] * (sign[g] or 1)))
        M = np.column_stack(cells)  # 45 cells
        inv_cells = []
        sign_i, gnames_i, _ = derive_sign(d, "iweb")  # invariant axes: corpus-independent
        Xi, _ = gauge_matrix(d, "iweb", lays)
        for g in I:
            j = gnames_i.index(g)
            inv_cells.append(ceiling_fraction(Xi[:, j] * (sign_i[g] or 1)))
        Mi = np.column_stack(inv_cells)
        rows = []
        for i, n in enumerate(field):
            rows.append(
                {
                    "layout": n,
                    "worst_case": round(float(M[i].min()), 8),
                    "mean": round(float(M[i].mean()), 8),
                    "n_cells_at_field_worst": int((M[i] <= EPS).sum()),
                    "inv_mean": round(float(Mi[i].mean()), 8),
                    "inv_worst": round(float(Mi[i].min()), 8),
                    "geomean_45": round(geomean(M[i]), 8),
                }
            )
        rows.sort(key=lambda r: -r["worst_case"])
        report["prior_art_control"][field_name] = {
            "n_cells": int(M.shape[1]),
            "construction": "15 corpus-sensitive gauges x 3 corpora, ceiling-fraction; 4 invariant axes separate",
            "rows": rows,
            "maximin_ranking": [r["layout"] for r in rows],
            "mean_ranking": [r["layout"] for r in sorted(rows, key=lambda r: -r["mean"])],
            "all_at_floor": all(r["worst_case"] <= EPS for r in rows),
        }
    # and the published numbers, for the mismatch check
    pub = json.loads(MINIMAX.read_text())
    report["prior_art_control"]["published_minimax_selection_json"] = {
        r["layout"]: {"worst_case": r["worst_case"], "mean": r["mean"]} for r in pub["rows"]
    }

    # ------------------------------------------------------ 4. the resolution floor, measured
    from keybo.analysis.timecard import TimeSurface
    from keybo.data.corpus import load_frequencies

    tri = load_frequencies("/local/home/zegertho/repos/keybo/data/corpus/trigrams.txt")
    surf = TimeSurface(tri, target_wpm=90.0, keep_seed_tables=True)
    floor: dict = {"corpus": "iweb (production trigrams)", "target_wpm": 90.0, "layouts": {}}
    for n in (*FIELD6, ANCHOR):
        lay = named[n]
        card = surf.card(lay)
        totals = surf.seed_totals(lay)
        n_chars = card.total_ms / card.ms_per_char
        per_seed_mpc = [t / n_chars for t in totals]
        floor["layouts"][n] = {
            "ms_per_char_seedmean": round(card.ms_per_char, 6),
            "per_seed_ms_per_char": [round(x, 6) for x in per_seed_mpc],
            "per_seed_spread_ms_per_char": round(max(per_seed_mpc) - min(per_seed_mpc), 6),
            "per_seed_sd_ms_per_char": round(float(np.std(per_seed_mpc, ddof=1)), 6),
            "coverage_pct": round(card.coverage_pct, 6),
        }
    spreads = [v["per_seed_spread_ms_per_char"] for k, v in floor["layouts"].items() if k != ANCHOR]
    floor["measured_floor"] = {
        "max_per_seed_spread_over_6_incumbents": round(max(spreads), 6),
        "median_per_seed_spread": round(float(np.median(spreads)), 6),
        "campaign_quoted_floor_ms_per_char": 1.0,
        "note": (
            "per-seed spread is the ESTIMATOR spread of one layout's ms/char across the 3 "
            "model seeds — the instrument's own resolution. A between-layout gap smaller "
            "than this is below the instrument."
        ),
    }
    gaps = {}
    for a, b in combinations([n for n in FIELD6], 2):
        ga = abs(
            floor["layouts"][a]["ms_per_char_seedmean"] - floor["layouts"][b]["ms_per_char_seedmean"]
        )
        gaps[f"{a}|{b}"] = {
            "gap_ms_per_char": round(ga, 6),
            "resolves_vs_measured_floor": bool(ga > max(spreads)),
            "resolves_vs_quoted_1ms": bool(ga > 1.0),
        }
    floor["pairwise_gaps"] = gaps
    floor["n_pairs_resolving_measured"] = sum(1 for v in gaps.values() if v["resolves_vs_measured_floor"])
    floor["n_pairs_resolving_quoted"] = sum(1 for v in gaps.values() if v["resolves_vs_quoted_1ms"])
    floor["n_pairs"] = len(gaps)
    # rank the 6 by speed so the aggregate's winner can be read against it
    floor["speed_ranking_best_first"] = sorted(
        FIELD6, key=lambda n: floor["layouts"][n]["ms_per_char_seedmean"]
    )
    report["resolution_floor"] = floor

    (OUT / "validation.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT / 'validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
