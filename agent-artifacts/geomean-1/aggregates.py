"""GEOMEAN-1 step 3 — the candidate aggregates, and the four validation tests.

Six candidates, each stating its ORIENTATION, TRANSFORM, WEIGHTING and NORMALIZATION
explicitly (problems (a)-(e) of the brief), plus the two pieces of prior art:

  A0  PLAIN-GEOMEAN (the baseline to beat, and the naive version the user already granted
      is broken). Orientation: derived. Transform: ceiling-fraction to (0,1]. Weights: equal
      over all 19 gauges. Normalization: min-max within the scoring field.
  A1  MEAN-CEILING (SELECT-MAXIMIN-1's `mean` column — PRIOR ART, re-derived here so the
      comparison is on one frame, not stitched across two boards: trap 13).
  A2  MAXIMIN (SELECT-MAXIMIN-1's `worst_case` column — PRIOR ART, the DEGENERATE one).
  A3  GROUPED-GEOMEAN: the fix for problem (d). Gauges are first collapsed into
      correlation-derived GROUPS (the |rho| >= 0.8 clusters from correlation.json, computed
      on the archive-only sub-pool, iWeb — one fixed grouping used for every corpus so the
      grouping is not refit per test), each group is reduced to its geometric mean, and the
      aggregate is the geometric mean OVER GROUPS. Weights: equal PER GROUP, which is the
      modelling claim ("the frame measures ~5 constructs; weight constructs, not columns").
  A4  RANK-GEOMEAN: transform = within-field percentile rank mapped to (0,1]. Kills the
      scale problem (c) and the negative-value problem (b) by construction, and is invariant
      to any monotone reparameterization of a gauge — the strongest defence against
      "the transform IS the modelling decision". Weights: equal per group (as A3).
  A5  GROUPED-RANK-GEOMEAN-QFREE: A4 + A3's grouping + the field EXCLUDES qwerty by
      construction, so the "does it survive dropping the anchor" question cannot arise for
      it. This is the candidate I actually expect to be least artefactual.

TRANSFORMS (problem (b): several gauges are large NEGATIVE numbers; oxey-style goes negative
for good layouts; scissor approaches 0):
  * ceiling-fraction: x -> (x - worst) / (best - worst) mapped into [eps, 1]. Sign-safe for
    negatives because it is affine in x and anchored on the FIELD's own worst/best. Requires
    a field; the field choice IS the with/without-qwerty test.
  * percentile-rank: x -> (rank + 1) / (n + 1) in (0,1). Sign-free, scale-free, monotone-
    reparameterization-invariant. Cannot be gamed by a heavy tail.
Both map into (0,1] so a geometric mean is defined and positive (problem (b) solved rather
than declared).

THE FOUR VALIDATION TESTS (deliverable 3):
  (i)   both normalizations: field = {6 incumbents + qwerty} vs field = {6 incumbents}.
        SELECT-MAXIMIN-1 failed exactly here.
  (ii)  leave-one-gauge-out: 19 refits, one per dropped gauge. Does rank-1 hold?
  (iii) three corpora: iWeb / blend-v1 / blend-v1-no-anchor.
  (iv)  margin vs the ~1 ms/char resolution floor, using the REAL per-seed instrument
        (`TimeSurface(keep_seed_tables=True).seed_totals`) rather than a quoted constant.

MODELED/gauge only. Every number names its corpus.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path("/local/home/zegertho/agent/state/geomean/artifacts/geomean-1")
EPS = 1e-6

#: The 6 incumbents/candidates the campaign selects among, plus the qwerty anchor.
FIELD6 = (
    "keybo-lsb", "keybo-lsb+lm", "lsb-sib", "archive-1843", "archive-1846", "flagship-c3",
)
ANCHOR = "qwerty"


def load_pool() -> dict:
    return json.loads((OUT / "pool_gauges.json").read_text())


def load_corr() -> dict:
    return json.loads((OUT / "correlation.json").read_text())


def gauge_matrix(d: dict, corpus: str, layouts: list[str]) -> tuple[np.ndarray, list[str]]:
    S, I = d["frame"]["sensitive"], d["frame"]["invariant"]
    rows = [
        list(d["sensitive"][corpus][lay]) + [d["invariant"][lay][g] for g in I]
        for lay in layouts
    ]
    return np.asarray(rows, float), list(S) + list(I)


# --------------------------------------------------------------------------- transforms
def ceiling_fraction(col: np.ndarray) -> np.ndarray:
    """(x - field worst) / (field best - field worst), clipped into [EPS, 1].

    `col` is ALREADY oriented so higher = better, so worst = min and best = max. A
    degenerate column (all equal) maps to all-1.0: it carries no information, and mapping it
    to 1.0 rather than 0 keeps it a multiplicative no-op in a geometric mean instead of
    zeroing every layout (which is how a maximin over a field-worst-somewhere frame
    collapses to 0.0000 for everyone).
    """
    lo, hi = col.min(), col.max()
    if hi == lo:
        return np.ones_like(col)
    return np.clip((col - lo) / (hi - lo), EPS, 1.0)


def percentile_rank(col: np.ndarray) -> np.ndarray:
    """(rank + 1) / (n + 1) in (0,1); higher = better. Ties get the average rank."""
    order = np.argsort(col, kind="mergesort")
    ranks = np.empty(len(col), float)
    sc = col[order]
    i = 0
    while i < len(col):
        j = i
        while j + 1 < len(col) and sc[j + 1] == sc[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return (ranks + 1.0) / (len(col) + 1.0)


def geomean(v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted geometric mean via logs (never a product — underflow is real at 19 legs)."""
    v = np.clip(v, EPS, None)
    if w is None:
        w = np.ones(len(v))
    w = w / w.sum()
    return float(np.exp(np.sum(w * np.log(v))))


# --------------------------------------------------------------------------- grouping
def grouping_from_correlation(corr: dict, gauges: list[str], thr: str = "thr_0.8") -> list[list[str]]:
    """The FIXED grouping: |rho| >= 0.8 clusters on the archive-only sub-pool, iWeb.

    Fixed once and reused for every corpus and every leave-one-out refit, so the grouping is
    a declared modelling choice rather than something refit to each test (refitting it would
    let the grouping absorb the very perturbation the test applies).
    """
    groups = corr["clusters"]["iweb"]["archive_only"][thr]
    named = {g for grp in groups for g in grp}
    # constants (sfr) are excluded from the correlation matrix; carry them as their own group
    # so the frame is covered and their inertness is visible rather than silent.
    extra = [[g] for g in gauges if g not in named]
    return [list(g) for g in groups] + extra


# --------------------------------------------------------------------------- aggregates
def score_field(
    X: np.ndarray, gauges: list[str], sign: dict[str, int], groups: list[list[str]]
) -> dict[str, np.ndarray]:
    """Every candidate's score for every row of X (the FIELD is exactly X's rows)."""
    s = np.array([sign[g] if sign[g] else 1 for g in gauges], float)
    Xo = X * s

    CF = np.column_stack([ceiling_fraction(Xo[:, j]) for j in range(Xo.shape[1])])
    PR = np.column_stack([percentile_rank(Xo[:, j]) for j in range(Xo.shape[1])])

    idx = {g: j for j, g in enumerate(gauges)}
    present = [[g for g in grp if g in idx] for grp in groups]
    present = [grp for grp in present if grp]

    def grouped(M: np.ndarray) -> np.ndarray:
        per_group = np.column_stack(
            [np.array([geomean(M[i, [idx[g] for g in grp]]) for i in range(len(M))]) for grp in present]
        )
        return np.array([geomean(per_group[i]) for i in range(len(M))])

    return {
        "A0_plain_geomean_ceilfrac": np.array([geomean(CF[i]) for i in range(len(CF))]),
        "A1_mean_ceilfrac": CF.mean(axis=1),
        "A2_maximin_ceilfrac": CF.min(axis=1),
        "A3_grouped_geomean_ceilfrac": grouped(CF),
        "A4_rank_geomean_flat": np.array([geomean(PR[i]) for i in range(len(PR))]),
        "A5_grouped_rank_geomean": grouped(PR),
    }


CANDIDATES = (
    "A0_plain_geomean_ceilfrac",
    "A1_mean_ceilfrac",
    "A2_maximin_ceilfrac",
    "A3_grouped_geomean_ceilfrac",
    "A4_rank_geomean_flat",
    "A5_grouped_rank_geomean",
)


def rank_of(scores: np.ndarray, layouts: list[str]) -> list[str]:
    """Best -> worst (scores are all higher = better by construction)."""
    return [layouts[i] for i in np.argsort(-scores, kind="mergesort")]


def main() -> int:
    d = load_pool()
    corr = load_corr()
    S, I = d["frame"]["sensitive"], d["frame"]["invariant"]
    gauges = list(S) + list(I)
    named = d["named"]
    groups = grouping_from_correlation(corr, gauges)

    report: dict = {
        "purpose": "GEOMEAN-1: candidate aggregates + validation tests (i)(ii)(iii)",
        "frame": d["frame"],
        "candidates": list(CANDIDATES),
        "grouping": {
            "source": "correlation.json clusters[iweb][archive_only][thr_0.8], FIXED",
            "groups": groups,
            "n_groups": len(groups),
        },
        "fields": {
            "with_qwerty": [*FIELD6, ANCHOR],
            "without_qwerty": list(FIELD6),
        },
        "results": {},
        "leave_one_gauge_out": {},
        "pool_wide": {},
    }

    for corpus in ("iweb", "blend", "noanchor"):
        # orientation is derived on the POOL (4367 layouts), never on the 6-7 row field:
        # a 6-row field cannot establish a direction, and deriving it there would let the
        # field's own composition set the sign.
        pool_layouts = list(d["sensitive"][corpus].keys())
        Xpool, gnames = gauge_matrix(d, corpus, pool_layouts)
        assert gnames == gauges
        med = np.median(Xpool, axis=0)
        qi = pool_layouts.index(named[ANCHOR])
        sign = {}
        for j, g in enumerate(gauges):
            sign[g] = -1 if Xpool[qi, j] > med[j] else (+1 if Xpool[qi, j] < med[j] else 0)

        report["results"][corpus] = {}
        for field_name, field in (
            ("with_qwerty", [*FIELD6, ANCHOR]),
            ("without_qwerty", list(FIELD6)),
        ):
            lays = [named[n] for n in field]
            X, _ = gauge_matrix(d, corpus, lays)
            scores = score_field(X, gauges, sign, groups)
            report["results"][corpus][field_name] = {
                cand: {
                    "scores": {n: round(float(scores[cand][i]), 8) for i, n in enumerate(field)},
                    "ranking": [field[i] for i in np.argsort(-scores[cand], kind="mergesort")],
                }
                for cand in CANDIDATES
            }

        # ---- (ii) leave-one-gauge-out, both fields
        report["leave_one_gauge_out"][corpus] = {}
        for field_name, field in (
            ("with_qwerty", [*FIELD6, ANCHOR]),
            ("without_qwerty", list(FIELD6)),
        ):
            lays = [named[n] for n in field]
            X, _ = gauge_matrix(d, corpus, lays)
            base = score_field(X, gauges, sign, groups)
            base_rank = {c: [field[i] for i in np.argsort(-base[c], kind="mergesort")] for c in CANDIDATES}
            per_cand: dict[str, dict] = {c: {"base_top": base_rank[c][0], "drops": {}} for c in CANDIDATES}
            for drop in gauges:
                keep = [g for g in gauges if g != drop]
                kj = [gauges.index(g) for g in keep]
                sc = score_field(X[:, kj], keep, sign, groups)
                for c in CANDIDATES:
                    r = [field[i] for i in np.argsort(-sc[c], kind="mergesort")]
                    per_cand[c]["drops"][drop] = {
                        "top": r[0],
                        "top_changed": r[0] != base_rank[c][0],
                        "full_ranking_changed": r != base_rank[c],
                        "ranking": r,
                    }
            for c in CANDIDATES:
                dr = per_cand[c]["drops"]
                per_cand[c]["n_drops_changing_top"] = sum(1 for v in dr.values() if v["top_changed"])
                per_cand[c]["n_drops_changing_ranking"] = sum(
                    1 for v in dr.values() if v["full_ranking_changed"]
                )
                per_cand[c]["gauges_whose_removal_changes_top"] = [
                    g for g, v in dr.items() if v["top_changed"]
                ]
            report["leave_one_gauge_out"][corpus][field_name] = per_cand

        # ---- a pool-wide field: the aggregate as an OBJECTIVE, not just a 7-row selector.
        # (An aggregate that only exists on a 6-row field is not a selection rule you can
        # search against; the search would renormalize every generation.)
        scores_pool = score_field(Xpool, gauges, sign, groups)
        report["pool_wide"][corpus] = {
            cand: {
                "top10": [
                    {"layout": pool_layouts[i], "score": round(float(scores_pool[cand][i]), 8)}
                    for i in np.argsort(-scores_pool[cand], kind="mergesort")[:10]
                ],
                "incumbent_percentile": {
                    n: round(
                        float(
                            (scores_pool[cand] < scores_pool[cand][pool_layouts.index(named[n])]).mean()
                        ),
                        6,
                    )
                    for n in (*FIELD6, ANCHOR)
                },
                "incumbent_rank": {
                    n: int(
                        1
                        + (scores_pool[cand] > scores_pool[cand][pool_layouts.index(named[n])]).sum()
                    )
                    for n in (*FIELD6, ANCHOR)
                },
            }
            for cand in CANDIDATES
        }

    # ---- cross-corpus agreement of each candidate's 6-field ranking (test iii)
    agree: dict[str, dict] = {}
    for field_name in ("with_qwerty", "without_qwerty"):
        agree[field_name] = {}
        for cand in CANDIDATES:
            ranks = {c: report["results"][c][field_name][cand]["ranking"] for c in ("iweb", "blend", "noanchor")}
            tops = {c: r[0] for c, r in ranks.items()}
            n_inv = {}
            for a, b in combinations(ranks, 2):
                ra, rb = ranks[a], ranks[b]
                n_inv[f"{a}|{b}"] = sum(
                    1
                    for x, y in combinations(ra, 2)
                    if (ra.index(x) < ra.index(y)) != (rb.index(x) < rb.index(y))
                )
            agree[field_name][cand] = {
                "tops": tops,
                "top_stable_across_corpora": len(set(tops.values())) == 1,
                "identical_ranking_across_corpora": len({tuple(r) for r in ranks.values()}) == 1,
                "pairwise_inversions": n_inv,
                "rankings": ranks,
            }
    report["cross_corpus"] = agree

    (OUT / "aggregates.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT / 'aggregates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
