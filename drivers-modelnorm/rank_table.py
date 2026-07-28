"""MODELNORM-1 deliverables A and B — the candidate table, and does normalizing CHANGE a ranking?

Deliverable A: every candidate on the normalized scale, with the anchors and their uncertainty.

Deliverable B: does normalizing change any ranking versus the raw scores? Answered at three
levels, because "the ranking" is three different objects and conflating them is how this
question gets answered wrongly:

  1. **per model** — normalization is a per-model affine map with positive scale, so it is
     rank-preserving WITHIN a model by construction. Asserted, not assumed: a change here
     would be an implementation bug, and the assertion is what would catch it.
  2. **the aggregate** — this is where a change can be real. The comparison is the normalized
     equal-weight blend versus the two obvious raw aggregates: the raw MEAN of the three
     surfaces' predicted ms, and the raw mean of saved-vs-qwerty percentages. A raw mean is
     dominated by whichever surface has the widest span, which is exactly the scale-break the
     design exists to fix.
  3. **against the resolution floor** — any reordering is only meaningful if it exceeds what
     the instrument can resolve. The PAIRED floor is computed on THIS pool (trap 7 of the
     brief: a paired floor must name its pool; n=8 gave 0.2222 elsewhere, other artifacts
     0.4964 at n=10 and 0.1406 at n=11 — none of those is transferable).

The paired floor here is derived from the only replicate structure these surfaces have: the
THREE MODELS are three measurements of the same underlying quantity, so a candidate's
per-model spread is the nuisance and the paired difference between two candidates ACROSS
models is the signal. That decomposition is reported explicitly (candidate x model two-way),
including the model main effect's share of SS, so the floor is readable rather than borrowed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelnorm_eval as MN  # noqa: E402


def ranking(values: dict[str, float], higher_is_better: bool) -> list[str]:
    return [
        name for name, _ in sorted(
            values.items(), key=lambda kv: (-kv[1] if higher_is_better else kv[1])
        )
    ]


def adjacent_transpositions(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Pairs whose ORDER differs between two rankings (all discordant pairs, not just adjacent
    ones — named honestly: a pair is listed if a puts x before y and b does not)."""
    position_b = {name: index for index, name in enumerate(b)}
    out = []
    for x, y in itertools.combinations(a, 2):
        if position_b[x] > position_b[y]:
            out.append((x, y))
    return out


def two_way_decomposition(matrix: np.ndarray) -> dict:
    """Candidate x model two-way ANOVA-style decomposition of a (C, M) value matrix.

    Reports each effect's share of total SS. The MODEL main effect is the nuisance that a
    paired comparison cancels; the CANDIDATE main effect is the signal; the interaction is
    what a paired comparison cannot cancel and therefore what actually limits resolution.
    """
    grand = matrix.mean()
    candidate_effect = matrix.mean(axis=1) - grand
    model_effect = matrix.mean(axis=0) - grand
    interaction = matrix - grand - candidate_effect[:, None] - model_effect[None, :]
    ss_candidate = float(matrix.shape[1] * (candidate_effect**2).sum())
    ss_model = float(matrix.shape[0] * (model_effect**2).sum())
    ss_interaction = float((interaction**2).sum())
    total = ss_candidate + ss_model + ss_interaction
    return {
        "ss_candidate": ss_candidate,
        "ss_model": ss_model,
        "ss_interaction": ss_interaction,
        "ss_total": total,
        "model_main_effect_share_of_ss": ss_model / total,
        "candidate_main_effect_share_of_ss": ss_candidate / total,
        "interaction_share_of_ss": ss_interaction / total,
        "residual_sd_after_removing_both_main_effects": float(interaction.std(ddof=0)),
    }


def paired_floor(matrix: np.ndarray, names: list[str]) -> dict:
    """The PAIRED resolution floor on THIS pool, in the units of ``matrix``.

    For every candidate pair, the difference is measured once per model; the SPREAD of those
    three per-model differences is the pair's own uncertainty. The floor is the max such
    spread over pairs (conservative) alongside the median (typical). Reported with the pool
    named, because a paired floor is a property of its pool and is not transferable.
    """
    spreads = {}
    for i, j in itertools.combinations(range(matrix.shape[0]), 2):
        differences = matrix[i] - matrix[j]
        spreads[f"{names[i]} vs {names[j]}"] = {
            "mean_difference": float(differences.mean()),
            "per_model_differences": differences.tolist(),
            "spread_max_minus_min": float(differences.max() - differences.min()),
            "sd": float(differences.std(ddof=1)),
            "sign_agrees_across_all_three_models": bool(
                np.all(differences > 0) or np.all(differences < 0)
            ),
        }
    values = np.array([v["spread_max_minus_min"] for v in spreads.values()])
    sds = np.array([v["sd"] for v in spreads.values()])
    return {
        "pool": f"the {matrix.shape[0]} candidates of this arm, measured on "
                f"{matrix.shape[1]} native surfaces (blend-v1, 90 WPM baked)",
        "n_candidates": int(matrix.shape[0]),
        "n_pairs": len(spreads),
        "floor_conservative_max_pair_spread": float(values.max()),
        "floor_median_pair_spread": float(np.median(values)),
        "floor_max_pair_sd": float(sds.max()),
        "floor_median_pair_sd": float(np.median(sds)),
        "pairs_whose_sign_agrees_on_all_three_models": int(
            sum(1 for v in spreads.values() if v["sign_agrees_across_all_three_models"])
        ),
        "per_pair": spreads,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra", action="append", default=[],
                        help="NAME=LAYOUT rows to append (e.g. a blend champion)")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    surf = MN.NativeSurfaces(corpus=args.corpus)
    anchors = MN.load_anchors(Path(args.anchors))
    normalizer = MN.BlendNormalizer(anchors)

    layouts = dict(MN.CANDIDATES)
    for spec in args.extra:
        name, _, layout = spec.partition("=")
        layouts[name] = layout

    names = list(layouts)
    fits = np.stack([surf.fit_of_layout(layouts[n]) for n in names])       # (C,3) predicted ms
    normalized = normalizer.normalize(fits)                                # (C,3) 1=best
    qwerty_fit = surf.fit_of_layout(MN.CANDIDATES["qwerty30m"])
    saved = 100.0 * (1.0 - fits / qwerty_fit)                              # (C,3) saved% vs qwerty

    # ---- 1. per-model rankings, raw vs normalized ----
    per_model = {}
    for index, model in enumerate(MN.MODELS):
        raw = {n: float(fits[i, index]) for i, n in enumerate(names)}
        nor = {n: float(normalized[i, index]) for i, n in enumerate(names)}
        raw_rank = ranking(raw, higher_is_better=False)   # lower ms is better
        norm_rank = ranking(nor, higher_is_better=True)   # higher normalized is better
        per_model[model] = {
            "raw_ms": raw,
            "normalized": nor,
            "raw_ranking_best_first": raw_rank,
            "normalized_ranking_best_first": norm_rank,
            "rankings_identical": raw_rank == norm_rank,
            "discordant_pairs": adjacent_transpositions(raw_rank, norm_rank),
        }
        assert raw_rank == norm_rank, (
            f"{model}: normalization changed a WITHIN-MODEL ranking. It is a positive-scale "
            "affine map and cannot; this is an implementation bug.\n"
            f"  raw : {raw_rank}\n  norm: {norm_rank}"
        )

    # ---- 2. the aggregate: normalized blend vs the raw aggregates ----
    blend = {n: float(normalizer.blend(fits[i])) for i, n in enumerate(names)}
    raw_mean_ms = {n: float(fits[i].mean()) for i, n in enumerate(names)}
    raw_mean_saved = {n: float(saved[i].mean()) for i, n in enumerate(names)}
    raw_min_saved = {n: float(saved[i].min()) for i, n in enumerate(names)}

    blend_rank = ranking(blend, higher_is_better=True)
    aggregates = {
        "raw_mean_predicted_ms": (raw_mean_ms, False),
        "raw_mean_saved_vs_qwerty_pct": (raw_mean_saved, True),
        "raw_min_saved_vs_qwerty_pct_the_scale_broken_floor": (raw_min_saved, True),
    }
    comparisons = {}
    for label, (values, higher) in aggregates.items():
        other = ranking(values, higher_is_better=higher)
        discordant = adjacent_transpositions(blend_rank, other)
        comparisons[label] = {
            "ranking_best_first": other,
            "values": values,
            "identical_to_normalized_blend": other == blend_rank,
            "discordant_pairs_vs_normalized_blend": discordant,
            "n_discordant_pairs": len(discordant),
        }

    # ---- 3. does any change clear the resolution floor? ----
    # Floor on the NORMALIZED scale (the scale the blend decides on), and on saved% for a
    # cross-artifact-readable number.
    floor_normalized = paired_floor(normalized, names)
    floor_saved = paired_floor(saved, names)
    decomposition_normalized = two_way_decomposition(normalized)
    decomposition_saved = two_way_decomposition(saved)

    # for each discordant pair, is the blend gap bigger than the floor?
    verdicts = {}
    for label, comparison in comparisons.items():
        rows = []
        for x, y in comparison["discordant_pairs_vs_normalized_blend"]:
            gap = abs(blend[x] - blend[y])
            rows.append({
                "pair": [x, y],
                "blend_gap": gap,
                "floor_conservative": floor_normalized["floor_conservative_max_pair_spread"],
                "floor_this_pair_spread": floor_normalized["per_pair"].get(
                    f"{x} vs {y}", floor_normalized["per_pair"].get(f"{y} vs {x}", {})
                ).get("spread_max_minus_min"),
                "clears_conservative_floor": bool(
                    gap > floor_normalized["floor_conservative_max_pair_spread"]
                ),
            })
        verdicts[label] = rows

    # ---- the design-defect measurement: how much of [0,1] do real candidates occupy? ----
    occupancy = {
        model: {
            "min_normalized": float(normalized[:, i].min()),
            "max_normalized": float(normalized[:, i].max()),
            "window_width": float(normalized[:, i].max() - normalized[:, i].min()),
        }
        for i, model in enumerate(MN.MODELS)
    }
    blend_values = np.array(list(blend.values()))

    blob = {
        "what": "MODELNORM-1 deliverables A and B: the candidate table on the normalized "
                "scale, and whether normalizing changes any ranking",
        "identity": surf.identity(),
        "anchors": {
            "zero": anchors.zero, "one": anchors.one,
            "zero_statistic": anchors.zero_statistic, "zero_n": anchors.zero_n,
            "zero_seed": anchors.zero_seed, "zero_sd": anchors.zero_sd,
            "one_provenance": anchors.one_provenance,
            "span": {m: anchors.zero[m] - anchors.one[m] for m in MN.MODELS},
        },
        "layouts": layouts,
        "table": [
            {
                "name": name,
                "layout": layouts[name],
                "raw_ms": {m: float(fits[i, j]) for j, m in enumerate(MN.MODELS)},
                "saved_vs_qwerty30m_pct": {m: float(saved[i, j]) for j, m in enumerate(MN.MODELS)},
                "normalized": {m: float(normalized[i, j]) for j, m in enumerate(MN.MODELS)},
                "equal_weight_blend": blend[name],
            }
            for i, name in enumerate(names)
        ],
        "B1_per_model_rankings": per_model,
        "B1_verdict": (
            "normalization changes NO within-model ranking, and cannot: it is a per-model "
            "affine map with positive scale. Asserted in code."
        ),
        "B2_aggregate_rankings": {
            "normalized_equal_weight_blend_best_first": blend_rank,
            "normalized_equal_weight_blend_values": blend,
            "compared_against": comparisons,
        },
        "B3_against_the_resolution_floor": {
            "floor_on_the_normalized_scale": floor_normalized,
            "floor_on_saved_vs_qwerty_pct": floor_saved,
            "two_way_decomposition_normalized": decomposition_normalized,
            "two_way_decomposition_saved_pct": decomposition_saved,
            "per_comparison_verdicts": verdicts,
            "note": (
                "the paired floor is computed on THIS pool and is NOT transferable — a paired "
                "floor must name its pool. The replicate structure used is the three models as "
                "three measurements of one quantity, so the MODEL main effect is the nuisance "
                "that a paired difference cancels and the candidate x model INTERACTION is what "
                "actually limits resolution."
            ),
        },
        "design_defect_scale_occupancy": {
            "per_model": occupancy,
            "blend_min": float(blend_values.min()),
            "blend_max": float(blend_values.max()),
            "blend_window_width": float(blend_values.max() - blend_values.min()),
            "why_this_matters": (
                "the '0' anchor is the mean of RANDOM layouts, and random layouts are 2.5-3 sd "
                "worse than qwerty, so the [0,1] scale spends most of its range on a region no "
                "candidate occupies. A narrow occupancy window is a property of the ANCHORING, "
                "not of the layouts."
            ),
        },
        "modelled_only": (
            "MODELLED ONLY: fitted surfaces on the .native frame at a BAKED 90 WPM; tau "
            "saturated at 1.0, Phase-D cancelled. No layout is promoted or adopted."
        ),
    }
    Path(args.out).write_text(json.dumps(blob, indent=1))
    print(f"WROTE {args.out}")

    width = max(len(n) for n in names) + 1
    print("\n== normalized scale (1 = per-model optimum, 0 = random-pool mean) "
          "| corpus blend-v1, .native, 90 WPM ==")
    print(f"  {'name':{width}s} " + " ".join(f"{m:>10s}" for m in MN.MODELS) + "   blend(1,1,1)")
    for i, name in enumerate(names):
        print(f"  {name:{width}s} " + " ".join(f"{normalized[i, j]:10.6f}" for j in range(3))
              + f"   {blend[name]:.6f}")
    print("\n== B: does normalizing change a ranking? ==")
    print("  per model: NO (affine, positive scale — asserted)")
    for label, comparison in comparisons.items():
        print(f"  vs {label}: {comparison['n_discordant_pairs']} discordant pair(s)")
        for x, y in comparison["discordant_pairs_vs_normalized_blend"]:
            print(f"      blend puts {x} > {y}; that aggregate disagrees "
                  f"(blend gap {abs(blend[x] - blend[y]):.6f})")
    print(f"\n  paired floor on the normalized scale (pool = these {len(names)} candidates x 3 models):")
    print("      conservative (max pair spread) = "
          f"{floor_normalized['floor_conservative_max_pair_spread']:.6f}")
    print(f"      median pair spread             = {floor_normalized['floor_median_pair_spread']:.6f}")
    print("      pairs whose sign agrees on all 3 models = "
          f"{floor_normalized['pairs_whose_sign_agrees_on_all_three_models']}/{floor_normalized['n_pairs']}")
    print(f"      model main effect = {100*decomposition_normalized['model_main_effect_share_of_ss']:.2f}% of SS; "
          f"interaction = {100*decomposition_normalized['interaction_share_of_ss']:.2f}%")
    print(f"\n  scale occupancy: blend window = {blob['design_defect_scale_occupancy']['blend_window_width']:.6f} "
          "of the 0-1 range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
