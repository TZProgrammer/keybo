"""MODELNORM-1 — assemble the anchor set of record and TEST its stability (trap 1 + trap 2).

Reads the six per-model searches (3 models x 2 independent seeds, identical budget) plus the
step-1 random-pool artifact, then:

* builds the anchors JSON the blend search consumes — ``zero`` from the random pool, ``one``
  from the per-model optimum;
* measures the "1" anchor's uncertainty as the **seed-to-seed disagreement**, and states it
  as a fraction of the anchor span and against the blend's decision margin. If two seeds
  disagree by more than the decision margin the normalization is NOT stable and this driver
  says so in the artifact rather than quietly proceeding;
* reports each model's convergence evidence — the best-so-far curve, the epoch at which the
  champion was last improved, and how many of the 40 islands independently reached within a
  hair of the champion (restarts agreeing);
* quantifies the "1" anchor **against the prior ceiling-fraction anchoring** on the same
  layouts, which is the specific improvement the user's design claims.

The anchor of record uses the **worse (slower) of the two seeds' optima** per model — the
conservative choice: it cannot flatter a model whose search converged better, and it makes
the anchor a lower bound stated as one (trap 1: the "1" anchor IS a lower bound on the true
optimum, never the optimum).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelnorm_eval as MN  # noqa: E402


def _read_json(path):
    """json.load with the handle closed (ruff SIM115)."""
    with open(path) as handle:
        return json.load(handle)


def _write_json(path, payload):
    """json.dump with the handle closed (ruff SIM115)."""
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


SEEDS = {"s1": 20260728, "s2": 20260901}


def convergence(run: dict) -> dict:
    """Per-model convergence evidence from one search's per-epoch curve + island bests."""
    curve = run["curve"]
    best = run["champion"]["fitness"]
    last_improvement = 0
    previous = float("inf")
    for point in curve:
        if point["best_fit"] < previous - 1e-9:
            last_improvement = point["epoch"]
            previous = point["best_fit"]
    island_bests = np.array([i["best_fit"] for i in run["per_island_best"]])
    return {
        "epochs_run": run["epochs_run"],
        "unique_evals": run["unique_evals"],
        "champion_fitness": best,
        "last_epoch_that_improved_the_champion": last_improvement,
        "epochs_since_last_improvement": run["epochs_run"] - last_improvement,
        "islands": int(island_bests.size),
        "islands_within_0.01pct_of_champion": int((island_bests <= best * 1.0001).sum()),
        "islands_within_0.10pct_of_champion": int((island_bests <= best * 1.001).sum()),
        "island_best_spread_pct_of_champion": float(
            100.0 * (island_bests.max() - island_bests.min()) / abs(best)
        ),
        "best_so_far_curve": [
            {"epoch": p["epoch"], "unique": p["unique"], "best_fit": p["best_fit"]}
            for p in curve
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None, help="directory holding anchor-<MODEL>-<TAG>.json")
    parser.add_argument("--step1", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zero-n", type=int, default=100,
                        help="which random-pool size defines the '0' anchor of record")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs or Path(args.out).parent / "runs")
    surf = MN.NativeSurfaces(corpus=args.corpus)
    step1 = _read_json(args.step1)

    # ---- the "1" anchors, both seeds ----
    per_seed: dict[str, dict[str, dict]] = {}
    for model in MN.MODELS:
        per_seed[model] = {}
        for tag in SEEDS:
            path = runs_dir / f"anchor-{model}-{tag}.json"
            if not path.is_file():
                raise SystemExit(f"missing anchor run {path}")
            run = _read_json(path)
            if run["objective"] != f"solo:{model}":
                raise SystemExit(f"{path} holds objective {run['objective']!r}, expected solo:{model}")
            if run["seed"] != SEEDS[tag]:
                raise SystemExit(f"{path} holds seed {run['seed']}, expected {SEEDS[tag]}")
            # re-derive the champion's fit from the layout string: never trust the recorded
            # fitness without checking it reproduces (trap 20 / "a label is not its referent")
            recomputed = float(surf.fit_of_layout(run["champion"]["layout"])[MN.MODELS.index(model)])
            drift = abs(recomputed - run["champion"]["fitness"]) / abs(recomputed)
            if drift > 1e-12:
                raise SystemExit(
                    f"{path}: champion fitness {run['champion']['fitness']} does not reproduce "
                    f"from its layout ({recomputed}, relative drift {drift:.3e})"
                )
            per_seed[model][tag] = {
                "seed": SEEDS[tag],
                "layout": run["champion"]["layout"],
                "fit": recomputed,
                "budget_requested": run["budget_requested"],
                "unique_evals": run["unique_evals"],
                "islands": run["islands"],
                "epochs_run": run["epochs_run"],
                "convergence": convergence(run),
            }

    # budgets MUST match across models and seeds, else the anchors are not comparable
    budgets = {
        (m, t): (per_seed[m][t]["budget_requested"], per_seed[m][t]["islands"])
        for m in MN.MODELS for t in SEEDS
    }
    if len(set(budgets.values())) != 1:
        raise SystemExit("anchor searches ran at DIFFERENT budgets, so the scales are not "
                         f"comparable: {budgets}")

    # ---- the "0" anchor of record ----
    pool = step1["pools"][str(args.zero_n)]
    zero = pool["mean"]

    # ---- anchor of record: the CONSERVATIVE (slower) of the two seeds ----
    one = {m: max(per_seed[m][t]["fit"] for t in SEEDS) for m in MN.MODELS}
    one_layout = {
        m: max(per_seed[m].values(), key=lambda d: d["fit"])["layout"] for m in MN.MODELS
    }
    best_seen = {m: min(per_seed[m][t]["fit"] for t in SEEDS) for m in MN.MODELS}
    best_layout = {
        m: min(per_seed[m].values(), key=lambda d: d["fit"])["layout"] for m in MN.MODELS
    }

    span = {m: zero[m] - one[m] for m in MN.MODELS}
    seed_gap = {m: abs(per_seed[m]["s1"]["fit"] - per_seed[m]["s2"]["fit"]) for m in MN.MODELS}
    seed_gap_pct_of_span = {m: 100.0 * seed_gap[m] / span[m] for m in MN.MODELS}

    # ---- the prior anchoring, for the comparison the design's claim rests on ----
    ceiling = MN.ceiling_fraction_anchors(surf, MN.CANDIDATES)
    improvement = {
        m: {
            "ceiling_fraction_one_best_of_8_candidates": ceiling[m],
            "search_one_conservative": one[m],
            "search_one_best_of_two_seeds": best_seen[m],
            "search_beats_ceiling_by_ms": ceiling[m] - one[m],
            "search_beats_ceiling_pct_of_search_span": 100.0 * (ceiling[m] - one[m]) / span[m],
            "ceiling_span": zero[m] - ceiling[m],
            "span_ratio_search_over_ceiling": span[m] / (zero[m] - ceiling[m]),
        }
        for m in MN.MODELS
    }

    anchors = MN.Anchors(
        zero=zero, one=one, zero_statistic="mean", zero_n=args.zero_n,
        zero_seed=step1["anchor_of_record"]["seed"], zero_sd=pool["sd"],
        one_provenance={
            "kind": "per-model island memetic search, IDENTICAL budget across models and seeds",
            "statistic": "the SLOWER of two independent seeds (conservative: cannot flatter a "
                         "model whose search converged better)",
            "budget_requested": per_seed[MN.MODELS[0]]["s1"]["budget_requested"],
            "islands": per_seed[MN.MODELS[0]]["s1"]["islands"],
            "seeds": SEEDS,
            "layout_of_record": one_layout,
            "best_layout_seen": best_layout,
            "best_fit_seen": best_seen,
            "is_a_lower_bound_not_the_optimum": (
                "an optimizer output bounds the true optimum from one side only; every "
                "normalized score is therefore an upper bound on the true normalized score"
            ),
        },
    )
    normalizer = MN.BlendNormalizer(anchors)
    direction = MN.assert_direction(surf, normalizer, one_layout)

    # ---- is the normalization STABLE? (trap 1's explicit stop condition) ----
    # The decision margin is what the blend must resolve: the smallest gap between two
    # ADJACENT candidates on the equal-weight blend. If seed-to-seed anchor disagreement
    # exceeds it, the ordering the blend produces is not attributable to the layouts.
    candidate_blend = {
        name: float(normalizer.blend(surf.fit_of_layout(layout)))
        for name, layout in MN.CANDIDATES.items()
    }
    ordered = sorted(candidate_blend.values(), reverse=True)
    decision_margin = float(min(np.abs(np.diff(ordered))))
    # a seed disagreement of g ms on model m perturbs that model's normalized scores by
    # about g/span_m; the blend sees w_m times that.
    perturbation = float(
        sum(normalizer.weights[i] * seed_gap[m] / span[m] for i, m in enumerate(MN.MODELS))
    )
    stable = perturbation < decision_margin

    blob = {
        "what": "MODELNORM-1 anchors of record + their stability evidence",
        "identity": surf.identity(),
        "zero_anchor": {
            "value": zero, "statistic": "mean", "n": args.zero_n,
            "seed": step1["anchor_of_record"]["seed"], "sd": pool["sd"],
            "se": pool["se_of_mean"],
            "n100_vs_n1000_shift_in_se": step1["anchor_movement"][
                "n100_to_n1000_shift_in_se_of_n100"],
        },
        "one_anchor": {
            "value": one, "layout": one_layout,
            "statistic": "slower of two independent seeds (conservative)",
            "best_seen": best_seen, "best_layout": best_layout,
            "per_seed": per_seed,
        },
        "span": span,
        "span_pct_of_zero": {m: 100.0 * span[m] / zero[m] for m in MN.MODELS},
        "stability": {
            "seed_to_seed_gap_ms": seed_gap,
            "seed_to_seed_gap_pct_of_span": seed_gap_pct_of_span,
            "blend_decision_margin": decision_margin,
            "decision_margin_definition": (
                "the smallest gap between two ADJACENT candidates of the 8 on the equal-weight "
                "normalized blend — what the blend must resolve to order them"
            ),
            "anchor_induced_blend_perturbation": perturbation,
            "normalization_is_stable": bool(stable),
            "verdict": (
                "STABLE: seed-to-seed anchor disagreement perturbs the equal-weight blend by "
                f"{perturbation:.6f}, below the {decision_margin:.6f} decision margin"
                if stable else
                "NOT STABLE: seed-to-seed anchor disagreement perturbs the equal-weight blend "
                f"by {perturbation:.6f}, which EXCEEDS the {decision_margin:.6f} decision "
                "margin — the ordering is not attributable to the layouts"
            ),
        },
        "direction_assertions": direction,
        "vs_prior_ceiling_fraction_anchoring": improvement,
        "modelled_only": (
            "MODELLED ONLY: fitted surfaces on the .native frame at a BAKED 90 WPM; tau "
            "saturated at 1.0, Phase-D cancelled. No layout is promoted or adopted."
        ),
    }
    # the anchors JSON the search consumes: a flat, minimal contract
    anchors_out = Path(args.out)
    _write_json(anchors_out, {
        "zero": zero, "one": one, "zero_statistic": "mean", "zero_n": args.zero_n,
        "zero_seed": step1["anchor_of_record"]["seed"], "zero_sd": pool["sd"],
        "one_provenance": anchors.one_provenance,
    })
    _write_json(anchors_out.with_name(anchors_out.stem + "-evidence.json"), blob)

    print(f"WROTE {anchors_out} and {anchors_out.with_name(anchors_out.stem + '-evidence.json')}")
    print("\n== anchors (predicted ms over blend-v1, .native, 90 WPM) ==")
    for m in MN.MODELS:
        print(f"  {m:10s} zero={zero[m]:.6e}  one={one[m]:.6e}  span={span[m]:.6e} "
              f"({100*span[m]/zero[m]:.3f}% of zero)")
    print("\n== '1' anchor stability across two independent seeds ==")
    for m in MN.MODELS:
        print(f"  {m:10s} s1={per_seed[m]['s1']['fit']:.6e}  s2={per_seed[m]['s2']['fit']:.6e}  "
              f"gap={seed_gap[m]:.4e} = {seed_gap_pct_of_span[m]:.4f}% of span")
    print(f"\n  blend decision margin  = {decision_margin:.6f}")
    print(f"  anchor-induced perturb = {perturbation:.6f}")
    print(f"  => {blob['stability']['verdict']}")
    print("\n== convergence evidence ==")
    for m in MN.MODELS:
        for tag in SEEDS:
            c = per_seed[m][tag]["convergence"]
            print(f"  {m:10s} {tag}: {c['unique_evals']:,} unique, champion last improved at "
                  f"epoch {c['last_epoch_that_improved_the_champion']}/{c['epochs_run']} "
                  f"({c['epochs_since_last_improvement']} epochs quiet), "
                  f"{c['islands_within_0.10pct_of_champion']}/{c['islands']} islands within 0.10%")
    print("\n== user's search-anchoring vs the prior ceiling-fraction anchoring ==")
    for m in MN.MODELS:
        i = improvement[m]
        print(f"  {m:10s} search '1' beats best-of-8-candidates by {i['search_beats_ceiling_by_ms']:.4e} ms "
              f"= {i['search_beats_ceiling_pct_of_search_span']:.3f}% of the search span; "
              f"span ratio {i['span_ratio_search_over_ceiling']:.4f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
