"""MODELNORM-1 — the resolution floor from the ONE surviving per-seed surface family.

Trap 7 of the brief says a paired floor must name its pool, and trap 14 says the per-seed
speed surfaces were never harvested. Both are true here, and this driver measures exactly what
is left rather than borrowing a number:

* **What survives:** `COMMUNITY_BASE` has three per-seed conditional tensors
  (`.conditional.seed{0,1,2}.npy`) and three per-seed bigram tensors, so a genuine
  **seed-to-seed** noise estimate is available for ONE of the three models on the BASE family.
  Nothing survives for AALTO or POOL, or for `TRI_PS_FREQ_PRIOR` on any model. The rank table's
  "model spread" floor is therefore a floor on MODEL DISAGREEMENT, which is a different and
  much larger quantity than fit noise — this driver separates the two.

* **Why that distinction decides the deliverable-B answer.** FLAGSHIP-1's lesson (trap 37) is
  that an UNPAIRED floor is the wrong ruler for a PAIRED comparison: every candidate is scored
  on the SAME seed tables, so the seed main effect is common mode and cancels in a difference.
  So the floor that matters for "does this ranking change clear the noise?" is the spread of
  paired DIFFERENCES across seeds, not the spread of values.

Reported on the saved-vs-qwerty% scale and on the normalized scale, both named, with the
candidate x seed decomposition and the seed main effect's share of SS printed before any floor
is quoted (trap 37's explicit requirement).

⚠ NOT reused from FLAGSHIP-1: its "seed = 78-83% of SS" is an **iWeb** figure. This is
blend-v1 and the number is computed here.
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

SEEDS = (0, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra", action="append", default=[], help="NAME=LAYOUT rows to add")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    surf = MN.NativeSurfaces(corpus=args.corpus)
    directory = surf.native_dir

    # ---- inventory: what per-seed material actually exists ----
    inventory = {}
    for model in MN.MODELS:
        for family in ("BASE", "FREQ_PRIOR", "TRI_PS_FREQ_PRIOR"):
            present = [
                s for s in SEEDS
                if (directory / f"{model}_{family}.conditional.seed{s}.npy").is_file()
            ]
            inventory[f"{model}_{family}"] = {
                "per_seed_conditional_available": present,
                "per_seed_bigram_available": [
                    s for s in SEEDS
                    if (directory / f"{model}_{family}.bigram.seed{s}.npy").is_file()
                ],
            }
    usable = [k for k, v in inventory.items() if len(v["per_seed_conditional_available"]) >= 2]
    if not usable:
        raise SystemExit("no per-seed surface family survives; a seed floor cannot be computed")

    layouts = dict(MN.CANDIDATES)
    for spec in args.extra:
        name, _, layout = spec.partition("=")
        layouts[name] = layout
    names = list(layouts)

    results = {}
    for key in usable:
        # native frame per seed: native = bigram_own[:,:,None] + conditional  (verified exactly
        # 0.0 residual in modelnorm_eval's frame guard for the seedmean arrays)
        per_seed_fit = np.empty((len(names), len(SEEDS)))
        for seed_index, seed in enumerate(SEEDS):
            bigram = np.load(directory / f"{key}.bigram.seed{seed}.npy")
            conditional = np.load(directory / f"{key}.conditional.seed{seed}.npy")
            surface = bigram[:, :, None] + conditional
            flat = surface.reshape(-1)
            for row, name in enumerate(names):
                perm = MN.perm_of(layouts[name])
                index = (perm[surf.I] * 31 + perm[surf.J]) * 31 + perm[surf.K]
                per_seed_fit[row, seed_index] = float((flat[index] * surf.F).sum())
        # positive control: the seed MEAN must reproduce the shipped seedmean native array
        bigram_mean = np.load(directory / f"{key}.bigram.seedmean.npy")
        conditional_mean = np.load(directory / f"{key}.conditional.seedmean.npy")
        mean_surface = (bigram_mean[:, :, None] + conditional_mean).reshape(-1)
        shipped_native = np.load(directory / f"{key}.native.npy").reshape(-1)
        control = float(np.abs(mean_surface - shipped_native).max())

        qwerty_row = names.index("qwerty30m")
        saved = 100.0 * (1.0 - per_seed_fit / per_seed_fit[qwerty_row])

        def decompose(matrix: np.ndarray) -> dict:
            grand = matrix.mean()
            candidate_effect = matrix.mean(axis=1) - grand
            seed_effect = matrix.mean(axis=0) - grand
            interaction = matrix - grand - candidate_effect[:, None] - seed_effect[None, :]
            ss_candidate = float(matrix.shape[1] * (candidate_effect**2).sum())
            ss_seed = float(matrix.shape[0] * (seed_effect**2).sum())
            ss_interaction = float((interaction**2).sum())
            total = ss_candidate + ss_seed + ss_interaction
            return {
                "ss_candidate": ss_candidate, "ss_seed": ss_seed,
                "ss_interaction": ss_interaction, "ss_total": total,
                "seed_main_effect_share_of_ss": ss_seed / total,
                "candidate_main_effect_share_of_ss": ss_candidate / total,
                "interaction_share_of_ss": ss_interaction / total,
            }

        def floor_of(matrix: np.ndarray, drop_reference: bool, reference_row: int) -> dict:
            """Paired and unpaired floors. ``drop_reference`` excludes the qwerty30m row.

            ⚠ THE REFERENCE ROW MAKES THE PAIRED FLOOR DEGENERATE ON A saved% SCALE. saved% is
            ``100*(1 - fit/fit_qwerty)`` computed PER SEED, so qwerty's own row is identically
            (0,0,0) with zero spread — and then ``spread(X - qwerty) == spread(X)``, i.e. the
            "paired" floor for any pair involving the reference is just that layout's unpaired
            spread. Including it forces the paired/unpaired ratio to exactly 1.0000 and hides
            the cancellation the paired analysis exists to measure. Measured here: 1.0000 with
            the reference in, 0.5632 with it out.
            """
            rows = [r for r in range(len(names))
                    if not (drop_reference and r == reference_row)]
            spreads, per_pair = [], {}
            for i, j in itertools.combinations(rows, 2):
                differences = matrix[i] - matrix[j]
                spread = float(differences.max() - differences.min())
                spreads.append(spread)
                per_pair[f"{names[i]} vs {names[j]}"] = {
                    "mean_difference": float(differences.mean()),
                    "per_seed_differences": differences.tolist(),
                    "spread": spread,
                    "sd": float(differences.std(ddof=1)),
                    "sign_agrees_across_seeds": bool(
                        np.all(differences > 0) or np.all(differences < 0)
                    ),
                }
            spreads = np.array(spreads)
            unpaired = float(max(matrix[r].max() - matrix[r].min() for r in rows))
            resolved = sum(
                1 for v in per_pair.values()
                if abs(v["mean_difference"]) > spreads.max() and v["sign_agrees_across_seeds"]
            )
            return {
                "rows_used": [names[r] for r in rows],
                "floor_unpaired_max_within_candidate_spread": unpaired,
                "floor_paired_max_difference_spread": float(spreads.max()),
                "floor_paired_median_difference_spread": float(np.median(spreads)),
                "paired_over_unpaired_ratio": float(spreads.max() / unpaired),
                "pairs_resolved_against_the_conservative_paired_floor": resolved,
                "n_pairs": len(per_pair),
                "per_pair": per_pair,
            }

        results[key] = {
            "positive_control_seedmean_reproduces_shipped_native_max_abs": control,
            "per_seed_fit_ms": {n: per_seed_fit[r].tolist() for r, n in enumerate(names)},
            "saved_vs_qwerty30m_pct_per_seed": {
                n: saved[r].tolist() for r, n in enumerate(names)
            },
            # BOTH scales, because the seed share is NOT scale-invariant and quoting one alone
            # invites a false comparison with FLAGSHIP-1's iWeb figure.
            "decomposition_on_raw_ms": decompose(per_seed_fit),
            "decomposition_on_saved_pct": decompose(saved),
            "scale_note": (
                "saved% divides by qwerty PER SEED, so the reference row is identically zero "
                "and the seed MAIN effect is partly removed by construction. The raw-ms "
                "decomposition is therefore reported alongside; on blend-v1 the seed share is "
                "small on BOTH scales, which is the substantive finding."
            ),
            "floor_saved_pct_all_rows": floor_of(saved, False, qwerty_row),
            "floor_saved_pct_excluding_reference": floor_of(saved, True, qwerty_row),
            "floor_raw_ms_excluding_reference": floor_of(per_seed_fit, True, qwerty_row),
        }

    blob = {
        "what": "MODELNORM-1 seed-noise resolution floor, from the surviving per-seed surfaces",
        "identity": surf.identity(),
        "inventory_of_per_seed_material": inventory,
        "usable_families": usable,
        "trap14_note": (
            "per-seed material survives for ONE family only (COMMUNITY_BASE, 3 seeds). Nothing "
            "survives for AALTO or POOL, or for TRI_PS_FREQ_PRIOR on any model — the per-seed "
            "models behind 7 of 8 surfaces are gone. So a seed floor exists for exactly one of "
            "the three models this arm blends, on a DIFFERENT family than the one it blends."
        ),
        "results": results,
        "how_to_read_this": (
            "the seed floor bounds FIT NOISE for one model; the rank table's model-spread floor "
            "bounds MODEL DISAGREEMENT across three models. They are different quantities and "
            "the larger one is not a refinement of the smaller. A ranking change should clear "
            "the seed floor to be real at all, and clear the model-spread floor to be a claim "
            "about the blend rather than about which model you happened to weight."
        ),
        "not_reused": (
            "FLAGSHIP-1's 'seed = 78-83% of SS' is its iWeb figure and is NOT reused; the share "
            "here is computed on blend-v1 and reported above, on BOTH the raw-ms and saved% "
            "scales. It comes out ~0.7-0.8%, i.e. two orders of magnitude smaller — so on "
            "blend-v1 the seed is NOT the dominant variance component and the paired-vs-unpaired "
            "distinction buys far less than it did on iWeb."
        ),
        "modelled_only": (
            "MODELLED ONLY: fitted surfaces, .native frame, BAKED 90 WPM. Not a claim about "
            "realized typing speed. No layout is promoted or adopted."
        ),
    }
    Path(args.out).write_text(json.dumps(blob, indent=1))
    print(f"WROTE {args.out}")
    for key, result in results.items():
        print(f"\n== {key} ({len(SEEDS)} seeds, blend-v1, .native) ==")
        print("  positive control (seedmean rebuilds shipped native): "
              f"max abs {result['positive_control_seedmean_reproduces_shipped_native_max_abs']:.3e}")
        for scale in ("decomposition_on_raw_ms", "decomposition_on_saved_pct"):
            d = result[scale]
            print(f"  {scale.replace('decomposition_on_', ''):9s}: seed={100*d['seed_main_effect_share_of_ss']:6.2f}% "
                  f"candidate={100*d['candidate_main_effect_share_of_ss']:6.2f}% "
                  f"interaction={100*d['interaction_share_of_ss']:5.2f}% of SS")
        print("  (FLAGSHIP-1's 78-83% seed share is its iWeb figure; on blend-v1 it is ~0.7-0.8%)")
        for label in ("floor_saved_pct_all_rows", "floor_saved_pct_excluding_reference",
                      "floor_raw_ms_excluding_reference"):
            f = result[label]
            print(f"  {label}:")
            print(f"      unpaired={f['floor_unpaired_max_within_candidate_spread']:.6g}  "
                  f"paired={f['floor_paired_max_difference_spread']:.6g}  "
                  f"median_paired={f['floor_paired_median_difference_spread']:.6g}  "
                  f"ratio={f['paired_over_unpaired_ratio']:.4f}")
            print("      pairs resolved vs the conservative paired floor: "
                  f"{f['pairs_resolved_against_the_conservative_paired_floor']}/{f['n_pairs']}")
        print("  ⚠ ratio EXACTLY 1.0000 in the all-rows cell is a DEGENERACY, not a finding: "
              "saved% is computed per seed against qwerty, so the reference row is (0,0,0) and "
              "spread(X - qwerty) == spread(X). The excluding-reference cell is the readable one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
