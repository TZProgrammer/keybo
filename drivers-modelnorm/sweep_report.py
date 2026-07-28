"""MODELNORM-1 deliverable D — the PREFERENCE SWEEP: does the weight behave as a preference?

Reads the blend searches produced at (1,1,1), (1,0,0), (0,1,0), (0,0,1) and (2,1,1) — all at
IDENTICAL budget, islands, epochs and seed, so a difference between cells is the WEIGHT and
not the draw — then answers the question that makes the weight interpretable:

* **Positive control on the whole scheme.** A (1,0,0) search maximizes exactly one model's
  normalized score, whose maximum is 1.0 BY CONSTRUCTION at that model's "1" anchor. So a
  correct implementation must return blend == 1.000000 and the anchor's own layout. That is an
  end-to-end check of the anchors, the normalization and the search in one number — and it is
  the check that would catch an anchor/objective mismatch, which no unit test can see.
* **Do the solo champions differ?** Reported as pairwise Hamming distance over the 30 slots.
  High correlation on a WIDE random pool does not imply agreement in the NARROW near-optimal
  band (trap 52), so this is measured rather than inferred from the correlation matrix.
* **Does an uneven weight move the champion the right way?** (2,1,1) must land between (1,0,0)
  and (1,1,1) in normalized AALTO score. If it does not, the weight is not acting as a
  preference and the deliverable's claim fails.
* **What does each cell cost the other models?** For every cell, every model's normalized
  score, so the tradeoff a preference buys is visible rather than asserted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelnorm_eval as MN  # noqa: E402

REPO = Path("/tmp/modelnorm")

#: cell name -> the weights it was run at. Must match run_sweep.sh.
CELLS = {
    "equal": (1.0, 1.0, 1.0),
    "aalto-only": (1.0, 0.0, 0.0),
    "community-only": (0.0, 1.0, 0.0),
    "pool-only": (0.0, 0.0, 1.0),
    "aalto-pref": (2.0, 1.0, 1.0),
}


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def analyze_ms(layouts: list[str], corpus: str | None) -> dict[str, float]:
    """ms/char per layout from the shipped CLI, with SET-CONTAINMENT asserted (never a count:
    `analyze` adds a `--ref` row, so a count check is wrong in the other direction)."""
    unique = list(dict.fromkeys(layouts))
    command = ["uv", "run", "--no-sync", "python", "-m", "keybo.cli", "analyze", "--json"]
    if corpus:
        command += ["--corpus", corpus]
    command += unique
    proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"analyze failed rc={proc.returncode}: {proc.stderr[-3000:]}")
    blob = json.loads(proc.stdout)
    got = {row["layout"]: row for row in blob["rows"].values()}
    missing = set(unique) - set(got)
    assert not missing, f"analyze DROPPED {len(missing)} requested layout(s): {sorted(missing)}"
    return {layout: got[layout]["time"]["ms_per_char"] for layout in unique}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    surf = MN.NativeSurfaces(corpus=args.corpus)
    anchors = MN.load_anchors(Path(args.anchors))
    runs_dir = Path(args.runs)

    cells = {}
    for name, weights in CELLS.items():
        path = runs_dir / f"blend-{name}.json"
        if not path.is_file():
            print(f"  (skipping {name}: {path} not present)")
            continue
        run = json.load(open(path))
        assert run["objective"] == "blend", f"{path} is not a blend run"
        assert tuple(run["weights"]) == weights, (
            f"{path} ran at weights {run['weights']}, expected {list(weights)}"
        )
        cells[name] = run

    if "equal" not in cells:
        raise SystemExit("the equal-weight cell is required")

    # every cell must have run at the same budget/seed, else a difference is the draw
    signature = {
        n: (r["budget_requested"], r["islands"], r["seed"], r["polish_sweeps"], r["ga_share"])
        for n, r in cells.items()
    }
    assert len(set(signature.values())) == 1, (
        f"sweep cells did NOT run at identical settings, so a difference between them is not "
        f"attributable to the weight: {signature}"
    )

    layouts = {n: r["champion"]["layout"] for n, r in cells.items()}
    ms = analyze_ms([*layouts.values(), *MN.CANDIDATES.values()], args.corpus)

    equal_normalizer = MN.BlendNormalizer(anchors, dict(zip(MN.MODELS, CELLS["equal"], strict=True)))
    rows = {}
    for name, layout in layouts.items():
        fits = surf.fit_of_layout(layout)
        normalized = equal_normalizer.normalize(fits)
        own = MN.BlendNormalizer(anchors, dict(zip(MN.MODELS, CELLS[name], strict=True)))
        rows[name] = {
            "weights": list(CELLS[name]),
            "champion": layout,
            "normalized": {m: float(v) for m, v in zip(MN.MODELS, normalized, strict=True)},
            "normalized_floor_min_over_models": float(normalized.min()),
            "own_objective_blend": float(own.blend(fits)),
            "equal_weight_blend": float(equal_normalizer.blend(fits)),
            "ms_per_char": ms[layout],
            "unique_evals": cells[name]["unique_evals"],
            "recorded_fitness": cells[name]["champion"]["fitness"],
        }

    # ---- positive control: a solo cell must hit blend == 1.0 at that model's anchor ----
    control = {}
    for name, model in (("aalto-only", "AALTO"), ("community-only", "COMMUNITY"),
                        ("pool-only", "POOL")):
        if name not in rows:
            continue
        expected_layout = anchors.one_provenance["layout_of_record"][model]
        got_blend = rows[name]["own_objective_blend"]
        control[name] = {
            "model": model,
            "own_blend": got_blend,
            "hits_1.0": abs(got_blend - 1.0) < 1e-9,
            "champion": rows[name]["champion"],
            "anchor_layout_of_record": expected_layout,
            "reproduces_the_anchor_layout": rows[name]["champion"] == expected_layout,
            "note": (
                "a solo blend's maximum is 1.0 BY CONSTRUCTION at that model's '1' anchor, so "
                "this is an end-to-end control on the anchors, the normalization AND the search "
                "in one number — it is what would catch an anchor/objective mismatch that no "
                "unit test can see"
            ),
        }
        assert control[name]["hits_1.0"], (
            f"{name}: solo blend reached {got_blend}, not 1.0 — the anchor and the objective "
            f"disagree, so the normalization is not the one the anchors describe"
        )

    # ---- do the solo champions differ? (trap 52: wide-pool correlation != band agreement) ----
    solo = {n: rows[n]["champion"] for n in ("aalto-only", "community-only", "pool-only")
            if n in rows}
    distances = {
        f"{a} vs {b}": hamming(solo[a], solo[b])
        for a, b in itertools.combinations(solo, 2)
    }

    # ---- does the uneven weight behave as a preference? ----
    preference = None
    if {"aalto-only", "equal", "aalto-pref"} <= set(rows):
        solo_a = rows["aalto-only"]["normalized"]["AALTO"]
        equal_a = rows["equal"]["normalized"]["AALTO"]
        pref_a = rows["aalto-pref"]["normalized"]["AALTO"]
        between = min(solo_a, equal_a) <= pref_a <= max(solo_a, equal_a)
        preference = {
            "aalto_normalized_at_1_0_0": solo_a,
            "aalto_normalized_at_1_1_1": equal_a,
            "aalto_normalized_at_2_1_1": pref_a,
            "monotone_2_1_1_lies_between": bool(between),
            "moved_toward_aalto_vs_equal": bool(pref_a > equal_a),
            "verdict": (
                "the weight ACTS AS A PREFERENCE: raising AALTO's weight from 1 to 2 moved the "
                "champion's AALTO score toward the AALTO-only optimum without reaching it"
                if between and pref_a > equal_a else
                "the weight did NOT behave monotonically as a preference on this axis — see the "
                "three numbers above"
            ),
        }

    # ---- what does each preference COST the other models? ----
    cost = {}
    for name in rows:
        best_elsewhere = {}
        for index, model in enumerate(MN.MODELS):
            solo_name = {"AALTO": "aalto-only", "COMMUNITY": "community-only",
                         "POOL": "pool-only"}[model]
            if solo_name in rows:
                best_elsewhere[model] = (
                    rows[name]["normalized"][model] - rows[solo_name]["normalized"][model]
                )
            del index
        cost[name] = {
            "shortfall_vs_each_models_own_optimum": best_elsewhere,
            "worst_shortfall": min(best_elsewhere.values()) if best_elsewhere else None,
        }

    blob = {
        "what": "MODELNORM-1 deliverable D: the preference sweep",
        "identity": surf.identity(),
        "identical_run_settings": {
            "budget_islands_seed_polish_gashare": list(next(iter(signature.values()))),
            "why": "a difference between cells is then the WEIGHT and not the draw",
        },
        "cells": rows,
        "positive_control_solo_cells_hit_1.0": control,
        "solo_champion_pairwise_hamming_distance_of_30": distances,
        "trap52_note": (
            "the three models correlate 0.83-0.95 on a WIDE random pool (participation ratio "
            "1.17 of 3), but that does NOT imply they agree in the NARROW near-optimal band. "
            "The Hamming distances above are the direct measurement of band agreement."
        ),
        "preference_monotonicity": preference,
        "cost_of_each_preference": cost,
        "ms_per_char": {n: rows[n]["ms_per_char"] for n in rows},
        "frozen_comparison_ms_per_char": {
            "arm-B": ms[MN.CANDIDATES["arm-B"]],
            "keybo-lsb": ms[MN.CANDIDATES["keybo-lsb"]],
            "flagship-c3": ms[MN.CANDIDATES["flagship-c3"]],
            "arm-A": ms[MN.CANDIDATES["arm-A"]],
            "qwerty30m": ms[MN.CANDIDATES["qwerty30m"]],
        },
        "modelled_only": (
            "MODELLED ONLY: .native frame, BAKED 90 WPM, tau saturated at 1.0, Phase-D "
            "cancelled. No layout is promoted or adopted."
        ),
    }
    Path(args.out).write_text(json.dumps(blob, indent=1))
    print(f"WROTE {args.out}")

    print("\n== D: preference sweep (identical budget/seed per cell; blend-v1, .native, 90 WPM) ==")
    print(f"  {'cell':16s} {'weights':10s} {'champion':32s} "
          + " ".join(f"{m[:4]:>8s}" for m in MN.MODELS) + f" {'ms/char':>10s}")
    for name, row in rows.items():
        weights = ",".join(str(int(w)) for w in row["weights"])
        print(f"  {name:16s} {weights:10s} {row['champion']:32s} "
              + " ".join(f"{row['normalized'][m]:8.5f}" for m in MN.MODELS)
              + f" {row['ms_per_char']:10.4f}")
    print("\n  positive control — a solo cell's own blend must be EXACTLY 1.0:")
    for name, result in control.items():
        print(f"    {name:16s} own_blend={result['own_blend']:.9f} hits_1.0={result['hits_1.0']} "
              f"reproduces_anchor_layout={result['reproduces_the_anchor_layout']}")
    print("\n  solo champion pairwise Hamming distance (of 30 slots):")
    for pair, distance in distances.items():
        print(f"    {pair:34s} {distance}/30")
    if preference:
        print(f"\n  preference monotonicity on AALTO: "
              f"(1,0,0)={preference['aalto_normalized_at_1_0_0']:.5f}  "
              f"(2,1,1)={preference['aalto_normalized_at_2_1_1']:.5f}  "
              f"(1,1,1)={preference['aalto_normalized_at_1_1_1']:.5f}")
        print(f"    => {preference['verdict']}")
    print("\n  ms/char vs the frozen set: arm-B "
          f"{blob['frozen_comparison_ms_per_char']['arm-B']:.4f}")
    for name, row in sorted(rows.items(), key=lambda kv: kv[1]["ms_per_char"]):
        print(f"    {name:16s} {row['ms_per_char']:10.4f}  "
              f"vs arm-B {row['ms_per_char'] - blob['frozen_comparison_ms_per_char']['arm-B']:+8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
