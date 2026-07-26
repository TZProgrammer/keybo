"""REHUNT adjudication — the per-cell answer, frozen verdict vs RE-RUN verdict.

For each (corpus, frame, incumbent) cell in the 14-flip list, four states are distinguished, and
they are NOT the same claim:

  frozen-dominates / readjudicated-NOT / rerun-DOMINATES
      The frozen layout died, but a hunt pointed at the corrected bar found a DIFFERENT one that
      lives. The frozen verdict's *conclusion* survives on a new witness; its *evidence* does not.

  frozen-dominates / readjudicated-NOT / rerun-NOT
      A real null from a TARGETED hunt (never an archive scan). The correction removed the claim
      and a search at comparable budget could not replace it.

Plus the two nulls that must survive, checked directly rather than assumed:

  NULL 1 — "no layout dominates all five" (NO-ANCHOR-1). Every cell's IDEAL(all5) target must
      still fail. Also reports the corrected all-five wfd shortfall against the frozen corrupt one
      (WFD-FRAMES-1 measured 5.21e11 -> 2.67e12 on no-anchor arm B: the correct axis blocks HARDER).

  NULL 2 — "the wscissor axis is inert" (WSCISSOR-ARMB-1). Read as placebo -> real, i.e.
      narrow11 -> wide11, NEVER ten -> wide11: going ten -> wide11 changes the axis AND the frame
      size, so the marginal effect is unattributable without a same-SIZE placebo (trap 17). The
      placebo is deliberately NESTED in the real axis (narrow support ⊂ wide), which understates
      the real axis's cost and so makes an "inert" verdict conservative.

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. Nothing promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ART = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts")

#: The 7 frozen cells and the (corpus, arm, frame) each re-runs as. Keys match the launcher tags.
CELL_TO_FROZEN = {
    "blend-armA-ten": "gen-on-blend/hunt-blend-armA-norm.json",
    "blend-armB-ten": "gen-on-blend/hunt-blend-armB-norm.json",
    "noanchor-armA-ten": "noanchor-1/hunt-noanchor-armA-norm.json",
    "noanchor-armB-ten": "noanchor-1/hunt-noanchor-armB-norm.json",
    "blend-armA-twelve": "wscissor-gen-1/runs/whunt-blend-twelve.json",
    "iweb-armA-twelve": "wscissor-gen-1/runs/whunt-iweb-twelve.json",
    "noanchor-armA-twelve": "wscissor-gen-1/runs/whunt-noanchor-twelve.json",
}

#: The brief's exact target list: the 14 (cell, incumbent) pairs whose verdict flipped.
THE_14 = [
    ("blend-armA-ten", "lsb-sib"),
    ("blend-armB-ten", "lsb-sib"),
    ("noanchor-armA-ten", "keybo-lsb"),
    ("noanchor-armA-ten", "lsb-sib"),
    ("noanchor-armA-ten", "archive-1843"),
    ("noanchor-armA-ten", "keybo-lsb+lm"),
    ("noanchor-armB-ten", "lsb-sib"),
    ("noanchor-armB-ten", "keybo-lsb+lm"),
    ("blend-armA-twelve", "lsb-sib"),
    ("iweb-armA-twelve", "lsb-sib"),
    ("noanchor-armA-twelve", "keybo-lsb"),
    ("noanchor-armA-twelve", "lsb-sib"),
    ("noanchor-armA-twelve", "archive-1843"),
    ("noanchor-armA-twelve", "keybo-lsb+lm"),
]

#: WFD-FRAMES-1's re-adjudicated all-five wfd shortfalls (no-anchor arm B), to be re-derived.
FROZEN_ALL5_SHORTFALL = {"corrupt": 520_816_457_900, "corrected": 2_668_822_579_700}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--preflight", default=str(HERE / "runs" / "rehunt-preflight.json"))
    ap.add_argument("--verify", default=str(HERE / "runs" / "rehunt-verification.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    preflight = load(Path(args.preflight))
    verify = load(Path(args.verify))
    if preflight is None or preflight["verdict"] != "PASS":
        raise SystemExit("preflight missing or FAILED — everything downstream is void")
    verified = {}
    if verify:
        for r in verify["rows"]:
            verified[(r["run"].replace("rehunt-", "").replace(".json", ""), r["target"])] = r

    # ---- per-cell table ------------------------------------------------------------------
    cells = {}
    for tag, frozen_rel in CELL_TO_FROZEN.items():
        rerun = load(runs_dir / f"rehunt-{tag}.json")
        frozen = load(ART / frozen_rel)
        if rerun is None:
            cells[tag] = {"status": "MISSING", "frozen_artifact": frozen_rel}
            continue
        pf = {
            r["target"]: r
            for r in preflight["rows"]
            if r["file"] == frozen_rel
        }
        per_target = {}
        for target, best in rerun["per_target_best"].items():
            frozen_best = frozen["per_target_best"].get(target, {})
            v = verified.get((tag, target))
            per_target[target] = {
                "frozen_dominates": bool(frozen_best.get("dominates_target", False)),
                "frozen_layout": frozen_best.get("best_layout"),
                "frozen_n_ge": frozen_best.get("best_n_ge"),
                "readjudicated_dominates": (
                    None if target not in pf else pf[target]["dominates_corrected"]
                ),
                "readjudicated_n_ge": (
                    None if target not in pf else pf[target]["n_ge_corrected"]
                ),
                "rerun_dominates": bool(best["dominates_target"]),
                "rerun_layout": best["best_layout"],
                "rerun_n_ge": int(best["best_n_ge"]),
                "rerun_n_strict": int(best["best_n_strict_better"]),
                "rerun_max_n_ge_across_seeds": int(best.get("max_n_ge_across_seeds", 0)),
                "rerun_deficit": float(best["best_deficit"]),
                "rerun_blocking_axes": {
                    a: v2 for a, v2 in best["residual_shortfall"].items() if v2 > 0
                },
                "rerun_layout_is_new": bool(
                    best["best_layout"] != frozen_best.get("best_layout")
                ),
                "slow_path_confirms": None if v is None else v["slow_dominates"],
                "slow_path_max_rel_err": None if v is None else v["max_rel_err"],
                "axes_won": None if v is None else v["axes_won"],
                "wscissor_pct_change": None if v is None else v["wscissor_pct_change"],
                "unique_layouts": int(best.get("unique_layouts_all_seeds", 0)),
            }
        cells[tag] = {
            "status": "ok",
            "frozen_artifact": frozen_rel,
            "corpus": rerun["corpus"],
            "corpus_label": rerun["corpus_label"],
            "arm": rerun["arm"],
            "frame": rerun["frame"],
            "frame_size": len(rerun["frame_axes"]),
            "wfd_mode": rerun["wfd_mode"],
            "unique_layouts_total": rerun["unique_layouts_total"],
            "wall_s": rerun["wall_s"],
            "frozen_dominated": frozen["dominated_targets"],
            "rerun_dominated": rerun["dominated_targets"],
            "per_target": per_target,
        }

    # ---- THE 14, one row each -------------------------------------------------------------
    print("== THE 14 FLIPPED CELLS, re-run against the CORRECTED wfd bar ==\n")
    print(
        f"{'cell':22s} {'incumbent':14s} {'frozen':7s} {'readj':6s} {'RERUN':6s} "
        f"{'n_ge':>7s} {'new?':5s} {'slow':5s}  blocked_by / axes_won"
    )
    the14 = []
    for tag, target in THE_14:
        cell = cells.get(tag, {})
        if cell.get("status") != "ok":
            print(f"{tag:22s} {target:14s} -- cell {cell.get('status', 'ABSENT')}")
            the14.append({"cell": tag, "target": target, "status": cell.get("status", "ABSENT")})
            continue
        t = cell["per_target"][target]
        blocked = ", ".join(f"{a}:{v:.4g}" for a, v in sorted(
            t["rerun_blocking_axes"].items(), key=lambda kv: -kv[1]
        )[:3])
        detail = blocked or (",".join(t["axes_won"] or []) if t["rerun_dominates"] else "(none)")
        print(
            f"{tag:22s} {target:14s} "
            f"{'DOM' if t['frozen_dominates'] else 'no':7s} "
            f"{'DOM' if t['readjudicated_dominates'] else 'NOT':6s} "
            f"{'DOM' if t['rerun_dominates'] else 'NOT':6s} "
            f"{t['rerun_n_ge']:>3d}/{cell['frame_size']:<3d} "
            f"{('yes' if t['rerun_layout_is_new'] else 'same'):5s} "
            f"{str(t['slow_path_confirms']):5s}  {detail}"
        )
        the14.append(
            {
                "cell": tag,
                "corpus": cell["corpus"],
                "corpus_label": cell["corpus_label"],
                "arm": cell["arm"],
                "frame": cell["frame"],
                "frame_size": cell["frame_size"],
                "target": target,
                "status": "ok",
                **t,
            }
        )
    resolved = [r for r in the14 if r.get("status") == "ok"]
    found = [r for r in resolved if r.get("rerun_dominates")]
    print(
        f"\n  cells where a re-run FOUND a dominator against the corrected bar: "
        f"{len(found)}/{len(resolved)}"
    )
    print(
        f"  cells where a TARGETED hunt found NONE (a real null, not an archive-only null): "
        f"{len(resolved) - len(found)}/{len(resolved)}"
    )

    # ---- NULL 1: no layout dominates all five --------------------------------------------
    print("\n== NULL 1 (NO-ANCHOR-1): does any layout dominate ALL FIVE incumbents? ==")
    null1 = {}
    for tag, cell in cells.items():
        if cell.get("status") != "ok":
            continue
        t = cell["per_target"].get("IDEAL(all5)")
        if not t:
            continue
        null1[tag] = {
            "rerun_dominates_all5": t["rerun_dominates"],
            "rerun_n_ge": t["rerun_n_ge"],
            "frame_size": cell["frame_size"],
            "max_n_ge_across_seeds": t["rerun_max_n_ge_across_seeds"],
            "blocking_axes": t["rerun_blocking_axes"],
            "wfd_shortfall": t["rerun_blocking_axes"].get("wfd"),
        }
        print(
            f"  {tag:22s} IDEAL(all5) {'DOMINATES ❌' if t['rerun_dominates'] else 'NOT ✅':14s} "
            f"n_ge={t['rerun_n_ge']}/{cell['frame_size']} "
            f"(max seen {t['rerun_max_n_ge_across_seeds']})  "
            f"wfd shortfall={t['rerun_blocking_axes'].get('wfd', 0.0):.4g}"
        )
    survives1 = all(not v["rerun_dominates_all5"] for v in null1.values()) and bool(null1)
    print(f"  NULL 1 {'SURVIVES ✅' if survives1 else 'CONTRADICTED ❌ (suspect your setup)'}")

    # ---- NULL 2: the wscissor axis is inert (placebo-differenced) -------------------------
    print("\n== NULL 2 (WSCISSOR-ARMB-1): is the wscissor axis inert? placebo -> real ==")
    print("   read narrow11 -> wide11, NEVER ten -> wide11 (trap 17: frame size is a 2nd factor)")
    null2 = {}
    for corpus in ("iweb", "blend", "noanchor"):
        counts = {}
        for frame in ("ten", "narrow11", "wide11", "twelve"):
            run = load(runs_dir / f"rehunt-{corpus}-armA-{frame}.json")
            if run is None:
                continue
            # A raw dominator COUNT conflates two different failures, and only one of them is
            # attributable to the frame:
            #   AXIS-BLOCKED  deficit > 0 — the candidate cannot reach the incumbent on some
            #                 axis. THIS is a frame effect.
            #   SELF-TIE      deficit == 0, n_strict == 0, no blocking axis, and the best find
            #                 IS the incumbent — the search reached the incumbent's own quality
            #                 and merely failed to find a STRICT win. That is SA stochasticity,
            #                 not the added axis blocking anything.
            # Differencing counts without splitting these makes the marginal a noise statistic.
            per_target = {}
            for target, best in run["per_target_best"].items():
                if target.startswith("IDEAL"):
                    continue
                blocking = {a: v for a, v in best["residual_shortfall"].items() if v > 0}
                per_target[target] = {
                    "dominates": bool(best["dominates_target"]),
                    "deficit": float(best["best_deficit"]),
                    "n_strict": int(best["best_n_strict_better"]),
                    "self_tie": bool(
                        not best["dominates_target"]
                        and best["best_deficit"] <= 1e-9
                        and best["best_layout"] == run["incumbent_axes"][target]["layout"]
                    ),
                    "blocking_axes": blocking,
                }
            counts[frame] = {
                "n_dominated": len(run["dominated_targets"]),
                "dominated": run["dominated_targets"],
                "n_self_tie": sum(v["self_tie"] for v in per_target.values()),
                "n_axis_blocked": sum(
                    1 for v in per_target.values() if not v["dominates"] and v["blocking_axes"]
                ),
                "axis_blocked_targets": [
                    t for t, v in per_target.items() if not v["dominates"] and v["blocking_axes"]
                ],
                "unique_layouts": run["unique_layouts_total"],
                "per_target": per_target,
            }
        if "narrow11" in counts and "wide11" in counts:
            marginal = counts["wide11"]["n_dominated"] - counts["narrow11"]["n_dominated"]
            naive = (
                counts["wide11"]["n_dominated"] - counts["ten"]["n_dominated"]
                if "ten" in counts
                else None
            )
            # The attributable marginal: does the real axis BLOCK an incumbent the placebo does
            # not? Only axis-blocked non-dominance counts.
            blocked_marginal = counts["wide11"]["n_axis_blocked"] - counts["narrow11"][
                "n_axis_blocked"
            ]
            # every count difference that is NOT axis-blocked is a self-tie, i.e. search noise
            count_diff_is_all_self_tie = bool(marginal != 0 and blocked_marginal == 0)
            null2[corpus] = {
                "counts": counts,
                "marginal_placebo_to_real": marginal,
                "naive_ten_to_wide11": naive,
                "attributable_marginal_axis_blocked": blocked_marginal,
                "count_difference_is_all_self_tie": count_diff_is_all_self_tie,
                # INERT means the real axis blocks nothing the placebo does not. A count wobble
                # produced purely by self-ties does not make the axis active.
                "wscissor_inert": blocked_marginal == 0,
            }
            print(
                f"  {corpus:9s} ten={counts.get('ten', {}).get('n_dominated', '--')} "
                f"narrow11(placebo)={counts['narrow11']['n_dominated']} "
                f"wide11(real)={counts['wide11']['n_dominated']} "
                f"twelve={counts.get('twelve', {}).get('n_dominated', '--')}   "
                f"count marginal={marginal:+d}  "
                f"ATTRIBUTABLE (axis-blocked) marginal={blocked_marginal:+d}"
                f"{'   [count wobble is ALL self-ties -> search noise]' if count_diff_is_all_self_tie else ''}"
            )
            for frame in ("ten", "narrow11", "wide11", "twelve"):
                if frame in counts:
                    c = counts[frame]
                    print(
                        f"      {frame:9s} dom={c['n_dominated']} self_tie={c['n_self_tie']} "
                        f"axis_blocked={c['n_axis_blocked']} {c['axis_blocked_targets']}"
                    )
    # Absence of data is NOT a contradiction — distinguish the states explicitly, or a missing
    # placebo cell reads as a refuted null.
    if not null2:
        null2_state = "NOT TESTED (no narrow11/wide11 placebo pair present)"
    elif all(v["wscissor_inert"] for v in null2.values()):
        wobble = [c for c, v in null2.items() if v["count_difference_is_all_self_tie"]]
        null2_state = (
            "SURVIVES ✅ (the real axis blocks NOTHING the same-size placebo does not: "
            "attributable axis-blocked marginal is 0 on every corpus)"
        )
        if wobble:
            null2_state += (
                f"; the raw COUNT wobbles on {wobble}, but every count difference is a SELF-TIE "
                "(deficit 0, no blocking axis, best find IS the incumbent) — i.e. the search "
                "failing to find a strict win, not the axis blocking"
            )
    else:
        hot = [c for c, v in null2.items() if not v["wscissor_inert"]]
        null2_state = (
            f"NON-ZERO attributable marginal on {hot} — the real axis blocks an incumbent the "
            "placebo does not; investigate before calling it a discovery"
        )
    survives2 = bool(null2) and all(v["wscissor_inert"] for v in null2.values())
    print(f"  NULL 2 {null2_state}")

    out = {
        "preflight_verdict": preflight["verdict"],
        "preflight_flips_reproduced": preflight["flips_reproduced"],
        "preflight_reverse_flips": preflight["reverse_flips"],
        "zero_reuse_verification": (
            None
            if verify is None
            else {
                "verdict": verify["verdict"],
                "max_rel_err_candidates": verify["zero_reuse_max_rel_err_candidates"],
                "max_rel_err_incumbents": verify["zero_reuse_max_rel_err_incumbents"],
                "max_rel_err_wfd": verify["zero_reuse_max_rel_err_wfd"],
                "n_verdict_disagreements": verify["n_verdict_disagreements"],
            }
        ),
        "the_14": the14,
        "n_cells_with_rerun_dominator": len(found),
        "n_cells_resolved": len(resolved),
        "cells": cells,
        "null_1_no_all5_dominator": {
            "survives": survives1,
            "per_cell": null1,
            "frozen_all5_wfd_shortfall_noanchor_armB": FROZEN_ALL5_SHORTFALL,
        },
        "null_2_wscissor_inert": {
            "survives": survives2,
            "state": null2_state,
            "corpora_tested": sorted(null2),
            "per_corpus": null2,
        },
        "null_semantics": (
            "Every cell here is a TARGETED per-incumbent hunt warm-started from the incumbent "
            "itself, so an empty result IS a real null. No number in this file comes from an "
            "archive scan; 'no dominator found in the archive' is a different and weaker claim."
        ),
        "note": (
            "MODELED/gauge only; tau saturated at 1.0, Phase-D cancelled. Clearing the 12-axis "
            "bar on a corpus is NOT a 'best layout' claim and NOT an adoption claim."
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
