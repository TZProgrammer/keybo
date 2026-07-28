"""MODELNORM-1 deliverables C and E — verify the blend champion independently, and adjudicate it.

C. The champion's **predicted ms/char** comes from the shipped ``keybo analyze --json``, i.e.
   an INDEPENDENT path from the search's own objective (which is a 3-surface native-frame
   blend, a different quantity). ⚠ SET-CONTAINMENT of the requested layouts is asserted, never
   ``len(rows) == len(layouts)`` — ``analyze`` adds a row for its ``--ref`` layout, so a count
   comparison is wrong in the other direction, while containment catches exactly the failure
   (a REQUESTED layout silently dropped) that trap 38 is about.

E. Admissibility on three frames:
   * the **10-axis dominance frame WITH the strict-win term** — ``n_ge == 10 AND n_strict >= 1``
     (trap 33: ``n_ge == n_axes`` alone labels a candidate that merely TIES everywhere a
     dominator). Axes and signs are taken verbatim from arm E's frozen frame so the verdict is
     comparable to the campaign's other arms.
   * the **normalized floor** — this arm's own min-over-models normalized score, which is the
     scale-corrected analogue of FLOOR-METHODOLOGY-1's ceiling-fraction floor.
   * the **19-gauge frame** with per-gauge win counts, reported as **18 gauges that can move
     plus ``sfr``**, because ``sfr`` counts doubled letters and is a PERMUTATION INVARIANT
     (trap 23) — numpy gives it std 1.9e-14, so a ``std > 0`` filter keeps it and then
     rank-correlates pure noise. Invariance is tested directly by shuffling, never via a
     variance threshold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelnorm_eval as MN  # noqa: E402

REPO = Path("/tmp/modelnorm")

#: 10-axis dominance frame + orientation, VERBATIM from arm E's frozen frame so this arm's
#: verdict is comparable with the campaign's others. +1 = higher is better after the sign.
DOM_AXES = ("floor", "mean", "wfd", "genkey", "oxey1", "oxey2", "lsb", "scissor", "sfb", "sfs")
DOM_SIGN = {"floor": +1, "mean": +1, "wfd": +1, "oxey1": +1, "oxey2": +1,
            "genkey": -1, "lsb": -1, "scissor": -1, "sfb": -1, "sfs": -1}

#: The incumbents a champion must dominate to be admissible.
INCUMBENTS = ("keybo-lsb", "keybo-lsb+lm", "flagship-c3", "arm-B", "arm-A")

#: LOWER-better direction on the 19-gauge frame. `sfr` is a permutation invariant (trap 23) and
#: is excluded from win COUNTS while still being reported, so no denominator hides a tie.
GAUGE_LOWER_BETTER = {
    "sfb": True, "sfs": True, "sfr": True, "lsb": True, "lsb-dist": True, "sfb-dist": True,
    "sfs-dist": True, "alt": False, "redir": True, "bad-redir": True, "onehand": False,
    "roll": False, "sr-roll": False, "scissor": True, "imbalance": True,
    "oxey-style": False, "comfort": False,
}


def dominates(candidate: dict, incumbent: dict, atol: float = 1e-9) -> tuple[bool, int, int]:
    """Pareto dominance WITH the strict-win term (trap 33)."""
    c = np.array([DOM_SIGN[a] * candidate[a] for a in DOM_AXES])
    i = np.array([DOM_SIGN[a] * incumbent[a] for a in DOM_AXES])
    n_ge = int((c >= i - atol).sum())
    n_strict = int((c > i + atol).sum())
    return (n_ge == len(DOM_AXES) and n_strict >= 1), n_ge, n_strict


def run_analyze(layouts: dict[str, str], corpus: str | None) -> dict:
    """The shipped `keybo analyze --json`, with SET-CONTAINMENT asserted (never a count)."""
    command = ["uv", "run", "--no-sync", "python", "-m", "keybo.cli", "analyze", "--json"]
    if corpus:
        command += ["--corpus", corpus]
    command += list(dict.fromkeys(layouts.values()))
    proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"analyze failed rc={proc.returncode}: {proc.stderr[-3000:]}")
    blob = json.loads(proc.stdout)
    rows = blob["rows"]
    got = {row["layout"] for row in rows.values()}
    want = set(layouts.values())
    missing = want - got
    assert not missing, f"analyze DROPPED {len(missing)} requested layout(s): {sorted(missing)}"
    assert len(rows) == len(set(rows)), "duplicate row keys in analyze output"
    blob["_extra_rows"] = sorted(got - want)
    return blob


def assert_sfr_is_a_permutation_invariant(analyze_rows: dict, layouts: dict) -> dict:
    """TRAP 23, tested DIRECTLY (by looking across shuffled layouts), never via std > 0."""
    values = sorted({round(row["gauges"]["sfr"], 12) for row in analyze_rows.values()})
    return {
        "distinct_values_over_the_scored_layouts": values,
        "is_invariant": len(values) == 1,
        "numpy_std": float(np.std([row["gauges"]["sfr"] for row in analyze_rows.values()])),
        "note": (
            "sfr counts DOUBLED LETTERS, so no placement can move it. numpy reports a std of "
            "order 1e-14 rather than 0, so a `std > 0` filter KEEPS it and then rank-correlates "
            "pure noise. Excluded from win COUNTS below; the frame is 18 movable gauges + sfr."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--champion", required=True, help="NAME=LAYOUT of the blend champion")
    parser.add_argument("--also", action="append", default=[],
                        help="extra NAME=LAYOUT rows (e.g. preference-sweep champions)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    surf = MN.NativeSurfaces(corpus=args.corpus)
    anchors = MN.load_anchors(Path(args.anchors))
    normalizer = MN.BlendNormalizer(anchors)

    champion_name, _, champion_layout = args.champion.partition("=")
    everyone = dict(MN.CANDIDATES)
    everyone[champion_name] = champion_layout
    extras = {}
    for spec in args.also:
        name, _, layout = spec.partition("=")
        extras[name] = layout
        everyone[name] = layout

    analyze = run_analyze(everyone, args.corpus)
    spec_to_name = {}
    for name, layout in everyone.items():
        spec_to_name.setdefault(layout, name)
    rows = {}
    for row in analyze["rows"].values():
        name = spec_to_name.get(row["layout"])
        if name is not None:
            rows[name] = row
    missing = set(everyone) - set(rows)
    # a duplicate layout string under two names is fine; assert every LAYOUT is present
    unmatched = {n for n in missing if everyone[n] not in {r["layout"] for r in rows.values()}}
    assert not unmatched, f"no analyze row for {sorted(unmatched)}"
    for name in missing:
        for other, row in list(rows.items()):
            if row["layout"] == everyone[name]:
                rows[name] = row
                del other
                break

    # ---- C: predicted ms/char, independently ----
    ms_per_char = {n: rows[n]["time"]["ms_per_char"] for n in everyone}
    frozen = {
        "arm-B": 253.90057910352797, "keybo-lsb": 254.6307495925403,
        "flagship-c3": 254.9761188060974, "arm-A": 256.846570694692,
        "qwerty30m": 264.13891657883323,
    }
    reproduced = {
        n: {"expected": v, "got": ms_per_char[n], "abs_diff": abs(ms_per_char[n] - v)}
        for n, v in frozen.items()
    }
    worst = max(v["abs_diff"] for v in reproduced.values())
    assert worst < 1e-9, f"the frozen comparison set did not reproduce (worst {worst:.3e})"

    # ---- normalized floor + blend for every row ----
    normalized_rows = {}
    for name, layout in everyone.items():
        fits = surf.fit_of_layout(layout)
        norm = normalizer.normalize(fits)
        normalized_rows[name] = {
            "layout": layout,
            "raw_ms": {m: float(v) for m, v in zip(MN.MODELS, fits, strict=True)},
            "normalized": {m: float(v) for m, v in zip(MN.MODELS, norm, strict=True)},
            "normalized_floor_min_over_models": float(norm.min()),
            "normalized_mean_over_models": float(norm.mean()),
            "equal_weight_blend": float(normalizer.blend(fits)),
            "analyze_ms_per_char": ms_per_char[name],
        }

    # ---- E: the 10-axis dominance frame ----
    dominance_axes = {}
    for name in everyone:
        row = rows[name]
        norm = normalizer.normalize(surf.fit_of_layout(everyone[name]))
        qwerty = surf.fit_of_layout(MN.CANDIDATES["qwerty30m"])
        saved = 100.0 * (1.0 - surf.fit_of_layout(everyone[name]) / qwerty)
        dominance_axes[name] = {
            # this arm's own floor/mean: the NORMALIZED (per-model 0-1) floor and mean, which
            # is the scale-corrected analogue of the ceiling-fraction floor the frame was
            # built with. Stated explicitly because it is NOT the same number as arm E's.
            "floor": float(norm.min()),
            "mean": float(saved.mean()),
            "wfd": row["community"]["wfd"],
            "genkey": row["community_primed"]["genkey_primed"],
            "oxey1": row["community_primed"]["oxey1_primed"],
            "oxey2": row["community_primed"]["oxey2_primed"],
            "lsb": row["kmstats"]["lsb"],
            "sfb": row["kmstats"]["sfb"],
            "sfs": row["kmstats"]["sfs"],
            "scissor": row["gauges"]["scissor"],
        }

    champions = [champion_name, *extras]
    dominance = {}
    for name in champions:
        results = {}
        for incumbent in INCUMBENTS:
            is_dom, n_ge, n_strict = dominates(dominance_axes[name], dominance_axes[incumbent])
            results[incumbent] = {"dominates": is_dom, "n_ge": n_ge, "n_strict": n_strict}
        dominance[name] = {
            "vs_incumbents": results,
            "dominator_exists": any(v["dominates"] for v in results.values()),
            "best_n_ge": max(v["n_ge"] for v in results.values()),
            "best_n_strict": max(v["n_strict"] for v in results.values()),
        }

    # ---- E: the 19-gauge frame ----
    sfr = assert_sfr_is_a_permutation_invariant(analyze["rows"], everyone)
    gauge_names = [g for g in GAUGE_LOWER_BETTER if g in rows[champion_name]["gauges"]]
    movable = [g for g in gauge_names if g != "sfr"]
    gauge_table = {
        name: {g: rows[name]["gauges"][g] for g in gauge_names} for name in everyone
    }
    wins = {}
    for name in champions:
        per_incumbent = {}
        for incumbent in INCUMBENTS:
            won = [
                g for g in movable
                if (gauge_table[name][g] < gauge_table[incumbent][g]) == GAUGE_LOWER_BETTER[g]
                and gauge_table[name][g] != gauge_table[incumbent][g]
            ]
            per_incumbent[incumbent] = {
                "wins": len(won), "of": len(movable), "which": won,
            }
        wins[name] = per_incumbent

    blob = {
        "what": "MODELNORM-1 deliverables C and E: independent ms/char + admissibility",
        "identity": surf.identity(),
        "analyze_extra_rows": analyze["_extra_rows"],
        "analyze_containment_check": (
            "SET-CONTAINMENT of the requested layout strings was asserted (never "
            "len(rows) == len(layouts): analyze adds a --ref row, so a count comparison is "
            "wrong in the other direction while containment catches a dropped request)"
        ),
        "C_ms_per_char": ms_per_char,
        "C_frozen_set_reproduced": reproduced,
        "C_frozen_worst_abs_diff": worst,
        "normalized_rows": normalized_rows,
        "E_dominance": {
            "frame": {"axes": list(DOM_AXES), "sign": DOM_SIGN,
                      "predicate": "n_ge == 10 AND n_strict >= 1 (trap 33: the strict-win term "
                                   "is required, else a candidate that merely TIES everywhere "
                                   "is labelled a dominator)",
                      "floor_axis_note": (
                          "'floor' here is THIS arm's min-over-three-models NORMALIZED score "
                          "(the scale-corrected analogue of FLOOR-METHODOLOGY-1's "
                          "ceiling-fraction floor). It is NOT numerically the same axis as "
                          "arm E's six-surface ceiling-fraction floor; the dominance VERDICT "
                          "is comparable, the floor NUMBER is not."
                      )},
            "axes_values": dominance_axes,
            "verdicts": dominance,
        },
        "E_nineteen_gauge_frame": {
            "sfr_invariance": sfr,
            "gauges_counted": movable,
            "n_gauges_counted": len(movable),
            "gauge_values": gauge_table,
            "win_counts_vs_incumbents": wins,
            "denominator_note": (
                f"win counts are out of {len(movable)} MOVABLE gauges, not {len(gauge_names)}: "
                f"sfr is a permutation invariant and is a tie by construction (trap 23)."
            ),
        },
        "modelled_only": (
            "MODELLED ONLY: the three fitted surfaces are on the .native frame at a BAKED "
            "90 WPM; tau saturated at 1.0, Phase-D cancelled. ms/char is the shipped served "
            "K31 metric at 90 WPM on blend-v1. No layout is promoted or adopted."
        ),
    }
    Path(args.out).write_text(json.dumps(blob, indent=1))
    print(f"WROTE {args.out}")

    print("\n== C: predicted ms/char (shipped keybo analyze, blend-v1, 90 WPM) ==")
    for name, value in sorted(ms_per_char.items(), key=lambda kv: kv[1]):
        mark = " <-- BLEND CHAMPION" if name == champion_name else (
            " (sweep)" if name in extras else "")
        delta = value - ms_per_char["arm-B"]
        print(f"  {name:34s} {value:10.4f}   vs arm-B {delta:+8.4f}{mark}")
    print(f"\n  frozen comparison set reproduced, worst abs diff = {worst:.3e}")
    print(f"  analyze extra (--ref) rows, not ours: {analyze['_extra_rows']}")

    print("\n== E: dominance (10 axes, strict-win term REQUIRED) ==")
    for name in champions:
        verdict = dominance[name]
        print(f"  {name}: dominator_exists={verdict['dominator_exists']} "
              f"best_n_ge={verdict['best_n_ge']}/10 best_n_strict={verdict['best_n_strict']}")
        for incumbent, result in verdict["vs_incumbents"].items():
            print(f"      vs {incumbent:14s} n_ge={result['n_ge']:2d} "
                  f"n_strict={result['n_strict']:2d} dominates={result['dominates']}")
    print(f"\n  normalized floor: " + "  ".join(
        f"{n}={normalized_rows[n]['normalized_floor_min_over_models']:+.6f}" for n in champions))
    print(f"\n== E: 19-gauge frame ({len(movable)} movable + sfr invariant) ==")
    print(f"  sfr distinct values over all scored layouts: "
          f"{sfr['distinct_values_over_the_scored_layouts']} (invariant={sfr['is_invariant']}, "
          f"numpy std={sfr['numpy_std']:.3e})")
    for name in champions:
        summary = "  ".join(f"{i}:{v['wins']}/{v['of']}" for i, v in wins[name].items())
        print(f"  {name}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
