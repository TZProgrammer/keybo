"""MULTIWPM-1 analysis: does the argmin LAYOUT change, and is any difference RESOLVABLE?

Reads run_arms.py's JSON and answers (a)/(b)/(c) from the preregistration.

The decisive design point: winners are re-scored on the SHIPPED seed-averaged `TimeSurface`
(bigram+trigram, seeds 0-2) that `analyze` reports, NOT on the single search model. An arm
scored only on its own search surface would be marking its own homework, and the resolution
floor this campaign quotes (median 0.135 ms/char over 91 board pairs) is a floor on THAT
surface's ms/char, so the comparison has to happen there or the floor does not apply.

Decision rule (registered before results):
  |d ms/char| <  0.135  -> NULL (same layout for practical purposes)
  0.135 <= |d| < 0.243  -> resolvable but MARGINAL; needs >=7/8 sign-consistent seeds
  |d| >= 0.243          -> resolvable

Usage: analyze_arms.py <arms.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from statistics import mean, stdev

import numpy as np

from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.analysis.timecard import TimeSurface
from keybo.data.corpus import load_frequencies, production_corpus_dir

FLOOR = 0.135  # median over 91 board pairs (PREREGISTRATIONS.md:10405)
P90 = 0.243
EVAL_WPMS = (90.0, 100.0, 110.0, 120.0, 130.0, 140.0)


def verdict(delta: float, sign_consistency: int, n: int) -> str:
    """The registered decision rule, applied to a mean gap and its seed sign-consistency."""
    a = abs(delta)
    if a < FLOOR:
        return f"NULL (|{delta:+.4f}| < floor {FLOOR})"
    if a < P90:
        ok = sign_consistency >= max(7, int(np.ceil(0.875 * n)))
        return (
            f"{'RESOLVABLE-MARGINAL' if ok else 'NULL (marginal, sign-inconsistent)'} "
            f"(floor <= |{delta:+.4f}| < p90 {P90}; {sign_consistency}/{n} same sign)"
        )
    return f"RESOLVABLE (|{delta:+.4f}| >= p90 {P90})"


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    data = json.loads(open(src).read())
    arms = data["arms"]
    n = data["n_seeds"]

    print("building the shipped seed-averaged surfaces (this is the slow part)...", flush=True)
    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    surfaces = {w: TimeSurface(tri, target_wpm=w) for w in EVAL_WPMS}

    bi_f = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
    sk_f = load_frequencies(str(production_corpus_dir(None) / "1-skip31.txt"))
    km = KmStats(bi_f, sk_f, tri)

    out: dict = {
        "band": data["band"],
        "n_seeds": n,
        "floor_ms_per_char": FLOOR,
        "p90_ms_per_char": P90,
        "eval_wpms": list(EVAL_WPMS),
        "note": "all ms/char are on the SHIPPED seed-averaged TimeSurface (bigram+trigram, seeds 0-2)",
        "per_arm": {},
        "vs_control": {},
        "degeneracy_check": {},
    }

    # --- per-arm: winners, their ms/char curves on the shipped surface, and gauges ------------
    curves: dict[str, dict[int, dict[float, float]]] = {}
    for arm, blk in arms.items():
        curves[arm] = {}
        boards = []
        for row in blk["per_seed"]:
            lay = row["layout"]
            boards.append(lay)
            curves[arm][row["seed"]] = {w: surfaces[w].card(lay).ms_per_char for w in EVAL_WPMS}
        gauges = [km.stats(b) for b in boards]
        out["per_arm"][arm] = {
            "objective": blk["objective"],
            "layouts": boards,
            "n_distinct_layouts": len(set(boards)),
            "ms_per_char_by_wpm": {
                f"{w:g}": {
                    "mean": mean(curves[arm][s][w] for s in curves[arm]),
                    "sd": stdev([curves[arm][s][w] for s in curves[arm]]) if n > 1 else 0.0,
                    "min": min(curves[arm][s][w] for s in curves[arm]),
                    "max": max(curves[arm][s][w] for s in curves[arm]),
                }
                for w in EVAL_WPMS
            },
            "gauges_mean": {g: mean(x[g] for x in gauges) for g in STAT_NAMES},
            "gauges_sd": {g: (stdev([x[g] for x in gauges]) if n > 1 else 0.0) for g in STAT_NAMES},
        }

    # --- (a) does the LAYOUT differ from control, per matched seed? ---------------------------
    ctrl = arms["control90"]["per_seed"]
    ctrl_by_seed = {r["seed"]: r["layout"] for r in ctrl}
    for arm, blk in arms.items():
        if arm == "control90":
            continue
        same = 0
        hamming = []
        for row in blk["per_seed"]:
            c = ctrl_by_seed[row["seed"]]
            if row["layout"] == c:
                same += 1
            hamming.append(sum(1 for x, y in zip(row["layout"], c, strict=True) if x != y))
        out["vs_control"][arm] = {
            "identical_layout_seeds": f"{same}/{n}",
            "hamming_to_control_per_seed": hamming,
            "hamming_mean": mean(hamming),
        }

    # --- degeneracy check: is rawminimax byte-identical to control? ---------------------------
    if "rawminimax" in arms:
        raw = {r["seed"]: r["layout"] for r in arms["rawminimax"]["per_seed"]}
        ident = sum(1 for s in ctrl_by_seed if raw.get(s) == ctrl_by_seed[s])
        out["degeneracy_check"] = {
            "rawminimax_identical_to_control_seeds": f"{ident}/{n}",
            "confirms_prediction": ident == n,
        }

    # --- (b)/(c): the gap at each pace, paired by seed ----------------------------------------
    for arm in arms:
        if arm == "control90":
            continue
        per_wpm = {}
        for w in EVAL_WPMS:
            d = [curves[arm][s][w] - curves["control90"][s][w] for s in sorted(curves[arm])]
            m = mean(d)
            sign = sum(1 for x in d if (x < 0) == (m < 0))
            per_wpm[f"{w:g}"] = {
                "delta_ms_per_char_mean": m,
                "delta_sd": stdev(d) if n > 1 else 0.0,
                "delta_per_seed": d,
                "sign_consistent_seeds": f"{sign}/{n}",
                "verdict": verdict(m, sign, n),
            }
        out["vs_control"][arm]["delta_vs_control_by_wpm"] = per_wpm

    # --- (c) proper: BEST-OF-ARM cross-evaluation (the user's actual question) ----------------
    # Take each arm's single best board BY ITS OWN objective (what a user would ship) and
    # compare the whole set at every pace. This is the "does a 90-optimized layout regress at
    # 120" read, decoupled from per-seed search noise.
    best_of = {}
    for arm, blk in arms.items():
        row = min(blk["per_seed"], key=lambda r: r["arm_fitness"])
        best_of[arm] = row["layout"]
    out["best_of_arm"] = {
        "layouts": best_of,
        "ms_per_char_by_wpm": {
            arm: {f"{w:g}": surfaces[w].card(lay).ms_per_char for w in EVAL_WPMS}
            for arm, lay in best_of.items()
        },
        "gauges": {arm: km.stats(lay) for arm, lay in best_of.items()},
    }
    bc = out["best_of_arm"]["ms_per_char_by_wpm"]
    out["best_of_arm"]["delta_vs_control90"] = {
        arm: {w: bc[arm][w] - bc["control90"][w] for w in bc[arm]}
        for arm in bc
        if arm != "control90"
    }

    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    # --- printed summary ----------------------------------------------------------------------
    print(f"\n{'=' * 78}\n(a) LAYOUT IDENTITY vs control90 (matched seeds)\n{'=' * 78}")
    for arm, v in out["vs_control"].items():
        print(f"  {arm:12s} identical: {v['identical_layout_seeds']:6s}  mean hamming: {v['hamming_mean']:.2f}/30")
    print(f"\n  degeneracy check: rawminimax == control on "
          f"{out['degeneracy_check'].get('rawminimax_identical_to_control_seeds')} seeds")

    print(f"\n{'=' * 78}\n(b)/(c) GAP vs control90 in ms/char, SHIPPED surface, paired by seed\n{'=' * 78}")
    for arm, v in out["vs_control"].items():
        if "delta_vs_control_by_wpm" not in v:
            continue
        print(f"\n  {arm}:")
        for w, d in v["delta_vs_control_by_wpm"].items():
            print(f"    wpm {w:>4s}: d={d['delta_ms_per_char_mean']:+.4f} "
                  f"(sd {d['delta_sd']:.4f}, {d['sign_consistent_seeds']})  {d['verdict']}")

    print(f"\n{'=' * 78}\nARM ms/char ON SHIPPED SURFACE (mean over {n} seeds +- sd)\n{'=' * 78}")
    hdr = "  " + "arm".ljust(12) + "".join(f"{f'wpm{w:g}':>20s}" for w in EVAL_WPMS)
    print(hdr)
    for arm, v in out["per_arm"].items():
        cells = "".join(
            f"{v['ms_per_char_by_wpm'][f'{w:g}']['mean']:>13.4f}+-{v['ms_per_char_by_wpm'][f'{w:g}']['sd']:<5.3f}"
            for w in EVAL_WPMS
        )
        print(f"  {arm:12s}{cells}")

    print(f"\n{'=' * 78}\nBEST-OF-ARM cross-evaluation (each arm's own best board)\n{'=' * 78}")
    for arm, lay in best_of.items():
        cells = "".join(f"{bc[arm][f'{w:g}']:>12.4f}" for w in EVAL_WPMS)
        print(f"  {arm:12s} {lay}{cells}")

    print(f"\n{'=' * 78}\nGAUGE DRIFT (mean +- sd over seeds; drift is material only if > within-arm sd)\n{'=' * 78}")
    watch = ("sfb", "roll", "sr-roll", "alt")
    print("  " + "arm".ljust(12) + "".join(f"{g:>18s}" for g in watch))
    for arm, v in out["per_arm"].items():
        print(f"  {arm:12s}" + "".join(
            f"{v['gauges_mean'][g]:>11.3f}+-{v['gauges_sd'][g]:<5.3f}" for g in watch
        ))

    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
