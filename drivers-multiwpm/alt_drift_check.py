"""Is the `alt` drop (45.5 -> ~41.5) an OBJECTIVE effect, or a search-depth artifact?

Both band arms and the wpm=120 endpoint arm drop kmstats `alt` by ~4 points vs the wpm=90
control, and it replicates across the 4-point and 7-point bands. That looks like a real
signature — but there is a confound: the band arms evaluate 4-7 table lookups per candidate
where the control evaluates 1, and the control's `alt` sd is 0.92 while the band arms' is
4.7-5.5. A four-point shift with a five-point sd is not obviously a shift.

The discriminating test: `alt` is a property of the BOARD, and every arm's boards live at a
different fitness level. So regress `alt` on the board's shipped-surface ms/char across ALL
arms' boards pooled. If arm identity adds nothing once fitness is controlled, the "drift" is
just "these arms found deeper optima", not "the range objective prefers less alternation".

Also reports the CONTROL arm's own alt-vs-fitness slope, which is the within-arm null.

Usage: alt_drift_check.py <arms_band4.json> <arms_band7.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from statistics import mean, stdev

import numpy as np
from scipy import stats

from keybo.analysis.kmstats import KmStats
from keybo.analysis.timecard import TimeSurface
from keybo.data.corpus import load_frequencies, production_corpus_dir

WATCH = ("sfb", "roll", "sr-roll", "alt")


def main() -> int:
    out_path = sys.argv[-1]
    boards: list[tuple[str, str, int]] = []  # (arm, layout, band_id)
    for band_id, path in enumerate(sys.argv[1:-1]):
        data = json.loads(open(path).read())
        for arm, blk in data["arms"].items():
            if arm == "rawminimax":
                continue  # byte-identical to control; would double-weight the control
            for row in blk["per_seed"]:
                boards.append((arm, row["layout"], band_id))

    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    bi = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
    sk = load_frequencies(str(production_corpus_dir(None) / "1-skip31.txt"))
    km = KmStats(bi, sk, tri)
    surf = TimeSurface(tri, target_wpm=90.0)

    rows = []
    for arm, lay, band_id in boards:
        g = km.stats(lay)
        rows.append(
            {
                "arm": arm,
                "layout": lay,
                "band": band_id,
                "ms_per_char_90": surf.card(lay).ms_per_char,
                **{k: g[k] for k in WATCH},
            }
        )

    ms = np.array([r["ms_per_char_90"] for r in rows])
    out: dict = {"n_boards": len(rows), "per_gauge": {}}

    print(f"pooled boards: {len(rows)} (rawminimax excluded as a control duplicate)\n")
    for gauge in WATCH:
        y = np.array([r[gauge] for r in rows])
        # 1. does fitness alone explain the gauge?
        lr = stats.linregress(ms, y)
        resid = y - (lr.intercept + lr.slope * ms)
        # 2. after removing the fitness trend, does ARM still separate?
        by_arm_resid = {}
        for arm in sorted({r["arm"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["arm"] == arm]
            by_arm_resid[arm] = [float(resid[i]) for i in idx]
        groups = list(by_arm_resid.values())
        f_stat, f_p = stats.f_oneway(*groups)
        # raw (uncontrolled) arm means, for the contrast
        raw_by_arm = {
            arm: [y[i] for i, r in enumerate(rows) if r["arm"] == arm]
            for arm in sorted({r["arm"] for r in rows})
        }
        out["per_gauge"][gauge] = {
            "fitness_slope": lr.slope,
            "fitness_r2": lr.rvalue**2,
            "fitness_p": lr.pvalue,
            "anova_on_residuals_F": float(f_stat),
            "anova_on_residuals_p": float(f_p),
            "raw_arm_mean": {a: mean(v) for a, v in raw_by_arm.items()},
            "raw_arm_sd": {a: (stdev(v) if len(v) > 1 else 0.0) for a, v in raw_by_arm.items()},
            "residual_arm_mean": {a: mean(v) for a, v in by_arm_resid.items()},
        }
        print(f"[{gauge}]")
        print(f"  vs ms/char@90: slope={lr.slope:+.4f}  R2={lr.rvalue**2:.4f}  p={lr.pvalue:.2e}")
        print(f"  ANOVA on fitness-residuals, arm as factor: F={f_stat:.3f}  p={f_p:.4f}"
              f"   -> {'arm STILL separates' if f_p < 0.05 else 'arm adds NOTHING once fitness is controlled'}")
        print("  raw arm means:      " + "  ".join(f"{a}={mean(v):.3f}" for a, v in raw_by_arm.items()))
        print("  residual arm means: " + "  ".join(f"{a}={mean(v):+.3f}" for a, v in by_arm_resid.items()))
        print()

    with open(out_path, "w") as f:
        json.dump({"summary": out, "boards": rows}, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
