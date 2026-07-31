"""Does the combined gauge REORDER the incumbent field, or only pick a different champion?

MODELNORM-1 registered "0 discordant pairs" -- normalization reorders nothing. That is a claim
about the SCALE (an affine positive rescale cannot reorder WITHIN one model). This arm's P7 asks a
DIFFERENT question: does WEIGHTING across models reorder? Those are not the same claim and must be
scored separately, or a true finding gets attached to the wrong question.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from keybo.testkit import assert_module_under

assert_module_under("keybo", "/tmp/normgauge")

HERE = Path(__file__).resolve().parent
FIELD = [
    "keybo-lsb", "keybo-lsb+lm", "keybo-c30m", "flagship-c3", "archive-1843", "archive-1846",
    "lsb-sib", "qwerty30m", "graphite", "semimak", "arm-A", "arm-B",
]


def discordant(a: dict[str, float], b: dict[str, float], names: list[str]) -> list[tuple]:
    """Pairs whose ORDER differs between two scorings (both higher-is-better)."""
    out = []
    for x, y in itertools.combinations(names, 2):
        if (a[x] - a[y]) * (b[x] - b[y]) < 0:
            out.append((x, y, a[x] - a[y], b[x] - b[y]))
    return out


def main() -> int:
    report = json.loads((HERE / "blend-report.json").read_text())
    rows = report["normalized_rows"]
    ms = {n: report["gauge_table"]["rows"][n]["ms_per_char"] for n in FIELD}

    scorings = {
        "registered": {n: rows[n]["blend:registered"] for n in FIELD},
        "equal": {n: rows[n]["blend:equal"] for n in FIELD},
        "drop-pool": {n: rows[n]["blend:drop-pool"] for n in FIELD},
        "solo-AALTO": {n: rows[n]["aalto-n"] for n in FIELD},
        "solo-COMMUNITY": {n: rows[n]["comm-n"] for n in FIELD},
        "solo-POOL": {n: rows[n]["pool-n"] for n in FIELD},
        # ms/char is lower-is-better, so negate to put it on a higher-is-better footing.
        "ms/char (negated)": {n: -ms[n] for n in FIELD},
        # The RAW (un-normalized) mean of the three fits, also negated: this is the comparison
        # MODELNORM made, and it is the one that must show 0 discordant pairs.
        "raw mean fit (negated)": {
            n: -float(np.mean([rows[n]["fit_ms"][p] for p in ("AALTO", "COMMUNITY", "POOL")]))
            for n in FIELD
        },
    }

    print("=== A) WITHIN one model: normalization is an affine rescale, so it CANNOT reorder ===")
    for pool, gauge in (("AALTO", "aalto-n"), ("COMMUNITY", "comm-n"), ("POOL", "pool-n")):
        raw = {n: -rows[n]["fit_ms"][pool] for n in FIELD}      # negated: lower fit is better
        norm = {n: rows[n][gauge] for n in FIELD}
        bad = discordant(raw, norm, FIELD)
        rho = spearmanr([raw[n] for n in FIELD], [norm[n] for n in FIELD]).statistic
        print(f"  {pool:10s} discordant pairs vs its own raw fit: {len(bad)}  (spearman {rho:+.6f})")
    print("  => reproduces MODELNORM-1's null. This is the claim that survives.")

    print()
    print("=== B) ACROSS weightings: does the WEIGHT reorder the field? (a different question) ===")
    labels = list(scorings)
    print(f"  {'pair':44s} {'discordant':>10s} {'of':>4s} {'spearman':>9s}")
    total_pairs = len(FIELD) * (len(FIELD) - 1) // 2
    for x, y in itertools.combinations(labels, 2):
        bad = discordant(scorings[x], scorings[y], FIELD)
        rho = spearmanr([scorings[x][n] for n in FIELD], [scorings[y][n] for n in FIELD]).statistic
        print(f"  {x + ' vs ' + y:44s} {len(bad):10d} {total_pairs:4d} {rho:+9.4f}")

    print()
    print("=== C) the discordant pairs the REGISTERED weighting creates vs solo-AALTO ===")
    for x, y, da, db in discordant(scorings["solo-AALTO"], scorings["registered"], FIELD):
        print(f"  {x:14s} vs {y:14s}  aalto-n gap {da:+.4f}  registered gap {db:+.4f}")

    print()
    print("=== D) leaderboard under each weighting (top 5) ===")
    for label, score in scorings.items():
        top = sorted(FIELD, key=lambda n: -score[n])[:5]
        print(f"  {label:24s} {' > '.join(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
