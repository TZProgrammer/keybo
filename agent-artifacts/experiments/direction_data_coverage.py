"""DIRECTION-DATA-1: does the TRAINING DATA contain the cell the user is asking about?

Section 5 of this investigation could only say "our data may not speak to this cell". The
tables are on disk after all, so the question is answerable rather than merely flagged.

The user's cell, stated as a geometric predicate on the recorded key positions (the tables
store positions, so no layout assumption is needed): three keys, one hand, ONE ROW, THREE
DISTINCT fingers, and the direction of travel by |column| reverses between the two
constituent bigrams. Count the observed trigram samples that satisfy it, per source layout.

Two facts this establishes that no amount of re-reading the null could:
  * whether the fitted trigram model has ANY direct evidence about same-row three-finger
    reversals, and how much;
  * whether the 2026-07-05 "redirects are not super-additive" probe -- which read qwerty
    rows only -- could have sampled the cell, as opposed to merely admitting it.

Reads the tables STREAMING and counts only; nothing is re-fitted.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

TRISTROKES = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"


def classify(a, b, c) -> str | None:
    """Bucket a recorded position triple the way the 2026-07-05 probe bucketed it."""
    if 0 in (a[0], b[0], c[0]):
        return None  # thumb/space
    if not (G.hand(a[0]) == G.hand(b[0]) == G.hand(c[0]) != 0):
        return None  # not a same-hand run
    d1, d2 = abs(b[0]) - abs(a[0]), abs(c[0]) - abs(b[0])
    if d1 == 0 or d2 == 0:
        return "run-flat"
    return "run-continue" if (d1 > 0) == (d2 > 0) else "run-redirect"


def main() -> int:
    assert_module_under("keybo", REPO)
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(TRISTROKES)
    if not path.is_file():
        raise SystemExit(f"training table not found: {path}")

    # rows and SAMPLES per bucket; the sample count is what a fit actually sees.
    rows = defaultdict(int)
    samples = defaultdict(int)
    # the user's cell: run-redirect AND same row AND three distinct fingers
    cell_rows = defaultdict(int)
    cell_samples = defaultdict(int)
    cell_paths: dict[str, int] = {}
    layouts = set()
    total_rows = 0

    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            layout = parts[0]
            try:
                positions = ast.literal_eval(parts[1])
            except (ValueError, SyntaxError):
                continue
            if not (isinstance(positions, tuple) and len(positions) == 3):
                continue
            total_rows += 1
            layouts.add(layout)
            a, b, c = positions
            bucket = classify(a, b, c)
            if bucket is None:
                continue
            n_samples = len(parts) - 4  # the per-observation tuples
            rows[(layout, bucket)] += 1
            samples[(layout, bucket)] += n_samples
            if bucket != "run-redirect":
                continue
            fingers = tuple(G.finger(p[0]).name for p in (a, b, c))
            if a[1] == b[1] == c[1] and len(set(fingers)) == 3:
                cell_rows[layout] += 1
                cell_samples[layout] += n_samples
                key = "->".join(f[1] for f in fingers)
                cell_paths[key] = cell_paths.get(key, 0) + n_samples

    print(f"table: {path}")
    print(f"rows={total_rows:,}  layouts={sorted(layouts)} (n={len(layouts)})")
    print(f"\n{'layout':10s} {'bucket':14s} {'rows':>8s} {'samples':>12s}")
    for (layout, bucket), n in sorted(rows.items()):
        print(f"{layout:10s} {bucket:14s} {n:>8,} {samples[(layout, bucket)]:>12,}")

    print("\nTHE USER'S CELL — run-redirect AND same-row AND three distinct fingers:")
    if not cell_rows:
        print("  ZERO rows in the entire training table. The fitted trigram model has NO")
        print("  direct evidence about this pattern; its price there is EXTRAPOLATION.")
    for layout in sorted(cell_rows):
        print(f"  {layout:10s} rows={cell_rows[layout]:,}  samples={cell_samples[layout]:,}")
    if cell_paths:
        print(
            "  finger paths present (by samples): "
            + ", ".join(f"{k}={v:,}" for k, v in sorted(cell_paths.items(), key=lambda kv: -kv[1]))
        )
        print(
            f"  is the user's R->I->M present? "
            f"{'YES' if 'R->I->M' in cell_paths else 'NO'}"
            + (f" ({cell_paths['R->I->M']:,} samples)" if "R->I->M" in cell_paths else "")
        )
    qw_cell = cell_samples.get("qwerty", 0)
    print(
        f"\n  qwerty-only (what the 2026-07-05 additivity probe could read): "
        f"{cell_rows.get('qwerty', 0):,} rows / {qw_cell:,} samples"
    )
    print(
        "  ⇒ that probe's run-redirect null "
        + ("DID have access to the cell." if qw_cell else "COULD NOT have sampled the cell.")
    )

    out = {
        "table": str(path),
        "total_rows": total_rows,
        "layouts": sorted(layouts),
        "per_layout_bucket": {
            f"{k[0]}|{k[1]}": {"rows": v, "samples": samples[k]} for k, v in sorted(rows.items())
        },
        "users_cell": {
            "rows_per_layout": dict(cell_rows),
            "samples_per_layout": dict(cell_samples),
            "finger_paths_by_samples": cell_paths,
            "qwerty_samples": qw_cell,
        },
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_data_coverage.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
