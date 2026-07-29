"""FIND-phase: EXACT per-surface-cell training support for AALTO and COMMUNITY.

The 643x in my brief is scissor-neighbourhood + covered-pair-filtered. For a WEIGHT I need
support on the frame the gauge actually uses: the 31^3 surface cells, weighted by the
corpus mass that lands on them. This walks the raw stroke tables ONCE and counts, per
(slot_a, slot_b, slot_c) cell, the number of TRAINING SAMPLES each source contributed.

Cell semantics: a stroke row's `positions` is a tuple of geometry positions; the surface is
indexed by SLOT INDEX in ROW_STAGGERED_30.slots order, with space at 30. Same mapping the
feature pipeline uses, so a row maps to exactly one cell.

Scope stated: rows are filtered exactly as the training recipe filters them
(wpm_threshold=40 per CELL_KW wpm_lo, min_samples=10 per min_cell_samples) so the counted
support is the support the FIT saw, not the support the raw file holds.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

from keybo.testkit import assert_module_under
assert_module_under("keybo", "/tmp/normgauge")
from keybo.geometry import ROW_STAGGERED_30

OUT = Path("/tmp/normgauge/drivers-normgauge/support-cells.json")
E2E = Path("/local/home/zegertho/keybo-e2e")
COMM = Path("/local/home/zegertho/repos/keybo/data/community/processed")

# The exact source subsets scissorsupport identified by EXACT PRACTICE-TERM KEY-SET MATCH.
AALTO_LABELS = {"azerty", "dvorak", "qwerty", "qwertz"}
COMM_LABELS = {"colemak@rowStagger#alite", "custom-aa426873@rowStagger#vg",
               "custom-d42a1f92@rowStagger#ddn", "mtgap-variant@rowStagger#richarddavison"}
WPM_LO, MIN_SAMPLES = 40, 10

SLOT_OF = {p: n for n, p in enumerate(ROW_STAGGERED_30.slots)}
SLOT_OF[ROW_STAGGERED_30.space_position] = 30


def scan(path: Path, keep_labels: set[str], ngram_len: int = 3):
    """(counts[29791], rows[29791], n_rows_kept, n_rows_seen, labels_seen)."""
    counts = np.zeros(29791, dtype=np.int64)
    rows_per_cell = np.zeros(29791, dtype=np.int64)
    kept = seen = 0
    labels = defaultdict(int)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            layout, pos_str, ngram, _freq, *tokens = parts
            seen += 1
            labels[layout] += 1
            if layout not in keep_labels or len(ngram) != ngram_len:
                continue
            # count samples at/above the wpm floor without ast.literal_eval per token
            n = 0
            for tok in tokens:
                # token form "(wpm, interval, pid, flag)"
                c = tok.find(",")
                if c <= 1:
                    continue
                try:
                    if int(tok[1:c]) >= WPM_LO:
                        n += 1
                except ValueError:
                    continue
            if n < MIN_SAMPLES:
                continue
            try:
                positions = tuple(tuple(t) for t in json.loads(pos_str.replace("(", "[").replace(")", "]")))
            except Exception:
                continue
            try:
                a, b, c3 = (SLOT_OF[p] for p in positions)
            except KeyError:
                continue           # a position outside the 31-slot board
            idx = a * 961 + b * 31 + c3
            counts[idx] += n
            rows_per_cell[idx] += 1
            kept += 1
    return counts, rows_per_cell, kept, seen, dict(labels)


def main() -> int:
    t0 = time.time()
    res = {}
    for name, path, labels in (
        ("AALTO", E2E / "tristrokes31_cond_v1.tsv", AALTO_LABELS),
        ("COMMUNITY", COMM / "tristrokes_last_community.tsv", COMM_LABELS),
    ):
        counts, rows, kept, seen, seen_labels = scan(path, labels)
        res[name] = dict(
            path=str(path), samples=int(counts.sum()), rows_kept=kept, rows_seen=seen,
            cells_covered=int((counts > 0).sum()),
            labels_kept=sorted(labels), labels_in_file=seen_labels,
        )
        np.save(f"/tmp/normgauge/drivers-normgauge/support-{name}.npy", counts)
        np.save(f"/tmp/normgauge/drivers-normgauge/rows-{name}.npy", rows)
        print(f"[{time.time()-t0:6.1f}s] {name}: samples={counts.sum():,} rows_kept={kept:,} "
              f"of {seen:,} cells_covered={int((counts>0).sum()):,}/29791", flush=True)
    a, c = res["AALTO"]["samples"], res["COMMUNITY"]["samples"]
    res["ratio_whole_surface"] = a / c
    print(f"\nwhole-surface sample ratio AALTO/COMMUNITY = {a/c:.2f}x")
    OUT.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
