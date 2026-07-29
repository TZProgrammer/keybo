"""Exhaustive predicate census for bad_scissor, ASSERTED against the module's own docstring.

The docstring publishes a table of counts. Rather than retype those numbers (two of two
hand-transcriptions by a prior arm were wrong), this driver PARSES the table out of the
docstring and asserts the enumerated behaviour matches it. A mismatch is a
disclosure-vs-behaviour defect, which is the highest-value finding available.

Everything is enumerated over all ordered slot pairs on the shipped geometry — no corpus,
no layout, so this is a pure statement about the predicate.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

from keybo.analysis import bad_scissor as BS
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31
from keybo.testkit import assert_module_under

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)
# Positive control on the enumeration itself: the predicate must be reachable at all.
assert_module_under("keybo.analysis.bad_scissor", ROOT)


def parse_docstring_table(doc: str) -> dict[str, int]:
    """Pull the ``label  count`` rows out of the docstring's RST simple table."""
    rows: dict[str, int] = {}
    for line in doc.splitlines():
        m = re.match(r"^(\S.*?)\s{2,}(\d+)\s*$", line.rstrip())
        if m:
            label = re.sub(r"\s+", " ", m.group(1)).strip()
            rows[label] = int(m.group(2))
        else:
            m2 = re.match(r"^\s+(\.\.\.\S.*?)\s{2,}(\d+)\s*$", line.rstrip())
            if m2:
                label = re.sub(r"\s+", " ", m2.group(1)).strip()
                rows[label] = int(m2.group(2))
    return rows


def census(geometry) -> dict:
    slots = sorted(geometry.slots)
    ordered_all = list(itertools.product(slots, slots))
    ordered_distinct = [(a, b) for a, b in ordered_all if a != b]

    def wide(a, b) -> bool:
        return (
            C.same_hand(geometry, a, b)
            and not C.same_finger(geometry, a, b)
            and abs(a[1] - b[1]) == 2
        )

    narrow = {(a, b) for a, b in ordered_all if C.is_scissor(geometry, a, b)}
    wide_set = {(a, b) for a, b in ordered_all if wide(a, b)}
    bad = {(a, b) for a, b in ordered_all if BS.bad_scissor(geometry, a, b)}

    def kinds(a, b):
        return {BS._kind(geometry, a[0]), BS._kind(geometry, b[0])}

    dy = {}
    for a, b in bad:
        dy[abs(a[1] - b[1])] = dy.get(abs(a[1] - b[1]), 0) + 1

    def weak_on_top(a, b) -> bool:
        """The class the docstring says the exclusions all belong to."""
        _w, _wx, wy, sy, _s = BS._weak_and_strong(geometry, a, b)
        return wy > sy

    narrow_minus_bad = narrow - bad
    wide_minus_bad = wide_set - bad

    # Symmetry, exhaustively, over ALL ordered pairs.
    asym = [(a, b) for a, b in ordered_all
            if BS.bad_scissor(geometry, a, b) != BS.bad_scissor(geometry, b, a)]
    asym_finger = [(a, b) for a, b in ordered_all
                   if BS.bad_scissor_finger(geometry, a, b) != BS.bad_scissor_finger(geometry, b, a)]
    asym_cell = [(a, b) for a, b in ordered_all
                 if BS.bad_scissor_cell(geometry, a, b) != BS.bad_scissor_cell(geometry, b, a)]

    fingers: dict[str, int] = {}
    cells: dict[str, int] = {}
    cell_to_pairs: dict[str, set] = {}
    for a, b in bad:
        f = BS.bad_scissor_finger(geometry, a, b)
        fingers[f] = fingers.get(f, 0) + 1
        c = BS.bad_scissor_cell(geometry, a, b)
        cells[c] = cells.get(c, 0) + 1
        cell_to_pairs.setdefault(c, set()).add((frozenset(kinds(a, b)), abs(a[1] - b[1])))

    adjacent = sum(1 for a, b in bad if C.is_adjacent(geometry, a, b))

    return {
        "n_slots": len(slots),
        "ordered_pairs_including_self": len(ordered_all),
        "ordered_pairs_distinct": len(ordered_distinct),
        "narrow_is_scissor": len(narrow),
        "narrow_dy_support": sorted({abs(a[1] - b[1]) for a, b in narrow}),
        "wide": len(wide_set),
        "bad": len(bad),
        "bad_dy_split": dict(sorted(dy.items())),
        "bad_dy_support": sorted(dy),
        "bad_middle_pinky": sum(1 for a, b in bad if kinds(a, b) == {"middle", "pinky"}),
        "bad_adjacent": adjacent,
        "bad_nonadjacent": len(bad) - adjacent,
        "narrow_minus_bad": len(narrow_minus_bad),
        "narrow_minus_bad_all_weak_on_top": all(weak_on_top(a, b) for a, b in narrow_minus_bad),
        "wide_minus_bad": len(wide_minus_bad),
        "wide_minus_bad_all_weak_on_top": all(weak_on_top(a, b) for a, b in wide_minus_bad),
        "bad_minus_narrow": len(bad - narrow),
        "bad_minus_wide": len(bad - wide_set),
        "bad_and_narrow": len(bad & narrow),
        "bad_and_wide": len(bad & wide_set),
        "symmetry_violations_predicate": len(asym),
        "symmetry_violations_finger": len(asym_finger),
        "symmetry_violations_cell": len(asym_cell),
        "fingers_charged": dict(sorted(fingers.items())),
        "index_ever_charged": any("index" in f for f in fingers),
        "cells": dict(sorted(cells.items())),
        "n_cells": len(cells),
        "cell_collisions": {c: len(v) for c, v in sorted(cell_to_pairs.items()) if len(v) > 1},
        "fingers_not_in_FINGER_ORDER": sorted(set(fingers) - set(BS.FINGER_ORDER)),
    }


def main() -> int:
    doc = BS.__doc__ or ""
    table = parse_docstring_table(doc)
    print("=== DOCSTRING TABLE AS PARSED (the disclosure) ===")
    for k, v in table.items():
        print(f"  {k!r:60s} = {v}")

    c = census(ROW_STAGGERED_30)
    print("\n=== ENUMERATED BEHAVIOUR on ROW_STAGGERED_30 ===")
    for k, v in c.items():
        print(f"  {k:42s} = {v}")

    # Map docstring labels -> computed values. Labels come from the parse, so a docstring
    # edit that renames a row shows up as a KeyError here rather than a silent skip.
    want = {
        "ordered position pairs on ``ROW_STAGGERED_30``": "ordered_pairs_including_self",
        "narrow ``is_scissor``": "narrow_is_scissor",
        "wide (same-hand, distinct-finger, dy == 2)": "wide",
        "``bad-scissor``": "bad",
        "...of which dy == 1": None,
        "...of which dy == 2": None,
        "...of which middle-pinky": "bad_middle_pinky",
        # NB: the docstring is a non-raw string, so its ``\\`` is ONE literal backslash, and
        # the parser collapses runs of whitespace. Getting this wrong produced two FALSE
        # "docstring row NOT FOUND" failures on the first run — my bug, not the module's.
        "narrow \\ bad (excluded: all weak-on-TOP)": "narrow_minus_bad",
        "wide \\ bad (excluded: all weak-on-TOP)": "wide_minus_bad",
    }
    print("\n=== DISCLOSURE vs BEHAVIOUR ===")
    failures = []
    checked = 0
    for label, key in want.items():
        if label not in table:
            failures.append(f"docstring row {label!r} NOT FOUND — parser or docstring drifted")
            continue
        claimed = table[label]
        if key is None:
            n = 1 if "dy == 1" in label else 2
            actual = c["bad_dy_split"].get(n, 0)
        else:
            actual = c[key]
        ok = claimed == actual
        checked += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {label!r:58s} claims {claimed:4d} · actual {actual:4d}")
        if not ok:
            failures.append(f"{label!r}: docstring claims {claimed}, code does {actual}")

    print(f"\n  ({checked} rows checked)")

    # Prose claims stated outside the table.
    print("\n=== PROSE CLAIMS ===")
    prose = [
        ("symmetric for all 900 pairs", c["symmetry_violations_predicate"] == 0),
        ("both index fingers always structurally uncharged",
         not c["index_ever_charged"]),
        ("every excluded narrow pair is weak-on-TOP", c["narrow_minus_bad_all_weak_on_top"]),
        ("every excluded wide pair is weak-on-TOP", c["wide_minus_bad_all_weak_on_top"]),
        ("supports are NOT nested (bad-narrow and narrow-bad both nonempty)",
         c["bad_minus_narrow"] > 0 and c["narrow_minus_bad"] > 0),
        ("no charged finger falls outside FINGER_ORDER",
         c["fingers_not_in_FINGER_ORDER"] == []),
        ("cell key does not collide (one (fingerpair,dy) class per key string)",
         c["cell_collisions"] == {}),
    ]
    for what, ok in prose:
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(f"prose: {what} is FALSE")

    # The four-row board must be refused.
    print("\n=== _check_geometry on a non-three-row board ===")
    print(f"  ROW_STAGGERED_31 rows = {sorted({y for _x, y in ROW_STAGGERED_31.slots})}")
    try:
        BS.BadScissor({}).share.__self__._check_geometry(ROW_STAGGERED_31)
        print("  ROW_STAGGERED_31 ACCEPTED")
        r31_refused = False
    except ValueError as e:
        print(f"  ROW_STAGGERED_31 REFUSED: {e}")
        r31_refused = True
    c["row_staggered_31_refused"] = r31_refused
    c31 = census(ROW_STAGGERED_31)
    print(f"  (census on 31 anyway: bad={c31['bad']}, slots={c31['n_slots']}, "
          f"dy={c31['bad_dy_split']}, sym_viol={c31['symmetry_violations_predicate']})")

    out = ROOT / "agent-artifacts/bsaudit/census_predicate.json"
    out.write_text(json.dumps(
        {"docstring_table": table, "row_staggered_30": c, "row_staggered_31": c31,
         "failures": failures}, indent=2, default=str))
    print(f"\nwrote {out}")

    if failures:
        print(f"\n!!! {len(failures)} DISCLOSURE/BEHAVIOUR MISMATCH(ES):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL disclosure rows and prose claims MATCH the enumerated behaviour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
