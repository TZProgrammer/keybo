"""lmscissor (b)+(d): re-estimate the Aalto row-travel surface, disaggregated by
LOWER-FINGER IDENTITY and ADJACENCY — the two dimensions the shipped predicate collapses.

Same estimand as BADSCISSOR-1's bs01_surface.py (deliberately, so numbers are comparable):
  relative_excess(cell) = mean(dur | cell, wpm-bucket) / mean(dur | same-row two-finger
                          baseline, same bucket) - 1, averaged over supported buckets.
Same corrections: punctuation dropped from cells AND baseline, space dropped, >=3 distinct
bigrams, >=200 raw samples, >=20 participants.

The question: the shipped predicate excludes ALL weak-finger-on-TOP pairs on the strength of a
"-0.0179 at n=1.64M" figure. That figure is the aggregate of the weak-on-top dy=2 class. If the
cheapness is driven by LOWER-KEY-IS-INDEX (the spec's own §2.2 contrast is literally
"lower key = index finger"), then excluding weak-on-top pairs whose lower key is MIDDLE or RING
generalizes past the evidence — and `bl` (lower=middle, upper=pinky, dy=2) is exactly such a pair.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
OUT = Path("/tmp/lmscissor_surface.json")

PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
BUCKET_WIDTH = 20
MIN_RAW = 200
MIN_PIDS = 20
MIN_BIGRAMS = 3

_ABS_COL_FINGER = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
_DEX = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}
_ROW = {3: "top", 2: "home", 1: "bottom"}


def finger(x: int) -> str:
    return _ABS_COL_FINGER[abs(x)]


def bucket_of(wpm: int) -> int | None:
    for b in BUCKETS:
        if b - BUCKET_WIDTH / 2 <= wpm < b + BUCKET_WIDTH / 2:
            return b
    return None


def parse_positions(field: str):
    """'((-4, 3), (-5, 2))' -> ((-4,3),(-5,2)).  Fast path for the fixed shape."""
    nums = field.replace("(", " ").replace(")", " ").replace(",", " ").split()
    if len(nums) != 4:
        return None
    try:
        a, b, c, d = (int(n) for n in nums)
    except ValueError:
        return None
    return (a, b), (c, d)


def parse_samples(tokens):
    """'(92, 176, 100001, 0)' -> (wpm, duration, pid). Manual split (ast.literal_eval is 10x slower)."""
    for tok in tokens:
        tok = tok.strip()
        if not tok.startswith("("):
            continue
        parts = tok[1:-1].split(",") if tok.endswith(")") else tok[1:].split(",")
        if len(parts) < 3:
            continue
        try:
            yield int(parts[0]), float(parts[1]), int(parts[2])
        except ValueError:
            continue


def is_adjacent_cols(ax: int, bx: int) -> bool:
    """Mirror of classify.is_adjacent for two same-hand distinct-finger columns."""
    if {abs(ax), abs(bx)} == {6, 4}:
        return True
    return abs(abs(ax) - abs(bx)) == 1


def main() -> None:
    cells: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    baseline: dict[int, list] = defaultdict(list)
    cell_bigrams: dict[str, set] = defaultdict(set)
    meta_of: dict[str, dict] = {}
    n_rows = 0

    with open(TSV, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            n_rows += 1
            ngram = parts[2]
            if PUNCT & set(ngram) or " " in ngram:
                continue
            pos = parse_positions(parts[1])
            if pos is None:
                continue
            a, b = pos
            ax, ay = a
            bx, by = b
            if ax == 0 or bx == 0:
                continue
            if (ax > 0) != (bx > 0):
                continue
            fa, fb = finger(ax), finger(bx)
            if fa == fb:
                continue
            samples = list(parse_samples(parts[4:]))
            if not samples:
                continue

            if ay == by:  # same-row two-finger -> baseline
                for wpm, dur, pid in samples:
                    bk = bucket_of(wpm)
                    if bk is not None:
                        baseline[bk].append((dur, pid))
                continue

            dy = abs(ay - by)
            weakest = min((fa, fb), key=lambda f: _DEX[f])
            strongest = max((fa, fb), key=lambda f: _DEX[f])
            weak_y = ay if fa == weakest else by
            strong_y = by if fa == weakest else ay
            weak_is_lower = weak_y < strong_y
            lower_kind = fa if ay < by else fb
            upper_kind = fb if ay < by else fa
            adjacent = is_adjacent_cols(ax, bx)
            pair = f"{strongest}-{weakest}"

            # The three cell keys this probe needs.
            keys = {
                # A: the shipped predicate's own view
                f"A|{'weakLOWER' if weak_is_lower else 'weakTOP'}|dy{dy}",
                # B: split it by adjacency — the dimension the -0.0179 evidence never separated
                f"B|{'weakLOWER' if weak_is_lower else 'weakTOP'}|dy{dy}|"
                f"{'adj' if adjacent else 'nonadj'}",
                # C: split the weak-on-TOP class by WHICH FINGER IS LOWER (index vs not)
                f"C|{'weakLOWER' if weak_is_lower else 'weakTOP'}|dy{dy}|"
                f"lower={'index' if lower_kind == 'index' else 'nonindex'}",
                # D: the fully explicit class — lower finger x upper finger x dy
                f"D|lower={lower_kind}|upper={upper_kind}|dy{dy}",
                # E: pair x orientation x dy (for the middle-pinky story specifically)
                f"E|{pair}|{'weakLOWER' if weak_is_lower else 'weakTOP'}|dy{dy}",
            }
            for key in keys:
                cell_bigrams[key].add(ngram)
                meta_of.setdefault(
                    key,
                    {
                        "pair": pair,
                        "dy": dy,
                        "weak_is_lower": weak_is_lower,
                        "adjacent": adjacent,
                        "lower_kind": lower_kind,
                        "upper_kind": upper_kind,
                    },
                )
                bucket_list = cells[key]
                for wpm, dur, pid in samples:
                    bk = bucket_of(wpm)
                    if bk is not None:
                        bucket_list[bk].append((dur, pid))

    base_ms = {
        bk: sum(d for d, _ in v) / len(v) for bk, v in baseline.items() if len(v) >= MIN_RAW
    }
    print(f"rows read: {n_rows}", flush=True)
    print(f"baseline buckets: { {k: round(v,2) for k,v in sorted(base_ms.items())} }", flush=True)

    out = {}
    for key, by_bucket in sorted(cells.items()):
        rel, n_raw, pids = [], 0, set()
        for bk in sorted(by_bucket):
            if bk not in base_ms:
                continue
            v = by_bucket[bk]
            d = [x for x, _ in v]
            p = {q for _, q in v}
            n_raw += len(d)
            pids |= p
            if len(d) >= MIN_RAW and len(p) >= MIN_PIDS:
                rel.append(sum(d) / len(d) / base_ms[bk] - 1.0)
        nbg = len(cell_bigrams[key])
        measured = bool(n_raw >= MIN_RAW and len(pids) >= MIN_PIDS and nbg >= MIN_BIGRAMS and rel)
        out[key] = {
            "status": "MEASURED" if measured else "UNMEASURED",
            "rel": (sum(rel) / len(rel)) if rel else None,
            "n_buckets": len(rel),
            "n_raw": n_raw,
            "n_pids": len(pids),
            "n_bigrams": nbg,
            "bigrams": sorted(cell_bigrams[key])[:30],
            **meta_of[key],
        }

    json.dump({"baseline_ms": base_ms, "cells": out}, open(OUT, "w"), indent=2)

    def show(prefix: str, title: str) -> None:
        print(f"\n{'='*100}\n{title}\n{'='*100}", flush=True)
        rows_ = [(k, v) for k, v in out.items() if k.startswith(prefix)]
        rows_.sort(key=lambda kv: -(kv[1]["rel"] if kv[1]["rel"] is not None else -9))
        print(f"{'cell':<52}{'rel':>10}{'n_raw':>12}{'pids':>8}{'bigrams':>9}  status")
        for k, v in rows_:
            r = f"{v['rel']:+.4f}" if v["rel"] is not None else "  n/a  "
            print(
                f"{k:<52}{r:>10}{v['n_raw']:>12}{v['n_pids']:>8}{v['n_bigrams']:>9}  {v['status']}",
                flush=True,
            )

    show("A|", "A. THE SHIPPED PREDICATE'S OWN VIEW (orientation x dy)")
    show("B|", "B. SPLIT BY ADJACENCY — did the weak-on-TOP exclusion generalize past its evidence?")
    show("C|", "C. SPLIT THE WEAK-ON-TOP CLASS BY WHETHER THE LOWER KEY IS THE INDEX FINGER")
    show("D|", "D. FULLY EXPLICIT: lower finger x upper finger x dy")
    show("E|", "E. PER FINGER-PAIR x orientation x dy (the middle-pinky story)")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
