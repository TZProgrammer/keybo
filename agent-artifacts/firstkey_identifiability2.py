"""TASK-1 GATE, v2 — corrected universes.

v1 excluded a==b and b==c, but the REAL data contains 541 and 451 such rows respectively.
Three universes now, in increasing authority:

  U1  full enumeration, repeats allowed (a==b / b==c / a==c all permitted)  -- the widest
  U2  full enumeration, repeats excluded (v1's universe)                    -- for comparison
  U3  the ACTUAL 16,643 recorded trigram position-triples from
      tristrokes31_cond_v1.tsv                                             -- THE authority

For each: does any pair of trigrams agree on all 46 served columns yet DISAGREE on key a's
absolute row/finger? Also reports the same for b and c as method controls.
"""

from __future__ import annotations

import ast
import itertools
import json
from collections import defaultdict

from keybo.features import classify as C
from keybo.features.ngram import _trigram_row_from_positions
from keybo.features.schema import TRIGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_31

G = ROW_STAGGERED_31
SLOTS = [*G.slots, G.space_position]
NAMES = TRIGRAM_FEATURE_NAMES
TSV = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"


def row_onehot(p):
    y = p[1]
    return (float(y == 1), float(y == 2), float(y == 3))


def finger_onehot(p):
    ax = abs(p[0])
    return (float(ax in (5, 6)), float(ax == 4), float(ax == 3), float(ax in (1, 2)))


def block(p):
    return (*row_onehot(p), *finger_onehot(p), float(C.is_lateral(p[0])))


def probe(trigrams, label, with_wpm=None):
    """with_wpm: optional dict trig->set of wpm buckets; if given, wpm joins the served row."""
    buckets = defaultdict(list)
    for t in trigrams:
        a, b, c = t
        row = _trigram_row_from_positions(G, a, b, c, 0.0)
        key = tuple(round(row[n], 12) for n in NAMES)
        buckets[key].append(t)
    out = {"universe": label, "n_trigrams": len(trigrams), "n_distinct_served_rows": len(buckets)}
    for name, idx in (("a_first", 0), ("b_second", 1), ("c_third", 2)):
        amb_row = amb_fing = amb_blk = 0
        n_in_amb = 0
        ex = []
        for trigs in buckets.values():
            if len({row_onehot(t[idx]) for t in trigs}) > 1:
                amb_row += 1
            if len({finger_onehot(t[idx]) for t in trigs}) > 1:
                amb_fing += 1
            blks = {block(t[idx]) for t in trigs}
            if len(blks) > 1:
                amb_blk += 1
                n_in_amb += len(trigs)
                if len(ex) < 4:
                    ex.append({"trigrams": [list(map(list, t)) for t in trigs[:3]],
                               "n_colliding": len(trigs), "distinct_blocks": len(blks)})
        out[name] = {
            "buckets_ambiguous_row": amb_row,
            "buckets_ambiguous_finger": amb_fing,
            "buckets_ambiguous_block": amb_blk,
            "n_trigrams_in_ambiguous_buckets": n_in_amb,
            "frac_ambiguous": round(n_in_amb / max(len(trigrams), 1), 6),
            "examples": ex,
        }
    return out


def main():
    res = {}

    # U1: repeats allowed
    u1 = [t for t in itertools.product(SLOTS, repeat=3)]
    res["U1_enum_repeats_allowed"] = probe(u1, "enum, repeats allowed")

    # U2: v1's universe
    u2 = [t for t in itertools.product(SLOTS, repeat=3) if t[0] != t[1] and t[1] != t[2]]
    res["U2_enum_no_adjacent_repeat"] = probe(u2, "enum, a!=b and b!=c")

    # U3: THE AUTHORITY -- real recorded position triples
    real = []
    seen_layouts = set()
    with open(TSV, encoding="utf-8") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            seen_layouts.add(p[0])
            real.append(tuple(ast.literal_eval(p[1])))
    res["U3_real_data"] = probe(real, f"real data, layouts={sorted(seen_layouts)}")
    res["U3_real_data"]["n_distinct_position_triples"] = len(set(real))
    # and the deduplicated version (a layout/ngram pair repeats across layouts)
    res["U3_real_dedup"] = probe(sorted(set(real)), "real data, distinct triples")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
