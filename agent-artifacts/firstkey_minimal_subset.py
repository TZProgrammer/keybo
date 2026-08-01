"""TASK-1, part 3: HOW is key A's absolute position carried? Structural, or a fingerprint?

Parts 1-2 established (i) the served 46-column row NEVER collides on key A's absolute row/finger,
yet (ii) a depth-3 tree recovers it only 0.6946 / 0.5153, and (iii) dropping the eleven continuous
geometry columns collapses recovery to ~baseline. That combination points at a specific mechanism:
A is identified by ARITHMETIC on continuous spans, not by a cheap conjunction of discrete
predicates. This script pins that down, because the distinction is the whole justification for the
arm:

  * If A's position were a cheap CONJUNCTION of two or three discrete columns, a bg0_ one-hot would
    be near-redundant and the A/B would be close to a duplicate no-op (SGDIST-SHIP-1's situation).
  * If it takes a continuous ARITHMETIC combination, the one-hot is a genuine re-encoding that a
    depth-3 tree cannot cheaply reproduce, and the A/B is a real capacity test.

Three probes:
  P1. GREEDY MINIMAL SUBSET — grow a column set until A's row (then A's finger) is exactly
      determined (zero collisions). Report the subset and whether its members are discrete or
      continuous.
  P2. DISCRETE-ONLY CEILING — is A's absolute position determined AT ALL by the 35 discrete
      columns, at any depth? (Part 2 said 0.5272 accuracy; here: does it even separate?)
  P3. PER-CONTINUOUS-COLUMN NECESSITY — drop each continuous column alone and re-count collisions.
      A column whose removal introduces collisions is NECESSARY for exact identification.
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

CONTINUOUS = {
    "sg_dx", "sg_dy", "sg_distance",
    "bg1_dx", "bg1_dy", "bg1_distance", "bg1_angle",
    "bg2_dx", "bg2_dy", "bg2_distance", "bg2_angle",
}


def a_row(p):
    return p[1]


def a_finger(p):
    ax = abs(p[0])
    return "pinky" if ax in (5, 6) else "ring" if ax == 4 else "middle" if ax == 3 else (
        "index" if ax in (1, 2) else "space"
    )


def a_block(p):
    return (a_row(p), a_finger(p), C.is_lateral(p[0]))


def load_real():
    out = []
    with open(TSV, encoding="utf-8") as f:
        for ln in f:
            out.append(tuple(ast.literal_eval(ln.split("\t")[1])))
    return sorted(set(out))


def collisions(rowvals, targets, cols_idx):
    """How many trigrams sit in a bucket that is ambiguous in `targets`, using only cols_idx."""
    buckets = defaultdict(set)
    counts = defaultdict(int)
    for vals, t in zip(rowvals, targets, strict=True):
        key = tuple(vals[i] for i in cols_idx)
        buckets[key].add(t)
        counts[key] += 1
    return sum(n for k, n in counts.items() if len(buckets[k]) > 1)


def _score(rowvals, targets, cols_idx):
    """(collisions, -n_buckets) — the greedy objective.

    Collisions alone PLATEAUS: with a 3-tuple target, no SINGLE first column reduces the collision
    count at all (every bucket it makes is still internally ambiguous), so a collisions-only greedy
    reports "no column helps" and falsely concludes the target is undetermined — contradicting the
    exhaustive result from part 1 that all 46 columns DO determine it. The bucket count is the
    tie-break that escapes the plateau: prefer the column that partitions most finely when the
    collision count cannot yet move.
    """
    buckets = defaultdict(set)
    counts = defaultdict(int)
    for vals, t in zip(rowvals, targets, strict=True):
        key = tuple(vals[i] for i in cols_idx)
        buckets[key].add(t)
        counts[key] += 1
    coll = sum(n for k, n in counts.items() if len(buckets[k]) > 1)
    return coll, -len(buckets)


def greedy_minimal(rowvals, targets, label, universe_n, cap=25):
    """Grow a column set greedily until collisions hit 0 (or nothing improves either objective)."""
    chosen: list[int] = []
    remaining = list(range(len(NAMES)))
    trace = []
    cur = _score(rowvals, targets, chosen)
    while cur[0] > 0 and len(chosen) < cap:
        best, best_s = None, cur
        for i in remaining:
            s = _score(rowvals, targets, [*chosen, i])
            if s < best_s:
                best, best_s = i, s
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
        cur = best_s
        trace.append(
            {
                "added": NAMES[best],
                "kind": "continuous" if NAMES[best] in CONTINUOUS else "discrete",
                "collisions_remaining": cur[0],
                "frac_remaining": round(cur[0] / universe_n, 6),
                "n_buckets": -cur[1],
            }
        )
    cur_coll = cur[0]
    return {
        "target": label,
        "exactly_determined": cur_coll == 0,
        "collisions_remaining_at_stop": cur_coll,
        "hit_cap": len(chosen) >= cap and cur_coll > 0,
        "n_columns_needed": len(chosen),
        "subset": [NAMES[i] for i in chosen],
        "n_continuous_in_subset": sum(1 for i in chosen if NAMES[i] in CONTINUOUS),
        "n_discrete_in_subset": sum(1 for i in chosen if NAMES[i] not in CONTINUOUS),
        "trace": trace,
    }


def main():
    real = load_real()
    enum = [t for t in itertools.product(SLOTS, repeat=3)]
    out = {}

    for uname, universe in (("real_distinct_triples", real), ("full_enumeration", enum)):
        rowvals = []
        for a, b, c in universe:
            row = _trigram_row_from_positions(G, a, b, c, 0.0)
            rowvals.append([round(row[n], 12) for n in NAMES])
        n = len(universe)
        res = {"n": n}

        # P1 greedy minimal subsets
        for label, fn in (("a_row", a_row), ("a_finger", a_finger), ("a_block", a_block)):
            targets = [fn(t[0]) for t in universe]
            res[f"P1_minimal_{label}"] = greedy_minimal(rowvals, targets, label, n)

        # P2 discrete-only ceiling: exact determination using ONLY discrete columns?
        disc = [i for i, nm in enumerate(NAMES) if nm not in CONTINUOUS]
        res["P2_discrete_only"] = {
            "n_discrete_cols": len(disc),
            "collisions_a_row": collisions(rowvals, [a_row(t[0]) for t in universe], disc),
            "collisions_a_finger": collisions(rowvals, [a_finger(t[0]) for t in universe], disc),
            "collisions_a_block": collisions(rowvals, [a_block(t[0]) for t in universe], disc),
            "frac_a_block": round(
                collisions(rowvals, [a_block(t[0]) for t in universe], disc) / n, 6
            ),
        }

        # P3 per-continuous-column necessity (drop one, keep all 45 others)
        allc = list(range(len(NAMES)))
        tb = [a_block(t[0]) for t in universe]
        nec = {}
        for nm in sorted(CONTINUOUS):
            keep = [i for i in allc if NAMES[i] != nm]
            c = collisions(rowvals, tb, keep)
            nec[nm] = {"collisions_without_it": c, "necessary": c > 0}
        res["P3_necessity"] = nec
        res["P3_necessary_columns"] = [k for k, v in nec.items() if v["necessary"]]
        out[uname] = res
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
