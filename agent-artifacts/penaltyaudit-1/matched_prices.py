"""Matched pattern-class prices on the served bigram surface.

WHY MATCHED. A raw class contrast (mean ms of the class minus mean ms of ALTERNATE) is
confounded by composition: the classes differ systematically in which KEY they land on,
and the landing key is by far the largest single term in the surface (bottom-row pinky
148.3 ms vs home-row middle 125.2 ms on AALTO -- a 23 ms spread, larger than most class
contrasts). So a class that happens to contain more bottom-row landings looks expensive
even if its defining geometry is free.

The estimator here strata-matches on the landing key's FULL schema signature
(row, finger, lateral) -- everything the model records about the second key -- and averages
the within-stratum (member - non-member) difference, weighted by min(n_member, n_nonmember).
That is the PINKY-GAP matched-cell design applied to the serve grid instead of to data.

FRAME (label every number with this):
  * ROW_STAGGERED_30, 30 letter slots, SPACE EXCLUDED (thumb; hand()==0 would pollute
    ALTERNATE).
  * ordered distinct pairs only (870); same-key repeats are a separate class.
  * ms entries of the served table T2 = g(geometry, wpm=90). NO per-ngram practice term b.
  * three genuinely-independent-ish bigram tables: AALTO (= the shipped production table),
    COMMUNITY, POOL. POOL is a SUPERSET of the other two, not an independent sample
    (ledger COMM+POOL-INVEST-1 correction addendum (1)).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G

FULL = [*G.slots, G.space_position]
IX = {p: i for i, p in enumerate(FULL)}
SLOTS = list(G.slots)


def land_sig(p):
    """Everything the schema records about the LANDING key: row, finger, lateral flag."""
    ax = abs(p[0])
    fing = "pinky" if ax in (5, 6) else "ring" if ax == 4 else "middle" if ax == 3 else "index"
    return (p[1], fing, ax in (1, 6))


def origin_sig(p):
    ax = abs(p[0])
    fing = "pinky" if ax in (5, 6) else "ring" if ax == 4 else "middle" if ax == 3 else "index"
    return (p[1], fing)


def matched(T, member, nonmember, strata, pairs=None):
    """Within-stratum mean(member) - mean(nonmember), weighted by min cell count."""
    if pairs is None:
        pairs = [(a, b) for a in SLOTS for b in SLOTS if a != b]
    cells = defaultdict(lambda: ([], []))
    for a, b in pairs:
        k = strata((a, b))
        if k is None:
            continue
        t = T[IX[a], IX[b]]
        if member((a, b)):
            cells[k][0].append(t)
        elif nonmember((a, b)):
            cells[k][1].append(t)
    num = den = 0.0
    deltas, weights = [], []
    for mem, non in cells.values():
        if not mem or not non:
            continue
        d = float(np.mean(mem) - np.mean(non))
        w = float(min(len(mem), len(non)))
        num += w * d
        den += w
        deltas.append(d)
        weights.append(w)
    if den == 0:
        return None
    deltas = np.array(deltas)
    return {
        "delta_ms": num / den,
        "n_strata": len(deltas),
        "frac_pos": float(np.average(deltas > 0, weights=weights)),
        "p10": float(np.percentile(deltas, 10)),
        "p90": float(np.percentile(deltas, 90)),
    }


# ------------------------------------------------------------------- predicates
def alt(ab):
    a, b = ab
    return C.classify_positions(G, a, b) is C.BigramClass.ALTERNATE


def shb(ab):
    a, b = ab
    return C.classify_positions(G, a, b) is C.BigramClass.SAME_HAND


def sfb(ab):
    a, b = ab
    return C.same_finger(G, a, b)


def rowspan(ab):
    a, b = ab
    return abs(a[1] - b[1])


def adjacent(ab):
    a, b = ab
    return C.is_adjacent(G, a, b)


def scissor(ab):
    a, b = ab
    return C.is_scissor(G, a, b)


def lsb(ab):
    a, b = ab
    return C.is_lsb(G, a, b)


def colgap(ab):
    a, b = ab
    return abs(abs(a[0]) - abs(b[0]))


TRUE = lambda ab: True  # noqa: E731
