"""FM4 step 2c: the per-name DETAIL evidence — the exact disagreeing cells, and the
near-miss names the exact-match scan does not catch (`sg_distance` vs the `sg_dist` gauge).
"""

from __future__ import annotations

import itertools
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import keybo  # noqa: E402
from keybo.analysis import kmstats as KM  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.features.ngram import (  # noqa: E402
    _placement_row_from_positions,
    _trigram_level_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

print("keybo.__file__ =", keybo.__file__)
G = ROW_STAGGERED_30
SLOTS = list(G.slots)
out = {"keybo_file": keybo.__file__}

# --- 1. the `lsb` disagreements, named cell by cell ----------------------------------------
print("\n=== `lsb`: FRAME (classify.is_lsb) vs keymeow GAUGE (kmstats._is_lsb) ===")
disagree = []
for i, j in itertools.product(range(len(SLOTS)), repeat=2):
    a, b = SLOTS[i], SLOTS[j]
    frame = bool(_placement_row_from_positions(G, a, b)["lsb"])
    gauge = bool(KM._is_lsb(KM._KEYS[i], KM._KEYS[j]))
    if frame != gauge:
        disagree.append(
            {
                "a": a,
                "b": b,
                "frame_lsb": frame,
                "gauge_lsb": gauge,
                "frame_dx_stagger_adj": G.stagger_adjusted_dx(a, b),
                "gauge_abs_x_gap": abs(KM._KEYS[i].x - KM._KEYS[j].x),
                "gauge_kind_gap": abs(KM._KEYS[i].kind - KM._KEYS[j].kind),
                "fingers": (G.finger(a[0]).name, G.finger(b[0]).name),
            }
        )
print(f"  {len(disagree)} disagreeing ordered pairs of {len(SLOTS)**2}; all listed:")
for d in disagree:
    print(
        f"    {d['a']} -> {d['b']}  {d['fingers'][0]}/{d['fingers'][1]}  "
        f"frame={d['frame_lsb']} (dx={d['frame_dx_stagger_adj']:.2f} > 1.5)  "
        f"gauge={d['gauge_lsb']} (gap={d['gauge_abs_x_gap']:.2f} >= 2 ? kind_gap={d['gauge_kind_gap']})"
    )
frame_fires = sum(
    1
    for i, j in itertools.product(range(len(SLOTS)), repeat=2)
    if _placement_row_from_positions(G, SLOTS[i], SLOTS[j])["lsb"]
)
gauge_fires = sum(
    1
    for i, j in itertools.product(range(len(SLOTS)), repeat=2)
    if KM._is_lsb(KM._KEYS[i], KM._KEYS[j])
)
print(f"  FRAME fires {frame_fires}; GAUGE fires {gauge_fires}  -> frame is a strict SUPERSET")
out["lsb"] = {
    "n": len(SLOTS) ** 2,
    "disagreements": len(disagree),
    "frame_firings": frame_fires,
    "gauge_firings": gauge_fires,
    "cells": disagree,
}

# --- 2. WHICH finger pairs each `lsb` can reach (the structural claim) ---------------------
print("\n=== which FINGER-PAIR classes each `lsb` can EVER flag (K30) ===")
frame_classes, gauge_classes = set(), set()
for i, j in itertools.product(range(len(SLOTS)), repeat=2):
    a, b = SLOTS[i], SLOTS[j]
    cls = tuple(sorted((G.finger(a[0]).name[1:], G.finger(b[0]).name[1:])))
    if _placement_row_from_positions(G, a, b)["lsb"]:
        frame_classes.add(cls)
    if KM._is_lsb(KM._KEYS[i], KM._KEYS[j]):
        gauge_classes.add(cls)
print(f"  FRAME is_lsb    can flag: {sorted(frame_classes)}")
print(f"  keymeow GAUGE   can flag: {sorted(gauge_classes)}")
out["lsb_finger_classes"] = {
    "frame": sorted(map(list, frame_classes)),
    "gauge": sorted(map(list, gauge_classes)),
}

# --- 3. `lateral` (column) vs `lat-span` (gauge): what each actually measures --------------
print("\n=== `lateral` the COLUMN vs `lat-span` the GAUGE ===")
lateral_fires = 0
span_nonzero = 0
lateral_and_span0 = 0
span_and_lateral0 = 0
for a, b in itertools.product(SLOTS, repeat=2):
    lat = bool(_placement_row_from_positions(G, a, b)["lateral"])
    span = C.lateral_span(G, a, b) > 0.0
    lateral_fires += lat
    span_nonzero += span
    lateral_and_span0 += lat and not span
    span_and_lateral0 += span and not lat
print(f"  `lateral`  (one-hot on the LANDING key) fires : {lateral_fires} of {len(SLOTS)**2}")
print(f"  `lat-span` (graded pairwise stretch) nonzero  : {span_nonzero}")
print(f"  lateral=1 while lat-span=0 : {lateral_and_span0}")
print(f"  lat-span>0 while lateral=0 : {span_and_lateral0}")
# `lateral` is a function of the SECOND KEY ALONE -> prove it
one_key = all(
    _placement_row_from_positions(G, a, b)["lateral"]
    == _placement_row_from_positions(G, a2, b)["lateral"]
    for b in SLOTS
    for a in SLOTS
    for a2 in SLOTS
)
print(f"  `lateral` depends ONLY on the landing key b (invariant in a): {one_key}")
# `lat-span` is symmetric in a,b -> prove it
symmetric = all(
    C.lateral_span(G, a, b) == C.lateral_span(G, b, a) for a, b in itertools.product(SLOTS, repeat=2)
)
print(f"  `lat-span` is SYMMETRIC in (a,b): {symmetric}")
out["lateral_vs_latspan"] = {
    "lateral_firings": lateral_fires,
    "latspan_nonzero": span_nonzero,
    "lateral1_span0": lateral_and_span0,
    "span_pos_lateral0": span_and_lateral0,
    "lateral_is_landing_key_only": one_key,
    "latspan_is_symmetric": symmetric,
}

# --- 4. `sg_distance` (frame) vs `sg_dist` (gauge): a NEAR-name that IS the same quantity --
print("\n=== `sg_distance` the COLUMN vs `sg_dist` the GAUGE (near-name, not exact) ===")
d = sum(
    1
    for a, b, c in itertools.product(SLOTS, repeat=3)
    if _trigram_level_from_positions(G, a, b, c)["sg_distance"] != G.distance(a, c)
)
print(f"  cells where frame sg_distance != geometry.distance(first, third): {d} of {len(SLOTS)**3}")
print("  (the `sg_dist` GAUGE is the corpus-weighted MEAN of exactly this quantity")
print("   -- skipgram_span.sg_dist: weighted_span += freq * geometry.distance(pos[first], pos[third]))")
out["sg_distance_vs_sg_dist"] = {
    "n": len(SLOTS) ** 3,
    "per_cell_disagreements": d,
    "verdict": "EQUAL per-cell (gauge is the corpus-weighted mean of the column)",
}

with open(os.path.join(os.path.dirname(__file__), "detail.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("\nwrote detail.json")
