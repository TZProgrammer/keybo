"""THE SHARPER TEST: is a candidate feature NEW INFORMATION, or a function of the existing 20?

Swap-dependence is necessary but NOT sufficient.  A candidate can be order-dependent and
still be perfectly determined by the shipped 20-column vector — in which case adding it
gives the model no channel it did not already have, and any "gain" would be pure noise.

Method: group the 870 ordered distinct pairs by their EXACT shipped 20-feature vector
(wpm excluded — it is constant across a serve grid).  Within each collision group, check
whether the candidate is constant.  If it is constant in every group, the candidate is a
deterministic function of the existing features => WORTHLESS.
"""

from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")
sys.path.insert(0, "/local/home/zegertho/agent/state/direction/scratch")

from keybo.features.ngram import _placement_row_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402
from swap_test import DICTS, SCALARS  # noqa: E402


def run(g, label):
    slots = list(g.slots)
    pairs = [(a, b) for a in slots for b in slots if a != b]
    groups = defaultdict(list)
    for a, b in pairs:
        r = _placement_row_from_positions(g, a, b)
        key = tuple(round(v, 9) for v in r.values())
        groups[key].append((a, b))
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"\n=== {label}: {len(pairs)} ordered distinct pairs -> "
          f"{len(groups)} distinct shipped-20 vectors ===")
    print(f"  collision-group sizes: max {sizes[0]}, "
          f"n_groups>1 = {sum(1 for s in sizes if s > 1)}, "
          f"n_pairs_in_collisions = {sum(s for s in sizes if s > 1)}")

    # how many pairs have their SWAP in the same group (i.e. featurewise identical)?
    key_of = {}
    for k, v in groups.items():
        for p in v:
            key_of[p] = k
    same = sum(1 for a, b in pairs if key_of[(a, b)] == key_of[(b, a)])
    print(f"  pairs whose REVERSE has the identical shipped-20 vector: {same} / {len(pairs)}")

    print("  candidate                              varies_in_groups  n_pairs_in_varying  verdict")
    for name, fn in SCALARS.items():
        varying = 0
        npairs = 0
        for k, v in groups.items():
            vals = {round(fn(g, a, b), 9) for a, b in v}
            if len(vals) > 1:
                varying += 1
                npairs += len(v)
        verdict = "NEW INFORMATION" if varying else "!! DETERMINED by existing 20 (worthless)"
        print(f"  {name:38s} {varying:5d}              {npairs:5d}          {verdict}")
    for name, fn in DICTS.items():
        keys = list(fn(g, pairs[0][0], pairs[0][1]).keys())
        for col in keys:
            varying = 0
            npairs = 0
            for k, v in groups.items():
                vals = {round(fn(g, a, b)[col], 9) for a, b in v}
                if len(vals) > 1:
                    varying += 1
                    npairs += len(v)
            verdict = "NEW INFORMATION" if varying else "!! DETERMINED (worthless)"
            print(f"  {name[:12]+'.'+col:38s} {varying:5d}              {npairs:5d}          {verdict}")


if __name__ == "__main__":
    run(ROW_STAGGERED_30, "ROW_STAGGERED_30")
    run(ROW_STAGGERED_31, "ROW_STAGGERED_31")
