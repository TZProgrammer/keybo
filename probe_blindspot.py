"""lmscissor question (a): quantify the two-row NON-adjacent-finger blind spot.

For both layouts, partition ALL same-hand distinct-finger bigram mass by (dy, finger-pair,
adjacency, which-finger-is-lower) and label each cell with what each gauge prices.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import _DEX, bad_scissor  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}

corpus_dir = production_corpus_dir(None)
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"corpus = {corpus_dir.name}  (bigrams.txt)")


def kind(x: int) -> str:
    return G.finger(x).value.split("-")[1]


results = {}
for label, spec in LAYOUTS.items():
    lay = Layout(spec, G)
    denom = 0
    # cell -> mass
    cells: dict[tuple, int] = {}
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg:
            continue
        if not all(lay.has_key(c) for c in bg):
            continue
        denom += freq
        a, b = lay.pos(bg[0]), lay.pos(bg[1])
        if not C.same_hand(G, a, b) or C.same_finger(G, a, b):
            continue
        dy = abs(a[1] - b[1])
        if dy == 0:
            continue
        ka, kb = kind(a[0]), kind(b[0])
        pair = "-".join(sorted((ka, kb), key=lambda k: -_DEX[k]))
        adjacent = C.is_adjacent(G, a, b)
        # which finger holds the LOWER key
        lower_kind = ka if a[1] < b[1] else kb
        upper_kind = kb if a[1] < b[1] else ka
        weak_is_lower = _DEX[lower_kind] < _DEX[upper_kind]
        key = (dy, pair, adjacent, weak_is_lower)
        cells[key] = cells.get(key, 0) + freq

    rows = []
    for (dy, pair, adjacent, weak_lower), mass in sorted(
        cells.items(), key=lambda kv: -kv[1]
    ):
        pp = 100.0 * mass / denom
        # which gauges price this cell?
        narrow = dy == 2 and adjacent
        wide = dy == 2
        bad = weak_lower
        rows.append(
            {
                "dy": dy,
                "pair": pair,
                "adjacent": adjacent,
                "weak_is_lower": weak_lower,
                "mass": mass,
                "pp": pp,
                "narrow": narrow,
                "wide_dy2": wide,
                "bad_scissor": bad,
            }
        )
    results[label] = {"denom": denom, "rows": rows}

    print(f"\n{'='*106}\n{label}   denominator (space-excluded, layout-restricted) = {denom}\n{'='*106}")
    print(
        f"{'dy':>2} {'finger-pair':<15} {'adj':<5} {'weak_lo':<7} {'share pp':>9}  "
        f"{'narrow':<7}{'wide-dy2':<9}{'bad-sc':<7}"
    )
    for r in rows:
        print(
            f"{r['dy']:>2} {r['pair']:<15} {str(r['adjacent']):<5} {str(r['weak_is_lower']):<7} "
            f"{r['pp']:>9.4f}  {'YES' if r['narrow'] else '-':<7}"
            f"{'YES' if r['wide_dy2'] else '-':<9}{'YES' if r['bad_scissor'] else '-':<7}"
        )

    # ---- the headline aggregates -------------------------------------------------------
    def agg(pred) -> float:
        return sum(r["pp"] for r in rows if pred(r))

    print(f"\n  -- aggregates ({label}, blend-v1, pp of space-excluded bigram mass) --")
    print(f"  ALL same-hand distinct-finger row-travel:        {agg(lambda r: True):8.4f}")
    print(f"  dy=2 total (any finger pair):                    {agg(lambda r: r['dy']==2):8.4f}")
    print(f"    dy=2 ADJACENT   (= narrow support):             {agg(lambda r: r['dy']==2 and r['adjacent']):8.4f}")
    print(f"    dy=2 NONADJACENT:                              {agg(lambda r: r['dy']==2 and not r['adjacent']):8.4f}")
    print(f"  dy=1 total:                                      {agg(lambda r: r['dy']==1):8.4f}")
    print()
    print(f"  bad-scissor priced (weak lower, any dy):          {agg(lambda r: r['bad_scissor']):8.4f}")
    print(f"    ... of which dy=1:                             {agg(lambda r: r['bad_scissor'] and r['dy']==1):8.4f}")
    print(f"    ... of which dy=2:                             {agg(lambda r: r['bad_scissor'] and r['dy']==2):8.4f}")
    print()
    print("  >>> THE BLIND SPOT: dy=2 mass that bad-scissor does NOT price (weak on TOP):")
    print(f"      dy=2 & NOT weak_lower  (total):              {agg(lambda r: r['dy']==2 and not r['bad_scissor']):8.4f}")
    print(f"        ... nonadjacent (narrow blind too):        {agg(lambda r: r['dy']==2 and not r['bad_scissor'] and not r['adjacent']):8.4f}")
    print(f"        ... involving the PINKY:                   {agg(lambda r: r['dy']==2 and not r['bad_scissor'] and 'pinky' in r['pair']):8.4f}")
    print()
    print("  >>> dy=2 mass NO gauge prices (nonadjacent AND weak-on-top):")
    print(f"      {agg(lambda r: r['dy']==2 and not r['adjacent'] and not r['bad_scissor']):8.4f}")
    print("  >>> dy=2 NONADJACENT total (narrow blind, regardless of orientation):")
    print(f"      {agg(lambda r: r['dy']==2 and not r['adjacent']):8.4f}")
    print(f"        ... involving PINKY:                       {agg(lambda r: r['dy']==2 and not r['adjacent'] and 'pinky' in r['pair']):8.4f}")
    print(f"        ... middle-pinky specifically:             {agg(lambda r: r['dy']==2 and not r['adjacent'] and r['pair']=='middle-pinky'):8.4f}")

# ---- cross-layout comparison of the key aggregates ---------------------------------
print(f"\n{'='*106}\nCROSS-LAYOUT DELTAS (blend-v1)\n{'='*106}")


def get(label, pred):
    return sum(r["pp"] for r in results[label]["rows"] if pred(r))


checks = [
    ("bad-scissor priced total", lambda r: r["bad_scissor"]),
    ("  dy=1 priced", lambda r: r["bad_scissor"] and r["dy"] == 1),
    ("  dy=2 priced", lambda r: r["bad_scissor"] and r["dy"] == 2),
    ("dy=2 total (any orientation/adjacency)", lambda r: r["dy"] == 2),
    ("  dy=2 adjacent (narrow support)", lambda r: r["dy"] == 2 and r["adjacent"]),
    ("  dy=2 NONadjacent", lambda r: r["dy"] == 2 and not r["adjacent"]),
    ("  dy=2 middle-pinky (all nonadj)", lambda r: r["dy"] == 2 and r["pair"] == "middle-pinky"),
    ("  dy=2 any-pinky", lambda r: r["dy"] == 2 and "pinky" in r["pair"]),
    ("dy=1 total", lambda r: r["dy"] == 1),
    ("dy=1 any-pinky", lambda r: r["dy"] == 1 and "pinky" in r["pair"]),
    ("ALL row-travel same-hand distinct-finger", lambda r: True),
]
print(f"{'quantity':<45}{'keybo-lsb':>12}{'keybo-lsb+lm':>14}{'delta':>10}")
for name, pred in checks:
    a, b = get("keybo-lsb", pred), get("keybo-lsb+lm", pred)
    print(f"{name:<45}{a:>12.4f}{b:>14.4f}{b-a:>+10.4f}")

json.dump(results, open("/tmp/lmscissor_blindspot.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_blindspot.json")
