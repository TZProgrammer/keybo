"""DIRECTION-HOSTILE-2: repairs the two attacks that came back inconclusive or fatal.

A4 as first written was IMPOSSIBLE, not merely negative: it looked for a monotone and a
non-monotone order of the SAME three keys sharing both endpoints, but with three distinct
keys the endpoints determine the middle key, so no such pair can exist. It reported "0
pairs" and I nearly filed that as "not separable". Redone here two ways that can actually
run (B1, B2).

A3 refuted my generalization (monotone is cheapest in only 110/378 = 29.1% of same-hand
three-finger triples, BELOW the 2-in-6 = 33.3% chance rate). B3 asks the sharper question
that the refutation leaves open: is there a paired penalty at all, and what does the model
price INSTEAD of abstract direction?
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features.ngram import trigram_features_from_positions  # noqa: E402
from keybo.features.schema import TRIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

_SKIP = ("sg_same_finger", "sg_dx", "sg_dy", "sg_distance")


def _is_monotone(order) -> bool:
    d1 = abs(G.slots[order[1]][0]) - abs(G.slots[order[0]][0])
    d2 = abs(G.slots[order[2]][0]) - abs(G.slots[order[1]][0])
    return d1 * d2 > 0


def _same_hand_three_finger_triples():
    for combo in itertools.combinations(range(30), 3):
        pos = [G.slots[s] for s in combo]
        if len({G.hand(p[0]) for p in pos}) != 1:
            continue
        if len({G.finger(p[0]).name for p in pos}) != 3:
            continue
        yield combo, pos


def b1_skip_matched_across_triples(cost) -> dict:
    """Compare redirect vs monotone orders whose SKIPGRAM features are identical.

    Since a single triple cannot supply both (see the module docstring), match ACROSS
    orders on the exact skipgram feature tuple. Same first->third relationship, different
    direction class: if the redirect is still costlier, the penalty is not the skipgram.
    """
    buckets: dict[tuple, dict[str, list[float]]] = {}
    idx = [TRIGRAM_FEATURE_NAMES.index(n) for n in _SKIP]
    for combo, _pos in _same_hand_three_finger_triples():
        for order in itertools.permutations(combo):
            vec = trigram_features_from_positions(G, tuple(G.slots[s] for s in order), wpm=90.0)
            key = tuple(round(float(vec[i]), 9) for i in idx)
            slot = buckets.setdefault(key, {"mono": [], "redir": []})
            slot["mono" if _is_monotone(order) else "redir"].append(cost(order))
    usable = {k: v for k, v in buckets.items() if v["mono"] and v["redir"]}
    deltas = [float(np.mean(v["redir"]) - np.mean(v["mono"])) for v in usable.values()]
    arr = np.array(deltas)
    print(f"B1  skipgram-matched redirect-vs-monotone (matched on the exact {list(_SKIP)} tuple)")
    print(f"    {len(usable)} usable skipgram buckets of {len(buckets)}")
    print(
        f"    mean(redirect) - mean(monotone): mean {arr.mean():+.4f} ms, "
        f"median {np.median(arr):+.4f}, min {arr.min():+.4f}, max {arr.max():+.4f}"
    )
    print(
        f"    buckets where the redirect is costlier: {(arr > 0).sum()}/{len(arr)} "
        f"= {100 * (arr > 0).mean():.1f}%"
    )
    return {
        "n_buckets": len(usable),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "frac_costlier": float((arr > 0).mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def b2_paired_within_triple(cost) -> dict:
    """Within each triple: mean of its 2 monotone orders vs mean of its 4 redirect orders.

    This is the paired form of A3 and asks a different question than "is monotone the
    argmin": it asks whether monotone is cheaper ON AVERAGE for that triple, which is the
    quantity a corpus-weighted objective actually integrates.
    """
    per_triple, same_row = [], []
    for combo, pos in _same_hand_three_finger_triples():
        orders = list(itertools.permutations(combo))
        mono = [cost(o) for o in orders if _is_monotone(o)]
        redir = [cost(o) for o in orders if not _is_monotone(o)]
        d = float(np.mean(redir) - np.mean(mono))
        per_triple.append(d)
        if len({p[1] for p in pos}) == 1:
            same_row.append(d)
    arr, sr = np.array(per_triple), np.array(same_row)
    print("\nB2  paired within-triple: mean(4 redirect orders) - mean(2 monotone orders)")
    print(
        f"    all {len(arr)} triples : mean {arr.mean():+.4f} ms, median "
        f"{np.median(arr):+.4f}, positive in {100 * (arr > 0).mean():.1f}%"
    )
    print(
        f"    same-row ({len(sr)})   : mean {sr.mean():+.4f} ms, median "
        f"{np.median(sr):+.4f}, positive in {100 * (sr > 0).mean():.1f}%"
    )
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "frac_positive": float((arr > 0).mean()),
        "same_row_n": len(sr),
        "same_row_mean": float(sr.mean()),
        "same_row_frac_positive": float((sr > 0).mean()),
    }


def b3_what_is_priced_instead(cost) -> dict:
    """A3 said monotone-cheapest is BELOW chance. So what IS the cheapest order?

    Characterize the argmin order of every triple by the finger that lands LAST and by
    whether a pinky is involved, to name the thing the model prefers instead of direction.
    """
    last_finger: dict[str, int] = {}
    first_finger: dict[str, int] = {}
    n = 0
    for combo, _pos in _same_hand_three_finger_triples():
        best = min(itertools.permutations(combo), key=cost)
        lf = G.finger(G.slots[best[2]][0]).name[1]
        ff = G.finger(G.slots[best[0]][0]).name[1]
        last_finger[lf] = last_finger.get(lf, 0) + 1
        first_finger[ff] = first_finger.get(ff, 0) + 1
        n += 1
    print("\nB3  what the model actually prefers (argmin order of each triple)")
    print(
        "    finger that lands LAST : "
        + ", ".join(f"{k}={100 * v / n:.1f}%" for k, v in sorted(last_finger.items()))
    )
    print(
        "    finger that goes FIRST : "
        + ", ".join(f"{k}={100 * v / n:.1f}%" for k, v in sorted(first_finger.items()))
    )
    return {
        "n": n,
        "last_finger_pct": {k: 100.0 * v / n for k, v in last_finger.items()},
        "first_finger_pct": {k: 100.0 * v / n for k, v in first_finger.items()},
    }


def b4_the_users_cell_specifically(cost) -> dict:
    """The user's exact cell, stated as the only comparison their claim makes.

    Not "is monotone the argmin of six" (refuted in general) but: on the three slots a
    layout devotes to y/u/o, does the model charge more for the order that spells the word
    ``you`` when that order reverses direction than when it does not? Reported per triple
    geometry actually used by a registry layout.
    """
    slots = {p: i for i, p in enumerate(G.slots)}
    cells = {
        "ring/middle/index same-row (keybo-lsb, flagship-c3, BALL-1, armB)": (
            (-4, 3),
            (-3, 3),
            (-2, 3),
        ),
        "pinky/ring/middle same-row (p13stab-win, semimak)": ((5, 3), (4, 3), (3, 3)),
        "middle/index/ring same-row (p16-balance)": ((3, 3), (2, 3), (4, 3)),
    }
    out = {}
    for label, triple in cells.items():
        ids = tuple(slots[p] for p in triple)
        priced = {o: cost(o) for o in itertools.permutations(ids)}
        mono = {o: c for o, c in priced.items() if _is_monotone(o)}
        redir = {o: c for o, c in priced.items() if not _is_monotone(o)}
        cheapest_mono = min(mono.values())
        cheapest_redir = min(redir.values())
        out[label] = {
            "cheapest_monotone_ms": cheapest_mono,
            "cheapest_redirect_ms": cheapest_redir,
            "monotone_advantage_ms": cheapest_redir - cheapest_mono,
            "argmin_is_monotone": min(priced, key=priced.get) in mono,
        }
        print(f"\nB4  {label}")
        print(
            f"    cheapest monotone {cheapest_mono:.3f} ms  vs cheapest redirect "
            f"{cheapest_redir:.3f} ms  -> monotone advantage "
            f"{cheapest_redir - cheapest_mono:+.3f} ms"
        )
        print(
            f"    is the overall cheapest order monotone? "
            f"{'YES' if out[label]['argmin_is_monotone'] else 'NO'}"
        )
    return out


def main() -> int:
    assert_module_under("keybo", REPO)
    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    surface = TimeSurface(tri, target_wpm=90.0)

    def cost(order):
        a, b, c = order
        return float(surface._T2[a, b] + surface._Tc[a, b, c])

    out = {
        "B1_skip_matched": b1_skip_matched_across_triples(cost),
        "B2_paired_within_triple": b2_paired_within_triple(cost),
        "B3_what_is_priced_instead": b3_what_is_priced_instead(cost),
        "B4_users_cell": b4_the_users_cell_specifically(cost),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_hostile2.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
