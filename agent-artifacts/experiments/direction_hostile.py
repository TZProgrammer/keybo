"""DIRECTION-HOSTILE-1: the self-separation pass — attacks this investigation's own claims.

Read as a hostile stranger, four of my own numbers are attackable. Each attack is run as
code, and each can kill the claim it targets:

A1  **The two "BETTER" layouts are below the campaign's own resolution floor.**
    keybo-lsb improves by 0.001622 ms/char. The campaign's MEASURED floor on blend-v1 is
    0.6654644 ms/char. 0.001622 is 410x SMALLER. So "the swapped variant is better" is
    NOT a resolvable claim -- it is a tie. Quantify the ratio and say so.

A2  **Is the 14.293 ms per-occurrence figure resolvable?** It is a per-trigram table
    difference, not a per-char corpus mean, so the ms/char floor does not directly apply.
    But the three seeds behind the surface disagree with each other; measure that spread
    on THIS quantity. If seed spread swamps 14.293, the per-occurrence claim dies too.

A3  **Is the roll-cheapest ordering a general property or a coincidence of one triple?**
    My headline generalizes from the ring/middle/index same-row triple. Test it on every
    same-hand three-distinct-finger triple on the board: how often is the monotone order
    the cheapest of the six? A low rate makes the headline a special case.

A4  **Does the redirect FEATURE actually drive the price, or is it the skipgram?**
    C3 showed 10 features move, only one of which is ``redirect``; ``sg_dx``/``sg_distance``
    move too. If the cost tracks skip distance rather than the direction flag, then the
    model is pricing "how far apart the 1st and 3rd keys are", NOT clunkiness -- a
    materially different claim. Separate them by finding triples where the two disagree.
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
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.testkit import assert_module_under, assert_operands_computed  # noqa: E402

#: The campaign's MEASURED ms/char resolution floors, per corpus (parent's frozen numbers).
#: Quoted to be compared AGAINST, not relied on as an input to any computation here.
FLOOR_BLEND = 0.6654644
FLOOR_IWEB = 0.7185664

#: The A/B deltas DIRECTION-SWAP-1 measured through the shipped CLI, for the floor check.
SWAP_DELTAS = {
    "keybo-lsb": -0.001622,
    "keybo-lsb+lm": -0.002453,
    "p13stab-win": +0.194932,
    "p16-balance": +0.350223,
    "BALL-1": +0.106704,
    "colemak": +0.468265,
    "dvorak": +0.939092,
    "keybo-c30m": +0.838458,
    "qwerty": +1.811403,
    "qwerty30m": +1.711750,
}


def a1_floor_check() -> dict:
    print("A1  is any whole-layout swap delta RESOLVABLE against the measured floor?")
    print(f"    blend-v1 measured floor = {FLOOR_BLEND} ms/char")
    rows = {}
    for name, delta in sorted(SWAP_DELTAS.items(), key=lambda kv: kv[1]):
        ratio = abs(delta) / FLOOR_BLEND
        verdict = "RESOLVABLE" if abs(delta) > FLOOR_BLEND else "below floor (a TIE)"
        rows[name] = {"delta": delta, "floor_ratio": ratio, "verdict": verdict}
        print(f"    {name:14s} {delta:+.6f}  = {ratio:7.4f}x floor  {verdict}")
    resolvable = [k for k, v in rows.items() if v["verdict"] == "RESOLVABLE"]
    print(f"    ⇒ {len(resolvable)} of {len(rows)} resolvable: {resolvable}")
    print("    ⇒ BOTH 'BETTER' results (keybo-lsb, +lm) are ~275-410x BELOW the floor:")
    print("      the honest statement is 'no resolvable whole-layout difference',")
    print("      NOT 'the swapped variant is faster'.")
    return {"floor": FLOOR_BLEND, "per_layout": rows, "resolvable": resolvable}


def a2_seed_spread(tri_freqs: dict) -> dict:
    """Is 14.293 ms/occurrence bigger than the disagreement among the three seeds?"""
    surface = TimeSurface(tri_freqs, target_wpm=90.0, keep_seed_tables=True)
    ring, middle, index_ = (-4, 3), (-3, 3), (-2, 3)
    slots = {p: i for i, p in enumerate(G.slots)}
    roll = tuple(slots[p] for p in (ring, middle, index_))
    redirect = tuple(slots[p] for p in (ring, index_, middle))

    def _cost(T2, Tc, s) -> float:
        """Explicit parameters, not a closure: a lambda capturing the loop variables here
        would bind them late (ruff B023), which is harmless only because it is called in
        the same iteration -- a footgun not worth leaving in a probe whose whole output is
        the difference between three seeds."""
        return float(T2[s[0], s[1]] + Tc[s[0], s[1], s[2]])

    per_seed = [
        _cost(T2, Tc, redirect) - _cost(T2, Tc, roll)
        for T2, Tc in zip(surface._T2s, surface._Tcs, strict=True)
    ]
    mean_tables = float(
        (surface._T2[redirect[0], redirect[1]] + surface._Tc[redirect[0], redirect[1], redirect[2]])
        - (surface._T2[roll[0], roll[1]] + surface._Tc[roll[0], roll[1], roll[2]])
    )
    assert_operands_computed([*per_seed, mean_tables], "A2 seed spread")
    spread = max(per_seed) - min(per_seed)
    sd = float(np.std(per_seed, ddof=1))
    print("\nA2  seed spread on the per-occurrence redirect penalty (ring/middle/index)")
    print(f"    per-seed deltas: {', '.join(f'{v:.4f}' for v in per_seed)}")
    print(f"    seed-mean-tables delta (the headline) = {mean_tables:.6f} ms")
    print(f"    spread(max-min) = {spread:.4f} ms   sd = {sd:.4f} ms")
    ok = min(per_seed) > 0
    print(
        f"    all three seeds agree on the SIGN? {'YES' if ok else 'NO'}"
        f"  (min seed delta {min(per_seed):+.4f})"
    )
    print(
        f"    headline / spread = {mean_tables / spread:.2f}x" if spread else "    spread is zero"
    )
    return {
        "per_seed": per_seed,
        "mean_tables_delta": mean_tables,
        "spread": spread,
        "sd": sd,
        "all_seeds_same_sign": ok,
    }


def a3_is_monotone_cheapest_generally(tri_freqs: dict) -> dict:
    """Over every same-hand 3-distinct-finger triple: is the monotone order the cheapest?"""
    surface = TimeSurface(tri_freqs, target_wpm=90.0)
    cost = lambda s: float(surface._T2[s[0], s[1]] + surface._Tc[s[0], s[1], s[2]])  # noqa: E731
    n = wins = same_row_n = same_row_wins = 0
    losses = []
    for combo in itertools.combinations(range(30), 3):
        pos = [G.slots[s] for s in combo]
        if len({G.hand(p[0]) for p in pos}) != 1:
            continue
        if len({G.finger(p[0]).name for p in pos}) != 3:
            continue
        orders = list(itertools.permutations(combo))
        priced = sorted(orders, key=cost)
        # the two monotone orders (strictly inward, strictly outward, by |column|)
        monotone = [
            o
            for o in orders
            if (abs(G.slots[o[1]][0]) - abs(G.slots[o[0]][0]))
            * (abs(G.slots[o[2]][0]) - abs(G.slots[o[1]][0]))
            > 0
        ]
        n += 1
        cheapest_is_monotone = priced[0] in monotone
        wins += cheapest_is_monotone
        rows_same = len({p[1] for p in pos}) == 1
        if rows_same:
            same_row_n += 1
            same_row_wins += cheapest_is_monotone
        if not cheapest_is_monotone:
            losses.append(
                {
                    "slots": list(combo),
                    "cheapest": list(priced[0]),
                    "cheapest_fingers": "->".join(G.finger(G.slots[s][0]).name for s in priced[0]),
                    "same_row": rows_same,
                }
            )
    print("\nA3  is the MONOTONE order the cheapest of the six, generally?")
    print(f"    same-hand 3-distinct-finger triples: {n}")
    print(f"    monotone order is cheapest: {wins}/{n} = {100 * wins / n:.1f}%")
    print(
        f"    restricted to SAME-ROW triples: {same_row_wins}/{same_row_n} = "
        f"{100 * same_row_wins / same_row_n:.1f}%"
    )
    if losses:
        print(
            f"    {len(losses)} counterexamples, e.g. "
            f"{losses[0]['cheapest_fingers']} (same_row={losses[0]['same_row']})"
        )
    return {
        "n": n,
        "monotone_cheapest": wins,
        "pct": 100.0 * wins / n,
        "same_row_n": same_row_n,
        "same_row_monotone_cheapest": same_row_wins,
        "same_row_pct": 100.0 * same_row_wins / same_row_n if same_row_n else None,
        "counterexamples": losses[:10],
    }


def a4_redirect_flag_vs_skip_distance(tri_freqs: dict) -> dict:
    """Does the price track the direction FLAG, or just how far apart keys 1 and 3 are?

    Find same-hand triples where the redirect order has the SAME first->third distance as
    the monotone order. If the penalty survives there, the flag carries signal the skip
    distance does not. If it vanishes, my claim should be restated as a skip-distance claim.
    """
    surface = TimeSurface(tri_freqs, target_wpm=90.0)
    cost = lambda s: float(surface._T2[s[0], s[1]] + surface._Tc[s[0], s[1], s[2]])  # noqa: E731
    matched, deltas = 0, []
    for combo in itertools.combinations(range(30), 3):
        pos = [G.slots[s] for s in combo]
        if len({G.hand(p[0]) for p in pos}) != 1 or len({G.finger(p[0]).name for p in pos}) != 3:
            continue
        for o in itertools.permutations(combo):
            mono = (abs(G.slots[o[1]][0]) - abs(G.slots[o[0]][0])) * (
                abs(G.slots[o[2]][0]) - abs(G.slots[o[1]][0])
            ) > 0
            if mono:
                continue
            # a monotone order over the same triple with the SAME endpoints (so identical
            # skipgram features), differing only in where the middle key falls
            for m in itertools.permutations(combo):
                if m[0] != o[0] or m[2] != o[2]:
                    continue
                mm = (abs(G.slots[m[1]][0]) - abs(G.slots[m[0]][0])) * (
                    abs(G.slots[m[2]][0]) - abs(G.slots[m[1]][0])
                ) > 0
                if not mm:
                    continue
                matched += 1
                deltas.append(cost(o) - cost(m))
    print("\nA4  redirect flag vs skipgram distance (endpoints held FIXED, so the")
    print("    skipgram features sg_dx/sg_dy/sg_distance are IDENTICAL by construction)")
    if not deltas:
        print("    no endpoint-matched pairs exist ⇒ the two cannot be separated this way")
        return {"n_pairs": 0, "separable": False}
    arr = np.array(deltas)
    print(f"    endpoint-matched (redirect, monotone) pairs: {matched}")
    print(
        f"    redirect minus monotone: mean {arr.mean():+.4f} ms, median "
        f"{np.median(arr):+.4f}, min {arr.min():+.4f}, max {arr.max():+.4f}"
    )
    print(f"    fraction where the redirect is MORE expensive: {100 * (arr > 0).mean():.1f}%")
    print("    ⇒ the penalty survives with skipgram features held identical, so it is not")
    print(
        "      reducible to first->third distance."
        if (arr > 0).mean() > 0.5
        else "    ⇒ the penalty does NOT survive; restate as a skip-distance claim."
    )
    return {
        "n_pairs": matched,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_redirect_costlier": float((arr > 0).mean()),
        "separable": True,
    }


def main() -> int:
    assert_module_under("keybo", REPO)
    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    out = {
        "A1_floor_check": a1_floor_check(),
        "A2_seed_spread": a2_seed_spread(tri),
        "A3_monotone_cheapest_generally": a3_is_monotone_cheapest_generally(tri),
        "A4_flag_vs_skip_distance": a4_redirect_flag_vs_skip_distance(tri),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_hostile.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
