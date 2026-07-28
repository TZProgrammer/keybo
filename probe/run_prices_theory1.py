"""Produce the ranked, matched pattern-class price table across the three bigram sources.

Every row is a CETERIS-PARIBUS contrast: member minus non-member, averaged within strata
that hold the landing key's full schema signature (and, where noted, the row span) fixed.
"""

from __future__ import annotations

import json

import numpy as np

import matched_prices as M
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G

S = "/local/home/zegertho/agent/state/theory/scratch/"
TABLES = {
    "AALTO": np.load(S + "T2_prod.npy"),
    "COMMUNITY": np.load(S + "T2_comm.npy"),
    "POOL": np.load(S + "T2_pool.npy"),
}

LAND = lambda ab: M.land_sig(ab[1])  # noqa: E731
LAND_ROW = lambda ab: (M.land_sig(ab[1]), M.rowspan(ab))  # noqa: E731
LAND_CLASS = lambda ab: (M.land_sig(ab[1]), C.classify_positions(G, *ab).value)  # noqa: E731


def same_hand_2f(ab):
    return M.shb(ab)


# ---------------------------------------------------------------- the test battery
# name -> (member, nonmember, strata, note)
TESTS = {
    # ============ the big structural axes ============
    "same_finger (SFB) vs same-hand 2-finger": (
        M.sfb, same_hand_2f, LAND_ROW,
        "the SFB penalty proper: one finger doing both keys, vs two fingers, same landing key + same row span"),
    "same-hand 2-finger vs alternate hands": (
        same_hand_2f, M.alt, LAND_ROW,
        "the cost of staying on one hand at all"),
    "same_finger (SFB) vs alternate hands": (
        M.sfb, M.alt, LAND_ROW,
        "SFB vs the fastest class (composition of the two above)"),
    "repeat (same key twice) vs alternate": (
        lambda ab: False, M.alt, LAND,
        "PLACEHOLDER - repeats need the diagonal, handled separately"),
    # ============ row geometry ============
    "row span 1 vs row span 0 (same hand)": (
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 1,
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 0, LAND,
        "one-row reach vs flat, same hand two fingers"),
    "row span 2 vs row span 0 (same hand)": (
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 2,
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 0, LAND,
        "full two-row reach vs flat"),
    "row span 2 vs row span 1 (same hand)": (
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 2,
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 1, LAND,
        "is the second row of travel as dear as the first?"),
    "row span 1 vs 0 (SFB only)": (
        lambda ab: M.sfb(ab) and M.rowspan(ab) == 1,
        lambda ab: M.sfb(ab) and M.rowspan(ab) == 0, LAND,
        "row travel WITHIN the same finger"),
    "row span 2 vs 1 (SFB only)": (
        lambda ab: M.sfb(ab) and M.rowspan(ab) == 2,
        lambda ab: M.sfb(ab) and M.rowspan(ab) == 1, LAND,
        "the 2u SFB vs the 1u SFB"),
    # ============ column geometry ============
    "column gap 2 vs 1 (same hand, same row)": (
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 0 and M.colgap(ab) == 2,
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 0 and M.colgap(ab) == 1, LAND,
        "skipping a finger, flat"),
    "column gap 3+ vs 1 (same hand, same row)": (
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 0 and M.colgap(ab) >= 3,
        lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 0 and M.colgap(ab) == 1, LAND,
        "widest flat same-hand reach"),
    # ============ the named community patterns ============
    "scissor (adj finger, 2 rows) vs adj finger flat": (
        M.scissor, lambda ab: M.adjacent(ab) and M.rowspan(ab) == 0, LAND,
        "the scissor as the community means it"),
    "scissor vs NON-adjacent same-hand 2-row": (
        M.scissor, lambda ab: same_hand_2f(ab) and M.rowspan(ab) == 2 and not M.adjacent(ab),
        LAND,
        "is ADJACENCY what makes a 2-row reach a scissor, or just the 2 rows?"),
    "half-scissor (adj, 1 row) vs adj finger flat": (
        lambda ab: M.adjacent(ab) and M.rowspan(ab) == 1,
        lambda ab: M.adjacent(ab) and M.rowspan(ab) == 0, LAND,
        "the 1u version"),
    "lsb (index/middle stretch) vs same-hand non-lsb": (
        M.lsb, lambda ab: same_hand_2f(ab) and not M.lsb(ab), LAND_ROW,
        "lateral stretch bigram"),
    "adjacent fingers vs non-adjacent (same hand, matched row span)": (
        M.adjacent, lambda ab: same_hand_2f(ab) and not M.adjacent(ab), LAND_ROW,
        "does finger adjacency help or hurt, holding row span?"),
}


def repeats_vs_alt(T):
    """Repeats need the diagonal, which the ordered-distinct grid excludes."""
    from collections import defaultdict

    cells = defaultdict(lambda: ([], []))
    for a in M.SLOTS:
        cells[M.land_sig(a)][0].append(T[M.IX[a], M.IX[a]])
    for a in M.SLOTS:
        for b in M.SLOTS:
            if a != b and M.alt((a, b)):
                cells[M.land_sig(b)][1].append(T[M.IX[a], M.IX[b]])
    num = den = 0.0
    ds = []
    for mem, non in cells.values():
        if not mem or not non:
            continue
        d = float(np.mean(mem) - np.mean(non))
        w = float(min(len(mem), len(non)))
        num += w * d
        den += w
        ds.append(d)
    return {"delta_ms": num / den, "n_strata": len(ds),
            "frac_pos": float(np.mean(np.array(ds) > 0)),
            "p10": float(np.percentile(ds, 10)), "p90": float(np.percentile(ds, 90))}


def main():
    results = {}
    for name, (mem, non, st, note) in TESTS.items():
        if name.startswith("repeat"):
            results[name] = {s: repeats_vs_alt(T) for s, T in TABLES.items()}
        else:
            results[name] = {s: M.matched(T, mem, non, st) for s, T in TABLES.items()}
        results[name]["_note"] = note

    hdr = f"{'matched contrast (ms, WPM 90, served frame)':58s}"
    hdr += "".join(f"{s[:5]:>10s}" for s in TABLES) + " | " + "".join(f"{s[:4]+'%+':>8s}" for s in TABLES)
    print(hdr)
    print("-" * len(hdr))
    order = sorted(
        (n for n in results),
        key=lambda n: -abs(results[n]["AALTO"]["delta_ms"]) if results[n]["AALTO"] else 0,
    )
    for name in order:
        r = results[name]
        line = f"{name:58s}"
        for s in TABLES:
            line += f"{r[s]['delta_ms']:10.2f}" if r[s] else f"{'--':>10s}"
        line += " | "
        for s in TABLES:
            line += f"{100*r[s]['frac_pos']:7.0f}%" if r[s] else f"{'--':>8s}"
        # cross-source sign agreement
        signs = {np.sign(r[s]["delta_ms"]) for s in TABLES if r[s]}
        line += "  AGREE" if len(signs) == 1 else "  ** SPLIT **"
        print(line)
    with open(S + "matched_prices.json", "w") as f:
        json.dump({k: {s: v for s, v in r.items()} for k, r in results.items()}, f, indent=1,
                  default=float)
    print(f"\nwrote {S}matched_prices.json")


if __name__ == "__main__":
    main()
