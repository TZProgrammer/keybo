"""ARM H — extract the exact 1-swap-ball facts the prereg quotes.

The 1-swap ball around arm B is EXHAUSTIVE (all 435 transpositions) and involves NO ARM H
search, so it is a prereg INPUT of the same kind as ARM G's D-of-existing-layouts table:
frozen-geometry enumeration around a frozen layout under frozen gauge definitions.

It answers the one question that decides ARM H's DESIGN:
  * is the feasible set non-trivially non-empty (does anything but arm B satisfy the 13
    hard axis constraints)?  and
  * how far outside arm B's speed does the nearest feasible-and-better layout sit?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import evobj as EV  # noqa: E402

ARMB = "flmpg-yuo,sntdcireahkxbwv'.jzq"
TARGET = "oxey-style"


def main() -> int:
    fe = EV.FastEval(corpus=None, weights_json=None, with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armh"), fe.corpus_dir
    from keybo.analysis.evidence_scorer import EXPECTED_SIGN, LIVE_GAUGES
    live = list(LIVE_GAUGES)
    con = [g for g in live if g != TARGET]
    dirs = {g: float(EXPECTED_SIGN[g]) for g in live}

    b = EV.perm_of(ARMB)
    gb = fe.gauges(np.stack([b]))
    ref = {g: float(gb[g][0]) for g in live}
    ref_ms = float(gb["_ms_per_char"][0])

    pairs = [(i, j) for i in range(30) for j in range(i + 1, 30)]
    nb = np.repeat(b[None, :], len(pairs), axis=0)
    for r, (i, j) in enumerate(pairs):
        nb[r, i], nb[r, j] = b[j], b[i]
    g = fe.gauges(nb)

    n_viol = np.zeros(len(nb), dtype=int)
    for a in con:
        n_viol += (dirs[a] * (g[a] - ref[a])) > 1e-9
    feas = n_viol == 0

    out: dict = {"armB_ms": ref_ms, "armB_oxey": ref[TARGET], "n_ball": len(nb)}
    rows = []
    for i in np.where(feas)[0]:
        lay = EV.layout_of(nb[i])
        rows.append({
            "layout": lay,
            "swap": list(pairs[int(i)]),
            "ms": float(g["_ms_per_char"][i]),
            "ms_minus_armB": float(g["_ms_per_char"][i]) - ref_ms,
            "oxey": float(g[TARGET][i]),
            "oxey_minus_armB": float(g[TARGET][i]) - ref[TARGET],
            "per_axis_excess": {a: float(dirs[a] * (float(g[a][i]) - ref[a])) for a in con},
        })
    out["feasible_in_1swap_ball"] = rows

    # oxey range across the WHOLE ball and across the oxey-improving subset -- this
    # calibrates the penalty scale (LAMBDA must dominate the achievable oxey gain).
    out["oxey_range_in_ball"] = {
        "min": float(g[TARGET].min()), "max": float(g[TARGET].max()),
        "span": float(g[TARGET].max() - g[TARGET].min()),
    }
    out["ms_range_in_ball"] = {
        "min": float(g["_ms_per_char"].min()), "max": float(g["_ms_per_char"].max()),
        "min_minus_armB": float(g["_ms_per_char"].min()) - ref_ms,
    }
    # how many clear a range of candidate speed bands, jointly with the 13 axes?
    bands = {}
    for edge in (0.0, 0.02, 0.05, 0.0617, 0.1, 0.1234, 0.2, 0.5, 1.0, 1e9):
        okms = g["_ms_per_char"] <= ref_ms + edge + 1e-9
        both = feas & okms
        better = both & (g[TARGET] < ref[TARGET] - 1e-6)
        bands[f"{edge:g}"] = {
            "n_ms_ok": int(okms.sum()), "n_ms_ok_and_13ax": int(both.sum()),
            "n_ms_ok_and_13ax_and_oxey_better": int(better.sum()),
            "best_oxey": (float(g[TARGET][better].min()) if better.any() else None),
        }
    out["joint_feasibility_by_speed_band"] = bands

    # WHICH axis is the tightest binder -- the count of the ball violating each axis, and
    # (for the EMPTY-set answer the brief asks for) the SMALLEST excess on each axis among
    # layouts that violate ONLY that axis.
    solo = {}
    for a in con:
        ex_a = dirs[a] * (g[a] - ref[a])
        others = np.zeros(len(nb), dtype=int)
        for c in con:
            if c != a:
                others += (dirs[c] * (g[c] - ref[c])) > 1e-9
        only_a = (ex_a > 1e-9) & (others == 0)
        solo[a] = {"n_violating_axis": int((ex_a > 1e-9).sum()),
                   "n_violating_ONLY_this_axis": int(only_a.sum()),
                   "min_excess_when_only_violator": (float(ex_a[only_a].min())
                                                     if only_a.any() else None)}
    out["per_axis_binding"] = solo

    # hand-partition invariance (ULTRAAUDIT-INTERIM): which ball members share arm B's
    # left/right character partition, making `alt`/`imbalance` ties BY CONSTRUCTION?
    from keybo.geometry import ROW_STAGGERED_30 as G
    slot_hand = np.array([G.hand(s[0]) for s in G.slots], dtype=int)
    assert len(slot_hand) == 30 and set(slot_hand.tolist()) == {-1, 1}, slot_hand

    def partition(perm31: np.ndarray) -> frozenset:
        # perm[i] = SLOT of char i; a char's hand is the hand of its slot
        return frozenset((i, int(slot_hand[int(perm31[i])])) for i in range(30))

    pb = partition(b)
    same_part = np.array([partition(nb[i]) == pb for i in range(len(nb))])
    out["hand_partition"] = {
        "n_ball_sharing_armB_partition": int(same_part.sum()),
        "note": ("`alt` and `imbalance` are HAND-PARTITION INVARIANTS (ULTRAAUDIT-INTERIM); "
                 "for any candidate sharing arm B's partition they are ties BY CONSTRUCTION "
                 "and must be reported as such, never counted as earned."),
    }
    for r in rows:
        p = partition(EV.perm_of(r["layout"]))
        r["shares_armB_hand_partition"] = (p == pb)

    json.dump(out, open(HERE / "ball-probe.json", "w"), indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nWROTE {HERE / 'ball-probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
