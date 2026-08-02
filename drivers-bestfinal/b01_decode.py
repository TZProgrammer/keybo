"""B01 — decode the PRICEBAND-1 frontier perms into LAYOUT STRINGS, then re-verify.

The frontier `F(c) = min{ms/char : sfb <= c}` was certified by PRICEBAND-1 but its report only
printed SCORES. `c07_warm.json[cap]['perm']` holds the achieving permutation, so the board is
recoverable exactly — no re-search needed (prereg contingency not triggered).

Verifies, for every cap: decode(perm) -> string -> re-score on the SHIPPED gauge and the
SHIPPED sfb, and confirm both reproduce the stored `best` / `sfb_at_best`.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "drivers-priceband"))

import keybo  # noqa: E402

WT = os.path.abspath(os.path.join(HERE, ".."))
assert keybo.__file__.startswith(WT), f"WRONG KEYBO: {keybo.__file__} not under {WT}"

from boards import FIELD  # noqa: E402
from fasteval import CHARS, FastSurface  # noqa: E402
from fastsfb import FastGauges  # noqa: E402

WARM = "/local/home/zegertho/agent/state/priceband/artifacts/c07_warm.json"
OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b01_frontier_boards.json"


def decode(perm):
    """perm maps char-index -> slot-index (31 entries, space last). Invert to slot-major string."""
    p = list(perm)
    assert len(p) == 31, len(p)
    assert p[30] == 30, f"space must sit on the space slot, got {p[30]}"
    slot_to_char = {}
    for ci, slot in enumerate(p[:30]):
        assert slot not in slot_to_char, f"collision at slot {slot}"
        slot_to_char[slot] = CHARS[ci]
    assert len(slot_to_char) == 30
    return "".join(slot_to_char[i] for i in range(30))


def main():
    fs = FastSurface()
    fg = FastGauges()
    warm = json.load(open(WARM))

    # Reconciliation gate FIRST: the two published anchors must reproduce on my build.
    recon = {}
    for name in ("arm-B", "BALL-1"):
        recon[name] = {
            "published": {"arm-B": 253.900579, "BALL-1": 253.966426}[name],
            "measured": fs.ms_per_char(FIELD[name]),
            "sfb": fg.sfb_only(fg.perm(FIELD[name])),
        }
        recon[name]["abs_diff"] = abs(recon[name]["measured"] - recon[name]["published"])
    print("== RECONCILIATION (must be < 1e-5) ==")
    for k, v in recon.items():
        print(f"  {k:8s} published {v['published']:.6f}  measured {v['measured']:.6f}  "
              f"diff {v['abs_diff']:.2e}  sfb {v['sfb']:.4f}")
        assert v["abs_diff"] < 1e-5, f"RECONCILIATION FAILED for {k}"

    rows = {}
    for cap, rec in warm.items():
        lay = decode(rec["perm"])
        ms = fs.ms_per_char(lay)
        sfb = fg.sfb_only(fg.perm(lay))
        rows[cap] = {
            "cap": rec["cap"],
            "layout": lay,
            "ms_stored": rec["best"],
            "ms_remeasured": ms,
            "ms_absdiff": abs(ms - rec["best"]),
            "sfb_stored": rec["sfb_at_best"],
            "sfb_remeasured": sfb,
            "sfb_absdiff": abs(sfb - rec["sfb_at_best"]),
            "n_restarts": rec["n"],
            "matches_field_board": next((b for b, s in FIELD.items() if s == lay), None),
        }

    print("\n== FRONTIER BOARDS DECODED AND RE-VERIFIED ==")
    print(f"{'cap':>10} {'layout':32} {'ms(stored)':>12} {'ms(remeas)':>12} {'d':>9} "
          f"{'sfb':>8} {'=field?':>12}")
    ok = True
    for cap in sorted(rows, key=lambda x: float(x)):
        r = rows[cap]
        flag = "" if (r["ms_absdiff"] < 1e-6 and r["sfb_absdiff"] < 1e-9) else "  <-- MISMATCH"
        if flag:
            ok = False
        print(f"{r['cap']:>10.4g} {r['layout']:32} {r['ms_stored']:12.6f} "
              f"{r['ms_remeasured']:12.6f} {r['ms_absdiff']:9.1e} {r['sfb_remeasured']:8.4f} "
              f"{str(r['matches_field_board']):>12}{flag}")

    inf = rows["1000000000.0"]
    print(f"\nUnconstrained F(inf) board = {inf['layout']}  "
          f"({inf['matches_field_board']}) at {inf['ms_remeasured']:.6f}")

    json.dump({"reconciliation": recon, "frontier": rows, "all_verified": ok},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}   all_verified={ok}")
    assert ok, "at least one frontier board failed to re-verify"


if __name__ == "__main__":
    main()
