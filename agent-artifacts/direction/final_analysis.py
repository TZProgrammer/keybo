"""THE CLEAN CROSS-SOURCE TEST: common support set + hand symmetry.

Two defects in the per-surface support runs that this fixes, because both could manufacture
a cross-source "split" that is really an artifact:

1. **The three surfaces' "both_supported" subsets are DIFFERENT KEY SETS** (100 pairs on
   AALTO, 61 on COMMUNITY, 112 on POOL). Comparing -9.04 ms against -0.63 ms across
   different key sets is exactly TOOLING-TRAPS #16: a contrast whose groups differ in
   composition cannot isolate the axis it names. Fix: restrict every surface to the pairs
   supported in ALL THREE, and compare on that single common set.

2. **Hand symmetry is a falsification test the effect must pass.** A real biomechanical
   inroll advantage is a property of the hand, so it should appear on BOTH hands at similar
   magnitude. If it is left/right asymmetric, the model is fitting something
   data-specific (the training corpus is ~98.7% qwerty, whose letter placement is not
   hand-symmetric) rather than a motion cost. AALTO's per-finger-pair table looked strongly
   right-hand-loaded, so this needs measuring, not eyeballing.

Saves the served tensors so any further slicing costs nothing.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

WT = "/local/home/zegertho/agent/state/direction/wt-direction"
sys.path.insert(0, f"{WT}/src")
sys.path.insert(0, "/local/home/zegertho/agent/state/direction/scratch")

from keybo.features import classify as C  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from refit import REG_LOLO, load_surface  # noqa: E402

OUT = Path("/local/home/zegertho/agent/state/direction/artifacts")
TENSOR_DIR = OUT / "tensors"
TENSOR_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]
SUPPORT_FLOOR = 10
SURFACES = ["AALTO", "COMMUNITY", "POOL"]
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def serve_grid(model, geom, direction=False, placebo=False) -> np.ndarray:
    positions = [*geom.slots, geom.space_position]
    X = np.vstack(
        [
            bigram_features_from_positions(
                geom, (a, b), wpm=90.0, direction=direction, placebo=placebo
            )
            for a in positions
            for b in positions
        ]
    )
    return model.to_ms(model.predict(X), X).reshape(len(positions), len(positions))


def main() -> None:
    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]
    slot_of = {p: i for i, p in enumerate(positions)}

    # --- the roll-pair frame (identical across surfaces) --------------------------------
    roll_pairs = []
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i >= j:
                continue
            if not (C.same_hand(geom, a, b) and not C.same_finger(geom, a, b)):
                continue
            if abs(a[0]) == abs(b[0]):
                continue
            outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
            roll_pairs.append(
                {
                    "key": f"{outer}->{inner}",
                    "inward": (slot_of[outer], slot_of[inner]),
                    "outward": (slot_of[inner], slot_of[outer]),
                    "row_span": abs(a[1] - b[1]),
                    "hand": "left" if a[0] < 0 else "right",
                }
            )

    support: dict[str, set[str]] = {}
    tensors: dict[str, dict[str, list[np.ndarray]]] = {}

    for surface in SURFACES:
        rows = load_surface(surface)
        log(f"{surface}: {len(rows)} rows")
        observed: dict[tuple[int, int], int] = defaultdict(int)
        for r in rows:
            if len(r.positions) != 2:
                continue
            a, b = r.positions
            if a in slot_of and b in slot_of:
                observed[(slot_of[a], slot_of[b])] += len(r.samples)
        support[surface] = {
            p["key"]
            for p in roll_pairs
            if observed.get(p["inward"], 0) >= SUPPORT_FLOOR
            and observed.get(p["outward"], 0) >= SUPPORT_FLOOR
        }
        log(f"  {surface}: {len(support[surface])} of {len(roll_pairs)} pairs supported")

        tensors[surface] = {}
        for arm, kw in [("v1", {}), ("placebo", {"placebo": True}), ("v2", {"direction": True})]:
            per_seed = []
            for seed in SEEDS:
                m = train_bigram_model(
                    rows, target_wpm=90.0, geometry=geom, random_state=seed, n_jobs=10,
                    **kw, **REG_LOLO,
                )
                T = serve_grid(
                    m, geom, direction=bool(kw.get("direction")), placebo=bool(kw.get("placebo"))
                )
                np.save(TENSOR_DIR / f"T2_{surface}_{arm}_seed{seed}.npy", T)
                per_seed.append(T)
            tensors[surface][arm] = per_seed
            log(f"  {surface}/{arm}: 3 seeds done (tensors saved)")
        del rows

    common = set.intersection(*[support[s] for s in SURFACES])
    log(f"COMMON supported pairs across all 3 surfaces: {len(common)}")

    def contrast(T: np.ndarray, subset) -> float:
        if not subset:
            return float("nan")
        return float(np.mean([T[p["inward"]] - T[p["outward"]] for p in subset]))

    def attributable(surface: str, subset) -> tuple[float, float, list[float]]:
        """(seed-mean attributable effect, seed spread, per-seed) for v2 minus placebo."""
        per_seed = [
            contrast(v, subset) - contrast(p, subset)
            for v, p in zip(tensors[surface]["v2"], tensors[surface]["placebo"], strict=True)
        ]
        return float(np.mean(per_seed)), float(np.std(per_seed)), [float(x) for x in per_seed]

    out: dict = {
        "meta": {
            "question": (
                "Is the inroll-vs-outroll direction effect (a) consistent across sources when "
                "measured on ONE COMMON supported key set, and (b) symmetric across hands?"
            ),
            "support_floor_raw_samples": SUPPORT_FLOOR,
            "seeds": SEEDS,
            "n_roll_pairs": len(roll_pairs),
            "why_common_set": (
                "Per-surface 'both_supported' subsets are different KEY SETS, so comparing "
                "their contrasts confounds the source with the composition "
                "(TOOLING-TRAPS #16). The common set removes that."
            ),
            "hand_symmetry_rationale": (
                "A biomechanical inroll advantage is a property of the hand and should appear "
                "on both at similar magnitude. Left/right asymmetry indicates the model is "
                "fitting corpus-specific placement (~98.7% qwerty, OQ-1), not motion cost."
            ),
            "pool_caveat": "POOL is NOT independent — it contains AALTO and COMMUNITY.",
        },
        "support_counts": {s: len(support[s]) for s in SURFACES},
        "n_common_supported": len(common),
    }

    # --- A. common-set cross-source comparison ------------------------------------------
    subsets = {
        "common_all_spans": [p for p in roll_pairs if p["key"] in common],
        "common_flat_span0": [p for p in roll_pairs if p["key"] in common and p["row_span"] == 0],
        "common_span1": [p for p in roll_pairs if p["key"] in common and p["row_span"] == 1],
        "common_span2": [p for p in roll_pairs if p["key"] in common and p["row_span"] == 2],
    }
    common_res: dict = {}
    for label, sub in subsets.items():
        entry = {"n_pairs": len(sub)}
        for s in SURFACES:
            m, sd, ps = attributable(s, sub)
            entry[s] = {"attributable_ms": m, "seed_spread": sd, "per_seed": ps}
            entry[f"{s}_v1"] = contrast(np.mean(tensors[s]["v1"], axis=0), sub)
            entry[f"{s}_v2"] = contrast(np.mean(tensors[s]["v2"], axis=0), sub)
        signs = {np.sign(entry[s]["attributable_ms"]) for s in SURFACES}
        entry["sign_agrees_across_sources"] = len(signs) == 1
        entry["all_outside_own_seed_spread"] = all(
            abs(entry[s]["attributable_ms"]) > entry[s]["seed_spread"] for s in SURFACES
        )
        common_res[label] = entry
    out["common_set"] = common_res

    # --- B. hand symmetry ---------------------------------------------------------------
    hand_res: dict = {}
    for label, span in [("all_spans", None), ("flat_span0", 0)]:
        entry: dict = {}
        for hand in ("left", "right"):
            sub = [
                p
                for p in roll_pairs
                if p["key"] in common
                and p["hand"] == hand
                and (span is None or p["row_span"] == span)
            ]
            entry[hand] = {"n_pairs": len(sub)}
            for s in SURFACES:
                m, sd, _ = attributable(s, sub)
                entry[hand][s] = {"attributable_ms": m, "seed_spread": sd}
        for s in SURFACES:
            lo = entry["left"][s]["attributable_ms"]
            ro = entry["right"][s]["attributable_ms"]
            entry.setdefault("left_minus_right_ms", {})[s] = lo - ro
            entry.setdefault("sign_agrees_across_hands", {})[s] = bool(
                np.sign(lo) == np.sign(ro)
            )
        hand_res[label] = entry
    out["hand_symmetry"] = hand_res

    (OUT / "direction_final_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {OUT / 'direction_final_analysis.json'}")

    # --- report --------------------------------------------------------------------------
    print("\n" + "=" * 94)
    print("CLEAN CROSS-SOURCE TEST — one COMMON supported key set + hand symmetry")
    print("=" * 94)
    print(f"\nsupported roll pairs (of {len(roll_pairs)}): " +
          ", ".join(f"{s} {len(support[s])}" for s in SURFACES) +
          f"  =>  COMMON {len(common)}")
    print("\nA) attributable inroll-minus-outroll (v2 - placebo), ms, ON THE COMMON SET")
    print("   negative = inrolls faster. '+-' is the 3-seed spread.")
    print(f"{'subset':20s} {'n':>4s} " + " ".join(f"{s:>22s}" for s in SURFACES) +
          "   sign agrees?  all>spread?")
    for label, e in common_res.items():
        cells = " ".join(
            f"{e[s]['attributable_ms']:+13.3f}+-{e[s]['seed_spread']:6.3f}" for s in SURFACES
        )
        print(f"{label:20s} {e['n_pairs']:4d} {cells}   "
              f"{str(e['sign_agrees_across_sources']):>10s}  {str(e['all_outside_own_seed_spread']):>10s}")

    print("\nB) HAND SYMMETRY on the common set (a real biomechanical effect should match)")
    for label, e in hand_res.items():
        print(f"\n   {label}: left n={e['left']['n_pairs']}, right n={e['right']['n_pairs']}")
        for s in SURFACES:
            lo, ro = e["left"][s], e["right"][s]
            print(f"     {s:10s} left {lo['attributable_ms']:+8.3f}+-{lo['seed_spread']:.3f}   "
                  f"right {ro['attributable_ms']:+8.3f}+-{ro['seed_spread']:.3f}   "
                  f"L-R {e['left_minus_right_ms'][s]:+8.3f}   "
                  f"same sign: {e['sign_agrees_across_hands'][s]}")
    print("\nALL-DONE")


if __name__ == "__main__":
    main()
