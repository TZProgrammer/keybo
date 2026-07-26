"""IS THE DIRECTION EFFECT LEARNED, OR INVENTED? — the training-support check.

The serve grid asks the model about all 930 ordered slot pairs, but the model only ever SAW
the pairs that occur in the stroke data. A direction column can only carry a learned effect
where BOTH orderings of a key pair were observed; where only one ordering (or neither) was
seen, the model's asymmetry on that pair is unconstrained extrapolation into a null space —
exactly the failure mode that produced the goodhart-row-blindness incident (the optimizer
queries OFF the training distribution).

This matters most for the FLAT (row-span 0) rolls, because that is where directed_angle newly
separates inward from outward (54 of 870 pairs), and where the measured attributable effect is
largest. So the question is not rhetorical: it decides whether that number is a finding.

For each unordered same-hand two-finger key pair this reports:
  * n_samples in each ordering (raw stroke samples, summed over participants/wpm)
  * whether BOTH orderings clear a support floor
and then re-runs the inroll-vs-outroll contrast restricted to WELL-SUPPORTED pairs only.

If the effect survives on supported pairs, it is learned. If it only exists on unsupported
pairs, it is the model inventing structure where it has no data — and must be reported as
such, not as a direction effect.
"""

from __future__ import annotations

import json
import os
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
SEEDS = [0, 1, 2]
#: A pair ordering counts as SUPPORTED at >= this many raw stroke samples. 10 mirrors
#: validate.build_cells' min_cell_samples floor (a starved cell is noise there too).
SUPPORT_FLOOR = 10
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
    surface = os.environ.get("KEYBO_SURFACE", "AALTO")
    out_path = OUT / (os.environ.get("KEYBO_OUT") or f"direction_support_{surface}.json")
    n_jobs = int(os.environ.get("KEYBO_NJOBS", "14"))
    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]
    slot_of = {p: i for i, p in enumerate(positions)}

    rows = load_surface(surface)
    log(f"{surface}: {len(rows)} rows")

    # --- observed support per ORDERED slot pair, straight from the stroke table ---------
    observed: dict[tuple[int, int], int] = defaultdict(int)
    for r in rows:
        if len(r.positions) != 2:
            continue
        a, b = r.positions
        if a in slot_of and b in slot_of:
            observed[(slot_of[a], slot_of[b])] += len(r.samples)

    # --- the same roll-pair enumeration structure.py uses ------------------------------
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
            oi, ii = slot_of[outer], slot_of[inner]
            n_in, n_out = observed.get((oi, ii), 0), observed.get((ii, oi), 0)
            roll_pairs.append(
                {
                    "inward": (oi, ii),
                    "outward": (ii, oi),
                    "row_span": abs(a[1] - b[1]),
                    "n_inward": n_in,
                    "n_outward": n_out,
                    "both_supported": bool(n_in >= SUPPORT_FLOOR and n_out >= SUPPORT_FLOOR),
                    "either_zero": bool(n_in == 0 or n_out == 0),
                }
            )

    by_span: dict[int, dict] = {}
    for span in sorted({p["row_span"] for p in roll_pairs}):
        sub = [p for p in roll_pairs if p["row_span"] == span]
        by_span[span] = {
            "n_pairs": len(sub),
            "n_both_supported": sum(1 for p in sub if p["both_supported"]),
            "n_either_zero": sum(1 for p in sub if p["either_zero"]),
            "median_min_n": float(np.median([min(p["n_inward"], p["n_outward"]) for p in sub])),
        }
    log("support by row span: " + json.dumps(by_span))

    # --- refit and re-run the contrast, restricted to supported pairs ------------------
    tensors: dict[str, list[np.ndarray]] = {}
    for arm, kw in [("v1", {}), ("placebo", {"placebo": True}), ("v2", {"direction": True})]:
        per_seed = []
        for seed in SEEDS:
            m = train_bigram_model(
                rows, target_wpm=90.0, geometry=geom, random_state=seed, n_jobs=n_jobs,
                **kw, **REG_LOLO,
            )
            per_seed.append(
                serve_grid(
                    m, geom, direction=bool(kw.get("direction")), placebo=bool(kw.get("placebo"))
                )
            )
        tensors[arm] = per_seed
        log(f"  {arm}: 3 seeds done")

    def contrast(T: np.ndarray, subset) -> float:
        """inward-ms minus outward-ms, unweighted mean. Negative = inrolls faster."""
        if not subset:
            return float("nan")
        return float(np.mean([T[p["inward"]] - T[p["outward"]] for p in subset]))

    mean = {k: np.mean(v, axis=0) for k, v in tensors.items()}
    out: dict = {
        "meta": {
            "surface": surface,
            "support_floor_raw_samples": SUPPORT_FLOOR,
            "question": (
                "Does the measured inroll-vs-outroll direction effect survive when the "
                "contrast is restricted to key pairs whose BOTH orderings were actually "
                "observed in training? If not, it is extrapolation into a null space, not a "
                "learned effect."
            ),
            "seeds": SEEDS,
        },
        "support_by_row_span": {str(k): v for k, v in by_span.items()},
        "pairs": roll_pairs,
    }

    subsets = {
        "all": roll_pairs,
        "both_supported": [p for p in roll_pairs if p["both_supported"]],
        "either_unobserved": [p for p in roll_pairs if p["either_zero"]],
    }
    res: dict = {}
    for label, sub in subsets.items():
        entry = {"n_pairs": len(sub)}
        for arm in ("v1", "placebo", "v2"):
            entry[arm] = contrast(mean[arm], sub)
            entry[f"{arm}_per_seed"] = [contrast(T, sub) for T in tensors[arm]]
        entry["attributable"] = entry["v2"] - entry["placebo"]
        entry["attributable_per_seed"] = [
            v - p for v, p in zip(entry["v2_per_seed"], entry["placebo_per_seed"], strict=True)
        ]
        entry["attributable_seed_spread"] = float(np.std(entry["attributable_per_seed"]))
        res[label] = entry
    out["contrast_by_support"] = res

    # and the same split WITHIN the flat (span-0) rolls, where the effect was largest
    flat = [p for p in roll_pairs if p["row_span"] == 0]
    flat_res: dict = {}
    for label, sub in {
        "flat_all": flat,
        "flat_both_supported": [p for p in flat if p["both_supported"]],
        "flat_either_unobserved": [p for p in flat if p["either_zero"]],
    }.items():
        entry = {"n_pairs": len(sub)}
        for arm in ("v1", "placebo", "v2"):
            entry[arm] = contrast(mean[arm], sub)
        entry["attributable"] = entry["v2"] - entry["placebo"]
        entry["attributable_per_seed"] = [
            contrast(v, sub) - contrast(p, sub)
            for v, p in zip(tensors["v2"], tensors["placebo"], strict=True)
        ]
        entry["attributable_seed_spread"] = float(np.std(entry["attributable_per_seed"]))
        flat_res[label] = entry
    out["flat_rolls_by_support"] = flat_res

    out_path.write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {out_path}")

    print("\n" + "=" * 88)
    print(f"TRAINING SUPPORT — {surface} — is the direction effect learned or extrapolated?")
    print("=" * 88)
    print(f"\nsupport per unordered same-hand two-finger key pair (floor {SUPPORT_FLOOR} samples):")
    print(f"{'row span':10s} {'pairs':>6s} {'both supported':>15s} {'either UNOBSERVED':>18s} {'median min n':>13s}")
    for span, v in by_span.items():
        print(f"{span:<10d} {v['n_pairs']:6d} {v['n_both_supported']:15d} "
              f"{v['n_either_zero']:18d} {v['median_min_n']:13.0f}")

    print("\ninroll minus outroll (negative = inrolls faster), unweighted ms:")
    print(f"{'subset':22s} {'n':>5s} {'v1':>9s} {'placebo':>9s} {'v2':>9s} {'ATTRIB':>9s} {'spread':>8s}")
    for label, e in res.items():
        print(f"{label:22s} {e['n_pairs']:5d} {e['v1']:9.3f} {e['placebo']:9.3f} "
              f"{e['v2']:9.3f} {e['attributable']:9.3f} {e['attributable_seed_spread']:8.3f}")
    print("\nFLAT (row-span 0) rolls only — where directed_angle newly separates:")
    for label, e in flat_res.items():
        print(f"{label:22s} {e['n_pairs']:5d} {e['v1']:9.3f} {e['placebo']:9.3f} "
              f"{e['v2']:9.3f} {e['attributable']:9.3f} {e['attributable_seed_spread']:8.3f}")
    print("ALL-DONE")


if __name__ == "__main__":
    main()
