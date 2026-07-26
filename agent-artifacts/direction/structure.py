"""WHAT DOES A DIRECTION-AWARE OBJECTIVE FAVOUR? — the structural question.

The brief's question 5: "does a direction-aware objective favour different structure (inward
vs outward rolls, which finger ends higher)?"

Method — MATCHED REVERSES on the layout-neutral serve grid, which is the only honest frame
here (OQ-1: the corpus is ~98.7% qwerty, so anything read off raw training data is
correlation, not price):

  For every unordered pair {a,b}, the served tensor gives BOTH orderings. The direction
  effect on that pair is  T2[a,b] - T2[b,a].  Under v1 this difference is entirely the
  landing-key price (THEORY-1 D2 proved the non-landing features are swap-identical); under
  v2 it may additionally carry a genuine direction term.

  So the ATTRIBUTABLE direction effect for a pair is
        (v2 asymmetry) - (placebo asymmetry)
  and the placebo — same width, zero new information — is what makes it attributable rather
  than a frame-width artifact (TOOLING-TRAPS #17).

Two structural questions, each on the classes the community actually argues about:

  1. INROLL vs OUTROLL. For same-hand two-finger pairs, is travelling toward the index
     finger faster than travelling toward the pinky, on the SAME key pair? Under v1 this is
     0.000 by construction; the v2 number is the real answer.
  2. WHICH FINGER ENDS HIGHER. For the vertical version: on the same pair of keys spanning
     rows, does it matter which finger ends up on the higher row? (theory-1 found
     _PREFERRED_HEIGHT sign-splits by finger pair; this re-asks it where direction CAN be
     expressed.)

Reported per finger pair, weighted by iWeb bigram frequency AND unweighted, in ms.
"""

from __future__ import annotations

import hashlib
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
CORPUS = Path("/local/home/zegertho/repos/keybo/data/corpus")
C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
SEEDS = [0, 1, 2]
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def serve_grid(model, geom, direction: bool = False, placebo: bool = False) -> np.ndarray:
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


def qwerty_pair_weights(geom) -> dict[tuple[int, int], float]:
    """iWeb frequency mass per ORDERED slot pair, using qwerty's own char->slot map.

    Used ONLY as a weighting so the classes are not dominated by pairs nobody types. Stated
    plainly because it IS a qwerty-conditional weighting (OQ-1); the unweighted number is
    reported alongside so the reader can see whether the weighting drives anything.
    """
    slot_of = {ch: i for i, ch in enumerate(C30M)}
    slot_of[" "] = 30
    w: dict[tuple[int, int], float] = defaultdict(float)
    for line in (CORPUS / "bigrams.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or len(parts[0]) != 2:
            continue
        a, b = parts[0]
        if a in slot_of and b in slot_of:
            w[(slot_of[a], slot_of[b])] += float(parts[1])
    return w


def main() -> None:
    surface = os.environ.get("KEYBO_SURFACE", "AALTO")
    out_path = OUT / (os.environ.get("KEYBO_OUT") or f"direction_structure_{surface}.json")
    n_jobs = int(os.environ.get("KEYBO_NJOBS", "16"))
    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]

    rows = load_surface(surface)
    log(f"{surface}: {len(rows)} rows")
    weights = qwerty_pair_weights(geom)

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
                    m,
                    geom,
                    direction=bool(kw.get("direction")),
                    placebo=bool(kw.get("placebo")),
                )
            )
            log(f"  {arm} seed {seed} done")
        tensors[arm] = per_seed

    out: dict = {
        "meta": {
            "surface": surface,
            "frame": "served g(geometry,wpm) only, wpm 90, ROW_STAGGERED_30 + space",
            "seeds": SEEDS,
            "corpus_md5": hashlib.md5((CORPUS / "bigrams.txt").read_bytes()).hexdigest(),
            "estimator": (
                "MATCHED REVERSES on the serve grid: for each unordered pair, "
                "T2[a,b] - T2[b,a]. The ATTRIBUTABLE direction effect is the v2 asymmetry "
                "MINUS the placebo asymmetry (same-width control, TOOLING-TRAPS #17)."
            ),
            "weighting_caveat": (
                "Frequency weights use QWERTY's char->slot map on iWeb, so the weighted "
                "number is qwerty-conditional (OQ-1). Unweighted reported alongside."
            ),
        }
    }

    # --- 1. INROLL vs OUTROLL, matched on the key pair --------------------------------
    # For a same-hand two-finger pair, compare the inward ordering against the outward one.
    # Under v1 this is identically 0 for the non-landing features, but the LANDING key
    # differs between the two orderings, so v1 still shows a number — that number IS the
    # landing-key price (THEORY-1 D2). The v2-minus-placebo delta is the direction term.
    roll_pairs = []
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i >= j:
                continue
            if not (C.same_hand(geom, a, b) and not C.same_finger(geom, a, b)):
                continue
            if abs(a[0]) == abs(b[0]):
                continue
            # orient: (outer -> inner) is the INWARD ordering
            outer, inner = (a, b) if abs(a[0]) > abs(b[0]) else (b, a)
            oi, ii = positions.index(outer), positions.index(inner)
            fp = tuple(sorted([geom.finger(a[0]).value, geom.finger(b[0]).value]))
            roll_pairs.append(
                {
                    "inward": (oi, ii),  # outer -> inner  == travelling toward the index
                    "outward": (ii, oi),
                    "finger_pair": f"{fp[0]}|{fp[1]}",
                    "row_span": abs(a[1] - b[1]),
                }
            )
    log(f"{len(roll_pairs)} unordered same-hand two-finger key pairs")

    def inroll_minus_outroll(T: np.ndarray, subset=None) -> tuple[float, float]:
        """(unweighted mean, freq-weighted mean) of inward-ms minus outward-ms.

        NEGATIVE means the inward ordering is FASTER (an inroll advantage).
        """
        vals, ws = [], []
        for p in roll_pairs if subset is None else subset:
            d = T[p["inward"]] - T[p["outward"]]
            vals.append(d)
            ws.append(weights.get(p["inward"], 0.0) + weights.get(p["outward"], 0.0))
        vals = np.array(vals)
        ws = np.array(ws)
        wm = float((vals * ws).sum() / ws.sum()) if ws.sum() else float("nan")
        return float(vals.mean()), wm

    mean = {k: np.mean(v, axis=0) for k, v in tensors.items()}
    res: dict = {}
    for arm in ("v1", "placebo", "v2"):
        u, w = inroll_minus_outroll(mean[arm])
        per_seed_u = [inroll_minus_outroll(T)[0] for T in tensors[arm]]
        res[arm] = {
            "inroll_minus_outroll_ms_unweighted": u,
            "inroll_minus_outroll_ms_freqweighted": w,
            "per_seed_unweighted": [float(x) for x in per_seed_u],
            "seed_spread": float(np.std(per_seed_u)),
        }
    res["attributable_direction_effect_ms"] = {
        "unweighted": res["v2"]["inroll_minus_outroll_ms_unweighted"]
        - res["placebo"]["inroll_minus_outroll_ms_unweighted"],
        "freqweighted": res["v2"]["inroll_minus_outroll_ms_freqweighted"]
        - res["placebo"]["inroll_minus_outroll_ms_freqweighted"],
        "note": "v2 minus placebo. NEGATIVE = inrolls faster. Compare to the seed spread.",
    }
    # by finger pair, and by row span (flat rolls are where directed_angle newly separates)
    by_fp: dict[str, dict] = {}
    for fp in sorted({p["finger_pair"] for p in roll_pairs}):
        sub = [p for p in roll_pairs if p["finger_pair"] == fp]
        row = {"n_pairs": len(sub)}
        for arm in ("v1", "placebo", "v2"):
            u, w = inroll_minus_outroll(mean[arm], sub)
            row[arm] = {"unweighted": u, "freqweighted": w}
        row["attributable"] = row["v2"]["unweighted"] - row["placebo"]["unweighted"]
        by_fp[fp] = row
    res["by_finger_pair"] = by_fp
    by_span: dict[str, dict] = {}
    for span in sorted({p["row_span"] for p in roll_pairs}):
        sub = [p for p in roll_pairs if p["row_span"] == span]
        row = {"n_pairs": len(sub)}
        for arm in ("v1", "placebo", "v2"):
            u, w = inroll_minus_outroll(mean[arm], sub)
            row[arm] = {"unweighted": u, "freqweighted": w}
        row["attributable"] = row["v2"]["unweighted"] - row["placebo"]["unweighted"]
        by_span[str(span)] = row
    res["by_row_span"] = by_span
    out["inroll_vs_outroll"] = res

    # --- 2. WHICH FINGER ENDS HIGHER (the vertical direction question) -----------------
    # Same key pair spanning rows: compare "ends on the higher row" vs "ends on the lower".
    # NOTE this is confounded with the landing row by construction (reversing a two-row pair
    # swaps which key you land on — theory-1's self-audit #3 hit exactly this), which is
    # PRECISELY why the placebo subtraction is required: the landing-row price is identical
    # in both arms, so it cancels in v2-minus-placebo.
    vert = []
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i >= j or not C.same_hand(geom, a, b) or C.same_finger(geom, a, b):
                continue
            if a[1] == b[1]:
                continue
            lo, hi = (a, b) if a[1] < b[1] else (b, a)
            vert.append((positions.index(lo), positions.index(hi)))

    def up_minus_down(T: np.ndarray) -> tuple[float, float]:
        """(unweighted, weighted) ms for 'ending on the HIGHER row' minus 'ending lower'."""
        vals, ws = [], []
        for loi, hii in vert:
            vals.append(T[loi, hii] - T[hii, loi])
            ws.append(weights.get((loi, hii), 0.0) + weights.get((hii, loi), 0.0))
        vals, ws = np.array(vals), np.array(ws)
        return (
            float(vals.mean()),
            float((vals * ws).sum() / ws.sum()) if ws.sum() else float("nan"),
        )

    vres: dict = {"n_pairs": len(vert)}
    for arm in ("v1", "placebo", "v2"):
        u, w = up_minus_down(mean[arm])
        ps = [up_minus_down(T)[0] for T in tensors[arm]]
        vres[arm] = {
            "end_higher_minus_end_lower_ms_unweighted": u,
            "end_higher_minus_end_lower_ms_freqweighted": w,
            "seed_spread": float(np.std(ps)),
        }
    vres["attributable_direction_effect_ms"] = (
        vres["v2"]["end_higher_minus_end_lower_ms_unweighted"]
        - vres["placebo"]["end_higher_minus_end_lower_ms_unweighted"]
    )
    vres["confound_note"] = (
        "Reversing a two-row pair ALWAYS swaps the landing row, so the raw number is the "
        "bottom-row price (theory-1 T4) under a direction-shaped name. Only the "
        "v2-minus-placebo delta is a direction effect: the landing-row price is identical "
        "in both arms and cancels."
    )
    out["which_finger_ends_higher"] = vres

    out_path.write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {out_path}")

    # --- report ------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"STRUCTURE — {surface} — matched reverses on the served serve grid (3-seed mean)")
    print("=" * 90)
    print("\n1) INROLL minus OUTROLL on the SAME key pair (negative = inrolls faster), ms:")
    print(f"{'arm':10s} {'unweighted':>12s} {'freq-weighted':>14s} {'seed spread':>12s}")
    for arm in ("v1", "placebo", "v2"):
        r = res[arm]
        print(f"{arm:10s} {r['inroll_minus_outroll_ms_unweighted']:12.4f} "
              f"{r['inroll_minus_outroll_ms_freqweighted']:14.4f} {r['seed_spread']:12.4f}")
    ad = res["attributable_direction_effect_ms"]
    print(f"{'ATTRIB':10s} {ad['unweighted']:12.4f} {ad['freqweighted']:14.4f}"
          f"   <- v2 - placebo = the direction effect")
    print("\n   by finger pair (unweighted ms, attributable = v2 - placebo):")
    for fp, r in sorted(by_fp.items(), key=lambda kv: -abs(kv[1]["attributable"])):
        print(f"     {fp:28s} n={r['n_pairs']:3d}  v1 {r['v1']['unweighted']:+8.3f}  "
              f"placebo {r['placebo']['unweighted']:+8.3f}  v2 {r['v2']['unweighted']:+8.3f}  "
              f"ATTRIB {r['attributable']:+8.3f}")
    print("\n   by row span:")
    for sp, r in by_span.items():
        print(f"     span {sp}  n={r['n_pairs']:3d}  v1 {r['v1']['unweighted']:+8.3f}  "
              f"placebo {r['placebo']['unweighted']:+8.3f}  v2 {r['v2']['unweighted']:+8.3f}  "
              f"ATTRIB {r['attributable']:+8.3f}")

    print(f"\n2) ENDING ON THE HIGHER ROW minus ending lower ({len(vert)} pairs), ms:")
    for arm in ("v1", "placebo", "v2"):
        r = vres[arm]
        print(f"   {arm:10s} unweighted {r['end_higher_minus_end_lower_ms_unweighted']:+9.4f}  "
              f"weighted {r['end_higher_minus_end_lower_ms_freqweighted']:+9.4f}  "
              f"spread {r['seed_spread']:.4f}")
    print(f"   ATTRIBUTABLE (v2 - placebo): {vres['attributable_direction_effect_ms']:+9.4f} ms")
    print("   " + vres["confound_note"])
    print("ALL-DONE")


if __name__ == "__main__":
    main()
