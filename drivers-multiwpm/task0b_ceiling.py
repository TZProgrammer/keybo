"""TASK 0b — find the ACTUAL wpm response ceiling of the shipped models.

Task 0 already refuted "flat above 120" (log-ratio kept moving 120->140, and Part C found
26-52 distinct `wpm` split thresholds ABOVE 120, up to 213). This driver pins the real ceiling:

  1. Fine sweep of raw predict() (LOGRAT space) on FIXED feature vectors, wpm 100..260 step 1.
     A frozen tree ensemble emits a bit-identical value; the first wpm past which every
     prediction is bit-identical to the wpm=260 prediction IS the ensemble's ceiling.
  2. Histogram of the `wpm` split thresholds by decade, so the density of learned structure
     above 120 is visible rather than asserted.
"""

from __future__ import annotations

import json

import numpy as np

from keybo.analysis.timecard import _SEEDS, _load_gz_model
from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30


def wpm_thresholds(m) -> list[float]:
    booster = m._regressor.get_booster()
    wpm_key = f"f{m.metadata.feature_names.index('wpm')}"
    out: list[float] = []

    def walk(node):
        if "split" in node:
            if node["split"] in (wpm_key, "wpm"):
                out.append(float(node["split_condition"]))
            for ch in node.get("children", []):
                walk(ch)

    for tree_json in booster.get_dump(dump_format="json"):
        walk(json.loads(tree_json))
    return out


def main() -> int:
    geo = ROW_STAGGERED_30
    positions = [*geo.slots, geo.space_position]
    pairs = [(positions[i], positions[j]) for i in (0, 3, 12, 15, 27) for j in (1, 4, 13, 20, 29)]
    triples = [
        (positions[i], positions[j], positions[k])
        for i in (0, 12, 27)
        for j in (4, 15)
        for k in (13, 29)
    ]

    bi = [_load_gz_model(f"bigram_reg31_seed{s}") for s in _SEEDS]
    tri = [_load_gz_model(f"trigram_cond31_seed{s}") for s in _SEEDS]

    wpms = list(range(100, 261))
    result: dict = {"ceiling": {}, "thresholds": {}, "sweep": {}}

    for name, models, feat, items in (
        ("bigram", bi, bigram_features_from_positions, pairs),
        ("trigram", tri, trigram_features_from_positions, triples),
    ):
        # raw predict() per wpm, seed by seed (no averaging: a per-seed freeze is the fact)
        per_seed_curves = []
        for m in models:
            curve = []
            for w in wpms:
                X = np.vstack([feat(geo, it, wpm=float(w)) for it in items])
                curve.append(m.predict(X))
            per_seed_curves.append(np.array(curve))  # (n_wpm, n_items)

        ceilings = []
        for si, curve in enumerate(per_seed_curves):
            top = curve[-1]
            # first index from which EVERY later row is bit-identical to the last row
            frozen_from = len(wpms) - 1
            for i in range(len(wpms) - 1, -1, -1):
                if np.array_equal(curve[i], top):
                    frozen_from = i
                else:
                    break
            ceilings.append(wpms[frozen_from])
            print(f"[ceiling] {name} seed{si}: predictions frozen from wpm={wpms[frozen_from]}")
        result["ceiling"][name] = ceilings

        ths = [wpm_thresholds(m) for m in models]
        hist = {}
        for lo in range(0, 260, 20):
            hist[f"{lo}-{lo + 20}"] = [sum(1 for t in s if lo <= t < lo + 20) for s in ths]
        result["thresholds"][name] = {
            "n_split_nodes_on_wpm": [len(s) for s in ths],
            "max_threshold": [max(s) if s else None for s in ths],
            "hist_by_20wpm_bin": hist,
        }
        print(f"[thresholds] {name}: split NODES on wpm per seed = {[len(s) for s in ths]}")
        print(f"[thresholds] {name}: max threshold per seed = {[max(s) for s in ths]}")
        for k, v in hist.items():
            print(f"    {k:>8}: {v}")

        # seed-mean sweep for the report table (log-ratio; the space the trees emit)
        mean_curve = np.mean([c.mean(axis=1) for c in per_seed_curves], axis=0)
        result["sweep"][name] = {
            "wpm": wpms,
            "logratio_mean": [float(x) for x in mean_curve],
        }

    with open("task0b_ceiling.json", "w") as f:
        json.dump(result, f, indent=2)

    # a compact printed table at the wpms the report quotes
    print("\n[sweep] seed-mean raw log-ratio (the trees' own output):")
    for w in (100, 110, 120, 130, 140, 160, 180, 200, 210, 213, 215, 220, 240, 260):
        if w not in wpms:
            continue
        i = wpms.index(w)
        b = result["sweep"]["bigram"]["logratio_mean"][i]
        t = result["sweep"]["trigram"]["logratio_mean"][i]
        print(f"  wpm={w:4d}  bigram={b:.9f}  trigram={t:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
