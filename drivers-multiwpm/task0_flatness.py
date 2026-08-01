"""TASK 0 — empirically test the parent's claim that the fitted surface is FLAT above wpm=120.

The claim: all six shipped k31 models are stamped wpm_range=[60,120]; XGBoost does not
extrapolate, so a tree ensemble emits a CONSTANT beyond the trained envelope and any
wpm > ~120 returns ~the 120 prediction.

The test: score ONE fixed layout on the shipped production surface (the same TimeSurface
`analyze` uses) at wpm = 60,90,100,110,115,120,125,130,140 and print predicted ms/char.

Two DIFFERENT quantities are reported, and the distinction matters:

  raw ms/char        -- what the surface predicts. This FALLS with wpm by construction even
                        if the trees are frozen, because a LOGRAT model's ms is
                        exp(pred) * 12000 / wpm: the 1/wpm factor is arithmetic, not learned.
  logratio = pred    -- log(ms * wpm / 12000), i.e. the quantity the trees actually emit.
                        THIS is where the clamp shows: if the trees are frozen above 120,
                        the log-ratio is IDENTICAL at 120/125/130/140.

Flatness of the LOGRATIO above 120 is the parent's claim. Reporting only raw ms would make a
frozen model look like it was still responding to wpm (the 1/wpm arithmetic), which is the
exact trap the brief warns about ("do not compare raw predict() outputs as if they were ms" --
the converse trap is equally real).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from keybo.analysis.timecard import TimeSurface, _load_gz_model, _SEEDS
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.layouts import NAMED_LAYOUTS

WPMS = (60.0, 90.0, 100.0, 110.0, 115.0, 120.0, 125.0, 130.0, 140.0)


def main() -> int:
    lay = NAMED_LAYOUTS["qwerty"]
    probe = sys.argv[1] if len(sys.argv) > 1 else lay
    # Resolve a NAME to its board, and refuse anything that is not a full 30-char layout.
    # Without this, `task0_flatness.py qwerty` scores the literal 6-char string "qwerty":
    # TimeSurface.card builds slot_of from the string it is handed, so 24 of 30 keys simply
    # go missing, ~95% of the corpus is skipped as uncoverable, and the run still prints a
    # plausible ms/char. Caught in this campaign by a 19.6x total_ms discrepancy between two
    # runs of the same driver.
    probe = NAMED_LAYOUTS.get(probe, probe)
    if len(probe) != 30 or len(set(probe)) != 30:
        raise SystemExit(
            f"probe layout must be 30 distinct chars (or a NAMED_LAYOUTS key); got {probe!r} "
            f"({len(probe)} chars, {len(set(probe))} distinct)"
        )
    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))

    geo = ROW_STAGGERED_30
    positions = [*geo.slots, geo.space_position]

    # --- Part A: the corpus-level gauge, exactly as `analyze` computes it -----------------
    rows_a = []
    for wpm in WPMS:
        surf = TimeSurface(tri, target_wpm=wpm)
        card = surf.card(probe)
        rows_a.append({"wpm": wpm, "ms_per_char": card.ms_per_char, "total_ms": card.total_ms})
        print(f"[A] wpm={wpm:6.1f}  ms/char={card.ms_per_char:.6f}  total_ms={card.total_ms:.1f}")

    # --- Part B: the RAW TREE OUTPUT (log-ratio space), seed-averaged ---------------------
    # Same feature vectors as Part A's tables, but we keep the models' native output so the
    # tree clamp is visible without the 1/wpm arithmetic on top.
    bi_models = [_load_gz_model(f"bigram_reg31_seed{s}") for s in _SEEDS]
    tri_models = [_load_gz_model(f"trigram_cond31_seed{s}") for s in _SEEDS]

    # A small fixed sample of position pairs/triples so this is cheap and deterministic.
    pairs = [(positions[i], positions[j]) for i in (0, 3, 12, 15, 27) for j in (1, 4, 13, 20, 29)]
    triples = [
        (positions[i], positions[j], positions[k])
        for i in (0, 12, 27)
        for j in (4, 15)
        for k in (13, 29)
    ]

    rows_b = []
    for wpm in WPMS:
        Xb = np.vstack([bigram_features_from_positions(geo, p, wpm=wpm) for p in pairs])
        Xt = np.vstack([trigram_features_from_positions(geo, t, wpm=wpm) for t in triples])
        lb = np.mean([m.predict(Xb) for m in bi_models], axis=0)
        lt = np.mean([m.predict(Xt) for m in tri_models], axis=0)
        msb = np.mean([m.predict_ms(Xb) for m in bi_models], axis=0)
        rows_b.append(
            {
                "wpm": wpm,
                "bigram_logratio_mean": float(lb.mean()),
                "bigram_logratio_sum": float(lb.sum()),
                "trigram_logratio_mean": float(lt.mean()),
                "bigram_ms_mean": float(msb.mean()),
            }
        )
        print(
            f"[B] wpm={wpm:6.1f}  bigram logratio mean={lb.mean():.9f}  "
            f"trigram logratio mean={lt.mean():.9f}  bigram ms mean={msb.mean():.6f}"
        )

    # --- Part C: does the wpm feature even reach a split above 120? -----------------------
    # Direct evidence: dump every threshold on the `wpm` feature across all six boosters.
    thresholds = {}
    for name, models in (("bigram", bi_models), ("trigram", tri_models)):
        got = []
        for m in models:
            booster = m._regressor.get_booster() if hasattr(m, "_regressor") else None
            if booster is None:
                continue
            dump = booster.get_dump(dump_format="json")
            fnames = m.metadata.feature_names
            wpm_idx = fnames.index("wpm")
            wpm_key = f"f{wpm_idx}"

            def walk(node, out):
                if "split" in node:
                    if node["split"] in (wpm_key, "wpm"):
                        out.append(float(node["split_condition"]))
                    for ch in node.get("children", []):
                        walk(ch, out)

            out: list[float] = []
            for tree_json in dump:
                walk(json.loads(tree_json), out)
            got.append(sorted(set(out)))
        thresholds[name] = got
        for i, t in enumerate(got):
            hi = [x for x in t if x > 120.0]
            print(
                f"[C] {name} seed{i}: {len(t)} distinct wpm split thresholds, "
                f"range [{min(t) if t else float('nan'):.3f}, {max(t) if t else float('nan'):.3f}], "
                f"{len(hi)} above 120 -> {hi[:8]}"
            )

    out = {
        "probe_layout": probe,
        "part_a_corpus_surface": rows_a,
        "part_b_raw_tree_output": rows_b,
        "part_c_wpm_split_thresholds": thresholds,
    }
    with open(sys.argv[2] if len(sys.argv) > 2 else "task0_flatness.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
