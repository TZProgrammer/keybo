"""KITCHEN-SINK measurement driver — matched-pair widened vs kitchen-sink, out-of-sample.

Extends the reviewed ``retrain_direction_measure.py`` design rather than re-deriving it. Emits
every table as GENERATED output and writes one JSON sidecar so the report is reproducible from a
single artifact. Run:

    PYTHONPATH=src OMP_NUM_THREADS=1 python agent-artifacts/kitchensink_measure.py \
        --bistrokes /path/bistrokes31_v1.tsv --tristrokes /path/tristrokes31_cond_v1.tsv \
        --out agent-artifacts/kitchensink_results.json

Design, all pre-registered in ``agent-artifacts/KITCHENSINK-preregistration.md``:

- **The baseline arm is the WIDENED frame, not the narrow one.** This branch builds on
  ``sfgated-eval``, so the widened frame is the incumbent here and the kitchen-sink block is the
  single variable under test. The narrow arm is ALSO run, for a three-way table and to confirm
  comparability against the two prior rounds.
- The A/B is one ``validate()`` call per arm, differing ONLY in the frame flag. Same rows, seeds,
  folds, cell floor.
- High-wpm gate: the matched cell's WIDENED ``bucket_rhos`` are the baseline; ``gated=False`` is
  reported UNSCOREABLE, never passing. A regression on ALL THREE seeds is STRUCTURAL; inconsistent
  is NOISE.
- Transfer: PAIRED per-(fold, seed) deltas (MOR-FIX-1) — never a mean of ratios, which can reorder
  across folds with unequal ceilings.
- Importance: separate FULL-DATA models per arm (the model the field would actually be scored by).
- Ranking: ``fitness()`` over the 15-name catalog through the reviewed scorer, NOT a
  re-implementation. This is the SEPARATE surface from the 4 LOLO folds.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from keybo.cli.analyze import _EXTRA_NAMED
from keybo.data.strokes import load_strokes
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_KITCHENSINK_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.model_scorer import BigramModelScorer, TrigramModelScorer
from keybo.training.train import train_bigram_model, train_trigram_model
from keybo.training.validate import validate
from keybo.verdicts import HIGH_WPM_FLOOR, bucket_regression_report

SEEDS = [0, 1, 2]
ALL_NAMED = {**NAMED_LAYOUTS, **_EXTRA_NAMED}

#: The three frames, in widening order. ``flags`` go straight to validate()/train_*().
ARMS = {
    "narrow": {},
    "widened": {"direction": True},
    "kitchensink": {"kitchensink": True},
}


def _feature_names(ngram: str, arm: str) -> list[str]:
    if ngram == "bigram":
        return {
            "narrow": BIGRAM_FEATURE_NAMES,
            "widened": BIGRAM_DIRECTION_FEATURE_NAMES,
            "kitchensink": BIGRAM_KITCHENSINK_FEATURE_NAMES,
        }[arm]
    return {
        "narrow": TRIGRAM_FEATURE_NAMES,
        "widened": TRIGRAM_DIRECTION_FEATURE_NAMES,
        "kitchensink": TRIGRAM_KITCHENSINK_FEATURE_NAMES,
    }[arm]


def _new_columns(ngram: str) -> list[str]:
    """The kitchen-sink columns — what this arm ADDS to the widened incumbent."""
    incumbent = set(_feature_names(ngram, "widened"))
    return [n for n in _feature_names(ngram, "kitchensink") if n not in incumbent]


# --- transfer (LOLO) ---------------------------------------------------------------------


def run_transfer(rows, ngram: str) -> dict:
    """Every arm through validate(), then attach a high-wpm verdict to the kitchen-sink cells.

    The gate's baseline is the matched (holdout, seed) cell of the WIDENED arm — the incumbent on
    this branch. A vs-narrow verdict is attached too, so the report can state both without
    re-running anything.
    """
    reports = {}
    for arm, flags in ARMS.items():
        print(f"  [{ngram}] arm={arm} ...", flush=True)
        reports[arm] = validate(rows, seeds=SEEDS, ngram=ngram, progress=False, **flags)

    def buckets_of(arm: str) -> dict[tuple[str, int], dict[int, float]]:
        out: dict[tuple[str, int], dict[int, float]] = {}
        for holdout, fold in reports[arm]["folds"].items():
            for rec in fold["seeds"]:
                out[(holdout, rec["seed"])] = {int(k): v for k, v in rec["bucket_rhos"].items()}
        return out

    wide_b, narrow_b = buckets_of("widened"), buckets_of("narrow")
    for holdout, fold in reports["kitchensink"]["folds"].items():
        for rec in fold["seeds"]:
            cand = {int(k): v for k, v in rec["bucket_rhos"].items()}
            key = (holdout, rec["seed"])
            rec["high_wpm_gate_vs_widened"] = bucket_regression_report(
                cand,
                wide_b.get(key, {}),
                f"kitchensink {holdout} seed={rec['seed']} vs widened",
                floor=HIGH_WPM_FLOOR,
            )
            rec["high_wpm_gate_vs_narrow"] = bucket_regression_report(
                cand,
                narrow_b.get(key, {}),
                f"kitchensink {holdout} seed={rec['seed']} vs narrow",
                floor=HIGH_WPM_FLOOR,
            )
    return reports


def paired_fold_deltas(reports: dict, baseline: str = "widened") -> list[dict]:
    """rho_kitchensink - rho_<baseline> per (holdout, seed) — the MOR-FIX-1 statistic."""
    base, cand = reports[baseline], reports["kitchensink"]
    rows = []
    for holdout in base["folds"]:
        b_by_seed = {r["seed"]: r for r in base["folds"][holdout]["seeds"]}
        c_by_seed = {r["seed"]: r for r in cand["folds"][holdout]["seeds"]}
        for seed in SEEDS:
            rb, rc = b_by_seed[seed], c_by_seed[seed]
            gate = rc.get(f"high_wpm_gate_vs_{baseline}", {})
            rows.append(
                {
                    "holdout": holdout,
                    "seed": seed,
                    "baseline_arm": baseline,
                    "ceiling": base["ceilings"][holdout],
                    "rho_baseline": rb["rho"],
                    "rho_kitchensink": rc["rho"],
                    "delta": rc["rho"] - rb["rho"],
                    "tau_baseline": rb["tau_all4"],
                    "tau_kitchensink": rc["tau_all4"],
                    "wmae_baseline": rb["wmae"],
                    "wmae_kitchensink": rc["wmae"],
                    "hw_gate_passed": gate.get("passed"),
                    "hw_gate_gated": gate.get("gated"),
                    "hw_regressing_buckets": gate.get("regressing_high_buckets", []),
                    "hw_deltas": gate.get("deltas", {}),
                }
            )
    return rows


def sign_consistency(deltas: list[dict]) -> dict:
    """Per-fold sign agreement across seeds — the pre-registered decision input.

    A fold counts as sign-consistent only if ALL of its seeds share a strictly non-zero sign, so a
    fold that is positive on two seeds and negative on the third is MIXED, not a win.
    """
    by_fold: dict[str, list[float]] = {}
    for row in deltas:
        by_fold.setdefault(row["holdout"], []).append(row["delta"])
    out = {}
    for fold, ds in sorted(by_fold.items()):
        pos, neg = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
        out[fold] = {
            "deltas": ds,
            "mean": float(np.mean(ds)),
            "n_pos": pos,
            "n_neg": neg,
            "verdict": "WIN" if pos == len(ds) else ("LOSS" if neg == len(ds) else "MIXED"),
        }
    return out


def structural_high_wpm(deltas: list[dict]) -> dict:
    """Split high-wpm regressions into STRUCTURAL (every seed of a fold) vs NOISE.

    The distinction the brief requires: the last two rounds both failed structurally on dvorak
    b120, and a regression on one seed of three is a different claim from one on all three.
    """
    by_fold: dict[str, list[list[int]]] = {}
    n_seeds: dict[str, int] = {}
    ungated = []
    for row in deltas:
        if row["hw_gate_gated"] is not True:
            ungated.append((row["holdout"], row["seed"]))
        by_fold.setdefault(row["holdout"], []).append(row["hw_regressing_buckets"])
        n_seeds[row["holdout"]] = n_seeds.get(row["holdout"], 0) + 1
    out = {"structural": {}, "noise": {}, "ungated_cells": ungated}
    for fold, per_seed in sorted(by_fold.items()):
        counts: dict[int, int] = {}
        for buckets in per_seed:
            for b in buckets:
                counts[b] = counts.get(b, 0) + 1
        for bucket, hits in sorted(counts.items()):
            entry = {"fold": fold, "bucket": bucket, "seeds_hit": hits, "of": n_seeds[fold]}
            key = "structural" if hits == n_seeds[fold] else "noise"
            out[key][f"{fold}_b{bucket}"] = entry
    return out


# --- importance --------------------------------------------------------------------------


def run_importance(rows, ngram: str) -> dict:
    """Total-gain importance from FULL-DATA kitchen-sink models per seed.

    ``used`` (gain > 0) and ``share of total gain`` answer different questions: a column can be
    USED on every seed and still be noise, which is exactly what the two prior direction rounds
    found. Both are reported so neither reads as the other.
    """
    names = _feature_names(ngram, "kitchensink")
    new_cols = _new_columns(ngram)
    train_fn = train_bigram_model if ngram == "bigram" else train_trigram_model

    per_seed = []
    for seed in SEEDS:
        model = train_fn(rows, target_wpm=90.0, kitchensink=True, random_state=seed, n_jobs=1)
        gain = model._regressor.get_booster().get_score(importance_type="gain")
        by_name = {nm: gain.get(f"f{i}", 0.0) for i, nm in enumerate(names)}
        total = sum(by_name.values()) or 1.0
        new_gain = sum(by_name[c] for c in new_cols)
        per_seed.append(
            {
                "seed": seed,
                "total_gain": total,
                "new_columns_gain": new_gain,
                "new_columns_frac": new_gain / total,
                "by_name": by_name,
                "used_new_columns": sorted(c for c in new_cols if by_name[c] > 0),
            }
        )
    used_all_seeds = sorted(
        c for c in new_cols if all(s["by_name"][c] > 0 for s in per_seed)
    )
    return {
        "ngram": ngram,
        "new_columns": new_cols,
        "n_new_columns": len(new_cols),
        "mean_new_columns_frac": float(np.mean([s["new_columns_frac"] for s in per_seed])),
        "used_on_all_seeds": used_all_seeds,
        "mean_col_frac": {
            nm: float(np.mean([s["by_name"][nm] / s["total_gain"] for s in per_seed]))
            for nm in names
        },
        "per_seed": per_seed,
    }


# --- ranking over the named-layout field -------------------------------------------------


def run_ranking(bi_rows, tri_rows, corpus) -> dict:
    """Rank the 15-name catalog under widened vs kitchen-sink FULL-DATA models (seed 0).

    The catalog is the SEPARATE scoring surface — not the 4 LOLO folds. Uses the reviewed model
    scorer, so charset intersection is handled by its own ``has_key``. Reports the argmin's
    stability and the median adjacent gap, because churn confined to near-ties is not a collapse.
    """
    from scipy.stats import kendalltau, spearmanr

    bigrams, trigrams = corpus
    layouts = {name: Layout(s, ROW_STAGGERED_30) for name, s in ALL_NAMED.items()}
    out = {}
    for ngram, rows, freqs, train_fn, ScorerCls, freq_arg in [
        ("bigram", bi_rows, bigrams, train_bigram_model, BigramModelScorer, "bigram_freqs"),
        ("trigram", tri_rows, trigrams, train_trigram_model, TrigramModelScorer, "trigram_freqs"),
    ]:
        if not rows:
            continue
        fits = {}
        for arm in ("widened", "kitchensink"):
            flags = ARMS[arm]
            model = train_fn(rows, target_wpm=90.0, random_state=0, n_jobs=1, **flags)
            scorer = ScorerCls(model, **{freq_arg: freqs}, target_wpm=90.0, **flags)
            fits[arm] = {name: scorer.fitness(lay) for name, lay in layouts.items()}

        names = list(layouts)
        order = {a: sorted(fits[a], key=lambda k: fits[a][k]) for a in fits}
        rank = {a: {n: i for i, n in enumerate(order[a])} for a in fits}
        tau = kendalltau(
            [rank["widened"][n] for n in names], [rank["kitchensink"][n] for n in names]
        ).statistic
        rho = spearmanr(
            [fits["widened"][n] for n in names], [fits["kitchensink"][n] for n in names]
        ).statistic
        # adjacent-gap profile on the incumbent, as % of fitness — the near-tie context
        sorted_fit = [fits["widened"][n] for n in order["widened"]]
        gaps = [
            100.0 * (b - a) / a for a, b in zip(sorted_fit, sorted_fit[1:], strict=False) if a
        ]
        moved = [n for n in names if rank["widened"][n] != rank["kitchensink"][n]]
        out[ngram] = {
            "n_layouts": len(names),
            "fit_widened": fits["widened"],
            "fit_kitchensink": fits["kitchensink"],
            "order_widened": order["widened"],
            "order_kitchensink": order["kitchensink"],
            "kendall_tau": float(tau),
            "spearman_rho": float(rho),
            "n_positions_moved": len(moved),
            "moved": moved,
            "argmin_widened": order["widened"][0],
            "argmin_kitchensink": order["kitchensink"][0],
            "argmin_stable": order["widened"][0] == order["kitchensink"][0],
            "median_adjacent_gap_pct": float(np.median(gaps)) if gaps else float("nan"),
            "argmin_margin_pct": float(gaps[0]) if gaps else float("nan"),
        }
    return out


# --- main --------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bistrokes", required=True)
    ap.add_argument("--tristrokes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-bigram", action="store_true")
    ap.add_argument("--skip-trigram", action="store_true")
    ap.add_argument("--skip-ranking", action="store_true")
    ap.add_argument(
        "--only-ranking",
        action="store_true",
        help="load both row sets but run ONLY the ranking measurement (so the two LOLO arms can "
        "run as separate concurrent processes and be merged afterwards)",
    )
    args = ap.parse_args()

    from pathlib import Path

    from keybo.data.corpus import load_frequencies

    do_bigram, do_trigram = not args.skip_bigram, not args.skip_trigram
    print("loading strokes...", flush=True)
    bi_rows = (
        load_strokes(args.bistrokes, ngram_len=2, wpm_threshold=0, min_samples=1)
        if do_bigram
        else []
    )
    tri_rows = (
        load_strokes(args.tristrokes, ngram_len=3, wpm_threshold=0, min_samples=1)
        if do_trigram
        else []
    )
    print(f"  bistroke rows: {len(bi_rows)}  tristroke rows: {len(tri_rows)}", flush=True)

    results: dict = {
        "seeds": SEEDS,
        "high_wpm_floor": HIGH_WPM_FLOOR,
        "arms": {a: dict(f) for a, f in ARMS.items()},
        "scope": {
            "n_bistroke_rows": len(bi_rows),
            "n_tristroke_rows": len(tri_rows),
            "lolo_folds": sorted({r.layout for r in (bi_rows or tri_rows)}),
        },
    }

    for ngram, rows in (("bigram", bi_rows), ("trigram", tri_rows)):
        if not rows or args.only_ranking:
            continue
        print(f"[{ngram}] transfer (LOLO, 3 arms x 3 seeds x 4 folds)...", flush=True)
        reports = run_transfer(rows, ngram)
        print(f"[{ngram}] importance (full-data kitchensink)...", flush=True)
        importance = run_importance(rows, ngram)
        deltas_w = paired_fold_deltas(reports, "widened")
        deltas_n = paired_fold_deltas(reports, "narrow")
        results[ngram] = {
            "transfer": reports,
            "paired_vs_widened": deltas_w,
            "paired_vs_narrow": deltas_n,
            "sign_consistency_vs_widened": sign_consistency(deltas_w),
            "sign_consistency_vs_narrow": sign_consistency(deltas_n),
            "high_wpm_vs_widened": structural_high_wpm(deltas_w),
            "importance": importance,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  checkpoint -> {args.out}", flush=True)

    if not args.skip_ranking:
        print("ranking over the named-layout catalog...", flush=True)
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
        results["ranking"] = run_ranking(
            bi_rows,
            tri_rows,
            (
                load_frequencies(str(corpus_dir / "bigrams.txt")),
                load_frequencies(str(corpus_dir / "trigrams.txt")),
            ),
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"results -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
