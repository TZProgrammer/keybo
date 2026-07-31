"""RETRAIN-DIRECTION measurement driver — matched-pair narrow vs widened, out-of-sample.

Emits every table in the report as GENERATED output (never hand-transcribed) and writes a JSON
sidecar so the report's numbers are reproducible from one artifact. Run:

    OMP_NUM_THREADS=1 python agent-artifacts/retrain_direction_measure.py \
        --bistrokes /path/bistrokes31_v1.tsv --tristrokes /path/tristrokes31_cond_v1.tsv \
        --out agent-artifacts/retrain_direction_results.json

Design (all pre-registered in agent-artifacts/RETRAIN-DIRECTION-preregistration.md):

- The A/B is a SINGLE call to keybo.training.validate.validate() per arm, differing ONLY in
  direction=True/False. Same rows, same seeds, same folds, same cell floor, same everything.
  So a rho delta is attributable to the added columns and nothing else.
- High-wpm gate: the NARROW arm's per-fold/seed bucket_rhos are the baseline; the widened arm is
  re-validated with baseline_buckets set to the matched cell's narrow bucket_rhos, so each
  fold/seed carries its own high_wpm_gate verdict. A gate that cannot run (gated=False) is
  reported UNSCOREABLE, not passing.
- Importance: separate FULL-DATA models (no holdout) per arm, so the importance is of the model
  the field would actually be scored by — total gain per column, from the booster.
- Ranking: model.fitness() over the 15-name catalog under BOTH arms, on the production corpus,
  through the reviewed scorer (not a re-implementation).
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
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
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


def _feature_names(ngram: str, direction: bool) -> list[str]:
    if ngram == "bigram":
        return BIGRAM_DIRECTION_FEATURE_NAMES if direction else BIGRAM_FEATURE_NAMES
    return TRIGRAM_DIRECTION_FEATURE_NAMES if direction else TRIGRAM_FEATURE_NAMES


def _new_columns(ngram: str) -> list[str]:
    narrow = set(_feature_names(ngram, False))
    return [n for n in _feature_names(ngram, True) if n not in narrow]


# --- transfer (LOLO) ---------------------------------------------------------------------


def run_transfer(rows, ngram: str) -> dict:
    """Both arms through validate(), then re-run the widened arm with the narrow arm's bucket
    rhos as the high-wpm baseline so each fold/seed cell gets its non-regression verdict."""
    narrow = validate(rows, seeds=SEEDS, ngram=ngram, direction=False, progress=False)
    widened = validate(rows, seeds=SEEDS, ngram=ngram, direction=True, progress=False)

    # Index narrow bucket_rhos by (holdout, seed) to serve as each widened cell's baseline.
    narrow_buckets: dict[tuple[str, int], dict[int, float]] = {}
    for holdout, fold in narrow["folds"].items():
        for rec in fold["seeds"]:
            narrow_buckets[(holdout, rec["seed"])] = {
                int(k): v for k, v in rec["bucket_rhos"].items()
            }

    # Attach a high-wpm gate verdict to every widened cell (narrow arm = baseline).
    for holdout, fold in widened["folds"].items():
        for rec in fold["seeds"]:
            cand = {int(k): v for k, v in rec["bucket_rhos"].items()}
            base = narrow_buckets.get((holdout, rec["seed"]), {})
            rec["high_wpm_gate_vs_narrow"] = bucket_regression_report(
                cand, base, f"widened {holdout} seed={rec['seed']} vs narrow", floor=HIGH_WPM_FLOOR
            )
    return {"narrow": narrow, "widened": widened, "narrow_buckets_index": True}


def paired_fold_deltas(transfer: dict) -> list[dict]:
    """rho_widened - rho_narrow per (holdout, seed) — the MOR-FIX-1 statistic (never a mean of
    ratios). Also carries each arm's fraction-of-ceiling for context."""
    narrow, widened = transfer["narrow"], transfer["widened"]
    rows = []
    for holdout in narrow["folds"]:
        n_by_seed = {r["seed"]: r for r in narrow["folds"][holdout]["seeds"]}
        w_by_seed = {r["seed"]: r for r in widened["folds"][holdout]["seeds"]}
        ceiling = narrow["ceilings"][holdout]
        for seed in SEEDS:
            rn, rw = n_by_seed[seed], w_by_seed[seed]
            rows.append(
                {
                    "holdout": holdout,
                    "seed": seed,
                    "ceiling": ceiling,
                    "rho_narrow": rn["rho"],
                    "rho_widened": rw["rho"],
                    "delta": rw["rho"] - rn["rho"],
                    "tau_narrow": rn["tau_all4"],
                    "tau_widened": rw["tau_all4"],
                    "wmae_narrow": rn["wmae"],
                    "wmae_widened": rw["wmae"],
                    "hw_gate_passed": rw["high_wpm_gate_vs_narrow"]["passed"],
                    "hw_gate_gated": rw["high_wpm_gate_vs_narrow"]["gated"],
                    "hw_regressing_buckets": rw["high_wpm_gate_vs_narrow"][
                        "regressing_high_buckets"
                    ],
                }
            )
    return rows


# --- importance --------------------------------------------------------------------------


def run_importance(rows, ngram: str) -> dict:
    """Total-gain importance from a FULL-DATA widened model per seed (the model the field would
    be scored by). Reports the new columns' share of total gain, averaged over seeds."""
    names = _feature_names(ngram, True)
    new_cols = _new_columns(ngram)
    train_fn = train_bigram_model if ngram == "bigram" else train_trigram_model

    per_seed = []
    for seed in SEEDS:
        model = train_fn(rows, target_wpm=90.0, direction=True, random_state=seed, n_jobs=1)
        booster = model._regressor.get_booster()
        gain = booster.get_score(importance_type="gain")  # {'f0': ..., 'f12': ...}
        total = sum(gain.values()) or 1.0
        by_name = {}
        for i, nm in enumerate(names):
            by_name[nm] = gain.get(f"f{i}", 0.0)
        new_gain = sum(by_name[c] for c in new_cols)
        per_seed.append(
            {
                "seed": seed,
                "total_gain": total,
                "new_columns_gain": new_gain,
                "new_columns_frac": new_gain / total,
                "by_name": by_name,
                "used_columns": sorted(k for k, v in by_name.items() if v > 0),
            }
        )
    mean_frac = float(np.mean([s["new_columns_frac"] for s in per_seed]))
    # Per-column mean fraction across seeds, for the generated importance table.
    col_frac = {}
    for nm in names:
        col_frac[nm] = float(
            np.mean([s["by_name"][nm] / s["total_gain"] for s in per_seed])
        )
    return {
        "ngram": ngram,
        "new_columns": new_cols,
        "mean_new_columns_frac": mean_frac,
        "per_seed": per_seed,
        "mean_col_frac": col_frac,
    }


# --- ranking over the named-layout field -------------------------------------------------


def run_ranking(bi_rows, tri_rows, corpus) -> dict:
    """Rank the named-layout field under narrow vs widened FULL-DATA models (seed 0), per ngram.
    Uses the reviewed model scorer; charset intersection is handled by the scorer's has_key."""
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
        model_n = train_fn(rows, target_wpm=90.0, direction=False, random_state=0, n_jobs=1)
        model_w = train_fn(rows, target_wpm=90.0, direction=True, random_state=0, n_jobs=1)
        scorer_n = ScorerCls(model_n, **{freq_arg: freqs}, target_wpm=90.0, direction=False)
        scorer_w = ScorerCls(model_w, **{freq_arg: freqs}, target_wpm=90.0, direction=True)

        fit_n = {name: scorer_n.fitness(lay) for name, lay in layouts.items()}
        fit_w = {name: scorer_w.fitness(lay) for name, lay in layouts.items()}
        order_n = sorted(fit_n, key=lambda k: fit_n[k])
        order_w = sorted(fit_w, key=lambda k: fit_w[k])
        rank_n = {name: i for i, name in enumerate(order_n)}
        rank_w = {name: i for i, name in enumerate(order_w)}
        names = list(layouts)
        tau = kendalltau([rank_n[n] for n in names], [rank_w[n] for n in names]).statistic
        rho = spearmanr([fit_n[n] for n in names], [fit_w[n] for n in names]).statistic
        moved = [n for n in names if rank_n[n] != rank_w[n]]
        out[ngram] = {
            "fit_narrow": fit_n,
            "fit_widened": fit_w,
            "order_narrow": order_n,
            "order_widened": order_w,
            "rank_narrow": rank_n,
            "rank_widened": rank_w,
            "kendall_tau": float(tau),
            "spearman_rho": float(rho),
            "n_positions_moved": len(moved),
            "moved": moved,
            "argmin_narrow": order_n[0],
            "argmin_widened": order_w[0],
        }
    return out


# --- main --------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bistrokes", required=True)
    ap.add_argument("--tristrokes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-trigram", action="store_true", help="bigram-only (faster smoke run)")
    ap.add_argument("--skip-bigram", action="store_true", help="trigram-only (merge with a prior run)")
    args = ap.parse_args()

    from keybo.data.corpus import load_frequencies

    do_bigram = not args.skip_bigram
    do_trigram = not args.skip_trigram

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

    # production corpus for the ranking measurement
    from pathlib import Path

    corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"
    bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
    trigrams = load_frequencies(str(corpus_dir / "trigrams.txt"))

    results: dict = {"seeds": SEEDS, "high_wpm_floor": HIGH_WPM_FLOOR}

    ngrams = [n for n, ok in (("bigram", do_bigram), ("trigram", do_trigram)) if ok]
    for ngram in ngrams:
        rows = bi_rows if ngram == "bigram" else tri_rows
        print(f"[{ngram}] transfer (LOLO, both arms)...", flush=True)
        transfer = run_transfer(rows, ngram)
        print(f"[{ngram}] importance (full-data widened)...", flush=True)
        importance = run_importance(rows, ngram)
        results[ngram] = {
            "transfer": transfer,
            "paired_fold_deltas": paired_fold_deltas(transfer),
            "importance": importance,
        }
        # Checkpoint after each ngram so a crash in the heavier trigram arm cannot lose the
        # bigram result.
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  checkpoint -> {args.out}", flush=True)

    print("ranking over the named-layout field...", flush=True)
    results["ranking"] = run_ranking(bi_rows, tri_rows, (bigrams, trigrams))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"results -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
