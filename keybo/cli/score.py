"""`keybo score` — compare named layouts on the learned objective."""

from __future__ import annotations

import argparse

from keybo.data.corpus import load_frequencies
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import BASELINE, NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.model_scorer import BigramModelScorer


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to a saved bigram model (.json)")
    parser.add_argument("--bigram-freqs", required=True, help="Path to the bigram frequency file")
    parser.add_argument("--target-wpm", type=float, default=90.0)
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=sorted(NAMED_LAYOUTS),
        help="Named layouts to compare (default: all known)",
    )


def run(args: argparse.Namespace) -> int:
    model = XGBoostTypingModel.load(args.model)
    freqs = load_frequencies(args.bigram_freqs)
    scorer = BigramModelScorer(model, bigram_freqs=freqs, target_wpm=args.target_wpm)

    baseline_score = scorer.fitness(Layout(NAMED_LAYOUTS[BASELINE], ROW_STAGGERED_30))
    print(f"{BASELINE} score: {baseline_score:.0f} (baseline)")

    for name in args.layouts:
        if name == BASELINE or name not in NAMED_LAYOUTS:
            continue
        score = scorer.fitness(Layout(NAMED_LAYOUTS[name], ROW_STAGGERED_30))
        improvement = (baseline_score - score) / baseline_score * 100 if baseline_score else 0.0
        print(f"{name} score: {score:.0f} (improvement over {BASELINE}: {improvement:+.2f}%)")
    return 0
