"""`keybo tune` — hyperparameter search for the typing-time model."""

from __future__ import annotations

import argparse
import json

from keybo.cli._paths import ensure_writable_output
from keybo.data.strokes import load_strokes
from keybo.training.train import build_training_matrix
from keybo.training.tune import tune_hyperparameters


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strokes", required=True, help="Path to the bistroke/tristroke TSV")
    parser.add_argument("--ngram", choices=["bigram", "trigram"], default="bigram")
    parser.add_argument("--output", default="best_hyperparams.json", help="Where to write params")
    parser.add_argument("--target-wpm", type=float, default=90.0)
    parser.add_argument("--wpm-threshold", type=int, default=60)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)


def run(args: argparse.Namespace) -> int:
    # Fail fast: the search below can run for a long time; don't discover a bad output then.
    ensure_writable_output(args.output, "--output")
    ngram_len = 2 if args.ngram == "bigram" else 3
    rows = load_strokes(
        args.strokes,
        ngram_len=ngram_len,
        wpm_threshold=args.wpm_threshold,
        min_samples=args.min_samples,
    )
    if not rows:
        print("No stroke rows survived filtering; check the input and thresholds.")
        return 1

    X, y = build_training_matrix(rows, ngram=args.ngram, target_wpm=args.target_wpm)
    best = tune_hyperparameters(X, y, n_iter=args.n_iter, cv=args.cv, seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best hyperparameters -> {args.output}: {best}")
    return 0
