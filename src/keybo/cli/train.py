"""`keybo train` — fit a typing-time model from stroke data."""

from __future__ import annotations

import argparse

from keybo.data.strokes import load_strokes
from keybo.training.train import train_bigram_model, train_trigram_model


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strokes", required=True, help="Path to the bistroke/tristroke TSV")
    parser.add_argument("--ngram", choices=["bigram", "trigram"], default="bigram")
    parser.add_argument("--output", required=True, help="Where to write the model (.json)")
    parser.add_argument("--target-wpm", type=float, default=90.0)
    parser.add_argument("--wpm-threshold", type=int, default=60, help="Drop samples below this WPM")
    parser.add_argument("--min-samples", type=int, default=25, help="Min samples to keep a row")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)


def run(args: argparse.Namespace) -> int:
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

    trainer = train_bigram_model if args.ngram == "bigram" else train_trigram_model
    model = trainer(
        rows,
        target_wpm=args.target_wpm,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    model.save(args.output)
    print(f"Trained {args.ngram} model on {len(rows)} rows -> {args.output}")
    return 0
