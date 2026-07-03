"""Shared CLI helper: build the right IScorer from --ngram + frequency-file args.

Both `optimize` and `score` need to turn CLI arguments into a scorer, so the logic lives
here to keep them DRY and consistent.
"""

from __future__ import annotations

import argparse

from keybo.data.corpus import load_frequencies
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.base import IScorer
from keybo.scoring.model_scorer import BigramModelScorer, TrigramModelScorer

_DEFAULT_BIGRAMS = "data/corpus/bigrams.txt"
_DEFAULT_TRIGRAMS = "data/corpus/trigrams.txt"


def add_scorer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model + ngram + frequency-file arguments shared by optimize and score."""
    parser.add_argument("--model", required=True, help="Path to a saved model (.json)")
    parser.add_argument(
        "--ngram",
        choices=["bigram", "trigram"],
        default="bigram",
        help="Which objective to score with (must match the model)",
    )
    parser.add_argument(
        "--bigram-freqs",
        default=_DEFAULT_BIGRAMS,
        help="Bigram frequency file (bigram objective only)",
    )
    parser.add_argument(
        "--trigram-freqs",
        default=_DEFAULT_TRIGRAMS,
        help="Trigram frequency file (trigram objective only)",
    )
    parser.add_argument("--target-wpm", type=float, default=90.0)


def build_scorer(args: argparse.Namespace) -> IScorer:
    """Load the model and construct the scorer the CLI args ask for.

    Validates that the loaded model was trained for the requested n-gram order, so a bigram
    model can't silently be used as a trigram objective (or vice versa). Loads ONLY the
    frequency file the chosen objective consumes: the trigram scorer intentionally ignores
    constituent bigram/skipgram frequencies (train/serve parity), so requiring — or even
    reading — those files here would fail on a discarded input (e.g. running from a cwd
    without the repo's data/corpus/).
    """
    model = XGBoostTypingModel.load(args.model)
    if model.metadata.ngram != args.ngram:
        raise SystemExit(
            f"model was trained for ngram={model.metadata.ngram!r} but --ngram={args.ngram!r} "
            f"was requested; retrain or pass --ngram {model.metadata.ngram}"
        )

    if args.ngram == "bigram":
        return BigramModelScorer(
            model, bigram_freqs=load_frequencies(args.bigram_freqs), target_wpm=args.target_wpm
        )
    return TrigramModelScorer(
        model,
        trigram_freqs=load_frequencies(args.trigram_freqs),
        target_wpm=args.target_wpm,
    )
