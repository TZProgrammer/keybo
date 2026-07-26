"""Shared CLI helper: build the right IScorer from --ngram + frequency-file args.

Both `optimize` and `score` need to turn CLI arguments into a scorer, so the logic lives
here to keep them DRY and consistent.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

from keybo.data.corpus import load_frequencies
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.base import IScorer
from keybo.scoring.model_scorer import BigramModelScorer, TrigramModelScorer


def freq_path(args: argparse.Namespace, table: str) -> str:
    """One frequency-table path: the explicit flag if given, else the production corpus's.

    Resolution is LAZY (here, not at parser-construction time) for two reasons: the default
    must follow ``KEYBO_CORPUS`` at run time, and ``build_parser`` builds every subcommand's
    arguments on every invocation — so resolving eagerly would make a bad ``KEYBO_CORPUS``
    break even ``keybo --help``. Before CORPUS-SWAP-1 these defaults were repo-relative
    literals, so the default corpus silently depended on the cwd.
    """
    from keybo.data.corpus import production_corpus_dir

    flag = "bigram_freqs" if table == "bigrams.txt" else "trigram_freqs"
    explicit = getattr(args, flag, None)
    if explicit:
        return str(explicit)
    return str(production_corpus_dir(getattr(args, "corpus", None)) / table)


def add_scorer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model + ngram + frequency-file arguments shared by optimize and score."""
    from keybo.data.corpus import CORPUS_ENV_VAR, PRODUCTION_DEFAULT, known_corpora

    parser.add_argument("--model", required=True, help="Path to a saved model (.json)")
    parser.add_argument(
        "--ngram",
        choices=["bigram", "trigram"],
        default="bigram",
        help="Which objective to score with (must match the model)",
    )
    parser.add_argument(
        "--bigram-freqs",
        default=None,
        help="Bigram frequency file (bigram objective only; default: --corpus's bigrams.txt)",
    )
    parser.add_argument(
        "--trigram-freqs",
        default=None,
        help="Trigram frequency file (trigram objective only; default: --corpus's trigrams.txt)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            f"Corpus supplying the default frequency tables: {' | '.join(known_corpora())}, "
            f"or a directory (default {PRODUCTION_DEFAULT}; env: {CORPUS_ENV_VAR})"
        ),
    )
    parser.add_argument("--target-wpm", type=float, default=90.0)


def load_freqs(args: argparse.Namespace) -> dict[str, int]:
    """Load the frequency table the chosen n-gram objective consumes.

    Reads ONLY the file the objective uses (the bigram file for --ngram bigram, the trigram
    file for trigram): the trigram scorer intentionally ignores constituent bigram/skipgram
    frequencies (train/serve parity), so touching those files here would fail on a discarded
    input (e.g. running from a cwd without the repo's data/corpus/).
    """
    table = "bigrams.txt" if args.ngram == "bigram" else "trigrams.txt"
    return load_frequencies(freq_path(args, table))


def build_scorer(args: argparse.Namespace, freqs: Mapping[str, int] | None = None) -> IScorer:
    """Load the model and construct the scorer the CLI args ask for.

    Validates that the loaded model was trained for the requested n-gram order, so a bigram
    model can't silently be used as a trigram objective (or vice versa). ``freqs`` lets a
    caller supply an already-loaded (e.g. coverage-restricted) frequency table; when omitted
    the objective's own file is loaded via :func:`load_freqs`.

    Warns on stderr — without erroring — when ``--target-wpm`` falls outside the model's
    trained ``wpm_range``: the trees clamp WPM at the boundary, so an out-of-range value is
    unvalidated extrapolation, but a power user may still want it.
    """
    model = XGBoostTypingModel.load(args.model)
    if model.metadata.ngram != args.ngram:
        raise SystemExit(
            f"model was trained for ngram={model.metadata.ngram!r} but --ngram={args.ngram!r} "
            f"was requested; retrain or pass --ngram {model.metadata.ngram}"
        )

    lo, hi = model.metadata.wpm_range
    if not lo <= args.target_wpm <= hi:
        print(
            f"WARNING: --target-wpm {args.target_wpm:g} is outside the model's trained WPM "
            f"range {model.metadata.wpm_range}; predictions are unvalidated extrapolation "
            f"(trees clamp at the boundary).",
            file=sys.stderr,
        )

    if freqs is None:
        freqs = load_freqs(args)

    if args.ngram == "bigram":
        return BigramModelScorer(model, bigram_freqs=freqs, target_wpm=args.target_wpm)
    return TrigramModelScorer(model, trigram_freqs=freqs, target_wpm=args.target_wpm)
