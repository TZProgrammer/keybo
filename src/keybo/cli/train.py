"""`keybo train` — fit a typing-time model from stroke data."""

from __future__ import annotations

import argparse
import json
import os

from keybo.cli._paths import ensure_writable_output
from keybo.data.strokes import load_strokes
from keybo.training.train import train_bigram_model, train_trigram_model

# Fallback estimator params when neither a --hyperparams file nor an explicit flag supplies
# them. These mirror the historical CLI defaults (and XGBoostTypingModel's own defaults).
_DEFAULT_N_ESTIMATORS = 300
_DEFAULT_MAX_DEPTH = 5


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strokes", required=True, help="Path to the bistroke/tristroke TSV")
    parser.add_argument("--ngram", choices=["bigram", "trigram"], default="bigram")
    parser.add_argument("--output", required=True, help="Where to write the model (.json)")
    parser.add_argument("--target-wpm", type=float, default=90.0)
    # Defaults 0/1 match the harness-VALIDATED recipe (every LOLO result was measured
    # at these values; the old 60/25 silently trained a different model than validated).
    parser.add_argument("--wpm-threshold", type=int, default=0, help="Drop samples below this WPM")
    parser.add_argument("--min-samples", type=int, default=1, help="Min samples to keep a row")
    parser.add_argument(
        "--hyperparams",
        help="Path to a JSON dict of XGBoost params (e.g. from `keybo tune`). "
        "Explicit --n-estimators/--max-depth override the file's values.",
    )
    # default=None so we can tell an explicitly-passed flag from an unset one and apply the
    # right precedence (explicit flag > --hyperparams file > built-in default).
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def _resolve_params(args: argparse.Namespace) -> dict:
    """Merge the --hyperparams file with explicit flags per the precedence rule.

    Precedence (highest first): an explicitly-passed --n-estimators/--max-depth flag, then
    the --hyperparams JSON file, then the built-in defaults. Any *other* params in the JSON
    (e.g. learning_rate, subsample from `keybo tune`) flow straight through to the trainer.
    """
    params: dict = {}
    if args.hyperparams:
        with open(args.hyperparams) as f:
            params.update(json.load(f))

    # Explicit flags win over the file; otherwise keep the file's value, else the default.
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    else:
        params.setdefault("n_estimators", _DEFAULT_N_ESTIMATORS)
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    else:
        params.setdefault("max_depth", _DEFAULT_MAX_DEPTH)

    return params


def run(args: argparse.Namespace) -> int:
    # Fail fast on anything that would kill the run AFTER the expensive stages: an
    # uncreatable output dir (XGBoost's C++ writer error is opaque and arrives hours in)
    # or a missing --hyperparams file.
    ensure_writable_output(args.output, "--output")
    if args.hyperparams and not os.path.exists(args.hyperparams):
        raise SystemExit(f"--hyperparams file not found: {args.hyperparams}")
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

    params = _resolve_params(args)
    trainer = train_bigram_model if args.ngram == "bigram" else train_trigram_model
    # progress is an explicit kwarg, deliberately NOT merged into params: params is recorded
    # as hyperparameter provenance and forwarded to XGBoost, where a stray key would be
    # silently ignored.
    model = trainer(rows, target_wpm=args.target_wpm, progress=not args.no_progress, **params)

    # Record the resolved hyperparameters actually used, for provenance. ModelMetadata is a
    # frozen dataclass, but `extra` is a mutable dict field, so mutating it in place is fine
    # (only attribute *reassignment* is blocked by frozen).
    model.metadata.extra["hyperparams"] = params

    model.save(args.output)
    print(f"Trained {args.ngram} model on {len(rows)} rows -> {args.output}")
    return 0
