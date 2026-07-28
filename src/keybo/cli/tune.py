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
    parser.add_argument("--wpm-threshold", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective",
        choices=["lolo", "cv-mae"],
        default="lolo",
        help="lolo (default): score candidates by leave-one-layout-out TRANSFER "
        "(rho/ceiling gated on ranking tau) — what the optimizer needs. cv-mae: the "
        "legacy pooled-CV fit objective (rewards memorization; kept for comparison).",
    )
    parser.add_argument(
        "--lolo-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Training seeds per candidate for --objective lolo",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=None,
        help="Smallest RELATIVE margin a lolo selection must clear to be reported as a "
        "winner (default: keybo.training.tune.LOLO_MIN_MARGIN, derived from the ceiling "
        "reweighting bound). 0 disables the check.",
    )
    parser.add_argument(
        "--allow-unresolvable-margin",
        action="store_true",
        help="Warn instead of refusing when the top two candidates are closer than "
        "--min-margin. Off by default: a champion chosen inside the resolvable margin is "
        "indistinguishable from a real one in the output.",
    )
    parser.add_argument(
        "--allow-unevaluated-objective",
        action="store_true",
        help="Proceed even if NO fold yields a finite rho/ceiling, in which case every "
        "candidate ties at -inf and the tau gate alone picks the winner. Off by default: "
        "the resulting params file is indistinguishable from a real one, so the default is "
        "to refuse. Use only when you want the tau-gated result knowingly.",
    )


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

    if args.objective == "lolo":
        import numpy as np

        from keybo.training.tune import (
            LOLO_MIN_MARGIN,
            ObjectiveNotEvaluated,
            tune_lolo,
        )
        from keybo.verdicts import MarginTooSmall

        rng = np.random.default_rng(args.seed)
        candidates = [
            {
                "n_estimators": int(rng.integers(100, 500)),
                "max_depth": int(rng.integers(2, 6)),
                "learning_rate": float(rng.uniform(0.02, 0.2)),
                "min_child_weight": int(rng.integers(1, 6)),
                "subsample": float(rng.uniform(0.6, 1.0)),
            }
            for _ in range(args.n_iter)
        ]
        try:
            best, leaderboard = tune_lolo(
                rows,
                candidates=candidates,
                seeds=args.lolo_seeds,
                ngram=args.ngram,
                allow_unevaluated_objective=args.allow_unevaluated_objective,
                min_margin=(LOLO_MIN_MARGIN if args.min_margin is None else args.min_margin),
                allow_unresolvable_margin=args.allow_unresolvable_margin,
            )
        except (ObjectiveNotEvaluated, MarginTooSmall) as exc:
            # Refuse BEFORE writing --output: a params file from an unevaluated objective is
            # indistinguishable from a real one once on disk, and this command's own final
            # line calls it "Best hyperparameters".
            raise SystemExit(f"keybo tune --objective lolo: {exc}") from exc
        for params, score in leaderboard[:5]:
            print(f"  rho/ceiling {score:+.4f}  {params}")
    else:
        X, y = build_training_matrix(rows, ngram=args.ngram, target_wpm=args.target_wpm)
        best = tune_hyperparameters(X, y, n_iter=args.n_iter, cv=args.cv, seed=args.seed)
    with open(args.output, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best hyperparameters -> {args.output}: {best}")
    return 0
