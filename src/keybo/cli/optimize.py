"""`keybo optimize` — search for a layout that minimizes predicted typing time."""

from __future__ import annotations

import argparse

from keybo.data.corpus import load_frequencies
from keybo.geometry import ROW_STAGGERED_30
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt
from keybo.scoring.model_scorer import BigramModelScorer


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to a saved bigram model (.json)")
    parser.add_argument("--bigram-freqs", required=True, help="Path to the bigram frequency file")
    parser.add_argument(
        "--start", default=NAMED_LAYOUTS["qwerty"], help="Starting layout (30 chars)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    parser.add_argument("--alpha", type=float, default=0.999, help="Geometric cooling rate")
    parser.add_argument("--target-wpm", type=float, default=90.0)
    parser.add_argument("--max-outer", type=int, default=None, help="Cap on cooling iterations")
    parser.add_argument("--no-local-search", action="store_true", help="Skip the 2-opt polish")


def run(args: argparse.Namespace) -> int:
    model = XGBoostTypingModel.load(args.model)
    freqs = load_frequencies(args.bigram_freqs)
    scorer = BigramModelScorer(model, bigram_freqs=freqs, target_wpm=args.target_wpm)

    from keybo.layout import Layout

    layout = Layout(args.start, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=args.seed, alpha=args.alpha, max_outer=args.max_outer)
    best = sa.optimize(layout, scorer)
    if not args.no_local_search:
        best = two_opt(best, scorer)

    print(f"Best fitness: {scorer.fitness(best):.0f}")
    print(best.render())
    return 0
