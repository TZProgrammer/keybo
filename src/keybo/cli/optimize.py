"""`keybo optimize` — search for a layout that minimizes predicted typing time."""

from __future__ import annotations

import argparse

from keybo.cli._scorer import add_scorer_arguments, build_scorer
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_scorer_arguments(parser)
    parser.add_argument(
        "--start", default=NAMED_LAYOUTS["qwerty"], help="Starting layout (30 chars)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    parser.add_argument("--alpha", type=float, default=0.999, help="Geometric cooling rate")
    parser.add_argument("--max-outer", type=int, default=None, help="Cap on cooling iterations")
    parser.add_argument("--no-local-search", action="store_true", help="Skip the 2-opt polish")


def run(args: argparse.Namespace) -> int:
    scorer = build_scorer(args)

    layout = Layout(args.start, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=args.seed, alpha=args.alpha, max_outer=args.max_outer)
    best = sa.optimize(layout, scorer)
    if not args.no_local_search:
        best = two_opt(best, scorer)

    print(f"Best fitness: {scorer.fitness(best):.0f}")
    print(best.render())
    return 0
