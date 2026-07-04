"""`keybo optimize` — search for a layout that minimizes predicted typing time."""

from __future__ import annotations

import argparse
import json

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
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Run the search N times with seeds seed..seed+N-1 and keep the best "
        "(guards against a single unlucky local minimum)",
    )
    parser.add_argument(
        "--out",
        help="Write the best result to this path as JSON (layout, fitness, and run config)",
    )


def _one_attempt(args: argparse.Namespace, scorer, seed: int) -> Layout:
    """Run a single SA (+ optional 2-opt polish) from a fresh starting layout."""
    # A fresh Layout per attempt: SA mutates the layout it searches, so reusing one would
    # start later attempts from a different (mutated) board and break seed determinism.
    layout = Layout(args.start, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=seed, alpha=args.alpha, max_outer=args.max_outer)
    best = sa.optimize(layout, scorer)
    if not args.no_local_search:
        best = two_opt(best, scorer)
    return best


def run(args: argparse.Namespace) -> int:
    if args.attempts < 1:
        raise SystemExit(f"--attempts must be >= 1 (got {args.attempts})")
    scorer = build_scorer(args)

    best_layout: Layout | None = None
    best_fitness = float("inf")
    for i in range(args.attempts):
        candidate = _one_attempt(args, scorer, seed=args.seed + i)
        fitness = scorer.fitness(candidate)
        print(f"attempt {i + 1}/{args.attempts}: fitness {fitness:.0f}")
        if fitness < best_fitness:
            best_fitness = fitness
            best_layout = candidate

    assert best_layout is not None  # attempts >= 1, so the loop always ran at least once
    print(f"Best fitness: {best_fitness:.0f}")
    print(best_layout.render())

    if args.out:
        result = {
            "layout": "".join(best_layout.chars),
            "fitness": best_fitness,
            "ngram": args.ngram,
            "target_wpm": args.target_wpm,
            "seed": args.seed,
            "attempts": args.attempts,
            "model": args.model,
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    return 0
