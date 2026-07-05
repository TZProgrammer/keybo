"""`keybo optimize` — search for a layout that minimizes predicted typing time."""

from __future__ import annotations

import argparse
import json

from keybo.cli._paths import ensure_writable_output
from keybo.cli._scorer import add_scorer_arguments, build_scorer, load_freqs
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt
from keybo.scoring.inspect import layout_diagnostics


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
    parser.add_argument(
        "--comfort-weight",
        type=float,
        default=0.0,
        help="Add comfort_weight * comfort-penalty (ms-equivalents; see keybo.scoring."
        "comfort DEFAULT_COMFORT — documented PREFERENCES, not measurements) to the "
        "measured speed objective. 0 = pure speed. Bigram objective only.",
    )
    parser.add_argument(
        "--comfort-config",
        help="JSON file overriding individual comfort weights by name",
    )
    parser.add_argument(
        "--oxey-weight",
        type=float,
        default=0.0,
        help="Add oxey_weight * community-heuristic pattern score (oxeylyzer-style "
        "sfb/dsfb/roll/redirect judgment — keybo.scoring.oxey; a documented PREFERENCE "
        "approximation, incl. patterns our data measured time-neutral). 0 = off. "
        "Bigram objective only; loads the skipgram+trigram corpora beside --bigram-freqs.",
    )
    parser.add_argument(
        "--finger-load-weight",
        type=float,
        default=0.0,
        help="Add finger-utilization balancing (sum of load^2/capacity; the semimak "
        "principle as an explicit comfort term — see keybo.scoring.comfort."
        "FingerLoadScorer; PREFERENCE weights, measured to have no speed mechanism). "
        "0 = off. Bigram objective only.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Disable the QAP-table fast path (bigram objective only) and score through "
        "the model on every evaluation — ~1000x slower, same objective",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def _one_attempt(args: argparse.Namespace, scorer, seed: int) -> Layout:
    """Run a single SA (+ optional 2-opt polish) from a fresh starting layout."""
    # A fresh Layout per attempt: SA mutates the layout it searches, so reusing one would
    # start later attempts from a different (mutated) board and break seed determinism.
    layout = Layout(args.start, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(
        seed=seed, alpha=args.alpha, max_outer=args.max_outer, progress=not args.no_progress
    )
    best = sa.optimize(layout, scorer)
    if not args.no_local_search:
        best = two_opt(best, scorer)
    return best


def run(args: argparse.Namespace) -> int:
    if args.attempts < 1:
        raise SystemExit(f"--attempts must be >= 1 (got {args.attempts})")
    if args.out:
        # Validate before the (long) search, not when writing the result at the end.
        ensure_writable_output(args.out, "--out")
    scorer = build_scorer(args)
    if args.comfort_weight or args.finger_load_weight or args.oxey_weight:
        if args.ngram != "bigram":
            raise SystemExit(
                "--comfort-weight/--finger-load-weight currently support the bigram objective only"
            )
        from keybo.scoring.comfort import (
            ComfortBigramScorer,
            CompositeScorer,
            FingerLoadScorer,
        )

        overrides = {}
        if args.comfort_config:
            with open(args.comfort_config, encoding="utf-8") as f:
                overrides = json.load(f)
        freqs = load_freqs(args)
        if args.comfort_weight:
            comfort = ComfortBigramScorer(freqs, weights=overrides)
            scorer = CompositeScorer(scorer, comfort, comfort_weight=args.comfort_weight)
        if args.finger_load_weight:
            fl = FingerLoadScorer(bigram_freqs=freqs)
            scorer = CompositeScorer(scorer, fl, comfort_weight=args.finger_load_weight)
        if args.oxey_weight:
            import os

            from keybo.data.corpus import load_frequencies
            from keybo.scoring.oxey import OxeyStyleScorer

            corpus_dir = os.path.dirname(args.bigram_freqs)
            oxey = OxeyStyleScorer(
                freqs,
                load_frequencies(os.path.join(corpus_dir, "1-skip.txt")),
                load_frequencies(os.path.join(corpus_dir, "trigrams.txt")),
            )
            scorer = CompositeScorer(scorer, oxey, comfort_weight=args.oxey_weight)
    search_scorer = scorer
    if (
        args.ngram == "bigram"
        and not args.no_table
        and not args.comfort_weight
        and not args.finger_load_weight
        and not args.oxey_weight
    ):
        # Exact same objective, ~1000x faster per evaluation (parity-tested). The search
        # explores permutations of --start's charset, which is what the table fixes.
        from keybo.models.xgboost_model import XGBoostTypingModel
        from keybo.scoring.table_scorer import TableBigramScorer

        model = XGBoostTypingModel.load(args.model)
        search_scorer = TableBigramScorer(
            model, load_freqs(args), target_wpm=args.target_wpm, chars=args.start
        )

    best_layout: Layout | None = None
    best_fitness = float("inf")
    for i in range(args.attempts):
        candidate = _one_attempt(args, search_scorer, seed=args.seed + i)
        fitness = scorer.fitness(candidate)
        print(f"attempt {i + 1}/{args.attempts}: fitness {fitness:.0f}")
        if fitness < best_fitness:
            best_fitness = fitness
            best_layout = candidate

    assert best_layout is not None  # attempts >= 1, so the loop always ran at least once
    print(f"Best fitness: {best_fitness:.0f}")
    print(best_layout.render())

    # Auto-E5 structural postflight (Goodhart gate): every search ends with the numbers
    # that catch a degenerate optimum — see agent-artifacts/goodhart-row-blindness.md.
    diag = layout_diagnostics(best_layout, load_freqs(args) if args.ngram == "bigram" else {})
    if diag["row_share"]["home"] or diag["sfb_share"] or diag["finger_load"]:
        loads = {k: v for k, v in diag["finger_load"].items() if k != "thumb"}
        max_f = max(loads, key=loads.get) if loads else "n/a"
        print(
            f"structure: home-row share {diag['row_share']['home']:.1%} | "
            f"sfb share {diag['sfb_share']:.2%} | "
            f"max finger load {max_f} {loads.get(max_f, 0):.1%}"
        )

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
