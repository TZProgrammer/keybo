"""`keybo score` — compare named layouts on the learned objective."""

from __future__ import annotations

import argparse

from keybo.cli._scorer import add_scorer_arguments, build_scorer
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import BASELINE, NAMED_LAYOUTS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_scorer_arguments(parser)
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=sorted(NAMED_LAYOUTS),
        help="Named layouts to compare (default: all known)",
    )


def run(args: argparse.Namespace) -> int:
    scorer = build_scorer(args)

    baseline_score = scorer.fitness(Layout(NAMED_LAYOUTS[BASELINE], ROW_STAGGERED_30))
    print(f"{BASELINE} score: {baseline_score:.0f} (baseline)")

    for name in args.layouts:
        if name == BASELINE or name not in NAMED_LAYOUTS:
            continue
        score = scorer.fitness(Layout(NAMED_LAYOUTS[name], ROW_STAGGERED_30))
        improvement = (baseline_score - score) / baseline_score * 100 if baseline_score else 0.0
        print(f"{name} score: {score:.0f} (improvement over {BASELINE}: {improvement:+.2f}%)")
    return 0
