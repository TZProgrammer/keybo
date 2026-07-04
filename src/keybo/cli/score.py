"""`keybo score` — compare named layouts on the learned objective.

Comparability fix (fable-audit finding 4): named layouts carry different punctuation, so the
scorer's per-layout ``has_key`` filter skips a *different* corpus subset for each — qwerty
scores its ';' bigrams, graphite its '-' bigrams. A raw "improvement over qwerty" then mixes
that coverage artifact into the geometry effect. So before scoring we restrict the corpus to
the n-grams typable on EVERY compared layout (:func:`common_ngrams`) and build ONE scorer
from that common subset, making the reported fitnesses strictly comparable.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping

from keybo.cli._scorer import add_scorer_arguments, build_scorer, load_freqs
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


def common_ngrams(freqs: Mapping[str, int], layouts: Iterable[Layout]) -> dict[str, int]:
    """Restrict ``freqs`` to the n-grams typable on EVERY layout in ``layouts``.

    An n-gram survives only if every one of its characters has a key on every layout, so the
    result is the intersection of what the compared boards can type — the subset on which
    their fitnesses are directly comparable.
    """
    layouts = list(layouts)
    return {
        ng: f for ng, f in freqs.items() if all(all(lay.has_key(c) for c in ng) for lay in layouts)
    }


def run(args: argparse.Namespace) -> int:
    # The layouts actually being compared: the baseline plus every requested named layout.
    names = [BASELINE] + [n for n in args.layouts if n != BASELINE and n in NAMED_LAYOUTS]
    layouts = {name: Layout(NAMED_LAYOUTS[name], ROW_STAGGERED_30) for name in names}

    freqs = load_freqs(args)
    common = common_ngrams(freqs, layouts.values())

    total_weight = sum(freqs.values())
    common_weight = sum(common.values())
    pct = common_weight / total_weight * 100 if total_weight else 0.0
    print(f"scoring on the {len(common)}-{args.ngram} common subset ({pct:.1f}% of corpus weight)")

    # ONE scorer, built from the common subset, used for every layout (comparability).
    scorer = build_scorer(args, freqs=common)

    baseline_score = scorer.fitness(layouts[BASELINE])
    print(f"{BASELINE} score: {baseline_score:.0f} (baseline)")

    for name in names:
        if name == BASELINE:
            continue
        score = scorer.fitness(layouts[name])
        improvement = (baseline_score - score) / baseline_score * 100 if baseline_score else 0.0
        print(f"{name} score: {score:.0f} (improvement over {BASELINE}: {improvement:+.2f}%)")
    return 0
