"""`keybo optimize` — search for a layout that minimizes predicted typing time."""

from __future__ import annotations

import argparse
import json

from keybo.cli._paths import ensure_writable_output
from keybo.cli._scorer import add_scorer_arguments, build_scorer, freq_path, load_freqs
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt
from keybo.scoring import model_norm as MN
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
    parser.add_argument(
        "--model-weight",
        action="append",
        metavar="GAUGE=W",
        help="Optimize the per-model NORMALIZED gauges instead of the bigram speed objective: "
        f"{', '.join(MN.GAUGE_NAMES)}, each 0 at a random-layout pool's MEAN and 1 at that "
        "model's own searched optimum (keybo.scoring.model_norm). Repeatable, e.g. "
        "--model-weight aalto-n=0.54 --model-weight comm-n=0.40 --model-weight pool-n=0.06; "
        "weights are normalized to sum to 1, so they express a PREFERENCE, not a scale. "
        "Requires --model-anchors. C30M charset only",
    )
    parser.add_argument(
        "--model-anchors",
        help="JSON anchor artifact for --model-weight (see keybo.scoring.model_norm.Anchors). "
        "It carries the frame, corpus, per-surface digests, pool seed/n and search budget the "
        "anchors were built from, and REFUSES to score if today's surfaces disagree with it — a "
        "gauge whose anchors are not reproducible is not a gauge",
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


def _parse_model_weights(specs: list[str]) -> dict[str, float]:
    """``["aalto-n=0.54", ...]`` -> ``{"AALTO": 0.54, ...}``, refusing anything ambiguous.

    Refuses an unknown gauge name and a repeated one rather than silently taking the last: a
    typo that scored zero weight on the gauge the user meant would be invisible in the output.
    """
    weights: dict[str, float] = {}
    for spec in specs:
        name, _, raw = spec.partition("=")
        name = name.strip()
        if name not in MN.POOL_OF_GAUGE:
            raise SystemExit(
                f"unknown --model-weight gauge {name!r}; expected one of "
                f"{', '.join(MN.GAUGE_NAMES)}"
            )
        if not raw:
            raise SystemExit(f"--model-weight {spec!r} has no value; expected GAUGE=W")
        pool = MN.POOL_OF_GAUGE[name]
        if pool in weights:
            raise SystemExit(f"--model-weight {name!r} given more than once")
        try:
            weights[pool] = float(raw)
        except ValueError:
            raise SystemExit(f"--model-weight {spec!r}: {raw!r} is not a number") from None
    return weights


def _build_model_blend(args: argparse.Namespace):
    """The normalized-gauge blend scorer for ``--model-weight``, with its gates run up front."""
    if not args.model_anchors:
        raise SystemExit(
            "--model-weight requires --model-anchors: the 0/1 anchors define the scale, so "
            "without them a weight has no meaning"
        )
    if not MN.S.is_c30m(args.start):
        raise SystemExit(
            f"--model-weight needs a C30M start layout (the fitted surfaces index only "
            f"{MN.S.C30M!r}); --start is {args.start!r}"
        )
    weights = _parse_model_weights(args.model_weight)
    anchors = MN.Anchors.read(args.model_anchors)
    fits = MN.SurfaceFits(corpus=getattr(args, "corpus", None))
    # Both gates BEFORE the (long) search rather than after: the direction guard catches a sign
    # or scale error, and the drift check refuses anchors that do not describe today's surfaces
    # instead of silently rescaling against them.
    anchors.assert_direction()
    if anchors.provenance.get("probe_fits"):
        anchors.assert_matches_surfaces(fits, anchors.provenance["probe_layout"])
    spec = MN.BlendSpec(weights=weights, rule="user-supplied --model-weight preference")
    print(f"objective: normalized model blend — {spec.describe()}")
    print(f"  anchors: {args.model_anchors}")
    print(f"  {MN.interpretation_note()}")
    print(f"  caveat: {MN.frame_caveat()}")
    return MN.ModelBlendScorer(anchors, spec, fits)


def run(args: argparse.Namespace) -> int:
    if args.attempts < 1:
        raise SystemExit(f"--attempts must be >= 1 (got {args.attempts})")
    if args.out:
        # Validate before the (long) search, not when writing the result at the end.
        ensure_writable_output(args.out, "--out")
    # `getattr` with a default, not `args.model_weight`: callers legitimately hand-build a
    # namespace holding only the fields their own path needs (two shipped tests construct a
    # `SimpleNamespace` for the comfort/oxey branches), and a new flag must not make those
    # callers raise AttributeError just for existing.
    if getattr(args, "model_weight", None):
        if args.comfort_weight or args.finger_load_weight or args.oxey_weight:
            raise SystemExit(
                "--model-weight replaces the speed objective with the normalized model blend, so "
                "it cannot be combined with --comfort-weight/--finger-load-weight/--oxey-weight "
                "(those are ms-equivalent terms on a different scale and would be added to a "
                "0-1 quantity)"
            )
        # One scorer for both roles: there is no faster table path for this objective, so the
        # search and the final scoring are the same evaluator (and must be, or the reported
        # fitness could differ from the one the search optimized).
        blend = _build_model_blend(args)
        return _run_search(args, blend, blend)
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
            from pathlib import Path

            from keybo.data.corpus import PRODUCTION_SKIPGRAMS, load_frequencies

            # Sibling tables come from whatever directory the bigram table came from -- the
            # user's --bigram-freqs, else the resolved production corpus (CORPUS-SWAP-1;
            # `freq_path` is what keeps this working now that the default is not a literal).
            # `PRODUCTION_SKIPGRAMS` (1-skip31.txt) IS the trigram marginalization and is the
            # table every frozen board -- and `analyze` since ALLGAUGE-1 -- is computed on.
            # This path used to hardcode `1-skip.txt`, the "different, unreproducible pass":
            # identical on blend-v1 (byte-identical files) but 4.3-4.6% apart on iWeb's
            # optimized layouts vs 0.08% on qwerty, so the search objective silently
            # disagreed with the gauge that reports on it.
            skipgram_path = Path(freq_path(args, "bigrams.txt")).with_name(PRODUCTION_SKIPGRAMS)
            skipgrams = load_frequencies(str(skipgram_path)) if skipgram_path.exists() else {}
            comfort = ComfortBigramScorer(
                freqs,
                weights=overrides,
                skipgram_freqs=skipgrams,
            )
            scorer = CompositeScorer(scorer, comfort, comfort_weight=args.comfort_weight)
        if args.finger_load_weight:
            fl = FingerLoadScorer(bigram_freqs=freqs)
            scorer = CompositeScorer(scorer, fl, comfort_weight=args.finger_load_weight)
        if args.oxey_weight:
            import os

            from keybo.data.corpus import PRODUCTION_SKIPGRAMS, load_frequencies
            from keybo.scoring.oxey import OxeyStyleScorer

            corpus_dir = os.path.dirname(freq_path(args, "bigrams.txt"))
            # `PRODUCTION_SKIPGRAMS`, not `1-skip.txt` -- see the comfort branch above.
            oxey = OxeyStyleScorer(
                freqs,
                load_frequencies(os.path.join(corpus_dir, PRODUCTION_SKIPGRAMS)),
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

    return _run_search(args, scorer, search_scorer)


def _run_search(args: argparse.Namespace, scorer, search_scorer) -> int:
    """The best-of-N loop, postflight, and result file — shared by every objective.

    Factored out so ``--model-weight`` runs the SAME search, the SAME 2-opt polish and the SAME
    Goodhart postflight as the speed objective. A second copy of this loop would be a second
    place for the two paths to drift apart.
    """
    best_layout: Layout | None = None
    best_fitness = float("inf")
    for i in range(args.attempts):
        candidate = _one_attempt(args, search_scorer, seed=args.seed + i)
        fitness = scorer.fitness(candidate)
        # The blend is a 0-1 quantity, so `.0f` (right for a ms objective) would print every
        # attempt as "-1" and hide the whole search. Format by magnitude, not by objective name.
        precision = ".6f" if abs(fitness) < 1000 else ".0f"
        print(f"attempt {i + 1}/{args.attempts}: fitness {fitness:{precision}}")
        if fitness < best_fitness:
            best_fitness = fitness
            best_layout = candidate

    assert best_layout is not None  # attempts >= 1, so the loop always ran at least once
    precision = ".6f" if abs(best_fitness) < 1000 else ".0f"
    print(f"Best fitness: {best_fitness:{precision}}")
    if isinstance(scorer, MN.ModelBlendScorer):
        # Report the gauges themselves, not just the negated blend the optimizer minimized.
        normalized = scorer.normalized(best_layout)
        print(
            "normalized gauges: "
            + "  ".join(f"{MN.GAUGE_OF_POOL[p]}={v:.6f}" for p, v in normalized.items())
        )
        print(f"blend (higher is better): {scorer.blend(best_layout):.6f}")
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
        if isinstance(scorer, MN.ModelBlendScorer):
            # The objective is not reconstructible from --ngram/--target-wpm, so it is recorded:
            # a result file that cannot name its own objective is not reproducible.
            result["objective"] = "normalized model blend"
            result["model_weights"] = dict(scorer.spec.weights)
            result["model_anchors"] = args.model_anchors
            result["normalized_gauges"] = {
                MN.GAUGE_OF_POOL[p]: v for p, v in scorer.normalized(best_layout).items()
            }
            result["blend_higher_is_better"] = scorer.blend(best_layout)
            result["frame_caveat"] = MN.frame_caveat()
            result["interpretation"] = MN.interpretation_note()
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    return 0
