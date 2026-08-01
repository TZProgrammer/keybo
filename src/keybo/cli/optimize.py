"""`keybo optimize` — search for a layout that minimizes predicted typing time."""

from __future__ import annotations

import argparse
import json

from keybo.analysis.timecard import GaugeTrigramScorer
from keybo.cli._paths import ensure_writable_output
from keybo.cli._scorer import add_scorer_arguments, build_scorer, freq_path, load_freqs
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import three_opt, two_opt
from keybo.scoring import model_norm as MN
from keybo.scoring.inspect import layout_diagnostics

#: Parity tolerance for ``--gauge-objective``: how far the search objective may sit from
#: ``analyze``'s own ms/char before the run is refused. The table path reconciles to ~1.2e-14
#: (float noise over a 10^11-magnitude sum), so 1e-12 is ~100x of headroom while still
#: rejecting the plausible-but-wrong constructions of this objective by 10 orders of magnitude.
_GAUGE_PARITY_TOLERANCE = 1e-12


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
        "--three-opt",
        action="store_true",
        help="Extend the local-search polish from 2-opt (swap pairs) to 3-opt (reorder "
        "triples, keybo.optimize.local_search.three_opt). Off by default because it is "
        "~4-6x the evaluations; a 2-opt optimum is NOT a 3-opt optimum (measured: the "
        "campaign incumbents graphite/semimak each admit a 3-opt-only improvement of "
        # `%%`, not `%`: argparse runs every help string through `% params`, so a bare
        # percent sign raises ValueError and takes out `optimize --help` entirely -- a
        # crash in the one invocation that carries no other output to notice it by.
        "-0.27%% ms/char, above the 0.135 model-seed floor). Incompatible with "
        "--no-local-search",
    )
    parser.add_argument(
        "--polish-incumbent",
        action="append",
        metavar="LAYOUT",
        default=None,
        help="Apply the SAME polish the searched layout gets to this incumbent (name or "
        "30-char string) and report both, so the comparison is symmetric. Repeatable. "
        "Without this, a searched layout gets SA + local search while an incumbent is "
        "scored as-is, and the reported gap includes the polish the incumbent never got "
        "(measured: 71-91%% of the gap vs the campaign's arm B is polish, not layout). "
        "Requires the incumbent to be a permutation of --start's charset",
    )
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
    parser.add_argument(
        "--gauge-objective",
        action="store_true",
        help="Search the REPORTED gauge (`analyze`'s ms/char: the K31 seed-averaged "
        "T2 + Tcond surface over the trigram corpus) instead of the default bigram objective. "
        "The default objective and this gauge rank layouts INVERTED (spearman 0.672; the "
        "selection tax at the argmin is 4.97 resolution floors) because the cubic term carries "
        "most of the gauge's variance, so no restart budget closes the gap — see "
        "SEARCHPARAMS-1 / NORMOPT-1. Opt-in: it does NOT change the default. Parity-gated "
        "against `analyze` before the search starts. C30M charset only; cannot be combined "
        "with --no-table, --ngram trigram, --model-weight or the comfort/finger-load/oxey terms",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def _polish(args: argparse.Namespace, layout: Layout, scorer) -> Layout:
    """THE polish, in one place — so a searched layout and an incumbent get the same one.

    Factored out (rather than inlined in :func:`_one_attempt`) precisely because the defect
    A6 names is an ASYMMETRIC comparison: a searched layout got SA + 2-opt while an incumbent
    it was reported against got scored as-is. A second copy of these two lines is a second
    place for the two roles to diverge, which is how the asymmetry arose in the first place.

    ``--three-opt`` runs 3-opt AFTER 2-opt rather than instead of it: both reach the same 3-opt
    optimum from these boards, but the cheap moves first is measurably fewer evaluations on 5
    of 7 campaign boards (e.g. semimak 56,686 vs 95,325), because every improving swap 2-opt
    takes is one the 3-opt scan would have re-derived over C(n,3) triples instead of C(n,2)
    pairs.
    """
    if args.no_local_search:
        return layout
    polished = two_opt(layout, scorer)
    if args.three_opt:
        polished = three_opt(polished, scorer)
    return polished


def _one_attempt(args: argparse.Namespace, scorer, seed: int) -> Layout:
    """Run a single SA (+ optional local-search polish) from a fresh starting layout."""
    # A fresh Layout per attempt: SA mutates the layout it searches, so reusing one would
    # start later attempts from a different (mutated) board and break seed determinism.
    layout = Layout(args.start, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(
        seed=seed, alpha=args.alpha, max_outer=args.max_outer, progress=not args.no_progress
    )
    best = sa.optimize(layout, scorer)
    return _polish(args, best, scorer)


def _resolve_incumbents(specs: list[str], start: str) -> list[tuple[str, str]]:
    """``["graphite", "<30 chars>"] -> [(label, layout), ...]``, refusing anything unscorable.

    REFUSES rather than skipping, and does so here — before the (long) search — for the same
    reason ``--model-weight``'s two gates run up front: a comparison silently missing the arm
    the user asked for reads exactly like one where that arm lost.

    A non-permutation of ``--start``'s charset is refused rather than scored, because the
    reported quantity is a corpus-restricted mean: a board with a different charset covers
    different corpus rows, so its number would be a different denominator's mean printed in the
    same column. That is the cross-charset comparison ``analyze`` renders as N/A.
    """
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in specs:
        layout = NAMED_LAYOUTS.get(spec, spec)
        label = spec if spec in NAMED_LAYOUTS else "(literal)"
        if len(layout) != len(start) or sorted(layout) != sorted(start):
            raise SystemExit(
                f"--polish-incumbent {spec!r} is not a permutation of --start's charset, so it "
                f"cannot be polished or compared on the same objective: the search explores "
                f"permutations of --start (the table fixes that charset), and a different "
                f"charset covers different corpus n-grams — its score would be a different "
                f"denominator's mean printed in the same column. Give an incumbent over "
                f"{''.join(sorted(start))!r}, or run a separate search from it"
            )
        if layout in seen:
            raise SystemExit(
                f"--polish-incumbent {spec!r} given more than once (as {spec!r} and by "
                f"another name); each incumbent is polished once"
            )
        seen.add(layout)
        resolved.append((label, layout))
    return resolved


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


def _build_gauge_objective(args: argparse.Namespace):
    """The reported-gauge search objective for ``--gauge-objective``, gates run up front.

    Every gate REFUSES rather than falling back to the default objective: a run that silently
    searched the bigram table while its output was labelled with the gauge is exactly the
    ``present != effective`` defect this flag exists to fix (the shipped ``--compiler-opt-level``
    no-op, the ``1-skip.txt`` divergence, and ``target_space='lograt'`` are the same shape).
    """
    from keybo.analysis import surfaces as SF
    from keybo.analysis.timecard import gauge_search_scorer

    if args.no_table:
        raise SystemExit(
            "--gauge-objective cannot be combined with --no-table: the gauge's no-table path is "
            "the analyzer's own ~50 ms-per-layout loop, which a multi-million-evaluation search "
            "cannot use. The table path is exact (parity-gated below), not an approximation"
        )
    if args.ngram != "bigram":
        raise SystemExit(
            f"--gauge-objective replaces the objective with the reported gauge, so --ngram "
            f"{args.ngram!r} is redundant and ambiguous: the gauge already contains both the "
            f"bigram and the trigram term. Drop --ngram (it selects the DEFAULT objective's "
            f"order, and the gauge is neither)"
        )
    if args.comfort_weight or args.finger_load_weight or args.oxey_weight:
        raise SystemExit(
            "--gauge-objective cannot be combined with --comfort-weight/--finger-load-weight/"
            "--oxey-weight: those terms are added to the DEFAULT bigram objective's scale, and "
            "adding them here would mean the search no longer optimizes the published gauge — "
            "which is the one property this flag provides"
        )
    if getattr(args, "model_weight", None):
        raise SystemExit(
            "--gauge-objective and --model-weight are two different replacement objectives "
            "(the measured ms/char gauge vs the normalized model blend); pass exactly one"
        )
    if not SF.is_c30m(args.start):
        raise SystemExit(
            f"--gauge-objective needs a C30M start layout (the K31 surface's charset is "
            f"{SF.C30M!r}, and which corpus rows the objective keeps depends on it, so another "
            f"charset's total is not comparable to any published ms/char); --start is "
            f"{args.start!r}"
        )

    scorer = gauge_search_scorer(
        chars=args.start, target_wpm=args.target_wpm, corpus=getattr(args, "corpus", None)
    )
    # PARITY GATE, before the (long) search rather than after it — the same order
    # `--model-weight` runs its anchor gates in. An objective that cannot be tied to the
    # published gauge must not consume an hour of search first: the naive way to build this
    # objective is 1.5e-2 off (~11 resolution floors) and looks entirely plausible.
    deviation = scorer.parity_rel_dev(Layout(args.start, ROW_STAGGERED_30))
    if deviation > _GAUGE_PARITY_TOLERANCE:
        raise SystemExit(
            f"--gauge-objective failed its parity gate: the search objective deviates from "
            f"`analyze`'s ms/char by rel {deviation:.3e} on --start, above the "
            f"{_GAUGE_PARITY_TOLERANCE:.0e} tolerance. Refusing to search an objective that is "
            f"not the gauge it will be reported on"
        )
    print("objective: the REPORTED gauge — analyze's ms/char (K31 T2+Tcond, seed-averaged)")
    print(
        f"  parity vs analyze on --start: rel dev {deviation:.3e} (tolerance "
        f"{_GAUGE_PARITY_TOLERANCE:.0e})"
    )
    print(
        "  NOTE this is NOT the default objective; the two rank layouts inverted (SEARCHPARAMS-1)"
    )
    return scorer


def run(args: argparse.Namespace) -> int:
    if args.attempts < 1:
        raise SystemExit(f"--attempts must be >= 1 (got {args.attempts})")
    if args.out:
        # Validate before the (long) search, not when writing the result at the end.
        ensure_writable_output(args.out, "--out")
    # `getattr` with a default, not `args.three_opt` / `args.polish_incumbent`: callers
    # legitimately hand-build a namespace holding only the fields their own path needs (shipped
    # tests construct a `SimpleNamespace` for the comfort/oxey branches), and a new flag must
    # not make those callers raise AttributeError just for existing. Normalized ONCE here so
    # every path below — including `_polish`, which both roles call — reads a real attribute.
    args.three_opt = bool(getattr(args, "three_opt", False))
    args.polish_incumbent = list(getattr(args, "polish_incumbent", None) or [])
    # BOTH gates before the (long) search, and REFUSING rather than falling back: an
    # unsupported combination that silently degrades to the default polish produces a number
    # labelled "3-opt" that no 3-opt ever touched.
    if args.three_opt and args.no_local_search:
        raise SystemExit(
            "--three-opt extends the local-search polish, but --no-local-search disables it; "
            "they cannot be combined. Drop one — passing both would report a '3-opt' result "
            "that ran no local search at all"
        )
    incumbents = (
        _resolve_incumbents(args.polish_incumbent, args.start) if (args.polish_incumbent) else []
    )
    if incumbents and args.no_local_search:
        raise SystemExit(
            "--polish-incumbent applies the searched layout's polish to an incumbent, but "
            "--no-local-search means there is no polish to apply; the comparison would be "
            "as-is vs as-is. Drop --no-local-search, or score the incumbent with `keybo score`"
        )
    # `getattr` with a default, not `args.model_weight`: callers legitimately hand-build a
    # namespace holding only the fields their own path needs (two shipped tests construct a
    # `SimpleNamespace` for the comfort/oxey branches), and a new flag must not make those
    # callers raise AttributeError just for existing.
    if getattr(args, "gauge_objective", False):
        # One scorer for both roles, as `--model-weight` does: best-of-N must SELECT on the
        # gauge too. Searching the gauge but ranking attempts by the bigram objective would
        # reintroduce the 4.97-floor selection tax the flag exists to remove.
        gauge = _build_gauge_objective(args)
        return _run_search(args, gauge, gauge)
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
        return _run_search(args, blend, blend, incumbents)
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

    return _run_search(args, scorer, search_scorer, incumbents)


def _run_search(
    args: argparse.Namespace,
    scorer,
    search_scorer,
    incumbents: list[tuple[str, str]] | None = None,
) -> int:
    """The best-of-N loop, postflight, and result file — shared by every objective.

    Factored out so ``--model-weight`` runs the SAME search, the SAME local-search polish and
    the SAME Goodhart postflight as the speed objective. A second copy of this loop would be a
    second place for the two paths to drift apart.
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
    if isinstance(scorer, GaugeTrigramScorer):
        # The gauge's fitness is a corpus TOTAL (~2e11 ms); every published number is the
        # normalized ms/char, so print it here rather than leaving the reader to divide by a
        # coverage-dependent denominator they would have to look up.
        print(f"ms/char (analyze's gauge): {scorer.ms_per_char(best_layout):.6f}")
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

    incumbent_rows = _report_incumbents(args, scorer, search_scorer, incumbents, best_fitness)

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
        if isinstance(scorer, GaugeTrigramScorer):
            # `"ngram": "bigram"` is recorded for every run, so without this a gauge result
            # file would be indistinguishable from a default one — the same
            # unreproducible-artifact failure the blend branch above guards against. The parity
            # deviation travels with the result so a reader can see the objective was TIED to
            # the gauge at run time, not merely labelled with its name.
            result["objective"] = "reported gauge — analyze's ms/char (K31 T2+Tcond, seed-mean)"
            result["ms_per_char"] = scorer.ms_per_char(best_layout)
            result["gauge_parity_rel_dev"] = scorer.parity_rel_dev(best_layout)
            result["gauge_parity_tolerance"] = _GAUGE_PARITY_TOLERANCE
            result["corpus"] = getattr(args, "corpus", None)
        if args.three_opt or incumbent_rows:
            # Recorded ONLY when a new flag was passed, so the default artifact keeps exactly the
            # key set `tests/cli/test_optimize_attempts.py` pins (that test asserts the whole set
            # with `==`, and rightly: a result file that silently grows keys is one a consumer
            # cannot validate). A default run's polish is already implied by --no-local-search's
            # absence; a NON-default one is not, which is why it is named here.
            #
            # This is also where an unconditional `result["polish"] = ...` first draft was caught:
            # adding a key to the default artifact IS a default-behaviour change, however small.
            result["polish"] = _polish_label(args)
        if incumbent_rows:
            # A result file carrying only the searched layout's fitness cannot distinguish "beat a
            # polished incumbent" from "beat an unpolished one" — the asymmetry this flag removes.
            result["incumbents"] = incumbent_rows
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    return 0


def _polish_label(args: argparse.Namespace) -> str:
    """Name the polish that actually ran — the label every reported number is scoped to."""
    if args.no_local_search:
        return "none"
    return "2-opt+3-opt" if args.three_opt else "2-opt"


def _report_incumbents(
    args: argparse.Namespace,
    scorer,
    search_scorer,
    incumbents: list[tuple[str, str]] | None,
    best_fitness: float,
) -> list[dict]:
    """Score each incumbent AS-IS and AFTER the searched layout's own polish, and print both.

    The three columns are the point. ``as-is`` is what the campaign reported (and what the gap
    was computed against); ``polished`` is the symmetric number; ``polish`` is the difference
    between them — i.e. exactly how much of the reported gap was the polish the incumbent never
    got rather than the layout. Measured on the campaign gauge, that term is 71-91% of the gap
    against arm B, so printing the as-is column alone is what made the comparison misread.

    The incumbent is polished with ``search_scorer`` (the fast table) and reported with
    ``scorer``, the same split the searched layout gets — otherwise "the same polish" would be
    the same NAME for a different objective.
    """
    if not incumbents:
        return []
    rows: list[dict] = []
    print(
        f"\n== incumbents, polished the SAME way as the searched layout ({_polish_label(args)}) =="
    )
    print(f"{'incumbent':<14}{'as-is':>14}{'polished':>14}{'polish':>12}{'gap vs best':>14}")
    for label, layout_str in incumbents:
        as_is = scorer.fitness(Layout(layout_str, ROW_STAGGERED_30))
        polished_layout = _polish(args, Layout(layout_str, ROW_STAGGERED_30), search_scorer)
        polished = scorer.fitness(polished_layout)
        precision = ".6f" if abs(as_is) < 1000 else ".0f"
        shown = label if label != "(literal)" else layout_str
        print(
            f"{shown[:13]:<14}{as_is:>14{precision}}{polished:>14{precision}}"
            f"{polished - as_is:>+12{precision}}{polished - best_fitness:>+14{precision}}"
        )
        rows.append(
            {
                "incumbent": label,
                "layout": layout_str,
                "polished_layout": "".join(polished_layout.chars),
                "fitness_as_is": as_is,
                "fitness_polished": polished,
                "polish_gain": polished - as_is,
                "gap_vs_best_polished": polished - best_fitness,
            }
        )
    print(
        "`polish` is how much of an as-is gap was the polish the incumbent never got, not the "
        "layout — compare `gap vs best` (both polished), never `as-is` against a searched result"
    )
    return rows
