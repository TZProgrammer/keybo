"""`keybo shap-diff` — per-feature attribution of the ms/char gap between two layouts."""

from __future__ import annotations

import argparse
import json

from keybo.analysis.shap_diff import format_report as format_shap_diff
from keybo.analysis.shap_diff import shap_diff
from keybo.cli._paths import ensure_writable_output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("layout_a", help="Baseline layout: a registry name or a 30-char string")
    parser.add_argument("layout_b", help="Comparison layout: a registry name or a 30-char string")
    parser.add_argument(
        "--corpus",
        help="Corpus name or directory for the frequency weighting (default: the production "
        "corpus; '--corpus iweb' reproduces the campaign's frozen boards)",
    )
    parser.add_argument("--target-wpm", type=float, default=90.0, help="Scoring WPM")
    parser.add_argument(
        "--channel",
        choices=["t2", "tcond", "both"],
        default="both",
        help="Which term of the gauge to decompose: 't2' is the bigram table, 'tcond' the "
        "conditioned-trigram increment, 'both' (default) is the only setting whose decomposed "
        "share can reach 100%% -- a single channel is a PARTIAL answer and the report says so",
    )
    parser.add_argument(
        "--top-bigrams",
        "--top-ngrams",
        dest="top_ngrams",
        type=int,
        default=5,
        help="N-grams to name per leading column (0 to skip the block)",
    )
    parser.add_argument(
        "--no-columns",
        action="store_true",
        help="Print only the BLOCK table. Blocks are the primary result either way: a block sum "
        "is invariant to how SHAP split credit between correlated columns, and the per-column "
        "split is not unique",
    )
    parser.add_argument(
        "--control",
        choices=["bigram-table", "tcond-marginal", "shuffle"],
        help="Run a NEGATIVE control instead of the real decomposition. All MUST fail "
        "reconciliation. 'bigram-table' weights the T2 channel by bigrams.txt rather than the "
        "trigram marginal; 'tcond-marginal' weights the Tcond channel by that same marginal "
        "broadcast over the third character (the error of assuming the T2 correction transfers "
        "to a term indexed by all three); 'shuffle' permutes the per-cell SHAP deltas across "
        "cells. The two weighting controls break the EXTERNAL gauge tie while PASSING the "
        "internal sums-back bar, and shuffle does the reverse -- a control that RECONCILES "
        "means the identity is vacuous",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Seed for --control shuffle")
    parser.add_argument("--json", help="Also write the full result as JSON to this path")


def run(args: argparse.Namespace) -> int:
    # Imported here so the module's own import stays cheap: `analyze` owns the layout registry
    # (name -> 30-char string) and pulling it in at module scope would drag the whole gauge
    # stack into every `keybo --help`.
    from keybo.cli.analyze import _resolve

    name_a, lay_a = _resolve(args.layout_a)
    name_b, lay_b = _resolve(args.layout_b)
    if lay_a == lay_b:
        print(f"{name_a} and {name_b} are the same layout; there is no gap to decompose")
        return 1
    if args.json:
        ensure_writable_output(args.json, "--json")

    channel = args.channel
    kwargs: dict = {}
    if args.control == "bigram-table":
        from keybo.data.corpus import load_frequencies, production_corpus_dir

        kwargs["weighting"] = "bigram-table"
        kwargs["control_bigram_freqs"] = load_frequencies(
            str(production_corpus_dir(args.corpus) / "bigrams.txt")
        )
        # A T2-channel control cannot run against a Tcond-only decomposition. Narrow the channel
        # and SAY SO, rather than erroring or — worse — running a "control" that touches nothing
        # and then reports itself as having correctly failed for an unrelated reason.
        if channel == "tcond":
            print("--control bigram-table is a T2-channel control; running --channel t2")
            channel = "t2"
    elif args.control == "tcond-marginal":
        kwargs["tcond_weighting"] = "tcond-marginal"
        if channel == "t2":
            print("--control tcond-marginal is a Tcond-channel control; running --channel tcond")
            channel = "tcond"
    elif args.control == "shuffle":
        kwargs["shuffle_seed"] = args.shuffle_seed

    diff = shap_diff(
        lay_a,
        lay_b,
        name_a=name_a,
        name_b=name_b,
        target_wpm=args.target_wpm,
        corpus=args.corpus,
        channel=channel,
        **kwargs,
    )
    print(format_shap_diff(diff, top_bigrams_k=args.top_ngrams, columns=not args.no_columns))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(diff.to_dict(top_ngrams_k=max(args.top_ngrams, 1)), handle, indent=2)
        print(f"\nwrote {args.json}")

    if args.control:
        # A control is EXPECTED to fail; inverting the exit code makes that expectation
        # machine-checkable, so a control that silently starts passing (i.e. an identity that
        # holds even when the attribution is destroyed) surfaces as a non-zero exit.
        if diff.reconciles():
            print(
                f"\n!! CONTROL {args.control!r} RECONCILED — it was supposed to FAIL. The "
                "reconciliation is not testing the attribution. !!"
            )
            return 1
        print(f"\ncontrol {args.control!r} failed reconciliation, as required")
        return 0
    return 0 if diff.reconciles() else 1
