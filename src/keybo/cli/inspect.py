"""`keybo inspect` — structural diagnostics for a layout vs the named layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keybo.cli._paths import ensure_writable_output
from keybo.data.corpus import (
    CORPUS_ENV_VAR,
    PRODUCTION_DEFAULT,
    corpus_identity,
    corpus_name_for,
    known_corpora,
    load_frequencies,
    production_corpus_dir,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.inspect import layout_diagnostics


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layout",
        required=True,
        help="A 30-char layout string, or a named layout (qwerty, dvorak, ...)",
    )
    # Resolved LAZILY in run() (CORPUS-SWAP-1): the default must follow KEYBO_CORPUS at run
    # time, and build_parser() builds every subcommand's arguments on every invocation, so
    # an eager resolve would let a bad KEYBO_CORPUS break even `keybo --help`.
    parser.add_argument(
        "--bigram-freqs",
        default=None,
        help="Bigram frequency file (default: the --corpus bigrams.txt)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            f"Corpus supplying the default bigram table: {' | '.join(known_corpora())}, or a "
            f"directory (default {PRODUCTION_DEFAULT}; env: {CORPUS_ENV_VAR})"
        ),
    )
    parser.add_argument("--out", help="Write the full diagnostics JSON to this path")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def run(args: argparse.Namespace) -> int:
    if args.out:
        ensure_writable_output(args.out, "--out")
    bigram_path = args.bigram_freqs or str(production_corpus_dir(args.corpus) / "bigrams.txt")
    freqs = load_frequencies(bigram_path)
    layout_str = NAMED_LAYOUTS.get(args.layout, args.layout)

    target = layout_diagnostics(Layout(layout_str, ROW_STAGGERED_30), freqs)
    named = {
        name: layout_diagnostics(Layout(s, ROW_STAGGERED_30), freqs)
        for name, s in sorted(NAMED_LAYOUTS.items())
    }

    cols = ["layout", *sorted(NAMED_LAYOUTS)]
    diags = {"layout": target, **named}

    print(f"inspect: {args.layout}")
    # Name the corpus: every share below is corpus-weighted, and the default changed
    # (iWeb -> blend-v1) at CORPUS-SWAP-1.
    print(f"corpus:  {corpus_name_for(Path(bigram_path).parent)} ({bigram_path})")
    print(f"{'':<18}" + "".join(f"{c:>10}" for c in cols))
    fingers = list(target["finger_load"])
    for f in fingers:
        print(
            f"load {f:<13}" + "".join(_fmt_pct(diags[c]["finger_load"][f]).rjust(10) for c in cols)
        )
    for r in ("top", "home", "bottom", "space"):
        print(f"row {r:<14}" + "".join(_fmt_pct(diags[c]["row_share"][r]).rjust(10) for c in cols))
    for m in ("alternate", "same_hand", "same_finger"):
        print(
            f"motion {m:<11}"
            + "".join(_fmt_pct(diags[c]["motion_share"][m]).rjust(10) for c in cols)
        )
    for key, label in (
        ("sfb_share", "sfb"),
        ("scissor_share", "scissor"),
        ("lsb_share", "lsb"),
        ("excluded_weight_share", "excluded"),
    ):
        print(f"{label:<18}" + "".join(_fmt_pct(diags[c][key]).rjust(10) for c in cols))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "corpus_provenance": corpus_identity(Path(bigram_path).parent),
                    "layout": target,
                    "named": named,
                },
                f,
                indent=2,
            )
        print(f"report -> {args.out}")
    return 0
