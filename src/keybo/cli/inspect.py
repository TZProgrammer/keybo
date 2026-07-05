"""`keybo inspect` — structural diagnostics for a layout vs the named layouts."""

from __future__ import annotations

import argparse
import json

from keybo.cli._paths import ensure_writable_output
from keybo.data.corpus import load_frequencies
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.inspect import layout_diagnostics

_DEFAULT_BIGRAMS = "data/corpus/bigrams.txt"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layout",
        required=True,
        help="A 30-char layout string, or a named layout (qwerty, dvorak, ...)",
    )
    parser.add_argument("--bigram-freqs", default=_DEFAULT_BIGRAMS, help="Bigram frequency file")
    parser.add_argument("--out", help="Write the full diagnostics JSON to this path")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def run(args: argparse.Namespace) -> int:
    if args.out:
        ensure_writable_output(args.out, "--out")
    freqs = load_frequencies(args.bigram_freqs)
    layout_str = NAMED_LAYOUTS.get(args.layout, args.layout)

    target = layout_diagnostics(Layout(layout_str, ROW_STAGGERED_30), freqs)
    named = {
        name: layout_diagnostics(Layout(s, ROW_STAGGERED_30), freqs)
        for name, s in sorted(NAMED_LAYOUTS.items())
    }

    cols = ["layout", *sorted(NAMED_LAYOUTS)]
    diags = {"layout": target, **named}

    print(f"inspect: {args.layout}")
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
            json.dump({"layout": target, "named": named}, f, indent=2)
        print(f"report -> {args.out}")
    return 0
