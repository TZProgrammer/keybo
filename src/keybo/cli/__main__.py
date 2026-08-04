"""`python -m keybo` — dispatch to the workflow subcommands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from keybo.cli import (
    analyze,
    build_corpus,
    effect_curves,
    fetch_data,
    frame_collapse,
    inspect,
    layout_diff,
    optimize,
    process_data,
    score,
    shap_diff,
    shap_report,
    train,
    tune,
    validate,
)

# subcommand name -> module exposing add_arguments(parser) and run(args).
# Ordered along the pipeline: fetch-data -> (build-corpus) -> process-data -> train ->
# (tune) -> validate -> optimize / score, then the analysis tools.
_COMMANDS = {
    "fetch-data": fetch_data,
    "build-corpus": build_corpus,
    "process-data": process_data,
    "train": train,
    "tune": tune,
    "validate": validate,
    "optimize": optimize,
    "score": score,
    "analyze": analyze,
    "inspect": inspect,
    "shap-report": shap_report,
    "shap-diff": shap_diff,
    "effect-curves": effect_curves,
    "layout-diff": layout_diff,
    "frame-collapse": frame_collapse,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keybo", description="Data-driven keyboard layout optimizer"
    )
    subparsers = parser.add_subparsers(dest="command")
    for name, module in _COMMANDS.items():
        sub = subparsers.add_parser(name, help=module.__doc__)
        module.add_arguments(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    return _COMMANDS[args.command].run(args)


if __name__ == "__main__":
    raise SystemExit(main())
