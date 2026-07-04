"""`keybo process-data` — turn a raw keystroke dump into a stroke table."""

from __future__ import annotations

import argparse

from keybo.cli._paths import ensure_writable_output
from keybo.data.keystrokes import process_dataset, write_ngram_tsv


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--files-dir", required=True, help="Directory of *_keystrokes.txt files")
    parser.add_argument("--metadata", required=True, help="Path to metadata_participants.txt")
    parser.add_argument("--ngram", choices=["bigram", "trigram", "skipgram"], default="bigram")
    parser.add_argument("--time-mode", choices=["full", "last"], default="full")
    parser.add_argument("--output", required=True, help="Where to write the stroke TSV")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def run(args: argparse.Namespace) -> int:
    # Validate before the multi-hour processing pass, not at the final write.
    ensure_writable_output(args.output, "--output")
    aggregated = process_dataset(
        args.files_dir,
        args.metadata,
        ngram=args.ngram,
        time_mode=args.time_mode,
        progress=not args.no_progress,
    )
    if not aggregated:
        print("No n-gram occurrences extracted; check the dataset and filters.")
        return 1
    write_ngram_tsv(aggregated, args.output)
    print(f"Wrote {len(aggregated)} {args.ngram}s -> {args.output}")
    return 0
