"""`keybo fetch-data` — download the public 136M Keystrokes dataset."""

from __future__ import annotations

import argparse

from keybo.data.download import KEYSTROKES_URL, fetch_keystrokes


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="dataset", help="Where to download + extract")
    parser.add_argument("--url", default=KEYSTROKES_URL, help="Override the source URL")
    parser.add_argument("--force", action="store_true", help="Re-extract even if present")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def run(args: argparse.Namespace) -> int:
    files_dir = fetch_keystrokes(
        args.out_dir,
        url=args.url,
        force=args.force,
        show_progress=not args.no_progress,
    )
    print(f"Keystroke dataset ready -> {files_dir}")
    print(
        f"Next: keybo process-data --files-dir {files_dir} "
        f"--metadata {files_dir}/metadata_participants.txt --ngram bigram --output bistrokes.tsv"
    )
    return 0
