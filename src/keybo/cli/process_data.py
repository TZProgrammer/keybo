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
    parser.add_argument(
        "--hesitation-cap",
        type=float,
        default=3.0,
        help="Drop windows containing an interval > CAP x the session's median clean "
        "interval — hesitations (cognition), not typing motion. 0 disables. (default 3.0; "
        "adopted 2026-07-06: improved every LOLO metric ~23%%, rho/ceiling 0.97->1.01)",
    )
    parser.add_argument("--output", required=True, help="Where to write the stroke TSV")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def run(args: argparse.Namespace) -> int:
    # Validate before the multi-hour processing pass, not at the final write.
    ensure_writable_output(args.output, "--output")
    counters: dict = {}
    aggregated = process_dataset(
        args.files_dir,
        args.metadata,
        ngram=args.ngram,
        time_mode=args.time_mode,
        progress=not args.no_progress,
        counters=counters,
        hesitation_cap=args.hesitation_cap,
    )
    if not aggregated:
        print("No n-gram occurrences extracted; check the dataset and filters.")
        return 1
    write_ngram_tsv(aggregated, args.output)
    print(f"Wrote {len(aggregated)} {args.ngram}s -> {args.output}")
    _print_counters(counters)
    return 0


def _print_counters(c: dict) -> None:
    files = c.get("files_processed", 0)
    sessions = c.get("session_total", 0)
    no_single = c.get("session_no_single_char_rows", 0)
    no_correct = c.get("session_no_correct_chars", 0)
    bad_time = c.get("session_bad_time", 0)
    kept = c.get("window_kept", 0)
    non_contig = c.get("window_non_contiguous", 0)
    banned = c.get("window_banned_key", 0)
    multi = c.get("window_multi_char", 0)
    off_layout = c.get("window_off_layout", 0)
    w_bad_time = c.get("window_bad_time", 0)
    hesitation = c.get("window_hesitation", 0)
    print(
        f"files: {files}  sessions: {sessions}"
        f" (no-single: {no_single}, no-correct: {no_correct}, bad-time: {bad_time})"
    )
    print(
        f"windows: kept {kept} | non-contiguous {non_contig} | banned-key {banned}"
        f" | multi-char {multi} | off-layout {off_layout} | bad-time {w_bad_time}"
        f" | hesitation {hesitation}"
    )
