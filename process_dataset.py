"""
Script to extract n-gram (bigram, trigram, or skipgram) information from the
136M keystroke dataset using the improved keyboard utilities from classifier.py.

Dataset assumptions:
  - Keystroke logs are in: dataset/Keystrokes/files/
  - Metadata is in: dataset/Keystrokes/files/metadata_participants.txt
  - Files not matching the "<PARTICIPANT_ID>_keystrokes.txt" pattern (e.g. README) are ignored.

For each keystroke log file, the script:
  - Groups keystrokes by test session.
  - Aligns the keystroke sequence with the expected sentence (using difflib) to flag correct keys.
  - Computes session WPM (using only correct keystrokes), where session duration is measured
    from the PRESS_TIME of the first key to the RELEASE_TIME of the last key.
  - Slides a window (with optional skip) over correct keystrokes to extract n-grams.
  - Filters out any n-gram occurrence that includes "SHIFT" or "BKSP"/"BACKSPACE",
    or that is typed within 2 keypresses of a backspace.
  - Maps each n-gram to physical key positions using the Keyboard class from classifier.py.
  - Aggregates occurrences and writes a TSV output sorted by frequency (highest first).

For each n-gram, the duration is computed as follows:
  - "full" mode: last key's RELEASE_TIME minus first key's PRESS_TIME.
  - "last" mode: the duration (RELEASE_TIME minus PRESS_TIME) of the last key.

Output format:
  ((pos1), (pos2), ...)<tab>ngram<tab>frequency<tab>(wpm, duration)<tab>(wpm, duration)...
  
Author: [Your Name]
Date: [Date]
"""

import os
import csv
import re
import difflib
import argparse
from collections import defaultdict
import sys

csv.field_size_limit(sys.maxsize)

from classifier import Keyboard

# Define a set of banned keys.
BANNED_KEYS = {"SHIFT", "BKSP", "BACKSPACE"}

def load_metadata(metadata_path: str) -> dict:
    """Loads metadata from the specified file, filtering for participants with 9-10 fingers
    and keyboard type 'full' or 'laptop'."""
    metadata = {}
    with open(metadata_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["FINGERS"].strip() != "9-10":
                continue
            kt = row["KEYBOARD_TYPE"].strip().lower()
            if kt not in {"full", "laptop"}:
                continue
            pid = row["PARTICIPANT_ID"].strip()
            row["LAYOUT"] = row["LAYOUT"].strip().lower()
            metadata[pid] = row
    return metadata

def mark_correct_keystrokes(records: list, expected: str, user_input: str) -> list:
    """
    Uses difflib to align the concatenated keystroke string with the expected sentence.
    Marks each record with an 'is_correct' boolean flag.
    (Note: This approach does not robustly handle backspaces or shift keys.)
    """
    typed = "".join(rec["LETTER"] for rec in records)
    correct_flags = [False] * len(records)
    matcher = difflib.SequenceMatcher(None, expected, typed)
    for block in matcher.get_matching_blocks():
        exp_start, typed_start, size = block
        for i in range(typed_start, typed_start + size):
            if i < len(correct_flags):
                correct_flags[i] = True
    for rec, flag in zip(records, correct_flags):
        rec["is_correct"] = flag
    return records

def process_test_session(session_records: list, kb: Keyboard, n: int, skip: int, time_mode: str) -> list:
    """
    Processes one test session:
      - Aligns keystrokes with the expected sentence.
      - Computes session WPM using correct keystrokes, where session duration is from the PRESS_TIME of the first key to the RELEASE_TIME of the last key.
      - Slides a window (with optional skip) over correct keystrokes to extract n-grams.
      - Filters out any n-gram occurrence that contains a banned key or that is typed within 2 keypresses after a backspace.
    
    Returns a list of occurrences: (positions, ngram_string, session_wpm, ngram_duration)
    """
    occurrences = []
    if not session_records:
        return occurrences

    expected = session_records[0]["SENTENCE"]
    user_input = session_records[0]["USER_INPUT"]

    records = mark_correct_keystrokes(session_records, expected, user_input)
    correct_records = [r for r in records if r.get("is_correct", False)]
    if not correct_records:
        return occurrences

    try:
        first_press = float(correct_records[0]["PRESS_TIME"])
        last_release = float(correct_records[-1]["RELEASE_TIME"])
    except ValueError:
        return occurrences
    duration_min = max((last_release - first_press) / 60000, 0.001)
    session_wpm = round((len(correct_records) / 5) / duration_min)

    window_length = n + skip * (n - 1)
    if len(correct_records) < window_length:
        return occurrences

    for i in range(len(correct_records) - window_length + 1):
        indices = [i + j * (skip + 1) for j in range(n)]
        # Skip this n-gram if any key in the window is banned.
        if any(correct_records[idx]["LETTER"].upper() in BANNED_KEYS for idx in indices):
            continue
        # Also, ensure the n-gram is typed at least 2 keypresses after a backspace.
        if i >= 2:
            prev1 = correct_records[i-1]["LETTER"].upper()
            prev2 = correct_records[i-2]["LETTER"].upper()
            if prev1 in {"BKSP", "BACKSPACE"} or prev2 in {"BKSP", "BACKSPACE"}:
                continue

        ngram_letters = [correct_records[idx]["LETTER"] for idx in indices]
        try:
            first_press_ng = float(correct_records[indices[0]]["PRESS_TIME"])
            last_release_ng = float(correct_records[indices[-1]]["RELEASE_TIME"])
            last_press_ng = float(correct_records[indices[-1]]["PRESS_TIME"])
        except ValueError:
            continue

        if time_mode == "full":
            ngram_duration = last_release_ng - first_press_ng
        elif time_mode == "last":
            ngram_duration = last_release_ng - last_press_ng
        else:
            ngram_duration = 0

        ngram_str = "".join(ngram_letters)
        positions = tuple(kb.get_pos(ch) for ch in ngram_str)
        occurrences.append((positions, ngram_str, session_wpm, int(ngram_duration)))
    return occurrences

def process_keystroke_file(file_path: str, kb: Keyboard, n: int, skip: int, time_mode: str) -> list:
    """
    Processes a single keystroke log file (for one participant).
    The file is expected to be tab-delimited with columns:
      PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT,
      KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE.
    
    Returns a list of n-gram occurrences.
    """
    occurrences = []
    with open(file_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        sessions = defaultdict(list)
        for row in reader:
            if not row.get("TEST_SECTION_ID") or not row.get("LETTER"):
                continue
            sessions[row["TEST_SECTION_ID"]].append(row)
        for sess_id, recs in sessions.items():
            occ = process_test_session(recs, kb, n, skip, time_mode)
            occurrences.extend(occ)
    return occurrences

def aggregate_occurrences(all_occurrences: list) -> dict:
    """
    Aggregates n-gram occurrences into a dictionary keyed by (positions, ngram_string).
    Each value contains:
      - "frequency": total count
      - "occurrences": list of (wpm, duration) tuples.
    """
    ngram_data = defaultdict(lambda: {"frequency": 0, "occurrences": []})
    for occ in all_occurrences:
        key = (occ[0], occ[1])
        ngram_data[key]["frequency"] += 1
        ngram_data[key]["occurrences"].append((occ[2], occ[3]))
    return ngram_data

def write_ngram_output(ngram_data: dict, output_file: str) -> None:
    """
    Writes aggregated n-gram data to a TSV file.
    Each line is formatted as:
      ((pos1), (pos2), ...)<tab>ngram<tab>frequency<tab>(wpm, duration)<tab>...
    The n-grams are sorted by frequency (highest first).
    """
    sorted_data = sorted(ngram_data.items(), key=lambda item: item[1]["frequency"], reverse=True)
    with open(output_file, "w") as f:
        for (positions, ngram_str), info in sorted_data:
            pos_str = "(" + ", ".join(str(pos) for pos in positions) + ")"
            freq = info["frequency"]
            occ_str = "\t".join(str(occ) for occ in info["occurrences"])
            f.write(f"{pos_str}\t{ngram_str}\t{freq}\t{occ_str}\n")
    print(f"Output written to {output_file}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract n-gram information from the 136M keystroke dataset using improved keyboard handling."
    )
    parser.add_argument("--files_dir", default="dataset/Keystrokes/files/",
                        help="Directory containing keystroke log files.")
    parser.add_argument("--metadata_file", default="dataset/Keystrokes/files/metadata_participants.txt",
                        help="Path to metadata_participants.txt")
    parser.add_argument("--output_file", default="bistrokes.tsv",
                        help="Name of the output TSV file.")
    parser.add_argument("--ngram", choices=["bigram", "trigram", "skipgram"], default="bigram",
                        help="Type of n-gram to extract.")
    parser.add_argument("--time_mode", choices=["full", "last"], default="full",
                        help="Duration mode: 'full' (from first key PRESS_TIME to last key RELEASE_TIME) or 'last' (last key's press-release duration).")
    args = parser.parse_args()

    if not os.path.exists(args.metadata_file):
        print(f"Metadata file not found: {args.metadata_file}")
        return
    metadata = load_metadata(args.metadata_file)
    if not metadata:
        print("No participants met filtering criteria.")
        return

    if args.ngram == "bigram":
        n, skip = 2, 0
    elif args.ngram == "trigram":
        n, skip = 3, 0
    elif args.ngram == "skipgram":
        n, skip = 2, 1
    else:
        raise ValueError("Unsupported ngram type.")

    all_files = [fname for fname in os.listdir(args.files_dir) if re.match(r"^\d+_keystrokes\.txt$", fname)]
    total_files = len(all_files)
    processed_files = 0
    all_occurrences = []

    for i, fname in enumerate(all_files):
        file_path = os.path.join(args.files_dir, fname)
        pid = fname.split("_")[0]
        if pid not in metadata:
            continue
        layout_key = metadata[pid].get("LAYOUT", "qwerty")
        if layout_key not in {"qwerty", "azerty", "dvorak", "qwertz"}:
            layout_key = "qwerty"
        if layout_key == "qwerty":
            rows = ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]
        elif layout_key == "azerty":
            rows = ["azertyuiop", "qsdfghjkl;", "wxcvbnm,./"]
        elif layout_key == "dvorak":
            rows = ["',.pyfgcrl", "aoeuidhtns;", "qjkxbmwvz"]
        elif layout_key == "qwertz":
            rows = ["qwertzuiop", "asdfghjklö", "yxcvbnm,.-"]
        else:
            rows = ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"]
        kb = Keyboard(rows, spacebar_pos=(0, -1))
        processed_files += 1
        percentage = (processed_files / total_files) * 100
        print(f"{percentage:.2f}% Processing participant {pid} with layout {layout_key} from file {fname} ...")
        occ = process_keystroke_file(file_path, kb, n, skip, args.time_mode)
        all_occurrences.extend(occ)
        print(f"Progress: {percentage:.2f}% complete", end="\r", flush=True)

    print()  # Newline after progress.
    if not all_occurrences:
        print("No n-gram occurrences extracted. Check data and filtering criteria.")
        return

    aggregated = aggregate_occurrences(all_occurrences)
    write_ngram_output(aggregated, args.output_file)

if __name__ == "__main__":
    main()
