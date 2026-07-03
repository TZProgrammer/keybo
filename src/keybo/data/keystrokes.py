"""Process a raw keystroke dump into bistroke/tristroke tables (workflow D).

Given per-participant keystroke logs, this:
  - groups keystrokes into test sessions,
  - aligns each session's typed characters against the expected sentence to flag correct
    keys,
  - computes a session WPM from the correct keys,
  - slides an n-gram window over the correct keys, dropping windows that contain banned
    keys, keys not on the participant's layout, or multi-character "letter" fields,
  - maps each surviving n-gram to physical key positions and records ``(wpm, duration)``.

Fixes bug #6: the character map is built with the current geometry API. The old code called
``Keyboard(rows, spacebar_pos=(0, -1))`` — a signature its Keyboard class never accepted —
so the whole pipeline raised ``TypeError`` before doing any work.
"""

from __future__ import annotations

import csv
import difflib
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

from keybo.geometry import ROW_STAGGERED_30

csv.field_size_limit(sys.maxsize)

_KEYSTROKE_FILE_RE = re.compile(r"^\d+_keystrokes\.txt$")

BANNED_KEYS = {"SHIFT", "BKSP", "BACKSPACE"}

# n-gram type -> (n, skip): bigram = 2 adjacent, trigram = 3 adjacent, skipgram = 2 with a
# one-key gap.
NGRAM_SPECS = {
    "bigram": (2, 0),
    "trigram": (3, 0),
    "skipgram": (2, 1),
}

# The letter rows for each supported layout, top to bottom (10 keys each).
_LAYOUT_ROWS = {
    "qwerty": ["qwertyuiop", "asdfghjkl;", "zxcvbnm,./"],
    "azerty": ["azertyuiop", "qsdfghjklm", "wxcvbn,;:!"],
    "dvorak": ["',.pyfgcrl", "aoeuidhtns", ";qjkxbmwvz"],
    "qwertz": ["qwertzuiop", "asdfghjklö", "yxcvbnm,.-"],
}


@dataclass
class Occurrence:
    positions: tuple[tuple[int, int], ...]
    ngram: str
    wpm: int
    duration: int


def build_char_map(layout: str) -> dict[str, tuple[int, int]]:
    """Map each character of a named layout to its physical ``(x, y)`` position.

    Uses the canonical geometry slots, so it is consistent with the rest of the package.
    The space key maps to ``(0, 0)``.
    """
    if layout not in _LAYOUT_ROWS:
        raise ValueError(f"unknown layout {layout!r}; choose from {sorted(_LAYOUT_ROWS)}")
    chars = "".join(_LAYOUT_ROWS[layout])
    char_map = dict(zip(chars, ROW_STAGGERED_30.slots, strict=True))
    char_map[" "] = (0, 0)
    return char_map


def compute_session_wpm(first_press_ms: float, last_press_ms: float, n_correct: int) -> int:
    """Standard WPM = (chars / 5) / minutes, with a floor on the duration."""
    minutes = max((last_press_ms - first_press_ms) / 60000.0, 0.001)
    return round((n_correct / 5) / minutes)


def mark_correct_flags(typed: str, expected: str) -> list[bool]:
    """Flag which typed characters align with the expected sentence (difflib matching).

    Note: like the original, this does not model backspaces/shift precisely; it is a
    best-effort alignment of the raw typed string against the target.
    """
    flags = [False] * len(typed)
    matcher = difflib.SequenceMatcher(None, expected, typed)
    for _exp_start, typed_start, size in matcher.get_matching_blocks():
        for i in range(typed_start, typed_start + size):
            if i < len(flags):
                flags[i] = True
    return flags


def _letter(record: dict) -> str:
    return record.get("LETTER", "")


def extract_occurrences(
    records: list[dict],
    char_map: dict[str, tuple[int, int]],
    n: int,
    skip: int,
    time_mode: str,
) -> list[Occurrence]:
    """Slide an n-gram window over one session's records, applying all filters."""
    if not records:
        return []

    expected = records[0].get("SENTENCE", "")
    typed = "".join(_letter(r) for r in records)
    flags = mark_correct_flags(typed, expected)
    # Keep each correct key's ORIGINAL stream index. This is what lets us reject a window
    # whose keys were not actually typed consecutively -- if an incorrect key sat between
    # two correct ones, their original indices are non-contiguous, so they must not be
    # spliced into an "adjacent" n-gram (which would carry a duration spanning the mistake).
    correct = [(idx, r) for idx, (r, ok) in enumerate(zip(records, flags, strict=False)) if ok]
    if not correct:
        return []

    try:
        first_press = float(correct[0][1]["PRESS_TIME"])
        last_press = float(correct[-1][1]["PRESS_TIME"])
    except (ValueError, KeyError):
        return []
    session_wpm = compute_session_wpm(first_press, last_press, len(correct))

    allowed = set(char_map)
    span = n + skip * (n - 1)  # number of consecutive keys the window covers
    occurrences: list[Occurrence] = []

    for i in range(len(correct) - span + 1):
        indices = [i + j * (skip + 1) for j in range(n)]
        orig_indices = [correct[idx][0] for idx in indices]
        # Reject the window unless its keys were originally consecutive in the raw stream
        # (span keys covering exactly `span` original positions). A gap means a mistyped or
        # navigation key was removed from between them.
        if orig_indices[-1] - orig_indices[0] != span - 1:
            continue
        window = [correct[idx][1] for idx in indices]
        letters = [_letter(r) for r in window]

        if any(ltr.upper() in BANNED_KEYS for ltr in letters):
            continue
        if any(len(ltr) != 1 for ltr in letters):
            continue
        if any(ltr.lower() not in allowed for ltr in letters):
            continue

        try:
            first = float(window[0]["PRESS_TIME"])
            last = float(window[-1]["PRESS_TIME"])
            prev = float(window[-2]["PRESS_TIME"])
        except (ValueError, KeyError):
            continue

        if time_mode == "full":
            duration = int(last - first)
        elif time_mode == "last":
            duration = int(last - prev)
        else:
            duration = 0

        ngram = "".join(letters)
        positions = tuple(char_map[ltr.lower()] for ltr in letters)
        occurrences.append(Occurrence(positions, ngram, session_wpm, duration))
    return occurrences


def group_sessions(records: list[dict]) -> dict[str, list[dict]]:
    """Group raw rows into sessions by TEST_SECTION_ID, keeping single-character letters."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if not row.get("TEST_SECTION_ID") or len(row.get("LETTER", "")) != 1:
            continue
        sessions[row["TEST_SECTION_ID"]].append(row)
    return sessions


def aggregate_occurrences(occurrences: list[Occurrence]) -> dict:
    """Aggregate occurrences by (positions, ngram) into frequency + sample list."""
    data: dict = defaultdict(lambda: {"frequency": 0, "occurrences": []})
    for occ in occurrences:
        key = (occ.positions, occ.ngram)
        data[key]["frequency"] += 1
        data[key]["occurrences"].append((occ.wpm, occ.duration))
    return data


def write_ngram_tsv(aggregated: dict, output_path: str) -> None:
    """Write aggregated n-gram data to a TSV sorted by frequency (highest first)."""
    ordered = sorted(aggregated.items(), key=lambda kv: kv[1]["frequency"], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for (positions, ngram), info in ordered:
            pos_str = "(" + ", ".join(str(p) for p in positions) + ")"
            samples = "\t".join(str(s) for s in info["occurrences"])
            f.write(f"{pos_str}\t{ngram}\t{info['frequency']}\t{samples}\n")


# --- file-level orchestration (used by the CLI) ---------------------------------------


def load_participant_metadata(path: str, min_wpm: float = 40.0) -> dict[str, dict]:
    """Load participant metadata, keeping touch typists (9-10 fingers) above a WPM floor
    who use a full/laptop keyboard of a supported layout."""
    metadata: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("FINGERS", "").strip() != "9-10":
                continue
            try:
                if float(row.get("AVG_WPM_15", "0").strip()) < min_wpm:
                    continue
            except ValueError:
                continue
            if row.get("KEYBOARD_TYPE", "").strip().lower() not in {"full", "laptop"}:
                continue
            layout = row.get("LAYOUT", "qwerty").strip().lower()
            if layout not in _LAYOUT_ROWS:
                continue
            row["LAYOUT"] = layout
            metadata[row["PARTICIPANT_ID"].strip()] = row
    return metadata


def process_keystroke_file(
    path: str, char_map: dict[str, tuple[int, int]], n: int, skip: int, time_mode: str
) -> list[Occurrence]:
    """Process one participant's keystroke log into occurrences."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    occurrences: list[Occurrence] = []
    for session_records in group_sessions(rows).values():
        occurrences.extend(extract_occurrences(session_records, char_map, n, skip, time_mode))
    return occurrences


def process_dataset(
    files_dir: str, metadata_path: str, ngram: str, time_mode: str = "full"
) -> dict:
    """Process an entire keystroke dump directory into an aggregated n-gram table."""
    if ngram not in NGRAM_SPECS:
        raise ValueError(f"unsupported ngram {ngram!r}; choose from {sorted(NGRAM_SPECS)}")
    n, skip = NGRAM_SPECS[ngram]

    metadata = load_participant_metadata(metadata_path)
    char_maps = {name: build_char_map(name) for name in _LAYOUT_ROWS}

    all_occurrences: list[Occurrence] = []
    for fname in os.listdir(files_dir):
        if not _KEYSTROKE_FILE_RE.match(fname):
            continue
        pid = fname.split("_")[0]
        if pid not in metadata:
            continue
        char_map = char_maps[metadata[pid]["LAYOUT"]]
        all_occurrences.extend(
            process_keystroke_file(os.path.join(files_dir, fname), char_map, n, skip, time_mode)
        )
    return aggregate_occurrences(all_occurrences)
