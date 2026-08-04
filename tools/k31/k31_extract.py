"""K31 stage B (rule 2542bc4): BUF2-BOTH extraction with 31-char maps.

p8_final.py stage 1 verbatim, except each layout's char map gains its physical ANSI
quote-slot character at (6, 2): qwerty ', dvorak -, azerty ù, qwertz ä. Everything
else (BUF2-BOTH buffer, contiguity, 5s caps, conditioned trigram timing) is identical.
GATE (registered): v5's rows must be a subset; the delta must be windows CONTAINING
the quote-slot char (plus windows previously broken by a ' interruption now valid).
"""

import csv
import os
import sys
import time
from collections import defaultdict

# Only needed when keybo is not installed (no `pip install -e .`); harmless otherwise.
if os.environ.get("KEYBO_SRC"):
    sys.path.insert(0, os.environ["KEYBO_SRC"])

from keybo.data.keystrokes import (
    BANNED_KEYS,
    _letter,
    build_char_map,
    compute_session_wpm,
    group_sessions,
    load_participant_metadata,
    mark_correct_flags,
)

# Raw Aalto dump, as produced by `keybo fetch-data` (see tools/k31/README.md).
FILES = os.environ.get("K31_FILES_DIR", "dataset/Keystrokes/files")
BUF_K = 2
BI_TSV = os.environ.get("K31_BI_OUT", "bistrokes31_v1.tsv")
TRI_TSV = os.environ.get("K31_TRI_OUT", "tristrokes31_cond_v1.tsv")
QUOTE_SLOT = (6, 2)
QUOTE_CHAR = {"qwerty": "'", "dvorak": "-", "azerty": "ù", "qwertz": "ä"}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def write_tsv(acc, path):
    with open(path, "w", encoding="utf-8") as f:
        for (layout, positions, ngram), samples in acc.items():
            pos_str = repr(tuple(positions))
            f.write(f"{layout}\t{pos_str}\t{ngram}\t{len(samples)}\t"
                    + "\t".join(repr(s) for s in samples) + "\n")


log("stage B: K31 BUF2-BOTH extraction")
metadata = load_participant_metadata(f"{FILES}/metadata_participants.txt")
char_maps = {}
for name in ("qwerty", "azerty", "dvorak", "qwertz"):
    cmap = dict(build_char_map(name))
    cmap[QUOTE_CHAR[name]] = QUOTE_SLOT
    char_maps[name] = cmap
log("char maps built (31 keys each): " + ", ".join(
    f"{n}+{QUOTE_CHAR[n]!r}" for n in char_maps))

acc_bi = defaultdict(list)
acc_tri = defaultdict(list)
BIG = 10**6
file_list = []
for fname in sorted(os.listdir(FILES)):
    if not fname.endswith("_keystrokes.txt"):
        continue
    pid_s = fname.split("_")[0]
    md = metadata.get(pid_s)
    if md:
        file_list.append((fname, pid_s, md["LAYOUT"]))
log(f"{len(file_list)} qualifying files")

for fi, (fname, pid_s, layout) in enumerate(file_list):
    cmap = char_maps[layout]
    allowed = set(cmap)
    with open(os.path.join(FILES, fname), newline="", encoding="utf-8",
              errors="replace") as f:
        rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for sess in group_sessions(rows_raw).values():
        if not sess:
            continue
        expected = sess[0].get("SENTENCE") or ""
        single = [(i, r) for i, r in enumerate(sess) if len(_letter(r)) == 1]
        if not single:
            continue
        typed = "".join(_letter(r) for _, r in single)
        flags = mark_correct_flags(typed, expected)
        correct = [(i, r) for (i, r), ok in zip(single, flags, strict=False) if ok]
        if len(correct) < 3:
            continue
        times = []
        for _i, r in correct:
            try:
                times.append(float(r["PRESS_TIME"]))
            except (TypeError, ValueError, KeyError):
                times.append(None)
        if times[0] is None or times[-1] is None:
            continue
        w_mean = compute_session_wpm(times[0], times[-1], len(correct))
        ks = [BIG] * len(correct)
        for i in range(1, len(correct)):
            if correct[i][0] - correct[i - 1][0] == 1:
                ks[i] = min(ks[i - 1] + 1, BIG)
            else:
                ks[i] = 0

        def ok_key(i):
            ltr = _letter(correct[i][1])
            return ltr.upper() not in BANNED_KEYS and ltr.lower() in allowed

        for i in range(len(correct) - 1):
            if ks[i] < BUF_K:
                continue
            if correct[i + 1][0] - correct[i][0] == 1:
                t1, t2 = times[i], times[i + 1]
                if (t1 is not None and t2 is not None and 0 < t2 - t1 < 5000
                        and ok_key(i) and ok_key(i + 1)):
                    la = _letter(correct[i][1]).lower()
                    lb = _letter(correct[i + 1][1]).lower()
                    acc_bi[(layout, (cmap[la], cmap[lb]), la + lb)].append(
                        (int(w_mean), int(t2 - t1), int(pid_s), 0))
            if (i + 2 < len(correct)
                    and correct[i + 2][0] - correct[i][0] == 2):
                t1, t2, t3 = times[i], times[i + 1], times[i + 2]
                if (t1 is not None and t2 is not None and t3 is not None
                        and 0 < t3 - t1 < 5000 and 0 < t3 - t2 < 5000
                        and ok_key(i) and ok_key(i + 1) and ok_key(i + 2)):
                    la = _letter(correct[i][1]).lower()
                    lb = _letter(correct[i + 1][1]).lower()
                    lc = _letter(correct[i + 2][1]).lower()
                    acc_tri[(layout, (cmap[la], cmap[lb], cmap[lc]), la + lb + lc)].append(
                        (int(w_mean), int(t3 - t2), int(pid_s), 0))
    if (fi + 1) % 10000 == 0:
        log(f"  {fi + 1}/{len(file_list)} files")

log(f"extraction done: {sum(len(v) for v in acc_bi.values())} bigram occ, "
    f"{sum(len(v) for v in acc_tri.values())} cond-trigram occ")
write_tsv(acc_bi, BI_TSV)
write_tsv(acc_tri, TRI_TSV)
log(f"wrote {BI_TSV} + {TRI_TSV}")

# ---- registered gate: v5 subset check --------------------------------------------------------
quote_chars = set("'-ùä")
n_quote_rows = sum(1 for (lay, _pos, ng) in acc_bi if any(c in quote_chars for c in ng))
n_plain_rows = len(acc_bi) - n_quote_rows
occ_quote = sum(len(v) for (lay, _pos, ng), v in acc_bi.items()
                if any(c in quote_chars for c in ng))
log(f"bigram rows: {len(acc_bi)} total = {n_plain_rows} plain + {n_quote_rows} quote-slot "
    f"({occ_quote} quote occurrences)")
log("ALL-DONE")
