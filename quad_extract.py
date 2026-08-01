"""Quadgram (4-key) extraction — faithful n=4 extension of k31_extract.py.

Matches the trigram frame construction EXACTLY except n=4:
  - BUF2-BOTH buffer (ks[i] >= BUF_K), 31-char maps + quote slot,
  - CONDITIONED last-interval timing: target = t4 - t3 (press3->press4),
    the n=4 analogue of the trigram's t3 - t2,
  - caps: 0 < t4 - t1 < 5000  AND  0 < t4 - t3 < 5000  (span cap + last-interval cap,
    the n=4 analogue of the trigram's 0<t3-t1<5000 and 0<t3-t2<5000),
  - all four keys contiguous in the raw stream and on-layout / not banned.

Usage:
    python quad_extract.py <out.tsv> [max_files]     # max_files omitted => all
"""

import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/tmp/quadgram-wt/src")

from keybo.data.keystrokes import (
    BANNED_KEYS,
    _letter,
    build_char_map,
    compute_session_wpm,
    group_sessions,
    load_participant_metadata,
    mark_correct_flags,
)

FILES = "/local/home/zegertho/keybo-e2e/dataset/Keystrokes/files"
BUF_K = 2
QUOTE_SLOT = (6, 2)
QUOTE_CHAR = {"qwerty": "'", "dvorak": "-", "azerty": "ù", "qwertz": "ä"}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)


def write_tsv(acc, path):
    with open(path, "w", encoding="utf-8") as f:
        for (layout, positions, ngram), samples in acc.items():
            pos_str = repr(tuple(positions))
            f.write(
                f"{layout}\t{pos_str}\t{ngram}\t{len(samples)}\t"
                + "\t".join(repr(s) for s in samples)
                + "\n"
            )


def main():
    out_path = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None

    log("quadgram extraction (BUF2-BOTH, conditioned last-interval t4-t3)")
    metadata = load_participant_metadata(f"{FILES}/metadata_participants.txt")
    char_maps = {}
    for name in ("qwerty", "azerty", "dvorak", "qwertz"):
        cmap = dict(build_char_map(name))
        cmap[QUOTE_CHAR[name]] = QUOTE_SLOT
        char_maps[name] = cmap

    file_list = []
    for fname in sorted(os.listdir(FILES)):
        if not fname.endswith("_keystrokes.txt"):
            continue
        pid_s = fname.split("_")[0]
        md = metadata.get(pid_s)
        if md:
            file_list.append((fname, pid_s, md["LAYOUT"]))
    if max_files is not None:
        file_list = file_list[:max_files]
    log(f"{len(file_list)} qualifying files")

    acc_quad = defaultdict(list)
    n_sessions = 0
    for fi, (fname, pid_s, layout) in enumerate(file_list):
        cmap = char_maps[layout]
        allowed = set(cmap)
        with open(
            os.path.join(FILES, fname), newline="", encoding="utf-8", errors="replace"
        ) as f:
            rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        for sess in group_sessions(rows_raw).values():
            if not sess:
                continue
            n_sessions += 1
            expected = sess[0].get("SENTENCE") or ""
            single = [(i, r) for i, r in enumerate(sess) if len(_letter(r)) == 1]
            if not single:
                continue
            typed = "".join(_letter(r) for _, r in single)
            flags = mark_correct_flags(typed, expected)
            correct = [(i, r) for (i, r), ok in zip(single, flags, strict=False) if ok]
            if len(correct) < 4:
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
            ks = [10**6] * len(correct)
            for i in range(1, len(correct)):
                if correct[i][0] - correct[i - 1][0] == 1:
                    ks[i] = min(ks[i - 1] + 1, 10**6)
                else:
                    ks[i] = 0

            def ok_key(i):
                ltr = _letter(correct[i][1])
                return ltr.upper() not in BANNED_KEYS and ltr.lower() in allowed

            for i in range(len(correct) - 3):
                if ks[i] < BUF_K:
                    continue
                # all four keys contiguous in the raw stream
                if correct[i + 3][0] - correct[i][0] != 3:
                    continue
                t1, t2, t3, t4 = times[i], times[i + 1], times[i + 2], times[i + 3]
                if None in (t1, t2, t3, t4):
                    continue
                if not (0 < t4 - t1 < 5000 and 0 < t4 - t3 < 5000):
                    continue
                if not (ok_key(i) and ok_key(i + 1) and ok_key(i + 2) and ok_key(i + 3)):
                    continue
                la = _letter(correct[i][1]).lower()
                lb = _letter(correct[i + 1][1]).lower()
                lc = _letter(correct[i + 2][1]).lower()
                ld = _letter(correct[i + 3][1]).lower()
                acc_quad[
                    (layout, (cmap[la], cmap[lb], cmap[lc], cmap[ld]), la + lb + lc + ld)
                ].append((int(w_mean), int(t4 - t3), int(pid_s), 0))
        if (fi + 1) % 10000 == 0:
            log(
                f"  {fi + 1}/{len(file_list)} files | "
                f"{len(acc_quad)} distinct quads | "
                f"{sum(len(v) for v in acc_quad.values())} occ"
            )

    total_occ = sum(len(v) for v in acc_quad.values())
    log(
        f"extraction done: {len(acc_quad)} distinct quadgrams, {total_occ} occurrences, "
        f"{n_sessions} sessions"
    )
    write_tsv(acc_quad, out_path)
    log(f"wrote {out_path}")
    log("ALL-DONE")


if __name__ == "__main__":
    main()
