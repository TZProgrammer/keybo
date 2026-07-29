"""ARM-1b widening: re-extract NON-QWERTY participant files with the AVG_WPM_15 floor RELAXED.

JUSTIFICATION for the single relaxation (min_wpm 40 -> 0):
  build_cells (validate.py:92) ALREADY gates every sample at session-wpm in [40,140). The
  participant-level AVG_WPM_15>=40 floor (keystrokes.py:318) is a REDUNDANT coarser pre-filter
  on a per-participant metadata variable the models never bucket on. Dropping it defers wpm
  filtering entirely to the session-wpm gate the models actually use, and lets a participant
  whose AVG_WPM_15 summary is <40 but who has fast SESSIONS contribute those fast sessions.
  FINGERS=="9-10" and KEYBOARD_TYPE in {full,laptop} are KEPT (relaxing them is a population /
  geometry change, registered as NOT done here).

Extraction is k31_extract.py VERBATIM (BUF2-BOTH conditioned trigrams, 31-char maps, 5s caps),
applied only to non-qwerty files. Two modes:
  --min-wpm 40  => POSITIVE CONTROL: must reproduce the shipped table's non-qwerty rows exactly.
  --min-wpm 0   => the WIDENED non-qwerty rows.
Writes ONLY non-qwerty rows to <out>. Qwerty is spliced from the shipped table separately.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from collections import defaultdict

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")
from keybo.data.keystrokes import (BANNED_KEYS, _letter, build_char_map,
                                    compute_session_wpm, group_sessions, mark_correct_flags)

FILES = "/local/home/zegertho/keybo-e2e/dataset/Keystrokes/files"
META = f"{FILES}/metadata_participants.txt"
BUF_K = 2
QUOTE_SLOT = (6, 2)
QUOTE_CHAR = {"qwerty": "'", "dvorak": "-", "azerty": "ù", "qwertz": "ä"}
NONQW = {"azerty", "dvorak", "qwertz"}
BIG = 10**6
csv.field_size_limit(sys.maxsize)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

def load_meta_nonqw(min_wpm: float) -> dict[str, dict]:
    """Same predicate structure as load_participant_metadata but with tunable wpm floor and
    restricted to non-qwerty supported layouts."""
    md = {}
    with open(META, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if (row.get("FINGERS") or "").strip() != "9-10":
                continue
            try:
                if float((row.get("AVG_WPM_15") or "0").strip()) < min_wpm:
                    continue
            except ValueError:
                continue
            if (row.get("KEYBOARD_TYPE") or "").strip().lower() not in {"full", "laptop"}:
                continue
            layout = (row.get("LAYOUT") or "qwerty").strip().lower()
            if layout not in NONQW:
                continue
            row["LAYOUT"] = layout
            md[(row.get("PARTICIPANT_ID") or "").strip()] = row
    return md

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-wpm", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metadata = load_meta_nonqw(args.min_wpm)
    char_maps = {}
    for name in ("azerty", "dvorak", "qwertz"):
        cmap = dict(build_char_map(name)); cmap[QUOTE_CHAR[name]] = QUOTE_SLOT
        char_maps[name] = cmap
    file_list = []
    for fname in sorted(os.listdir(FILES)):
        if not fname.endswith("_keystrokes.txt"): continue
        md = metadata.get(fname.split("_")[0])
        if md: file_list.append((fname, fname.split("_")[0], md["LAYOUT"]))
    log(f"min_wpm={args.min_wpm}: {len(file_list)} qualifying non-qwerty files")

    acc_tri = defaultdict(list)
    for fi, (fname, pid_s, layout) in enumerate(file_list):
        cmap = char_maps[layout]; allowed = set(cmap)
        with open(os.path.join(FILES, fname), newline="", encoding="utf-8", errors="replace") as f:
            rows_raw = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        for sess in group_sessions(rows_raw).values():
            if not sess: continue
            expected = sess[0].get("SENTENCE") or ""
            single = [(i, r) for i, r in enumerate(sess) if len(_letter(r)) == 1]
            if not single: continue
            typed = "".join(_letter(r) for _, r in single)
            flags = mark_correct_flags(typed, expected)
            correct = [(i, r) for (i, r), ok in zip(single, flags, strict=False) if ok]
            if len(correct) < 3: continue
            times = []
            for _i, r in correct:
                try: times.append(float(r["PRESS_TIME"]))
                except (TypeError, ValueError, KeyError): times.append(None)
            if times[0] is None or times[-1] is None: continue
            w_mean = compute_session_wpm(times[0], times[-1], len(correct))
            ks = [BIG] * len(correct)
            for i in range(1, len(correct)):
                ks[i] = min(ks[i-1]+1, BIG) if correct[i][0]-correct[i-1][0] == 1 else 0
            def ok_key(i):
                ltr = _letter(correct[i][1])
                return ltr.upper() not in BANNED_KEYS and ltr.lower() in allowed
            for i in range(len(correct) - 1):
                if ks[i] < BUF_K: continue
                if (i + 2 < len(correct) and correct[i+2][0]-correct[i][0] == 2):
                    t1, t2, t3 = times[i], times[i+1], times[i+2]
                    if (t1 is not None and t2 is not None and t3 is not None
                            and 0 < t3-t1 < 5000 and 0 < t3-t2 < 5000
                            and ok_key(i) and ok_key(i+1) and ok_key(i+2)):
                        la = _letter(correct[i][1]).lower()
                        lb = _letter(correct[i+1][1]).lower()
                        lc = _letter(correct[i+2][1]).lower()
                        acc_tri[(layout, (cmap[la], cmap[lb], cmap[lc]), la+lb+lc)].append(
                            (int(w_mean), int(t3-t2), int(pid_s), 0))
        if (fi + 1) % 200 == 0:
            log(f"  {fi+1}/{len(file_list)} files")
    log(f"extraction done: {sum(len(v) for v in acc_tri.values())} cond-trigram occ, "
        f"{len(acc_tri)} rows")
    with open(args.out, "w", encoding="utf-8") as f:
        for (layout, positions, ngram), samples in acc_tri.items():
            f.write(f"{layout}\t{repr(tuple(positions))}\t{ngram}\t{len(samples)}\t"
                    + "\t".join(repr(s) for s in samples) + "\n")
    log(f"wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
