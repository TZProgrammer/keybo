"""Ingest community monkeytype captures (Kiakl form responses) into the production
stroke-TSV schema (KIAKL-INGEST registration, PREREGISTRATIONS.md 2026-07-12).

Source schema (per json file): a list of sessions::

    {"data": [{"key": "s", "interval": 456.2, "correct": true}, ...],
     "keyboardType": "ortholinear" | "rowStagger" | "angleMod",
     "layout": "`1234...=<13 top><11 home><10 bottom>...",  # monkeytype full string
     "sessionID": 1708268370386, "website": "https://monkeytype.com/"}

``interval`` is press-to-press ms from the PREVIOUS event (first event: 0). There are
no release timestamps, so ``hold`` is always -1 in the output.

Fixed ingestion rules (registered before processing — see KIAKL-INGEST):
 dedup by sessionID; user = submitter (pid 200001+); layout label =
 ``<name>@<kbt>#<submitter>``; windows require all-correct events and every interval
 in (0, 5000], resetting at session boundaries and after any incorrect event; wpm =
 session-level from correct intervals; output = production TSV rows on
 ROW_STAGGERED_30 slots with corpus-table frequencies.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from keybo.geometry import ROW_STAGGERED_30

#: known 30-key cores (top+home+bottom rows, 10 each) for naming; unmatched strings
#: get a ``custom-<8hex>`` slug instead — never a silently wrong name.
KNOWN_LAYOUTS = {
    "qwerty": "qwertyuiopasdfghjkl;zxcvbnm,./",
    "colemak": "qwfpgjluy;arstdhneiozxcvbkm,./",
    "colemak-dh": "qwfpbjluy;arstgmneiozxcdvkh,./",
    "canary": "wlypbzfou;crstgmneiaqjvdkxh/,.",
    "mtgap": 'ypoujkdlcwinea,mhtsrq";.:bfgvx',
    "recurva": "frdpvqjuoysntcb.heaizxkgwml,;/",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "sturdy": "vmlcpxfouj;strdy.naeizkqgwbh',",
}
#: >= this many exact slot matches against a known core counts as a variant of it
_VARIANT_THRESHOLD = 24

MAX_INTERVAL_MS = 5000.0

#: the 30-key qwerty core in slot order — the FIXED physical-key labeling. The capture's
#: ``key`` field is the QWERTY LABEL of the physical key pressed (monkeytype layout
#: emulation records the raw browser key), NOT the produced character (KIAKL-INGEST
#: Amendment 3: decoding every label's correct-key stream through qwerty30->main30
#: yields fluent English; as-is yields gibberish).
QWERTY30 = "qwertyuiopasdfghjkl;zxcvbnm,./"
_QWERTY_INDEX = {c: i for i, c in enumerate(QWERTY30)}

#: shifted-label recovery (Amendment 4a): shifted presses are valid typing; unshift the
#: label before decoding. Letters lowercase; qwerty punctuation shift pairs map back.
_UNSHIFT = {"<": ",", ">": ".", "?": "/", ":": ";"}


def decode_event_key(key: str, main30: str) -> str:
    """The character the emulated layout PRODUCED for a captured key label.

    ``key`` is the qwerty label of the physical key pressed; the produced character is
    the session layout's char on that physical slot: ``main30[qwerty_index(key)]``.
    Identity for qwerty sessions. Undecodable labels (control keys like ``Backspace``,
    chars outside the 30-key core) are returned unchanged — they are absent from the
    layout's char->slot map, so they break windows exactly as before.
    """
    if key == " ":
        return " "
    k = key.lower() if len(key) == 1 and key.isalpha() else _UNSHIFT.get(key, key)
    idx = _QWERTY_INDEX.get(k)
    return main30[idx] if idx is not None else key


def dedup_prefix_streams(sessions: list[SessionRecord]) -> list[SessionRecord]:
    """Drop re-exported sessions (Amendment 4b): sessionID is an export timestamp, so a
    double-export gets a fresh id and survives the id dedup. A re-export's event stream
    is a strict prefix of the later export's (same keys, same intervals, same flags —
    the later copy has a few trailing events). Keep the longest of each prefix chain.

    Comparison includes intervals: legitimate repeats of the same monkeytype text have
    identical (key, correct) streams but different timings — those are kept.
    """
    streams = [tuple(s.events) for s in sessions]
    order = sorted(range(len(sessions)), key=lambda i: streams[i])
    drop: set[int] = set()
    for a, b in zip(order, order[1:], strict=False):
        sa, sb = streams[a], streams[b]
        if len(sa) < len(sb) and sb[: len(sa)] == sa:
            drop.add(a)
        elif sa == sb:
            drop.add(b)
    return [s for i, s in enumerate(sessions) if i not in drop]


def main30_from_monkeytype(layout_str: str) -> str | None:
    """Extract the 30-key core from monkeytype's full layout string.

    The string concatenates the unshifted rows — number row (13 keys, ``\\`1..=``),
    top row (13, trailing ``[]\\``), home row (11, trailing ``'``), bottom row (10) —
    followed by the shifted repeat. We take the first 10 of top/home/bottom.
    """
    layout_str = layout_str.lstrip()  # one capture variant has a leading space
    if len(layout_str) < 47:
        return None
    top, home, bottom = layout_str[13:26], layout_str[26:37], layout_str[37:47]
    return top[:10] + home[:10] + bottom[:10]


def identify_layout(main30: str) -> str:
    best_name, best_n = None, -1
    for name, ref in KNOWN_LAYOUTS.items():
        n = sum(1 for a, b in zip(main30, ref, strict=False) if a == b)
        if n > best_n:
            best_name, best_n = name, n
    if best_n == 30:
        return best_name
    if best_n >= _VARIANT_THRESHOLD:
        return f"{best_name}-variant"
    import hashlib

    return "custom-" + hashlib.sha1(main30.encode()).hexdigest()[:8]


def submitter_slug(filename: str) -> str:
    """Form filenames look like ``typingdata... - Firstname Lastname.json``."""
    stem = Path(filename).stem
    who = stem.split(" - ")[-1] if " - " in stem else stem
    who = re.sub(r"\(\d+\)\s*$", "", who)  # "(1)" download-duplicate counters
    slug = re.sub(r"[^a-z0-9]+", "", who.lower())
    return slug or "anon"


@dataclass
class SessionRecord:
    session_id: int
    submitter: str
    layout_label: str
    main30: str
    wpm: int
    events: list[tuple[str, float, bool]]  # (key, interval_ms, correct)
    source_stem: str = ""  # originating filename stem, for corpus tagging


@dataclass
class IngestReport:
    files_seen: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    sessions_total: int = 0
    sessions_deduped: int = 0
    sessions_prefix_deduped: int = 0
    sessions_kept: int = 0
    events_kept: int = 0
    labels: dict[str, int] = field(default_factory=dict)


def _session_wpm(events: list[tuple[str, float, bool]]) -> int:
    """Session-level WPM from correct-event intervals ((chars/5) per minute)."""
    total_ms = sum(iv for _, iv, ok in events if ok and 0 < iv <= MAX_INTERVAL_MS)
    n_ok = sum(1 for _, _, ok in events if ok)
    if total_ms <= 0 or n_ok < 10:
        return 0
    return int(round((n_ok / 5.0) / (total_ms / 60000.0)))


def load_sessions(json_paths: list[Path], report: IngestReport) -> list[SessionRecord]:
    """Parse, dedup (by sessionID), label, and wpm-annotate all sessions."""
    seen_ids: set[int] = set()
    out: list[SessionRecord] = []
    for path in sorted(json_paths):
        report.files_seen.append(path.name)
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            report.files_skipped.append(f"{path.name} (unparseable)")
            continue
        if not (isinstance(doc, list) and doc and isinstance(doc[0], dict) and "data" in doc[0]):
            report.files_skipped.append(f"{path.name} (not a session list)")
            continue
        who = submitter_slug(path.name)
        for sess in doc:
            report.sessions_total += 1
            sid = sess.get("sessionID")
            if sid in seen_ids:
                report.sessions_deduped += 1
                continue
            seen_ids.add(sid)
            main30 = main30_from_monkeytype(sess.get("layout", ""))
            if main30 is None:
                continue
            events = [
                (e.get("key", ""), float(e.get("interval", -1)), bool(e.get("correct")))
                for e in sess.get("data", [])
            ]
            wpm = _session_wpm(events)
            if wpm <= 0:
                continue
            label = f"{identify_layout(main30)}@{sess.get('keyboardType', '?')}#{who}"
            # source_stem only for files WITHOUT a " - submitter" suffix (the GK zip
            # files); form-response files must never match the GK stem overrides.
            stem = path.stem if " - " not in path.stem else ""
            out.append(SessionRecord(sid, who, label, main30, wpm, events, source_stem=stem))
    # Amendment 4b: sessionID is an export timestamp — re-exports get fresh ids. Drop
    # sessions whose event stream is a strict prefix (or exact copy) of another's.
    before = len(out)
    out = dedup_prefix_streams(out)
    report.sessions_prefix_deduped = before - len(out)
    for sess in out:
        report.sessions_kept += 1
        report.events_kept += len(sess.events)
        report.labels[sess.layout_label] = report.labels.get(sess.layout_label, 0) + 1
    return out


def _char_positions(main30: str) -> dict[str, tuple[int, int]]:
    pos = {c: p for c, p in zip(main30, ROW_STAGGERED_30.slots, strict=False)}
    pos[" "] = ROW_STAGGERED_30.space_position
    return pos


def extract_windows(
    sessions: list[SessionRecord], pids: dict[str, int], n: int, time_mode: str
) -> dict[tuple[str, tuple, str], list[tuple[int, int, int, int]]]:
    """Valid n-gram windows -> production samples, honoring the registered rules.

    ``time_mode``: "full" = press1->pressN duration; "last" = press(N-1)->pressN.
    A window is valid iff all n events are correct, all n-1 within-window intervals
    are in (0, MAX_INTERVAL_MS], and no event before position 0 was incorrect in a
    way that contaminates event 0's own landing (event 0's interval is NOT part of
    the window, so only correctness of the window's events matters — but an
    incorrect event anywhere inside breaks every window covering it).
    """
    cells: dict[tuple[str, tuple, str], list] = defaultdict(list)
    for sess in sessions:
        cmap = _char_positions(sess.main30)
        pid = pids[sess.submitter]
        ev = sess.events
        for i in range(len(ev) - n + 1):
            win = ev[i : i + n]
            if not all(ok for _, _, ok in win):
                continue
            ivs = [iv for _, iv, _ in win[1:]]
            if any(iv <= 0 or iv > MAX_INTERVAL_MS for iv in ivs):
                continue
            # Amendment 3: the captured key is a QWERTY LABEL; decode to the produced
            # char. cmap[produced] is then the PHYSICAL slot pressed (main30[i] sits on
            # slot i, so cmap[main30[qidx(label)]] == slots[qidx(label)]).
            keys = [decode_event_key(k, sess.main30) for k, _, _ in win]
            if any(k not in cmap for k in keys):
                continue
            positions = tuple(cmap[k] for k in keys)
            ngram = "".join(keys)
            dur = ivs[-1] if time_mode == "last" else sum(ivs)
            cells[(sess.layout_label, positions, ngram)].append(
                (sess.wpm, int(round(dur)), pid, -1)
            )
    return cells


def write_tsv(
    cells: dict, corpus_freqs: dict[str, int], out_path: Path, min_samples: int = 1
) -> int:
    rows = 0
    with open(out_path, "w") as f:
        for (label, positions, ngram), samples in sorted(cells.items()):
            if len(samples) < min_samples:
                continue
            freq = corpus_freqs.get(ngram, 0)
            parts = [label, repr(positions), ngram, str(freq)]
            parts += [repr(s) for s in samples]
            f.write("\t".join(parts) + "\n")
            rows += 1
    return rows
