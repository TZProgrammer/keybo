"""CLI: process the raw community zips into production stroke TSVs.

Usage::

    python -m keybo.data.community_cli <raw_dir> <out_dir>

``raw_dir`` should contain the extracted Kiakl form-response jsons (any nesting).
Writes ``bistrokes_community.tsv``, ``tristrokes_community.tsv``,
``tristrokes_last_community.tsv``, and ``ingest_report.json`` to ``out_dir``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from keybo.data.community import IngestReport, extract_windows, load_sessions, write_tsv

#: community pids start here — disjoint from aalto participant ids
PID_BASE = 200001


def _load_corpus(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            out[parts[0]] = int(parts[1])
    return out


def main() -> None:
    raw_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(__file__).resolve().parents[3] / "data" / "corpus"

    report = IngestReport()
    sessions = load_sessions(sorted(raw_dir.rglob("*.json")), report)
    # KIAKL-INGEST amendment: the gk_typingdata.zip files carry no submitter in the
    # filename (they are Grzegorz Kulesza's — same pid as his form submissions), and
    # two of them are non-natural text (pseudo-words / rare-char-boosted words) that
    # must stay distinguishable via a corpus tag on the label.
    gk_stems = {"typingdata": None, "typingdata0003": "+pseudo", "typingdata1278": "+rareboost"}
    for sess in sessions:
        if sess.source_stem in gk_stems:
            old = sess.submitter
            sess.submitter = "grzegorzkulesza"
            sess.layout_label = sess.layout_label.replace(f"#{old}", "#grzegorzkulesza")
            tag = gk_stems[sess.source_stem]
            if tag:
                sess.layout_label += tag
    report.labels = {}
    for sess in sessions:
        report.labels[sess.layout_label] = report.labels.get(sess.layout_label, 0) + 1
    pids = {who: PID_BASE + i for i, who in enumerate(sorted({s.submitter for s in sessions}))}

    bi = extract_windows(sessions, pids, n=2, time_mode="full")
    tri = extract_windows(sessions, pids, n=3, time_mode="full")
    tri_last = extract_windows(sessions, pids, n=3, time_mode="last")

    bi_freq = _load_corpus(corpus_dir / "bigrams.txt")
    tri_freq = _load_corpus(corpus_dir / "trigrams.txt")
    n_bi = write_tsv(bi, bi_freq, out_dir / "bistrokes_community.tsv")
    n_tri = write_tsv(tri, tri_freq, out_dir / "tristrokes_community.tsv")
    n_tl = write_tsv(tri_last, tri_freq, out_dir / "tristrokes_last_community.tsv")

    summary = {
        "pids": pids,
        "rows": {"bistrokes": n_bi, "tristrokes": n_tri, "tristrokes_last": n_tl},
        "sessions_total": report.sessions_total,
        "sessions_deduped": report.sessions_deduped,
        "sessions_kept": report.sessions_kept,
        "events_kept": report.events_kept,
        "files_skipped": report.files_skipped,
        "labels": dict(sorted(report.labels.items(), key=lambda kv: -kv[1])),
    }
    (out_dir / "ingest_report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
