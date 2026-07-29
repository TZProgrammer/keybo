"""Measure `finger-travel` and `off-home` over the full layout field (FT round, 2026-07-28).

Emits ONE json artifact holding, per layout: the travel shares + absolute total + dispersion,
the off-home columns, and every gauge/time value pulled from the SHIPPED `keybo analyze` path
(never a reimplementation) so the correlation analysis compares like with like.

Two guards run BEFORE anything is measured:

* ``assert_module_under`` — this repo's venv carries an editable ``.pth`` into a DIFFERENT
  clone's ``src``, so a probe can silently score the wrong tree while every printed path looks
  right. Verified today on this box.
* the layout field is **derived** from the shipped registries and (for the three campaign
  candidates) grepped out of ``PREREGISTRATIONS.md`` — never retyped. Two of two
  hand-transcriptions by a prior arm were wrong.

Run: ``PYTHONPATH=src python agent-artifacts/ft_board.py <out.json>``
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "PREREGISTRATIONS.md"

#: The three adoption candidates that are NOT in a shipped registry. Held as (label, regex) and
#: resolved by grepping the ledger, so the 30-char string is never typed here.
GREPPED = (
    ("arm-B", r"flmpg-yuo,sntdcireahkxbwv'\.jzq"),
    ("BALL-1", r"flmpg-yuo,sntcdireahkxbwv'\.jzq"),
    ("armH-hdln", r"flmpg-,uoysntcdireahkxvwb\.'jzq"),
)


def grepped_candidates() -> dict[str, str]:
    """Pull the campaign candidate strings out of the ledger; refuse anything unverified."""
    text = LEDGER.read_text(errors="replace")
    out: dict[str, str] = {}
    for label, pattern in GREPPED:
        found = set(re.findall(pattern, text))
        if len(found) != 1:
            raise SystemExit(
                f"{label}: expected exactly one distinct match for {pattern!r} in the ledger, "
                f"got {len(found)} — refusing to guess which is the candidate"
            )
        (layout,) = found
        if len(layout) != 30 or len(set(layout)) != 30:
            raise SystemExit(f"{label}: {layout!r} is not a 30-char permutation")
        out[label] = layout
    return out


def field() -> dict[str, str]:
    """The full field: 15 registry layouts + 3 grepped candidates. Deduplicated by STRING.

    ``keybo-lsb``, ``keybo-lsb+lm`` and ``flagship-c3`` are in ``_EXTRA_NAMED`` already, so the
    brief's six adoption candidates are covered by the registry plus the three grepped ones.
    """
    from keybo.cli.analyze import _EXTRA_NAMED
    from keybo.layouts import NAMED_LAYOUTS

    registry = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
    if len(registry) != 15:
        raise SystemExit(f"expected 15 registry layouts, found {len(registry)} — field changed")
    everything = {**registry, **grepped_candidates()}
    by_string: dict[str, str] = {}
    for name, layout in everything.items():
        if layout in by_string:
            raise SystemExit(f"{name!r} and {by_string[layout]!r} are the same layout {layout!r}")
        by_string[layout] = name
    return everything


def analyze(layouts: dict[str, str]) -> dict:
    """Run the SHIPPED analyzer over the field and return its json, keyed by our labels.

    Passing raw 30-char strings (not registry names) keeps one code path for all 18 layouts,
    and the returned rows are re-keyed back to our labels by matching row["layout"].
    """
    command = [
        sys.executable,
        "-m",
        "keybo.cli.__main__",
        "analyze",
        *sorted(set(layouts.values())),
        "--no-model-scores",
        "--json",
    ]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"analyze failed rc={done.returncode}:\n{done.stderr[-4000:]}")
    payload = json.loads(done.stdout)
    by_string = {row["layout"]: row for row in payload["rows"].values()}
    missing = [name for name, lay in layouts.items() if lay not in by_string]
    if missing:
        raise SystemExit(f"analyze returned no row for {missing} — refusing a partial board")
    return {
        "corpus": payload["corpus_provenance"],
        "rows": {name: by_string[lay] for name, lay in layouts.items()},
    }


def main(out_path: str) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from keybo.analysis.finger_travel import FingerTravel, OffHomeUsage
    from keybo.data.corpus import PRODUCTION_SKIPGRAMS, load_frequencies, production_corpus_dir
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout
    from keybo.testkit import assert_module_under

    assert_module_under("keybo.analysis.finger_travel", ROOT / "src")
    assert_module_under("keybo.cli.analyze", ROOT / "src")

    layouts = field()
    corpus_dir = production_corpus_dir(None)
    bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
    trigrams = load_frequencies(str(corpus_dir / "trigrams.txt"))
    skipgrams = load_frequencies(str(corpus_dir / PRODUCTION_SKIPGRAMS))
    assert skipgrams, "skipgram table must load even though only bigrams/trigrams are used here"

    travel = FingerTravel(bigrams)
    off_home = OffHomeUsage(bigrams)

    print(f"analyzing {len(layouts)} layouts through the shipped path…", file=sys.stderr)
    shipped = analyze(layouts)

    board = {}
    for name, lay30 in sorted(layouts.items()):
        layout = Layout(lay30, ROW_STAGGERED_30)
        row = shipped["rows"][name]
        board[name] = {
            "layout": lay30,
            "travel": travel.report(layout),
            "travel_static_shares": _shares(travel.static_per_finger(layout)),
            "travel_static_total": sum(travel.static_per_finger(layout).values()),
            "travel_slowness_weighted_shares": travel.slowness_weighted_shares(layout),
            "travel_lag2_shares": travel.lag2_shares(layout, trigrams),
            "off_home": off_home.report(layout),
            "gauges": row["gauges"],
            "time": row["time"],
            "bad_scissor_share": row["bad_scissor"]["share"],
            "scissor_by_finger": row["scissor_by_finger"],
        }
        print(
            f"  {name:<14} travel_total={board[name]['travel']['total']:.4g}"
            f"  pinky_off={board[name]['off_home']['pinky']['off_home']:.2f}",
            file=sys.stderr,
        )

    Path(out_path).write_text(
        json.dumps(
            {
                "generated_by": "agent-artifacts/ft_board.py",
                "prereg": "docs/finger-travel-preregistration.md",
                "corpus": shipped["corpus"],
                "note": (
                    "travel/off_home computed by keybo.analysis.finger_travel; gauges and time "
                    "come from the SHIPPED `keybo analyze --json` path, not a reimplementation"
                ),
                "rows": board,
            },
            indent=1,
        )
    )
    print(f"wrote {out_path}", file=sys.stderr)


def _shares(charged: dict[str, float]) -> dict[str, float]:
    total = sum(charged.values())
    return {k: (100.0 * v / total if total else 0.0) for k, v in charged.items()}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "agent-artifacts/ft_board.json")
