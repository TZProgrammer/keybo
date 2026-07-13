"""`keybo analyze` — the keybo keyboard analyzer (KAN-1, rule b330ab4).

One command, one corpus, every gauge:

* **predicted typing time** (the primary metric — no community tool has it):
  total predicted ms on the measured-keystroke surface, ms/char, percent time
  saved vs a reference layout, with per-finger and costliest-bigram attribution;
* **community scores**, each an exact parity-gated port run on its own native
  corpus convention: genkey Score, oxeylyzer-1, oxeylyzer-2 (+ its
  weighted-finger-distance component);
* **keymeow-class statistics** (sfb/sfs/lsb/alt/roll/redir + distances) on the
  shared analyzer corpus.

Layouts are 30-char row-major strings (top/home/bottom rows, left to right) or
names from the built-in registry. Mixed-charset comparisons are allowed; each
layout's corpus coverage is reported so a charset that misses corpus mass is
visible instead of silently flattered.
"""

from __future__ import annotations

import argparse
import json as _json

from keybo.analysis.community import community_suite, pinned_char
from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.analysis.timecard import default_surface
from keybo.layouts import NAMED_LAYOUTS

#: campaign layouts worth having on tap (docs/layout-*.md); the registry stays small
_EXTRA_NAMED = {
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "p16-balance": "frlwg'uyoksntdc.ieahvxmpb,-jqz",
    "p13stab-win": "rcgkmq.ouylsthd,naeixwbfvpjz;/",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "layouts",
        nargs="+",
        help="Layouts to analyze: registry names and/or 30-char row-major strings",
    )
    parser.add_argument(
        "--ref",
        default="qwerty",
        help="Reference layout for '%% time saved' (name or 30-char string; default qwerty)",
    )
    parser.add_argument(
        "--target-wpm",
        type=float,
        default=90.0,
        help="WPM the time surface is evaluated at (default 90)",
    )
    parser.add_argument(
        "--attribution",
        action="store_true",
        help="Also print per-finger time shares and the costliest bigrams per layout",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of text")


def _resolve(spec: str) -> tuple[str, str]:
    """Layout spec -> (display name, 30-char string)."""
    key = spec.lower()
    if key in NAMED_LAYOUTS:
        return key, NAMED_LAYOUTS[key]
    if key in _EXTRA_NAMED:
        return key, _EXTRA_NAMED[key]
    if len(spec) == 30:
        return spec[:8] + "…", spec
    raise SystemExit(
        f"unknown layout {spec!r}: not a registry name "
        f"({', '.join(sorted({**NAMED_LAYOUTS, **_EXTRA_NAMED}))}) and not a 30-char string"
    )


def run(args: argparse.Namespace) -> int:
    specs = [_resolve(s) for s in args.layouts]
    ref_name, ref_lay = _resolve(args.ref)
    if all(name != ref_name for name, _ in specs):
        specs.insert(0, (ref_name, ref_lay))

    surf = default_surface(args.target_wpm)
    kms = _shared_kmstats()

    ref_card = surf.card(ref_lay)
    rows: dict[str, dict] = {}
    for name, lay in specs:
        card = surf.card(lay, ref_total_ms=ref_card.total_ms)
        gk, v1, o2 = community_suite(pinned_char(lay))
        rows[name] = {
            "layout": lay,
            "time": {
                "ms_per_char": card.ms_per_char,
                "saved_vs_ref_pct": card.saved_vs_ref_pct,
                "coverage_pct": card.coverage_pct,
            },
            "community": {
                "genkey": gk.score(lay),
                "oxeylyzer1": v1.score(lay),
                "oxeylyzer2": o2.score(lay),
                "wfd": o2.wfd(lay),
            },
            "kmstats": kms.stats(lay),
        }
        if args.attribution:
            total = card.total_ms or 1.0
            rows[name]["attribution"] = {
                "finger_time_pct": {f: 100.0 * v / total for f, v in card.per_finger_ms.items()},
                "top_bigrams_ms_per_char": [
                    (bg, ms / max(surf.total_mass, 1)) for bg, ms in card.top_bigrams
                ],
            }

    if args.json:
        print(_json.dumps({"target_wpm": args.target_wpm, "ref": ref_name, "rows": rows}, indent=1))
        return 0

    _print_report(rows, ref_name, args.attribution)
    return 0


def _shared_kmstats() -> KmStats:
    from pathlib import Path

    from keybo.data.corpus import load_frequencies

    root = Path(__file__).resolve().parents[3]
    return KmStats(
        load_frequencies(str(root / "data" / "corpus" / "bigrams.txt")),
        load_frequencies(str(root / "data" / "corpus" / "1-skip.txt")),
        load_frequencies(str(root / "data" / "corpus" / "trigrams.txt")),
    )


def _print_report(rows: dict[str, dict], ref_name: str, attribution: bool) -> None:
    names = list(rows)
    w = max(len(n) for n in names) + 2
    print(f"== predicted typing time (measured-keystroke surface; ref = {ref_name}) ==")
    print(f"{'layout':<{w}}{'ms/char':>9}{'saved%':>8}{'coverage%':>11}")
    for n in names:
        t = rows[n]["time"]
        saved = f"{t['saved_vs_ref_pct']:+.2f}" if t["saved_vs_ref_pct"] is not None else "-"
        print(f"{n:<{w}}{t['ms_per_char']:>9.2f}{saved:>8}{t['coverage_pct']:>11.1f}")
    print("\n== community scores (exact ports, native corpora) ==")
    print(f"{'layout':<{w}}{'genkey↓':>9}{'oxey1↑':>13}{'oxey2↑':>16}{'wfd↑':>16}")
    for n in names:
        c = rows[n]["community"]
        print(
            f"{n:<{w}}{c['genkey']:>9.2f}{c['oxeylyzer1']:>13}{c['oxeylyzer2']:>16}{c['wfd']:>16}"
        )
    print("\n== keymeow-class stats, shared corpus (% of layout-covered mass) ==")
    print(f"{'layout':<{w}}" + "".join(f"{s:>9}" for s in STAT_NAMES))
    for n in names:
        k = rows[n]["kmstats"]
        print(f"{n:<{w}}" + "".join(f"{k[s]:>9.3f}" for s in STAT_NAMES))
    if attribution:
        for n in names:
            a = rows[n].get("attribution")
            if not a:
                continue
            fingers = {f: p for f, p in a["finger_time_pct"].items() if p > 0}
            print(f"\n== attribution: {n} ==")
            print(
                "finger time%: "
                + "  ".join(
                    f"{f} {p:.1f}" for f, p in sorted(fingers.items(), key=lambda kv: -kv[1])
                )
            )
            print(
                "costliest bigrams (ms/char): "
                + "  ".join(f"{bg!r} {v:.4f}" for bg, v in a["top_bigrams_ms_per_char"][:8])
            )
