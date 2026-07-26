"""`keybo analyze` — the keybo keyboard analyzer (KAN-1 b330ab4; ALLGAUGE-1).

One command, one corpus, every gauge the campaign uses:

* **predicted typing time** (the primary metric — no community tool has it):
  total predicted ms on the measured-keystroke surface, ms/char, percent time
  saved vs a reference layout, with per-finger and costliest-bigram attribution;
* **community scores**, each an exact parity-gated port run on its own native
  corpus convention: genkey Score, oxeylyzer-1, oxeylyzer-2 (+ its
  weighted-finger-distance component), and each one's **primed** counterpart;
* **the 15 corpus-sensitive gauges** of the campaign's frozen all-gauge frame: the
  11 keymeow-class statistics plus ``scissor``, ``imbalance``, ``oxey-style`` and
  ``comfort``;
* **per-finger scissor** — the aggregate scissor gauge split across the eight
  fingers as an exact partition (:mod:`keybo.analysis.scissor_fingers`);
* **the oxeylyzer redirect family** including bad-redirect
  (:mod:`keybo.analysis.redirects`);
* **the three fitted model surfaces** (aalto / community / pool)
  (:mod:`keybo.analysis.surfaces`).

Three conventions this command commits to, each because getting it wrong has already
cost the campaign a wrong number:

**Skipgrams come from ``1-skip31.txt``.** That file *is* the trigram marginalization
``skip(a,c) = sum_b tri(a,b,c)`` (``data/build_corpus.py``, verified byte-exact there) and
is the table every frozen campaign board was computed on. ``1-skip.txt`` is a different,
unreproducible pass; analyze used it until ALLGAUGE-1, which is why its ``sfs`` /
``sfs-dist`` / ``oxey-style`` / ``comfort`` did not reproduce any frozen board.

**One frame per gauge, named.** ``scissor`` in particular is not one quantity across the
campaign's artifacts — the same layout reads 0.12670 on the all-gauge board's
``tb_objective_axes`` and 0.13891 on the three-corpus board. This command reports the
``oxey.pattern_shares`` convention, whose denominator is the **layout-restricted bigram
mass**, and every gauge's denominator is stated in its own module.

**Charset-dependent cells render N/A, never a number.** dvorak is not a C30M layout (it
carries both ``;`` and ``'`` and lacks ``-``), so the oxeylyzer 31-key boards and the
modeled surfaces cannot score it. Before ALLGAUGE-1 that raised ``ValueError``.

Layouts are 30-char row-major strings (top/home/bottom rows, left to right) or
names from the built-in registry. Mixed-charset comparisons are allowed; each
layout's corpus coverage is reported so a charset that misses corpus mass is
visible instead of silently flattered.
"""

from __future__ import annotations

import argparse
import json as _json
from pathlib import Path

from keybo.analysis import surfaces as S
from keybo.analysis.community import community_suite, pinned_char
from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.analysis.redirects import REDIRECT_CLASSES, RedirectFamily
from keybo.analysis.scissor_fingers import FINGER_NAMES, ScissorByFinger
from keybo.analysis.timecard import default_surface
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.comfort import ComfortBigramScorer
from keybo.scoring.oxey import OxeyStyleScorer
from keybo.scoring.scissor_severity import (
    DEFAULT_SEVERITY,
    FLAT,
    ScissorSeverity,
    SeverityWeights,
)

#: campaign layouts worth having on tap (docs/layout-*.md); the registry stays small
_EXTRA_NAMED = {
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "p16-balance": "frlwg'uyoksntdc.ieahvxmpb,-jqz",
    "p13stab-win": "rcgkmq.ouylsthd,naeixwbfvpjz;/",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
}

#: The 15 corpus-sensitive gauges of the frozen all-gauge frame, in board order.
GAUGE_NAMES = (*STAT_NAMES, "scissor", "imbalance", "oxey-style", "comfort")

#: Sentinel rendered for a cell a layout's charset cannot support.
NA = "N/A"


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
        help=(
            "WPM the measured-keystroke time surface is evaluated at (default 90). "
            "NOTE the fitted model surfaces are baked at 90 WPM and do not move with this"
        ),
    )
    parser.add_argument(
        "--attribution",
        action="store_true",
        help="Also print per-finger time shares and the costliest bigrams per layout",
    )
    parser.add_argument(
        "--model-family",
        default=S.DEFAULT_FAMILY,
        choices=S.FAMILIES,
        help=(
            "Which fitted surface family the model columns report "
            f"(default {S.DEFAULT_FAMILY}, the campaign's peak model)"
        ),
    )
    parser.add_argument(
        "--surface-dir",
        default=None,
        help=(
            "Directory holding <NAME>.standardized.npy[.gz] model surfaces; overrides the "
            "vendored data/surfaces (env: KEYBO_SURFACE_DIR)"
        ),
    )
    parser.add_argument(
        "--no-model-scores",
        action="store_true",
        help="Skip the fitted model-surface columns",
    )
    parser.add_argument(
        "--no-time",
        action="store_true",
        help="Skip the measured-keystroke time card (the slow part: it loads 6 models)",
    )
    parser.add_argument(
        "--scissor-pairs",
        action="store_true",
        help="Also print scissor mass per adjacent FINGER PAIR (a second exact partition)",
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


def _oxeylyzer_scorable(lay30: str) -> bool:
    """Whether the oxeylyzer 31-key boards can score this layout.

    They need the 30 characters to be exactly one of the two known charsets, so that the
    layout plus its pinned quote-slot character forms a 31-character permutation. dvorak
    carries BOTH ``'`` and ``;`` and lacks ``-``, so it is neither.
    """
    c30m = set("qwertyuiopasdfghjkl'zxcvbnm,.-")
    classic = set("qwertyuiopasdfghjkl;zxcvbnm,./")
    return set(lay30) in (c30m, classic)


def run(args: argparse.Namespace) -> int:
    specs = [_resolve(s) for s in args.layouts]
    ref_name, ref_lay = _resolve(args.ref)
    if all(name != ref_name for name, _ in specs):
        specs.insert(0, (ref_name, ref_lay))

    if args.surface_dir is not None and not Path(args.surface_dir).is_dir():
        raise SystemExit(f"--surface-dir {args.surface_dir!r}: model surface directory not found")

    bigrams, skipgrams, trigrams = _shared_corpora()
    kms = KmStats(bigrams, skipgrams, trigrams)
    oxey = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    comfort = ComfortBigramScorer(bigrams, skipgram_freqs=skipgrams)
    bigram_mass = sum(bigrams.values())
    scissor_fingers = ScissorByFinger(bigrams)
    redirects = RedirectFamily(trigrams)
    severity = ScissorSeverity(bigrams)

    surf = None if args.no_time else default_surface(args.target_wpm)
    ref_card = surf.card(ref_lay) if surf is not None else None

    rows: dict[str, dict] = {}
    for name, lay in specs:
        layout = Layout(lay, ROW_STAGGERED_30)
        row: dict = {"layout": lay}

        if surf is not None:
            card = surf.card(lay, ref_total_ms=ref_card.total_ms)
            row["time"] = {
                "ms_per_char": card.ms_per_char,
                "saved_vs_ref_pct": card.saved_vs_ref_pct,
                "coverage_pct": card.coverage_pct,
            }
        else:
            card = None
            row["time"] = None

        # --- the 15 corpus-sensitive gauges, on the frozen board's convention ---
        gauges = dict(kms.stats(lay))
        shares = oxey.pattern_shares(layout)
        gauges["scissor"] = shares["scissor"]
        gauges["imbalance"] = shares["imbalance"]
        gauges["oxey-style"] = oxey.fitness(layout)
        # comfort is an ABSOLUTE corpus-scale ms-equivalent sum; dividing by the FULL
        # corpus bigram mass is the frozen board's convention (board_three_corpora.py).
        # Note this denominator differs from every other gauge's here -- stated, not hidden.
        gauges["comfort"] = comfort.fitness(layout) / bigram_mass
        row["gauges"] = gauges

        # Graded scissor: the same support, weighted by finger tier and reach direction.
        # A declared PREFERENCE (docs/scissor-severity-preregistration.md), not a
        # measurement -- so it is reported alongside the flat gauge, never instead of it.
        # `flat_control` re-derives the flat share through the graded code path at all
        # weights 1.0; it must equal `gauges["scissor"]`, which is the positive control
        # that the grading is a strict generalization rather than a rival metric.
        row["scissor_graded"] = {
            "share": severity.share(layout, DEFAULT_SEVERITY),
            "weights": DEFAULT_SEVERITY.label(),
            "flat_control": severity.share(layout, FLAT),
            "by_class": severity.breakdown(layout, DEFAULT_SEVERITY),
            "class_masses_unweighted": severity.class_masses(layout),
            "wide_support_share": severity.share(layout, SeverityWeights(support="wide")),
            "note": (
                "a declared preference, not a measurement; wide_support_share drops the "
                "column-adjacency gate (the only support where middle-pinky mass is visible)"
            ),
        }

        row["scissor_by_finger"] = scissor_fingers.shares(layout)
        row["scissor_by_finger_rule"] = "half-to-each-finger"
        if args.scissor_pairs:
            row["scissor_by_finger_pair"] = scissor_fingers.pair_shares(layout)
        row["redirects"] = redirects.shares(lay)

        # --- community scores: raw and primed; N/A when the charset cannot be boarded ---
        gk, v1, o2 = community_suite(pinned_char(lay))
        if _oxeylyzer_scorable(lay):
            row["community"] = {
                "genkey": gk.score(lay),
                "oxeylyzer1": v1.score(lay),
                "oxeylyzer2": o2.score(lay),
                "wfd": o2.wfd(lay),
            }
            row["community_primed"] = {
                "genkey_primed": gk.score_primed(lay),
                "oxey1_primed": float(v1.score_primed(lay)),
                "oxey2_primed": float(o2.score_primed(lay)),
                # The dominance boards' wfd pins ' on the quote slot; the components wfd
                # (row["community"]["wfd"]) pins whatever pinned_char() picks. Two
                # different numbers for the same name -- both reported, both labelled.
                "wfd": float(o2.wfd_apostrophe_pinned(lay)) if "'" in lay else None,
                "wfd_frame": "apostrophe-pinned (dominance-board convention)",
            }
        else:
            # genkey is charset-agnostic (it scores its own parsed corpus by character).
            row["community"] = {
                "genkey": gk.score(lay),
                "oxeylyzer1": None,
                "oxeylyzer2": None,
                "wfd": None,
            }
            row["community_primed"] = {
                "genkey_primed": gk.score_primed(lay),
                "oxey1_primed": None,
                "oxey2_primed": None,
                "wfd": None,
                "wfd_frame": "apostrophe-pinned (dominance-board convention)",
            }

        # --- fitted model surfaces ---
        if args.no_model_scores:
            row["model_scores"] = {
                "available": False,
                "reason": "skipped (--no-model-scores)",
                "surfaces": {},
                "family": args.model_family,
                "frame": S.FRAME_NOTE,
                "baked_wpm": S.BAKED_WPM,
                "wpm_matches_request": float(args.target_wpm) == S.BAKED_WPM,
                "wpm_note": S.wpm_note(args.target_wpm),
            }
        else:
            row["model_scores"] = S.model_scores(
                lay,
                family=args.model_family,
                target_wpm=args.target_wpm,
                ref_lay30=ref_lay,
                surface_dir=args.surface_dir,
            )

        if args.attribution and card is not None:
            total = card.total_ms or 1.0
            row["attribution"] = {
                "finger_time_pct": {f: 100.0 * v / total for f, v in card.per_finger_ms.items()},
                "top_bigrams_ms_per_char": [
                    (bg, ms / max(surf.total_mass, 1)) for bg, ms in card.top_bigrams
                ],
            }
        rows[name] = row

    if args.json:
        print(
            _json.dumps(
                {
                    "target_wpm": args.target_wpm,
                    "ref": ref_name,
                    "skipgram_table": "1-skip31.txt",
                    "gauge_frame": (
                        "wscissor-allgauge: kmstats + oxey.pattern_shares + comfort/bigram-mass"
                    ),
                    "model_family": args.model_family,
                    "rows": rows,
                },
                indent=1,
            )
        )
        return 0

    _print_report(rows, ref_name, args)
    return 0


def _shared_corpora() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """The production corpus: bigrams, ``1-skip31`` skipgrams, trigrams.

    ``1-skip31.txt`` (not ``1-skip.txt``) is the table the campaign's frozen gauge boards
    were computed on — see the module docstring.
    """
    from keybo.data.corpus import load_frequencies

    root = Path(__file__).resolve().parents[3]
    corpus = root / "data" / "corpus"
    return (
        load_frequencies(str(corpus / "bigrams.txt")),
        load_frequencies(str(corpus / "1-skip31.txt")),
        load_frequencies(str(corpus / "trigrams.txt")),
    )


def _cell(value: float | None, width: int, spec: str = ".3f") -> str:
    """Right-aligned number, or N/A — never a silently-wrong number."""
    if value is None:
        return f"{NA:>{width}}"
    return f"{value:>{width}{spec}}"


def _print_report(rows: dict[str, dict], ref_name: str, args: argparse.Namespace) -> None:
    names = list(rows)
    w = max(len(n) for n in names) + 2

    if not args.no_time:
        print(f"== predicted typing time (measured-keystroke surface; ref = {ref_name}) ==")
        print(f"{'layout':<{w}}{'ms/char':>9}{'saved%':>8}{'coverage%':>11}")
        for n in names:
            t = rows[n]["time"]
            saved = f"{t['saved_vs_ref_pct']:+.2f}" if t["saved_vs_ref_pct"] is not None else "-"
            print(f"{n:<{w}}{t['ms_per_char']:>9.2f}{saved:>8}{t['coverage_pct']:>11.1f}")
        print()

    print("== community scores (exact ports, native corpora) ==")
    print(f"{'layout':<{w}}{'genkey↓':>12}{'oxey1↑':>16}{'oxey2↑':>18}{'wfd↑':>18}")
    for n in names:
        c = rows[n]["community"]
        print(
            f"{n:<{w}}{_cell(c['genkey'], 12, '.2f')}"
            + "".join(_cell(c[k], width, ".0f") for k, width in (("oxeylyzer1", 16),))
            + _cell(c["oxeylyzer2"], 18, ".0f")
            + _cell(c["wfd"], 18, ".0f")
        )

    print("\n== community PRIMED (strain residual; the frozen all-gauge board's frame) ==")
    print(f"{'layout':<{w}}{'genkey′↓':>12}{'oxey1′↑':>16}{'oxey2′↑':>18}{'wfd(board)↑':>18}")
    for n in names:
        p = rows[n]["community_primed"]
        print(
            f"{n:<{w}}{_cell(p['genkey_primed'], 12, '.4f')}"
            + _cell(p["oxey1_primed"], 16, ".0f")
            + _cell(p["oxey2_primed"], 18, ".0f")
            + _cell(p["wfd"], 18, ".0f")
        )
    print(
        "wfd(board) pins ' on the quote slot (dominance-board convention); the wfd above "
        "it pins the layout's own quote character — two different quantities, same name"
    )

    print(f"\n== all-gauge frame, shared corpus (1-skip31); {len(GAUGE_NAMES)} gauges ==")
    print(f"{'layout':<{w}}" + "".join(f"{s:>11}" for s in GAUGE_NAMES))
    for n in names:
        g = rows[n]["gauges"]
        print(f"{n:<{w}}" + "".join(_cell(g[s], 11) for s in GAUGE_NAMES))

    print("\n== scissor by finger (% of layout-covered bigram mass; sums to `scissor`) ==")
    print(f"{'layout':<{w}}" + "".join(f"{f:>9}" for f in FINGER_NAMES) + f"{'total':>10}")
    for n in names:
        per_finger = rows[n]["scissor_by_finger"]
        print(
            f"{n:<{w}}"
            + "".join(f"{per_finger[f]:>9.4f}" for f in FINGER_NAMES)
            + f"{sum(per_finger.values()):>10.4f}"
        )
    print(f"attribution rule: {rows[names[0]]['scissor_by_finger_rule']} (an exact partition)")

    print("\n== scissor GRADED (a declared preference, not a measurement) ==")
    print(f"{'layout':<{w}}{'flat↓':>10}{'graded↓':>10}{'flat-ctl':>10}{'wide↓':>10}  {'weights'}")
    for n in names:
        graded = rows[n]["scissor_graded"]
        print(
            f"{n:<{w}}{rows[n]['gauges']['scissor']:>10.4f}{graded['share']:>10.4f}"
            f"{graded['flat_control']:>10.4f}{graded['wide_support_share']:>10.4f}"
            f"  {graded['weights']}"
        )
    print(
        "flat-ctl re-derives the flat gauge through the graded code path at all weights 1.0 — "
        "it must equal flat (a positive control that grading generalizes rather than replaces)"
    )

    if args.scissor_pairs:
        print("\n== scissor by adjacent finger PAIR (a second exact partition) ==")
        keys = sorted({k for n in names for k in rows[n].get("scissor_by_finger_pair", {})})
        print(f"{'layout':<{w}}" + "".join(f"{k:>10}" for k in keys))
        for n in names:
            pairs = rows[n].get("scissor_by_finger_pair", {})
            print(f"{n:<{w}}" + "".join(f"{pairs.get(k, 0.0):>10.4f}" for k in keys))

    print("\n== redirect family, oxeylyzer-1 classes (% of layout-covered trigram mass) ==")
    print(
        f"{'layout':<{w}}"
        + "".join(f"{c:>19}" for c in REDIRECT_CLASSES)
        + f"{'bad-total':>12}{'family':>10}"
    )
    for n in names:
        r = rows[n]["redirects"]
        print(
            f"{n:<{w}}"
            + "".join(f"{r[c]:>19.4f}" for c in REDIRECT_CLASSES)
            + f"{r['bad_redirects_total']:>12.4f}{r['redirects_family_total']:>10.4f}"
        )
    print("(the four classes are mutually exclusive; family total == the `redir` gauge above)")

    _print_model_scores(rows, names, w)

    if args.attribution:
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


def _print_model_scores(rows: dict[str, dict], names: list[str], w: int) -> None:
    """The three fitted model surfaces, or one clear line saying why not."""
    first = rows[names[0]]["model_scores"]
    scored = [n for n in names if rows[n]["model_scores"]["available"]]
    if not scored:
        print(f"\n== fitted model surfaces ==\n{first['reason']}")
        return
    surface_names = list(rows[scored[0]]["model_scores"]["surfaces"])
    print(f"\n== fitted model surfaces — family {first['family']}; {first['frame']} ==")
    print(f"{'layout':<{w}}" + "".join(f"{s:>30}" for s in surface_names))
    for n in names:
        scores = rows[n]["model_scores"]
        if not scores["available"]:
            print(
                f"{n:<{w}}"
                + "".join(f"{NA:>30}" for _ in surface_names)
                + f"  ({scores['reason']})"
            )
            continue
        cells = []
        for s in surface_names:
            cell = scores["surfaces"][s]
            saved = cell["saved_vs_ref_pct"]
            cells.append(
                f"{cell['fit'] / 1e9:>19.4f}Gms"
                + (f"{saved:>+7.2f}%" if saved is not None else f"{NA:>8}")
            )
        print(f"{n:<{w}}" + "".join(f"{c:>30}" for c in cells))
    print(f"note: {first['wpm_note']}")
