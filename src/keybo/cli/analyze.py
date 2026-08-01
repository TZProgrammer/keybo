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
from keybo.analysis.bad_scissor import ATTRIBUTION_RULE as BAD_SCISSOR_RULE
from keybo.analysis.bad_scissor import FINGER_ORDER as BAD_SCISSOR_FINGERS
from keybo.analysis.bad_scissor import BadScissor
from keybo.analysis.community import (
    C30M_CHARS,
    CLASSIC_CHARS,
    community_suite,
    legacy_board_of,
    pinned_char,
)
from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.analysis.redirects import REDIRECT_CLASSES, RedirectFamily
from keybo.analysis.scissor_fingers import FINGER_NAMES, ScissorByFinger
from keybo.analysis.timecard import default_surface
from keybo.data.corpus import (
    CORPUS_ENV_VAR,
    IWEB,
    PRODUCTION_DEFAULT,
    corpus_identity,
    known_corpora,
    production_corpus_dir,
)
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
    # The comfort micro-variant of keybo-lsb (SELECT-1's blind spot: it cuts middle-pinky
    # scissor mass ~55%). Named here so a corpus comparison can cite it as a row rather
    # than a 30-char string, which is how a transcription error gets into a board.
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
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
        "--corpus",
        default=None,
        help=(
            f"Corpus to score on: {' | '.join(known_corpora())}, or a directory holding the "
            f"tables (default {PRODUCTION_DEFAULT}; env: {CORPUS_ENV_VAR}). Use "
            f"'--corpus {IWEB}' to reproduce the campaign's frozen boards"
        ),
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
        return spec, spec
    raise SystemExit(
        f"unknown layout {spec!r}: not a registry name "
        f"({', '.join(sorted({**NAMED_LAYOUTS, **_EXTRA_NAMED}))}) and not a 30-char string"
    )


def _display(name: str) -> str:
    """Shorten a raw 30-char layout for table columns; registry names pass through.

    Only ever applied at PRINT time. The row key stays the full layout string, because
    ``rows`` is a dict: keying it on a truncation silently DROPPED a row whenever two
    layouts shared the first 8 characters (``keybo-lsb`` vs ``keybo-lsb+lm``,
    ``archive-1843`` vs ``archive-1846``) — two layouts in, one row out, exit 0.
    """
    return name[:8] + "…" if len(name) == 30 else name


def _oxeylyzer_scorable(lay30: str) -> bool:
    """Whether the oxeylyzer 31-key boards can score this layout.

    They need the 30 characters to be exactly one of the two known charsets, so that the
    layout plus its pinned quote-slot character forms a 31-character permutation. dvorak
    carries BOTH ``'`` and ``;`` and lacks ``-``, so it is neither.
    """
    return set(lay30) in (set(C30M_CHARS), set(CLASSIC_CHARS))


def _wfd_reconciliation(o2, lay30: str) -> dict | None:
    """How the frozen artifacts' wfd differs from the correct one, decomposed exactly.

    The campaign's wfd came from a board that is not a permutation of the 31 keys, so the
    gap is not a convention offset to be reported alongside: it is attributable, to the
    last integer, to the three key positions the bug disturbs. Reporting the decomposition
    (rather than a second gauge column) is what stops the legacy number being read as an
    alternative measurement of the same layout.

    ``None`` when the legacy board cannot be built (a classic-charset layout has no ``'``
    in its 30 block, and the legacy mapping is only *wrong* for C30M anyway).
    """
    if "'" not in lay30:
        return None
    correct = o2.wfd(lay30)
    legacy = o2.wfd_legacy_board(lay30)
    board = legacy_board_of(lay30, o2.chars)
    evicted = sorted(set(o2.chars) - set(board))
    duplicated = sorted({c for c in board if board.count(c) > 1})
    return {
        "correct_wfd": correct,
        "legacy_board_wfd": legacy,
        "delta": legacy - correct,
        "delta_pct_of_correct": 100.0 * (legacy - correct) / abs(correct),
        "legacy_board": board,
        "legacy_board_is_a_permutation": len(set(board)) == len(o2.chars),
        "evicted_characters": evicted,
        "duplicated_characters": duplicated,
        "why": (
            "the legacy board never assigns ';' a position, so it lands on dof 0 "
            "(top-left, left pinky) and evicts the character on slot 0; the dof that "
            "' vacated is refilled by index 0 ('q'). This is a bug, not a convention."
        ),
        "use": (
            "reconcile frozen artifacts only; correct_wfd is the quantity to rank or "
            "gate on. 14 of 42 frozen per-incumbent dominance verdicts do not survive "
            "the correction."
        ),
    }


def _raw_total_reconciliation(card, ref_card) -> dict | None:
    """How the frozen artifacts' raw-TOTALS saved% differs from the rankable one.

    Same contract as :func:`_wfd_reconciliation`, for the same reason: the raw-total
    number is not a second way of measuring the layout, it is the same measurement taken
    on a frame that only holds when both layouts cover the same corpus mass. A layout
    that can type MORE of the corpus accumulates a larger total for that reason alone,
    so the raw-total comparison charges it for its own coverage — which is how
    ``graphite`` came to be reported as slower than qwerty while being 5.5 ms/char
    faster. Reporting it as ``rankable + delta``, with the non-comparability named, is
    what stops a reader (or a ``--json`` consumer) ranking on it.

    ``None`` when there is no reference comparison to reconcile.
    """
    if card.saved_vs_ref_pct is None or card.raw_total_saved_vs_ref_pct is None:
        return None
    equal_coverage = card.coverage_pct == ref_card.coverage_pct
    return {
        "raw_total_saved_vs_ref_pct": card.raw_total_saved_vs_ref_pct,
        "delta": card.raw_total_saved_vs_ref_pct - card.saved_vs_ref_pct,
        "coverage_pct": card.coverage_pct,
        "ref_coverage_pct": ref_card.coverage_pct,
        "equal_coverage": equal_coverage,
        "comparable_across_charsets": False,
        "why_not_comparable": (
            "raw corpus TOTALS are accumulated over each layout's OWN typable subset, so "
            "they only compare at equal coverage; a wider charset covers more corpus mass "
            "and is charged for it, which can report a faster layout as slower"
        ),
        "rank_on": "saved_vs_ref_pct",
        "use": (
            "reconcile frozen artifacts only. It is exact at equal coverage (every frozen "
            "board this repo pins compares same-charset layouts, where delta == 0)."
        ),
    }


def run(args: argparse.Namespace) -> int:
    specs = [_resolve(s) for s in args.layouts]
    ref_name, ref_lay = _resolve(args.ref)
    if all(name != ref_name for name, _ in specs):
        specs.insert(0, (ref_name, ref_lay))

    # A row per distinct layout, or fail loudly. `rows` is a dict, so any two specs that
    # resolve to the same display name would silently collapse into one row with exit 0 —
    # invisible in the output and in every board built from it.
    seen: dict[str, str] = {}
    for name, lay in specs:
        if name in seen and seen[name] != lay:
            raise SystemExit(
                f"layout name collision: {name!r} resolves to two different layouts "
                f"({seen[name]!r} and {lay!r}); pass registry names to disambiguate"
            )
        seen[name] = lay

    if args.surface_dir is not None and not Path(args.surface_dir).is_dir():
        raise SystemExit(f"--surface-dir {args.surface_dir!r}: model surface directory not found")

    corpus_dir = production_corpus_dir(args.corpus)
    corpus_block = corpus_identity(corpus_dir)

    bigrams, skipgrams, trigrams = _shared_corpora(corpus_dir)
    kms = KmStats(bigrams, skipgrams, trigrams)
    oxey = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    comfort = ComfortBigramScorer(bigrams, skipgram_freqs=skipgrams)
    bigram_mass = sum(bigrams.values())
    scissor_fingers = ScissorByFinger(bigrams)
    redirects = RedirectFamily(trigrams)
    severity = ScissorSeverity(bigrams)
    bad_scissor = BadScissor(bigrams)

    surf = None if args.no_time else default_surface(args.target_wpm, args.corpus)
    ref_card = surf.card(ref_lay) if surf is not None else None

    rows: dict[str, dict] = {}
    for name, lay in specs:
        layout = Layout(lay, ROW_STAGGERED_30)
        row: dict = {"layout": lay}

        if surf is not None:
            card = surf.card(
                lay,
                ref_total_ms=ref_card.total_ms,
                ref_ms_per_char=ref_card.ms_per_char,
            )
            row["time"] = {
                "ms_per_char": card.ms_per_char,
                "saved_vs_ref_pct": card.saved_vs_ref_pct,
                "coverage_pct": card.coverage_pct,
                # The raw-corpus-TOTALS convention is NOT a second saved% -- it is the
                # frozen artifacts' number, valid only at equal coverage. It appears
                # only inside a labelled reconciliation, never beside saved% as though
                # the two were co-equal (same contract as wfd_legacy_reconciliation).
                "raw_total_reconciliation": _raw_total_reconciliation(card, ref_card),
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
        # `kmstats` is the 11 keymeow-class statistics ALONE -- a named external convention,
        # and the historical JSON key, so it stays. `gauges` is the campaign's 15-gauge frame
        # (these 11 plus scissor/imbalance/oxey-style/comfort). Same values on the shared 11;
        # kept as two keys because they name two different things, and dropping the older one
        # would silently break any consumer reading row["kmstats"].
        row["kmstats"] = {name: gauges[name] for name in STAT_NAMES}

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
            # The `down` weight is an ORIENTATION term, and the served gauge has no
            # direction-of-travel channel to support one (every relational/geometric feature
            # is a function of the UNORDERED pair; direction enters only via the landing-key
            # one-hots). So `down` is a PRIOR, flagged as such -- not a measured effect. It
            # is also the only order-dependent part of this column: at down=1.5 the severity
            # differs between (a,b) and (b,a) on 24 of 900 pairs, and at down=1.0 on 0.
            "orientation_term": {
                "weight": DEFAULT_SEVERITY.down,
                "status": "PRIOR — not measured; the served gauge cannot represent direction",
                "share_without_it": severity.share(
                    layout,
                    SeverityWeights(
                        pinky=DEFAULT_SEVERITY.pinky,
                        ring_ratio=DEFAULT_SEVERITY.ring_ratio,
                        down=1.0,
                        support=DEFAULT_SEVERITY.support,
                    ),
                ),
            },
            "note": (
                "a declared preference, not a measurement; wide_support_share drops the "
                "column-adjacency gate (the only support where middle-pinky mass is visible); "
                "the `down` orientation weight is a PRIOR the served gauge cannot corroborate"
            ),
        }

        row["scissor_by_finger"] = scissor_fingers.shares(layout)
        row["scissor_by_finger_rule"] = "half-to-each-finger"
        if args.scissor_pairs:
            row["scissor_by_finger_pair"] = scissor_fingers.pair_shares(layout)
        row["redirects"] = redirects.shares(lay)

        # bad-scissor: the sibling `badscissor` agent's specification, implemented exactly.
        # A DIFFERENT support from `scissor` -- a cross-cut, not a superset -- and a
        # DIFFERENT denominator (space-excluded, the kmstats convention), so it is reported
        # in its own block rather than beside the flat scissor gauge.
        row["bad_scissor"] = {
            "share": bad_scissor.share(layout),
            "by_finger": bad_scissor.by_finger(layout),
            "by_cell": bad_scissor.by_cell(layout),
            "attribution_rule": BAD_SCISSOR_RULE,
            "denominator": "layout-restricted bigram mass, space-EXCLUDED (kmstats convention)",
            "severity": "flat (1.0 per qualifying bigram) -- the spec refuted a distance grading",
        }

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
            }
            # The frozen dominance artifacts' wfd came from a CORRUPT board (';' on dof 0,
            # the slot-0 character evicted, 'q' duplicated) -- a bug, not a second
            # convention. It is reported as a reconciliation aid ONLY, in its own block,
            # never beside row["community"]["wfd"] as though the two were co-equal gauges.
            row["wfd_legacy_reconciliation"] = _wfd_reconciliation(o2, lay)
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
            }
            row["wfd_legacy_reconciliation"] = None

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
                trigram_path=str(corpus_dir / "trigrams.txt"),
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

    if len(rows) != len(specs):  # pragma: no cover — the collision check above precludes it
        raise SystemExit(
            f"internal error: {len(specs)} layouts requested but {len(rows)} rows produced; "
            "refusing to emit a table with a dropped row"
        )

    if args.json:
        print(
            _json.dumps(
                {
                    "target_wpm": args.target_wpm,
                    "ref": ref_name,
                    # WHICH corpus produced every number below. `corpus` is the label and
                    # `corpus_provenance.sha256` is the fact -- a report that names only a
                    # default is how two corpora get stitched into one table.
                    "corpus": corpus_block["corpus"],
                    "corpus_provenance": corpus_block,
                    "skipgram_table": corpus_block["skipgram_table"],
                    "gauge_frame": (
                        "wscissor-allgauge: kmstats + oxey.pattern_shares + comfort/bigram-mass"
                    ),
                    # The community gauges (genkey/oxeylyzer1/oxeylyzer2/wfd) run on their
                    # own VENDORED corpora and do not move with --corpus; so do the fitted
                    # model surfaces' timing values, which are baked at 90 WPM. --corpus
                    # changes the frequency WEIGHTING of the objective, never a fitted model.
                    "corpus_sensitive": "gauges, time, redirects, scissor_*, bad_scissor",
                    "corpus_invariant": "community, community_primed (vendored corpora)",
                    "model_family": args.model_family,
                    "rows": rows,
                },
                indent=1,
            )
        )
        return 0

    _print_report(rows, ref_name, args, corpus_block)
    return 0


def _shared_corpora(
    corpus_dir: Path,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """One corpus's bigrams, ``1-skip31`` skipgrams and trigrams.

    ``1-skip31.txt`` (not ``1-skip.txt``) is the table the campaign's frozen gauge boards
    were computed on — see the module docstring. Pinning it here is what makes an
    iWeb-vs-blend comparison a corpus comparison rather than a skipgram-convention change:
    the two files are DIFFERENT tables in iWeb and the same table in blend-v1.
    """
    from keybo.data.corpus import PRODUCTION_SKIPGRAMS, load_frequencies

    return (
        load_frequencies(str(corpus_dir / "bigrams.txt")),
        load_frequencies(str(corpus_dir / PRODUCTION_SKIPGRAMS)),
        load_frequencies(str(corpus_dir / "trigrams.txt")),
    )


def _cell(value: float | None, width: int, spec: str = ".3f") -> str:
    """Right-aligned number, or N/A — never a silently-wrong number."""
    if value is None:
        return f"{NA:>{width}}"
    return f"{value:>{width}{spec}}"


def _print_report(
    rows: dict[str, dict],
    ref_name: str,
    args: argparse.Namespace,
    corpus_block: dict,
) -> None:
    names = list(rows)
    # Truncate raw 30-char layouts for column width HERE, never in the row key (see _display).
    # If two truncations would collide, lengthen them until they do not: an ambiguous LABEL is
    # only cosmetic, but it reads exactly like the dropped row this truncation used to cause.
    shown = {n: _display(n) for n in names}
    for cut in range(9, 31):
        if len(set(shown.values())) == len(names):
            break
        shown = {n: (n[:cut] + "…" if len(n) == 30 else n) for n in names}
    w = max(len(shown[n]) for n in names) + 2

    # WHICH corpus, first line of the report: every corpus-sensitive number below depends
    # on it, and the default changed (iWeb -> blend-v1) at CORPUS-SWAP-1.
    tri_sha = corpus_block["sha256"].get("trigrams.txt", "?")[:12]
    print(
        f"== corpus: {corpus_block['corpus']} "
        f"({corpus_block['path']}; skipgrams={corpus_block['skipgram_table']}; "
        f"trigrams.sha256={tri_sha}) =="
    )
    print(
        "   the community scores below are corpus-INVARIANT (vendored corpora); "
        "the fitted model surfaces are baked at 90 WPM and are not re-fit by a corpus change\n"
    )

    if not args.no_time:
        print(f"== predicted typing time (measured-keystroke surface; ref = {ref_name}) ==")
        print(f"{'layout':<{w}}{'ms/char':>9}{'saved%':>8}{'coverage%':>11}")
        for n in names:
            t = rows[n]["time"]
            saved = f"{t['saved_vs_ref_pct']:+.2f}" if t["saved_vs_ref_pct"] is not None else "-"
            print(f"{shown[n]:<{w}}{t['ms_per_char']:>9.2f}{saved:>8}{t['coverage_pct']:>11.1f}")
        print("saved% is the ms/char comparison — per CHARACTER TYPED, so a charset that")
        print("covers less of the corpus cannot be flattered by it. Rank and gate on it.")
        _print_raw_total_reconciliation(rows, names, w, shown, ref_name)
        print()

    print("== community scores (exact ports, native corpora) ==")
    print(f"{'layout':<{w}}{'genkey↓':>12}{'oxey1↑':>16}{'oxey2↑':>18}{'wfd↑':>18}")
    for n in names:
        c = rows[n]["community"]
        print(
            f"{shown[n]:<{w}}{_cell(c['genkey'], 12, '.2f')}"
            + "".join(_cell(c[k], width, ".0f") for k, width in (("oxeylyzer1", 16),))
            + _cell(c["oxeylyzer2"], 18, ".0f")
            + _cell(c["wfd"], 18, ".0f")
        )

    print("\n== community PRIMED (strain residual; the frozen all-gauge board's frame) ==")
    print(f"{'layout':<{w}}{'genkey′↓':>12}{'oxey1′↑':>16}{'oxey2′↑':>18}")
    for n in names:
        p = rows[n]["community_primed"]
        print(
            f"{shown[n]:<{w}}{_cell(p['genkey_primed'], 12, '.4f')}"
            + _cell(p["oxey1_primed"], 16, ".0f")
            + _cell(p["oxey2_primed"], 18, ".0f")
        )
    print("wfd is NOT primed away — it is the same one gauge printed above (higher better)")

    _print_wfd_reconciliation(rows, names, w, shown)

    print(f"\n== all-gauge frame, shared corpus (1-skip31); {len(GAUGE_NAMES)} gauges ==")
    print(f"{'layout':<{w}}" + "".join(f"{s:>11}" for s in GAUGE_NAMES))
    for n in names:
        g = rows[n]["gauges"]
        print(f"{shown[n]:<{w}}" + "".join(_cell(g[s], 11) for s in GAUGE_NAMES))

    print("\n== scissor by finger (% of layout-covered bigram mass; sums to `scissor`) ==")
    print(f"{'layout':<{w}}" + "".join(f"{f:>9}" for f in FINGER_NAMES) + f"{'total':>10}")
    for n in names:
        per_finger = rows[n]["scissor_by_finger"]
        print(
            f"{shown[n]:<{w}}"
            + "".join(f"{per_finger[f]:>9.4f}" for f in FINGER_NAMES)
            + f"{sum(per_finger.values()):>10.4f}"
        )
    print(f"attribution rule: {rows[names[0]]['scissor_by_finger_rule']} (an exact partition)")

    print("\n== scissor GRADED (a declared preference, not a measurement) ==")
    print(f"{'layout':<{w}}{'flat↓':>10}{'graded↓':>10}{'flat-ctl':>10}{'wide↓':>10}  {'weights'}")
    for n in names:
        graded = rows[n]["scissor_graded"]
        print(
            f"{shown[n]:<{w}}{rows[n]['gauges']['scissor']:>10.4f}{graded['share']:>10.4f}"
            f"{graded['flat_control']:>10.4f}{graded['wide_support_share']:>10.4f}"
            f"  {graded['weights']}"
        )
    print(
        "flat-ctl re-derives the flat gauge through the graded code path at all weights 1.0 — "
        "it must equal flat (a positive control that grading generalizes rather than replaces)"
    )
    print(
        "⚠ the `down=` orientation weight is a PRIOR, not a measured effect: the served gauge "
        "has NO direction-of-travel channel (every relational feature is a function of the "
        "unordered pair), so it cannot corroborate one. no-orientation column: "
        + "  ".join(
            f"{n} {rows[n]['scissor_graded']['orientation_term']['share_without_it']:.4f}"
            for n in names
        )
    )

    if args.scissor_pairs:
        print("\n== scissor by adjacent finger PAIR (a second exact partition) ==")
        keys = sorted({k for n in names for k in rows[n].get("scissor_by_finger_pair", {})})
        print(f"{'layout':<{w}}" + "".join(f"{k:>10}" for k in keys))
        for n in names:
            pairs = rows[n].get("scissor_by_finger_pair", {})
            print(f"{shown[n]:<{w}}" + "".join(f"{pairs.get(k, 0.0):>10.4f}" for k in keys))

    print(
        "\n== bad-scissor: lower key on a non-index finger (% of layout-covered NO-SPACE mass) =="
    )
    print(
        f"{'layout':<{w}}{'share↓':>9}{'dy1':>9}{'dy2':>9}"
        + "".join(f"{f:>9}" for f in BAD_SCISSOR_FINGERS)
    )
    for n in names:
        bad = rows[n]["bad_scissor"]
        cells = bad["by_cell"]
        dy1 = sum(v for k, v in cells.items() if k.endswith("dy1"))
        dy2 = sum(v for k, v in cells.items() if k.endswith("dy2"))
        print(
            f"{shown[n]:<{w}}{bad['share']:>9.4f}{dy1:>9.4f}{dy2:>9.4f}"
            + "".join(f"{bad['by_finger'][f]:>9.4f}" for f in BAD_SCISSOR_FINGERS)
        )
    print(
        f"attribution: {rows[names[0]]['bad_scissor']['attribution_rule']} "
        "(so both index columns are structurally 0); "
        f"denominator: {rows[names[0]]['bad_scissor']['denominator']}"
    )
    print(
        "a CROSS-CUT of `scissor`, not a superset: it drops the 12 narrow / 36 wide pairs whose "
        "lower key is the index's and adds 72 single-row descents neither gauge sees — so the "
        "dy2 column above is the only part the incumbent scissor gauges can price"
    )
    print(
        "⚠ POSTURE DIAGNOSTIC, NOT A SPEED PREDICTOR. (1) frequency-controlled, "
        "overlap-restricted effect is +0.41 ms [+0.23, +0.55]; bigram FREQUENCY explains more "
        "variance than any geometric axis. (2) the mid-board ordering is NOT robust — only "
        "'qwerty is worst' and 'lsb-sib < archive-1843' survive every weighting, so do not pick "
        "a winner on small differences. (3) most of the flagged mass sits on a few bottom keys "
        "(`c`/`x`), so this measures a few qwerty-era letter placements, not a structural law — a "
        "96.6% figure printed here until 2026-07-28 is withdrawn as a number: it reproduces in no "
        "shipped frame (qwerty is 7.6% on iWeb, 10.0% on blend-v1). The "
        "per-finger split says WHERE THE MASS SITS, not which finger is strained — that causal "
        "claim is not identified ON THE AALTO SAMPLE (there the two groups share no bottom-row "
        "key). That limit is EMPIRICAL, not structural: the geometry does admit the missing "
        "comparisons, so a corpus supplying them could identify it. (4) ⚠ THE keybo-lsb vs "
        "keybo-lsb+lm ORDERING ON THIS GAUGE IS A SUPPORT-BOUNDARY ARTIFACT, not a posture "
        "difference: 100% of their gap is dy=1 (their dy2 mass is EXACTLY equal), exactly ONE "
        "finger moves (R-pinky), and 16 bigrams flip flag status — all dy=1, all R-pinky. It "
        "reproduces on both iWeb (+0.407) and blend-v1 (+0.363), so it is corpus-robust. The cell "
        "that explains it (2-row, non-adjacent, weaker-finger-on-top) is one this gauge's own "
        "by_cell breakdown CANNOT express, and it moves 1.48x the penalty in the OPPOSITE "
        "direction. Do not read this pair's ordering here as a posture claim."
    )

    print("\n== redirect family, oxeylyzer-1 classes (% of layout-covered trigram mass) ==")
    print(
        f"{'layout':<{w}}"
        + "".join(f"{c:>19}" for c in REDIRECT_CLASSES)
        + f"{'bad-total':>12}{'family':>10}"
    )
    for n in names:
        r = rows[n]["redirects"]
        print(
            f"{shown[n]:<{w}}"
            + "".join(f"{r[c]:>19.4f}" for c in REDIRECT_CLASSES)
            + f"{r['bad_redirects_total']:>12.4f}{r['redirects_family_total']:>10.4f}"
        )
    print("(the four classes are mutually exclusive; family total == the `redir` gauge above)")

    _print_model_scores(rows, names, w, shown)

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


def _print_raw_total_reconciliation(
    rows: dict[str, dict], names: list[str], w: int, shown: dict[str, str], ref_name: str
) -> None:
    """Reconcile the frozen artifacts' raw-TOTALS saved% against the rankable one.

    Printed only when a coverage difference makes the two conventions actually diverge —
    on an equal-coverage cohort they are the same number and a second block would be
    noise. Deliberately NOT a second ``saved%`` column: see
    :func:`_raw_total_reconciliation`.
    """
    live = [
        n
        for n in names
        if (rec := rows[n]["time"] and rows[n]["time"].get("raw_total_reconciliation"))
        and not rec["equal_coverage"]
    ]
    if not live:
        return
    print(
        "\n-- saved%: frozen-artifact reconciliation "
        "(raw corpus TOTALS — not comparable across charsets) --"
    )
    print(
        f"{'layout':<{w}}{'saved% (rankable)':>19}{'raw totals':>13}{'delta':>9}{'coverage%':>11}"
    )
    for n in live:
        t = rows[n]["time"]
        rec = t["raw_total_reconciliation"]
        print(
            f"{shown[n]:<{w}}{t['saved_vs_ref_pct']:>+19.2f}"
            f"{rec['raw_total_saved_vs_ref_pct']:>+13.2f}{rec['delta']:>+9.2f}"
            f"{rec['coverage_pct']:>11.1f}"
        )
    print(
        f"the raw-totals column divides each layout's total by {ref_name}'s, but each total is\n"
        "summed over a DIFFERENT corpus subset — a layout that types more of the corpus is\n"
        "charged for the extra mass, so the number can call a faster layout slower. Quote it\n"
        "only to reconcile a frozen artifact; rank and gate on saved% above."
    )


def _print_wfd_reconciliation(
    rows: dict[str, dict], names: list[str], w: int, shown: dict[str, str]
) -> None:
    """Reconcile the frozen artifacts' wfd against the correct one — as a decomposed delta.

    Deliberately NOT a second gauge column: the legacy number is not another way of
    measuring the layout, it is the same measurement taken on a board that cannot exist.
    Printing it as ``correct + delta``, with the corruption named, is what keeps a reader
    from stitching a comparison across the two (the failure this whole block exists for).
    """
    live = [n for n in names if rows[n].get("wfd_legacy_reconciliation")]
    if not live:
        return
    print("\n== wfd: frozen-artifact reconciliation (the legacy board is a BUG, not a frame) ==")
    print(f"{'layout':<{w}}{'wfd↑ (correct)':>22}{'legacy board':>22}{'delta':>20}{'delta%':>9}")
    for n in live:
        r = rows[n]["wfd_legacy_reconciliation"]
        print(
            f"{shown[n]:<{w}}{r['correct_wfd']:>22,}{r['legacy_board_wfd']:>22,}"
            f"{r['delta']:>+20,}{r['delta_pct_of_correct']:>+8.2f}%"
        )
    for n in live:
        r = rows[n]["wfd_legacy_reconciliation"]
        # quote each character: 'evicted=-' would otherwise read as "evicted: none" on
        # qwerty30m, whose evicted character genuinely IS '-'.
        evicted = ",".join(repr(c) for c in r["evicted_characters"]) or "none"
        duplicated = ",".join(repr(c) for c in r["duplicated_characters"]) or "none"
        print(
            f"  {shown[n]:<{w}} legacy board {r['legacy_board']!r}"
            f"  evicted={evicted} duplicated={duplicated}"
        )
    print(
        "the legacy board puts ';' on dof 0 (top-left, left pinky), evicts the character on\n"
        "slot 0, and duplicates 'q' — so it is not a permutation of the 31 keys. Quote it\n"
        "only to reconcile a frozen artifact; rank and gate on the correct wfd above."
    )


def _print_model_scores(
    rows: dict[str, dict], names: list[str], w: int, shown: dict[str, str]
) -> None:
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
                f"{shown[n]:<{w}}"
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
        print(f"{shown[n]:<{w}}" + "".join(f"{c:>30}" for c in cells))
    print(f"note: {first['wpm_note']}")
