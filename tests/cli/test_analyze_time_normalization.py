"""`keybo analyze`'s time card must commit to ONE normalization, and label the other.

The card reports two quantities computed on **different denominators**
(:mod:`keybo.analysis.timecard`)::

    ms_per_char      = total_ms / covered_mass          # per-CHARACTER rate
    saved_vs_ref_pct = (ref_total - total) / ref_total   # RAW corpus TOTALS

Raw totals are only comparable between layouts that cover the *same* corpus mass. A
layout whose charset types **more** of the corpus accumulates a larger ``total_ms`` for
that reason alone, so ``saved_vs_ref_pct`` charges it for the extra coverage and reads
as *slower* — while ``ms_per_char``, on the same row of the same table, reads *faster*.

That is the same failure shape as the legacy-wfd board (a number taken on a frame that
cannot be compared), and it gets the same treatment here: the raw-total quantity is
retained for reconciling frozen artifacts, but it is **named**, **not co-equal with the
rankable column**, and carries the exact delta to the coverage-normalized quantity — so
no reader and no script can rank on it by accident.

Every control NAMES its corpus, for the same reason ``tests/cli/test_analyze_allgauge.py``
does: taking the default would turn these into assertions about whatever the default is.
Both shipped corpora are exercised, and that is load-bearing rather than belt-and-braces —
the defect presents *differently* on each. On ``blend-v1`` (the production default)
``graphite`` reported a NEGATIVE saved% while being 5.5 ms/char faster than the reference
(an outright sign contradiction); on ``iweb`` no sign flips, but the two columns still
**rank** the cohort differently. A test that ran only on iWeb would have missed the sign
flip, so the rank agreement — which is violated on both — is the invariant asserted here.
"""

from __future__ import annotations

import json

import pytest

from keybo.cli.__main__ import main
from keybo.data.corpus import IWEB, PRODUCTION_DEFAULT

#: Both shipped corpora, NAMED. See the module docstring for why both are load-bearing.
BOTH_CORPORA = [PRODUCTION_DEFAULT, IWEB]

#: A charset-DIVERGENT cohort: qwerty/colemak carry ``;`` while semimak/graphite carry
#: ``-``, so their corpus coverage differs and the two conventions come apart. Every
#: layout in the shipped ``docs/analyzer.md`` example has *identical* coverage (92.5%),
#: which is exactly why the defect was invisible there.
MIXED_CHARSET_COHORT = ["qwerty", "colemak", "dvorak", "semimak", "graphite"]


def _run(capsys, argv: list[str], corpus: str = IWEB) -> dict:
    """Run `analyze` with the corpus NAMED (see the module docstring)."""
    argv = [argv[0], "--corpus", corpus, *argv[1:]]
    rc = main(argv)
    assert rc == 0, f"analyze {argv} returned {rc}"
    out = json.loads(capsys.readouterr().out)
    if "--json" in argv:
        assert out["corpus"] == corpus, f"must run on {corpus!r}, got {out['corpus']!r}"
    return out


@pytest.mark.slow
@pytest.mark.parametrize("corpus", BOTH_CORPORA)
def test_the_time_card_ranks_the_same_cohort_the_same_way_in_both_columns(capsys, corpus):
    """THE DEFECT: two columns of one table rank the same cohort differently.

    This is the regression test proper, and it asserts the invariant that fails on BOTH
    corpora. ``ms_per_char`` (lower faster) and the saved-percent (higher faster) are two
    presentations of one comparison, so they must induce the same order.

    Before the fix, on both corpora: ``colemak`` was 2nd by saved% and 4th by ms/char, in
    the same table. On ``blend-v1`` it degenerated further into a sign contradiction —
    ``graphite`` reported ``saved% = -0.26``, i.e. *slower than qwerty*, while its
    ``ms_per_char`` of 258.17 vs qwerty's 263.71 made it 5.5 ms/char FASTER.
    """
    out = _run(capsys, ["analyze", *MIXED_CHARSET_COHORT, "--ref", "qwerty", "--json"], corpus)
    time = {name: out["rows"][name]["time"] for name in MIXED_CHARSET_COHORT}

    # the cohort must genuinely exercise the defect, else this test proves nothing
    assert len({t["coverage_pct"] for t in time.values()}) > 1, (
        "cohort no longer has divergent coverage, so it cannot detect the defect"
    )

    by_saved = sorted(MIXED_CHARSET_COHORT, key=lambda n: -time[n]["saved_vs_ref_pct"])
    by_rate = sorted(MIXED_CHARSET_COHORT, key=lambda n: time[n]["ms_per_char"])
    assert by_saved == by_rate, (
        f"[{corpus}] the time card contradicts itself: ranked by saved% "
        f"{'>'.join(by_saved)} but by ms/char {'>'.join(by_rate)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("corpus", BOTH_CORPORA)
def test_no_layout_is_faster_by_one_column_and_slower_by_the_other(capsys, corpus):
    """The sharpest form of the defect: an outright SIGN contradiction.

    Observed on ``blend-v1`` before the fix (graphite). Kept separate from the rank test
    because it is the form a casual reader actually gets burned by — a minus sign on a
    layout that is in fact faster.
    """
    out = _run(capsys, ["analyze", *MIXED_CHARSET_COHORT, "--ref", "qwerty", "--json"], corpus)
    time = {name: out["rows"][name]["time"] for name in MIXED_CHARSET_COHORT}
    ref_mspc = time["qwerty"]["ms_per_char"]

    for name, t in time.items():
        faster_by_rate = t["ms_per_char"] < ref_mspc
        faster_by_saved = t["saved_vs_ref_pct"] > 0.0
        assert faster_by_rate == faster_by_saved, (
            f"[{corpus}] {name}: ms_per_char={t['ms_per_char']:.4f} vs ref "
            f"{ref_mspc:.4f} says faster={faster_by_rate}, but saved_vs_ref_pct="
            f"{t['saved_vs_ref_pct']:+.4f} says faster={faster_by_saved}"
        )


@pytest.mark.slow
def test_the_rankable_saved_percent_is_coverage_normalized_and_exact(capsys):
    """The rankable quantity is the per-CHARACTER comparison, and it reconciles exactly.

    Pins the arithmetic rather than a magic number: the shipped saved-percent must be
    derivable from the two ``ms_per_char`` values in the same table. A bit-exact
    numerator says nothing about the quotient, so the quotient is what gets asserted.
    """
    out = _run(capsys, ["analyze", *MIXED_CHARSET_COHORT, "--ref", "qwerty", "--json"])
    time = {name: out["rows"][name]["time"] for name in MIXED_CHARSET_COHORT}
    ref_mspc = time["qwerty"]["ms_per_char"]

    assert time["qwerty"]["saved_vs_ref_pct"] == pytest.approx(0.0), (
        "the reference layout saves 0% against itself by construction"
    )
    for name, t in time.items():
        expected = 100.0 * (ref_mspc - t["ms_per_char"]) / ref_mspc
        assert t["saved_vs_ref_pct"] == pytest.approx(expected, rel=1e-12), (
            f"{name}: saved% must be the coverage-normalized (ms/char) comparison"
        )


@pytest.mark.slow
def test_the_raw_total_convention_is_never_printed_without_its_label(capsys):
    """The PIN: the two conventions can never again be printed as co-equal columns.

    The raw-total number stays available for reconciling frozen artifacts, but only
    inside a block that (a) names it as the non-comparable convention, (b) carries the
    exact delta to the rankable one, and (c) tells the reader which to rank on. This is
    the same contract ``test_wfd_is_one_gauge_plus_a_labelled_legacy_reconciliation``
    pins for the legacy wfd board.
    """
    out = _run(capsys, ["analyze", *MIXED_CHARSET_COHORT, "--ref", "qwerty", "--json"])

    for name in MIXED_CHARSET_COHORT:
        rec = out["rows"][name]["time"]["raw_total_reconciliation"]
        assert rec is not None, f"{name}: the raw-total convention must be disclosed"
        # it is NOT a second saved% column: it is labelled and reconciled
        assert rec["comparable_across_charsets"] is False
        assert rec["delta"] == pytest.approx(
            rec["raw_total_saved_vs_ref_pct"] - out["rows"][name]["time"]["saved_vs_ref_pct"]
        ), f"{name}: the reconciliation delta must close exactly"
        assert "coverage" in rec["why_not_comparable"].lower()
        assert rec["rank_on"] == "saved_vs_ref_pct"


@pytest.mark.slow
def test_the_text_report_labels_the_raw_total_block_as_not_rankable(capsys):
    """A reader of the TEXT report must be told which column to rank on.

    The JSON contract above is machine-facing; this is the human-facing half. The
    defect was a *printed* contradiction, so the print frame is part of the fix.
    """
    argv = ["analyze", "--corpus", IWEB, *MIXED_CHARSET_COHORT, "--ref", "qwerty"]
    assert main(argv) == 0
    text = capsys.readouterr().out

    assert "raw corpus TOTALS" in text, "the non-comparable convention must be named"
    assert "not comparable across charsets" in text
    assert "rank and gate on" in text, "the reader must be told which column to rank on"
    # the rankable column is labelled as normalized where it is printed
    assert "saved%" in text


@pytest.mark.slow
def test_the_cohort_entry_point_wires_the_reference_rate_through():
    """``TimeSurface.cards()`` must be mixed-charset-safe without the caller's help.

    ``card(ref_total_ms=...)`` alone cannot distinguish the two conventions at unequal
    coverage — a bare total is only a well-defined reference when coverage matches. So the
    cohort entry point threads BOTH reference quantities, and this pins that it does:
    ``cards()`` must agree with the coverage-normalized comparison, not the raw-total one.
    """
    from keybo.analysis.timecard import default_surface
    from keybo.layouts import NAMED_LAYOUTS

    surf = default_surface(90.0)
    cohort = {n: NAMED_LAYOUTS[n] for n in MIXED_CHARSET_COHORT}
    cards = surf.cards(cohort, NAMED_LAYOUTS["qwerty"])
    ref_rate = cards["qwerty"].ms_per_char

    for name, card in cards.items():
        expected = 100.0 * (ref_rate - card.ms_per_char) / ref_rate
        assert card.saved_vs_ref_pct == pytest.approx(expected, rel=1e-12), name
    # and the cohort genuinely spans charsets, else the check is vacuous
    assert len({c.coverage_pct for c in cards.values()}) > 1


@pytest.mark.slow
def test_same_charset_layouts_are_unaffected_by_the_normalization(capsys):
    """CONTROL: when coverage is equal, both conventions agree identically.

    This is what makes the fix safe for the frozen reconciliation path: every frozen
    board this repo pins compares same-charset layouts, where the raw-total and
    normalized comparisons are algebraically the same number.
    """
    out = _run(capsys, ["analyze", "qwerty30m", "keybo-c30m", "--ref", "qwerty30m", "--json"])
    rows = out["rows"]
    assert (
        rows["qwerty30m"]["time"]["coverage_pct"] == rows["keybo-c30m"]["time"]["coverage_pct"]
    ), "control requires equal coverage"

    for name in ("qwerty30m", "keybo-c30m"):
        t = rows[name]["time"]
        rec = t["raw_total_reconciliation"]
        # equal coverage => the two conventions coincide to float precision
        assert rec["raw_total_saved_vs_ref_pct"] == pytest.approx(
            t["saved_vs_ref_pct"], abs=1e-9
        ), f"{name}: conventions must coincide at equal coverage"
        assert rec["delta"] == pytest.approx(0.0, abs=1e-9)
