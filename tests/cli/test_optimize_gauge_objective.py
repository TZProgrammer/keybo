"""`optimize --gauge-objective`: search the ruler the campaign REPORTS on, opt-in only.

WHAT DEFECT THIS PREVENTS (SEARCHPARAMS-1 / NORMOPT-1, both 2026-08-01, independently):
``optimize``'s default objective is the BIGRAM table, but every published number is
``analyze``'s ms/char = the K31 time surface's ``T2[a,b] + Tcond[a,b,c]`` over the TRIGRAM
corpus. Those two rank layouts INVERTED — spearman 0.672, and the selection tax at the
argmin is 4.97 resolution floors — because the cubic term carries the gauge's variance (sd
0.803 vs 0.274). So the search optimizes one ruler and the report grades on another, and no
restart budget closes the gap. This file pins the opt-in fix and, more importantly, the four
ways it could go wrong:

1. **A default change.** The whole value of the flag is that it is opt-in; a regression that
   made the trigram term default would silently invalidate every frozen board in
   PREREGISTRATIONS.md. ``test_default_invocation_still_builds_the_bigram_table_scorer``
   pins the default path to ``TableBigramScorer``, by type.
2. **Silent fallback.** This repo's most repeated defect shape is ``present != effective`` —
   a flag accepted and then ignored (see ``--compiler-opt-level``, ``1-skip.txt``,
   ``target_space='lograt'``). Every unsupported combination here must RAISE, so the refusal
   tests assert on ``SystemExit``, not on a fitness value.
3. **An objective that is not the gauge.** The naive reading — ``TableBigramScorer`` (on
   ``bigrams.txt``) + ``TableTrigramScorer`` (on one seed) — is off by rel 1.5e-2 on C30M,
   ~11 resolution floors, because (a) the gauge weights the bigram term by the TRIGRAM
   corpus's first-two-character marginal, not by ``bigrams.txt`` (kept mass 887,147,352 vs
   913,956,722), and (b) the gauge is the 3-seed MEAN of the tables, which no single-model
   scorer can reach. The parity tests tie the shipped objective to ``analyze``'s own number.
4. **Selecting on the wrong ruler.** ``--attempts`` keeps the best of N; if the search ran on
   the gauge but selection compared bigram fitness, the tax would survive the fix. The
   reporting scorer must BE the search scorer here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring import model_norm as MN
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.scoring.table_trigram import TableTrigramScorer

#: The charset every K31-surface board is a permutation of (``model_norm.S.C30M``).
C30M = MN.S.C30M

#: Boards the parity gate reconciles on: the reference charset plus every named layout that
#: is a permutation of it. ``qwerty``/``dvorak``/``colemak`` pin ``;``+``/`` instead of
#: ``'``+``-``, so they are a DIFFERENT charset and the table would exclude different corpus
#: rows — parity there would be comparing two objectives, not two implementations.
PARITY_BOARDS = {
    "C30M": C30M,
    "graphite": NAMED_LAYOUTS["graphite"],
    "semimak": NAMED_LAYOUTS["semimak"],
}


def _base_args(**overrides) -> SimpleNamespace:
    """A namespace with exactly the fields ``optimize.run`` reads, defaults as shipped.

    Built by hand rather than through the parser because these tests assert on WHICH SCORER
    the wiring builds; two shipped tests in this directory already use this idiom.
    """
    args = SimpleNamespace(
        attempts=1,
        out=None,
        seed=0,
        alpha=0.999,
        max_outer=None,
        no_local_search=False,
        no_progress=True,
        start=NAMED_LAYOUTS["qwerty"],
        ngram="bigram",
        no_table=False,
        comfort_weight=0.0,
        comfort_config=None,
        finger_load_weight=0.0,
        oxey_weight=0.0,
        model_weight=None,
        model_anchors=None,
        gauge_objective=False,
        target_wpm=90.0,
        corpus=None,
        bigram_freqs=None,
        trigram_freqs=None,
        model="unused",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _capture_search_scorer(monkeypatch, args):
    """Run ``optimize.run`` with the search itself stubbed out, returning the scorers built.

    Stubs ``_one_attempt`` (the expensive part) and the postflight, so what is measured is
    the SCORER-SELECTION wiring — the thing every test in this file is about — at unit speed.
    """
    from keybo.cli import optimize

    seen = {}

    # `incumbents` is accepted (and recorded) because the gauge path forwards it, exactly as the
    # blend and default paths do. It was ORIGINALLY omitted here, and that spy signature is what
    # hid a real defect once the --gauge-objective and --polish-incumbent branches were combined:
    # the production call dropped the argument, so --polish-incumbent was SILENTLY INERT under
    # --gauge-objective. A spy that accepts fewer arguments than the real callee can only ever pin
    # the narrower behaviour.
    def fake_run_search(inner_args, scorer, search_scorer, incumbents=()):
        seen["scorer"] = scorer
        seen["search_scorer"] = search_scorer
        seen["args"] = inner_args
        seen["incumbents"] = incumbents
        return 0

    monkeypatch.setattr(optimize, "_run_search", fake_run_search)
    rc = optimize.run(args)
    assert rc == 0, "the stubbed run must reach _run_search"
    return seen


# --------------------------------------------------------------------------------------
# 1. THE DEFAULT IS UNCHANGED — the one thing this change is not allowed to touch.
# --------------------------------------------------------------------------------------


def test_default_invocation_still_builds_the_bigram_table_scorer(monkeypatch):
    """No flag => ``TableBigramScorer``, exactly as before ``--gauge-objective`` existed.

    Every frozen board in PREREGISTRATIONS.md was produced by this path. If adding an opt-in
    objective moved the default, those boards would no longer be reproducible from the CLI
    that claims to produce them.
    """
    monkeypatch.setattr(
        "keybo.cli.optimize.build_scorer", lambda _args: _StubScorer(), raising=True
    )
    monkeypatch.setattr(
        "keybo.models.xgboost_model.XGBoostTypingModel.load",
        staticmethod(lambda _path: _StubModel()),
        raising=True,
    )
    monkeypatch.setattr("keybo.cli.optimize.load_freqs", lambda _args: {"th": 100, "he": 90})
    seen = _capture_search_scorer(monkeypatch, _base_args())
    assert isinstance(seen["search_scorer"], TableBigramScorer)
    assert not isinstance(seen["search_scorer"], TableTrigramScorer)


def test_gauge_objective_defaults_to_off_in_the_argument_parser():
    """The flag's parser default is ``False`` — the default invocation cannot opt in by accident."""
    import argparse

    from keybo.cli import optimize

    parser = argparse.ArgumentParser()
    optimize.add_arguments(parser)
    defaults = parser.parse_args(["--model", "m.json"])
    assert defaults.gauge_objective is False


# --------------------------------------------------------------------------------------
# 2. THE OPT-IN PATH SELECTS THE GAUGE OBJECTIVE — and uses it for BOTH roles.
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_gauge_objective_builds_a_trigram_table_scorer(monkeypatch):
    """``--gauge-objective`` searches the cubic objective, not the bigram table.

    Pinned by type: ``TableTrigramScorer`` is the evaluator whose table is indexed by a
    position TRIPLE, so its presence is what proves the cubic term is in the objective at all.
    """
    seen = _capture_search_scorer(monkeypatch, _base_args(gauge_objective=True, start=C30M))
    assert isinstance(seen["search_scorer"], TableTrigramScorer)


@pytest.mark.slow
def test_gauge_objective_reports_on_the_same_scorer_it_searches(monkeypatch):
    """Search scorer IS the reporting scorer, so best-of-N SELECTS on the gauge.

    The measured selection tax (4.97 floors at the argmin) comes from choosing among
    candidates with a ruler that disagrees with the report. Searching the gauge but ranking
    attempts by a different objective would reintroduce exactly that tax, and every printed
    fitness would be off the gauge it claims to optimize.
    """
    seen = _capture_search_scorer(
        monkeypatch, _base_args(gauge_objective=True, start=C30M, attempts=4)
    )
    assert seen["scorer"] is seen["search_scorer"]


@pytest.mark.slow
def test_gauge_objective_evaluates_hundreds_of_times_faster_than_the_analyzer(monkeypatch):
    """The opt-in objective must be a SEARCH objective, not just a correct one.

    ``TimeSurface.card()`` is ~50 ms per layout; simulated annealing evaluates millions of
    times, so a correct-but-slow gauge would be unusable and the flag would be a trap. The
    table path is ~0.2 ms. A 20x floor (measured margin is ~270x) keeps this robust on a
    loaded box while still failing if the implementation ever regresses to scoring through
    the analyzer.
    """
    import time

    from keybo.analysis.timecard import default_surface

    seen = _capture_search_scorer(monkeypatch, _base_args(gauge_objective=True, start=C30M))
    scorer = seen["search_scorer"]
    layout = Layout(C30M, ROW_STAGGERED_30)
    perm = scorer.permutation(layout)
    scorer.fitness_of_permutation(perm)  # warm

    start = time.perf_counter()
    for _ in range(200):
        scorer.fitness_of_permutation(perm)
    table_sec = (time.perf_counter() - start) / 200

    surface = default_surface(90.0)
    surface.card(C30M)  # warm
    start = time.perf_counter()
    surface.card(C30M)
    card_sec = time.perf_counter() - start

    assert table_sec * 20 < card_sec, (
        f"gauge table eval {table_sec * 1e3:.3f} ms vs analyzer card {card_sec * 1e3:.1f} ms — "
        "the opt-in objective is not fast enough to search with"
    )


# --------------------------------------------------------------------------------------
# 3. PARITY WITH `analyze` — the objective must BE the published gauge.
# --------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("board", sorted(PARITY_BOARDS))
def test_gauge_search_scorer_reconciles_to_the_analyzer_ms_per_char(board):
    """The opt-in objective reproduces ``analyze``'s ms/char to floating-point noise.

    This is the gate the parent required: an objective that cannot be tied to the published
    gauge must not ship. ``rel=1e-12`` is ~100x looser than the measured worst case (1.2e-14
    over six boards) so the gate cannot flake on summation order, but it is still 10 orders of
    magnitude tighter than the naive ``bigrams.txt``-weighted reading (1.5e-2) it must reject.
    """
    from keybo.analysis.timecard import default_surface, gauge_search_scorer

    layout_string = PARITY_BOARDS[board]
    scorer = gauge_search_scorer(chars=layout_string, target_wpm=90.0)
    card = default_surface(90.0).card(layout_string)

    layout = Layout(layout_string, ROW_STAGGERED_30)
    assert scorer.fitness(layout) == pytest.approx(card.total_ms, rel=1e-12)
    assert scorer.ms_per_char(layout) == pytest.approx(card.ms_per_char, rel=1e-12)


@pytest.mark.slow
def test_gauge_scorer_tracks_the_analyzer_across_permutations_not_just_one_board():
    """Parity holds under the moves the SEARCH makes, so it is not a one-board coincidence.

    A scorer could match on a reference board and drift on the permutations annealing
    actually explores (e.g. by mis-indexing space, whose slot is pinned last). Scoring random
    transpositions of C30M ties the two implementations over the search space itself.
    ``assert_discriminating`` first: if every sampled board scored the same, the agreement
    below would be vacuous.
    """
    from keybo.analysis.timecard import default_surface, gauge_search_scorer
    from keybo.testkit import assert_discriminating

    scorer = gauge_search_scorer(chars=C30M, target_wpm=90.0)
    surface = default_surface(90.0)
    rng = np.random.default_rng(20260801)

    mine, theirs = [], []
    for _ in range(8):
        chars = list(C30M)
        i, j = rng.choice(len(chars), size=2, replace=False)
        chars[i], chars[j] = chars[j], chars[i]
        board = "".join(chars)
        mine.append(scorer.fitness(Layout(board, ROW_STAGGERED_30)))
        theirs.append(surface.card(board).total_ms)

    assert_discriminating(theirs, "analyzer totals over sampled transpositions")
    for got, want in zip(mine, theirs, strict=True):
        assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.slow
def test_gauge_objective_is_not_the_naive_sum_of_the_two_shipped_table_scorers():
    """Pins the DEFECT the parity gate exists to catch, so parity cannot be met by accident.

    Wiring the two shipped table scorers together — the obvious reading of "add the trigram
    term" — is off by ~1.5e-2 relative (~11 resolution floors of 0.135 ms/char), because the
    gauge weights its bigram term by the trigram corpus's first-two-character marginal and
    averages three model seeds. This test asserts the naive sum really is that far off, so
    that if someone later "simplifies" the implementation into it, the parity tests above
    fail for a reason this comment explains rather than mysteriously.
    """
    from keybo.analysis.timecard import _load_gz_model, default_surface
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    surface = default_surface(90.0)
    bigram_freqs = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
    naive = TableBigramScorer(
        _load_gz_model("bigram_reg31_seed0"), bigram_freqs, target_wpm=90.0, chars=C30M
    ).fitness(Layout(C30M, ROW_STAGGERED_30)) + TableTrigramScorer(
        _load_gz_model("trigram_cond31_seed0"), surface.tri, target_wpm=90.0, chars=C30M
    ).fitness(Layout(C30M, ROW_STAGGERED_30))

    total = surface.card(C30M).total_ms
    relative_error = abs(naive - total) / total
    assert relative_error > 1e-3, (
        "the naive two-scorer sum now matches the gauge; if the gauge's definition changed, "
        "this test and the parity gate above must be re-derived together"
    )


# --------------------------------------------------------------------------------------
# 4. REFUSAL, NEVER SILENT FALLBACK — `present != effective` is this repo's top defect shape.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"no_table": True}, "--no-table"),
        ({"ngram": "trigram"}, "--ngram"),
        ({"comfort_weight": 5.0}, "--comfort-weight"),
        ({"finger_load_weight": 50.0}, "--finger-load-weight"),
        ({"oxey_weight": 1.0}, "--oxey-weight"),
        ({"model_weight": ["aalto-n=1.0"]}, "--model-weight"),
    ],
)
def test_gauge_objective_refuses_unsupported_combinations(overrides, expected):
    """Each unsupported combination raises and NAMES the conflicting flag.

    Silent fallback is the failure mode to avoid: a user who passes
    ``--gauge-objective --no-table`` and gets a bigram search would publish numbers labelled
    with an objective the run never used. The message must name the flag so the refusal is
    actionable rather than merely loud.
    """
    from keybo.cli import optimize

    args = _base_args(gauge_objective=True, start=C30M, **overrides)
    with pytest.raises(SystemExit) as excinfo:
        optimize.run(args)
    message = str(excinfo.value)
    assert "--gauge-objective" in message
    assert expected in message


def test_gauge_objective_refuses_a_start_layout_off_the_surfaces_charset():
    """A non-C30M start is refused, not silently scored against a mismatched table.

    The K31 surface is indexed by the 31 geometric positions, but the corpus rows the table
    keeps are chosen by CHARSET: a ``qwerty``-charset start (``;`` and ``/`` instead of ``'``
    and ``-``) excludes different corpus mass, so its total is not comparable to any published
    ms/char. The refusal must happen before the search, not after it.
    """
    from keybo.cli import optimize

    args = _base_args(gauge_objective=True, start=NAMED_LAYOUTS["qwerty"])
    with pytest.raises(SystemExit) as excinfo:
        optimize.run(args)
    assert "--gauge-objective" in str(excinfo.value)


@pytest.mark.slow
def test_gauge_objective_records_its_objective_and_parity_in_the_result_file(monkeypatch, tmp_path):
    """The ``--out`` JSON names the objective and carries its parity deviation.

    ``--ngram bigram`` is recorded for every run, so a gauge run whose result file said only
    ``"ngram": "bigram"`` would be indistinguishable from a default run — the same
    unreproducible-artifact failure ``--model-weight`` already guards against by recording
    ``objective``. The parity number travels with the result so a reader can see the objective
    was tied to the gauge at run time, not merely claimed to be.
    """
    from keybo.cli import optimize

    out = tmp_path / "gauge.json"
    args = _base_args(gauge_objective=True, start=C30M, out=str(out), attempts=1)
    monkeypatch.setattr(
        optimize,
        "_one_attempt",
        lambda _args, _scorer, seed: Layout(C30M, ROW_STAGGERED_30),
    )
    assert optimize.run(args) == 0

    result = json.loads(out.read_text())
    assert "gauge" in result["objective"]
    assert result["gauge_parity_rel_dev"] < 1e-12
    assert result["ms_per_char"] == pytest.approx(
        __import__("keybo.analysis.timecard", fromlist=["x"])
        .default_surface(90.0)
        .card(C30M)
        .ms_per_char,
        rel=1e-12,
    )


class _StubScorer:
    """Minimal ``IScorer``: the default-path test asserts on TYPE, so fitness is irrelevant."""

    @staticmethod
    def fitness(_layout):
        return 0.0


class _StubModel:
    """Stands in for the XGBoost model the default fast path loads from ``--model``."""

    metadata = SimpleNamespace(ngram="bigram", wpm_range=(60.0, 120.0), extra={})

    @staticmethod
    def predict_ms(vectors):
        return np.zeros(len(vectors))
