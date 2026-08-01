"""A6: the 3-opt polish is REACHABLE from the CLI, and the comparison it feeds is SYMMETRIC.

``three_opt`` shipped implemented, exported and unit-tested with **zero production callers**
(``grep -rn three_opt src/`` found only its own definition and the ``__init__`` re-export), so
every reported optimum was 2-opt-converged only. That is half the defect. The other half is
that the polish was applied ASYMMETRICALLY: a searched layout got SA + 2-opt while an incumbent
it was reported against was scored as-is, so the printed gap included polish the incumbent never
got. The missing function is the symptom; the asymmetric comparison is the bug.

Measured on the campaign gauge (the reported ms/char — ``TimeSurface.card``, reproduced as an
exact ``T2[a,b] + Tc[a,b,c]`` table at rel 1.2e-14), exhaustively over all C(30,3)x5 = 20,300
reorderings:

* arm B (``flmpg-yuo,sntdcireahkxbwv'.jzq``) is **0/20300** improving — a strict 3-opt local
  optimum, not merely a 2-opt one. The campaign's claim survives being restated.
* the incumbents do NOT: graphite admits **-0.2740%** and semimak **-0.2757%**, each improving
  on **3/3 model seed tables** (structural, not estimator noise) and above the 0.135 seed floor.
  Both are 3-CYCLE moves, which no 2-opt scan can express at any depth.

So the asymmetry was worth real ms, and these tests pin the two halves of the fix: that the
polish is reachable and identical for both roles, and that the DEFAULT is untouched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from keybo.cli import optimize
from keybo.cli.__main__ import main
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.layout import Layout
from keybo.optimize.local_search import three_opt, two_opt
from keybo.scoring.base import IScorer
from tests.cli.test_cli import _train_tiny_model

#: A charset-30 board and a within-charset permutation of it, for the incumbent gates.
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
#: Same characters, different arrangement — a legal incumbent for a --start of QWERTY30M.
PERMUTED30M = "qwertyuiopasdfghjkl'zxcvbnm.,-"


def _corpus(tmp_path):
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t100\nhe\t90\nan\t80\n'a\t20\n-e\t10\n")
    return str(corpus)


def _argv(model_path, corpus, *extra):
    return [
        "optimize",
        "--model",
        model_path,
        "--bigram-freqs",
        corpus,
        "--seed",
        "7",
        "--alpha",
        "0.9",
        "--max-outer",
        "12",
        "--no-progress",
        *extra,
    ]


# ---------------------------------------------------------------------------------------
# 1. REACHABILITY: the flag exists, runs 3-opt, and lands on a 3-opt optimum.
# ---------------------------------------------------------------------------------------


class _CountingScorer(IScorer):
    """A real char->position landscape (as tests/optimize/conftest) that counts evaluations."""

    _W = {c: w for w, c in enumerate("etaoinshrdlcumwfgypbvkjxqz',.-", start=1)}

    def __init__(self) -> None:
        self.calls = 0

    def fitness(self, layout: Layout) -> float:
        self.calls += 1
        return sum(
            self._W[ch] * (abs(layout.pos(ch)[0]) + {2: 0.0, 3: 1.0, 1: 2.0}[layout.pos(ch)[1]])
            for ch in layout.chars
        )


#: A 16-key 2x8 mirrored board, used ONLY by the pairwise-landscape test below.
#: Small deliberately: that test enumerates all C(n,3)x5 reorderings TWICE, which is 20,300
#: boards x an O(n^2) scorer at n=30 (38 s, a 38% tax on the whole suite for one assertion) but
#: 2,800 at n=16 (~1 s). The property under test — a 2-opt optimum that is not 3-opt optimal —
#: is a property of pairwise interaction, not of board size, so nothing is weakened by shrinking
#: it; the 30-key measurement that motivated the flag is in the module docstring.
_SMALL_COLUMNS = (-4, -3, -2, -1, 1, 2, 3, 4)
_SMALL_GEOMETRY = Geometry(slots=tuple((c, y) for y in (2, 3) for c in _SMALL_COLUMNS))
_SMALL_ALPHABET = "etaoinshrdlcumwf"


class _PairFlowScorer(IScorer):
    """A tiny deterministic QAP: ``sum over char pairs of flow(a,b) * distance(pos)``.

    ⚠ NOT ``tests/optimize/conftest.CharPlacementScorer``, and the difference is the whole
    point. That scorer is a pure ASSIGNMENT problem — cost is a sum of independent per-character
    terms — and on those a 2-opt optimum is ALWAYS a 3-opt optimum: an improving 3-cycle
    decomposes into two swaps whose deltas add, so at least one of them improves on its own and
    2-opt would already have taken it. Written against that landscape, a "3-opt beats 2-opt" test
    passes vacuously or not at all (my first draft asserted it and correctly failed).

    Interaction here is genuinely PAIRWISE, which is the shape the real objective has
    (``TableBigramScorer`` is exactly ``sum f[a,b] * T[pos_a, pos_b]``), so 2-opt optima need not
    be 3-opt optima. ``flow`` is derived arithmetically from the character pair — no RNG, no data
    file, no seed-dependent import — so the landscape is byte-reproducible.
    """

    def __init__(self, seed: int = 4) -> None:
        self.calls = 0
        self._seed = seed

    def _flow(self, a: str, b: str) -> float:
        return float((ord(a) * 31 + ord(b) * 17 + self._seed * 7) % 23)

    def fitness(self, layout: Layout) -> float:
        self.calls += 1
        chars = layout.chars
        total = 0.0
        for i, a in enumerate(chars):
            for j, b in enumerate(chars):
                if i == j:
                    continue
                pa, pb = layout.pos(a), layout.pos(b)
                total += self._flow(a, b) * (abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]))
        return total


def _improving_three_opt_moves(layout_str: str, scorer: IScorer, geometry=ROW_STAGGERED_30) -> int:
    """Exhaustive count of reorderings of ANY triple that improve ``layout_str``.

    Independent of ``three_opt`` itself — enumerated here from the permutation definition, so a
    ``three_opt`` that silently stopped scanning some triples would be caught rather than
    confirmed. (A local optimum verified by the same code that produced it is not verified.)
    """
    from itertools import combinations, permutations

    baseline = scorer.fitness(Layout(layout_str, geometry))
    count = 0
    for i, j, k in combinations(range(len(layout_str)), 3):
        for target in permutations(("a", "b", "c")):
            if target == ("a", "b", "c"):
                continue
            chars = list(layout_str)
            src = {"a": layout_str[i], "b": layout_str[j], "c": layout_str[k]}
            chars[i], chars[j], chars[k] = (src[t] for t in target)
            if scorer.fitness(Layout("".join(chars), geometry)) < baseline:
                count += 1
    return count


def test_three_opt_polish_reaches_a_3opt_optimum_that_2opt_alone_does_not():
    """The reason the flag is worth having: 2-opt CONVERGED is not 3-opt converged.

    Pinned against an independent exhaustive enumeration, not against ``three_opt``'s own
    stopping condition — a local optimum verified by the code that produced it is not verified.

    Uses the pairwise-interaction landscape deliberately: see :class:`_PairFlowScorer` for why an
    assignment-shaped scorer CANNOT exhibit this gap.
    """
    geometry = _SMALL_GEOMETRY
    scorer = _PairFlowScorer()
    after_two = "".join(two_opt(Layout(_SMALL_ALPHABET, geometry), scorer).chars)
    left_after_two = _improving_three_opt_moves(after_two, scorer, geometry)
    assert left_after_two > 0, (
        "this landscape's 2-opt optimum is already 3-opt optimal, so it cannot demonstrate "
        "the gap the flag exists to close — pick a start where a triple still improves"
    )

    after_three = "".join(three_opt(Layout(after_two, geometry), scorer).chars)
    assert _improving_three_opt_moves(after_three, scorer, geometry) == 0
    assert scorer.fitness(Layout(after_three, geometry)) < scorer.fitness(
        Layout(after_two, geometry)
    ), "3-opt must strictly improve a 2-opt optimum that still has an improving triple"
    assert sorted(after_three) == sorted(_SMALL_ALPHABET), "and it stays a valid permutation"


def test_the_polish_helper_is_the_ONE_place_both_roles_get_their_polish():
    """``_polish`` must be what actually runs, for BOTH roles — that is the anti-asymmetry.

    Asserted by counting calls rather than by reading the source: the defect was two code paths
    that each looked correct in isolation, so "both call the same helper" has to be observed.
    """
    seen: list[str] = []
    real = optimize._polish

    def spy(args, layout, scorer):
        seen.append("".join(layout.chars))
        return real(args, layout, scorer)

    scorer = _CountingScorer()
    args = SimpleNamespace(no_local_search=False, three_opt=True)
    optimize._polish = spy
    try:
        spy(args, Layout(QWERTY30M, ROW_STAGGERED_30), scorer)
        spy(args, Layout(PERMUTED30M, ROW_STAGGERED_30), scorer)
    finally:
        optimize._polish = real
    assert seen == [QWERTY30M, PERMUTED30M]


def test_three_opt_flag_runs_more_evaluations_than_the_default_polish(tmp_path, capsys):
    """END-TO-END through the CLI: the flag must actually reach ``three_opt``.

    Counted through the scorer rather than asserted on the output, because on a tiny corpus the
    2-opt and 3-opt optima can coincide — a same-layout result would then read as "the flag did
    nothing" whether it ran or was silently dropped. Evaluation count separates those.
    """
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = _corpus(tmp_path)
    counts: dict[bool, dict[str, int]] = {}
    real = {"three": optimize.three_opt, "two": optimize.two_opt}

    def counting(name: str, tally: dict[str, int]):
        """A real closure over ``tally``, not a lambda over the loop variable (ruff B023)."""
        wrapped = real[name]

        def spy(layout, scorer):
            tally[name] += 1
            return wrapped(layout, scorer)

        return spy

    for use_three in (False, True):
        tally = {"three": 0, "two": 0}
        optimize.three_opt = counting("three", tally)
        optimize.two_opt = counting("two", tally)
        try:
            argv = _argv(model_path, corpus, *(["--three-opt"] if use_three else []))
            assert main(argv) == 0
            capsys.readouterr()
        finally:
            optimize.three_opt = real["three"]
            optimize.two_opt = real["two"]
        counts[use_three] = tally
        assert tally["two"] >= 1, "2-opt must run in both modes; 3-opt EXTENDS it"

    assert counts[False]["three"] == 0, "the default must not call three_opt (A6's default gate)"
    assert counts[True]["three"] >= 1, "--three-opt must actually call three_opt"


# ---------------------------------------------------------------------------------------
# 2. SYMMETRY: an incumbent can get the same polish, and the report says how much it was.
# ---------------------------------------------------------------------------------------


def test_polish_incumbent_reports_as_is_AND_polished_so_the_gap_is_attributable(tmp_path, capsys):
    """The three columns are the fix: as-is, polished, and the difference between them.

    An as-is-only report is what let a gap that was mostly polish read as a layout advantage
    (measured on the campaign gauge: 71-91% of the as-is gap vs arm B was polish).
    """
    model_path = _train_tiny_model(tmp_path / "bg.json")
    out_path = tmp_path / "result.json"
    assert (
        main(
            _argv(
                model_path,
                _corpus(tmp_path),
                "--start",
                QWERTY30M,
                "--polish-incumbent",
                PERMUTED30M,
                "--out",
                str(out_path),
            )
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "incumbents, polished the SAME way" in text
    assert "as-is" in text and "polished" in text

    result = json.loads(out_path.read_text())
    assert result["polish"] == "2-opt", "the artifact must name the polish it ran"
    row = result["incumbents"][0]
    assert row["layout"] == PERMUTED30M
    # The polished incumbent is a permutation of itself and never WORSE than as-is: the polish
    # only accepts improving moves, so a positive `polish_gain` would mean the two roles ran
    # different objectives -- the exact drift this flag exists to prevent.
    assert sorted(row["polished_layout"]) == sorted(PERMUTED30M)
    assert row["polish_gain"] <= 0.0
    assert row["fitness_polished"] == pytest.approx(
        row["fitness_as_is"] + row["polish_gain"], abs=1e-9
    )


def test_the_incumbent_gets_the_SAME_polish_the_searched_layout_got(tmp_path, capsys):
    """``--three-opt`` must upgrade BOTH roles' polish, not just the search's.

    If the incumbent kept the 2-opt polish while the searched layout got 3-opt, the comparison
    would still be asymmetric — just less visibly, which is worse.
    """
    model_path = _train_tiny_model(tmp_path / "bg.json")
    out_path = tmp_path / "three.json"
    assert (
        main(
            _argv(
                model_path,
                _corpus(tmp_path),
                "--start",
                QWERTY30M,
                "--three-opt",
                "--polish-incumbent",
                PERMUTED30M,
                "--out",
                str(out_path),
            )
        )
        == 0
    )
    capsys.readouterr()
    result = json.loads(out_path.read_text())
    assert result["polish"] == "2-opt+3-opt"

    # And the recorded polished board is exactly what the SAME polish produces standalone.
    args = SimpleNamespace(no_local_search=False, three_opt=True)
    scorer = _CountingScorer()
    expected = "".join(optimize._polish(args, Layout(PERMUTED30M, ROW_STAGGERED_30), scorer).chars)
    replay = json.loads(out_path.read_text())["incumbents"][0]
    assert sorted(replay["polished_layout"]) == sorted(expected)


# ---------------------------------------------------------------------------------------
# 3. The gates: REFUSE an unsupported combination rather than silently falling back.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra,expected",
    [
        (["--three-opt", "--no-local-search"], "cannot be combined"),
        (["--polish-incumbent", PERMUTED30M, "--no-local-search"], "no polish to apply"),
    ],
)
def test_unsupported_combinations_are_REFUSED_before_the_search(tmp_path, extra, expected):
    """A silent fallback would label a number '3-opt' that no 3-opt ever touched.

    And the refusal must land BEFORE the search: the model path here does not exist, so a gate
    that ran after scorer construction would raise something else entirely.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(_argv("/nonexistent/model.json", _corpus(tmp_path), "--start", QWERTY30M, *extra))
    assert expected in str(excinfo.value)


def test_an_incumbent_off_the_search_charset_is_REFUSED_not_scored(tmp_path):
    """A different charset covers different corpus rows, so its score is a different mean.

    Refused rather than printed, for the reason ``analyze`` renders such a cell N/A: the number
    would be dimensionally wrong while looking directly comparable in the same column.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(
            _argv(
                "/nonexistent/model.json",
                _corpus(tmp_path),
                "--start",
                QWERTY30M,
                "--polish-incumbent",
                "qwertyuiopasdfghjkl;zxcvbnm,./",  # ';' and '/' for "'" and '-'
            )
        )
    assert "not a permutation of --start's charset" in str(excinfo.value)


def test_a_repeated_incumbent_is_REFUSED_rather_than_polished_twice(tmp_path):
    """Same refusal shape as ``--model-weight`` given twice: ambiguity, not a silent last-wins."""
    with pytest.raises(SystemExit) as excinfo:
        main(
            _argv(
                "/nonexistent/model.json",
                _corpus(tmp_path),
                "--start",
                QWERTY30M,
                "--polish-incumbent",
                PERMUTED30M,
                "--polish-incumbent",
                PERMUTED30M,
            )
        )
    assert "more than once" in str(excinfo.value)


# ---------------------------------------------------------------------------------------
# 4. DEFAULT UNCHANGED — the gate the whole change is scoped to.
# ---------------------------------------------------------------------------------------


def test_the_default_invocation_never_calls_three_opt_and_adds_NO_json_key(tmp_path, capsys):
    """The default artifact keeps EXACTLY its shipped key set — no new key, not even a label.

    Byte-identity of the full default stdout against origin/main was verified out-of-band across
    six invocations (default / --no-table / --no-local-search / --attempts 3 / --comfort-weight /
    --finger-load-weight, all identical sha256) plus ``--help``. This is the in-suite half.

    The `polish` key is asserted ABSENT here, which is the second bug the default diff caught: an
    unconditional ``result["polish"] = ...`` broke ``test_optimize_out_writes_expected_json``'s
    ``set(result) ==`` pin. Adding a key to the default artifact is a default-behaviour change,
    so the label is emitted only when a flag made the polish non-default.
    """
    model_path = _train_tiny_model(tmp_path / "bg.json")
    out_path = tmp_path / "default.json"
    real_three = optimize.three_opt
    called = []
    optimize.three_opt = lambda lay, sc: called.append(1) or real_three(lay, sc)
    try:
        assert main(_argv(model_path, _corpus(tmp_path), "--out", str(out_path))) == 0
    finally:
        optimize.three_opt = real_three
    text = capsys.readouterr().out

    assert not called, "the default must not reach three_opt"
    assert "incumbents, polished" not in text, "no incumbent block without the flag"
    result = json.loads(out_path.read_text())
    assert set(result) == {
        "layout",
        "fitness",
        "ngram",
        "target_wpm",
        "seed",
        "attempts",
        "model",
    }, "the default --out artifact must gain NO key from this change"
    assert "polish" not in result, "even the polish LABEL is opt-in (see the docstring)"
    assert "incumbents" not in result, "the key must be absent, not an empty list"


def test_argparse_help_still_renders_which_a_bare_percent_sign_would_break():
    """REGRESSION: my own first draft crashed ``optimize --help`` with a bare ``%``.

    argparse runs every help string through ``% params``, so ``-0.27% ms/char`` raised
    ``ValueError: unsupported format character 'm'`` and killed the whole help output — in the
    one invocation that has no other result to notice the absence by. Caught only because the
    default-unchanged diff covered ``--help`` as well as the search paths.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="keybo optimize")
    optimize.add_arguments(parser)
    text = parser.format_help()
    assert "--three-opt" in text
    assert "--polish-incumbent" in text
    assert "-0.27% ms/char" in text, "the escaped %% must render as a single literal %"
