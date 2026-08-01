"""``target_space`` and ``with_layouts`` must be REACHABLE and EFFECTIVE on the public API.

Before ``cb907aa`` both existed only on the private ``_build_matrix_full``: no caller of the public
``build_training_matrix`` could ask for anything but ``"MS"``, while all six shipped k31 models are
``"LOGRAT"`` — a hardwired train/serve mismatch (KAGGLE-1 FINAL, ledger ``cf6ee07``). ``layouts``
was computed on every call and thrown away, which is exactly the ``groups=`` argument the new
grouped cross-validation needs.

The defect shape being guarded against is **present != effective**: a parameter that is accepted
and then ignored looks identical to a working one at the call site. So these tests never merely
check that the keyword is accepted — they check that the returned ``y`` actually changed, that it
changed in the right direction (log scale), and that ``X`` did *not*.

⚠ One test here documents a defect rather than a fix — see
``test_an_UNKNOWN_target_space_is_silently_accepted_and_falls_back_to_MS``. The public function
does no validation at all, so a typo is silent. It is pinned as the CURRENT behaviour, deliberately
not asserted as correct.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from keybo.data.strokes import StrokeRow
from keybo.training.train import _TARGET_SPACES, build_training_matrix

#: Two layouts so ``with_layouts`` has something to distinguish, and durations chosen so the two
#: target spaces cannot coincide: LOGRAT is ``log(duration * wpm / 12000)``, which is negative for
#: the first row and positive for the second, while MS is the raw duration for both.
_ROWS = [
    StrokeRow(
        layout="qwerty",
        positions=((-1, 3), (1, 2)),
        ngram="th",
        frequency=5,
        samples=[(90, 120, 1, 50)],
    ),
    StrokeRow(
        layout="dvorak",
        positions=((1, 2), (-3, 3)),
        ngram="he",
        frequency=5,
        samples=[(60, 240, 2, 50)],
    ),
]


def _y(target_space: str | None = None) -> np.ndarray:
    kwargs = {} if target_space is None else {"target_space": target_space}
    return build_training_matrix(_ROWS, ngram="bigram", target_wpm=90, **kwargs)[1]


# --- target_space is reachable, and it actually CHANGES the targets ----------------------


def test_LOGRAT_is_reachable_from_the_PUBLIC_api_and_differs_from_MS() -> None:
    """The fix itself. A parameter accepted and ignored would pass a weaker version of this test.

    Before ``cb907aa`` this call raised ``TypeError: unexpected keyword argument``; the risk now is
    the opposite one, so the assertion is on the VALUES, not on the call succeeding.
    """
    ms, lograt = _y("MS"), _y("LOGRAT")
    assert not np.allclose(ms, lograt), "target_space is accepted but has no effect on y"


def test_LOGRAT_targets_are_on_a_LOG_scale_and_MS_targets_are_raw_milliseconds() -> None:
    """Not just "different" — different in the specific way the name promises.

    ``log(duration * wpm / 12000)`` straddles zero, while a duration in milliseconds cannot be
    negative. A transform that merely rescaled MS would satisfy the "differs" test above and fail
    this one.
    """
    ms, lograt = _y("MS"), _y("LOGRAT")
    assert ms.tolist() == [120.0, 240.0], "MS must be the raw IQR-mean duration"
    assert (ms > 0).all()
    assert lograt[0] < 0.0 < lograt[1], f"LOGRAT must straddle zero here, got {lograt.tolist()}"
    for y_value, (wpm, duration) in zip(lograt, [(90, 120), (60, 240)], strict=True):
        assert y_value == pytest.approx(math.log(duration * wpm / 12000.0))


def test_DOUBLING_a_duration_shifts_a_LOGRAT_target_by_exactly_log_two() -> None:
    """The log-scale property stated as a law rather than a table of values.

    A multiplicative change in duration must become an additive shift in the target — that is the
    whole reason the adopted bigram space is logarithmic (T-REL, -37% cross-layout wmae). Pinning
    the law means the test survives a change to the 12000 constant, which a value table would not.
    """

    def target(duration: int) -> float:
        row = StrokeRow(
            layout="qwerty",
            positions=((-1, 3), (1, 2)),
            ngram="th",
            frequency=5,
            samples=[(90, duration, 1, 50)],
        )
        return float(
            build_training_matrix([row], ngram="bigram", target_wpm=90, target_space="LOGRAT")[1][0]
        )

    assert target(240) - target(120) == pytest.approx(math.log(2))


def test_the_target_space_leaves_the_FEATURE_matrix_byte_identical() -> None:
    """Scope: ``target_space`` names the target space, so it must not touch X.

    If it did, a model trained in one space could not be compared against one trained in the other
    — the confound would be in the frame rather than in the objective.
    """
    x_ms = build_training_matrix(_ROWS, ngram="bigram", target_wpm=90, target_space="MS")[0]
    x_lograt = build_training_matrix(_ROWS, ngram="bigram", target_wpm=90, target_space="LOGRAT")[0]
    assert np.array_equal(x_ms, x_lograt), "target_space must not change the features"


def test_the_DEFAULT_is_still_MS_so_existing_callers_are_unaffected() -> None:
    """``cb907aa`` deliberately did not change the default; a silent flip would be a data change.

    Every caller that omits the argument — the ``cv-mae`` CLI path among them — must keep getting
    exactly what it got before the parameter existed.
    """
    assert np.array_equal(_y(), _y("MS"))


# --- the validation gap: a typo in target_space is SILENT (defect, pinned as-is) ----------


def test_an_UNKNOWN_target_space_is_silently_accepted_and_falls_back_to_MS() -> None:
    """⚠ DEFECT, documented not endorsed: the public function validates NOTHING.

    ``_train`` upper-cases its ``target_space`` and rejects anything outside ``_TARGET_SPACES``
    (``train.py``, ``unknown target_space ... (known: ['LOGRAT', 'MS'])``). ``build_training_matrix``
    does neither: it forwards the string to ``_build_matrix_full``, whose ``_group_target`` tests
    ``target_space == "LOGRAT"`` exactly and falls through to MS for anything else.

    So the parameter is now reachable but MISSPELLING it is invisible — the caller gets a
    plausible matrix in the wrong space, which is the same present-!=-effective failure the fix
    was meant to remove, one level down. Pinned as the CURRENT behaviour so the next reader sees
    it; if validation is added, this test should be INVERTED to a ``pytest.raises``, not deleted.
    """
    assert np.array_equal(_y("BOGUS"), _y("MS")), "current behaviour: unknown space -> MS targets"
    assert "BOGUS" not in _TARGET_SPACES


def test_a_LOWERCASE_lograt_silently_yields_MS_targets_not_LOGRAT_ones() -> None:
    """⚠ Same defect, in its most likely form: the case that a human would actually type.

    ``_train`` normalizes case, so ``target_space="lograt"`` works there. Through the public
    function it silently produces MS targets — the WORST version of this bug, because the caller
    asked for the right space by name and got the other one.
    """
    lowercase = _y("lograt")
    assert np.array_equal(lowercase, _y("MS")), "current behaviour: case is NOT normalized"
    assert not np.allclose(lowercase, _y("LOGRAT")), (
        "and it is NOT the space the caller asked for — the defect, stated plainly"
    )


def test_every_documented_target_space_is_reachable_and_produces_distinct_targets() -> None:
    """Guards the pair as a set: whatever ``_TARGET_SPACES`` lists must work HERE too.

    If a third space is added to the module constant and wired only into ``_train``, this fails —
    which is precisely how the original defect got in.
    """
    produced = {space: _y(space) for space in _TARGET_SPACES}
    assert set(produced) == {"MS", "LOGRAT"}
    for space, y_values in produced.items():
        assert y_values.shape == (len(_ROWS),), space
        assert np.all(np.isfinite(y_values)), space
    assert not np.allclose(produced["MS"], produced["LOGRAT"])


# --- with_layouts: the groups= argument grouped CV needs ---------------------------------


def test_with_layouts_returns_the_per_example_layout_labels_grouped_cv_needs() -> None:
    """The other half of fix 2: the labels were always computed and discarded.

    ``grouped_cv`` cannot work without them, so this is the join between fixes 1 and 2 — and the
    returned labels must be per-EXAMPLE (aligned to X's rows), not the per-row input order.
    """
    X, y, layouts = build_training_matrix(_ROWS, ngram="bigram", target_wpm=90, with_layouts=True)
    assert layouts.tolist() == ["qwerty", "dvorak"]
    assert len(layouts) == X.shape[0] == y.shape[0], "one label per EXAMPLE, not per input row"


def test_with_layouts_changes_only_the_ARITY_and_not_X_or_y() -> None:
    """A flag that also perturbed the data would make grouped and ungrouped runs incomparable."""
    two = build_training_matrix(_ROWS, ngram="bigram", target_wpm=90)
    three = build_training_matrix(_ROWS, ngram="bigram", target_wpm=90, with_layouts=True)
    assert len(two) == 2 and len(three) == 3
    assert np.array_equal(two[0], three[0]) and np.array_equal(two[1], three[1])


def test_with_layouts_composes_with_target_space_which_is_how_the_tuner_calls_it() -> None:
    """The two new parameters must be usable TOGETHER, since a real caller needs both at once.

    Grouped CV on the shipped LOGRAT space is the whole point of the pair; a signature that
    accepted them only separately would satisfy every test above and still be unusable.
    """
    X, y, layouts = build_training_matrix(
        _ROWS, ngram="bigram", target_wpm=90, target_space="LOGRAT", with_layouts=True
    )
    assert layouts.tolist() == ["qwerty", "dvorak"]
    assert np.allclose(y, _y("LOGRAT")), "target_space must survive being passed alongside"
    assert X.shape[0] == len(layouts)


def test_the_layout_labels_are_usable_as_grouped_cv_groups_end_to_end() -> None:
    """Closes the loop fixes 1 and 2 were split across: the labels must satisfy ``grouped_cv``.

    Asserting the labels' *content* is not enough — ``grouped_cv`` requires >= 2 distinct groups
    and splits on them, so this feeds the real output into the real splitter and checks that no
    layout lands on both sides of a fold.
    """
    from keybo.training.tune import grouped_cv

    rows = [
        StrokeRow(
            layout=layout,
            positions=((-1, 3), (1, 2)),
            ngram=ngram,
            frequency=5,
            samples=[(90, 120, 1, 50), (85, 130, 2, 55)],
        )
        for layout in ("qwerty", "dvorak", "azerty")
        for ngram in ("th", "he", "an")
    ]
    X, _y_unused, layouts = build_training_matrix(
        rows, ngram="bigram", target_wpm=90, with_layouts=True
    )
    splitter = grouped_cv(5, layouts)
    assert splitter.get_n_splits(groups=layouts) == 3, "clamped to the 3 layouts present"
    for train_idx, test_idx in splitter.split(X, groups=layouts):
        assert not set(layouts[train_idx]) & set(layouts[test_idx])
