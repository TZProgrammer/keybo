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

Two tests here were originally written to DOCUMENT a defect rather than a fix: the public function
did no validation, so ``"BOGUS"`` and ``"lograt"`` both silently produced MS targets. They carried
the instruction that adding validation should INVERT them to ``pytest.raises`` / the right-space
assertion rather than delete them. Validation now exists on the public boundary, so both are
inverted and both keep their history note — an inverted pin still guards the original defect (a
reader who sees the silent fallback come back has found a regression, with the whole story in the
docstring), where a deleted one would have guarded nothing.
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


# --- the validation gap, now CLOSED: a typo raises and case normalizes -------------------


def test_an_UNKNOWN_target_space_RAISES_instead_of_falling_back_to_MS() -> None:
    """The inverted defect pin. HISTORY: this test used to assert the SILENT fallback.

    ``_train`` always upper-cased its ``target_space`` and rejected anything outside
    ``_TARGET_SPACES``, but ``build_training_matrix`` did NEITHER: it forwarded the string to
    ``_build_matrix_full``, whose ``_group_target`` tested ``target_space == "LOGRAT"`` exactly
    and fell through to MS for anything else. So the parameter was reachable but MISSPELLING it
    was invisible — the caller got a plausible matrix in the wrong space, the same
    present-!=-effective failure that reaching the parameter was meant to remove, one level down.

    It was pinned here as the CURRENT behaviour with the instruction that adding validation should
    INVERT this test rather than delete it. Validation now lives on the public boundary
    (``normalize_target_space``), so the assertion is inverted and the history is kept: a reader
    who sees a silent MS fallback return has found a REGRESSION, not a new bug.
    """
    assert "BOGUS" not in _TARGET_SPACES
    with pytest.raises(ValueError, match="unknown target_space"):
        _y("BOGUS")


def test_a_LOWERCASE_lograt_now_yields_LOGRAT_targets_instead_of_MS_ones() -> None:
    """The inverted defect pin, in the form a human would actually type. HISTORY: this used to
    assert that ``"lograt"`` silently produced MS targets.

    That was the WORST version of the bug — the caller asked for the right space BY NAME and got
    the other one, with no error. ``_train`` normalized case all along, so the public function was
    the only reader that did not.

    Case NORMALIZES rather than raising, matching the two other readers of this field:
    ``_train``'s long-standing ``.upper()`` and
    :attr:`keybo.models.base.TypingModel.target_space` (``test_target_space_reads_sidecar_case_insensitively``).
    The assertion is therefore on the VALUES, not on the call merely succeeding — accepting
    ``"lograt"`` and still computing MS would be the original defect wearing a passing test.
    """
    lowercase = _y("lograt")
    assert np.allclose(lowercase, _y("LOGRAT")), "lowercase must mean the space it names"
    assert not np.allclose(lowercase, _y("MS")), (
        "and it must NOT be the silent MS fallback this test was written to document"
    )


def test_the_public_boundary_and_the_trainer_share_ONE_target_space_gate() -> None:
    """The two entry points must agree, since disagreeing is how the original defect arose.

    ``_train`` validated and ``build_training_matrix`` did not, so the same string meant different
    things depending on which door the caller used. Both now route through
    ``normalize_target_space``; this pins the AGREEMENT rather than each side separately, because
    two independently-correct checks can still drift apart.
    """
    from keybo.training.train import normalize_target_space

    for good, expected in [("MS", "MS"), ("ms", "MS"), ("LOGRAT", "LOGRAT"), ("LogRat", "LOGRAT")]:
        assert normalize_target_space(good) == expected

    for bad in ["BOGUS", "", "LOGRATIO", "MS "]:
        with pytest.raises(ValueError, match="unknown target_space"):
            normalize_target_space(bad)
        with pytest.raises(ValueError, match="unknown target_space"):
            _y(bad)


def test_group_target_refuses_an_unnormalized_space_instead_of_defaulting_to_MS() -> None:
    """Defence in depth at the row that computes the number.

    ``_group_target`` used to be ``if LOGRAT ... else MS``, so a space added to
    ``_TARGET_SPACES`` and wired only into ``_train`` would have produced a plausible MS matrix
    for it. This is unreachable through the public API by design — the callers normalize — and
    that is precisely why it is worth pinning: it is the backstop, not the gate.
    """
    from keybo.training.train import _group_target

    assert _group_target([120], 90, "MS") == pytest.approx(120.0)
    assert _group_target([120], 90, "LOGRAT") == pytest.approx(math.log(120 * 90 / 12000.0))
    with pytest.raises(ValueError, match="unknown target_space"):
        _group_target([120], 90, "lograt")


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
