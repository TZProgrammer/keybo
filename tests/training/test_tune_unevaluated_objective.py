"""``tune_lolo`` must refuse when its stated objective was never evaluated.

The lolo objective is mean held-out rho/ceiling, and ``-inf`` is also the "lost the tau
gate" sentinel. When the ceiling is unobtainable — every layout having one participant, so
``split_half_ceiling`` bisects nothing and returns nan — every candidate scores ``-inf``,
the two states become indistinguishable, and the tie-break promotes a champion whose
objective never ran. Measured on the shipped community strokes: all 11 layouts have exactly
1 participant and 0 folds repo-wide have >= 2, so this is the DEFAULT experience there, not
an edge case.
"""

from __future__ import annotations

import math

import pytest

from keybo.data.strokes import StrokeRow
from keybo.training.tune import (
    ObjectiveNotEvaluated,
    UnevaluatedObjectiveWarning,
    _ceiling_diagnosis,
    tune_lolo,
)
from keybo.training.validate import split_half_ceiling

_PARAMS = [
    {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.2, "subsample": 1.0},
    {"n_estimators": 45, "max_depth": 3, "learning_rate": 0.15, "subsample": 0.9},
]


def _rows(n_participants: int, n_layouts: int = 3, n_ngrams: int = 6, seed: int = 5):
    """Stroke rows with a controllable participant count per layout.

    Reuses ``test_validate``'s geometry-lawful fixture (real 2-char ngrams at real position
    pairs, duration linear in distance) rather than inventing a third fixture shape — the
    only thing varied here is the PARTICIPANT count, which is what drives the ceiling.
    """
    import numpy as np

    from .test_validate import _POSITIONS, _distance

    rng = np.random.default_rng(seed)
    rows = []
    for layout, ngrams in list(_POSITIONS.items())[:n_layouts]:
        for ngram, positions in list(ngrams.items())[:n_ngrams]:
            samples = []
            for pid in range(1, n_participants + 1):
                for _ in range(8):
                    wpm = int(rng.integers(65, 95))
                    dur = int(60 + 25 * _distance(positions) + rng.normal(0, 4))
                    samples.append((wpm, dur, pid, 50))
            rows.append(
                StrokeRow(
                    layout=layout,
                    positions=positions,
                    ngram=ngram,
                    frequency=1000,
                    samples=samples,
                )
            )
    return rows


# --- the precondition this whole file rests on ------------------------------------------


def test_one_participant_makes_the_ceiling_nan() -> None:
    """The mechanism, pinned directly: < 2 participants -> nan, not a small number."""
    single = split_half_ceiling(_rows(n_participants=1), n_boot=4, seed=0)
    assert math.isnan(single), "one participant cannot be bisected"
    plural = split_half_ceiling(_rows(n_participants=6), n_boot=4, seed=0)
    assert math.isfinite(plural) and plural > 0.0


# --- the refusal -------------------------------------------------------------------------


def test_tune_lolo_REFUSES_when_no_fold_yields_a_finite_score() -> None:
    with pytest.raises(ObjectiveNotEvaluated) as exc:
        tune_lolo(_rows(n_participants=1), _PARAMS, seeds=[0], min_cell_samples=4)
    msg = str(exc.value)
    # the message must be actionable: name the objective, the counts, and the cause
    assert "never evaluated" in msg
    assert "rho/ceiling" in msg
    assert "PARTICIPANTS" in msg, "must name WHY, not just that it failed"
    assert "0 of" in msg, "must report how many cells were finite"


def test_the_refusal_can_be_downgraded_but_only_EXPLICITLY() -> None:
    with pytest.warns(UnevaluatedObjectiveWarning):
        best, leaderboard = tune_lolo(
            _rows(n_participants=1),
            _PARAMS,
            seeds=[0],
            min_cell_samples=4,
            allow_unevaluated_objective=True,
        )
    assert best in _PARAMS
    # and the leaderboard must still SHOW that nothing was measured
    assert all(score == float("-inf") for _p, score in leaderboard)


def test_a_healthy_dataset_is_unaffected_and_scores_finitely() -> None:
    """The guard must not fire when the objective IS evaluable — no false refusal."""
    best, leaderboard = tune_lolo(
        _rows(n_participants=6), _PARAMS, seeds=[0], min_cell_samples=4
    )
    assert best in _PARAMS
    finite = [s for _p, s in leaderboard if math.isfinite(s)]
    assert finite, "at least the tau-gate winner must carry a finite rho/ceiling"


def test_the_diagnosis_names_the_participant_counts() -> None:
    text = _ceiling_diagnosis(_rows(n_participants=1, n_layouts=3), "bigram")
    assert "0 of 3 layouts have >= 2" in text
    assert "min 1, max 1" in text
    healthy = _ceiling_diagnosis(_rows(n_participants=5, n_layouts=2), "bigram")
    assert "2 of 2 layouts have >= 2" in healthy


def test_the_diagnosis_survives_empty_input() -> None:
    assert "No bigram rows" in _ceiling_diagnosis([], "bigram")


def test_a_NON_FINITE_frac_is_excluded_as_firmly_as_a_None_one(monkeypatch) -> None:
    """The filter must test isfinite, not just ``is not None``.

    A nan ceiling can yield ``rho/nan = nan`` — a float, not None — which passes an
    ``is not None`` check and then poisons ``np.mean`` to nan, making the score neither
    finite nor ``-inf`` and slipping past the refusal. This distinguishes the two guards;
    without it, deleting the ``math.isfinite`` clause is invisible to the suite (it was:
    the mutation passed 0 failures before this test existed).
    """
    import keybo.training.validate as validate_mod

    real_validate = validate_mod.validate  # captured BEFORE patching, so no recursion

    def fake_validate(*args, **kwargs):
        report = real_validate(*args, **kwargs)
        # every fold reports a FLOAT that is not finite, rather than None
        for fold in report["folds"].values():
            for m in fold["seeds"]:
                m["rho_frac_ceiling"] = float("nan")
        return report

    monkeypatch.setattr(validate_mod, "validate", fake_validate)
    with pytest.raises(ObjectiveNotEvaluated):
        tune_lolo(_rows(n_participants=6), _PARAMS, seeds=[0], min_cell_samples=4)


# --- what the old behaviour was, so a regression is unmistakable -------------------------


def test_the_pre_fix_behaviour_was_a_SILENT_champion() -> None:
    """Documents the defect: the tie-break alone chose, with no signal to the caller.

    Under ``allow_unevaluated_objective=True`` we reproduce it deliberately — every score is
    ``-inf`` and the returned params are simply the first tau-gate survivor. A caller who
    ignored the warning would read that as a recommendation, which is why the DEFAULT is to
    raise rather than to warn.
    """
    with pytest.warns(UnevaluatedObjectiveWarning):
        best, leaderboard = tune_lolo(
            _rows(n_participants=1),
            _PARAMS,
            seeds=[0],
            min_cell_samples=4,
            allow_unevaluated_objective=True,
        )
    scores = [s for _p, s in leaderboard]
    assert scores == [float("-inf")] * len(_PARAMS)
    # "best" is indistinguishable from a real winner by inspection alone — the point.
    assert isinstance(best, dict) and "max_depth" in best
