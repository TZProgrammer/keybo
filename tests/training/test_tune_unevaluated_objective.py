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
    LOLO_MIN_MARGIN,
    ObjectiveNotEvaluated,
    UnevaluatedObjectiveWarning,
    _ceiling_diagnosis,
    tune_lolo,
)
from keybo.training.validate import split_half_ceiling
from keybo.verdicts import MarginTooSmall

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
    """The unevaluated-objective guard must not fire when the objective IS evaluable.

    NOTE the MARGIN gate legitimately fires here: on this fixture the two candidates score
    0.9000-ish within 0.0023 of each other (0.26% relative), well inside the resolvable
    margin — which is exactly the situation the gate exists to catch, and a good illustration
    that near-ties are the NORMAL case for a 2-candidate sweep rather than a contrived one. So
    disable the margin gate here to isolate what this test is about, and let the dedicated
    margin tests below cover the gate itself.
    """
    best, leaderboard = tune_lolo(
        _rows(n_participants=6), _PARAMS, seeds=[0], min_cell_samples=4, min_margin=0.0
    )
    assert best in _PARAMS
    finite = [s for _p, s in leaderboard if math.isfinite(s)]
    assert finite, "at least the tau-gate winner must carry a finite rho/ceiling"
    # and the objective WAS evaluated — no -inf sweep
    assert len(finite) >= 1 and all(math.isfinite(s) for s in finite)


def test_the_margin_gate_fires_on_REAL_near_tied_candidates_not_only_synthetic_ones() -> None:
    """The 2-candidate sweep on the lawful fixture is genuinely inside the margin.

    Measured, not mocked: the two shipped _PARAMS differ by ~0.0023 rho/ceiling on this data,
    a 0.26% relative gap. Pinning it here means the gate's real-world bite is a contract, not
    an accident of the mocked tests.
    """
    with pytest.raises(MarginTooSmall) as exc:
        tune_lolo(_rows(n_participants=6), _PARAMS, seeds=[0], min_cell_samples=4)
    assert "inside what the scoring rule can resolve" in str(exc.value)


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


# --- the minimum-margin gate, THROUGH tune_lolo ------------------------------------------


def test_tune_lolo_REFUSES_a_selection_inside_the_resolvable_margin(monkeypatch) -> None:
    """The gate must fire at the real selection site, not only standalone.

    Two near-identical candidates on a healthy dataset: the objective IS evaluated, so the
    unevaluated-objective guard does not fire, and the margin gate is the only thing standing
    between a 0.1%-margin win and a params file that reads like any other.
    """
    import keybo.training.validate as validate_mod

    real_validate = validate_mod.validate
    calls = {"n": 0}
    # give candidate 0 a 0.1% edge — far inside LOLO_MIN_MARGIN
    fracs = [0.9000, 0.8991]

    def fake_validate(*args, **kwargs):
        report = real_validate(*args, **kwargs)
        value = fracs[min(calls["n"], len(fracs) - 1)]
        calls["n"] += 1
        for fold in report["folds"].values():
            for m in fold["seeds"]:
                m["rho_frac_ceiling"] = value
        return report

    monkeypatch.setattr(validate_mod, "validate", fake_validate)
    with pytest.raises(MarginTooSmall) as exc:
        tune_lolo(_rows(n_participants=6), _PARAMS, seeds=[0], min_cell_samples=4)
    assert "lolo hyperparameter selection" in str(exc.value)


def test_the_margin_gate_can_be_downgraded_or_disabled(monkeypatch) -> None:
    import keybo.training.validate as validate_mod

    real_validate = validate_mod.validate
    calls = {"n": 0}
    fracs = [0.9000, 0.8991]

    def fake_validate(*args, **kwargs):
        report = real_validate(*args, **kwargs)
        value = fracs[min(calls["n"], len(fracs) - 1)]
        calls["n"] += 1
        for fold in report["folds"].values():
            for m in fold["seeds"]:
                m["rho_frac_ceiling"] = value
        return report

    monkeypatch.setattr(validate_mod, "validate", fake_validate)
    rows = _rows(n_participants=6)
    # warn instead of raise
    with pytest.warns(UnevaluatedObjectiveWarning):
        best, _lb = tune_lolo(
            rows,
            _PARAMS,
            seeds=[0],
            min_cell_samples=4,
            allow_unresolvable_margin=True,
        )
    assert best in _PARAMS
    # or disable outright, for reproducing a historical selection
    calls["n"] = 0
    best2, _lb2 = tune_lolo(rows, _PARAMS, seeds=[0], min_cell_samples=4, min_margin=0.0)
    assert best2 in _PARAMS


def test_a_WIDE_margin_passes_the_gate_untouched(monkeypatch) -> None:
    """No false refusal: a decisive win must not be blocked."""
    import keybo.training.validate as validate_mod

    real_validate = validate_mod.validate
    calls = {"n": 0}
    fracs = [0.95, 0.60]  # a 37% relative gap, far outside the bound

    def fake_validate(*args, **kwargs):
        report = real_validate(*args, **kwargs)
        value = fracs[min(calls["n"], len(fracs) - 1)]
        calls["n"] += 1
        for fold in report["folds"].values():
            for m in fold["seeds"]:
                m["rho_frac_ceiling"] = value
        return report

    monkeypatch.setattr(validate_mod, "validate", fake_validate)
    best, leaderboard = tune_lolo(_rows(n_participants=6), _PARAMS, seeds=[0], min_cell_samples=4)
    assert best == _PARAMS[0]
    assert leaderboard[0][1] == pytest.approx(0.95)


def test_the_shipped_threshold_is_the_derived_one_not_a_round_number() -> None:
    """Pins the provenance: 0.03 comes from the ceiling reweighting bound, 0.0301."""
    from keybo.verdicts import reweighting_margin_bound

    bound = reweighting_margin_bound([(1 + c) / 2 for c in (0.709, 0.815)])
    assert pytest.approx(0.03, abs=1e-9) == LOLO_MIN_MARGIN
    assert bound >= LOLO_MIN_MARGIN, "the gate must not be looser than the bound it derives from"


# --- an EMPTY candidate list is a caller bug, not an internal error -----------------------


def test_tune_lolo_REFUSES_an_EMPTY_candidate_list_naming_the_actual_problem() -> None:
    """Was an ``UnboundLocalError`` about a local variable, which named the wrong thing.

    ``report`` is bound only inside the candidates loop and read after it, so zero candidates
    fell straight through to ``len(report["folds"])``. The error a caller saw was "cannot access
    local variable 'report'", which describes ``tune_lolo``'s internals rather than the empty
    list that caused it.

    Refusing beats initialising ``report``: with zero candidates there is no champion to return,
    so initialising it merely moves the crash twenty lines down to ``leaderboard[0][0]`` on the
    empty leaderboard — a different internal error for the same caller mistake.
    ``test_an_empty_candidate_list_no_longer_reaches_either_crash_site`` pins that second site
    directly.
    """
    with pytest.raises(ValueError, match="at least one candidate"):
        tune_lolo(_rows(n_participants=6), [], seeds=[0])


def test_the_empty_candidate_refusal_is_not_bypassed_by_allow_unevaluated_objective() -> None:
    """The flag that EXPOSED this bug must not also be a way around the new guard.

    The defect was latent because the ``ObjectiveNotEvaluated`` refusal pre-empted it on the
    default path; ``allow_unevaluated_objective=True`` downgraded that refusal to a warning and
    let execution reach the unbound read. So this is the reproduction case, and it must now hit a
    clear ValueError rather than either crash.
    """
    with pytest.raises(ValueError, match="at least one candidate"):
        tune_lolo(_rows(n_participants=1), [], seeds=[0], allow_unevaluated_objective=True)


def test_an_empty_candidate_list_no_longer_reaches_either_crash_site() -> None:
    """Pins WHY refusing was chosen over initialising: the second crash site is real.

    ``apply_tau_gate([])`` returns an empty gated list, so the final ``leaderboard[0][0]`` raises
    ``IndexError`` independently of ``report``. Asserting that here means a future reader who
    "simplifies" the guard into ``report = {"folds": {}}`` sees the test above fail with the
    IndexError this documents, rather than believing the fix was equivalent.
    """
    from keybo.training.tune import apply_tau_gate

    gated, saturated = apply_tau_gate([], n_groups=None)
    assert gated == [] and saturated is False
    with pytest.raises(IndexError):
        _ = sorted(gated, key=lambda pf: -pf[1])[0][0]


def test_a_SINGLE_candidate_still_works_so_the_guard_is_not_off_by_one() -> None:
    """One candidate is the smallest legitimate input and must be unaffected.

    A guard written as ``len(candidates) < 2`` would pass every test above and break the real
    single-candidate call, so the boundary is pinned rather than assumed. One candidate also
    cannot trip the margin gate, which needs two finite scores — that is expected, not a bug.
    """
    best, leaderboard = tune_lolo(
        _rows(n_participants=6), _PARAMS[:1], seeds=[0], min_cell_samples=4
    )
    assert best == _PARAMS[0]
    assert len(leaderboard) == 1
    assert math.isfinite(leaderboard[0][1])
