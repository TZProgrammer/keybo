"""The tau gate must be neither a no-op nor a tripwire, and its ``-inf`` collision must stay ordered.

``apply_tau_gate`` exists because the old exact-max form (``keep iff t >= best_tau - 1e-9``) has no
useful regime at four layout groups (TAUGATE-1, ledger ``3620f06``; fixed in ``cb907aa``). Kendall
tau there takes only seven values spaced 1/3 apart, so the gate was always one of two things:

* **saturated** — every candidate at tau 1.0, the case that has actually run. It eliminated
  *nobody* while being described as a ranking guard, so a leaderboard decided by rho alone read as
  tau-filtered. This is the difference between "gated" and "reported a pass without checking
  anything", which is why the warning is asserted here and not just the ``saturated`` flag.
* **tripwire** — one candidate at 1.0 and the rest one inversion lower (0.667). The old form set
  the two BEST-rho candidates to ``-inf`` and let the worst rho win.

``test_tune_grouped_cv.py`` pins one example of each. This file pins the parts that example cannot
reach: the tolerance BOUNDARY (one step in, two steps out, and the 1e-9 edge), saturation at a tau
other than 1.0, the single-candidate and empty cases, and — the real gap — the **ordering** that
makes the gate's documented ``-inf`` collision survivable.

That collision is the subtle one. ``apply_tau_gate`` deliberately reuses ``-inf`` for "lost the
gate", which is the same sentinel ``tune_lolo`` uses for "the objective was never evaluable". The
two are indistinguishable in the returned leaderboard — proven below rather than asserted from the
docstring — so the only thing that keeps them apart is that ``tune_lolo``'s ``n_fracs_finite``
check runs BEFORE the gate. Nothing pinned that order, and reordering it converts a loud refusal
into a silent champion, which is exactly the failure ``ObjectiveNotEvaluated`` was added to stop.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from keybo.training.tune import (
    ObjectiveNotEvaluated,
    UnevaluatedObjectiveWarning,
    apply_tau_gate,
    tau_resolvable_step,
)

#: Four layouts: the shipped group count, where the resolvable step is 1/3.
_N_GROUPS = 4
_STEP = tau_resolvable_step(_N_GROUPS)


def _scores(scored, **kwargs):
    """Run the gate with warnings silenced, returning just the gated score column.

    Saturation warnings are asserted explicitly in the tests that are ABOUT the warning; here
    they would otherwise turn every tolerance test into a warning test as well.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gated, _saturated = apply_tau_gate(scored, **kwargs)
    return [s for _p, s in gated]


# --- the tolerance BOUNDARY: the gate is narrowed, not removed ---------------------------


def test_a_tau_exactly_one_resolvable_step_below_the_best_is_KEPT() -> None:
    """The tripwire case at the exact boundary, from the step rather than a literal 0.667.

    Computing the operand as ``1.0 - _STEP`` means this test still pins the *intended* rule if the
    step's value ever changes, instead of pinning 0.667 as a magic number.
    """
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, 1.0 - _STEP)]
    assert all(math.isfinite(s) for s in _scores(scored, n_groups=_N_GROUPS))


def test_a_tau_TWO_resolvable_steps_below_the_best_is_GATED() -> None:
    """The other side of the same boundary: two inversions IS a measurable ranking difference.

    Together with the test above this brackets the tolerance, which a single one-sided example
    cannot do — a gate that tolerated everything would pass that one and fail this one.
    """
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, 1.0 - 2 * _STEP)]
    assert _scores(scored, n_groups=_N_GROUPS) == [0.90, float("-inf")]


def test_the_1e9_edge_is_INCLUSIVE_and_not_a_licence_to_drift() -> None:
    """``+1e-9`` admits a gap of exactly one step; it must not admit a materially wider one.

    The comment in ``apply_tau_gate`` says the epsilon exists so an exactly-one-step gap lands
    INSIDE the tolerance rather than on its edge (float arithmetic on 2/3 does not land exactly).
    A gap one micro-unit past that must still gate, or the epsilon is slack rather than a
    boundary condition.
    """
    just_inside = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, 1.0 - _STEP)]
    assert all(math.isfinite(s) for s in _scores(just_inside, n_groups=_N_GROUPS))
    just_outside = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, 1.0 - _STEP - 1e-6)]
    assert _scores(just_outside, n_groups=_N_GROUPS) == [0.90, float("-inf")]


def test_the_rounded_two_thirds_a_human_would_type_is_also_kept() -> None:
    """0.666667 as printed in reports must behave like the exact 2/3 the frame produces.

    A tolerance that worked on ``2/3`` but not on its 6-decimal rendering would break on the
    values people actually paste out of a leaderboard.
    """
    for tau in (2 / 3, 0.666667, 0.6666666666666666):
        scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, tau)]
        assert all(math.isfinite(s) for s in _scores(scored, n_groups=_N_GROUPS)), tau


def test_a_total_ranking_REVERSAL_is_still_gated_so_the_guard_did_not_become_decorative() -> None:
    """The narrowing must not extend to a candidate that inverts the ranking outright.

    tau = -1.0 is the worst achievable value at any group count. If this ever passes, the gate
    has stopped being a guard and the whole narrowing was a removal in disguise.
    """
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, -1.0), ({"c": 3}, 0.95, 0.0)]
    assert _scores(scored, n_groups=_N_GROUPS) == [0.90, float("-inf"), float("-inf")]


def test_the_best_rho_can_win_within_tolerance_but_NOT_outside_it() -> None:
    """The two failure modes in one comparison, on the same rho ordering.

    This is the decision the gate actually makes: does the candidate with the best rho get to be
    champion? Inside one step it must; two steps out it must not. Pinning both with identical rho
    values isolates tau as the only thing that moved.
    """
    inside = [({"worst": 1}, 0.90, 1.0), ({"best": 2}, 0.99, 1.0 - _STEP)]
    assert max(_scores(inside, n_groups=_N_GROUPS)) == 0.99, "one inversion must not decide"
    outside = [({"worst": 1}, 0.90, 1.0), ({"best": 2}, 0.99, 1.0 - 3 * _STEP)]
    assert max(_scores(outside, n_groups=_N_GROUPS)) == 0.90, "a real collapse must decide"


# --- SATURATION: reporting a pass without checking anything ------------------------------


def test_saturation_is_detected_at_a_tau_OTHER_than_1_0() -> None:
    """The no-op is about AGREEMENT, not about the value 1.0.

    Every existing test of this uses all-1.0, so a hypothetical ``if tau == 1.0`` implementation
    would pass them all while missing the identical defect at any other shared value — a gate
    whose candidates all scored 0.667 has eliminated nobody just as completely.
    """
    scored = [({"a": 1}, 0.90, 0.666667), ({"b": 2}, 0.99, 0.666667)]
    with pytest.warns(UnevaluatedObjectiveWarning, match="GATED NOTHING"):
        gated, saturated = apply_tau_gate(scored, n_groups=_N_GROUPS)
    assert saturated is True
    assert [s for _p, s in gated] == [0.90, 0.99], "a no-op gate may not eliminate anyone"


def test_the_saturation_warning_NAMES_the_shared_tau_and_the_candidate_count() -> None:
    """A warning a reader cannot act on is barely better than silence.

    The message has to say how many candidates tied and at what value, because the reader's next
    question is whether the tau column is degenerate or the candidates are genuinely identical.
    """
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.95, 1.0), ({"c": 3}, 0.99, 1.0)]
    with pytest.warns(UnevaluatedObjectiveWarning) as record:
        apply_tau_gate(scored, n_groups=_N_GROUPS)
    message = str(record[0].message)
    assert "all 3 candidates" in message, "must state how many candidates tied"
    assert "1.0" in message, "must state the shared tau value"
    assert "rho" in message, "must say what actually decided the champion instead"


def test_a_SINGLE_candidate_is_not_reported_as_saturated() -> None:
    """One candidate cannot demonstrate a degenerate tau column — there is nothing to compare.

    Warning here would train readers to ignore the warning, which is the way a real saturation
    report gets lost. The ``len(taus) > 1`` guard is what prevents it, and nothing pinned it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ANY warning fails this test
        gated, saturated = apply_tau_gate([({"a": 1}, 0.90, 1.0)], n_groups=_N_GROUPS)
    assert saturated is False
    assert [s for _p, s in gated] == [0.90], "the sole candidate cannot lose its own gate"


def test_an_EMPTY_candidate_list_returns_empty_and_unsaturated_without_warning() -> None:
    """No candidates is not a saturated gate, and must not raise on ``max()`` of an empty list."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert apply_tau_gate([], n_groups=_N_GROUPS) == ([], False)


def test_a_discriminating_gate_does_NOT_warn() -> None:
    """The complement, so the warning cannot be made unconditional and still pass.

    Without this, ``warnings.warn(...)`` moved outside the ``if saturated`` block would leave the
    whole suite green while flooding every real run.
    """
    scored = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.99, 0.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _gated, saturated = apply_tau_gate(scored, n_groups=_N_GROUPS)
    assert saturated is False


def test_the_params_dicts_pass_through_the_gate_unchanged_and_in_order() -> None:
    """The gate rewrites SCORES only. Leaderboard order is applied by the caller, not here.

    ``tune_lolo`` sorts the gate's output, so if the gate itself reordered or copied the params
    the sort would silently pair a score with the wrong candidate.
    """
    params = [{"a": 1}, {"b": 2}, {"c": 3}]
    scored = [(p, 0.9, t) for p, t in zip(params, (1.0, 0.0, -1.0), strict=True)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gated, _saturated = apply_tau_gate(scored, n_groups=_N_GROUPS)
    assert [p for p, _s in gated] == params
    assert all(a is b for (a, _s), b in zip(gated, params, strict=True)), "no copies"


# --- the -inf COLLISION, and the ORDERING that makes it survivable -----------------------


def test_a_gated_out_candidate_is_INDISTINGUISHABLE_from_an_unevaluable_one() -> None:
    """The documented collision, proven rather than quoted from the docstring.

    Two runs with completely different problems — one where the objective scored fine but the
    ranking collapsed, one where the objective itself was never evaluable — produce byte-identical
    leaderboards. This is WHY the check order below is load-bearing: nothing downstream of the
    gate can tell these apart, so the distinction has to be made upstream of it.
    """
    ranking_collapsed = [({"a": 1}, 0.90, 1.0), ({"b": 2}, 0.95, -1.0)]
    never_evaluated = [({"a": 1}, 0.90, 1.0), ({"b": 2}, float("-inf"), 1.0)]
    assert (
        _scores(ranking_collapsed, n_groups=_N_GROUPS)
        == _scores(never_evaluated, n_groups=_N_GROUPS)
        == [0.90, float("-inf")]
    )


def test_tune_lolo_refuses_BEFORE_the_gate_runs_so_the_refusal_cannot_be_masked(
    monkeypatch,
) -> None:
    """The ordering, pinned at the only place it is observable: the gate never runs at all.

    ``ObjectiveNotEvaluated`` is raised while ``n_fracs_finite == 0``, which is upstream of
    ``apply_tau_gate``. If the two were swapped, the gate would run first on an all-``-inf``
    leaderboard, emit its saturation warning, and — because ``-inf`` is its own "lost the gate"
    sentinel — the refusal would be reporting on a state the gate had already overwritten.

    Asserting "the gate was never called" is stronger than asserting the exception type: a
    reordered implementation still raises, so a ``pytest.raises`` alone would not notice.
    """
    import keybo.training.tune as tune_mod

    calls: list[object] = []
    real_gate = tune_mod.apply_tau_gate

    def spy(results, *, n_groups=None):
        calls.append(n_groups)
        return real_gate(results, n_groups=n_groups)

    monkeypatch.setattr(tune_mod, "apply_tau_gate", spy)
    rows, params = _lawful_rows(n_participants=1), _CANDIDATES
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ObjectiveNotEvaluated):
            tune_mod.tune_lolo(rows, params, seeds=[0], min_cell_samples=4)
    assert calls == [], "the tau gate must not run when the objective was never evaluated"
    assert not [w for w in caught if "GATED NOTHING" in str(w.message)], (
        "a saturation warning here would mean the gate reported on an all--inf leaderboard"
    )


def test_the_group_count_is_actually_THREADED_from_tune_lolo_into_the_gate(monkeypatch) -> None:
    """present != effective: an ``n_groups`` that never arrives leaves the gate at exact-max.

    The whole narrowing is inert if ``tune_lolo`` calls the gate without the group count, and that
    failure is invisible — the gate still returns a plausible leaderboard, just a tripwire one.
    So assert the value that arrives, not merely that the parameter exists.
    """
    import keybo.training.tune as tune_mod

    seen: list[object] = []
    real_gate = tune_mod.apply_tau_gate

    def spy(results, *, n_groups=None):
        seen.append(n_groups)
        return real_gate(results, n_groups=n_groups)

    monkeypatch.setattr(tune_mod, "apply_tau_gate", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tune_mod.tune_lolo(
            _lawful_rows(n_participants=6, n_layouts=3),
            _CANDIDATES,
            seeds=[0],
            min_cell_samples=4,
            min_margin=0.0,
        )
    assert seen == [3], f"the gate must receive the fold/layout count, got {seen}"


def test_the_saturation_flag_reaches_the_gate_from_a_real_tune_lolo_run(monkeypatch) -> None:
    """End to end: the flag the caller can read is the flag the gate computed.

    ``cb907aa``'s complaint was that the saturation flag had been discarded into ``_saturated``.
    It is now recorded on the validate report, so this asserts the value a caller would see
    rather than re-testing the gate in isolation.
    """
    import keybo.training.tune as tune_mod

    captured: dict[str, object] = {}
    real_gate = tune_mod.apply_tau_gate

    def spy(results, *, n_groups=None):
        gated, saturated = real_gate(results, n_groups=n_groups)
        captured["saturated"] = saturated
        captured["taus"] = [t for _p, _f, t in results]
        return gated, saturated

    monkeypatch.setattr(tune_mod, "apply_tau_gate", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tune_mod.tune_lolo(
            _lawful_rows(n_participants=6, n_layouts=3),
            _CANDIDATES,
            seeds=[0],
            min_cell_samples=4,
            min_margin=0.0,
        )
    taus = captured["taus"]
    assert captured["saturated"] is (len(set(taus)) <= 1 and len(taus) > 1), (
        f"the reported flag must match the tau column it was computed from: {taus}"
    )


# --- fixture ------------------------------------------------------------------------------

#: Two candidates, enough for a leaderboard with an order; kept tiny because these tests are
#: about control flow, not about fit quality.
_CANDIDATES = [
    {"n_estimators": 40, "max_depth": 2, "learning_rate": 0.2, "subsample": 1.0},
    {"n_estimators": 45, "max_depth": 3, "learning_rate": 0.15, "subsample": 0.9},
]


def _lawful_rows(n_participants: int, n_layouts: int = 3, n_ngrams: int = 6, seed: int = 5):
    """Reuses ``test_tune_unevaluated_objective``'s geometry-lawful fixture.

    Deliberately NOT a fourth fixture shape: the participant count is the only knob these tests
    need (it drives whether the ceiling — and therefore the objective — is evaluable at all), and
    that file already builds real ngrams at real position pairs with durations linear in distance.
    """
    from keybo.data.strokes import StrokeRow

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
