"""Hyperparameter search for the XGBoost typing-time model.

A randomized search over the usual XGBoost knobs, scored by cross-validated MAE. Returns the
best parameter dict (which the ``train`` step can then be given). Uses the modern ``device``
parameter, not the removed ``gpu_hist``/``gpu_id`` (bug #12).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import randint, uniform
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from keybo.verdicts import MarginTooSmall, require_margin

if TYPE_CHECKING:
    from collections.abc import Sequence

_PARAM_DISTRIBUTIONS = {
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.005, 0.1),
    "min_child_weight": randint(1, 6),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 0.5),
    "reg_alpha": uniform(0, 2),
    "reg_lambda": uniform(0, 2),
    "n_estimators": randint(200, 900),
}


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 50,
    cv: int = 5,
    seed: int = 42,
    device: str = "cpu",
    groups: Sequence[str] | np.ndarray | None = None,
) -> dict:
    """Randomized search for XGBoost params minimizing cross-validated MAE.

    Pass ``groups`` (one label per row, normally the layout) to score candidates on splits
    that hold a whole layout out. Without it the split is ungrouped and the reported MAE is
    optimistic by a measured **+0.0349** (positive on 5/5 seeds) — so the ungrouped path warns
    rather than returning a clean-looking number.

    ⚠ Prefer :func:`tune_lolo`. This objective's winners have never been shipped, and its
    believed CV MAE is not comparable across splitters: ``KFold(shuffle=True)`` reports the
    LOWEST MAE of the three options while being the MOST optimistic (+0.0635, 1.76x the
    unshuffled default), so a splitter chosen by reading this number is chosen backwards.
    """
    base = XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,
        device=device,
        random_state=seed,
    )
    if groups is None:
        warnings.warn(
            "tune_hyperparameters is running UNGROUPED: every fold trains and tests on the "
            "same layouts, so the reported CV MAE is optimistic (measured +0.0349, 5/5 seeds) "
            "and must not be compared against a grouped or held-out number. Pass "
            "groups=<one layout label per row> to score by transfer.",
            UnevaluatedObjectiveWarning,
            stacklevel=2,
        )
        splitter: int | GroupKFold = cv
    else:
        groups = np.asarray(groups)
        splitter = grouped_cv(cv, groups)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=splitter,
        refit=True,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y, groups=groups) if groups is not None else search.fit(X, y)
    return dict(search.best_params_)


def tau_resolvable_step(n_groups: int | None) -> float:
    """Smallest Kendall-tau difference that ``n_groups`` ranked items can actually express.

    Kendall tau is ``(concordant - discordant) / total_pairs``, so flipping ONE pair moves it
    by **two** pair-units, i.e. ``4 / (n * (n - 1))`` — not ``2 / (n * (n - 1))``, which is the
    normalized-concordance step and understates the real spacing by half. At n=4 that gives
    **0.3333**, matching the seven achievable values ``{-1, -2/3, -1/3, 0, 1/3, 2/3, 1}``
    enumerated in ``test_kendall_tau_over_four_layouts_takes_only_seven_values``. A "tau edge"
    narrower than one step is not a measurement, it is the same ranking.

    Returns ``0.0`` when ``n_groups`` is unknown or too small to rank, which makes the gate
    fall back to its historical exact-max behaviour rather than silently widening.
    """
    if n_groups is None or n_groups < 2:
        return 0.0
    return 4.0 / (n_groups * (n_groups - 1))


def apply_tau_gate(
    results: list[tuple[dict, float, float]],
    *,
    n_groups: int | None = None,
) -> tuple[list[tuple[dict, float]], bool]:
    """Gate candidates on held-out ranking quality, and REPORT when the gate did nothing.

    The gate keeps a candidate's rho score if its tau is within one *resolvable step* of the
    best tau observed, and sets it to ``-inf`` otherwise. Returns
    ``(gated, tau_was_saturated)``.

    Two failure modes motivated extracting this (TAUGATE-1, ledger ``3620f06``); at 4 layouts
    the old exact-max form had no regime in between:

    * **saturated** — every candidate at tau 1.0, which is the case that has actually run:
      the gate eliminates *nobody* while being described as a ranking guard. It now warns, so
      a leaderboard is never read as tau-filtered when it was not.
    * **tripwire** — one candidate at 1.0 and the rest one inversion lower (0.667 at n=4):
      the old form set the two BEST-rho candidates to ``-inf`` and let the worst rho win. One
      inversion is the finest distinction this frame can draw, so it is treated as a tie.

    A ranking collapse WIDER than one step still gates, so the guard is narrowed, not removed.

    NOTE the gate deliberately reuses ``-inf``, so a gated-out candidate and an unevaluable
    objective look identical here — which is why ``tune_lolo``'s ``n_fracs_finite`` check runs
    BEFORE this point rather than trying to distinguish them from the leaderboard.
    """
    taus = [t for _p, _f, t in results]
    if not taus:
        return [], False
    saturated = len(set(taus)) <= 1 and len(taus) > 1
    if saturated:
        warnings.warn(
            f"the tau gate GATED NOTHING: all {len(taus)} candidates share tau_heldout="
            f"{taus[0]!r}, so every candidate passed and the champion was decided by rho "
            f"alone. A saturated guard reports a pass without checking anything — do not read "
            f"this leaderboard as ranking-filtered.",
            UnevaluatedObjectiveWarning,
            stacklevel=2,
        )
    best_tau = max(taus)
    # A gap of EXACTLY one step is one discordant pair — the finest distinction the frame can
    # draw — so it must be inside the tolerance, not on its edge. Hence +1e-9, not -1e-9.
    tolerance = tau_resolvable_step(n_groups) + 1e-9
    return [
        (p, f if t >= best_tau - tolerance else float("-inf")) for p, f, t in results
    ], saturated


def grouped_cv(cv: int, groups: Sequence[str] | np.ndarray) -> GroupKFold:
    """``GroupKFold`` with ``n_splits`` CLAMPED to the number of distinct groups.

    The clamp is the whole point. ``GroupKFold(cv)`` raises ``ValueError: Cannot have number
    of splits n_splits=5 greater than the number of groups: 4`` — and 5 is the shipped default
    (``cli/tune.py``) while the training frame has 4 layouts, so the obvious "just pass
    GroupKFold" fix converts a silent-optimism bug into a hard crash on the default
    invocation. Clamping is a ceiling, not a rewrite: a ``cv`` below the group count is
    respected.

    ⚠ At ``n_splits == n_groups`` this IS leave-one-group-out, so its zero optimism and zero
    regret-vs-oracle are **definitions, not measurements** (KAGGLE-1 FINAL). Do not quote them
    as evidence that grouping improved anything; the evidence is the ungrouped path's
    optimism, which is measured against an independent honest estimate.
    """
    n_groups = len(set(np.asarray(groups).tolist()))
    if n_groups < 2:
        raise ValueError(
            f"grouped cross-validation needs at least 2 groups, got {n_groups}: a single group "
            f"cannot be held out from itself, and a 1-fold split would silently train and test "
            f"on the same layout — the defect this function exists to prevent"
        )
    return GroupKFold(n_splits=min(cv, n_groups))


#: Smallest RELATIVE margin a lolo selection must clear to be reported as a winner.
#:
#: Derived, not chosen: the score is a mean over folds of ``rho / ceiling``, so a change in the
#: ceiling convention reweights each fold by ``(1 + c) / 2``. Over this ledger's registered
#: ceilings ([0.709, 0.815]) ``reweighting_margin_bound`` gives **0.0301**, and a 400k-pair
#: random search found no ordering flip at a margin above 0.0056 — so the closed form is the
#: conservative side of the empirical one. Rounded to 0.03.
#:
#: The one documented shipped margin (the depth-5-vs-depth-3 comparison in ``tune_lolo``'s
#: docstring, ~0.06 rho/ceiling) is 2.0x this, which is why the shipped selection is robust and
#: this gate is a guard on FUTURE selections rather than a retraction of a past one.
LOLO_MIN_MARGIN = 0.03


class ObjectiveNotEvaluated(RuntimeError):
    """No candidate produced a finite score, so the stated objective never ran.

    ``tune_lolo`` scores candidates by mean held-out rho/ceiling and uses ``-inf`` as the
    "loses the tau gate" sentinel. When the ceiling itself is unobtainable — e.g. every
    layout has one participant, so ``split_half_ceiling`` bisects nothing and returns nan —
    EVERY candidate scores ``-inf`` too, the two states become indistinguishable, and the
    tie-break silently promotes a champion whose objective was never measured.

    Refusing is the right default because the failure is invisible in the output: the
    returned params look like any other recommendation. A caller who genuinely wants the
    tie-broken result can ask for it explicitly.
    """


class UnevaluatedObjectiveWarning(UserWarning):
    """The lolo objective was not evaluable and the refusal was explicitly downgraded."""


def _ceiling_diagnosis(rows, ngram: str) -> str:
    """One sentence naming WHY the ceiling is unobtainable, so the error is actionable.

    ``split_half_ceiling`` needs >= 2 distinct participants per held-out layout. That is the
    dominant cause of an all-nan ceiling column, and it is cheap to check directly, so the
    message states the counts rather than making the reader guess.
    """
    per_layout: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        for _wpm, _duration, pid, _hold in row.samples:
            per_layout[row.layout].add(pid)
    if not per_layout:
        return f"No {ngram} rows were supplied at all."
    counts = sorted(len(pids) for pids in per_layout.values())
    n_ok = sum(1 for c in counts if c >= 2)
    return (
        f"split_half_ceiling bisects PARTICIPANTS and returns nan below 2 of them; "
        f"{n_ok} of {len(counts)} layouts have >= 2 (participants per layout: "
        f"min {counts[0]}, max {counts[-1]})."
    )


def tune_lolo(
    rows,
    candidates: list[dict],
    seeds: list[int],
    ngram: str = "bigram",
    wpm_lo: int = 40,
    wpm_hi: int = 140,
    bucket_width: int = 20,
    min_cell_samples: int = 10,
    allow_unevaluated_objective: bool = False,
    min_margin: float = LOLO_MIN_MARGIN,
    allow_unresolvable_margin: bool = False,
    direction: bool = False,
    kitchensink: bool = False,
) -> tuple[dict, list[tuple[dict, float]]]:
    """Hyperparameter selection scored by TRANSFER, not fit (backlog C1).

    The randomized-CV search above optimizes pooled CV MAE, which *rewards* memorizing
    training-family idiosyncrasies — the exact failure the LOLO harness exists to catch
    (measured: default depth-5 lost ~0.06 rho/ceiling to depth-3 while winning CV fit).
    This selector runs each candidate through the leave-one-layout-out harness and scores
    it by mean held-out rho/ceiling, GATED on the pooled layout-ranking tau staying at
    the maximum achieved by any candidate (a candidate that wins rho by breaking the
    ranking loses; the tau gate is the same principle as the arm-matrix decision rule).

    Returns (best_params, leaderboard) with the leaderboard sorted best-first as
    (params, gated_score) pairs. Candidates are explicit — reproducible and testable;
    callers wanting a random search generate the candidate list themselves.

    Raises ``ObjectiveNotEvaluated`` if NO candidate produced a finite rho/ceiling — see
    that exception's docstring for why refusing beats returning a tie-broken champion.
    Pass ``allow_unevaluated_objective=True`` to downgrade the refusal to a warning; the
    returned leaderboard then carries ``-inf`` scores and the caller MUST treat the
    champion as unselected.

    ``direction`` / ``kitchensink`` select the FRAME every candidate is trained and scored on,
    forwarded verbatim to :func:`keybo.training.validate.validate`. Without them this selector
    could tune only the NARROW served frame (KITCHEN-SINK, 2026-07-31): it called ``validate``
    with no frame argument, so a widened-frame arm had no way to ask for its own hyperparameters
    and would have been compared against a narrow arm tuned on a different frame — a confound in
    the selection step rather than in the measurement. The frame is a property of the ARM, not of
    a candidate, so it is one argument here and not a key in every candidate dict.
    """
    from keybo.training.validate import validate

    results: list[tuple[dict, float, float]] = []  # (params, mean_frac, min_tau)
    n_folds_seen = 0
    n_fracs_finite = 0
    for params in candidates:
        report = validate(
            rows,
            seeds=seeds,
            ngram=ngram,
            wpm_lo=wpm_lo,
            wpm_hi=wpm_hi,
            bucket_width=bucket_width,
            min_cell_samples=min_cell_samples,
            n_boot=10,  # ceilings are shared context here, not the contest
            train_params=params,
            direction=direction,
            kitchensink=kitchensink,
        )
        # A fold contributes only if its rho/ceiling is present AND finite. `None` means the
        # harness could not form the ratio; a non-finite float means it formed one from a nan
        # ceiling. Both are "not measured" and must not average into a score.
        fracs = [
            float(m["rho_frac_ceiling"])
            for fold in report["folds"].values()
            for m in fold["seeds"]
            if m["rho_frac_ceiling"] is not None and math.isfinite(float(m["rho_frac_ceiling"]))
        ]
        n_folds_seen += sum(len(f["seeds"]) for f in report["folds"].values())
        n_fracs_finite += len(fracs)
        taus = [p["tau_heldout"] for p in report["pooled"]]
        mean_frac = float(np.mean(fracs)) if fracs else float("-inf")
        min_tau = float(min(taus)) if taus else float("-inf")
        results.append((params, mean_frac, min_tau))

    if n_fracs_finite == 0:
        ceilings = _ceiling_diagnosis(rows, ngram)
        message = (
            f"the lolo objective was never evaluated: 0 of {n_folds_seen} "
            f"(fold x seed) cells across {len(candidates)} candidates produced a finite "
            f"rho/ceiling, so every candidate scored -inf and the tau gate alone would "
            f"decide the champion. {ceilings} Fix the data or lower --min-samples; do not "
            f"read the returned params as selected by transfer."
        )
        if not allow_unevaluated_objective:
            raise ObjectiveNotEvaluated(message)
        warnings.warn(message, UnevaluatedObjectiveWarning, stacklevel=2)

    gated, tau_saturated = apply_tau_gate(results, n_groups=len(report["folds"]) or None)
    # The saturation flag was previously discarded into `_saturated`, so a leaderboard produced by a
    # gate that eliminated NOBODY was indistinguishable from a tau-filtered one. It is now recorded on
    # the report, which is the smallest change that makes the fact readable by a caller (TAUGATE-1).
    if isinstance(report, dict):
        report["tau_gate_saturated"] = bool(tau_saturated)
    leaderboard = sorted(gated, key=lambda pf: -pf[1])

    # Minimum-margin gate. The score is a mean over folds of rho/ceiling, so a change in how
    # the ceiling is computed reweights the folds; a win decided by less than that reweighting
    # can move is a convention artifact, not a measurement. The default bound is derived from
    # THIS ledger's registered ceilings via reweighting_margin_bound; pass min_margin=0.0 to
    # disable (e.g. reproducing a historical selection).
    finite = [s for _p, s in leaderboard if math.isfinite(s)]
    if len(finite) >= 2 and min_margin > 0.0:
        try:
            require_margin(finite, "lolo hyperparameter selection", min_margin=min_margin)
        except MarginTooSmall as exc:
            if not allow_unresolvable_margin:
                raise
            warnings.warn(str(exc), UnevaluatedObjectiveWarning, stacklevel=2)

    return leaderboard[0][0], leaderboard
