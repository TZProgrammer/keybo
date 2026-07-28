"""Hyperparameter search for the XGBoost typing-time model.

A randomized search over the usual XGBoost knobs, scored by cross-validated MAE. Returns the
best parameter dict (which the ``train`` step can then be given). Uses the modern ``device``
parameter, not the removed ``gpu_hist``/``gpu_id`` (bug #12).
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import randint, uniform
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

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
) -> dict:
    """Randomized search for XGBoost params minimizing cross-validated MAE."""
    base = XGBRegressor(
        objective="reg:squarederror",
        verbosity=0,
        device=device,
        random_state=seed,
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X, y)
    return dict(search.best_params_)


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
        )
        # A fold contributes only if its rho/ceiling is present AND finite. `None` means the
        # harness could not form the ratio; a non-finite float means it formed one from a nan
        # ceiling. Both are "not measured" and must not average into a score.
        fracs = [
            float(m["rho_frac_ceiling"])
            for fold in report["folds"].values()
            for m in fold["seeds"]
            if m["rho_frac_ceiling"] is not None
            and math.isfinite(float(m["rho_frac_ceiling"]))
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

    best_tau = max(r[2] for r in results)
    # tau gate: only candidates achieving the best observed ranking quality compete on rho.
    # NOTE the gate deliberately reuses -inf, so a gated-out candidate and an unevaluable
    # objective look identical HERE — which is why the n_fracs_finite check above must run
    # BEFORE this point rather than trying to distinguish them from the leaderboard.
    gated = [(p, f if t >= best_tau - 1e-9 else float("-inf")) for p, f, t in results]
    leaderboard = sorted(gated, key=lambda pf: -pf[1])
    return leaderboard[0][0], leaderboard
