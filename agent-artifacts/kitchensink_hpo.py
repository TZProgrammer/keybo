"""KITCHEN-SINK hyperparameter tuning — transfer-scored, on the kitchen-sink frame.

Uses ``tune_lolo`` (mean held-out rho/ceiling, tau-gated), NOT ``tune_hyperparameters``: the latter's
CV is ungrouped and optimistic by a measured +0.0349, its winners have never shipped, and
``shuffle=True`` is 1.76x worse while reporting the BEST cv number (KAGGLE-1 FINAL).

This run is only possible because ``tune_lolo`` gained a frame argument in this branch. Before that
it called ``validate()`` with no frame flag, so it could tune only the NARROW served frame — a
widened arm had no way to ask for its own hyperparameters, and comparing an untuned widened arm
against a tuned narrow one would have put the confound in the SELECTION step rather than the
measurement.

Two guards are deliberately left ON and their firing is a REPORTABLE RESULT, not an obstacle:

* ``min_margin`` (LOLO_MIN_MARGIN = 0.03) — the score is a mean over folds of rho/ceiling, so a
  change in the ceiling convention reweights folds by (1+c)/2; a win narrower than that bound is a
  convention artifact. If it raises, the honest output is "no candidate is resolvably best".
* the tau gate — expected to be SATURATED at 4 layouts (tau takes only 7 values, 1/3 apart:
  TAUGATE-1). A saturated gate eliminates nobody while reading as a ranking guard, so its warning is
  captured and reported rather than suppressed.

    PYTHONPATH=src OMP_NUM_THREADS=1 python agent-artifacts/kitchensink_hpo.py \
        --strokes /path/bistrokes31_v1.tsv --ngram bigram \
        --out agent-artifacts/ks_hpo_bigram.json
"""

from __future__ import annotations

import argparse
import json
import warnings

from keybo.data.strokes import load_strokes
from keybo.training.tune import (
    LOLO_MIN_MARGIN,
    ObjectiveNotEvaluated,
    tune_lolo,
)
from keybo.verdicts import MarginTooSmall

#: The registered grid: 4 depths x 2 tree counts x 2 learning rates = 16 explicit candidates.
#: ``min_child_weight`` and ``subsample`` are held at the shipped values — two of five knobs fixed
#: keeps the search inside what 4 folds can resolve. max_depth 3 is the shipped value
#: (goodhart-row-blindness: depth 3 bought the LOLO win that removing the row one-hots would have
#: bought, without deleting information the optimizer needs).
GRID = [
    {"max_depth": d, "n_estimators": n, "learning_rate": lr, "min_child_weight": 1}
    for d in (2, 3, 4, 5)
    for n in (200, 400)
    for lr in (0.05, 0.10)
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strokes", required=True)
    ap.add_argument("--ngram", choices=["bigram", "trigram"], default="bigram")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--frame",
        choices=["narrow", "widened", "kitchensink"],
        default="kitchensink",
        help="which feature frame every candidate is trained and scored on",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    rows = load_strokes(
        args.strokes,
        ngram_len=2 if args.ngram == "bigram" else 3,
        wpm_threshold=0,
        min_samples=1,
    )
    print(f"loaded {len(rows)} rows, {len({r.layout for r in rows})} layouts", flush=True)

    flags = {
        "narrow": {},
        "widened": {"direction": True},
        "kitchensink": {"kitchensink": True},
    }[args.frame]

    result: dict = {
        "ngram": args.ngram,
        "frame": args.frame,
        "seeds": args.seeds,
        "n_candidates": len(GRID),
        "grid": GRID,
        "min_margin": LOLO_MIN_MARGIN,
        "n_rows": len(rows),
        "layouts": sorted({r.layout for r in rows}),
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            best, leaderboard = tune_lolo(
                rows,
                candidates=GRID,
                seeds=args.seeds,
                ngram=args.ngram,
                **flags,
            )
            result["best_params"] = best
            result["leaderboard"] = [
                {"params": p, "score": (None if s == float("-inf") else s)} for p, s in leaderboard
            ]
            result["selected"] = True
        except (ObjectiveNotEvaluated, MarginTooSmall) as exc:
            # A refusal IS the result. The guard exists because a champion chosen inside the
            # resolvable margin is indistinguishable from a real one once it is a row in a table.
            result["selected"] = False
            result["refusal_type"] = type(exc).__name__
            result["refusal"] = str(exc)
            print(f"REFUSED by design: {type(exc).__name__}: {exc}", flush=True)
        result["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"-> {args.out}", flush=True)
    if result.get("selected"):
        print(f"best: {result['best_params']}", flush=True)
        for row in result["leaderboard"][:5]:
            print(f"  {row['score']}  {row['params']}", flush=True)
    for w in result["warnings"]:
        print(f"WARNING captured: {w[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
