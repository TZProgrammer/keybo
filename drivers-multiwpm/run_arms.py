"""MULTIWPM-1 arms: run the five registered objectives over matched seeds.

Every arm calls the SHIPPED search (`SimulatedAnnealing` + `two_opt`, defaults: alpha=0.999,
no max_outer, 2-opt polish ON, start=qwerty) through `keybo.cli.optimize._one_attempt`, so the
only varying factor is the scorer. Search hyperparameters are pinned at their defaults because
the sibling `searchparams` owns that axis.

Arms (see state/multiwpm/PREREGISTRATION.md):
  A control90  single point wpm=90 -- the incumbent
  B mean       mean over the band
  C minimax    max over the band of total_ms/total_ms(qwerty) at that pace
  D rawminimax max over the band of RAW total_ms -- registered to DEMONSTRATE the degeneracy
               (predicted a priori: byte-identical to A). Built by hand because
               RangeBigramScorer REFUSES it, which is the point.
  E point120   single point wpm=120 -- the fast-typist endpoint

Usage: run_arms.py <out.json> [band-csv] [n-seeds]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from keybo.cli.optimize import _one_attempt
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.base import IScorer
from keybo.scoring.range_scorer import RangeBigramScorer
from keybo.scoring.table_scorer import TableBigramScorer

MODELS = "/local/home/zegertho/agent/workspaces/multiwpm/models-inflated"
SEARCH_MODEL = f"{MODELS}/bigram_reg31_seed0.json"
QWERTY = NAMED_LAYOUTS["qwerty"]


class RawMinimaxScorer(IScorer):
    """max over the band of RAW total_ms — the DEGENERATE arm, kept only as a demonstration.

    `RangeBigramScorer` refuses this on purpose (total_ms is monotone decreasing in wpm, so the
    max always lands on the band's lowest pace and the objective IS the single-point one there).
    It is reconstructed here so the degeneracy is MEASURED against arm A rather than asserted.
    """

    def __init__(self, model, freqs, wpms, chars):
        self._s = [TableBigramScorer(model, freqs, target_wpm=w, chars=chars) for w in wpms]

    def fitness(self, layout):
        perm = self._s[0].permutation(layout)
        return float(max(s.fitness_of_permutation(perm) for s in self._s))


def _search_args(seed: int) -> argparse.Namespace:
    """Shipped defaults; only `seed` varies. `_one_attempt` reads exactly these five fields."""
    return argparse.Namespace(
        start=QWERTY,
        alpha=0.999,
        max_outer=None,
        no_local_search=False,
        no_progress=True,
        seed=seed,
    )


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "arms.json"
    band = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "90,100,110,120").split(",")]
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    model = XGBoostTypingModel.load(SEARCH_MODEL)
    freqs = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))

    def mk(kind):
        if kind == "control90":
            return RangeBigramScorer(model, freqs, [90.0], aggregation="endpoint", chars=QWERTY)
        if kind == "point120":
            return RangeBigramScorer(model, freqs, [120.0], aggregation="endpoint", chars=QWERTY)
        if kind == "mean":
            return RangeBigramScorer(model, freqs, band, aggregation="mean", chars=QWERTY)
        if kind == "minimax":
            return RangeBigramScorer(
                model, freqs, band, aggregation="minimax", chars=QWERTY, reference=QWERTY
            )
        if kind == "rawminimax":
            return RawMinimaxScorer(model, freqs, band, QWERTY)
        raise KeyError(kind)

    arms = ["control90", "mean", "minimax", "rawminimax", "point120"]
    # Per-pace evaluators for the winners' curves, on a wider grid than the band so (c) can be
    # read out-of-band too. 130/140 are IN the model's real domain (ceiling ~191-213) but thin.
    eval_wpms = [90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 130.0, 140.0]
    evals = {w: TableBigramScorer(model, freqs, target_wpm=w, chars=QWERTY) for w in eval_wpms}

    results: dict = {
        "band": band,
        "n_seeds": n_seeds,
        "search_model": SEARCH_MODEL,
        "search_defaults": {"alpha": 0.999, "max_outer": None, "two_opt": True, "start": "qwerty"},
        "eval_wpms": eval_wpms,
        "arms": {},
    }

    for arm in arms:
        scorer = mk(arm)
        desc = scorer.describe() if hasattr(scorer, "describe") else "max of RAW total_ms (degenerate)"
        print(f"\n=== arm {arm}: {desc}", flush=True)
        per_seed = []
        for seed in range(n_seeds):
            t0 = time.time()
            best = _one_attempt(_search_args(seed), scorer, seed=seed)
            board = "".join(best.chars)
            curve = {f"{w:g}": float(evals[w].fitness(best)) for w in eval_wpms}
            per_seed.append(
                {
                    "seed": seed,
                    "layout": board,
                    "arm_fitness": float(scorer.fitness(best)),
                    "per_wpm_total_ms": curve,
                    "secs": round(time.time() - t0, 2),
                }
            )
            print(f"  seed {seed}: {board}  fit={scorer.fitness(best):.6g}  ({per_seed[-1]['secs']}s)", flush=True)
        results["arms"][arm] = {"objective": desc, "per_seed": per_seed}

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
