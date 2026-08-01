"""Shared harness for the restart/schedule power analysis (searchparams).

Loads the SHIPPED default objective exactly as `keybo optimize` does:
  --ngram bigram (default), --target-wpm 90 (default), table fast path (default),
  --start qwerty (default), 2-opt polish (default on).

Everything here is read-only w.r.t. the repo; results go to state/searchparams/artifacts.
"""

from __future__ import annotations

import time

from keybo.analysis.timecard import _load_gz_model, default_surface
from keybo.cli._scorer import load_freqs
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.optimize.annealing import SimulatedAnnealing, stopping_point
from keybo.optimize.local_search import two_opt
from keybo.scoring.table_scorer import TableBigramScorer

START = NAMED_LAYOUTS["qwerty"]  # cli default --start
TARGET_WPM = 90.0  # cli default --target-wpm


def build_search_scorer(target_wpm: float = TARGET_WPM, start: str = START):
    """The EXACT scorer `keybo optimize` searches with, on the shipped default flags.

    cli/optimize.py: --ngram bigram + not --no-table + no comfort/oxey/finger-load
    -> TableBigramScorer(XGBoostTypingModel.load(args.model), load_freqs(args), target_wpm, chars=args.start)
    The shipped model artifacts are gzipped, so we inflate via the same helper analyze uses.
    """
    model = _load_gz_model("bigram_reg31_seed0")
    freqs = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
    return TableBigramScorer(model, freqs, target_wpm=target_wpm, chars=start)


def ms_per_char(lay30: str, target_wpm: float = TARGET_WPM) -> float:
    """The campaign's REPORTING gauge (the ruler the 0.135 floor is defined on).

    This is analyze's ms/char: the K31 trigram TimeSurface (T2 + Tc, 3-seed mean) over the
    production trigram corpus.  It is NOT the search objective -- reporting both is the point.
    """
    return default_surface(target_wpm).card(lay30).ms_per_char


class InstrumentedSA(SimulatedAnnealing):
    """SimulatedAnnealing + counters. Overrides NOTHING that affects the search path.

    `cool` and `optimize` are inherited verbatim; we only count calls to `cool` (one per
    outer iteration) and record T0. No core file is modified.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.outer_count = 0
        self.t0 = None
        self.t_final = None

    def estimate_initial_temperature(self, layout, scorer, acceptance=None, samples=1000):
        t0 = super().estimate_initial_temperature(layout, scorer, acceptance, samples)
        if self.t0 is None:
            self.t0 = t0
        return t0

    def cool(self, temperature):
        self.outer_count += 1
        self.t_final = out = super().cool(temperature)
        return out


def one_attempt(
    scorer,
    seed: int,
    alpha: float = 0.999,
    max_outer: int | None = None,
    local_search: bool = True,
    start: str = START,
    instrumented: bool = True,
):
    """One `_one_attempt` equivalent, with wall clock split SA / 2-opt.

    Mirrors cli/optimize.py::_one_attempt exactly: a FRESH Layout per attempt, SA, then
    (unless --no-local-search) the 2-opt polish.
    """
    layout = Layout(start, ROW_STAGGERED_30)
    cls = InstrumentedSA if instrumented else SimulatedAnnealing
    sa = cls(seed=seed, alpha=alpha, max_outer=max_outer, progress=False)
    t0 = time.perf_counter()
    best = sa.optimize(layout, scorer)
    t_sa = time.perf_counter() - t0
    fit_sa = scorer.fitness(best)
    t1 = time.perf_counter()
    if local_search:
        best = two_opt(best, scorer)
    t_ls = time.perf_counter() - t1
    fit = scorer.fitness(best)
    rec = {
        "seed": seed,
        "alpha": alpha,
        "max_outer": max_outer,
        "local_search": local_search,
        "layout": "".join(best.chars),
        "fitness": fit,
        "fitness_sa_only": fit_sa,
        "sec_sa": t_sa,
        "sec_2opt": t_ls,
        "sec": t_sa + t_ls,
    }
    if instrumented:
        rec.update(outer_count=sa.outer_count, t0=sa.t0, t_final=sa.t_final)
    return rec


def stop_budget(start: str = START) -> dict:
    key_count = len(start)
    return {
        "key_count": key_count,
        "stopping_point": stopping_point(key_count),
        "inner_per_outer": key_count,
        "outer_without_best_to_stop": stopping_point(key_count) / key_count,
    }
