"""Simulated annealing over layouts.

The algorithm follows the original project's design, which is sound:

- **Initial temperature** is estimated so that the average uphill (worsening) swap is
  accepted with probability ``acceptance`` — i.e. ``T0 = -avg(positive delta) / ln(x0)``.
- **Cooling** is geometric: ``T <- alpha * T``.
- **Stopping** uses the coupon-collector expectation for the number of swaps needed to have
  probably examined every key pair: ``ceil(C(k,2)*(ln C(k,2) + gamma) + 0.5)`` consecutive
  non-improving iterations.

The one substantive change is reproducibility: all randomness comes from a seeded
``random.Random`` (bug #11 — the old code never seeded), so a run is repeatable.
"""

from __future__ import annotations

import random
from math import ceil, exp, log

from keybo.layout import Layout
from keybo.optimize.base import IOptimizer
from keybo.scoring.base import IScorer

_EULER_MASCHERONI = 0.5772156649


def stopping_point(key_count: int) -> int:
    """Consecutive non-improving iterations before we assume convergence."""
    pairs = key_count * (key_count - 1) // 2
    return ceil(pairs * (log(pairs) + _EULER_MASCHERONI) + 0.5)


class SimulatedAnnealing(IOptimizer):
    def __init__(
        self,
        seed: int | None = None,
        alpha: float = 0.999,
        acceptance: float = 0.8,
        max_outer: int | None = None,
        progress: bool = False,
    ):
        self.alpha = alpha
        self.acceptance = acceptance
        # Hard cap on outer (cooling) iterations, so a plateau landscape can't run forever.
        self.max_outer = max_outer
        # Show a tqdm convergence bar on stderr (stdout stays clean for parseable output).
        # Progress is measured by `stays` filling toward the stopping point -- the quantity
        # that actually terminates the run -- so the bar resets when a new best is found:
        # honest annealing behavior, not a fake linear ETA.
        self.progress = progress
        self._rng = random.Random(seed)

    def cool(self, temperature: float) -> float:
        return temperature * self.alpha

    def estimate_initial_temperature(
        self, layout: Layout, scorer: IScorer, acceptance: float | None = None, samples: int = 1000
    ) -> float:
        """Estimate T0 so the mean uphill swap is accepted with probability ``acceptance``."""
        acceptance = self.acceptance if acceptance is None else acceptance
        base = scorer.fitness(layout)
        chars = list(layout.chars)
        n = len(chars)
        all_pairs = [(chars[i], chars[j]) for i in range(n) for j in range(i + 1, n)]
        sample_pairs = self._rng.sample(all_pairs, min(samples, len(all_pairs)))

        uphill = []
        for k1, k2 in sample_pairs:
            layout.swap(k1, k2)
            delta = scorer.fitness(layout) - base
            layout.undo()
            if delta > 0:
                uphill.append(delta)

        if not uphill:
            return 1.0
        avg_delta = sum(uphill) / len(uphill)
        return -avg_delta / log(acceptance)

    def optimize(self, layout: Layout, scorer: IScorer) -> Layout:
        key_count = len(layout.chars)
        stop = stopping_point(key_count)
        temperature = self.estimate_initial_temperature(layout, scorer)

        current = scorer.fitness(layout)
        best_fitness = current
        best_chars = "".join(layout.chars)

        # `stays` counts iterations since the last improvement to the GLOBAL best. Basing
        # convergence on the best (which is monotonic non-increasing and bounded below)
        # rather than on accepted moves guarantees termination even on a plateau or when
        # uphill/lateral moves are accepted for exploration -- the original counted accepted
        # moves and could never converge when equal-fitness swaps kept being "accepted".
        stays = 0
        outer = 0
        bar = None
        if self.progress:
            from tqdm import tqdm

            bar = tqdm(total=stop, desc="annealing (convergence)", unit="stay", leave=False)
        try:
            while stays < stop:
                if self.max_outer is not None and outer >= self.max_outer:
                    break
                outer += 1
                for _ in range(key_count):
                    layout.random_swap(self._rng)
                    candidate = scorer.fitness(layout)
                    delta = candidate - current

                    if delta <= 0 or self._rng.random() < exp(-delta / temperature):
                        # Accept improvements and equal moves outright; accept worsening moves
                        # with the Metropolis probability to escape local minima.
                        current = candidate
                    else:
                        layout.undo()

                    if current < best_fitness:
                        best_fitness = current
                        best_chars = "".join(layout.chars)
                        stays = 0
                    else:
                        stays += 1

                temperature = self.cool(temperature)
                if bar is not None:
                    # One cheap update per outer iteration: sync the bar to the convergence
                    # counter (it can move backward when a new best resets `stays`).
                    bar.n = min(stays, stop)
                    bar.set_postfix_str(f"best={best_fitness:.0f} T={temperature:.3g}")
                    bar.refresh()
        finally:
            if bar is not None:
                bar.close()

        return Layout(best_chars, layout.geometry)
