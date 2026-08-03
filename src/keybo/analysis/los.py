"""LOS — Likelihood Of Superiority for a paired layout-vs-layout speed comparison.

The question this answers is the user's: *"a 0.01 ms/char difference is not significant since the
stddev of our models is so high — I want the CONFIDENCE that one layout is faster than another, like
fishtest."* It is deliberately NOT a p-value. A p-value answers "is the mean margin nonzero", and
this project has a measured pathology where that question returns p=1.7e-04 on a margin smaller than
the instrument's own resolution and even flips sign across pricings (TOURNAMENT-1). An instrument that
returns 0.99 on such a margin launders noise as confidence and is worse than no instrument.

WHAT LOS IS, PRECISELY
----------------------
Our per-seed margin ``d_s = M_s(A) - M_s(B)`` (ms/char; negative = A faster) is DETERMINISTIC given
(board, model seed) — there are no independent games to accumulate the way fishtest accumulates
them. Our only replication is the model-training seed (the retrain RNG). So the paired sample is the
``n`` per-seed margins over a COMMON seed set, and with a common seed set the margin is exactly linear
(``mean_s d_s = M(A) - M(B)``), which makes the speed "faster-than" relation a sub-relation of a total
order — no intransitivity is possible and every LOS here is monotone in the mean margin.

Three estimands (all computed; they give different numbers and the DROP between them is the point):

All three are directional "confidence that A (the first argument) is faster than B", so their
equipoise is 0.5 and ``LOS(B vs A) = 1 - LOS(A vs B)`` exactly (fishtest's LOS is directional and
complementary the same way).

* :attr:`LOSResult.los_seed`   — P(A faster), SEED NOISE ONLY: the one-sided flat-prior posterior
  mass ``P(mu < 0)`` from the paired-t on the per-seed margins. 0.5 on a null, ~1 on any consistent
  margin however tiny — which is exactly why it ALONE reproduces the pathology. Reported as the
  upper bound / decomposition baseline, never as the answer.
* :attr:`LOSResult.los_design` — the PRIMARY. The design's measured RESOLUTION FLOOR is used as a
  Region Of Practical Equivalence (the Bayesian form of fishtest's INTERVAL hypothesis): the
  posterior of the true mean margin splits into three regions — A meaningfully faster
  ``P(mu < -floor)``, a TIE ``P(|mu| <= floor)``, and B meaningfully faster ``P(mu > +floor)`` — and
  ``los_design = P(mu < -floor) + 0.5 * P(|mu| <= floor)`` (the tie mass split evenly, i.e. equipoise
  on an unresolvable difference). Consequences, all structural rather than tuned: a NULL returns 0.5;
  a SUB-FLOOR margin returns ~0.5 no matter how tiny its p-value (all its mass is in the tie region),
  so ``los_design`` CANNOT return 0.99 on a within-resolution difference; a margin far beyond the
  floor returns ~1. This is the anti-pathology guarantee.
* :attr:`LOSResult.p_exceed`   — companion, NOT a sign confidence: ``P(|mu| > floor)`` = the
  probability the difference is LARGER than the instrument's resolution at all, in either direction.
  This is estimand (c) ("confidence the margin exceeds a meaningful threshold"); it answers "is there
  a resolvable difference" while ``los_design`` answers "which way, treating sub-floor as a tie".
* :attr:`LOSResult.los_typist` — degrades ``los_design`` by an extrapolation WRONG-SIGN HAZARD
  ``q(gap)`` (the measured co-observed sign-flip rate: on the ground the model actually observed,
  what fraction of pairwise speed signs reverse). Enters as a mixture
  ``(1-q)*los_design + q*(1-los_design)``, whose registered property is ``q -> 0.5  =>  los -> 0.5``
  for ANY input. An 81%-flip-hazard margin cannot be turned into confidence.

NOTE ON THE PREREGISTRATION: PREREG §3 literally defined ``los_design`` as the one-sided mass
``P(mu < -floor)``. That is a valid estimand (it is ``p_exceed`` restricted to one side) but it is
NOT a sign confidence — its equipoise sits at the floor, not at zero, so it returns ~0 on a null and,
worse, ~0 on a genuine but sub-floor A-faster margin, which reads as "confident B is faster". The
ROPE directional form above is the fix: it meets the prereg's own NULL-1 = 0.5 calibration bar (which
the one-sided form fails), and the prereg explicitly invited a better route. The correction is
reported, not hidden.

The floor is a PROPERTY OF THE COMPARISON DESIGN and must be MEASURED for the design in use, never
borrowed (this project has four different measured floors for four designs). :func:`split_half_floor`
measures it for the paired per-seed design: partition the seeds into two disjoint halves, score the
SAME board on each, and take ``|mean(H1) - mean(H2)|`` — truth is exactly 0 by construction, so all
spread is the instrument's own noise at the sample size a verdict is read at.

Every LOS is reported beside its ``mean margin``, ``floor`` and ``margin/floor`` — a probability
without its floor is the defect this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------------------------
# Extrapolation wrong-sign hazard q(gap): PICK2-1's MEASURED co-observed sign-flip rate, stratified
# by the fitted gap. On the trigrams the K31 study actually observed (13.06% of position trigrams),
# this fraction of pairwise speed signs REVERSES relative to the full-corpus prediction. It is a
# direct probability-that-the-sign-is-wrong, which is exactly what LOS is a statement about.
# Bins are [lo, hi) in ms/char of |mean margin|; the last is open-ended.
#   gap < 0.42        -> 0.81      (22/27 pairs flipped)
#   0.42 <= gap < 0.97 -> 0.74     (20/27)
#   0.97 <= gap < 3.04 -> 0.30     (8/27)
#   gap >= 3.04        -> 0.12     (7/57)
# ------------------------------------------------------------------------------------------------
_FLIP_BINS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.42, 0.81),
    (0.42, 0.97, 0.74),
    (0.97, 3.04, 0.30),
    (3.04, float("inf"), 0.12),
)


def flip_hazard(gap: float) -> float:
    """Measured wrong-sign hazard ``q`` for a |mean margin| ``gap`` (ms/char); PICK2-1 strata."""
    g = abs(float(gap))
    for lo, hi, q in _FLIP_BINS:
        if lo <= g < hi:
            return q
    return _FLIP_BINS[-1][2]


def apply_flip_hazard(los: float, q: float) -> float:
    """Mix a sign-confidence ``los`` with its wrong-sign hazard ``q``.

    ``(1-q)*los + q*(1-los)``. Registered properties, both load-bearing and unit-tested:
    ``q=0`` leaves ``los`` unchanged; ``q=0.5`` returns 0.5 EXACTLY for any ``los`` — an
    at-chance flip hazard cannot be laundered into confidence.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"hazard q must be in [0,1], got {q}")
    return (1.0 - q) * los + q * (1.0 - los)


def split_half_floor(
    per_seed_ms: np.ndarray,
    n_partitions: int = 2000,
    rng: np.random.Generator | None = None,
    pct: float = 90.0,
) -> dict:
    """Measure the resolution floor for the paired per-seed design (same-board split-half placebo).

    ``per_seed_ms`` is ``(n_boards, n_seeds)`` — per-seed ms/char for a panel of boards. For each
    board and each random even split of the seeds into two disjoint halves, the placebo margin is
    ``|mean(half1) - mean(half2)|``; its true value is 0 by construction, so its distribution is the
    instrument's noise. Returns the pooled percentiles across all boards x partitions.

    The returned ``pXX`` (default p90) is the floor AT HALF SAMPLE SIZE (each half has n/2 seeds),
    which is CONSERVATIVE for a verdict read at the full n; :func:`scale_floor_to_n` rescales it.
    """
    rng = rng or np.random.default_rng(20260803)
    per_seed_ms = np.asarray(per_seed_ms, dtype=np.float64)
    if per_seed_ms.ndim != 2:
        raise ValueError("per_seed_ms must be (n_boards, n_seeds)")
    n_boards, n_seeds = per_seed_ms.shape
    half = n_seeds // 2
    if half < 2:
        raise ValueError(f"need >=4 seeds to split, got {n_seeds}")
    draws = np.empty(n_boards * n_partitions, dtype=np.float64)
    k = 0
    for b in range(n_boards):
        row = per_seed_ms[b]
        for _ in range(n_partitions):
            perm = rng.permutation(n_seeds)
            h1, h2 = perm[:half], perm[half : 2 * half]
            draws[k] = abs(row[h1].mean() - row[h2].mean())
            k += 1
    draws = draws[:k]
    return {
        "p50": float(np.percentile(draws, 50)),
        "p90": float(np.percentile(draws, 90)),
        "p99": float(np.percentile(draws, 99)),
        "max": float(draws.max()),
        "mean": float(draws.mean()),
        "n_draws": int(k),
        "half_n": int(half),
        "n_seeds": int(n_seeds),
        "floor": float(np.percentile(draws, pct)),
        "pct": float(pct),
    }


def scale_floor_to_n(half_floor: float, half_n: int, target_n: int) -> float:
    """Rescale a split-HALF placebo floor to the resolution of a verdict read at ``target_n`` seeds.

    A split-half placebo margin is ``mean(H1) - mean(H2)`` where each half has ``half_n`` seeds, so
    its sd is ``sqrt(2) * sd_seed / sqrt(half_n)``. A single mean over ``target_n`` seeds has sd
    ``sd_seed / sqrt(target_n)``, and the paired A-B margin's sd is ``sqrt(2)`` of that when the two
    boards share the seed set. So the floor at the verdict's own n is
    ``half_floor * sqrt(half_n / target_n)``. Reported alongside the conservative unscaled floor; the
    headline verdict uses the unscaled (larger) one.
    """
    return float(half_floor * np.sqrt(half_n / target_n))


@dataclass
class LOSResult:
    """One A-vs-B LOS verdict. ``margin`` and every probability carry the floor beside them."""

    a: str
    b: str
    n_seeds: int
    mean_margin: float          # mean_s(ms_A - ms_B); negative = A faster
    sd_margin: float
    sem_margin: float
    t_stat: float
    p_two_sided: float          # paired-t, H0: mu=0 (the pathology quantity, reported for contrast)
    signs_a_faster: int         # # seeds with ms_A < ms_B
    signs_b_faster: int
    floor: float                # the resolution floor used for the ROPE (ms/char)
    margin_over_floor: float
    los_seed: float             # P(A faster), seed noise only (equipoise 0.5)
    los_design: float           # ROPE directional: P(mu<-F) + 0.5 P(|mu|<=F) (equipoise 0.5)
    los_typist: float           # los_design degraded by the wrong-sign hazard
    p_exceed: float             # P(|mu|>F): is there a resolvable difference at all (not directional)
    p_a_beyond: float           # P(mu<-F): A meaningfully faster
    p_b_beyond: float           # P(mu>+F): B meaningfully faster
    p_tie: float                # P(|mu|<=F): within resolution
    flip_hazard_q: float
    faster: str                 # "A" | "B" — the board the sign points to
    verdict: str                # "A-DECIDED" | "B-DECIDED" | "UNDECIDED"
    decided_threshold: float = 0.95
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "pair": f"{self.a} vs {self.b}", "n": self.n_seeds,
            "mean_margin": self.mean_margin, "sd": self.sd_margin,
            "signs": f"{self.signs_a_faster}/{self.signs_b_faster}",
            "floor": self.floor, "margin_over_floor": self.margin_over_floor,
            "p_two_sided": self.p_two_sided,
            "LOS_seed": self.los_seed, "LOS_design": self.los_design,
            "LOS_typist": self.los_typist, "p_exceed": self.p_exceed,
            "p_tie": self.p_tie, "q": self.flip_hazard_q,
            "faster": self.faster, "verdict": self.verdict,
        }


def _posterior_below(margin_samples: np.ndarray, threshold: float) -> float:
    """Flat-prior posterior mass ``P(mu < threshold)`` for the mean of the paired differences.

    The marginal posterior of mu under a flat prior on (mu, log sigma) is a location-scale Student-t
    with df=n-1, centre = sample mean, scale = sem — numerically the complement of the frequentist
    one-sided t-test, but framed as a probability ABOUT mu, which is what a confidence-of-superiority
    statement is. Returned for a single threshold; the three ROPE regions are differences of two of
    these, so the tie mass is exact and the three regions sum to 1 by construction.
    """
    x = np.asarray(margin_samples, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("need >=2 seeds")
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(n))
    if sem == 0.0:
        # zero variance: posterior is a point mass at the mean. Split the boundary evenly.
        return 1.0 if mean < threshold else (0.5 if mean == threshold else 0.0)
    return float(stats.t.cdf((threshold - mean) / sem, n - 1))


def compute_los(
    ms_a: np.ndarray,
    ms_b: np.ndarray,
    floor: float,
    a_name: str = "A",
    b_name: str = "B",
    decided_threshold: float = 0.95,
) -> LOSResult:
    """The instrument. ``ms_a``/``ms_b`` are ``(n_seeds,)`` per-seed ms/char over a COMMON seed set.

    The pairing is by seed index (that is what removes the near-common seed shift, r>0.957). ``floor``
    is the design's MEASURED resolution floor (ms/char); pass the one measured for THIS design.
    """
    ms_a = np.asarray(ms_a, dtype=np.float64)
    ms_b = np.asarray(ms_b, dtype=np.float64)
    if ms_a.shape != ms_b.shape or ms_a.ndim != 1:
        raise ValueError("ms_a and ms_b must be 1-D arrays of the same length (paired by seed)")
    if not (np.all(np.isfinite(ms_a)) and np.all(np.isfinite(ms_b))):
        raise ValueError("non-finite ms/char — refusing (the empty-intersection nan trap)")
    if floor < 0:
        raise ValueError("floor must be >= 0")
    d = ms_a - ms_b                       # negative = A faster
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    sem = sd / np.sqrt(n) if sd > 0 else 0.0
    t_stat = mean / sem if sem > 0 else (np.inf if mean != 0 else 0.0)
    p_two = float(2 * stats.t.sf(abs(t_stat), n - 1)) if sem > 0 else (0.0 if mean != 0 else 1.0)
    signs_a = int((d < 0).sum())
    signs_b = int((d > 0).sum())
    faster = a_name if mean < 0 else b_name

    # LOS_seed: directional P(A faster) = P(mu < 0), seed noise only. Equipoise 0.5.
    los_seed = _posterior_below(d, 0.0)

    # The three ROPE regions from the SAME posterior (fishtest's interval hypothesis, Bayesian form):
    #   p_a_beyond = P(mu < -floor)   A meaningfully faster
    #   p_b_beyond = P(mu > +floor)   B meaningfully faster
    #   p_tie      = P(|mu| <= floor) within the instrument's resolution
    p_a_beyond = _posterior_below(d, -floor)
    p_b_beyond = 1.0 - _posterior_below(d, floor)
    p_tie = max(0.0, 1.0 - p_a_beyond - p_b_beyond)
    p_exceed = p_a_beyond + p_b_beyond
    # LOS_design: directional confidence with the TIE mass split evenly (equipoise on unresolvable).
    los_design = p_a_beyond + 0.5 * p_tie
    # LOS_typist: degrade by the measured wrong-sign hazard for this gap.
    q = flip_hazard(mean)
    los_typist = apply_flip_hazard(los_design, q)

    if los_design >= decided_threshold:
        verdict = "A-DECIDED"
    elif los_design <= 1.0 - decided_threshold:
        verdict = "B-DECIDED"
    else:
        verdict = "UNDECIDED"

    return LOSResult(
        a=a_name, b=b_name, n_seeds=n, mean_margin=mean, sd_margin=sd, sem_margin=sem,
        t_stat=float(t_stat), p_two_sided=p_two, signs_a_faster=signs_a, signs_b_faster=signs_b,
        floor=float(floor), margin_over_floor=(abs(mean) / floor if floor > 0 else float("inf")),
        los_seed=los_seed, los_design=los_design, los_typist=los_typist,
        p_exceed=p_exceed, p_a_beyond=p_a_beyond, p_b_beyond=p_b_beyond, p_tie=p_tie,
        flip_hazard_q=q, faster=faster, verdict=verdict, decided_threshold=decided_threshold,
    )
