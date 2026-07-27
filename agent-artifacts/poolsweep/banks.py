"""Layout BANKS for the interpolated-pool sweep — the constructions ARE the argument.

Four pool KINDS, chosen so that pool KIND and pool SPREAD can be varied INDEPENDENTLY.
That independence is the whole point: EVSCORE-1 had two cells (random x400, archive x400)
in which "near-optimal" and "narrow / collinear" moved together, so neither could be
identified. Here each LINEAGE gets its own spread knob:

    lineage \\ spread |  narrow                      |  wide
    -----------------+------------------------------+---------------------------
    random           |  band-filtered random        |  plain random
    archive          |  plain archive               |  k-swap perturbed archive

* ``random``   -- ``rng.permutation(C30M)``, byte-identical to the shipped ``_load_pool``.
* ``archive``  -- the exact global 12-axis Pareto archive from ``frontier_map.json``, same
  read path as the shipped ``_load_pool`` (``is_c30m`` filter, dedup, order-preserving).
* ``bandrandom`` -- random permutations REJECTION-FILTERED to a narrow window of predicted
  time. Never search-optimized, so it carries the archive's *low variance* without the
  archive's *optimality*. The window is centred on the random pool's own median, NOT on the
  archive's level, because no random permutation is as fast as a Pareto layout -- that
  structural gap is reported as an identification limit rather than papered over.
* ``kswap``    -- archive layouts with ``k`` random transpositions applied. Optimized
  LINEAGE with deliberately widened variance; ``k`` interpolates archive -> random.

Plus two WITHIN-ARCHIVE slices that vary spread with lineage held exactly fixed:

* ``arcband``  -- the archive's own narrowest time slice (spread down, lineage fixed).
* ``arcfast`` / ``arcslow`` -- the archive's fastest / slowest slice at MATCHED spread,
  which is the only construction here that moves *near-optimality* while holding both
  lineage and spread fixed.

Every builder is a pure function of ``(size, seed)`` so a cell is reproducible from its
label alone. Nothing here writes to the repo, changes a default, or promotes a layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import keybo.analysis.evidence_scorer as E
import keybo.analysis.surfaces as S

FRONTIER_MAP = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/frontier_map.json"
SURFACE_DIR = (
    "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
    "old-new-layout-comparison/tri_frequency_old_new_surfaces"
)


def load_archive(path: str = FRONTIER_MAP, *, include_known: bool = False) -> list[str]:
    """The Pareto archive as layout strings, via the SHIPPED read path.

    ``include_known`` mirrors GEOMEAN-1's wider definition (archive + known_candidates);
    the default is the shipped ``_load_pool`` behaviour (``data["archive"]`` only), which is
    what EVSCORE-1's arms actually drew from. Count in / count out is asserted because the
    ``is_c30m`` filter drops malformed entries SILENTLY.
    """
    data = json.loads(Path(path).read_text())
    entries = list(data.get("archive") or [])
    if include_known:
        entries += list(data.get("known_candidates") or [])
    raw = [e["layout"] if isinstance(e, dict) else e for e in entries]
    kept = [c for c in raw if S.is_c30m(c)]
    unique = list(dict.fromkeys(kept))
    assert len(kept) == len(raw), f"is_c30m dropped {len(raw) - len(kept)} archive entries"
    return unique


def random_bank(n: int, seed: int) -> list[str]:
    """``n`` random C30M permutations — the shipped ``--pool random`` construction."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.permutation(list(S.C30M))) for _ in range(n)]


def kswap(layout: str, k: int, rng: np.random.Generator) -> str:
    """``layout`` with ``k`` random transpositions of distinct positions applied.

    A transposition of a permutation is still a permutation, so the result is C30M by
    construction — but it is asserted anyway, because trap 28 is exactly the habitat of a
    hand-rolled index shuffle that silently emits a non-permutation.
    """
    chars = list(layout)
    for _ in range(k):
        i, j = rng.choice(len(chars), 2, replace=False)
        chars[i], chars[j] = chars[j], chars[i]
    out = "".join(chars)
    assert sorted(out) == sorted(layout), "kswap produced a non-permutation"
    return out


def kswap_bank(archive: list[str], k: int, n: int, seed: int) -> list[str]:
    """``n`` k-swap perturbations drawn round-robin over ``archive`` (deduplicated)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(archive))
    out: dict[str, None] = {}
    attempt = 0
    while len(out) < n and attempt < 40 * n:
        base = archive[order[attempt % len(order)]]
        out.setdefault(kswap(base, k, rng), None)
        attempt += 1
    assert len(out) >= n, f"kswap_bank k={k} exhausted at {len(out)}/{n}"
    return list(out)[:n]


def ms_of(layouts: list[str], surface, objective) -> np.ndarray:
    return E.surface_ms_per_trigram(layouts, surface, objective)


def _window_sds(sorted_vals: np.ndarray, n: int) -> np.ndarray:
    """sd (ddof=1) of every contiguous width-``n`` window of a sorted array, vectorized.

    Via prefix sums of x and x^2, so a 200k-layout bank costs one pass instead of 200k
    Python-level ``.std()`` calls. ``maximum(..., 0)`` guards the catastrophic-cancellation
    negative that the sum-of-squares form can produce when a window is nearly constant —
    which is exactly the regime the narrow arms live in.
    """
    c1 = np.concatenate([[0.0], np.cumsum(sorted_vals)])
    c2 = np.concatenate([[0.0], np.cumsum(sorted_vals**2)])
    total = c1[n:] - c1[:-n]
    total_sq = c2[n:] - c2[:-n]
    var = np.maximum(total_sq - total**2 / n, 0.0) / (n - 1)
    return np.sqrt(var)


def band_filter(
    layouts: list[str],
    values: np.ndarray,
    *,
    target_sd: float,
    n: int,
    seed: int,
) -> tuple[list[str], dict]:
    """The width-``n`` window of ``values`` whose sd best matches ``target_sd``.

    Selection is a contiguous slice in sorted ``values``, tie-broken toward the MEDIAN so the
    arm's LEVEL stays at its lineage's own centre rather than drifting to an extreme. Reported
    with the ACHIEVED sd so a spread-matching claim is checkable rather than asserted: a
    "matched" arm whose sd is 3x the target is not matched, and the number says so.

    ⚠️ A bank can be too small to hit a small target: with 3000 random layouts the tightest
    width-400 window still has sd_A ~= 0.31, so several nominally different targets returned
    the IDENTICAL pool. That is why the caller passes a large bank and why
    ``sd_ratio_achieved_over_target`` is in the output — a degenerate match is visible in the
    artifact instead of being silently read as a matched arm.
    """
    assert len(layouts) >= n, f"band_filter: bank of {len(layouts)} too small for n={n}"
    order = np.argsort(values)
    sorted_vals = values[order]
    sds = _window_sds(sorted_vals, n)
    centre = (len(order) - n) / 2.0
    starts = np.arange(len(sds))
    penalty = np.abs(sds - target_sd) + np.where(sds > target_sd, 0.5 * (sds - target_sd), 0.0)
    penalty = penalty + 1e-9 * np.abs(starts - centre) * max(target_sd, 1e-12)
    start = int(np.argmin(penalty))
    idx = order[start : start + n]
    rng = np.random.default_rng(seed)
    idx = idx[rng.permutation(len(idx))]  # break the sort order so downstream folds are fair
    chosen = [layouts[i] for i in idx]
    achieved = float(values[idx].std(ddof=1))
    return chosen, {
        "target_sd": target_sd,
        "achieved_sd": achieved,
        "sd_ratio_achieved_over_target": achieved / target_sd if target_sd else float("nan"),
        "window_start_quantile": start / max(1, len(order) - n),
        "bank_size": len(layouts),
        "selected": n,
        "tightest_possible_sd": float(sds.min()),
        "degenerate_floor": bool(achieved > 1.5 * target_sd),
    }


def quantile_band_pool(
    layouts: list[str],
    values: np.ndarray,
    *,
    target_sd: float,
    n: int,
    seed: int,
    tol: float = 0.02,
) -> tuple[list[str], dict]:
    """``n`` layouts drawn from the central quantile band whose sd matches ``target_sd``.

    ⚠️ This replaces :func:`band_filter` for any arm that needs a WIDE spread, because a
    CONTIGUOUS width-``n`` window of a sorted 200k-layout bank can never be wide: 400 of
    200,000 spans 0.2% of the range, so every "wide" target saturated at the same pool
    (``u_A`` capped at 0.51) and several nominally different targets returned the IDENTICAL
    layout set. My own adversarial probe P1 is what exposed it — the ``band_filter`` ladder's
    apparent flatness was my construction's ceiling, not a property of the sources.

    The fix decouples spread from sample size: take the central quantile band of half-width
    ``h``, then SUBSAMPLE ``n`` from it. sd is monotone in ``h`` and spans ~0 to the bank's
    full sd, so a bisection on ``h`` hits any reachable target. ``achieved_sd`` is reported so
    a miss is visible rather than silently read as a match.
    """
    assert len(layouts) >= n, f"bank of {len(layouts)} too small for n={n}"
    order = np.argsort(values)
    centre = len(order) / 2.0
    rng = np.random.default_rng(seed)

    def draw(h: float) -> np.ndarray:
        half = max(n / 2.0, h * len(order) / 2.0)
        lo, hi = int(max(0, centre - half)), int(min(len(order), centre + half))
        band = order[lo:hi]
        if len(band) <= n:
            return band
        return band[rng.choice(len(band), n, replace=False)]

    lo_h, hi_h = 0.0, 1.0
    best = draw(1.0)
    if float(values[best].std(ddof=1)) > target_sd:
        for _ in range(48):
            mid = 0.5 * (lo_h + hi_h)
            cand = draw(mid)
            sd = float(values[cand].std(ddof=1))
            best = cand
            if abs(sd - target_sd) <= tol * max(target_sd, 1e-12):
                break
            if sd > target_sd:
                hi_h = mid
            else:
                lo_h = mid
    idx = best[rng.permutation(len(best))]
    chosen = [layouts[i] for i in idx]
    achieved = float(values[idx].std(ddof=1))
    return chosen, {
        "target_sd": target_sd,
        "achieved_sd": achieved,
        "sd_ratio_achieved_over_target": achieved / target_sd if target_sd else float("nan"),
        "selected": len(chosen),
        "bank_size": len(layouts),
        "construction": "central quantile band, then subsample n (spread decoupled from size)",
    }


def axis_band_pool(
    layouts: list[str],
    axis: np.ndarray,
    *,
    target_sd_frac: float,
    n: int,
    seed: int,
) -> tuple[list[str], dict]:
    """``n`` layouts from the central band of an arbitrary ``axis``, to a spread fraction.

    Used for the CONSENSUS / DISAGREEMENT contrast, which is the identifying experiment: the
    axis is a linear combination of the two sources' z-scores rather than one source's values.
    """
    pool, meta = quantile_band_pool(
        layouts, axis, target_sd=target_sd_frac * float(axis.std(ddof=1)), n=n, seed=seed
    )
    return pool, {**meta, "target_sd_frac_of_axis": target_sd_frac}


def joint_band_filter(
    layouts: list[str],
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    target_sd_a: float,
    target_sd_b: float,
    n: int,
    seed: int,
    oversample: int = 30,
) -> tuple[list[str], dict]:
    """A random-lineage pool restricted in BOTH sources at once, to matched spreads.

    This arm exists because the single-variable ``band_filter`` turned out to be the wrong
    control, and the smoke run is what showed it. Restricting random permutations on A alone
    gives ``u_A = 0.047`` but leaves ``u_B = 0.566``: A and B disagree strongly on random
    layouts, so squeezing A barely squeezes B. The archive, by contrast, restricts BOTH
    (``u_A = 0.042``, ``u_B = 0.161``). Comparing those two cells confounds "how narrow" with
    "narrow in what", so neither the spread nor the near-optimality reading is identified by
    them.

    Construction: take a generous window in A (``oversample * n`` layouts around the level
    that matches the archive's own A-window), then inside it choose the width-``n`` B-window
    whose B-sd best matches the target. The result is random-lineage — no search, no
    optimization, no archive ancestry — yet restricted in both sources to the archive's own
    ``u_A`` and ``u_B``. If its rho still collapses, the collapse is range restriction on the
    shared factor and near-optimality is not doing the work.
    """
    assert len(layouts) >= oversample * n, f"need >= {oversample * n} layouts, have {len(layouts)}"
    order_a = np.argsort(values_a)
    width = min(len(order_a), oversample * n)
    sds_a = _window_sds(values_a[order_a], width)
    # Pick the A-window whose sd is closest to (oversample-scaled) target: a wider window in A
    # is fine, because the second stage narrows B inside it; what matters is that the FINAL
    # pool's u_A lands near the target, which the caller checks from the reported sd.
    start_a = int(np.argmin(np.abs(sds_a - target_sd_a * np.sqrt(oversample))))
    idx_a = order_a[start_a : start_a + width]
    sub_b = values_b[idx_a]
    order_b = np.argsort(sub_b)
    sds_b = _window_sds(sub_b[order_b], n)
    start_b = int(np.argmin(np.abs(sds_b - target_sd_b)))
    idx = idx_a[order_b[start_b : start_b + n]]
    rng = np.random.default_rng(seed)
    idx = idx[rng.permutation(len(idx))]
    chosen = [layouts[i] for i in idx]
    return chosen, {
        "target_sd_a": target_sd_a,
        "target_sd_b": target_sd_b,
        "achieved_sd_a": float(values_a[idx].std(ddof=1)),
        "achieved_sd_b": float(values_b[idx].std(ddof=1)),
        "a_window_width": int(width),
        "bank_size": len(layouts),
        "oversample": oversample,
    }


def slice_by_value(
    layouts: list[str], values: np.ndarray, *, n: int, where: str, seed: int
) -> list[str]:
    """The ``n`` fastest / slowest / median-centred layouts by ``values``."""
    order = np.argsort(values)
    if where == "fast":
        idx = order[:n]
    elif where == "slow":
        idx = order[-n:]
    elif where == "mid":
        start = (len(order) - n) // 2
        idx = order[start : start + n]
    else:
        raise ValueError(f"unknown slice {where!r}")
    rng = np.random.default_rng(seed)
    idx = idx[rng.permutation(len(idx))]
    return [layouts[i] for i in idx]


def interpolate(
    random_pool: list[str], archive_pool: list[str], *, f: float, n: int, seed: int
) -> tuple[list[str], dict]:
    """A size-``n`` pool holding ``round(f*n)`` archive layouts and the rest random.

    Both sub-draws are shuffled together so no downstream fold or bootstrap can pick up the
    block structure as signal.
    """
    n_archive = int(round(f * n))
    n_random = n - n_archive
    assert n_archive <= len(archive_pool), f"need {n_archive} archive, have {len(archive_pool)}"
    assert n_random <= len(random_pool), f"need {n_random} random, have {len(random_pool)}"
    rng = np.random.default_rng(seed)
    a = [archive_pool[i] for i in rng.choice(len(archive_pool), n_archive, replace=False)]
    r = [random_pool[i] for i in rng.choice(len(random_pool), n_random, replace=False)]
    pool = a + r
    pool = [pool[i] for i in rng.permutation(len(pool))]
    assert len(set(pool)) == len(pool), "interpolated pool has duplicates"
    return pool, {"f": f, "n": n, "n_archive": n_archive, "n_random": n_random}
