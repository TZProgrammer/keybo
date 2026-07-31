"""Shared primitives for the ASYMMETRIC-RESTRICTION necessity probe.

Deliberately self-contained on top of ``keybo.analysis.surfaces`` (which is BIT-IDENTICAL
between ``main`` and the ``poolsweep`` branch — verified with ``git diff main poolsweep --
src/keybo/analysis/surfaces.py`` returning empty). The unmerged ``evidence_scorer`` module
is NOT imported: the only two things this probe needs from it are a surface loader and a
ms/trigram reducer, both of which are four lines each and are re-implemented here so the
probe runs on ``main`` with nothing cherry-picked. A POSITIVE CONTROL against the published
poolsweep numbers is what certifies the re-implementation (see ``control.py``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import keybo.analysis.surfaces as S

FRONTIER_MAP = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/frontier_map.json"
SURFACE_DIR = (
    "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
    "old-new-layout-comparison/tri_frequency_old_new_surfaces"
)
A_NAME, B_NAME = "AALTO_BASE", "COMMUNITY_BASE"


@dataclass(frozen=True)
class Surface:
    name: str
    frame: str
    array: np.ndarray
    sha256: str


def load_surface(name: str, frame: str = "native", surface_dir: str = SURFACE_DIR) -> Surface:
    """``<name>.<frame>.npy`` with its digest, shape and finiteness asserted.

    ``.native`` is mandatory for any cross-source claim: on ``.standardized`` every source
    carries AALTO's bigram tensor (verified elsewhere at 5.68e-14), so the two "sources"
    would share a component and the comparison would be vacuous.
    """
    path = Path(surface_dir) / f"{name}.{frame}.npy"
    array = np.load(path)
    assert array.shape == (31, 31, 31), f"{name}: shape {array.shape}"
    assert np.all(np.isfinite(array)), f"{name}: non-finite"
    return Surface(name, frame, array, hashlib.sha256(path.read_bytes()).hexdigest())


def ms_of(layouts: list[str], surface: Surface, objective) -> np.ndarray:
    """Corpus-weighted fit per layout in ms per SCORED trigram (rank-identical to the sum)."""
    mass = float(objective[3].sum())
    return np.array([S.score_fit(lay, surface.array, objective) / mass for lay in layouts])


def load_archive(path: str = FRONTIER_MAP) -> list[str]:
    """The 12-axis Pareto archive via the shipped read path (``data["archive"]``, is_c30m, dedup)."""
    data = json.loads(Path(path).read_text())
    entries = list(data.get("archive") or [])
    raw = [e["layout"] if isinstance(e, dict) else e for e in entries]
    kept = [c for c in raw if S.is_c30m(c)]
    assert len(kept) == len(raw), f"is_c30m dropped {len(raw) - len(kept)}"
    return list(dict.fromkeys(kept))


def random_bank(n: int, seed: int) -> list[str]:
    """``n`` random C30M permutations — byte-identical construction to the shipped ``--pool random``."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.permutation(list(S.C30M))) for _ in range(n)]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho as Pearson on midrank-transformed values (SciPy-equivalent, ties handled)."""
    from scipy.stats import rankdata

    ra, rb = rankdata(a), rankdata(b)
    if len(np.unique(ra)) < 2 or len(np.unique(rb)) < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def community_per_seed(surface_dir: str = SURFACE_DIR) -> dict[int, np.ndarray]:
    """COMMUNITY's 3 per-seed surfaces, reassembled the same way the shipped loader does.

    This is the WITHIN-instrument reliability channel. AALTO ships NO per-seed parts, so
    there is no second independent PAIR — the two-source limit is structural, not a
    sampling gap. Asserted rather than assumed, because "the files are absent" and "I
    looked in the wrong directory" are different states.
    """
    sd = Path(surface_dir)
    out: dict[int, np.ndarray] = {}
    for seed in (0, 1, 2):
        bg, cd = sd / f"COMMUNITY_BASE.bigram.seed{seed}.npy", sd / f"COMMUNITY_BASE.conditional.seed{seed}.npy"
        if bg.is_file() and cd.is_file():
            out[seed] = np.load(bg)[:, :, None] + np.load(cd)
    assert out, f"no COMMUNITY per-seed parts under {sd}"
    return out


def seed_ms(pool: list[str], array: np.ndarray, objective) -> np.ndarray:
    mass = float(objective[3].sum())
    return np.array([S.score_fit(lay, array, objective) / mass for lay in pool])


def within_instrument(pool: list[str], per_seed: dict[int, np.ndarray], objective) -> dict:
    """Mean pairwise Spearman between COMMUNITY's own per-seed refits on ``pool``.

    NOT an independent instrument (same participants, same source) — but it bounds how much
    of a cross-source collapse is REFIT NOISE rather than genuine instrument disagreement.
    High within + low cross ⇒ the instruments really disagree.
    """
    ys = {s: seed_ms(pool, a, objective) for s, a in per_seed.items()}
    pairs = {
        f"seed{i}|seed{j}": spearman(ys[i], ys[j])
        for i in sorted(ys)
        for j in sorted(ys)
        if i < j
    }
    return {"pairs": pairs, "mean": float(np.mean(list(pairs.values())))}


def profile(pool: list[str], sA: Surface, sB: Surface, objective, ref: dict) -> dict:
    """Everything a cell has to report, in ONE place so no arm can quietly omit a field.

    ``u_A``/``u_B`` are the ACHIEVED restriction fractions — this pool's sd in each source
    divided by the WIDE RANDOM bank's sd in that source. They are computed from the pool's
    own scores, never from the request, because "requested 0.25" and "achieved 0.25" are
    different claims and only the second one is a measurement.
    """
    yA, yB = ms_of(pool, sA, objective), ms_of(pool, sB, objective)
    zA = (yA - ref["mean_A"]) / ref["sd_A"]
    zB = (yB - ref["mean_B"]) / ref["sd_B"]
    c, d = (zA + zB) / 2.0, (zA - zB) / 2.0
    uA, uB = float(zA.std(ddof=1)), float(zB.std(ddof=1))
    var_c, var_d = float(c.var(ddof=1)), float(d.var(ddof=1))
    k = float(np.sqrt(var_c / var_d))
    return {
        "n": len(pool),
        "rho_spearman": spearman(yA, yB),
        "rho_pearson": pearson(yA, yB),
        "u_A": uA,
        "u_B": uB,
        "u_ratio": uA / uB,
        "log_u_asymmetry": float(abs(np.log(uA / uB))),
        "c_spread": float(np.sqrt(var_c)),
        "d_spread": float(np.sqrt(var_d)),
        "k_c_over_d": k,
        # The closed form under the equal-variance (symmetric) assumption. Its SLACK vs the
        # measured Pearson r is the archive's real signature: the identity is exact only
        # when sd_A == sd_B, so slack IS asymmetry, quantified.
        "rho_algebraic_equalvar": float((k**2 - 1.0) / (k**2 + 1.0)),
        "cov_zA_zB": float(np.cov(zA, zB, ddof=1)[0, 1]),
        "var_C_minus_var_D": var_c - var_d,
        "mean_A": float(yA.mean()),
        "mean_B": float(yB.mean()),
        "sd_A": float(yA.std(ddof=1)),
        "sd_B": float(yB.std(ddof=1)),
    }


def bootstrap_rho(yA: np.ndarray, yB: np.ndarray, *, boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(yA)
    draws = []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        r = spearman(yA[i], yB[i])
        if np.isfinite(r):
            draws.append(r)
    arr = np.array(draws)
    return {
        "rho": spearman(yA, yB),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "boot_sd": float(arr.std(ddof=1)),
        "n_boot": len(arr),
    }


def two_sample_delta(yA1, yB1, yA2, yB2, *, boot: int = 8000, seed: int = 0) -> dict:
    """Bootstrap difference of two INDEPENDENT pools' rho — the unpaired form.

    Kept for continuity with the prior round's A5 test (which used exactly this), so my
    number is comparable with its +0.1106. The PAIRED form is stronger and lives in
    :func:`paired_delta`.
    """
    r = np.random.default_rng(seed)
    obs = spearman(yA1, yB1) - spearman(yA2, yB2)
    n1, n2 = len(yA1), len(yA2)
    d = []
    for _ in range(boot):
        i1, i2 = r.integers(0, n1, n1), r.integers(0, n2, n2)
        a, b = spearman(yA1[i1], yB1[i1]), spearman(yA2[i2], yB2[i2])
        if np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    arr = np.array(d)
    return {
        "delta_rho": float(obs),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_two_sided": float(min((arr > 0).mean(), (arr < 0).mean()) * 2),
        "n_boot": len(arr),
    }
