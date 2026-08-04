"""INTERPFRAME-1 §3 — the six interpretability metrics, computed identically on any frame.

Imported by the baseline and POC drivers so both arms are scored by ONE implementation: two
copies of a metric is how an arm ends up winning on a definition difference.

Every metric is computed on the CORPUS-FREQUENCY-WEIGHTED serve grid, as registered. An
unweighted 31x31 enumeration over-represents cells the corpus never types, and the attribution
being explained IS a frequency-weighted sum, so the weighted population is the one the numbers
are about.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

# --- the same-property groupings, FIXED IN THE PREREG so they cannot be chosen after the fact --

#: Served 20-column bigram frame (prereg §3). Each set is columns derived from ONE underlying
#: quantity, so a pair inside a set carrying OPPOSITE-signed attributions is failure mode 2.
SERVED_SAME_PROPERTY = [
    {"bottom", "home", "top"},  # exactly collinear: they sum to 1 on every letter key
    {"pinky", "ring", "middle", "index", "lateral"},  # the finger block (not even a one-hot)
    {"dx", "dy", "distance"},  # mutually functionally dependent travel
    {"inwards", "outwards"},  # the swap-invariant pair
]

#: The interp frame's groups: columns derived from the same underlying per-key quantity.
#: ``row_load``/``row_arrival``/``bottom_bias`` all read the two keys' ROWS; ``row_span`` reads
#: rows too but as a same-hand SPAN, so it is grouped with the other span columns. Declared with
#: the same generosity as the served list — a grouping that flattered the new frame by splitting
#: its own related columns apart would make M3 meaningless.
INTERP_SAME_PROPERTY = [
    {"row_load", "row_arrival", "bottom_bias"},
    {"row_span", "lateral_span", "same_hand_travel"},
    {"hand_conflict", "finger_load", "off_home_column"},
]

#: Trigram frame: every bg1_/bg2_ mirror pair, plus the served bigram groups inside each block.
def trigram_same_property(names):
    groups = []
    for g in SERVED_SAME_PROPERTY:
        groups.append({f"bg1_{n}" for n in g} | {f"bg2_{n}" for n in g})
    # the mirror pairs themselves: bg1_X and bg2_X are the same PREDICATE on different keys
    for n in names:
        if n.startswith("bg1_"):
            mirror = "bg2_" + n[4:]
            if mirror in names:
                groups.append({n, mirror})
    groups.append({"redirect", "bad_redirect"})
    groups.append({"sg_dx", "sg_dy", "sg_distance"})
    return groups


def same_property_groups(names):
    """The registered grouping for whichever frame ``names`` is."""
    names = set(names)
    if any(n.startswith("bg1_") for n in names):
        return [g & names for g in trigram_same_property(names)]
    if "hand_conflict" in names:
        return [g & names for g in INTERP_SAME_PROPERTY]
    return [g & names for g in SERVED_SAME_PROPERTY]


# --- M1 MAXCORR / M1b MEANCORR -------------------------------------------------------------


def weighted_corr_matrix(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Frequency-weighted Pearson correlation matrix of ``X``'s columns.

    A column with zero weighted variance (a CONSTANT over the weighted grid) has no correlation
    with anything — its row/col is set to 0 rather than nan, because a nan would silently
    propagate into ``nanmax`` and make M1 depend on which columns happened to be constant.
    Constants are M2's business, and M2 counts them explicitly.
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mean = w @ X
    Xc = X - mean
    cov = (Xc * w[:, None]).T @ Xc
    var = np.diag(cov).copy()
    live = var > 1e-14
    sd = np.where(live, np.sqrt(np.maximum(var, 0.0)), 1.0)
    corr = cov / np.outer(sd, sd)
    corr[~live, :] = 0.0
    corr[:, ~live] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def m1_maxcorr(X: np.ndarray, w: np.ndarray, names) -> dict:
    """M1: max off-diagonal |r| (and M1b: the mean), plus which pair is worst."""
    corr = weighted_corr_matrix(X, w)
    off = np.abs(corr).copy()
    np.fill_diagonal(off, 0.0)
    iu = np.triu_indices(len(names), k=1)
    vals = off[iu]
    worst = int(np.argmax(vals))
    return {
        "maxcorr": float(vals.max()),
        "meancorr": float(vals.mean()),
        "worst_pair": [names[iu[0][worst]], names[iu[1][worst]]],
        "n_pairs_over_0.9": int((vals > 0.9).sum()),
        "n_pairs_over_0.7": int((vals > 0.7).sum()),
        "n_pairs": int(len(vals)),
    }


# --- M2 CONSTFRAC ---------------------------------------------------------------------------


def m2_constfrac(X: np.ndarray, w: np.ndarray, names, attrib_abs: np.ndarray) -> dict:
    """M2: share of total |attribution| carried by columns that are CONSTANT on the weighted grid.

    "Constant" is judged on the WEIGHTED grid, i.e. over the cells the corpus actually types — a
    column that varies only on zero-mass cells is constant for every purpose the attribution
    serves. Any mass here is credit a main-effect number cannot mechanically have earned.
    """
    w = np.asarray(w, dtype=np.float64)
    live = w > 0
    const = []
    for j, name in enumerate(names):
        col = X[live, j]
        if col.size and float(np.ptp(col)) <= 1e-12:
            const.append(name)
    total = float(np.abs(attrib_abs).sum())
    mass = float(sum(abs(attrib_abs[names.index(n)]) for n in const))
    return {
        "constfrac": (mass / total) if total > 0 else 0.0,
        "constant_columns": const,
        "constant_mass_ms_per_char": mass,
        "total_abs_mass_ms_per_char": total,
    }


# --- M3 SPLITPAIRS --------------------------------------------------------------------------


def m3_splitpairs(names, attrib: np.ndarray, min_abs: float = 1e-4) -> dict:
    """M3: same-property column pairs whose attributions have OPPOSITE signs.

    ``min_abs`` (0.0001 ms/char) excludes numerically-dead columns: a +1e-17 vs -1e-17 pair is a
    float artifact, not a mechanism fighting itself, and counting it would let a frame's score be
    set by its most inert columns. Registered as part of the metric rather than tuned after.
    """
    by_name = dict(zip(names, attrib, strict=True))
    pairs = []
    for group in same_property_groups(names):
        live = sorted(n for n in group if abs(by_name.get(n, 0.0)) >= min_abs)
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                if (by_name[a] > 0) != (by_name[b] > 0):
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "ms_a": float(by_name[a]),
                            "ms_b": float(by_name[b]),
                            "conflict_ms": float(min(abs(by_name[a]), abs(by_name[b]))),
                        }
                    )
    pairs.sort(key=lambda p: -p["conflict_ms"])
    return {
        "splitpairs": len(pairs),
        "conflict_mass_ms_per_char": float(sum(p["conflict_ms"] for p in pairs)),
        "pairs": pairs[:20],
        "min_abs_threshold": min_abs,
    }


# --- M4 MONOFRAC ----------------------------------------------------------------------------


def m4_monofrac(names, attrib: np.ndarray, honored: dict[str, bool]) -> dict:
    """M4: share of |attribution| on columns that are monotone-constrained AND VERIFIED honored.

    ``honored`` is keyed by column name; a column absent from it (unconstrained) contributes 0,
    and a constrained-but-unhonored or constrained-but-DEAD column also contributes 0 — present is
    not effective, and ADJ-2 PINKY-MONO measured a constrained column on this repo learning
    exactly zero magnitude.
    """
    total = float(np.abs(attrib).sum())
    good = float(sum(abs(a) for n, a in zip(names, attrib, strict=True) if honored.get(n)))
    return {
        "monofrac": (good / total) if total > 0 else 0.0,
        "honored_columns": sorted(n for n in names if honored.get(n)),
        "unhonored_or_unconstrained": sorted(n for n in names if not honored.get(n)),
    }


# --- M5 SIGNSTAB / M6 SEEDSTAB --------------------------------------------------------------


def sign_agreement(a: np.ndarray, b: np.ndarray, min_abs: float = 1e-4) -> dict:
    """Share of columns whose attribution SIGN agrees between two runs, plus the rank rho.

    Columns inert in BOTH runs are excluded from the share (their sign is noise); a column live in
    one and inert in the other COUNTS AS A DISAGREEMENT, because "this feature explains the gap
    here and not there" is exactly the instability the metric is for.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    live_either = (np.abs(a) >= min_abs) | (np.abs(b) >= min_abs)
    live_both = (np.abs(a) >= min_abs) & (np.abs(b) >= min_abs)
    if not live_either.any():
        return {
            "sign_agree_frac": float("nan"),
            "rho": float("nan"),
            "n_live_either": 0,
            "n_live_both": 0,
        }
    # NUMERATOR: live in both AND same sign. DENOMINATOR: live in either. So a column that
    # explains the gap in one run and vanishes in the other counts as a DISAGREEMENT rather than
    # being quietly dropped — that appear/disappear behaviour is the instability being measured.
    agreed = int((np.sign(a[live_both]) == np.sign(b[live_both])).sum())
    rho = float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")
    return {
        "sign_agree_frac": agreed / int(live_either.sum()),
        "rho": rho,
        "n_live_either": int(live_either.sum()),
        "n_live_both": int(live_both.sum()),
    }
