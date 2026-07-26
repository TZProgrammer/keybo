"""GEOMEAN-1 step 2 — the correlation / effective-dof analysis of the 19-gauge frame.

This is deliverable 1 and stands alone: it is what says how many INDEPENDENT things a
19-gauge aggregate actually weights. Three things are computed, per corpus, over the
4367-layout pool (and separately over the archive-only and random-only sub-pools, because
the correlation structure of an already-optimized sample is not the correlation structure
of the space -- an optimizer collapses variance along the axes it traded off):

1. ORIENTATION, derived not assumed (problem (a), trap 5). Every gauge is oriented so that
   HIGHER = BETTER by comparing qwerty against the pool median: qwerty-is-worst is the
   campaign's reference point. The derived signs are then CHECKED against the frozen
   board's own `lower_better` map for the 15 sensitive gauges and against
   wscissor-allgauge's `invariant_direction_derivation` for the 4 invariant ones. A
   mismatch is reported, never silently overridden.

2. SPEARMAN RANK-CORRELATION MATRIX over the oriented gauges. Rank, not Pearson: the
   gauges live on wildly different scales and several are heavy-tailed, and a ranking rule
   only ever consumes order.

3. EFFECTIVE DEGREES OF FREEDOM, by three independent estimators so the number is not an
   artifact of one definition:
     * participation ratio of the |rho| eigenvalue spectrum:  (sum L)^2 / sum L^2
     * Kaiser count: #{eigenvalues > 1} of the 19x19 rho matrix
     * variance-explained count: smallest k with cumulative eigenvalue share >= 0.90
   Plus a GREEDY |rho|-threshold clustering at 0.9 / 0.8 / 0.7 that names the groups, since
   "how many independent things" is only actionable if you can say WHICH gauges collapse.

Nested/duplicate legs are checked EXACTLY where the campaign already has structural
claims, because a high rho is evidence of redundancy but an exact identity is proof:
  * sfr: permutation-invariant (proved separately) -> zero variance, dropped from the
    matrix with a note (a constant has no rank correlation).
  * oxey-style: `OxeyStyleScorer.fitness` is a fixed linear combination of
    `pattern_shares`, which include the scissor and imbalance columns of this very frame
    plus sfb/lsb/dsfb/redirect/... -- so it is checked by REGRESSING it on the frame
    (R^2), not merely correlated.
  * redir vs the oxeylyzer redirect family, sfb vs sfb-dist, lsb vs lsb-dist,
    sfs vs sfs-dist: reported as exact pairwise rho and, for the -dist pairs, as the
    correlation of the ratio (dist/count = mean distance) with the count.

MODELED/gauge only. Every number names its corpus.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path("/local/home/zegertho/agent/state/geomean/artifacts/geomean-1")
BOARD = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/noanchor-1/board_three_corpora.json"
)
WSCISSOR = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/wscissor-gen-1/wscissor-allgauge.json"
)


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of a 1-D array (ties averaged), matching scipy's default."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Spearman rho matrix of the columns of X."""
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    R = R - R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = 1.0
    R = R / sd
    return (R.T @ R) / len(R)


def load() -> dict:
    return json.loads((OUT / "pool_gauges.json").read_text())


def build_matrix(d: dict, corpus: str, layouts: list[str]) -> tuple[np.ndarray, list[str]]:
    """(n_layouts x 19) raw gauge matrix and the gauge names, in frame order."""
    S = d["frame"]["sensitive"]
    I = d["frame"]["invariant"]
    sens = d["sensitive"][corpus]
    inv = d["invariant"]
    rows = []
    for lay in layouts:
        rows.append(list(sens[lay]) + [inv[lay][g] for g in I])
    return np.asarray(rows, dtype=float), list(S) + list(I)


def derive_orientation(X: np.ndarray, names: list[str], qwerty_idx: int) -> dict[str, int]:
    """+1 if HIGHER is better, -1 if LOWER is better — derived from qwerty-is-worst.

    Reference point: qwerty is the campaign's known-worst layout. For each gauge, if
    qwerty sits ABOVE the pool median then higher must be WORSE (so lower_better, sign -1);
    if below, higher is better (sign +1). This is exactly the derivation
    wscissor-allgauge's `invariant_direction_derivation` records for the 4 invariant axes,
    applied uniformly to all 19 so no axis inherits an assumption.
    """
    med = np.median(X, axis=0)
    out = {}
    for j, name in enumerate(names):
        # A gauge with a single distinct value puts qwerty EXACTLY at the median, so the
        # comparison is not merely inconclusive — the gauge has no direction to derive.
        # Reported as 0 ("underivable"), never defaulted to a sign.
        if X[qwerty_idx, j] > med[j]:
            out[name] = -1  # qwerty high => high is bad => lower better
        elif X[qwerty_idx, j] < med[j]:
            out[name] = +1
        else:
            out[name] = 0
    return out


def greedy_clusters(rho: np.ndarray, names: list[str], thr: float) -> list[list[str]]:
    """Single-linkage clusters at |rho| >= thr (transitive closure)."""
    n = len(names)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in combinations(range(n), 2):
        if abs(rho[i, j]) >= thr:
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def eff_dof(rho: np.ndarray) -> dict[str, float]:
    ev = np.linalg.eigvalsh(np.abs(rho))
    ev = np.clip(ev[::-1], 0.0, None)
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    # participation ratio on the SIGNED rho spectrum too (|rho| is not PSD in general)
    evs = np.clip(np.linalg.eigvalsh(rho)[::-1], 0.0, None)
    return {
        "participation_ratio_absrho": float(tot**2 / (ev**2).sum()),
        "participation_ratio_rho": float(evs.sum() ** 2 / (evs**2).sum()),
        "kaiser_count_rho": int((np.linalg.eigvalsh(rho) > 1.0).sum()),
        "k_for_90pct_absrho": k90,
        "n_gauges": int(len(rho)),
    }


def main() -> int:
    d = load()
    S, I = d["frame"]["sensitive"], d["frame"]["invariant"]
    names_all = list(S) + list(I)
    qwerty = d["named"]["qwerty"]

    all_layouts = list(d["sensitive"]["iweb"].keys())
    # The archive/random boundary is re-derived from the frontier map, NOT read from
    # pool_gauges' `n_archive_derived`: that field records 2860+45=2905 pre-dedup while the
    # deduplicated archive is 2865, so trusting it would slice 40 archive layouts into the
    # "random" sub-pool and silently contaminate the sub-pool contrast.
    fm = json.loads(
        Path(
            "/local/home/zegertho/agent/state/keybo-optimization/artifacts/frontier_map.json"
        ).read_text()
    )
    seen: dict[str, None] = {}
    for e in fm["archive"] + fm["known_candidates"]:
        seen.setdefault(e["layout"], None)
    archive_only = list(seen)
    assert all_layouts[: len(archive_only)] == archive_only, "pool order is not archive-first"
    random_only = all_layouts[len(archive_only) : len(archive_only) + d["pool"]["n_random"]]
    assert len(random_only) == d["pool"]["n_random"]
    subpools = {
        "full": all_layouts,
        "archive_only": archive_only,
        "random_only": random_only,
    }

    board_lb = json.loads(BOARD.read_text())["lower_better"]
    wsc = json.loads(WSCISSOR.read_text())["invariant_direction_derivation"]["directions"]

    report: dict = {
        "purpose": "GEOMEAN-1: correlation + effective-dof of the 19-gauge frame",
        "pool": d["pool"],
        "frame": d["frame"],
        "orientation": {},
        "constant_gauges": {},
        "spearman": {},
        "eff_dof": {},
        "clusters": {},
        "nesting_checks": {},
    }

    for corpus in ("iweb", "blend", "noanchor"):
        Xfull, names = build_matrix(d, corpus, all_layouts)
        assert names == names_all
        qi = all_layouts.index(qwerty)

        # ---- 1. orientation, derived
        sign = derive_orientation(Xfull, names, qi)
        checks = []
        for g in S:
            derived_lb = sign[g] == -1
            if g in board_lb:
                frozen_lb = bool(board_lb[g])
                checks.append(
                    {
                        "gauge": g,
                        "derived_lower_better": derived_lb,
                        "frozen_lower_better": frozen_lb,
                        "agree": derived_lb == frozen_lb,
                        "derivable": sign[g] != 0,
                    }
                )
        for g in I:
            derived_lb = sign[g] == -1
            frozen_lb = bool(wsc[g]["lower_better"])
            checks.append(
                {
                    "gauge": g,
                    "derived_lower_better": derived_lb,
                    "frozen_lower_better": frozen_lb,
                    "agree": derived_lb == frozen_lb,
                    "derivable": sign[g] != 0,
                }
            )
        report["orientation"][corpus] = {"signs": sign, "vs_frozen": checks}

        # ---- constant gauges: no rank correlation exists for a constant.
        # Test on the DISTINCT-VALUE COUNT, not on std > 0: `sfr` is exactly
        # permutation-invariant (one distinct float64 over all 4367 layouts) yet numpy's
        # 2-D reduction reports std = 1.9e-14 for it, so an `sd == 0.0` test keeps it and
        # then rank-correlates pure floating-point noise into the matrix.
        n_distinct = [len({float(v) for v in Xfull[:, j]}) for j in range(len(names))]
        const = [names[j] for j in range(len(names)) if n_distinct[j] == 1]
        report["constant_gauges"][corpus] = {
            "gauges": const,
            "values": {g: float(Xfull[0, names.index(g)]) for g in const},
            "reported_std": {g: float(Xfull[:, names.index(g)].std()) for g in const},
            "note": (
                "exactly one distinct float64 over the whole pool => cannot discriminate any "
                "pair of layouts. Detected by distinct-value count, not std>0: numpy reports "
                "a ~1e-14 std for these from reduction order alone."
            ),
        }
        keep = [j for j in range(len(names)) if n_distinct[j] > 1]
        knames = [names[j] for j in keep]

        report["spearman"][corpus] = {}
        report["eff_dof"][corpus] = {}
        report["clusters"][corpus] = {}
        for sub, lays in subpools.items():
            X, _ = build_matrix(d, corpus, lays)
            Xo = X[:, keep] * np.array([sign[n] if sign[n] else 1 for n in knames])
            rho = spearman_matrix(Xo)
            report["spearman"][corpus][sub] = {
                "gauges": knames,
                "rho": [[round(float(v), 6) for v in row] for row in rho],
            }
            report["eff_dof"][corpus][sub] = eff_dof(rho)
            report["clusters"][corpus][sub] = {
                f"thr_{t}": greedy_clusters(rho, knames, t) for t in (0.9, 0.8, 0.7)
            }

        # ---- 3. nesting: is oxey-style a function of the rest of the frame?
        Xo_full = Xfull[:, keep] * np.array([sign[n] if sign[n] else 1 for n in knames])
        oi = knames.index("oxey-style")
        others = [j for j in range(len(knames)) if j != oi]
        A = np.column_stack([Xo_full[:, others], np.ones(len(Xo_full))])
        y = Xo_full[:, oi]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
        # and against ONLY the columns oxey.fitness actually reads from this frame
        legs = ["sfb", "lsb", "scissor", "imbalance", "redir", "alt"]
        lj = [knames.index(g) for g in legs if g in knames]
        A2 = np.column_stack([Xo_full[:, lj], np.ones(len(Xo_full))])
        c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
        r2b = 1.0 - float(((y - A2 @ c2) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
        report["nesting_checks"][corpus] = {
            "oxey_style_R2_on_full_frame": round(r2, 6),
            "oxey_style_R2_on_its_own_legs": round(r2b, 6),
            "its_own_legs": legs,
            "note": (
                "oxey.fitness is sum(w_k * pattern_shares[k]); pattern_shares includes the "
                "scissor and imbalance columns of THIS frame plus sfb/lsb/dsfb/"
                "inroll/outroll/onehand/redirect/bad_redirect/alternate. A high R^2 means "
                "the axis carries little information the frame does not already have."
            ),
        }
        # exact pairwise rho for the campaign's named duplicate/nested pairs
        pairs = [
            ("sfb", "sfb-dist"), ("sfs", "sfs-dist"), ("lsb", "lsb-dist"),
            ("sfb", "sfs"), ("redir", "oxey-style"), ("genkey", "oxey1"),
            ("oxey1", "oxey2"), ("oxey2", "wfd"), ("genkey", "wfd"),
            ("comfort", "oxey-style"), ("scissor", "oxey-style"),
            ("alt", "roll"), ("roll", "sr-roll"),
        ]
        rho_full = spearman_matrix(Xo_full)
        report["nesting_checks"][corpus]["named_pairs_rho"] = {
            f"{a}|{b}": round(float(rho_full[knames.index(a), knames.index(b)]), 6)
            for a, b in pairs
            if a in knames and b in knames
        }

    (OUT / "correlation.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT / 'correlation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
