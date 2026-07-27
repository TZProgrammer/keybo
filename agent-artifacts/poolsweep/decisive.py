"""PHASE 5 — the IDENTIFYING experiment, after my own probes broke phase 2's cleanest cell.

Phase 4 (adversarial, run against my own verdict) returned two findings that force a sharper
design, and both are corrections to ME:

* **P1/P3: my ``band_filter`` construction saturated.** A CONTIGUOUS width-400 window of a
  200k bank spans 0.2% of the range, so every "wide" target collapsed onto the same pool
  (``u_A`` capped at 0.51) and different targets returned IDENTICAL layout sets. The apparent
  flatness of that ladder was my construction's ceiling, not a property of the sources. Fixed
  by :func:`banks.quantile_band_pool` (central quantile band, then subsample), which decouples
  spread from sample size.
* **P4: optimized-lineage pools sit ABOVE the attenuation curve** (residual +0.152 vs the
  random lineage's −0.091, Mann-Whitney p < 0.0001). So "restriction magnitude alone explains
  everything" is FALSE as stated: at equal geometric-mean restriction ``u``, an archive-derived
  pool agrees MORE across sources than a random-lineage pool does.

Those two facts together say the operative variable is not "how much spread was removed" but
**WHICH DIRECTION** was removed. That is a testable, identifying claim, and this phase tests it.

The decomposition. With ``zA``, ``zB`` the two sources' z-scores over a reference pool, define

    consensus    C = (zA + zB) / 2     -- the shared factor, what both sources agree on
    disagreement D = (zA − zB) / 2     -- the part they contradict each other on

``rho(A,B)`` is high when the C-spread dominates the D-spread and low when it does not.
So restriction has two distinguishable modes, and the whole question is which one the archive
performs:

* restrict C, leave D  -> rho COLLAPSES (the shared signal is gone, only noise is left)
* restrict D, leave C  -> rho RISES (the pool is now one where the sources agree)

THE IDENTIFYING TEST: build random-lineage pools restricted along C only and along D only, at
matched TOTAL spread, then ask which one the archive resembles. If the archive matches the
C-restricted arm, the collapse is range restriction *of the shared factor* — a statistical
correlate of selecting on predicted time, with no role for near-optimality per se. If the
archive lies off both, near-optimality is doing something extra.

This also finally explains P4 without special pleading: the k-swap ladder's residual is positive
because perturbing an archive layout re-introduces C-spread (it moves layouts back toward the
bulk on the axis both sources agree about), which is exactly what the attenuation formula's
single ``u`` cannot represent.

MODELLED ONLY. ``.native`` frame, asserted. Corpus named in every output row.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import keybo.analysis.evidence_scorer as E
import keybo.analysis.evidence_validation as V
import keybo.analysis.surfaces as S

import banks

A_NAME, B_NAME = "AALTO_BASE", "COMMUNITY_BASE"


def decompose(yA: np.ndarray, yB: np.ndarray, ref: dict) -> dict:
    """C/D decomposition of a pool, in units of the REFERENCE pool's sd (so cells compare).

    ``ref`` carries the wide random pool's mean/sd per source; z-scoring against a FIXED
    reference rather than per-pool is what makes ``c_spread`` / ``d_spread`` comparable across
    cells — a per-pool z-score would normalize away the very quantity being measured.
    """
    zA = (yA - ref["mean_A"]) / ref["sd_A"]
    zB = (yB - ref["mean_B"]) / ref["sd_B"]
    c = (zA + zB) / 2.0
    d = (zA - zB) / 2.0
    return {
        "c_spread": float(c.std(ddof=1)),
        "d_spread": float(d.std(ddof=1)),
        "c_over_d": float(c.std(ddof=1) / d.std(ddof=1)) if d.std(ddof=1) else float("inf"),
        "u_A": float(zA.std(ddof=1)),
        "u_B": float(zB.std(ddof=1)),
        "c_level": float(c.mean()),
    }


def rho_ci(yA, yB, *, boot, seed):
    rho = V._spearman(yA, yB)
    rng = np.random.default_rng(seed)
    n = len(yA)
    draws = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        r = V._spearman(yA[idx], yB[idx])
        if np.isfinite(r):
            draws.append(r)
    arr = np.array(draws)
    return {
        "rho": rho,
        "rho_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "sqrt_rho": float(np.sqrt(max(rho, 0.0))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bank", type=int, default=200_000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sA = E.load_target_surface(A_NAME, banks.SURFACE_DIR, "native")
    sB = E.load_target_surface(B_NAME, banks.SURFACE_DIR, "native")
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    ctx = E.gauge_context(args.corpus)
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    print(f"frame=native (asserted) corpus={ctx.corpus_name} n={args.n}", flush=True)

    def ms(pool):
        return banks.ms_of(pool, sA, objective), banks.ms_of(pool, sB, objective)

    archive = banks.load_archive()
    rand_bank = banks.random_bank(args.bank, args.seed)
    yA_r, yB_r = ms(rand_bank)
    ref = {
        "mean_A": float(yA_r.mean()), "sd_A": float(yA_r.std(ddof=1)),
        "mean_B": float(yB_r.mean()), "sd_B": float(yB_r.std(ddof=1)),
    }
    zA_r = (yA_r - ref["mean_A"]) / ref["sd_A"]
    zB_r = (yB_r - ref["mean_B"]) / ref["sd_B"]
    C_r, D_r = (zA_r + zB_r) / 2.0, (zA_r - zB_r) / 2.0
    print(
        f"reference bank ({len(rand_bank)}): C spread {C_r.std(ddof=1):.4f}  "
        f"D spread {D_r.std(ddof=1):.4f}  C/D {C_r.std(ddof=1)/D_r.std(ddof=1):.4f}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    cells = []

    def record(label, lineage, mode, pool, extra=None):
        yA, yB = ms(pool)
        row = {
            "label": label, "lineage": lineage, "restriction_mode": mode, "n": len(pool),
            "corpus": ctx.corpus_name,
            **rho_ci(yA, yB, boot=args.boot, seed=args.seed),
            **decompose(yA, yB, ref),
            "mean_A": float(yA.mean()), "mean_B": float(yB.mean()),
            **(extra or {}),
        }
        cells.append(row)
        print(
            f"  {label:<34} rho={row['rho']:+.4f} [{row['rho_ci95'][0]:+.3f},{row['rho_ci95'][1]:+.3f}] "
            f"C={row['c_spread']:.4f} D={row['d_spread']:.4f} C/D={row['c_over_d']:7.3f} "
            f"u_A={row['u_A']:.4f} u_B={row['u_B']:.4f}",
            flush=True,
        )
        return row

    print("\n[REF] the two corners", flush=True)
    ref_pool = [rand_bank[i] for i in rng.choice(len(rand_bank), args.n, replace=False)]
    record("random-wide", "random", "none", ref_pool)
    arc_pool = [archive[i] for i in rng.choice(len(archive), args.n, replace=False)]
    arc_row = record("archive-x400", "archive", "search", arc_pool)

    # ---- THE IDENTIFYING CONTRAST: restrict C only vs D only, both random lineage -------
    print("\n[ID] restrict CONSENSUS only vs DISAGREEMENT only (random lineage both)", flush=True)
    for frac in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7):
        pool, meta = banks.axis_band_pool(rand_bank, C_r, target_sd_frac=frac, n=args.n, seed=args.seed)
        record(f"restrictC-{frac:g}", "random", "consensus-only", pool, {"target_frac": frac, "band": meta})
    for frac in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7):
        pool, meta = banks.axis_band_pool(rand_bank, D_r, target_sd_frac=frac, n=args.n, seed=args.seed)
        record(f"restrictD-{frac:g}", "random", "disagreement-only", pool, {"target_frac": frac, "band": meta})

    # ---- The archive's OWN C/D profile, matched by a random-lineage pool -----------------
    # Match the archive on BOTH C and D simultaneously: restrict C to the archive's C-spread,
    # then inside that band restrict D to the archive's D-spread. Random lineage throughout.
    print("\n[MATCH] random-lineage pool matched to the archive's C AND D spreads", flush=True)
    target_c, target_d = arc_row["c_spread"], arc_row["d_spread"]
    print(f"  archive profile: C={target_c:.4f} D={target_d:.4f} C/D={arc_row['c_over_d']:.4f}", flush=True)
    wide_c, meta_c = banks.quantile_band_pool(
        rand_bank, C_r, target_sd=target_c * np.sqrt(40), n=40 * args.n, seed=args.seed
    )
    sub_idx = {lay: i for i, lay in enumerate(rand_bank)}
    keep = np.array([sub_idx[lay] for lay in wide_c])
    sub_pool, sub_meta = banks.quantile_band_pool(
        [rand_bank[i] for i in keep], D_r[keep], target_sd=target_d, n=args.n, seed=args.seed
    )
    record(
        "match-archive-CD", "random", "consensus+disagreement", sub_pool,
        {"target_c": target_c, "target_d": target_d, "c_stage": meta_c, "d_stage": sub_meta},
    )
    # And the reverse order, as a construction-robustness check on the same target.
    wide_d, meta_d = banks.quantile_band_pool(
        rand_bank, D_r, target_sd=target_d * np.sqrt(40), n=40 * args.n, seed=args.seed
    )
    keep_d = np.array([sub_idx[lay] for lay in wide_d])
    sub_pool2, sub_meta2 = banks.quantile_band_pool(
        [rand_bank[i] for i in keep_d], C_r[keep_d], target_sd=target_c, n=args.n, seed=args.seed
    )
    record(
        "match-archive-CD-reversed", "random", "consensus+disagreement", sub_pool2,
        {"target_c": target_c, "target_d": target_d, "d_stage": meta_d, "c_stage": sub_meta2},
    )

    # ---- k-swap ladder in C/D coordinates: what does perturbation actually restore? ------
    print("\n[KSWAP] the k-swap ladder in C/D coordinates", flush=True)
    for k in (1, 2, 3, 5, 8, 12, 20, 30):
        record(f"kswap{k}", "archive", "search+perturb", banks.kswap_bank(archive, k, args.n, args.seed), {"k_swaps": k})

    # ---- Within-archive slices in C/D coordinates ---------------------------------------
    print("\n[ARCSLICE] within-archive slices in C/D coordinates", flush=True)
    yA_a, _ = ms(archive)
    for where in ("fast", "mid", "slow"):
        record(
            f"archive-{where}", "archive", "search+slice",
            banks.slice_by_value(archive, yA_a, n=args.n, where=where, seed=args.seed),
            {"slice": where},
        )

    # ---- THE VERDICT: does rho depend on C/D ratio alone, regardless of lineage? --------
    finite = [c for c in cells if np.isfinite(c["rho"]) and np.isfinite(c["c_over_d"])]
    lx = np.log(np.array([c["c_over_d"] for c in finite]))
    ry = np.array([c["rho"] for c in finite])
    rand_mask = np.array([c["lineage"] == "random" for c in finite])
    # Fit the relation on the RANDOM lineage only, then measure the OPTIMIZED lineage's
    # residual against it. This is the same logic as phase 4's P4, but with C/D as the
    # predictor instead of a single restriction magnitude u.
    coef = np.polyfit(lx[rand_mask], ry[rand_mask], 1)
    resid = ry - np.polyval(coef, lx)
    from scipy.stats import mannwhitneyu, spearmanr

    stat = mannwhitneyu(resid[~rand_mask], resid[rand_mask], alternative="two-sided")
    verdict = {
        "note": (
            "rho regressed on log(C/D) — the ratio of consensus spread to disagreement spread "
            "— fitted on the RANDOM lineage only, then the OPTIMIZED lineage's residual "
            "measured against that fit. Near-zero, non-significant optimized residuals mean the "
            "single variable C/D accounts for both lineages, i.e. the ceiling is set by WHICH "
            "DIRECTION of variation the pool retains and near-optimality adds nothing beyond "
            "its effect on that direction."
        ),
        "n_cells": len(finite),
        "spearman_rho_vs_log_c_over_d_all": float(spearmanr(lx, ry).statistic),
        "spearman_rho_vs_log_c_over_d_random_only": float(spearmanr(lx[rand_mask], ry[rand_mask]).statistic),
        "fit_on_random_slope": float(coef[0]),
        "fit_on_random_intercept": float(coef[1]),
        "random_residual_mean": float(resid[rand_mask].mean()),
        "random_residual_sd": float(resid[rand_mask].std(ddof=1)),
        "optimized_residual_mean": float(resid[~rand_mask].mean()),
        "optimized_residual_sd": float(resid[~rand_mask].std(ddof=1)),
        "mannwhitney_p": float(stat.pvalue),
        "per_cell_residual": [
            {"label": c["label"], "lineage": c["lineage"], "c_over_d": c["c_over_d"],
             "rho": c["rho"], "residual": float(r)}
            for c, r in zip(finite, resid, strict=True)
        ],
    }
    print(
        f"\nVERDICT: rho vs log(C/D): spearman all {verdict['spearman_rho_vs_log_c_over_d_all']:+.4f}, "
        f"random-only {verdict['spearman_rho_vs_log_c_over_d_random_only']:+.4f}\n"
        f"  residual mean: random {verdict['random_residual_mean']:+.4f} "
        f"(sd {verdict['random_residual_sd']:.4f}) | optimized {verdict['optimized_residual_mean']:+.4f} "
        f"(sd {verdict['optimized_residual_sd']:.4f}); Mann-Whitney p={verdict['mannwhitney_p']:.4f}",
        flush=True,
    )

    payload = {
        "schema_version": 1,
        "note": (
            "PHASE 5, the identifying experiment. C = (zA+zB)/2 (consensus / shared factor), "
            "D = (zA-zB)/2 (disagreement), z-scored against the WIDE RANDOM reference bank so "
            "spreads are comparable across cells. rho(A,B) is high iff C-spread dominates "
            "D-spread. Restricting C collapses rho; restricting D raises it. MODELLED ONLY: tau "
            "saturated at 1.0 and Phase-D was cancelled, so nothing here speaks to realized "
            "typing speed."
        ),
        "corpus": ctx.corpus_name,
        "corpus_sha256": dict(ctx.identity.get("sha256", {})),
        "frame": "native",
        "surface_sha256": {A_NAME: sA.sha256, B_NAME: sB.sha256},
        "seed": args.seed,
        "n_per_cell": args.n,
        "random_bank_size": len(rand_bank),
        "reference": ref,
        "reference_C_spread": float(C_r.std(ddof=1)),
        "reference_D_spread": float(D_r.std(ddof=1)),
        "cells": cells,
        "verdict": verdict,
        "elapsed_seconds": time.time() - t0,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out} ({len(cells)} cells, {payload['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
