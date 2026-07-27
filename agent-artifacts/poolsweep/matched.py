"""PHASE 2 — SPREAD-MATCHED arms, effective dof per cell, and the attenuation prediction.

Phase 1 answered the qualitative question but left two loose ends that a reviewer would (and
should) attack:

1. ``bandrandom`` collapsed to rho +0.1955 while retaining **18x more** A-spread than the
   archive mid-slice (sd_A 0.3095 vs 0.0168). Good for the *necessity* argument (it is
   conservative), but it means the two arms are not spread-MATCHED, so the comparison is not
   yet a clean 2x2. Fixed here with a bank ~50x larger, from which a genuinely
   archive-matched narrow window can be cut.
2. The *statistical correlate* hypothesis makes a QUANTITATIVE prediction, not just a
   directional one. Classical range restriction (Thorndike case 2) says a pool whose spread in
   the selection variable is a fraction ``u`` of the reference pool's attenuates the observed
   correlation by a known function of ``u``. If the measured ceilings sit ON that curve, the
   collapse IS the statistical correlate; if the optimized and random lineages sit on
   DIFFERENT curves at matched ``u``, near-optimality carries information beyond spread.
   Computing the prediction is what turns "tracks spread" into a testable claim.

Also measured here, per cell: the effective-dof statistic in BOTH conventions (the shipped
Pearson participation ratio that ``EvidenceWeights`` reports, and GEOMEAN-1's Spearman
|rho|/rho participation ratio) so this round is comparable with both prior rounds.

MODELLED ONLY. ``.native`` frame, asserted. Nothing adopted, no default changed.
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
from gaugecache import GaugeCache

A_NAME, B_NAME = "AALTO_BASE", "COMMUNITY_BASE"


def eff_dof_all(X: np.ndarray) -> dict:
    """Effective dof of a pool's gauge matrix, in every convention the campaign has used.

    Three, because a single definition's number is an artifact of that definition:
    ``shipped_pearson`` is what ``EvidenceWeights.effective_dof`` reports and what the
    ``NARROW_POOL_DOF = 4.5`` flag compares against; ``spearman_absrho`` /
    ``spearman_rho`` are GEOMEAN-1's participation ratios, so this round's numbers are
    directly comparable with that round's 4.10 (archive) / 4.82 (random).

    ``sfr`` is already excluded from ``LIVE_GAUGES``, but a variance filter is applied on the
    Spearman path too: numpy gives a permutation invariant std 1.9e-14 rather than 0, so a
    ``std > 0`` filter KEEPS it and rank-correlates pure noise (trap 23).
    """
    from scipy.stats import spearmanr

    out: dict = {"n_gauges": int(X.shape[1])}
    out["shipped_pearson_participation_ratio"] = E.effective_dof(X)
    # Relative-variance filter: absolute std thresholds are what trap 23 defeats.
    scale = np.maximum(np.abs(X).mean(axis=0), 1e-30)
    keep = (X.std(axis=0) / scale) > 1e-9
    out["n_gauges_kept"] = int(keep.sum())
    out["dropped_gauges"] = [g for g, k in zip(E.LIVE_GAUGES, keep, strict=True) if not k]
    if keep.sum() < 3 or X.shape[0] < 3:
        out["spearman_absrho_participation_ratio"] = float("nan")
        out["spearman_rho_participation_ratio"] = float("nan")
        return out
    rho = np.atleast_2d(spearmanr(X[:, keep]).statistic)
    rho = np.nan_to_num(rho, nan=0.0)
    np.fill_diagonal(rho, 1.0)
    for label, matrix in (("absrho", np.abs(rho)), ("rho", rho)):
        ev = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
        total = ev.sum()
        out[f"spearman_{label}_participation_ratio"] = (
            float(total**2 / (ev**2).sum()) if total > 0 else float("nan")
        )
    return out


def attenuated_rho(rho_ref: float, u: float) -> float:
    """Thorndike case-2 prediction: rho under restriction to spread fraction ``u``.

    ``u = sd_restricted / sd_reference`` in the SELECTION variable. This is the null the
    "it is only a statistical correlate" hypothesis actually implies, so quoting it lets the
    reader check the mechanism claim against a number instead of a direction.
    """
    if not np.isfinite(u) or u <= 0:
        return float("nan")
    denominator = np.sqrt(u**2 + rho_ref**2 * (1.0 - u**2))
    return float(rho_ref * u / denominator) if denominator > 0 else float("nan")


def spearman_ci(a: np.ndarray, b: np.ndarray, *, boot: int, seed: int) -> dict:
    rho = V._spearman(a, b)
    if boot <= 0:
        return {"rho": rho}
    rng = np.random.default_rng(seed)
    n = len(a)
    draws = []
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(a[idx])) < 3 or len(np.unique(b[idx])) < 3:
            continue
        r = V._spearman(a[idx], b[idx])
        if np.isfinite(r):
            draws.append(r)
    arr = np.array(draws)
    return {
        "rho": rho,
        "rho_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "rho_boot_sd": float(arr.std(ddof=1)),
        "sqrt_rho": float(np.sqrt(max(rho, 0.0))),
        "n_boot": len(arr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--bank", type=int, default=200_000, help="random bank size for band cutting")
    ap.add_argument("--n", type=int, default=400, help="pool size per cell")
    ap.add_argument("--cache-dir", default="/tmp/poolsweep-cache")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sA = E.load_target_surface(A_NAME, banks.SURFACE_DIR, "native")
    sB = E.load_target_surface(B_NAME, banks.SURFACE_DIR, "native")
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    ctx = E.gauge_context(args.corpus)
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    print(f"frame=native (asserted) corpus={ctx.corpus_name} n={args.n} bank={args.bank}", flush=True)

    def ms(pool):
        return banks.ms_of(pool, sA, objective), banks.ms_of(pool, sB, objective)

    archive = banks.load_archive()
    print(f"archive bank {len(archive)}; drawing random bank {args.bank} ...", flush=True)
    rand_bank = banks.random_bank(args.bank, args.seed)
    yA_r, yB_r = ms(rand_bank)
    yA_a, yB_a = ms(archive)
    sd_ref_A, sd_ref_B = float(yA_r.std(ddof=1)), float(yB_r.std(ddof=1))
    print(
        f"random bank: sd_A {sd_ref_A:.4f} sd_B {sd_ref_B:.4f} mean_A {yA_r.mean():.3f}\n"
        f"archive bank: sd_A {yA_a.std(ddof=1):.4f} sd_B {yB_a.std(ddof=1):.4f} mean_A {yA_a.mean():.3f}",
        flush=True,
    )

    cache = GaugeCache(args.corpus, args.cache_dir, workers=args.workers)
    cells: list[dict] = []

    # Reference cell: the wide random pool. Its rho is `rho_ref` for the attenuation curve.
    ref_pool = [
        rand_bank[i]
        for i in np.random.default_rng(args.seed).choice(len(rand_bank), args.n, replace=False)
    ]
    yA_ref, yB_ref = ms(ref_pool)
    rho_ref = V._spearman(yA_ref, yB_ref)
    print(f"\nreference random x{args.n}: rho_ref = {rho_ref:+.4f}", flush=True)

    def record(label, lineage, spread, pool, extra=None):
        yA, yB = ms(pool)
        X = cache.matrix(pool)
        uA = float(np.std(yA, ddof=1)) / sd_ref_A
        uB = float(np.std(yB, ddof=1)) / sd_ref_B
        row = {
            "label": label,
            "lineage": lineage,
            "spread": spread,
            "n": len(pool),
            **spearman_ci(yA, yB, boot=args.boot, seed=args.seed),
            "sd_A": float(np.std(yA, ddof=1)),
            "sd_B": float(np.std(yB, ddof=1)),
            "mean_A": float(np.mean(yA)),
            "mean_B": float(np.mean(yB)),
            "u_A": uA,
            "u_B": uB,
            # Predicted rho under pure range restriction, selecting on A, on B, and on the
            # geometric mean of the two (the symmetric case).
            "rho_pred_restrict_A": attenuated_rho(rho_ref, uA),
            "rho_pred_restrict_B": attenuated_rho(rho_ref, uB),
            "rho_pred_restrict_geom": attenuated_rho(rho_ref, float(np.sqrt(uA * uB))),
            "eff_dof": eff_dof_all(X),
            **(extra or {}),
        }
        row["residual_vs_pred_geom"] = row["rho"] - row["rho_pred_restrict_geom"]
        cells.append(row)
        print(
            f"  {label:<30} rho={row['rho']:+.4f} [{row['rho_ci95'][0]:+.3f},{row['rho_ci95'][1]:+.3f}] "
            f"u_A={uA:.4f} u_B={uB:.4f} pred_geom={row['rho_pred_restrict_geom']:+.4f} "
            f"resid={row['residual_vs_pred_geom']:+.4f} dof(ship)={row['eff_dof']['shipped_pearson_participation_ratio']:.2f} "
            f"dof(gm)={row['eff_dof']['spearman_absrho_participation_ratio']:.2f}",
            flush=True,
        )
        return row

    print("\n[A] the two reference corners", flush=True)
    record("random-wide", "random", "wide", ref_pool)
    arc_mid = banks.slice_by_value(archive, yA_a, n=args.n, where="mid", seed=args.seed)
    record("archive-narrow", "archive", "narrow", arc_mid, {"slice": "mid"})
    arc_full = [
        archive[i]
        for i in np.random.default_rng(args.seed).choice(len(archive), args.n, replace=False)
    ]
    record("archive-x400-shipped-draw", "archive", "narrow", arc_full, {"slice": "random-draw"})

    # ---- [B] SPREAD-MATCHED band-random: the decisive necessity test -------------------
    # For each target spread, cut a random-lineage window whose sd matches an archive-derived
    # spread. If rho at matched spread is the same for both lineages, spread is the mechanism.
    print("\n[B] band-random on A alone (random lineage, A-spread swept)", flush=True)
    arc_sd_A = float(np.std(yA_a, ddof=1))
    arc_mid_sd_A = float(np.std(banks.ms_of(arc_mid, sA, objective), ddof=1))
    arc_mid_sd_B = float(np.std(banks.ms_of(arc_mid, sB, objective), ddof=1))
    arc_sd_B = float(np.std(yB_a, ddof=1))
    targets = {
        "match-archive-mid": arc_mid_sd_A,
        "match-archive-full": arc_sd_A,
        "sd1": sd_ref_A * 0.01,
        "sd2": sd_ref_A * 0.02,
        "sd5": sd_ref_A * 0.05,
        "sd10": sd_ref_A * 0.10,
        "sd25": sd_ref_A * 0.25,
        "sd50": sd_ref_A * 0.50,
        "sd75": sd_ref_A * 0.75,
    }
    for name, target in sorted(targets.items(), key=lambda kv: kv[1]):
        pool, meta = banks.band_filter(
            rand_bank, yA_r, target_sd=target, n=args.n, seed=args.seed
        )
        record(f"bandrandom-A-{name}", "random", "narrowed-A", pool, {"band": meta})

    # ---- [B2] JOINT band-random: restricted in BOTH sources, to the ARCHIVE's own u -----
    # THE decisive cell. The single-variable arm above is not spread-matched to the archive in
    # the way that matters: restricting A alone leaves u_B ~= 0.57 while the archive has
    # u_B ~= 0.16, so "narrow" means something different in the two arms. This arm matches
    # BOTH, with random lineage and no search anywhere in its construction.
    print("\n[B2] JOINT band-random — matched in BOTH sources (the decisive cell)", flush=True)
    for label, (ta, tb) in {
        "match-archive-mid": (arc_mid_sd_A, arc_mid_sd_B),
        "match-archive-full": (arc_sd_A, arc_sd_B),
        "match-archive-full-2x": (arc_sd_A * 2, arc_sd_B * 2),
        "match-archive-full-4x": (arc_sd_A * 4, arc_sd_B * 4),
    }.items():
        pool, meta = banks.joint_band_filter(
            rand_bank, yA_r, yB_r, target_sd_a=ta, target_sd_b=tb, n=args.n, seed=args.seed
        )
        record(f"jointband-{label}", "random", "narrowed-AB", pool, {"band": meta})

    # ---- [C] k-swap ladder: optimized lineage, spread widened -------------------------
    print("\n[C] k-swap ladder (archive lineage, spread widened by k transpositions)", flush=True)
    for k in (1, 2, 3, 5, 8, 12, 20, 30):
        pool = banks.kswap_bank(archive, k, args.n, args.seed)
        record(f"kswap{k}", "archive", "widened", pool, {"k_swaps": k})

    # ---- [D] within-archive: optimality varies, lineage AND spread as fixed as possible -
    print("\n[D] within-archive slices (lineage fixed; optimality varies)", flush=True)
    for where in ("fast", "mid", "slow"):
        pool = banks.slice_by_value(archive, yA_a, n=args.n, where=where, seed=args.seed)
        record(f"archive-{where}", "archive", "narrow", pool, {"slice": where})

    # ---- [E] size ladder at both extremes, spread-matched band arm included ------------
    print("\n[E] size ladder", flush=True)
    for size in (100, 200, 400, 800, 1600):
        for f, lineage in ((0.0, "random"), (1.0, "archive")):
            pool, meta = banks.interpolate(rand_bank, archive, f=f, n=size, seed=args.seed)
            record(f"size{size}-f{f:g}", lineage, "wide" if f == 0 else "narrow", pool, meta)

    # ---- [F] The DISAGREEMENT-SCALE diagnostic — a level, not a correlation -------------
    # rho is a RATIO (shared spread over total spread), so it cannot tell whether a collapse
    # came from the numerator shrinking or the denominator shrinking. The absolute sd of the
    # per-layout DISAGREEMENT between the two sources can: it is the numerator's complement in
    # the same physical units (ms/trigram). Trap 45's lesson in the other direction — there a
    # DIFFERENCE could not test a SHARED component; here the difference is exactly the thing
    # being asked about, and the shared part is what rho already reports.
    print("\n[F] disagreement scale per cell (ms/trigram, absolute)", flush=True)
    disagreement = []
    for label, pool in (
        ("random-wide", ref_pool),
        ("archive-narrow", arc_mid),
        ("archive-shipped-draw", arc_full),
    ):
        yA, yB = ms(pool)
        # Regress B on A (least squares on z-scores) and take the residual's sd: the part of
        # each source's ranking the other cannot account for, in ms/trigram.
        za = (yA - yA.mean()) / (yA.std(ddof=1) or 1.0)
        slope = float(np.polyfit(za, yB, 1)[0])
        resid_sd = float(np.std(yB - np.polyval(np.polyfit(za, yB, 1), za), ddof=1))
        disagreement.append(
            {
                "label": label,
                "sd_A": float(yA.std(ddof=1)),
                "sd_B": float(yB.std(ddof=1)),
                "slope_B_on_zA": slope,
                "residual_sd_B_given_A": resid_sd,
                "shared_sd_B": abs(slope),
                "shared_over_residual": abs(slope) / resid_sd if resid_sd else float("nan"),
            }
        )
        print(
            f"  {label:<22} sd_B={yB.std(ddof=1):.4f} shared(|slope|)={abs(slope):.4f} "
            f"resid={resid_sd:.4f} ratio={abs(slope) / resid_sd if resid_sd else float('nan'):.4f}",
            flush=True,
        )

    control = cache.positive_control(ref_pool + arc_mid, n=16)
    print(f"\nPOSITIVE CONTROL vs shipped gauge_matrix: {control}", flush=True)

    payload = {
        "schema_version": 1,
        "note": (
            "PHASE 2. rho = Spearman(AALTO_BASE, COMMUNITY_BASE) ms/trigram, .native frame. "
            "u_A/u_B = this pool's sd divided by the WIDE RANDOM bank's sd (the restriction "
            "fraction). rho_pred_* is the Thorndike case-2 attenuation of rho_ref under that "
            "restriction — the quantitative null implied by 'the collapse is only a "
            "range-restriction correlate'. A cell ABOVE its prediction has more cross-source "
            "agreement than restriction alone explains; BELOW means less. MODELLED ONLY."
        ),
        "corpus": ctx.corpus_name,
        "corpus_sha256": dict(ctx.identity.get("sha256", {})),
        "frame": "native",
        "surface_sha256": {A_NAME: sA.sha256, B_NAME: sB.sha256},
        "seed": args.seed,
        "n_per_cell": args.n,
        "bootstrap": args.boot,
        "random_bank_size": len(rand_bank),
        "archive_bank_size": len(archive),
        "rho_ref_random_wide": rho_ref,
        "sd_reference": {"A": sd_ref_A, "B": sd_ref_B},
        "gauge_cache_positive_control": control,
        "disagreement_scale": disagreement,
        "cells": cells,
        "elapsed_seconds": time.time() - t0,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out} ({len(cells)} cells, {payload['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
