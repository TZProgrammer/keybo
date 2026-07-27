"""PHASE 1 — the CEILING sweep: rho(AALTO_BASE, COMMUNITY_BASE) over every pool cell.

The ceiling needs no gauge matrix and no fit (0.0002 s/layout vs 0.144 s/layout), so the
whole f x size x kind grid is affordable here and the expensive scorer arms in phase 2 can
be aimed at the cells that turn out to matter.

Design: pool KIND (lineage) and pool SPREAD are crossed, because in EVSCORE-1 they moved
together and so neither was identified. Read the output as:

  * ceiling varies down the LINEAGE rows at matched spread  -> near-optimality mechanism
  * ceiling varies across the SPREAD columns at fixed lineage -> range restriction / dof

Everything is measured on the ``.native`` frame (asserted below) on corpus blend-v1 unless
``--corpus`` says otherwise. MODELLED ONLY.
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


def attenuation_dof(*value_arrays: np.ndarray) -> dict:
    """Range-restriction diagnostics for a pool, per source and pooled.

    Reported because the *statistical correlate* hypothesis makes a QUANTITATIVE prediction,
    not just a directional one. Under classical range restriction on a single latent factor
    the observed correlation attenuates as a function of the retained spread, so a pool's sd
    ratio versus the reference (wide random) pool predicts its ceiling. Reporting the sd's
    lets a reader check that prediction instead of taking a verdict on trust.
    """
    out = {}
    for name, y in zip(("A", "B"), value_arrays, strict=True):
        out[f"sd_{name}"] = float(np.std(y, ddof=1))
        out[f"mean_{name}"] = float(np.mean(y))
        out[f"cv_{name}"] = float(np.std(y, ddof=1) / np.mean(y))
    return out


def spearman_ci(a: np.ndarray, b: np.ndarray, *, boot: int, seed: int) -> dict:
    """Bootstrap CI on rho, so a cell-to-cell difference can be read against noise.

    Trap 34's lesson generalized: a single rho with no interval is unreadable, and this whole
    sweep is a comparison of rho ACROSS cells. Resampling is over layouts (the unit that
    varies between cells), paired across the two sources so the pool draw is common-mode.
    """
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
        "n_boot": len(arr),
        "sqrt_rho": float(np.sqrt(max(rho, 0.0))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--surface-frame", default="native", choices=E.SURFACE_FRAMES)
    args = ap.parse_args()

    t_start = time.time()
    frame = args.surface_frame
    sA = E.load_target_surface(A_NAME, banks.SURFACE_DIR, frame)
    sB = E.load_target_surface(B_NAME, banks.SURFACE_DIR, frame)
    # ASSERT the frame (brief constraint): a cross-source rho on `.standardized` shares
    # AALTO's bigram tensor with the source under test and is meaningless for this task.
    assert sA.frame == frame == "native", f"FRAME ASSERT FAILED: {sA.frame!r} != 'native'"
    assert sB.frame == "native", f"FRAME ASSERT FAILED: {sB.frame!r}"
    ctx = E.gauge_context(args.corpus)
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    print(f"frame={frame} (asserted)  corpus={ctx.corpus_name}", flush=True)
    print(f"surface sha256: A={sA.sha256[:16]} B={sB.sha256[:16]}", flush=True)

    def ms(pool):
        return banks.ms_of(pool, sA, objective), banks.ms_of(pool, sB, objective)

    archive = banks.load_archive()
    print(f"archive bank: {len(archive)} unique C30M layouts", flush=True)
    # A big random bank, drawn once, so every random-derived cell is a subset of ONE draw
    # (pool-draw noise is then common-mode across cells rather than a per-cell nuisance).
    rand_bank = banks.random_bank(4000, args.seed)
    yA_rand, yB_rand = ms(rand_bank)
    yA_arc, yB_arc = ms(archive)
    print(
        f"random bank sd: A {yA_rand.std(ddof=1):.4f} B {yB_rand.std(ddof=1):.4f} | "
        f"archive bank sd: A {yA_arc.std(ddof=1):.4f} B {yB_arc.std(ddof=1):.4f}",
        flush=True,
    )

    cells: list[dict] = []

    def record(label: str, kind: str, pool: list[str], extra: dict) -> dict:
        yA, yB = ms(pool)
        row = {
            "label": label,
            "kind": kind,
            "n": len(pool),
            "corpus": ctx.corpus_name,
            "frame": frame,
            **spearman_ci(yA, yB, boot=args.boot, seed=args.seed),
            **attenuation_dof(yA, yB),
            **extra,
        }
        cells.append(row)
        print(
            f"  {label:<34} n={len(pool):<5} rho={row['rho']:+.4f} "
            f"sdA={row['sd_A']:.4f} sdB={row['sd_B']:.4f}",
            flush=True,
        )
        return row

    # ---- (1) INTERPOLATION at FIXED size 400 -------------------------------------------
    print("\n[1] interpolation f=archive share, size FIXED 400", flush=True)
    fractions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    for f in fractions:
        pool, meta = banks.interpolate(rand_bank, archive, f=f, n=400, seed=args.seed)
        record(f"interp-f{f:g}-x400", "interp", pool, meta)

    # Fine grid near f=1, because a threshold at the very tip and a smooth decline imply
    # very different things for the campaign's conclusion (the brief's question 4).
    print("\n[1b] fine grid near f=1 (threshold vs smooth)", flush=True)
    for f in [0.9, 0.95, 0.975, 0.99, 0.995]:
        pool, meta = banks.interpolate(rand_bank, archive, f=f, n=400, seed=args.seed)
        record(f"interp-f{f:g}-x400", "interp", pool, meta)

    # ---- (2) SIZE x KIND, independent of each other ------------------------------------
    print("\n[2] size x kind (f=0 pure random, f=1 pure archive, f=0.5 mid)", flush=True)
    for size in (100, 200, 400, 800, 1600):
        for f in (0.0, 0.5, 1.0):
            pool, meta = banks.interpolate(rand_bank, archive, f=f, n=size, seed=args.seed)
            record(f"size{size}-f{f:g}", "size-kind", pool, meta)

    # ---- (3) THE 2x2: LINEAGE x SPREAD -------------------------------------------------
    # The archive's own sd is the narrow target; the random pool's is the wide reference.
    print("\n[3] the 2x2 — lineage x spread, spreads matched by construction", flush=True)
    arc400 = banks.slice_by_value(archive, yA_arc, n=400, where="mid", seed=args.seed)
    yA_a400, _ = ms(arc400)
    target_narrow_sd = float(np.std(yA_a400, ddof=1))
    print(f"  narrow spread target (archive mid-400 sd_A) = {target_narrow_sd:.4f}", flush=True)

    # (a) random lineage, NARROW spread — the decisive new arm.
    band, band_meta = banks.band_filter(
        rand_bank, yA_rand, target_sd=target_narrow_sd, n=400, seed=args.seed
    )
    record("bandrandom-x400-narrow", "2x2", band, {"lineage": "random", "spread": "narrow", **band_meta})

    # (b) archive lineage, WIDE spread — k-swap perturbation widens variance, keeps lineage.
    for k in (1, 2, 3, 5, 8, 12, 20):
        pool = banks.kswap_bank(archive, k, 400, args.seed)
        yA_k, _ = ms(pool)
        record(
            f"kswap{k}-x400",
            "2x2",
            pool,
            {
                "lineage": "archive",
                "spread": "widened",
                "k_swaps": k,
                "sd_ratio_vs_archive": float(np.std(yA_k, ddof=1) / target_narrow_sd),
            },
        )

    # (c) the two reference corners, at exactly n=400 and the same bank draws.
    record("archive-x400-mid", "2x2", arc400, {"lineage": "archive", "spread": "narrow"})
    rand400 = [rand_bank[i] for i in np.random.default_rng(args.seed).choice(len(rand_bank), 400, replace=False)]
    record("random-x400", "2x2", rand400, {"lineage": "random", "spread": "wide"})

    # ---- (4) WITHIN-ARCHIVE: move OPTIMALITY with lineage AND spread held fixed --------
    # This is the only construction that isolates near-optimality itself: same lineage, same
    # spread, different distance from the frontier tip.
    print("\n[4] within-archive slices — optimality varies, lineage fixed", flush=True)
    for where in ("fast", "mid", "slow"):
        pool = banks.slice_by_value(archive, yA_arc, n=400, where=where, seed=args.seed)
        yA_s, _ = ms(pool)
        record(
            f"archive-{where}400",
            "within-archive",
            pool,
            {
                "lineage": "archive",
                "slice": where,
                "sd_A_over_archive_mid": float(np.std(yA_s, ddof=1) / target_narrow_sd),
            },
        )

    # ---- (5) WITHIN-RANDOM: move SPREAD across a wide range, lineage fixed random ------
    # The quantitative range-restriction curve. If the ceiling is a spread phenomenon, THIS
    # is the curve every other cell should sit on.
    print("\n[5] within-random spread ladder — lineage fixed random", flush=True)
    for mult in (0.5, 1, 2, 4, 8, 16, 32):
        target = target_narrow_sd * mult
        if target > float(np.std(yA_rand, ddof=1)) * 0.98:
            continue
        pool, meta = banks.band_filter(
            rand_bank, yA_rand, target_sd=target, n=400, seed=args.seed
        )
        record(
            f"bandrandom-x400-sd{mult:g}x",
            "spread-ladder",
            pool,
            {"lineage": "random", "sd_multiple_of_archive": mult, **meta},
        )

    payload = {
        "schema_version": 1,
        "note": (
            "PHASE 1 ceiling sweep. rho = Spearman(AALTO_BASE ms/trigram, COMMUNITY_BASE "
            "ms/trigram) on the .native frame — the transfer ceiling. sqrt(rho) is the bound "
            "for a scorer INDEPENDENT of both sources (trap 47); rho itself bounds only a "
            "scorer whose information IS one source. MODELLED ONLY."
        ),
        "corpus": ctx.corpus_name,
        "corpus_sha256": dict(ctx.identity.get("sha256", {})),
        "frame": frame,
        "surface_sha256": {A_NAME: sA.sha256, B_NAME: sB.sha256},
        "seed": args.seed,
        "bootstrap": args.boot,
        "archive_bank_size": len(archive),
        "random_bank_size": len(rand_bank),
        "narrow_spread_target_sd_A": target_narrow_sd,
        "reference_sd": {
            "random_bank_sd_A": float(yA_rand.std(ddof=1)),
            "random_bank_sd_B": float(yB_rand.std(ddof=1)),
            "archive_bank_sd_A": float(yA_arc.std(ddof=1)),
            "archive_bank_sd_B": float(yB_arc.std(ddof=1)),
        },
        "cells": cells,
        "elapsed_seconds": time.time() - t_start,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}  ({len(cells)} cells, {payload['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
