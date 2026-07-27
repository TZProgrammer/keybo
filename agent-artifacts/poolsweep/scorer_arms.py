"""PHASE 3 — the SCORER arms: cross-source delta-rho per pool cell, via shipped machinery.

Phase 1/2 measure the CEILING (rho between the two independent sources). This phase measures
what EVSCORE-1's headline actually reported: the fitted scorer's cross-source advantage over
the community taste constants (``delta-rho`` vs the best rival), plus the effective dof and the
noise placebo — per pool cell, so the sweep's two halves can be read against each other.

Everything routes through the SHIPPED functions (``fit_evidence_weights``,
``cross_source_validation``, ``noise_placebo``, ``cross_source_agreement``) — this driver only
chooses pools and reports. That is deliberate: a hand-rolled reimplementation of a validated
constructor loses the validation (trap 28).

``--placebo-repeats`` defaults to **200**, not the shipped 20: p95 over 20 replicates is
essentially the maximum of 20 samples, and EVSCORE-1 withdrew a claim for exactly that reason
(trap 46). The shipped DEFAULT is deliberately left alone — changing it would be new work.
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
from matched import attenuated_rho, eff_dof_all, spearman_ci

A_NAME, B_NAME = "AALTO_BASE", "COMMUNITY_BASE"
#: The two independent sources only. POOL is not independent of either (it pools them), and
#: including the extra AALTO_*/COMMUNITY_* variants multiplies cells without adding a source.
SOURCES = (A_NAME, B_NAME)


def run_cell(
    label: str,
    lineage: str,
    spread: str,
    pool: list[str],
    *,
    surfaces: dict,
    ctx,
    objective,
    cache: GaugeCache,
    args,
    extra: dict | None = None,
) -> dict:
    """One pool cell: ceiling, effective dof, fitted-scorer delta-rho, and the placebo band."""
    t0 = time.time()
    X = cache.matrix(pool)
    targets = {n: E.surface_ms_per_trigram(pool, s, objective) for n, s in surfaces.items()}
    competitors = V.competitor_scores(pool)
    agreement = V.cross_source_agreement(targets)
    cells = V.cross_source_validation(
        pool,
        surfaces,
        ctx,
        objective,
        X=X,
        targets=targets,
        competitors=competitors,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    independent = [c for c in cells if c.independent]
    # The placebo is the shipped one, at 200 repeats (trap 46), on the first independent cell.
    placebo_test = independent[0].test_source if independent else B_NAME
    placebo = V.noise_placebo(
        pool,
        surfaces[B_NAME],
        targets[placebo_test],
        ctx,
        objective,
        X=X,
        fit_target=targets[B_NAME],
        repeats=args.placebo_repeats,
        seed=args.seed,
    )
    rows = []
    for cell in independent:
        best = max(
            (a for name, a in cell.agreement.items() if name != "evidence"),
            key=lambda a: a.spearman if np.isfinite(a.spearman) else -np.inf,
        )
        rows.append(
            {
                "fit_source": cell.fit_source,
                "test_source": cell.test_source,
                "evidence_rho": cell.agreement["evidence"].spearman,
                "best_rival": best.scorer,
                "best_rival_rho": best.spearman,
                "delta_rho": cell.agreement["evidence"].spearman - best.spearman,
                "delta_ci95": cell.advantages[best.scorer]["ci95"],
                "p_gt_0": cell.advantages[best.scorer]["p_gt_0"],
                "placebo_rho": cell.placebo_spearman,
                "rivals": {k: v.spearman for k, v in cell.agreement.items() if k != "evidence"},
            }
        )
    deltas = [r["delta_rho"] for r in rows if np.isfinite(r["delta_rho"])]
    weights = E.fit_evidence_weights(
        pool, surfaces[B_NAME], ctx, objective, pool_label=label, X=X, y=targets[B_NAME],
        seed=args.seed,
    )
    yA, yB = targets[A_NAME], targets[B_NAME]
    ceiling = V._spearman(yA, yB)
    row = {
        "label": label,
        "lineage": lineage,
        "spread": spread,
        "n": len(pool),
        **spearman_ci(yA, yB, boot=args.boot, seed=args.seed),
        "ceiling_mean_all_pairs": agreement["mean"],
        "sd_A": float(yA.std(ddof=1)),
        "sd_B": float(yB.std(ddof=1)),
        "mean_A": float(yA.mean()),
        "mean_B": float(yB.mean()),
        "eff_dof": eff_dof_all(X),
        "shipped_effective_dof": weights.effective_dof,
        "narrow_pool_flag": weights.effective_dof < E.NARROW_POOL_DOF,
        "transfer_warning": weights.transfer_warning(ceiling),
        "n_bad_signs": weights.sign_audit()["n_implausible"],
        "surrogate_r2_in_sample": weights.surrogate_r2_in_sample,
        "surrogate_r2_holdout": weights.surrogate_r2_holdout,
        "cells_won": sum(1 for d in deltas if d > 0),
        "n_cells": len(deltas),
        "mean_delta_rho": float(np.mean(deltas)) if deltas else float("nan"),
        "min_delta_rho": float(np.min(deltas)) if deltas else float("nan"),
        "cell_rows": rows,
        "placebo": {
            "repeats": placebo["repeats"],
            "abs_mean": placebo["spearman_abs_mean"],
            "abs_p95": placebo["spearman_abs_p95"],
            "mean": placebo["spearman_mean"],
        },
        "evidence_inside_placebo_band": bool(
            max((abs(r["evidence_rho"]) for r in rows), default=0.0)
            <= placebo["spearman_abs_p95"]
        ),
        "elapsed_seconds": time.time() - t0,
        **(extra or {}),
    }
    print(
        f"  {label:<28} ceil={row['rho']:+.4f} won={row['cells_won']}/{row['n_cells']} "
        f"dRho={row['mean_delta_rho']:+.4f} dof={row['shipped_effective_dof']:.2f} "
        f"placebo_p95={row['placebo']['abs_p95']:.4f} inband={row['evidence_inside_placebo_band']} "
        f"({row['elapsed_seconds']:.0f}s)",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bank", type=int, default=200_000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--placebo-repeats", type=int, default=200)
    ap.add_argument("--cache-dir", default="/tmp/poolsweep-cache")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    surfaces = {n: E.load_target_surface(n, banks.SURFACE_DIR, "native") for n in SOURCES}
    for name, surface in surfaces.items():
        assert surface.frame == "native", f"FRAME ASSERT FAILED for {name}: {surface.frame!r}"
    ctx = E.gauge_context(args.corpus)
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    print(
        f"frame=native (asserted) corpus={ctx.corpus_name} n={args.n} "
        f"placebo_repeats={args.placebo_repeats} (shipped default 20 is too few — trap 46)",
        flush=True,
    )

    archive = banks.load_archive()
    rand_bank = banks.random_bank(args.bank, args.seed)
    yA_r = banks.ms_of(rand_bank, surfaces[A_NAME], objective)
    yB_r = banks.ms_of(rand_bank, surfaces[B_NAME], objective)
    yA_a = banks.ms_of(archive, surfaces[A_NAME], objective)
    yB_a = banks.ms_of(archive, surfaces[B_NAME], objective)
    cache = GaugeCache(args.corpus, args.cache_dir, workers=args.workers)

    rng = np.random.default_rng(args.seed)
    ref_pool = [rand_bank[i] for i in rng.choice(len(rand_bank), args.n, replace=False)]
    arc_pool = [archive[i] for i in rng.choice(len(archive), args.n, replace=False)]
    arc_mid = banks.slice_by_value(archive, yA_a, n=args.n, where="mid", seed=args.seed)
    arc_sd_A, arc_sd_B = float(yA_a.std(ddof=1)), float(yB_a.std(ddof=1))

    plan: list[tuple] = [
        # The two EVSCORE-1 corners, reproduced through the same shipped path.
        ("random-wide", "random", "wide", ref_pool, {}),
        ("archive-shipped-draw", "archive", "narrow", arc_pool, {}),
    ]
    # The interpolation ladder the brief asks for, at fixed size (reported WITH the caveat that
    # a mixed pool is bimodal — see the report; the numbers are still the ones requested).
    for f in (0.125, 0.25, 0.5, 0.75, 0.875, 0.95):
        pool, meta = banks.interpolate(rand_bank, archive, f=f, n=args.n, seed=args.seed)
        plan.append((f"interp-f{f:g}", "mixed", "bimodal", pool, meta))
    # The decisive off-diagonal cells.
    joint, jmeta = banks.joint_band_filter(
        rand_bank, yA_r, yB_r, target_sd_a=arc_sd_A, target_sd_b=arc_sd_B, n=args.n, seed=args.seed
    )
    plan.append(("jointband-match-archive", "random", "narrowed-AB", joint, jmeta))
    band, bmeta = banks.band_filter(rand_bank, yA_r, target_sd=arc_sd_A, n=args.n, seed=args.seed)
    plan.append(("bandrandom-A-match-archive", "random", "narrowed-A", band, bmeta))
    for k in (1, 3, 8, 20):
        plan.append((f"kswap{k}", "archive", "widened", banks.kswap_bank(archive, k, args.n, args.seed), {"k_swaps": k}))
    plan.append(("archive-mid", "archive", "narrow", arc_mid, {"slice": "mid"}))
    for where in ("fast", "slow"):
        plan.append((
            f"archive-{where}", "archive", "narrow",
            banks.slice_by_value(archive, yA_a, n=args.n, where=where, seed=args.seed),
            {"slice": where},
        ))
    # Size ladder at both extremes: the brief's item 2, on the scorer statistic too.
    for size in (100, 200, 800):
        for f, lineage in ((0.0, "random"), (1.0, "archive")):
            pool, meta = banks.interpolate(rand_bank, archive, f=f, n=size, seed=args.seed)
            plan.append((f"size{size}-f{f:g}", lineage, "wide" if f == 0 else "narrow", pool, meta))

    wanted = None if args.arms == "all" else set(args.arms.split(","))
    print(f"\nrunning {len(plan) if wanted is None else len(wanted)} arms", flush=True)
    rows = []
    for label, lineage, spread, pool, extra in plan:
        if wanted is not None and label not in wanted:
            continue
        rows.append(
            run_cell(
                label, lineage, spread, pool,
                surfaces=surfaces, ctx=ctx, objective=objective, cache=cache, args=args,
                extra=extra,
            )
        )

    payload = {
        "schema_version": 1,
        "note": (
            "PHASE 3. Per pool cell: the transfer ceiling rho(AALTO_BASE,COMMUNITY_BASE), the "
            "fitted scorer's cross-source delta-rho vs the best community rival, effective dof "
            "in three conventions, and the noise placebo at 200 repeats (the shipped default of "
            "20 is ~its own maximum — trap 46; the default itself is UNCHANGED). .native frame. "
            "MODELLED ONLY: tau saturated at 1.0 and Phase-D was cancelled, so nothing here "
            "speaks to realized typing speed."
        ),
        "corpus": ctx.corpus_name,
        "corpus_sha256": dict(ctx.identity.get("sha256", {})),
        "frame": "native",
        "sources": list(SOURCES),
        "surface_sha256": {n: s.sha256 for n, s in surfaces.items()},
        "seed": args.seed,
        "n_per_cell": args.n,
        "placebo_repeats": args.placebo_repeats,
        "bootstrap": args.bootstrap,
        "gauge_cache_positive_control": cache.positive_control(ref_pool + arc_mid, n=16),
        "arms": rows,
        "elapsed_seconds": time.time() - t_start,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out} ({len(rows)} arms, {payload['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
