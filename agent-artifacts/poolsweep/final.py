"""PHASE 6 — the properly MATCHED cell, and an honest residual test.

Phase 5 identified the mechanism: rho(A,B) is a monotone function of ``C/D``, the ratio of
consensus spread to disagreement spread (Spearman +0.9963 over all 27 cells, +1.0000 within the
random lineage). Two defects in that phase's own execution have to be fixed before the verdict
can be trusted, and both are mine:

1. **The 2-D match failed.** ``match-archive-CD`` was supposed to hit the archive's ``C=0.085,
   D=0.080`` and instead landed at ``C=0.553`` — a sequential two-stage band with a
   ``sqrt(40)`` inflation factor that has no justification. Fixed here by selecting directly
   inside a 2-D BOX in the ``(C, D)`` plane, with the half-widths found by bisection on the
   achieved spreads. The retained fraction needed is ~2.5% of a 200k bank, so the cell is
   comfortably feasible; the earlier miss was a construction bug, not a structural limit.

2. **The residual test used the wrong functional form.** ``rho`` vs ``log(C/D)`` is monotone but
   SATURATING (it cannot leave [-1, +1]), so a straight-line fit misfits both tails — and that
   misfit, not a lineage effect, produced phase 5's "+0.3346 optimized residual, p = 0.026".
   Fitted properly (monotone interpolation of the random-lineage curve, evaluated only INSIDE
   its support) the question becomes answerable: at matched ``C/D``, does the optimized lineage
   sit off the random-lineage curve?

That second point is the campaign-relevant one, so it gets the careful treatment: a wrong
functional form manufacturing a significant p-value is exactly the shape of a false positive.

MODELLED ONLY. ``.native`` frame, asserted.
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


def box_match(
    layouts: list[str],
    c: np.ndarray,
    d: np.ndarray,
    *,
    target_c: float,
    target_d: float,
    n: int,
    seed: int,
) -> tuple[list[str], dict]:
    """``n`` layouts from a 2-D box in the (C, D) plane matching BOTH target spreads.

    Half-widths are found by bisection on the ACHIEVED spread of the selected set, jointly:
    each axis's half-width is scaled until that axis's realized sd matches its target. The
    box is centred on the bank's median in each coordinate, so the arm's LEVEL stays at the
    random lineage's own centre rather than drifting toward the archive's (which no random
    permutation can reach anyway — that structural gap is a stated limit, not something this
    construction can or should close).
    """
    rng = np.random.default_rng(seed)
    c0, d0 = float(np.median(c)), float(np.median(d))
    hc, hd = float(c.std(ddof=1)), float(d.std(ddof=1))
    chosen: np.ndarray | None = None
    for _ in range(60):
        mask = (np.abs(c - c0) <= hc) & (np.abs(d - d0) <= hd)
        idx = np.flatnonzero(mask)
        if len(idx) < n:
            hc, hd = hc * 1.35, hd * 1.35
            continue
        pick = idx[rng.choice(len(idx), n, replace=False)] if len(idx) > n else idx
        sd_c, sd_d = float(c[pick].std(ddof=1)), float(d[pick].std(ddof=1))
        chosen = pick
        ok_c = abs(sd_c - target_c) <= 0.05 * target_c
        ok_d = abs(sd_d - target_d) <= 0.05 * target_d
        if ok_c and ok_d:
            break
        # Multiplicative update per axis: shrink the axis that is too wide, grow the too narrow.
        hc *= float(np.clip(target_c / max(sd_c, 1e-12), 0.5, 2.0)) ** 0.7
        hd *= float(np.clip(target_d / max(sd_d, 1e-12), 0.5, 2.0)) ** 0.7
    assert chosen is not None, "box_match never found enough layouts"
    order = chosen[rng.permutation(len(chosen))]
    return [layouts[i] for i in order], {
        "target_c": target_c,
        "target_d": target_d,
        "achieved_c": float(c[order].std(ddof=1)),
        "achieved_d": float(d[order].std(ddof=1)),
        "half_width_c": hc,
        "half_width_d": hd,
        "candidates_in_box": int(len(np.flatnonzero((np.abs(c - c0) <= hc) & (np.abs(d - d0) <= hd)))),
        "bank_size": len(layouts),
    }


def monotone_curve_predict(
    x_fit: np.ndarray, y_fit: np.ndarray, x_query: np.ndarray
) -> np.ndarray:
    """Predict ``y`` at ``x_query`` from a monotone reference relation, or NaN outside support.

    The reference points are sorted, made monotone by a running maximum (the relation is
    monotone increasing by construction — a violation is sampling noise, not signal), then
    linearly interpolated. Queries outside ``[min, max]`` of the fitted support return NaN
    rather than an extrapolation: extrapolating a saturating relation is exactly what produced
    the spurious effect this function exists to avoid.
    """
    order = np.argsort(x_fit)
    xs, ys = x_fit[order], np.maximum.accumulate(y_fit[order])
    out = np.interp(x_query, xs, ys)
    return np.where((x_query < xs[0]) | (x_query > xs[-1]), np.nan, out)


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

    cells = []

    def profile(pool):
        yA, yB = ms(pool)
        zA = (yA - ref["mean_A"]) / ref["sd_A"]
        zB = (yB - ref["mean_B"]) / ref["sd_B"]
        c, d = (zA + zB) / 2.0, (zA - zB) / 2.0
        rho = V._spearman(yA, yB)
        rng = np.random.default_rng(args.seed)
        draws = []
        for _ in range(args.boot):
            i = rng.integers(0, len(yA), len(yA))
            r = V._spearman(yA[i], yB[i])
            if np.isfinite(r):
                draws.append(r)
        arr = np.array(draws)
        return {
            "rho": rho,
            "rho_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "c_spread": float(c.std(ddof=1)),
            "d_spread": float(d.std(ddof=1)),
            "c_over_d": float(c.std(ddof=1) / d.std(ddof=1)),
            "u_A": float(zA.std(ddof=1)),
            "u_B": float(zB.std(ddof=1)),
            "mean_A": float(yA.mean()),
            "mean_B": float(yB.mean()),
        }

    def record(label, lineage, pool, extra=None):
        row = {"label": label, "lineage": lineage, "n": len(pool), **profile(pool), **(extra or {})}
        cells.append(row)
        print(
            f"  {label:<32} rho={row['rho']:+.4f} [{row['rho_ci95'][0]:+.3f},{row['rho_ci95'][1]:+.3f}] "
            f"C={row['c_spread']:.4f} D={row['d_spread']:.4f} C/D={row['c_over_d']:8.3f}",
            flush=True,
        )
        return row

    rng = np.random.default_rng(args.seed)
    print("\n[REF] corners", flush=True)
    record("random-wide", "random", [rand_bank[i] for i in rng.choice(len(rand_bank), args.n, replace=False)])
    arc_row = record("archive-x400", "archive", [archive[i] for i in rng.choice(len(archive), args.n, replace=False)])
    yA_a, _ = ms(archive)

    # ---- THE MATCHED CELL, done properly this time --------------------------------------
    print("\n[MATCH] 2-D box in the (C,D) plane — random lineage, archive's C AND D", flush=True)
    matched_rows = []
    for name, (tc, td) in {
        "match-archive": (arc_row["c_spread"], arc_row["d_spread"]),
        "match-archive-2x": (arc_row["c_spread"] * 2, arc_row["d_spread"] * 2),
        "match-archive-halfC": (arc_row["c_spread"] * 0.5, arc_row["d_spread"]),
        "match-archive-2xC": (arc_row["c_spread"] * 2, arc_row["d_spread"]),
    }.items():
        pool, meta = box_match(rand_bank, C_r, D_r, target_c=tc, target_d=td, n=args.n, seed=args.seed)
        row = record(f"boxmatch-{name}", "random", pool, {"box": meta})
        matched_rows.append(row)
        print(
            f"      target C={tc:.4f} D={td:.4f} -> achieved C={meta['achieved_c']:.4f} "
            f"D={meta['achieved_d']:.4f} (box held {meta['candidates_in_box']} candidates)",
            flush=True,
        )

    # ---- A DENSE random-lineage reference CURVE, for the residual test -------------------
    # Restrict C and D jointly over a wide grid so the curve covers the archive's C/D and the
    # k-swap ladder's, i.e. so no query needs extrapolation.
    print("\n[CURVE] dense random-lineage reference curve over C/D", flush=True)
    curve = []
    for fc in (0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.32, 0.45, 0.6, 0.8, 1.0):
        for fd in (0.1, 0.3, 0.6, 1.0):
            pool, meta = box_match(
                rand_bank, C_r, D_r,
                target_c=fc * float(C_r.std(ddof=1)), target_d=fd * float(D_r.std(ddof=1)),
                n=args.n, seed=args.seed,
            )
            row = record(f"curve-C{fc:g}-D{fd:g}", "random", pool, {"target_fc": fc, "target_fd": fd})
            curve.append(row)

    # ---- The optimized-lineage cells to test against the curve ---------------------------
    print("\n[OPT] optimized-lineage cells", flush=True)
    opt = [arc_row]
    for k in (1, 2, 3, 5, 8, 12, 20, 30):
        opt.append(record(f"kswap{k}", "archive", banks.kswap_bank(archive, k, args.n, args.seed), {"k_swaps": k}))
    for where in ("fast", "mid", "slow"):
        opt.append(
            record(
                f"archive-{where}", "archive",
                banks.slice_by_value(archive, yA_a, n=args.n, where=where, seed=args.seed),
                {"slice": where},
            )
        )

    # ---- THE HONEST RESIDUAL TEST --------------------------------------------------------
    rand_cells = [c for c in cells if c["lineage"] == "random"]
    x_fit = np.log(np.array([c["c_over_d"] for c in rand_cells]))
    y_fit = np.array([c["rho"] for c in rand_cells])
    x_opt = np.log(np.array([c["c_over_d"] for c in opt]))
    pred = monotone_curve_predict(x_fit, y_fit, x_opt)
    resid = np.array([c["rho"] for c in opt]) - pred
    inside = np.isfinite(resid)
    from scipy.stats import spearmanr, wilcoxon

    test = (
        wilcoxon(resid[inside])
        if inside.sum() >= 6 and np.any(resid[inside] != 0)
        else None
    )
    # Leave-one-out check on the random lineage itself, so the curve's OWN scatter is the ruler.
    loo = []
    for i in range(len(rand_cells)):
        mask = np.ones(len(rand_cells), dtype=bool)
        mask[i] = False
        p = monotone_curve_predict(x_fit[mask], y_fit[mask], x_fit[i : i + 1])[0]
        if np.isfinite(p):
            loo.append(y_fit[i] - p)
    loo_arr = np.array(loo)

    verdict = {
        "note": (
            "The random-lineage cells define a monotone reference curve rho(log C/D). Each "
            "optimized-lineage cell is compared with that curve AT ITS OWN C/D, and cells "
            "outside the curve's support return NaN rather than an extrapolation. The ruler for "
            "'is a residual large?' is the curve's own leave-one-out scatter over the random "
            "lineage, not zero — a curve fitted from finite samples has intrinsic scatter."
        ),
        "n_random_curve_cells": len(rand_cells),
        "spearman_rho_vs_log_c_over_d_random": float(spearmanr(x_fit, y_fit).statistic),
        "spearman_rho_vs_log_c_over_d_all": float(
            spearmanr(
                np.log(np.array([c["c_over_d"] for c in cells])),
                np.array([c["rho"] for c in cells]),
            ).statistic
        ),
        "random_curve_loo_residual_sd": float(loo_arr.std(ddof=1)) if len(loo_arr) > 2 else None,
        "random_curve_loo_residual_abs_mean": float(np.abs(loo_arr).mean()) if len(loo_arr) else None,
        "optimized_cells": [
            {
                "label": c["label"], "c_over_d": c["c_over_d"], "rho": c["rho"],
                "curve_prediction": None if not np.isfinite(p) else float(p),
                "residual": None if not np.isfinite(r) else float(r),
                "in_curve_support": bool(np.isfinite(r)),
            }
            for c, p, r in zip(opt, pred, resid, strict=True)
        ],
        "optimized_residual_mean": float(resid[inside].mean()) if inside.any() else None,
        "optimized_residual_sd": float(resid[inside].std(ddof=1)) if inside.sum() > 2 else None,
        "n_optimized_in_support": int(inside.sum()),
        "wilcoxon_p": float(test.pvalue) if test is not None else None,
    }
    print("\n=== HONEST RESIDUAL TEST (monotone curve, no extrapolation) ===", flush=True)
    print(
        f"  random-lineage curve: {len(rand_cells)} cells, spearman(rho, log C/D) = "
        f"{verdict['spearman_rho_vs_log_c_over_d_random']:+.4f}; LOO residual sd "
        f"{verdict['random_curve_loo_residual_sd']}",
        flush=True,
    )
    for row in verdict["optimized_cells"]:
        print(
            f"    {row['label']:<18} C/D={row['c_over_d']:8.3f} rho={row['rho']:+.4f} "
            f"pred={row['curve_prediction'] if row['curve_prediction'] is None else f'{row['curve_prediction']:+.4f}'} "
            f"resid={row['residual'] if row['residual'] is None else f'{row['residual']:+.4f}'}",
            flush=True,
        )
    print(
        f"  optimized residual mean {verdict['optimized_residual_mean']} "
        f"(sd {verdict['optimized_residual_sd']}, n in support {verdict['n_optimized_in_support']}), "
        f"Wilcoxon p={verdict['wilcoxon_p']}",
        flush=True,
    )

    payload = {
        "schema_version": 1,
        "note": (
            "PHASE 6. The properly matched cell (2-D box in the C/D plane) plus an honest "
            "residual test against a monotone random-lineage reference curve. C = (zA+zB)/2, "
            "D = (zA-zB)/2, z-scored against the wide random bank. MODELLED ONLY."
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
        "archive_profile": {k: arc_row[k] for k in ("rho", "c_spread", "d_spread", "c_over_d")},
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
