"""PHASE 4 — adversarial probes against MY OWN verdict, before I write it down.

The phase-2 conclusion ("range restriction of the shared factor, not near-optimality") rests on
one comparison: a random-lineage pool restricted to the archive's own ``u_A``/``u_B`` reaches the
archive's ceiling. Four ways that could be wrong, each with its own probe:

P1  SELECTION-INDUCED COLLAPSE. ``joint_band_filter`` selects on the very variables whose
    correlation is then measured. Selecting a narrow B-window *inside* a narrow A-window could
    mechanically destroy rho regardless of restriction magnitude — in which case my "matched"
    cell is not evidence about the archive at all, it is an artifact of my own construction.
    Probe: build the SAME two-stage selection but at WIDE targets. If the construction itself
    kills rho, it will kill it there too; if rho survives at wide targets, the collapse is
    coming from the restriction magnitude and not from the selection procedure.

P2  A GENUINE RESIDUAL. Overlapping CIs are not equality (absence is not disproof). Probe:
    estimate the archive-vs-jointband difference DIRECTLY with a paired bootstrap over the
    common layout count, and put a CI on the DIFFERENCE rather than eyeballing two intervals —
    the same discipline EVSCORE-1's paired-advantage function applies to delta-rho.

P3  THE ARCHIVE'S OWN u IS AMBIGUOUS. The archive restricts A 24x and B 6.6x; which one do I
    match? A verdict that flips depending on that choice is not a verdict. Probe: sweep the
    joint-band arm over a 2-D grid of (u_A, u_B) targets and report the whole surface, so the
    reader sees the archive's cell sitting on it rather than one hand-picked match.

P4  MONOTONICITY / SHAPE. If rho is a smooth function of u alone, then "the mechanism is
    restriction" makes a strong prediction: every cell — regardless of lineage — should sit on
    ONE curve rho(u). Probe: fit that curve on the random-lineage cells only, then measure the
    optimized-lineage cells' residuals against it. Systematic positive residuals for the
    optimized lineage would mean near-optimality carries something extra after all.

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
from matched import attenuated_rho

A_NAME, B_NAME = "AALTO_BASE", "COMMUNITY_BASE"


def paired_rho_difference(
    yA1, yB1, yA2, yB2, *, boot: int, seed: int
) -> dict:
    """Bootstrap CI on ``rho(pool1) - rho(pool2)`` — a CI on the DIFFERENCE, not two CIs.

    The pools are disjoint layout sets, so this is an unpaired two-sample bootstrap: resample
    each pool independently per replicate and take the difference of the resulting rhos. That
    is the right null for "these two cells have the same ceiling", and it is the statistic a
    reader should demand before accepting "the CIs overlap, so they agree".
    """
    rng = np.random.default_rng(seed)
    observed = V._spearman(yA1, yB1) - V._spearman(yA2, yB2)
    n1, n2 = len(yA1), len(yA2)
    draws = []
    for _ in range(boot):
        i1, i2 = rng.integers(0, n1, n1), rng.integers(0, n2, n2)
        r1, r2 = V._spearman(yA1[i1], yB1[i1]), V._spearman(yA2[i2], yB2[i2])
        if np.isfinite(r1) and np.isfinite(r2):
            draws.append(r1 - r2)
    arr = np.array(draws)
    return {
        "delta_rho": float(observed),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_gt_0": float((arr > 0).mean()),
        "p_two_sided_ge_0": float(min((arr > 0).mean(), (arr < 0).mean()) * 2),
        "n_boot": len(arr),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bank", type=int, default=200_000)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sA = E.load_target_surface(A_NAME, banks.SURFACE_DIR, "native")
    sB = E.load_target_surface(B_NAME, banks.SURFACE_DIR, "native")
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    ctx = E.gauge_context(args.corpus)
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    print(f"frame=native (asserted) corpus={ctx.corpus_name}", flush=True)

    def ms(pool):
        return banks.ms_of(pool, sA, objective), banks.ms_of(pool, sB, objective)

    archive = banks.load_archive()
    rand_bank = banks.random_bank(args.bank, args.seed)
    yA_r, yB_r = ms(rand_bank)
    yA_a, yB_a = ms(archive)
    sd_ref_A, sd_ref_B = float(yA_r.std(ddof=1)), float(yB_r.std(ddof=1))
    arc_sd_A, arc_sd_B = float(yA_a.std(ddof=1)), float(yB_a.std(ddof=1))
    rng = np.random.default_rng(args.seed)
    ref_pool = [rand_bank[i] for i in rng.choice(len(rand_bank), args.n, replace=False)]
    arc_pool = [archive[i] for i in rng.choice(len(archive), args.n, replace=False)]
    yA_ref, yB_ref = ms(ref_pool)
    yA_arc, yB_arc = ms(arc_pool)
    rho_ref = V._spearman(yA_ref, yB_ref)
    out: dict = {
        "schema_version": 1,
        "corpus": ctx.corpus_name,
        "corpus_sha256": dict(ctx.identity.get("sha256", {})),
        "frame": "native",
        "seed": args.seed,
        "n_per_cell": args.n,
        "rho_ref_random_wide": rho_ref,
        "archive_rho": V._spearman(yA_arc, yB_arc),
        "sd_reference": {"A": sd_ref_A, "B": sd_ref_B},
        "archive_sd": {"A": arc_sd_A, "B": arc_sd_B},
        "archive_u": {"A": arc_sd_A / sd_ref_A, "B": arc_sd_B / sd_ref_B},
    }

    # ---- P1: does the two-stage selection ITSELF destroy rho? --------------------------
    print("\n[P1] two-stage selection at WIDE targets — is my construction the cause?", flush=True)
    p1 = []
    for mult in (1.0, 0.75, 0.5, 0.25, 0.1, 0.05):
        pool, meta = banks.joint_band_filter(
            rand_bank, yA_r, yB_r,
            target_sd_a=sd_ref_A * mult, target_sd_b=sd_ref_B * mult,
            n=args.n, seed=args.seed, oversample=30,
        )
        yA, yB = ms(pool)
        row = {
            "target_mult_of_reference": mult,
            "rho": V._spearman(yA, yB),
            "u_A": float(yA.std(ddof=1)) / sd_ref_A,
            "u_B": float(yB.std(ddof=1)) / sd_ref_B,
            "achieved_sd_A": float(yA.std(ddof=1)),
            "achieved_sd_B": float(yB.std(ddof=1)),
            **meta,
        }
        row["rho_pred"] = attenuated_rho(rho_ref, float(np.sqrt(row["u_A"] * row["u_B"])))
        p1.append(row)
        print(
            f"  mult={mult:<5} rho={row['rho']:+.4f} u_A={row['u_A']:.4f} u_B={row['u_B']:.4f} "
            f"pred={row['rho_pred']:+.4f}",
            flush=True,
        )
    out["P1_selection_artifact_check"] = p1
    wide = [r for r in p1 if r["target_mult_of_reference"] >= 0.75]
    out["P1_verdict"] = (
        "construction is NOT the cause: the same two-stage selection at wide targets retains "
        f"rho {min(r['rho'] for r in wide):+.4f}..{max(r['rho'] for r in wide):+.4f}"
        if wide and min(r["rho"] for r in wide) > 0.5
        else "INCONCLUSIVE — the two-stage selection may itself depress rho; treat the matched cell with caution"
    )
    print(f"  -> {out['P1_verdict']}", flush=True)

    # ---- P2: a CI on the DIFFERENCE archive - jointband -------------------------------
    print("\n[P2] paired CI on the DIFFERENCE (archive vs matched random)", flush=True)
    joint_full, jf_meta = banks.joint_band_filter(
        rand_bank, yA_r, yB_r, target_sd_a=arc_sd_A, target_sd_b=arc_sd_B,
        n=args.n, seed=args.seed,
    )
    yA_j, yB_j = ms(joint_full)
    diff = paired_rho_difference(yA_arc, yB_arc, yA_j, yB_j, boot=args.boot, seed=args.seed)
    out["P2_archive_minus_jointband"] = {
        **diff,
        "archive_rho": V._spearman(yA_arc, yB_arc),
        "jointband_rho": V._spearman(yA_j, yB_j),
        "jointband_u_A": float(yA_j.std(ddof=1)) / sd_ref_A,
        "jointband_u_B": float(yB_j.std(ddof=1)) / sd_ref_B,
        "jointband_meta": jf_meta,
    }
    print(
        f"  archive {V._spearman(yA_arc, yB_arc):+.4f} - jointband {V._spearman(yA_j, yB_j):+.4f} "
        f"= {diff['delta_rho']:+.4f} CI [{diff['ci95'][0]:+.4f},{diff['ci95'][1]:+.4f}] "
        f"p(two-sided)={diff['p_two_sided_ge_0']:.4f}",
        flush=True,
    )

    # ---- P3: the (u_A, u_B) surface, so no single match is load-bearing ---------------
    print("\n[P3] (u_A,u_B) grid — the whole surface, not one hand-picked match", flush=True)
    grid = []
    for ma in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        for mb in (0.05, 0.1, 0.16, 0.3, 0.6, 1.0):
            pool, meta = banks.joint_band_filter(
                rand_bank, yA_r, yB_r,
                target_sd_a=sd_ref_A * ma, target_sd_b=sd_ref_B * mb,
                n=args.n, seed=args.seed, oversample=30,
            )
            yA, yB = ms(pool)
            uA, uB = float(yA.std(ddof=1)) / sd_ref_A, float(yB.std(ddof=1)) / sd_ref_B
            grid.append(
                {
                    "target_u_A": ma, "target_u_B": mb,
                    "u_A": uA, "u_B": uB,
                    "rho": V._spearman(yA, yB),
                    "rho_pred_geom": attenuated_rho(rho_ref, float(np.sqrt(uA * uB))),
                }
            )
            print(
                f"  target u=({ma:.2f},{mb:.2f}) -> achieved ({uA:.4f},{uB:.4f}) "
                f"rho={grid[-1]['rho']:+.4f} pred={grid[-1]['rho_pred_geom']:+.4f}",
                flush=True,
            )
    out["P3_u_grid"] = grid
    near = [
        g for g in grid
        if 0.5 * out["archive_u"]["A"] <= g["u_A"] <= 2.0 * out["archive_u"]["A"]
        and 0.5 * out["archive_u"]["B"] <= g["u_B"] <= 2.0 * out["archive_u"]["B"]
    ]
    out["P3_cells_near_archive_u"] = near
    if near:
        print(
            f"  -> {len(near)} random-lineage cells within 2x of the archive's u: "
            f"rho {min(g['rho'] for g in near):+.4f}..{max(g['rho'] for g in near):+.4f} "
            f"(archive {out['archive_rho']:+.4f})",
            flush=True,
        )

    # ---- P4: ONE curve rho(u)? fit on random lineage, test optimized lineage ----------
    print("\n[P4] one curve rho(u)? fit on RANDOM lineage, residuals for OPTIMIZED", flush=True)
    random_cells = [
        {"u": float(np.sqrt(g["u_A"] * g["u_B"])), "rho": g["rho"], "label": f"grid({g['target_u_A']},{g['target_u_B']})"}
        for g in grid
    ]
    random_cells.append({"u": 1.0, "rho": rho_ref, "label": "random-wide"})
    optimized_cells = []
    for k in (0, 1, 2, 3, 5, 8, 12, 20, 30):
        pool = arc_pool if k == 0 else banks.kswap_bank(archive, k, args.n, args.seed)
        yA, yB = ms(pool)
        uA, uB = float(yA.std(ddof=1)) / sd_ref_A, float(yB.std(ddof=1)) / sd_ref_B
        optimized_cells.append(
            {
                "label": f"archive+{k}swaps" if k else "archive",
                "k_swaps": k, "u_A": uA, "u_B": uB,
                "u": float(np.sqrt(uA * uB)), "rho": V._spearman(yA, yB),
            }
        )
    # The attenuation formula IS the curve; no free parameter beyond rho_ref (measured).
    for cell in random_cells + optimized_cells:
        cell["rho_pred"] = attenuated_rho(rho_ref, cell["u"])
        cell["residual"] = cell["rho"] - cell["rho_pred"]
    r_res = np.array([c["residual"] for c in random_cells])
    o_res = np.array([c["residual"] for c in optimized_cells])
    from scipy.stats import mannwhitneyu

    stat = mannwhitneyu(o_res, r_res, alternative="two-sided")
    out["P4_one_curve"] = {
        "note": (
            "Thorndike case-2 attenuation of the measured rho_ref, evaluated at each pool's "
            "geometric-mean restriction u = sqrt(u_A*u_B). No free parameter is fitted. If both "
            "lineages sit on this one curve, restriction is the mechanism; a systematic positive "
            "residual for the optimized lineage would mean near-optimality adds agreement."
        ),
        "random_lineage_cells": random_cells,
        "optimized_lineage_cells": optimized_cells,
        "random_residual_mean": float(r_res.mean()),
        "random_residual_sd": float(r_res.std(ddof=1)),
        "optimized_residual_mean": float(o_res.mean()),
        "optimized_residual_sd": float(o_res.std(ddof=1)),
        "mannwhitney_u": float(stat.statistic),
        "mannwhitney_p": float(stat.pvalue),
    }
    print(
        f"  residual mean: random {r_res.mean():+.4f} (sd {r_res.std(ddof=1):.4f}, n={len(r_res)}) | "
        f"optimized {o_res.mean():+.4f} (sd {o_res.std(ddof=1):.4f}, n={len(o_res)})",
        flush=True,
    )
    print(f"  Mann-Whitney on residuals: U={stat.statistic:.1f} p={stat.pvalue:.4f}", flush=True)
    for cell in optimized_cells:
        print(
            f"    {cell['label']:<18} u={cell['u']:.4f} rho={cell['rho']:+.4f} "
            f"pred={cell['rho_pred']:+.4f} resid={cell['residual']:+.4f}",
            flush=True,
        )

    out["elapsed_seconds"] = time.time() - t0
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nwrote {args.out} ({out['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
