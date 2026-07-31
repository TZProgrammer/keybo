"""CONFIRMATORY PASS — three things the first pass left unsettled, all of them load-bearing.

The primary run produced a result that contradicts part of my own preregistered prediction, so
before any of it is read as a verdict it gets attacked on the three axes where it could be wrong:

C1  **IS THE WITHIN-LEG COMPARISON EVEN FAIR?** Within-instrument reliability is a correlation
    between COMMUNITY's own per-seed refits, so it is attenuated by restriction in the PER-SEED
    variables — not by ``u_B``, which is measured on the seedMEAN surface. If the archive's
    per-seed spread is wider than my matched pool's, then "the archive keeps within high" is a
    spread artifact and not a near-optimality effect. So: measure ``u_seed`` for every pool, and
    build a random-lineage arm matched to the archive's PER-SEED spread rather than its ``u_B``.
    This is the same "matched on WHAT?" trap that made the prior round's boxmatch unreadable.

C2  **DOES THE F4 TRIGGER SURVIVE REPLICATION?** The q-ladder ran ONE construction seed per q and
    triggered F4 (|rho(1/4) - rho(4)| = 0.2630 > 0.20). But the measured replicate sd of cross-rho
    is ~0.036-0.045 and ladder-q1 (seed 2000) landed at -0.1006 against a 12-seed sym mean of
    +0.0194 — i.e. single-draw ladder cells scatter by more than the effect. Replicated here.

C3  **IS THE RANDOM-LINEAGE `level-asym-4x` SIGNATURE REAL?** That cell reproduced BOTH legs
    (within +0.9676 vs archive +0.9647; cross +0.2998, CI containing the archive's +0.2184) from a
    random pool. If it replicates, the two-legged signature is NOT unique to near-optimality —
    it just is not reachable at the archive's narrowness. One draw is not a result.

MODELLED ONLY. ``.native`` frame, asserted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import keybo.analysis.surfaces as S

import nplib as N
from asym import box_match_2d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank", type=int, default=200_000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--boot", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=6, help="replicates per ladder/level cell")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sA, sB = N.load_surface(N.A_NAME), N.load_surface(N.B_NAME)
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    per_seed = N.community_per_seed()
    mass = float(objective[3].sum())
    print(f"frame=native (asserted) n={args.n} reps={args.reps}", flush=True)

    archive = N.load_archive()
    rand_bank = N.random_bank(args.bank, args.seed)
    yA_r, yB_r = N.ms_of(rand_bank, sA, objective), N.ms_of(rand_bank, sB, objective)
    ref = {"mean_A": float(yA_r.mean()), "sd_A": float(yA_r.std(ddof=1)),
           "mean_B": float(yB_r.mean()), "sd_B": float(yB_r.std(ddof=1))}

    # Per-seed reference spreads on the SAME bank, so a per-seed restriction fraction is
    # defined on exactly the axis the within-instrument statistic is computed on.
    print("scoring the bank on COMMUNITY's 3 per-seed surfaces (the within-leg axes) ...", flush=True)
    yS_r = {s: N.seed_ms(rand_bank, a, objective) for s, a in per_seed.items()}
    sd_ref_seed = {s: float(v.std(ddof=1)) for s, v in yS_r.items()}
    print(f"  per-seed reference sds: { {k: round(v, 4) for k, v in sd_ref_seed.items()} }", flush=True)

    rng0 = np.random.default_rng(args.seed)
    _ = rng0.choice(len(rand_bank), args.n, replace=False)
    arc_pool = [archive[i] for i in rng0.choice(len(archive), args.n, replace=False)]
    arc = N.profile(arc_pool, sA, sB, objective, ref)
    TARGET_UA, TARGET_UB = arc["u_A"], arc["u_B"]
    U_GEO = float(np.sqrt(TARGET_UA * TARGET_UB))

    cells: list[dict] = []

    def u_seed_of(pool):
        """Restriction fraction in each of COMMUNITY's per-seed variables, plus their geo-mean.

        This is the axis the WITHIN-instrument statistic actually lives on. ``u_B`` is measured
        on the seedMEAN surface and is NOT the same quantity — averaging 3 refits cancels part of
        their independent noise, so the mean's spread and a single seed's spread differ.
        """
        out = {s: float(N.seed_ms(pool, a, objective).std(ddof=1)) / sd_ref_seed[s]
               for s, a in per_seed.items()}
        return {"per_seed": out, "geo": float(np.exp(np.mean(np.log(list(out.values())))))}

    def record(label, kind, pool, extra=None):
        p = N.profile(pool, sA, sB, objective, ref)
        yA, yB = N.ms_of(pool, sA, objective), N.ms_of(pool, sB, objective)
        boot = N.bootstrap_rho(yA, yB, boot=args.boot, seed=args.seed)
        wit = N.within_instrument(pool, per_seed, objective)
        us = u_seed_of(pool)
        row = {"label": label, "kind": kind, **p, "rho_ci95": boot["ci95"],
               "rho_boot_sd": boot["boot_sd"], "leg_cross": p["rho_spearman"],
               "leg_within": wit["mean"], "within_instrument_pairs": wit["pairs"],
               "u_seed": us["per_seed"], "u_seed_geo": us["geo"], **(extra or {})}
        cells.append(row)
        print(f"  {label:<24} cross={p['rho_spearman']:+.4f} within={wit['mean']:+.4f} "
              f"u_A={p['u_A']:.4f} u_B={p['u_B']:.4f} u_seed_geo={us['geo']:.4f} q={p['u_ratio']:.4f}",
              flush=True)
        return row

    def cut(uA, uB, seed):
        return box_match_2d(rand_bank, yA_r, yB_r, target_sd_a=uA * ref["sd_A"],
                            target_sd_b=uB * ref["sd_B"], n=args.n, seed=seed)

    print("\n[REF]", flush=True)
    rng_ref = np.random.default_rng(args.seed)
    record("random-wide", "reference",
           [rand_bank[i] for i in rng_ref.choice(len(rand_bank), args.n, replace=False)])
    arc_row = record("archive-x400", "reference", arc_pool)
    print(f"  ARCHIVE u_seed per seed: { {k: round(v, 4) for k, v in arc_row['u_seed'].items()} }",
          flush=True)

    # ---- C1: is the within-leg comparison fair? -----------------------------------------
    # Arm 1: matched on u_B (what the primary run did). Arm 2: matched on the archive's own
    # PER-SEED spread. If arm 2 still has lower within-reliability than the archive, the gap is
    # not a spread artifact.
    print("\n[C1] within-leg fairness: match u_B, then match the archive's PER-SEED spread",
          flush=True)
    c1 = {"archive_u_seed_geo": arc_row["u_seed_geo"], "archive_within": arc_row["leg_within"],
          "arms": []}
    for r in range(args.reps):
        pool, meta = cut(TARGET_UA, TARGET_UB, 5000 + r)
        row = record(f"c1-match-uB-r{r}", "c1-match-uB", pool, {"box": meta})
        c1["arms"].append({"arm": "match-uB", "rep": r, "within": row["leg_within"],
                           "cross": row["leg_cross"], "u_seed_geo": row["u_seed_geo"],
                           "u_A": row["u_A"], "u_B": row["u_B"]})

    # Match the archive's per-seed spread by bisecting u_B (monotone: seedmean spread drives
    # per-seed spread) while HOLDING the asymmetry q at the archive's value, so this arm differs
    # from the archive in near-optimality and nothing else that the within leg cares about.
    print(f"\n  bisecting u_B to hit the archive's u_seed_geo = {arc_row['u_seed_geo']:.4f} "
          f"(q held at {TARGET_UA / TARGET_UB:.4f})", flush=True)
    q_fixed = TARGET_UA / TARGET_UB
    lo, hi = TARGET_UB, TARGET_UB * 20.0
    hist = []
    for it in range(9):
        mid = float(np.sqrt(lo * hi))
        pool, meta = cut(q_fixed * mid, mid, 6000)
        got = u_seed_of(pool)["geo"]
        hist.append({"iter": it, "u_B": mid, "u_seed_geo": got})
        print(f"    iter {it}: u_B={mid:.4f} -> u_seed_geo={got:.4f} "
              f"(target {arc_row['u_seed_geo']:.4f})", flush=True)
        if abs(got / arc_row["u_seed_geo"] - 1.0) <= 0.03:
            break
        if got < arc_row["u_seed_geo"]:
            lo = mid
        else:
            hi = mid
    u_b_seedmatched = hist[-1]["u_B"]
    c1["bisection"] = hist
    c1["u_B_that_matches_archive_u_seed"] = u_b_seedmatched
    for r in range(args.reps):
        pool, meta = cut(q_fixed * u_b_seedmatched, u_b_seedmatched, 6100 + r)
        row = record(f"c1-match-useed-r{r}", "c1-match-useed", pool, {"box": meta})
        c1["arms"].append({"arm": "match-useed", "rep": r, "within": row["leg_within"],
                           "cross": row["leg_cross"], "u_seed_geo": row["u_seed_geo"],
                           "u_A": row["u_A"], "u_B": row["u_B"]})

    # ---- C2: replicated q-ladder --------------------------------------------------------
    print("\n[C2] replicated q-ladder at FIXED u_geo (F4 retest)", flush=True)
    ladder: dict[float, list[dict]] = {}
    for q in (1 / 16, 1 / 4, 1.0, 4.0, 16.0):
        ladder[q] = []
        for r in range(args.reps):
            uA, uB = U_GEO * np.sqrt(q), U_GEO / np.sqrt(q)
            pool, meta = cut(uA, uB, 7000 + 100 * int(round(np.log2(q))) + r)
            row = record(f"c2-q{q:g}-r{r}", "c2-ladder", pool, {"requested_q": q, "box": meta})
            ladder[q].append(row)

    # ---- C3: replicated level-asym-4x (and its sym partner) -----------------------------
    print("\n[C3] replicated 4x-level arms (does the two-legged signature replicate?)", flush=True)
    lvl: dict[str, list[dict]] = {"asym-4x": [], "sym-4x": []}
    for r in range(args.reps):
        pool, meta = cut(TARGET_UA * 4, TARGET_UB * 4, 8000 + r)
        lvl["asym-4x"].append(record(f"c3-asym4x-r{r}", "c3-level", pool, {"box": meta}))
        pool, meta = cut(U_GEO * 4, U_GEO * 4, 8000 + r)
        lvl["sym-4x"].append(record(f"c3-sym4x-r{r}", "c3-level", pool, {"box": meta}))

    # ---- INFERENCE ----------------------------------------------------------------------
    from scipy.stats import mannwhitneyu

    def agg(rows, key):
        a = np.array([r[key] for r in rows])
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "min": float(a.min()),
                "max": float(a.max()), "n": len(a)}

    mu = [a for a in c1["arms"] if a["arm"] == "match-uB"]
    ms_ = [a for a in c1["arms"] if a["arm"] == "match-useed"]
    c1["match_uB"] = {"within": agg(mu, "within"), "cross": agg(mu, "cross"),
                      "u_seed_geo": agg(mu, "u_seed_geo")}
    c1["match_useed"] = {"within": agg(ms_, "within"), "cross": agg(ms_, "cross"),
                         "u_seed_geo": agg(ms_, "u_seed_geo")}
    c1["verdict"] = {
        "archive_within_minus_match_uB": arc_row["leg_within"] - c1["match_uB"]["within"]["mean"],
        "archive_within_minus_match_useed": arc_row["leg_within"] - c1["match_useed"]["within"]["mean"],
        "archive_cross_minus_match_uB": arc_row["leg_cross"] - c1["match_uB"]["cross"]["mean"],
        "archive_cross_minus_match_useed": arc_row["leg_cross"] - c1["match_useed"]["cross"]["mean"],
        "note": (
            "match-uB is matched to the archive on the SEEDMEAN restriction (u_A, u_B). "
            "match-useed is matched on the PER-SEED restriction, which is the axis the "
            "within-instrument statistic actually lives on. If the archive still exceeds "
            "match-useed on the within leg, the gap is not a spread artifact."
        ),
    }
    ladder_agg = {
        f"{q:g}": {"cross": agg(rows, "leg_cross"), "within": agg(rows, "leg_within"),
                   "achieved_q": agg(rows, "u_ratio")}
        for q, rows in ladder.items()
    }
    f4 = []
    for q in (1 / 16, 1 / 4):
        a = np.array([r["leg_cross"] for r in ladder[q]])
        b = np.array([r["leg_cross"] for r in ladder[1 / q]])
        u = mannwhitneyu(a, b)
        f4.append({"q": q, "inv_q": 1 / q, "cross_mean_q": float(a.mean()),
                   "cross_mean_inv_q": float(b.mean()),
                   "abs_diff_of_means": float(abs(a.mean() - b.mean())),
                   "pooled_replicate_sd": float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)),
                   "mannwhitney_p": float(u.pvalue),
                   "exceeds_0.20_threshold": bool(abs(a.mean() - b.mean()) > 0.20)})
    lvl_agg = {k: {"cross": agg(v, "leg_cross"), "within": agg(v, "leg_within"),
                   "u_B": agg(v, "u_B"), "u_seed_geo": agg(v, "u_seed_geo")}
               for k, v in lvl.items()}
    a4c = np.array([r["leg_cross"] for r in lvl["asym-4x"]])
    a4w = np.array([r["leg_within"] for r in lvl["asym-4x"]])
    c3 = {
        "asym4x_reproduces_both_legs": bool(a4w.mean() >= 0.90 and a4c.mean() <= 0.30),
        "asym4x_cross_ci_contains_archive": bool(
            a4c.mean() - 2 * a4c.std(ddof=1) <= arc_row["leg_cross"] <= a4c.mean() + 2 * a4c.std(ddof=1)
        ),
        "note": (
            "If this replicates, the archive's two-legged signature (high within + low cross) is "
            "NOT unique to near-optimality — but note it is reached at 4x the archive's u, i.e. "
            "from a much WIDER pool. Reproducing a signature at a different restriction is a "
            "weaker claim than reproducing it at matched restriction, and is reported as such."
        ),
    }
    print("\n=== CONFIRMATORY VERDICT ===", flush=True)
    print(f"  C1 archive within {arc_row['leg_within']:+.4f} | match-uB "
          f"{c1['match_uB']['within']['mean']:+.4f} (sd {c1['match_uB']['within']['sd']:.4f}, "
          f"u_seed_geo {c1['match_uB']['u_seed_geo']['mean']:.4f}) | match-useed "
          f"{c1['match_useed']['within']['mean']:+.4f} (sd {c1['match_useed']['within']['sd']:.4f}, "
          f"u_seed_geo {c1['match_useed']['u_seed_geo']['mean']:.4f} vs archive "
          f"{arc_row['u_seed_geo']:.4f})", flush=True)
    print(f"     archive - match-useed on WITHIN = {c1['verdict']['archive_within_minus_match_useed']:+.4f}; "
          f"on CROSS = {c1['verdict']['archive_cross_minus_match_useed']:+.4f}", flush=True)
    for row in f4:
        print(f"  C2 F4 q={row['q']:.4f} vs {row['inv_q']:g}: |Δmeans|={row['abs_diff_of_means']:.4f} "
              f"(pooled rep sd {row['pooled_replicate_sd']:.4f}) MW p={row['mannwhitney_p']:.4f} "
              f"-> {'>0.20 TRIGGERED' if row['exceeds_0.20_threshold'] else 'below 0.20'}", flush=True)
    print(f"  C3 asym-4x cross={a4c.mean():+.4f} (sd {a4c.std(ddof=1):.4f}) within={a4w.mean():+.4f} "
          f"(sd {a4w.std(ddof=1):.4f}) -> both legs reproduced: {c3['asym4x_reproduces_both_legs']}",
          flush=True)

    payload = {
        "schema_version": 1,
        "note": "CONFIRMATORY pass for the necessity probe. MODELLED ONLY, .native frame.",
        "frame": "native",
        "surface_sha256": {N.A_NAME: sA.sha256, N.B_NAME: sB.sha256},
        "seed": args.seed, "n_per_cell": args.n, "reps": args.reps, "bootstrap": args.boot,
        "reference": ref, "per_seed_reference_sd": sd_ref_seed,
        "archive_target": {"u_A": TARGET_UA, "u_B": TARGET_UB, "q": q_fixed, "u_geo": U_GEO,
                           "u_seed_geo": arc_row["u_seed_geo"], "u_seed": arc_row["u_seed"]},
        "C1_within_leg_fairness": c1,
        "C2_ladder_replicated": {"per_q": ladder_agg, "F4_tests": f4},
        "C3_level_4x_replicated": {"agg": lvl_agg, "verdict": c3},
        "cells": cells,
        "elapsed_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(payload, h, indent=2)
    print(f"\nwrote {args.out} ({len(cells)} cells, {payload['elapsed_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
