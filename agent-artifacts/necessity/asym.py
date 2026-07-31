"""THE NECESSITY PROBE — an asymmetrically restricted RANDOM pool at u_A/u_B ~ 0.25.

The confound being broken: the Pareto archive is BOTH near-optimal AND asymmetrically
restricted (u_A 0.0399, u_B 0.1605, q = 0.249). Those are entangled, so neither the
"near-optimality destroys cross-source agreement" reading nor the "restriction geometry does
it" reading is identified by the archive alone. A random-lineage pool restricted to the SAME
TWO-SIDED geometry but with no search anywhere in its construction separates them.

Why TWO parameters and not one: P4 fed Thorndike case-2 a single ``u = sqrt(u_A u_B)``,
discarding the asymmetry that is the whole question, so its null could only ever mean "no
effect DETECTED". Every cell here is specified by BOTH ``u_A`` and ``u_B``, and reports the
ACHIEVED pair. The `sym-match` control holds the geometric mean FIXED and changes ONLY the
asymmetry, which is the one-variable-at-a-time contrast the prior round never cut.

Two channels are reported for every cell, because the archive's signature is TWO-legged:
  * instrument-vs-instrument = cross-source Spearman(AALTO_BASE, COMMUNITY_BASE)  [archive +0.2184]
  * instrument-vs-itself     = mean pairwise Spearman over COMMUNITY's 3 per-seed refits [archive +0.9647]
A pool that reproduces only the first leg has not reproduced the archive; `boxmatch` sat at
within +0.4605, i.e. near its own refit-noise floor, so its low cross-rho was partly attenuation.

MODELLED ONLY. ``.native`` frame, asserted. Nothing adopted, no weight flipped, no default changed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import keybo.analysis.surfaces as S

import nplib as N


def box_match_2d(
    layouts: list[str],
    yA: np.ndarray,
    yB: np.ndarray,
    *,
    target_sd_a: float,
    target_sd_b: float,
    n: int,
    seed: int,
    iters: int = 80,
    tol: float = 0.02,
) -> tuple[list[str], dict]:
    """``n`` layouts from a 2-D box in the (y_A, y_B) plane hitting BOTH achieved sds.

    Selection is a rectangle centred on the bank's MEDIAN in each source, with the two
    half-widths driven by independent multiplicative feedback on the **achieved** sd of the
    selected set — not on the box geometry, which is the mistake that made the prior round's
    two-stage band land at C = 0.553 when it was aiming for 0.085. Feedback on the realized
    statistic converges to the realized statistic by construction.

    The exponent 0.7 damps the update so the two axes do not fight: shrinking the A-window
    changes which layouts are in the box and therefore also perturbs the achieved B-sd. Under-
    relaxation makes the coupled iteration contract instead of oscillating.

    A uniform sd target is reachable only if the box holds >= n candidates; when it does not,
    both half-widths grow rather than one, so the box keeps its aspect ratio while it inflates
    and the ratio being targeted is not silently destroyed by the rescue path.
    """
    rng = np.random.default_rng(seed)
    a0, b0 = float(np.median(yA)), float(np.median(yB))
    ha, hb = float(yA.std(ddof=1)), float(yB.std(ddof=1))
    best: tuple[float, np.ndarray] | None = None
    history = []
    for _ in range(iters):
        mask = (np.abs(yA - a0) <= ha) & (np.abs(yB - b0) <= hb)
        idx = np.flatnonzero(mask)
        if len(idx) < n:
            ha, hb = ha * 1.30, hb * 1.30  # inflate BOTH: preserve the aspect ratio
            continue
        pick = idx[rng.choice(len(idx), n, replace=False)]
        sd_a, sd_b = float(yA[pick].std(ddof=1)), float(yB[pick].std(ddof=1))
        # Score = worst relative miss across the two axes, so "matched" means BOTH matched.
        miss = max(abs(sd_a / target_sd_a - 1.0), abs(sd_b / target_sd_b - 1.0))
        history.append({"half_a": ha, "half_b": hb, "sd_a": sd_a, "sd_b": sd_b, "worst_rel_miss": miss})
        if best is None or miss < best[0]:
            best = (miss, pick)
        if miss <= tol:
            break
        ha *= float(np.clip(target_sd_a / max(sd_a, 1e-12), 0.5, 2.0)) ** 0.7
        hb *= float(np.clip(target_sd_b / max(sd_b, 1e-12), 0.5, 2.0)) ** 0.7
    assert best is not None, "box_match_2d never found enough layouts"
    miss, pick = best
    order = pick[rng.permutation(len(pick))]  # kill any residual selection order
    return [layouts[i] for i in order], {
        "target_sd_a": target_sd_a,
        "target_sd_b": target_sd_b,
        "achieved_sd_a": float(yA[order].std(ddof=1)),
        "achieved_sd_b": float(yB[order].std(ddof=1)),
        "worst_rel_miss": miss,
        "converged_within_tol": bool(miss <= tol),
        "n_iters": len(history),
        "bank_size": len(layouts),
        "selected": int(len(order)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seed", type=int, default=0, help="bank seed (the reference bank's own seed)")
    ap.add_argument("--bank", type=int, default=200_000)
    ap.add_argument("--n", type=int, default=400, help="pool size — MATCHED to the archive's 400")
    ap.add_argument("--boot", type=int, default=8000, help="same protocol as the A5 test being compared against")
    ap.add_argument("--replicates", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sA, sB = N.load_surface(N.A_NAME), N.load_surface(N.B_NAME)
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    per_seed = N.community_per_seed()
    aalto_per_seed = sorted(p.name for p in Path(N.SURFACE_DIR).glob("AALTO*seed*.npy"))
    print(f"frame=native (asserted) n={args.n} bank={args.bank} R={args.replicates}", flush=True)
    print(f"COMMUNITY per-seed {sorted(per_seed)} | AALTO per-seed {aalto_per_seed or 'NONE (structural)'}", flush=True)

    archive = N.load_archive()
    rand_bank = N.random_bank(args.bank, args.seed)
    yA_r = N.ms_of(rand_bank, sA, objective)
    yB_r = N.ms_of(rand_bank, sB, objective)
    ref = {
        "mean_A": float(yA_r.mean()), "sd_A": float(yA_r.std(ddof=1)),
        "mean_B": float(yB_r.mean()), "sd_B": float(yB_r.std(ddof=1)),
    }
    print(f"reference bank sd_A={ref['sd_A']:.6f} sd_B={ref['sd_B']:.6f} "
          f"mean_A={ref['mean_A']:.4f} mean_B={ref['mean_B']:.4f}", flush=True)

    # ---- the ARCHIVE reference cell, drawn EXACTLY as the published one was -------------
    rng0 = np.random.default_rng(args.seed)
    _ = rng0.choice(len(rand_bank), args.n, replace=False)   # consume: random-wide draw first
    arc_pool = [archive[i] for i in rng0.choice(len(archive), args.n, replace=False)]
    arc = N.profile(arc_pool, sA, sB, objective, ref)
    TARGET_UA, TARGET_UB = arc["u_A"], arc["u_B"]
    print(f"\nARCHIVE TARGET (achieved, not requested): u_A={TARGET_UA:.6f} u_B={TARGET_UB:.6f} "
          f"q={TARGET_UA / TARGET_UB:.4f} rho={arc['rho_spearman']:+.4f}", flush=True)
    U_GEO = float(np.sqrt(TARGET_UA * TARGET_UB))
    print(f"geometric-mean narrowness held fixed for the ladder: u_geo={U_GEO:.6f}", flush=True)

    cells: list[dict] = []

    def record(label, kind, pool, extra=None, *, want=None):
        """Profile a pool on BOTH channels and both noise scales. One place, so no arm can omit a field."""
        p = N.profile(pool, sA, sB, objective, ref)
        yA = N.ms_of(pool, sA, objective)
        yB = N.ms_of(pool, sB, objective)
        boot = N.bootstrap_rho(yA, yB, boot=args.boot, seed=args.seed)
        wit = N.within_instrument(pool, per_seed, objective)
        row = {
            "label": label, "kind": kind, **p,
            "rho_ci95": boot["ci95"], "rho_boot_sd": boot["boot_sd"], "n_boot": boot["n_boot"],
            "within_instrument_mean": wit["mean"], "within_instrument_pairs": wit["pairs"],
            # The two-legged signature, as a single readable pair.
            "leg_cross": p["rho_spearman"], "leg_within": wit["mean"],
            **(extra or {}),
        }
        if want is not None:
            row["requested_u_A"], row["requested_u_B"] = want
            row["u_A_rel_miss"] = p["u_A"] / want[0] - 1.0
            row["u_B_rel_miss"] = p["u_B"] / want[1] - 1.0
        cells.append(row)
        print(f"  {label:<26} cross={p['rho_spearman']:+.4f} [{boot['ci95'][0]:+.3f},{boot['ci95'][1]:+.3f}] "
              f"within={wit['mean']:+.4f} u_A={p['u_A']:.4f} u_B={p['u_B']:.4f} q={p['u_ratio']:.4f} "
              f"meanA={p['mean_A']:.2f}", flush=True)
        return row

    print("\n[REF] the two anchors", flush=True)
    rng_ref = np.random.default_rng(args.seed)
    ref_pool = [rand_bank[i] for i in rng_ref.choice(len(rand_bank), args.n, replace=False)]
    record("random-wide", "reference", ref_pool)
    arc_row = record("archive-x400", "reference", arc_pool)

    def cut(uA: float, uB: float, seed: int):
        return box_match_2d(
            rand_bank, yA_r, yB_r,
            target_sd_a=uA * ref["sd_A"], target_sd_b=uB * ref["sd_B"], n=args.n, seed=seed,
        )

    # ---- PRIMARY: asym vs sym, PAIRED over R construction seeds ------------------------
    # Same bank, same construction seed per pair, so the ONLY difference within a pair is the
    # asymmetry of the target. That is what makes it a paired test rather than two samples.
    print(f"\n[PRIMARY] asym (q={TARGET_UA / TARGET_UB:.3f}) vs sym (q=1) at FIXED u_geo={U_GEO:.4f}, "
          f"paired over R={args.replicates} construction seeds", flush=True)
    pairs = []
    for r in range(args.replicates):
        cseed = 1000 + r
        pa, ma = cut(TARGET_UA, TARGET_UB, cseed)
        ra = record(f"asym-match-r{r}", "asym", pa, {"box": ma, "construction_seed": cseed},
                    want=(TARGET_UA, TARGET_UB))
        ps, msy = cut(U_GEO, U_GEO, cseed)
        rs = record(f"sym-match-r{r}", "sym", ps, {"box": msy, "construction_seed": cseed},
                    want=(U_GEO, U_GEO))
        pairs.append({"replicate": r, "construction_seed": cseed,
                      "asym_cross": ra["leg_cross"], "sym_cross": rs["leg_cross"],
                      "asym_within": ra["leg_within"], "sym_within": rs["leg_within"],
                      "asym_u_A": ra["u_A"], "asym_u_B": ra["u_B"], "asym_q": ra["u_ratio"],
                      "sym_u_A": rs["u_A"], "sym_u_B": rs["u_B"], "sym_q": rs["u_ratio"]})

    # ---- q-LADDER at FIXED geometric-mean narrowness ------------------------------------
    # u_A = sqrt(q)*u_geo, u_B = u_geo/sqrt(q) => u_A*u_B = u_geo^2 for every q. So overall
    # narrowness is constant along the ladder and ONLY the asymmetry moves. Both q and 1/q are
    # included: if rho is symmetric in log q, asymmetry is a magnitude; if not, WHICH source is
    # squeezed matters, which the single-u curve could not express.
    print("\n[LADDER] q swept at FIXED u_geo — only the ASYMMETRY moves", flush=True)
    ladder = []
    for q in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0):
        uA, uB = U_GEO * np.sqrt(q), U_GEO / np.sqrt(q)
        pool, meta = cut(uA, uB, 2000)
        row = record(f"ladder-q{q:g}", "ladder", pool,
                     {"box": meta, "requested_q": q, "construction_seed": 2000}, want=(uA, uB))
        ladder.append(row)

    # ---- LEVEL ladder: is it the RATIO or the overall narrowness? -----------------------
    print("\n[LEVEL] asym and sym at 1x / 2x / 4x the u-levels (ratio fixed within each arm)", flush=True)
    for mult in (1.0, 2.0, 4.0):
        pool, meta = cut(TARGET_UA * mult, TARGET_UB * mult, 3000)
        record(f"level-asym-{mult:g}x", "level", pool, {"box": meta, "u_multiplier": mult,
               "construction_seed": 3000}, want=(TARGET_UA * mult, TARGET_UB * mult))
        pool, meta = cut(U_GEO * mult, U_GEO * mult, 3000)
        record(f"level-sym-{mult:g}x", "level", pool, {"box": meta, "u_multiplier": mult,
               "construction_seed": 3000}, want=(U_GEO * mult, U_GEO * mult))

    # ---- INFERENCE ----------------------------------------------------------------------
    from scipy.stats import wilcoxon

    def arr(key):
        return np.array([p[key] for p in pairs])

    d_cross = arr("asym_cross") - arr("sym_cross")
    d_within = arr("asym_within") - arr("sym_within")
    asym_cross, sym_cross = arr("asym_cross"), arr("sym_cross")

    def wil(d):
        if len(d) >= 6 and np.any(d != 0):
            w = wilcoxon(d)
            return {"statistic": float(w.statistic), "p_two_sided": float(w.pvalue)}
        return None

    # UNPAIRED archive-vs-asym: disjoint layout universes, different lineage. Labelled as such;
    # the paired test above is the one with the tight resolution and it does not apply here.
    yA_ar = N.ms_of(arc_pool, sA, objective)
    yB_ar = N.ms_of(arc_pool, sB, objective)
    rep0 = [c for c in cells if c["label"] == "asym-match-r0"][0]
    pool_r0, _ = cut(TARGET_UA, TARGET_UB, 1000)
    yA_a0, yB_a0 = N.ms_of(pool_r0, sA, objective), N.ms_of(pool_r0, sB, objective)
    delta_unpaired = N.two_sample_delta(yA_ar, yB_ar, yA_a0, yB_a0, boot=args.boot, seed=args.seed)

    # Replicate-level contrast vs the archive's single value: the ruler is the replicate sd of
    # the asym arm, i.e. construction noise, NOT an imported ms/char floor.
    archive_minus_asym_mean = float(arc_row["leg_cross"] - asym_cross.mean())
    verdict = {
        "note": (
            "The archive's signature is TWO-legged: HIGH within-instrument (+0.9647) with LOW "
            "cross-instrument (+0.2184). Necessity must be judged per leg. `asym` and `sym` are "
            "PAIRED (same bank, same construction seed, only the target's asymmetry differs); "
            "`archive` vs `asym` is UNPAIRED (disjoint layout universes, different lineage) and is "
            "labelled as such. The resolution floor is DERIVED here — the replicate sd of rho over "
            "R construction seeds — because the campaign's 0.17-0.24 / 0.72 floors are ms/char on "
            "layout TIME and this statistic is a correlation (the resolution quadruple fails on its "
            "`statistic` element, so importing that floor would be a units error)."
        ),
        "archive": {"cross": arc_row["leg_cross"], "within": arc_row["leg_within"],
                    "u_A": arc_row["u_A"], "u_B": arc_row["u_B"], "q": arc_row["u_ratio"]},
        "random_wide": {k: [c for c in cells if c["label"] == "random-wide"][0][v]
                        for k, v in (("cross", "leg_cross"), ("within", "leg_within"))},
        "asym": {
            "cross_mean": float(asym_cross.mean()), "cross_replicate_sd": float(asym_cross.std(ddof=1)),
            "cross_min": float(asym_cross.min()), "cross_max": float(asym_cross.max()),
            "within_mean": float(arr("asym_within").mean()),
            "within_replicate_sd": float(arr("asym_within").std(ddof=1)),
            "achieved_u_A_mean": float(arr("asym_u_A").mean()),
            "achieved_u_B_mean": float(arr("asym_u_B").mean()),
            "achieved_q_mean": float(arr("asym_q").mean()),
            "achieved_q_sd": float(arr("asym_q").std(ddof=1)),
        },
        "sym": {
            "cross_mean": float(sym_cross.mean()), "cross_replicate_sd": float(sym_cross.std(ddof=1)),
            "within_mean": float(arr("sym_within").mean()),
            "within_replicate_sd": float(arr("sym_within").std(ddof=1)),
            "achieved_q_mean": float(arr("sym_q").mean()),
        },
        "paired_asym_minus_sym_cross": {
            "mean": float(d_cross.mean()), "sd": float(d_cross.std(ddof=1)),
            "n": len(d_cross), "wilcoxon": wil(d_cross),
            "resolution_replicate_sd_of_difference": float(d_cross.std(ddof=1)),
        },
        "paired_asym_minus_sym_within": {
            "mean": float(d_within.mean()), "sd": float(d_within.std(ddof=1)),
            "n": len(d_within), "wilcoxon": wil(d_within),
        },
        "unpaired_archive_minus_asym_r0": delta_unpaired,
        "archive_minus_asym_replicate_mean_cross": archive_minus_asym_mean,
        "ladder_symmetry_in_log_q": [
            {"q": r["requested_q"], "achieved_q": r["u_ratio"], "cross": r["leg_cross"],
             "within": r["leg_within"]}
            for r in ladder
        ],
        # F5 gates everything: a cell that missed its two-sided target cannot test necessity.
        "F5_construction_validity": {
            "criterion": "max |achieved u / requested u - 1| <= 0.10 on BOTH sources, per asym cell",
            "worst_u_A_rel_miss": float(np.max(np.abs([c["u_A_rel_miss"] for c in cells
                                                        if c["kind"] == "asym"]))),
            "worst_u_B_rel_miss": float(np.max(np.abs([c["u_B_rel_miss"] for c in cells
                                                        if c["kind"] == "asym"]))),
        },
    }
    v = verdict["F5_construction_validity"]
    v["pass"] = bool(v["worst_u_A_rel_miss"] <= 0.10 and v["worst_u_B_rel_miss"] <= 0.10)

    print("\n=== VERDICT ===", flush=True)
    print(f"  archive       cross={arc_row['leg_cross']:+.4f}  within={arc_row['leg_within']:+.4f}  "
          f"q={arc_row['u_ratio']:.4f}", flush=True)
    print(f"  asym  (R={len(pairs)})  cross={asym_cross.mean():+.4f} (rep sd {asym_cross.std(ddof=1):.4f})  "
          f"within={arr('asym_within').mean():+.4f}  achieved q={arr('asym_q').mean():.4f}", flush=True)
    print(f"  sym   (R={len(pairs)})  cross={sym_cross.mean():+.4f} (rep sd {sym_cross.std(ddof=1):.4f})  "
          f"within={arr('sym_within').mean():+.4f}  achieved q={arr('sym_q').mean():.4f}", flush=True)
    print(f"  PAIRED asym-sym cross: {d_cross.mean():+.4f} (sd {d_cross.std(ddof=1):.4f}) "
          f"wilcoxon {verdict['paired_asym_minus_sym_cross']['wilcoxon']}", flush=True)
    print(f"  UNPAIRED archive-asym(r0): {delta_unpaired['delta_rho']:+.4f} "
          f"CI [{delta_unpaired['ci95'][0]:+.4f},{delta_unpaired['ci95'][1]:+.4f}] "
          f"p={delta_unpaired['p_two_sided']:.4f}", flush=True)
    print(f"  F5 construction validity: worst u_A miss {v['worst_u_A_rel_miss']:+.4f}, "
          f"worst u_B miss {v['worst_u_B_rel_miss']:+.4f} -> {'PASS' if v['pass'] else 'FAIL'}", flush=True)

    payload = {
        "schema_version": 1,
        "note": (
            "NECESSITY PROBE. Asymmetrically restricted RANDOM pool at the archive's own achieved "
            "u_A/u_B. Two-parameter by construction (the P4 misspecification was ONE u for a "
            "two-sided restriction). All u are ACHIEVED, measured from the pool's own scores. "
            "MODELLED ONLY, .native frame."
        ),
        "corpus": S.default_trigram_path(args.corpus),
        "frame": "native",
        "surface_sha256": {N.A_NAME: sA.sha256, N.B_NAME: sB.sha256},
        "community_per_seed": sorted(per_seed),
        "aalto_per_seed": aalto_per_seed,
        "seed": args.seed, "n_per_cell": args.n, "bootstrap": args.boot,
        "replicates": args.replicates,
        "random_bank_size": len(rand_bank), "archive_bank_size": len(archive),
        "reference": ref,
        "archive_target": {"u_A": TARGET_UA, "u_B": TARGET_UB, "q": TARGET_UA / TARGET_UB,
                           "u_geo": U_GEO},
        "identification_limit": (
            "LEVEL is structurally unmatchable: no random permutation reaches the archive's speed "
            f"(archive mean_A {arc_row['mean_A']:.2f} vs random-bank mean_A {ref['mean_A']:.2f} "
            "ms/trigram). Every random-lineage cell is centred on the random median. Pool size, "
            "replicate structure, scale and statistic ARE matched; level is not, and that is a "
            "stated limit rather than something this construction can close (trap 16)."
        ),
        "pairs": pairs,
        "cells": cells,
        "verdict": verdict,
        "elapsed_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(payload, h, indent=2)
    print(f"\nwrote {args.out} ({len(cells)} cells, {payload['elapsed_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
