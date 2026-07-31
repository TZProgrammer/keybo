"""BOUNDING PASS — the two numbers the confirmatory pass left as inequalities.

B1  **The C1 bisection was MIS-BRACKETED and I am fixing it rather than quoting it.** It searched
    ``u_B in [archive_u_B, 20 x archive_u_B]`` for the ``u_B`` whose PER-SEED spread matches the
    archive's ``u_seed_geo = 0.1617``. But a random pool cut at ``u_B = 0.1605`` already lands at
    ``u_seed_geo = 0.1815`` — ABOVE the target — so the true root is BELOW the bracket's lower
    end and the search plateaued at a 12% miss. The miss direction happens to be conservative
    (more per-seed spread => LESS attenuation => the random arm is FAVOURED, and it still lost),
    but "conservative" is a weaker claim than "matched", so the bracket is reopened downward here.

B2  **HOW MUCH signal does near-optimality add? The bound the brief asks for.** The archive
    reaches within-instrument reliability +0.9647 at ``u_B = 0.1605``. A random-lineage pool at
    the same asymmetry needs some LARGER ``u_B`` to reach the same reliability. That ratio is a
    direct, units-free answer to "bound by how much": it is how much extra seedmean spread a
    random pool must be given to make COMMUNITY's own refits agree as well as they do on the
    archive.

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
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    sA, sB = N.load_surface(N.A_NAME), N.load_surface(N.B_NAME)
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    per_seed = N.community_per_seed()
    archive = N.load_archive()
    rand_bank = N.random_bank(args.bank, args.seed)
    yA_r, yB_r = N.ms_of(rand_bank, sA, objective), N.ms_of(rand_bank, sB, objective)
    ref = {"mean_A": float(yA_r.mean()), "sd_A": float(yA_r.std(ddof=1)),
           "mean_B": float(yB_r.mean()), "sd_B": float(yB_r.std(ddof=1))}
    yS_r = {s: N.seed_ms(rand_bank, a, objective) for s, a in per_seed.items()}
    sd_ref_seed = {s: float(v.std(ddof=1)) for s, v in yS_r.items()}

    rng0 = np.random.default_rng(args.seed)
    _ = rng0.choice(len(rand_bank), args.n, replace=False)
    arc_pool = [archive[i] for i in rng0.choice(len(archive), args.n, replace=False)]
    arc = N.profile(arc_pool, sA, sB, objective, ref)
    Q = arc["u_A"] / arc["u_B"]
    arc_wit = N.within_instrument(arc_pool, per_seed, objective)["mean"]
    arc_useed = {s: float(N.seed_ms(arc_pool, a, objective).std(ddof=1)) / sd_ref_seed[s]
                 for s, a in per_seed.items()}
    ARC_USEED_GEO = float(np.exp(np.mean(np.log(list(arc_useed.values())))))
    print(f"archive: u_A={arc['u_A']:.4f} u_B={arc['u_B']:.4f} q={Q:.4f} within={arc_wit:+.4f} "
          f"u_seed_geo={ARC_USEED_GEO:.4f}", flush=True)

    def probe(u_b: float, seed: int) -> dict:
        """Cut a random pool at asymmetry Q and seedmean spread ``u_b``; report both channels."""
        pool, meta = box_match_2d(rand_bank, yA_r, yB_r, target_sd_a=Q * u_b * ref["sd_A"],
                                  target_sd_b=u_b * ref["sd_B"], n=args.n, seed=seed)
        us = {s: float(N.seed_ms(pool, a, objective).std(ddof=1)) / sd_ref_seed[s]
              for s, a in per_seed.items()}
        p = N.profile(pool, sA, sB, objective, ref)
        return {"requested_u_B": u_b, "achieved_u_A": p["u_A"], "achieved_u_B": p["u_B"],
                "achieved_q": p["u_ratio"], "cross": p["rho_spearman"],
                "within": N.within_instrument(pool, per_seed, objective)["mean"],
                "u_seed_geo": float(np.exp(np.mean(np.log(list(us.values()))))),
                "construction_seed": seed, "box": meta}

    def bisect(getter, target: float, lo: float, hi: float, *, label: str, iters: int = 14,
               tol: float = 0.02, seed: int = 9000) -> dict:
        """Monotone bisection on ``u_B``, with the bracket VERIFIED before it is trusted.

        A bisection whose bracket does not straddle the root returns its own endpoint and looks
        converged — the exact failure the confirmatory pass hit. So both ends are evaluated
        first and the bracket is expanded until it genuinely straddles, or reported as
        unbracketed.
        """
        print(f"\n[{label}] bisecting u_B for target {target:.4f}", flush=True)
        f_lo, f_hi = getter(probe(lo, seed)), getter(probe(hi, seed))
        print(f"  bracket check: u_B={lo:.5f} -> {f_lo:.4f} | u_B={hi:.5f} -> {f_hi:.4f}", flush=True)
        expand = 0
        while (f_lo - target) * (f_hi - target) > 0 and expand < 6:
            if abs(f_lo - target) < abs(f_hi - target):
                lo = lo / 3.0
                f_lo = getter(probe(lo, seed))
                print(f"  expand DOWN: u_B={lo:.5f} -> {f_lo:.4f}", flush=True)
            else:
                hi = hi * 3.0
                f_hi = getter(probe(hi, seed))
                print(f"  expand UP:   u_B={hi:.5f} -> {f_hi:.4f}", flush=True)
            expand += 1
        bracketed = (f_lo - target) * (f_hi - target) <= 0
        hist = [{"u_B": lo, "value": f_lo}, {"u_B": hi, "value": f_hi}]
        if bracketed:
            for _ in range(iters):
                mid = float(np.sqrt(lo * hi))
                r = probe(mid, seed)
                fm = getter(r)
                hist.append({"u_B": mid, "value": fm, "achieved_u_B": r["achieved_u_B"],
                             "cross": r["cross"], "within": r["within"],
                             "u_seed_geo": r["u_seed_geo"]})
                print(f"  u_B={mid:.5f} -> {fm:.4f} (target {target:.4f})", flush=True)
                if abs(fm - target) <= tol * max(abs(target), 1e-9):
                    break
                if (fm < target) == (f_lo < target):
                    lo, f_lo = mid, fm
                else:
                    hi, f_hi = mid, fm
        best = min(hist, key=lambda h: abs(h["value"] - target))
        return {"label": label, "target": target, "bracketed": bool(bracketed),
                "expansions": expand, "root_u_B": best["u_B"], "root_value": best["value"],
                "rel_miss": best["value"] / target - 1.0, "history": hist}

    # ---- B1: reopen the u_seed match downward -------------------------------------------
    b1 = bisect(lambda r: r["u_seed_geo"], ARC_USEED_GEO, arc["u_B"] / 4.0, arc["u_B"] * 2.0,
                label="B1 match archive u_seed_geo")
    print(f"\n[B1] replicating at the corrected u_B = {b1['root_u_B']:.5f}", flush=True)
    b1_reps = [probe(b1["root_u_B"], 9100 + r) for r in range(args.reps)]
    for r in b1_reps:
        print(f"  cross={r['cross']:+.4f} within={r['within']:+.4f} u_seed_geo={r['u_seed_geo']:.4f} "
              f"u_A={r['achieved_u_A']:.4f} u_B={r['achieved_u_B']:.4f} q={r['achieved_q']:.4f}",
              flush=True)

    # ---- B2: how much u_B does a RANDOM pool need to reach the archive's within? ---------
    b2 = bisect(lambda r: r["within"], arc_wit, arc["u_B"], arc["u_B"] * 4.0,
                label="B2 match archive within-reliability")
    print(f"\n[B2] replicating at u_B = {b2['root_u_B']:.5f}", flush=True)
    b2_reps = [probe(b2["root_u_B"], 9200 + r) for r in range(args.reps)]
    for r in b2_reps:
        print(f"  cross={r['cross']:+.4f} within={r['within']:+.4f} u_A={r['achieved_u_A']:.4f} "
              f"u_B={r['achieved_u_B']:.4f} q={r['achieved_q']:.4f}", flush=True)

    def agg(rows, key):
        a = np.array([r[key] for r in rows])
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": len(a)}

    b1a = {k: agg(b1_reps, k) for k in ("cross", "within", "u_seed_geo", "achieved_u_B")}
    b2a = {k: agg(b2_reps, k) for k in ("cross", "within", "achieved_u_B", "u_seed_geo")}
    ratio = b2a["achieved_u_B"]["mean"] / arc["u_B"]
    verdict = {
        "archive": {"u_A": arc["u_A"], "u_B": arc["u_B"], "q": Q, "cross": arc["rho_spearman"],
                    "within": arc_wit, "u_seed_geo": ARC_USEED_GEO, "u_seed": arc_useed},
        "B1_useed_matched_arm": {
            "corrected_u_B": b1["root_u_B"], "bracketed": b1["bracketed"],
            "rel_miss_on_u_seed": b1["rel_miss"], **b1a,
            "archive_minus_arm_within": arc_wit - b1a["within"]["mean"],
            "archive_minus_arm_cross": arc["rho_spearman"] - b1a["cross"]["mean"],
            "note": (
                "Matched to the archive on the PER-SEED restriction — the axis the "
                "within-instrument statistic actually lives on — with the bracket reopened "
                "downward after the confirmatory pass's search was found mis-bracketed."
            ),
        },
        "B2_within_matched_arm": {
            "u_B_needed": b2a["achieved_u_B"]["mean"], "bracketed": b2["bracketed"],
            **b2a,
            "u_B_ratio_random_over_archive": ratio,
            "note": (
                "THE QUANTITATIVE BOUND. A random-lineage pool at the archive's asymmetry needs "
                f"{ratio:.2f}x the archive's seedmean spread before COMMUNITY's own refits agree "
                "as well as they already do on the archive. Equivalently: at equal spread the "
                "archive carries that much more within-instrument-resolvable signal. Note the "
                "arm's CROSS-source rho does NOT come along for the ride, which is the whole "
                "dissociation."
            ),
        },
    }
    print("\n=== BOUNDS ===", flush=True)
    print(f"  B1 archive within {arc_wit:+.4f} vs u_seed-matched random {b1a['within']['mean']:+.4f} "
          f"(sd {b1a['within']['sd']:.4f}) at u_seed_geo {b1a['u_seed_geo']['mean']:.4f} vs archive "
          f"{ARC_USEED_GEO:.4f} -> gap {verdict['B1_useed_matched_arm']['archive_minus_arm_within']:+.4f}",
          flush=True)
    print(f"  B2 a random pool needs u_B = {b2a['achieved_u_B']['mean']:.4f} = "
          f"{ratio:.2f}x the archive's {arc['u_B']:.4f} to reach within {arc_wit:+.4f}; "
          f"its cross there is {b2a['cross']['mean']:+.4f} (archive {arc['rho_spearman']:+.4f})",
          flush=True)

    payload = {
        "schema_version": 1, "note": "BOUNDING pass. MODELLED ONLY, .native frame.",
        "frame": "native",
        "surface_sha256": {N.A_NAME: sA.sha256, N.B_NAME: sB.sha256},
        "seed": args.seed, "n_per_cell": args.n, "reps": args.reps,
        "reference": ref, "per_seed_reference_sd": sd_ref_seed,
        "B1_bisection": b1, "B1_replicates": b1_reps,
        "B2_bisection": b2, "B2_replicates": b2_reps,
        "verdict": verdict, "elapsed_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as h:
        json.dump(payload, h, indent=2)
    print(f"\nwrote {args.out} ({payload['elapsed_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
