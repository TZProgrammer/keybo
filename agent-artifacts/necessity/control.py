"""POSITIVE CONTROL — does my from-scratch reimplementation reproduce the PUBLISHED numbers?

Run BEFORE any new design, because every claim I go on to make is a difference against
poolsweep's cells. If my pipeline is off by any amount, a "difference" I report is my own
reimplementation drift wearing a finding's clothes. Two anchors, both published:

  archive x400 (seed 0, blend-v1, .native):  rho = +0.2184, u_A = 0.0399, u_B = 0.1605
  random wide x400 (same):                   rho = +0.7970, u_A = 0.9693, u_B = 0.9711
  within-COMMUNITY seed-pair mean:           archive +0.9647, random +0.9872

Also re-derives the ALGEBRAIC IDENTITY the parent asserts, on MY pools, to its own claimed
machine precision — the identity is load-bearing for reading slack as asymmetry.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import keybo.analysis.surfaces as S

import nplib as N

TOL_RHO = 5e-4  # published to 4 decimals, so agreement must be at that resolution


def main() -> int:
    t0 = time.time()
    sA, sB = N.load_surface(N.A_NAME), N.load_surface(N.B_NAME)
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    objective = S.trigram_objective(S.default_trigram_path(None))
    print(f"frame=native (asserted) surfaces: A={sA.sha256[:12]} B={sB.sha256[:12]}", flush=True)

    archive = N.load_archive()
    print(f"archive bank: {len(archive)} layouts (published: 2860)", flush=True)
    rand_bank = N.random_bank(200_000, 0)
    yA_r = N.ms_of(rand_bank, sA, objective)
    yB_r = N.ms_of(rand_bank, sB, objective)
    ref = {
        "mean_A": float(yA_r.mean()), "sd_A": float(yA_r.std(ddof=1)),
        "mean_B": float(yB_r.mean()), "sd_B": float(yB_r.std(ddof=1)),
    }
    print(f"reference bank: mean_A={ref['mean_A']:.6f} sd_A={ref['sd_A']:.6f} "
          f"mean_B={ref['mean_B']:.6f} sd_B={ref['sd_B']:.6f}", flush=True)
    print("  published:    mean_A=277.294976 sd_A=4.329948 mean_B=293.623124 sd_B=7.458071", flush=True)

    # Draw order must match final.py EXACTLY: one rng(seed), random-wide drawn first,
    # archive-x400 second, from the SAME generator. A fresh rng per pool would give a
    # different archive draw and a legitimately different rho — that is the trap here.
    rng = np.random.default_rng(0)
    ref_pool = [rand_bank[i] for i in rng.choice(len(rand_bank), 400, replace=False)]
    arc_pool = [archive[i] for i in rng.choice(len(archive), 400, replace=False)]

    checks = []
    published = {
        "random-wide": {"rho": 0.7970054187838673, "u_A": 0.9693, "u_B": 0.9711},
        "archive-x400": {"rho": 0.21842724017025103, "u_A": 0.0399, "u_B": 0.1605},
    }
    for label, pool in (("random-wide", ref_pool), ("archive-x400", arc_pool)):
        p = N.profile(pool, sA, sB, objective, ref)
        exp = published[label]
        d_rho = abs(p["rho_spearman"] - exp["rho"])
        ok = d_rho < TOL_RHO and abs(p["u_A"] - exp["u_A"]) < 5e-4 and abs(p["u_B"] - exp["u_B"]) < 5e-4
        checks.append({"cell": label, "measured": p, "published": exp, "abs_rho_delta": d_rho, "pass": bool(ok)})
        print(f"  {label:<14} rho={p['rho_spearman']:+.7f} (pub {exp['rho']:+.7f}, delta {d_rho:.2e}) "
              f"u_A={p['u_A']:.4f} (pub {exp['u_A']}) u_B={p['u_B']:.4f} (pub {exp['u_B']}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    # ---- within-instrument channel -------------------------------------------------------
    per_seed = N.community_per_seed()
    print(f"\nCOMMUNITY per-seed parts: {sorted(per_seed)}; AALTO per-seed: "
          f"{sorted(p.name for p in __import__('pathlib').Path(N.SURFACE_DIR).glob('AALTO*seed*.npy')) or 'NONE'}",
          flush=True)
    within_pub = {"random-wide": 0.9872, "archive-x400": 0.9647}
    for label, pool in (("random-wide", ref_pool), ("archive-x400", arc_pool)):
        w = N.within_instrument(pool, per_seed, objective)
        d = abs(w["mean"] - within_pub[label])
        ok = d < 5e-4
        checks.append({"cell": f"within:{label}", "measured": w, "published": within_pub[label],
                       "abs_rho_delta": d, "pass": bool(ok)})
        print(f"  within {label:<14} mean={w['mean']:+.7f} (pub {within_pub[label]:+.4f}, delta {d:.2e}) "
              f"{ {k: round(v, 4) for k, v in w['pairs'].items()} } {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- the algebraic identity, on MY pools --------------------------------------------
    print("\n[IDENTITY] cov(zA,zB) == var(C) - var(D), and r == (k^2-1)/(k^2+1) iff sd_A == sd_B",
          flush=True)
    ident = []
    for label, pool in (("random-wide", ref_pool), ("archive-x400", arc_pool)):
        p = N.profile(pool, sA, sB, objective, ref)
        e_ident = abs(p["cov_zA_zB"] - p["var_C_minus_var_D"])
        slack = p["rho_pearson"] - p["rho_algebraic_equalvar"]
        ident.append({"cell": label, "identity_abs_error": e_ident, "pearson_slack": slack,
                      "u_ratio": p["u_ratio"], "k": p["k_c_over_d"]})
        print(f"  {label:<14} |cov - (varC-varD)| = {e_ident:.4e}  "
              f"slack(Pearson - closed form) = {slack:+.5f}  u_A/u_B = {p['u_ratio']:.4f}  k={p['k_c_over_d']:.4f}",
              flush=True)

    n_fail = sum(1 for c in checks if not c["pass"])
    payload = {
        "schema_version": 1,
        "note": (
            "POSITIVE CONTROL for the necessity probe's from-scratch pipeline. Reproduces "
            "poolsweep-1's published archive/random cells and within-COMMUNITY seed reliability "
            "from main @ e6a5b9e with NOTHING cherry-picked (surfaces.py is bit-identical between "
            "main and the poolsweep branch). If this fails, no difference measured downstream can "
            "be attributed to a pool design."
        ),
        "frame": "native",
        "surface_sha256": {N.A_NAME: sA.sha256, N.B_NAME: sB.sha256},
        "archive_bank_size": len(archive),
        "random_bank_size": len(rand_bank),
        "reference": ref,
        "checks": checks,
        "identity": ident,
        "n_fail": n_fail,
        "elapsed_seconds": time.time() - t0,
    }
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/necesspool-out/control.json"
    __import__("pathlib").Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as h:
        json.dump(payload, h, indent=2)
    print(f"\nwrote {out} — {len(checks) - n_fail}/{len(checks)} checks PASS "
          f"({payload['elapsed_seconds']:.1f}s)", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
