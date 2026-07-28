"""REFLECTION SELF-AUDIT — attack this round's own load-bearing claims.

Five probes, each aimed at a claim I made rather than at confirming it:

A1  Is C/D ALGEBRAICALLY tied to rho? If rho can be written in terms of var(C) and var(D) by
    construction, the "Spearman +0.9991" is an identity wearing a finding's clothes, and the
    empirical content is only WHICH pools sit WHERE on that curve. Derive the closed form and
    measure the gap between it and the observed rho.

A2  Why does the RANDOM lineage sit off the parameter-free Thorndike curve by -0.091? A
    parameter-free curve that the reference group misses is misspecified, and a misspecified
    curve cannot test a group difference. Probe: does the residual track the ASYMMETRY
    |log(u_A/u_B)| (which Thorndike case-2 cannot represent, since it takes ONE u)?

A3  Is kswap1 still near-optimal? The k-swap ladder is the cleanest result only if one
    transposition keeps the layout on/near the frontier. Measure its ms/char and full gauge
    profile against the archive, and against the random pool as the far reference.

A4  Is there ANY third quasi-independent target? COMMUNITY ships per-seed parts. A seed-vs-seed
    pair is NOT an independent instrument (same data, same source) but it bounds how much of the
    cross-source collapse is FIT NOISE rather than instrument disagreement — which is the number
    that decides whether the two-source limit is structural.

A5  The paired difference test I actually reported was run on the JOINTBAND cell (u_A 0.2205),
    not on the properly matched BOXMATCH cell (u_A 0.0427). Run it on the right cell.

MODELLED ONLY. `.native` frame, asserted. No new claims are registered from here — output is a
digest for the reflection callback.
"""

from __future__ import annotations

import json
import time

import numpy as np

import keybo.analysis.evidence_scorer as E
import keybo.analysis.evidence_validation as V
import keybo.analysis.surfaces as S

import banks
from final import box_match, monotone_curve_predict

A_NAME, B_NAME = "AALTO_BASE", "COMMUNITY_BASE"
OUT = {}


def spearman(a, b):
    return V._spearman(a, b)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    t0 = time.time()
    sA = E.load_target_surface(A_NAME, banks.SURFACE_DIR, "native")
    sB = E.load_target_surface(B_NAME, banks.SURFACE_DIR, "native")
    assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
    ctx = E.gauge_context(None)
    objective = S.trigram_objective(S.default_trigram_path(None))
    print(f"frame=native (asserted) corpus={ctx.corpus_name}", flush=True)

    def ms(pool):
        return banks.ms_of(pool, sA, objective), banks.ms_of(pool, sB, objective)

    archive = banks.load_archive()
    rand_bank = banks.random_bank(200_000, 0)
    yA_r, yB_r = ms(rand_bank)
    ref = {
        "mean_A": float(yA_r.mean()), "sd_A": float(yA_r.std(ddof=1)),
        "mean_B": float(yB_r.mean()), "sd_B": float(yB_r.std(ddof=1)),
    }
    zA_r = (yA_r - ref["mean_A"]) / ref["sd_A"]
    zB_r = (yB_r - ref["mean_B"]) / ref["sd_B"]
    C_r, D_r = (zA_r + zB_r) / 2.0, (zA_r - zB_r) / 2.0
    rng = np.random.default_rng(0)
    ref_pool = [rand_bank[i] for i in rng.choice(len(rand_bank), 400, replace=False)]
    arc_pool = [archive[i] for i in rng.choice(len(archive), 400, replace=False)]
    yA_a_all, yB_a_all = ms(archive)

    def prof(pool):
        yA, yB = ms(pool)
        zA = (yA - ref["mean_A"]) / ref["sd_A"]
        zB = (yB - ref["mean_B"]) / ref["sd_B"]
        c, d = (zA + zB) / 2.0, (zA - zB) / 2.0
        vc, vd = float(c.var(ddof=1)), float(d.var(ddof=1))
        k = float(c.std(ddof=1) / d.std(ddof=1))
        return {
            "rho_spearman": spearman(yA, yB), "rho_pearson": pearson(yA, yB),
            "var_C": vc, "var_D": vd, "k_c_over_d": k,
            # THE CLOSED FORM under equal restricted variances:
            #   cov(zA,zB) = var(C) - var(D)  and  var(zA)+var(zB) = 2(var(C)+var(D))
            #   => r = (var C - var D) / (sd_A sd_B) >= (k^2-1)/(k^2+1), equality iff sd_A=sd_B
            "rho_algebraic_equalvar": (k * k - 1.0) / (k * k + 1.0),
            "cov_zA_zB": float(np.cov(zA, zB, ddof=1)[0, 1]),
            "var_C_minus_var_D": vc - vd,
            "u_A": float(zA.std(ddof=1)), "u_B": float(zB.std(ddof=1)),
            "log_u_asymmetry": float(abs(np.log(zA.std(ddof=1) / zB.std(ddof=1)))),
            "mean_A": float(yA.mean()), "mean_B": float(yB.mean()),
        }

    # ---------------- A1: is C/D algebra? ----------------
    print("\n[A1] is rho ALGEBRAICALLY determined by var(C), var(D)?", flush=True)
    # (i) the exact identity: cov(zA,zB) must EQUAL var(C) - var(D) for every pool.
    ident = []
    cells = {"random-wide": ref_pool, "archive": arc_pool}
    _arc_prof = prof(arc_pool)
    boxm, boxmeta = box_match(
        rand_bank, C_r, D_r,
        target_c=float(np.sqrt(_arc_prof["var_C"])),
        target_d=float(np.sqrt(_arc_prof["var_D"])),
        n=400, seed=0,
    )
    cells["boxmatch"] = boxm
    for k in (1, 3, 8, 20):
        cells[f"kswap{k}"] = banks.kswap_bank(archive, k, 400, 0)
    for frac in (0.02, 0.1, 0.4):
        cells[f"restrictC-{frac:g}"], _ = banks.axis_band_pool(rand_bank, C_r, target_sd_frac=frac, n=400, seed=0)
        cells[f"restrictD-{frac:g}"], _ = banks.axis_band_pool(rand_bank, D_r, target_sd_frac=frac, n=400, seed=0)
    profiles = {name: prof(pool) for name, pool in cells.items()}
    for name, p in profiles.items():
        ident.append({
            "label": name,
            "cov_zA_zB": p["cov_zA_zB"],
            "var_C_minus_var_D": p["var_C_minus_var_D"],
            "identity_abs_error": abs(p["cov_zA_zB"] - p["var_C_minus_var_D"]),
            "rho_pearson": p["rho_pearson"],
            "rho_algebraic_equalvar": p["rho_algebraic_equalvar"],
            "pearson_minus_algebraic": p["rho_pearson"] - p["rho_algebraic_equalvar"],
            "rho_spearman": p["rho_spearman"],
            "spearman_minus_algebraic": p["rho_spearman"] - p["rho_algebraic_equalvar"],
            "k_c_over_d": p["k_c_over_d"],
            "u_ratio": p["u_A"] / p["u_B"],
        })
    max_ident = max(r["identity_abs_error"] for r in ident)
    print(f"  EXACT identity cov(zA,zB) == var(C) - var(D): max abs error over {len(ident)} pools = {max_ident:.3e}", flush=True)
    print(f"  {'pool':<16}{'k=C/D':>9}{'algebraic':>11}{'pearson':>10}{'spearman':>10}{'P-alg':>9}{'S-alg':>9}{'u_A/u_B':>9}", flush=True)
    for r in sorted(ident, key=lambda r: r["k_c_over_d"]):
        print(
            f"  {r['label']:<16}{r['k_c_over_d']:>9.3f}{r['rho_algebraic_equalvar']:>11.4f}"
            f"{r['rho_pearson']:>10.4f}{r['rho_spearman']:>10.4f}"
            f"{r['pearson_minus_algebraic']:>+9.4f}{r['spearman_minus_algebraic']:>+9.4f}{r['u_ratio']:>9.3f}",
            flush=True,
        )
    OUT["A1_algebra"] = {
        "note": (
            "cov(zA,zB) = var(C) - var(D) is an EXACT ALGEBRAIC IDENTITY for any pool (C and D are "
            "the sum/difference halves). Since var(zA)+var(zB) = 2(var C + var D), Pearson r = "
            "(var C - var D)/(sd_A sd_B) >= (k^2-1)/(k^2+1) with k = sd(C)/sd(D), and equality holds "
            "iff sd_A = sd_B. So the monotone rho-vs-C/D curve is ALGEBRA, not an empirical law; the "
            "empirical content is WHICH pools land at WHICH k, plus the sd_A != sd_B slack and the "
            "Pearson->Spearman gap."
        ),
        "identity_max_abs_error": max_ident,
        "rows": ident,
    }

    # ---------------- A2: why is the random lineage off the Thorndike curve? ----------------
    print("\n[A2] the Thorndike single-u curve — is the random offset a MISSPECIFICATION?", flush=True)
    adv = json.load(open("/tmp/poolsweep/agent-artifacts/poolsweep/out/adversarial-blend-seed0.json"))
    p4 = adv["P4_one_curve"]
    rows = []
    for grp, key in (("random", "random_lineage_cells"), ("optimized", "optimized_lineage_cells")):
        for c in p4[key]:
            uA, uB = c.get("u_A"), c.get("u_B")
            asym = abs(np.log(uA / uB)) if (uA and uB) else None
            rows.append({"group": grp, "label": c["label"], "u": c["u"], "residual": c["residual"],
                         "u_A": uA, "u_B": uB, "log_u_asymmetry": asym})
        # the P3-grid random cells carry u_A/u_B; the single "random-wide" entry does not
    have = [r for r in rows if r["log_u_asymmetry"] is not None]
    from scipy.stats import spearmanr
    if len(have) >= 6:
        s_all = spearmanr([r["log_u_asymmetry"] for r in have], [r["residual"] for r in have])
        print(f"  spearman(residual, |log u_A/u_B|) over {len(have)} cells WITH both u = {s_all.statistic:+.4f} (p={s_all.pvalue:.4g})", flush=True)
    ru = [r for r in rows if r["group"] == "random"]
    ou = [r for r in rows if r["group"] == "optimized"]
    ra = [r["log_u_asymmetry"] for r in ru if r["log_u_asymmetry"] is not None]
    oa = [r["log_u_asymmetry"] for r in ou if r["log_u_asymmetry"] is not None]
    print(f"  |log u_A/u_B|: random mean {np.mean(ra):.4f} (n={len(ra)}) | optimized mean {np.mean(oa):.4f} (n={len(oa)})", flush=True)
    print(f"  residual mean: random {np.mean([r['residual'] for r in ru]):+.4f} | optimized {np.mean([r['residual'] for r in ou]):+.4f}", flush=True)
    OUT["A2_thorndike_misspecification"] = {
        "note": (
            "Thorndike case-2 takes ONE restriction ratio u, i.e. it models selection on a SINGLE "
            "variable. Every cell here is restricted in BOTH sources by different amounts, and I fed "
            "the curve u = sqrt(u_A u_B). If the residual tracks the ASYMMETRY |log(u_A/u_B)| — a "
            "quantity the single-u formula cannot represent — then the curve is misspecified and the "
            "nonzero RANDOM-lineage residual is the proof. A misspecified curve whose reference group "
            "misses it by -0.091 cannot be used to test a group difference, because the two groups "
            "occupy different regions of (u_A,u_B) space."
        ),
        "spearman_residual_vs_asymmetry": float(s_all.statistic) if len(have) >= 6 else None,
        "spearman_p": float(s_all.pvalue) if len(have) >= 6 else None,
        "random_mean_asymmetry": float(np.mean(ra)) if ra else None,
        "optimized_mean_asymmetry": float(np.mean(oa)) if oa else None,
        "random_mean_residual": float(np.mean([r["residual"] for r in ru])),
        "optimized_mean_residual": float(np.mean([r["residual"] for r in ou])),
        "rows": rows,
    }

    # ---------------- A3: is kswap1 still near-optimal? ----------------
    print("\n[A3] is kswap1 still near-optimal? gauge profile vs archive and random", flush=True)
    from gaugecache import GaugeCache
    cache = GaugeCache(None, "/tmp/poolsweep-cache", workers=96)
    k1 = cells["kswap1"]
    Xa, X1, Xr = cache.matrix(arc_pool), cache.matrix(k1), cache.matrix(ref_pool)
    yA_1, yB_1 = ms(k1)
    yA_ar, yB_ar = ms(arc_pool)
    yA_rr, yB_rr = ms(ref_pool)
    print(f"  ms/trigram  AALTO: archive {yA_ar.mean():.4f}  kswap1 {yA_1.mean():.4f}  random {yA_rr.mean():.4f}", flush=True)
    print(f"  ms/trigram  COMMUNITY: archive {yB_ar.mean():.4f}  kswap1 {yB_1.mean():.4f}  random {yB_rr.mean():.4f}", flush=True)
    # position of kswap1 on the archive->random axis, per source, as a fraction
    frac_A = (yA_1.mean() - yA_ar.mean()) / (yA_rr.mean() - yA_ar.mean())
    frac_B = (yB_1.mean() - yB_ar.mean()) / (yB_rr.mean() - yB_ar.mean())
    print(f"  kswap1 sits {frac_A*100:.1f}% (AALTO) / {frac_B*100:.1f}% (COMMUNITY) of the way from archive to random", flush=True)
    # how many kswap1 layouts are inside the archive's own per-gauge 1-99 pct band on ALL gauges
    lo, hi = np.percentile(Xa, 1, axis=0), np.percentile(Xa, 99, axis=0)
    inside_all = int(np.all((X1 >= lo) & (X1 <= hi), axis=1).sum())
    inside_all_rand = int(np.all((Xr >= lo) & (Xr <= hi), axis=1).sum())
    inside_all_arc = int(np.all((Xa >= lo) & (Xa <= hi), axis=1).sum())
    n_in_band_per_layout = np.sum((X1 >= lo) & (X1 <= hi), axis=1)
    print(f"  layouts inside the archive's 1-99pct band on ALL 14 gauges: archive {inside_all_arc}/400, kswap1 {inside_all}/400, random {inside_all_rand}/400", flush=True)
    print(f"  kswap1 gauges-in-band per layout: mean {n_in_band_per_layout.mean():.2f}/14 (min {n_in_band_per_layout.min()}, max {n_in_band_per_layout.max()})", flush=True)
    gauge_rows = []
    print(f"  {'gauge':<12}{'archive':>12}{'kswap1':>12}{'random':>12}{'k1 in z(arc)':>14}", flush=True)
    for i, g in enumerate(E.LIVE_GAUGES):
        ma, m1, mr = Xa[:, i].mean(), X1[:, i].mean(), Xr[:, i].mean()
        sa = Xa[:, i].std(ddof=1)
        z = (m1 - ma) / sa if sa else float("nan")
        span = (mr - ma)
        gauge_rows.append({
            "gauge": g, "archive_mean": float(ma), "kswap1_mean": float(m1), "random_mean": float(mr),
            "kswap1_z_in_archive_sd": float(z),
            "kswap1_frac_of_archive_to_random": float((m1 - ma) / span) if span else None,
        })
        print(f"  {g:<12}{ma:>12.4f}{m1:>12.4f}{mr:>12.4f}{z:>14.2f}", flush=True)
    OUT["A3_kswap1_optimality"] = {
        "note": (
            "One transposition is a LARGE move in gauge space even though it is a small move in "
            "ms/trigram. Reported both ways: the ms/char position on the archive->random axis, and "
            "the fraction of kswap1 layouts still inside the archive's own 1st-99th percentile band "
            "on ALL 14 gauges. The second is the honest test of 'still near-optimal'."
        ),
        "ms_per_trigram": {
            "archive_A": float(yA_ar.mean()), "kswap1_A": float(yA_1.mean()), "random_A": float(yA_rr.mean()),
            "archive_B": float(yB_ar.mean()), "kswap1_B": float(yB_1.mean()), "random_B": float(yB_rr.mean()),
        },
        "kswap1_frac_archive_to_random_A": float(frac_A),
        "kswap1_frac_archive_to_random_B": float(frac_B),
        "inside_archive_band_all_gauges": {
            "archive": inside_all_arc, "kswap1": inside_all, "random": inside_all_rand, "n": 400,
        },
        "kswap1_gauges_in_band_mean": float(n_in_band_per_layout.mean()),
        "gauges": gauge_rows,
    }

    # ---------------- A4: any third quasi-independent target? ----------------
    print("\n[A4] is there a third quasi-independent target? (per-seed COMMUNITY parts)", flush=True)
    from pathlib import Path
    sd = Path(banks.SURFACE_DIR)
    per_seed = {}
    for seed in (0, 1, 2):
        bg, cd = sd / f"COMMUNITY_BASE.bigram.seed{seed}.npy", sd / f"COMMUNITY_BASE.conditional.seed{seed}.npy"
        if bg.is_file() and cd.is_file():
            per_seed[seed] = np.load(bg)[:, :, None] + np.load(cd)
    print(f"  COMMUNITY per-seed surfaces available: {sorted(per_seed)}", flush=True)
    aalto_per_seed = sorted(p.name for p in sd.glob("AALTO*seed*.npy"))
    print(f"  AALTO per-seed parts: {aalto_per_seed or 'NONE'}", flush=True)
    mass = float(objective[3].sum())

    def seed_ms(pool, arr):
        return np.array([S.score_fit(l, arr, objective) / mass for l in pool])

    within = []
    for label, pool in (("random-wide", ref_pool), ("archive", arc_pool), ("boxmatch", boxm)):
        ys = {s: seed_ms(pool, a) for s, a in per_seed.items()}
        pairs = {f"seed{i}|seed{j}": spearman(ys[i], ys[j])
                 for i in sorted(ys) for j in sorted(ys) if i < j}
        cross = spearman(banks.ms_of(pool, sA, objective), banks.ms_of(pool, sB, objective))
        within.append({
            "pool": label, "within_community_seed_pairs": pairs,
            "within_mean": float(np.mean(list(pairs.values()))),
            "cross_source_rho": cross,
        })
        print(f"  {label:<12} WITHIN-COMMUNITY seed-pair rho mean {np.mean(list(pairs.values())):+.4f} "
              f"{ {k: round(v,4) for k,v in pairs.items()} }  vs CROSS-source {cross:+.4f}", flush=True)
    OUT["A4_third_source"] = {
        "note": (
            "COMMUNITY ships 3 per-seed parts; AALTO ships NONE. A seed-vs-seed pair is NOT an "
            "independent instrument (same participants, same source, and per THEORY-1 the bigram "
            "table is shared across fit methods within a source) — but it bounds how much of the "
            "cross-source collapse is FIT NOISE versus genuine instrument disagreement. If the "
            "within-source seed agreement stays HIGH on the archive while the cross-source rho "
            "collapses, the collapse is instrument disagreement, not fit noise. Either way there is "
            "no second INDEPENDENT PAIR, so the two-source limit is STRUCTURAL for these artifacts."
        ),
        "community_per_seed_available": sorted(per_seed),
        "aalto_per_seed_available": aalto_per_seed,
        "rows": within,
    }

    # ---------------- A5: the paired difference on the RIGHT cell ----------------
    print("\n[A5] paired difference archive vs the PROPERLY MATCHED boxmatch cell", flush=True)

    def two_sample_diff(yA1, yB1, yA2, yB2, boot=8000, seed=0):
        r = np.random.default_rng(seed)
        obs = spearman(yA1, yB1) - spearman(yA2, yB2)
        n1, n2 = len(yA1), len(yA2)
        d = []
        for _ in range(boot):
            i1, i2 = r.integers(0, n1, n1), r.integers(0, n2, n2)
            a, b = spearman(yA1[i1], yB1[i1]), spearman(yA2[i2], yB2[i2])
            if np.isfinite(a) and np.isfinite(b):
                d.append(a - b)
        arr = np.array(d)
        return {
            "delta_rho": float(obs),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "p_two_sided": float(min((arr > 0).mean(), (arr < 0).mean()) * 2),
            "n_boot": len(arr),
        }

    yA_b, yB_b = ms(boxm)
    d_box = two_sample_diff(yA_ar, yB_ar, yA_b, yB_b)
    print(f"  archive {spearman(yA_ar,yB_ar):+.4f} - boxmatch {spearman(yA_b,yB_b):+.4f} = {d_box['delta_rho']:+.4f} "
          f"CI [{d_box['ci95'][0]:+.4f},{d_box['ci95'][1]:+.4f}] p={d_box['p_two_sided']:.4f}", flush=True)
    print(f"  boxmatch achieved: u_A={prof(boxm)['u_A']:.4f} u_B={prof(boxm)['u_B']:.4f} k={prof(boxm)['k_c_over_d']:.4f} "
          f"| archive u_A={prof(arc_pool)['u_A']:.4f} u_B={prof(arc_pool)['u_B']:.4f} k={prof(arc_pool)['k_c_over_d']:.4f}", flush=True)
    OUT["A5_paired_difference_right_cell"] = {
        "note": (
            "The difference test I REPORTED (-0.0130, p=0.8715) was run on the JOINTBAND cell, whose "
            "u_A was 0.2205 — five times the archive's 0.0421, i.e. NOT well matched. The BOXMATCH "
            "cell is the properly matched one and is what the headline quotes, so the difference test "
            "belongs on it. This is the number that decides whether NECESSITY is established."
        ),
        "archive_vs_boxmatch": d_box,
        "boxmatch_profile": prof(boxm), "archive_profile": prof(arc_pool),
        "box_meta": boxmeta,
    }

    OUT["meta"] = {
        "corpus": ctx.corpus_name,
        "corpus_sha256": dict(ctx.identity.get("sha256", {})),
        "frame": "native",
        "surface_sha256": {A_NAME: sA.sha256, B_NAME: sB.sha256},
        "reference": ref,
        "elapsed_seconds": time.time() - t0,
    }
    path = "/tmp/poolsweep/agent-artifacts/poolsweep/out/audit-blend-seed0.json"
    with open(path, "w") as h:
        json.dump(OUT, h, indent=2)
    print(f"\nwrote {path} ({OUT['meta']['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
