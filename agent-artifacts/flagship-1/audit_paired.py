"""FLAGSHIP-1 self-audit — try to REFUTE my own "the floor is the wrong ruler" claim.

The claim: 78-83% of the variance the floor measures is a seed MAIN EFFECT, which is common
mode and cancels in a within-seed difference, so the unpaired floor over-rejects and the
paired resolution (~4x tighter) is the right ruler for "is A faster than B".

Four ways it could be wrong. Each is implemented as a test that CAN fail.

A. **THE COMMON-MODE FRACTION MIGHT BE AN ARTIFACT OF THE DECOMPOSITION ITSELF.** With 6
   layouts x 3 seeds and layouts that differ by ~0.3 ms/char while seeds differ by ~1-2
   ms/char, a two-way ANOVA will *mechanically* attribute most SS to seed. The test that
   bites: a PLACEBO where the layout effect is destroyed (shuffle layout labels within each
   seed). If the placebo's %seed is similar, the decomposition is measuring the design, not
   a property of the estimator — and my reading is weak. If the placebo's %layout collapses
   to ~0 while %seed rises, the real matrix's 13-20% layout share is a genuine signal.

B. **"COMMON MODE" MIGHT NOT BE MULTIPLICATIVE-FREE.** If seed s scales every layout's time
   by a factor rather than shifting it, then a DIFFERENCE does not cancel — it scales too,
   and the paired resolution understates. Test: regress each seed's layout vector on the
   seed-mean vector; if slopes are ~1, shifts (additive, cancels). If slopes differ from 1,
   the difference scales and I must report the scaled worst case.

C. **THE PAIRED RESOLUTION MIGHT BE UNDER-ESTIMATED BY USING max|resid| ON n=3.** With 3
   seeds the max residual is a low-precision statistic. Test: report the FULL distribution of
   per-pair per-seed deltas and compute, per pair, the worst-case (min |delta| over seeds)
   rather than the mean — a strictly more conservative statement — and re-count how many
   pairs survive THAT.

D. **SIGN UNANIMITY MIGHT BE VACUOUS.** If seeds are near-identical up to a shift, every pair
   is trivially sign-unanimous and the test carries no information. Test: a null where the
   layout signal is removed (as in A) — count how often 15/15 pairs come out sign-unanimous
   under that null. If unanimity is common under the null it is weak evidence, and I must say
   so rather than cite 12/15.

Nothing here is about realized typing speed. All four tests bound INSTRUMENT behaviour.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

CAND = [
    "keybo-lsb",
    "keybo-lsb+lm",
    "lsb-sib",
    "archive-1843",
    "archive-1846",
    "flagship-c3",
]
CORPORA = ["iweb", "blend-v1", "blend-v1-no-anchor"]


def decompose(M: np.ndarray) -> dict:
    grand = float(M.mean())
    le = M.mean(axis=1) - grand
    se = M.mean(axis=0) - grand
    resid = M - grand - le[:, None] - se[None, :]
    ssl = float((M.shape[1] * le**2).sum())
    sss = float((M.shape[0] * se**2).sum())
    ssr = float((resid**2).sum())
    tot = ssl + sss + ssr
    return {
        "pct_layout": 100 * ssl / tot,
        "pct_seed": 100 * sss / tot,
        "pct_interaction": 100 * ssr / tot,
        "max_abs_resid": float(np.abs(resid).max()),
    }


def main() -> None:
    src = json.loads(Path("/tmp/flagship-work/paired-resolution.json").read_text())
    rng = np.random.default_rng(31415)
    out: dict = {
        "what": "adversarial self-audit of FLAGSHIP-1's paired-resolution reinterpretation",
        "corpora": {},
    }

    for corpus in CORPORA:
        c = src["corpora"][corpus]
        M = np.array([c["rows"][n]["per_seed_ms_per_char"] for n in CAND])  # 6 x 3
        real = decompose(M)

        # ---- A: label-shuffle placebo (destroys the layout effect, keeps the seed effect) --
        B = 2000
        pl, ps, pi = [], [], []
        unanimous_null = []
        for _ in range(B):
            Ms = np.array([M[rng.permutation(len(CAND)), s] for s in range(3)]).T
            d = decompose(Ms)
            pl.append(d["pct_layout"])
            ps.append(d["pct_seed"])
            pi.append(d["pct_interaction"])
            n_un = sum(
                bool(np.all((Ms[i] - Ms[j]) > 0) or np.all((Ms[i] - Ms[j]) < 0))
                for i, j in itertools.combinations(range(len(CAND)), 2)
            )
            unanimous_null.append(n_un)
        placebo = {
            "B": B,
            "pct_layout_mean": float(np.mean(pl)),
            "pct_layout_p95": float(np.percentile(pl, 95)),
            "pct_seed_mean": float(np.mean(ps)),
            "pct_interaction_mean": float(np.mean(pi)),
            "real_pct_layout": real["pct_layout"],
            "real_pct_layout_exceeds_null_p95": bool(real["pct_layout"] > np.percentile(pl, 95)),
            "p_value_layout_share": float((np.array(pl) >= real["pct_layout"]).mean()),
        }

        # ---- D: is sign-unanimity vacuous under that null? --------------------------------
        real_un = c["n_pairs_sign_unanimous"]
        unanimity_null = {
            "null_mean_n_unanimous_of_15": float(np.mean(unanimous_null)),
            "null_p95": float(np.percentile(unanimous_null, 95)),
            "real_n_unanimous_of_15": real_un,
            "p_value": float((np.array(unanimous_null) >= real_un).mean()),
            "reading": (
                "if the null routinely produces >= the observed unanimity count, the "
                "unanimity statistic is vacuous and must not be cited as evidence"
            ),
        }

        # ---- B: additive shift or multiplicative scale? -----------------------------------
        mean_vec = M.mean(axis=1)
        slopes = {}
        for s in range(3):
            A = np.vstack([mean_vec - mean_vec.mean(), np.ones(len(CAND))]).T
            coef, *_ = np.linalg.lstsq(A, M[:, s] - M[:, s].mean(), rcond=None)
            slopes[f"seed{s}"] = float(coef[0])
        shift_vs_scale = {
            "slopes_of_seed_on_seedmean": slopes,
            "max_abs_slope_minus_1": float(max(abs(v - 1) for v in slopes.values())),
            "reading": (
                "slope ~1 => the seed acts as an additive SHIFT (cancels in a difference); "
                "slope far from 1 => it SCALES differences and the paired ruler must be "
                "inflated by that factor"
            ),
        }
        worst_scale = max(slopes.values())

        # ---- C: strictly-conservative per-pair statement (worst seed, not the mean) -------
        pairs = {}
        for i, j in itertools.combinations(range(len(CAND)), 2):
            d = M[i] - M[j]
            pairs[f"{CAND[i]}|{CAND[j]}"] = {
                "mean_delta": float(d.mean()),
                "worst_seed_abs_delta": float(np.abs(d).min()),
                "sign_unanimous": bool(np.all(d > 0) or np.all(d < 0)),
                # conservative: the smallest |delta| any seed gives must still beat the
                # paired resolution, AND the sign must be unanimous
                "survives_conservative": bool(
                    (np.all(d > 0) or np.all(d < 0))
                    and np.abs(d).min() > c["resolution_paired"] * worst_scale
                ),
            }
        n_cons = sum(p["survives_conservative"] for p in pairs.values())

        out["corpora"][corpus] = {
            "real_decomposition": real,
            "A_label_shuffle_placebo": placebo,
            "B_shift_vs_scale": shift_vs_scale,
            "C_conservative_pairs": {
                "resolution_paired": c["resolution_paired"],
                "inflated_by_worst_slope": c["resolution_paired"] * worst_scale,
                "n_survive_conservative_of_15": n_cons,
                "survivors": [k for k, v in pairs.items() if v["survives_conservative"]],
                "pairs": pairs,
            },
            "D_unanimity_null": unanimity_null,
        }

    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"wrote {sys.argv[1]}\n")
    for corpus in CORPORA:
        o = out["corpora"][corpus]
        a, b, cc, d = (
            o["A_label_shuffle_placebo"],
            o["B_shift_vs_scale"],
            o["C_conservative_pairs"],
            o["D_unanimity_null"],
        )
        print(f"=== {corpus} ===")
        print(
            f"  A placebo: real %layout {a['real_pct_layout']:.2f} vs null mean "
            f"{a['pct_layout_mean']:.2f} p95 {a['pct_layout_p95']:.2f} -> "
            f"exceeds_p95={a['real_pct_layout_exceeds_null_p95']} p={a['p_value_layout_share']:.4f}"
        )
        print(
            f"  B shift/scale: slopes {['%.4f' % v for v in b['slopes_of_seed_on_seedmean'].values()]} "
            f"max|slope-1| = {b['max_abs_slope_minus_1']:.4f}"
        )
        print(
            f"  C conservative: paired res {cc['resolution_paired']:.4f} -> inflated "
            f"{cc['inflated_by_worst_slope']:.4f}; {cc['n_survive_conservative_of_15']}/15 survive"
        )
        for s in cc["survivors"]:
            print(f"       + {s}")
        print(
            f"  D unanimity null: real {d['real_n_unanimous_of_15']}/15 vs null mean "
            f"{d['null_mean_n_unanimous_of_15']:.2f} p95 {d['null_p95']:.1f} p={d['p_value']:.4f}"
        )
        print()


if __name__ == "__main__":
    main()
