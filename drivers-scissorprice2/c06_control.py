"""P1 POSITIVE CONTROL + the mandated LITERAL sign-blind falsifier, for `scissor`.

P1: run MY OWN frontier machinery with gauge = **sfb** at PRICEBAND-1's INERT caps and check I
recover its published unconstrained level **253.9006 (= arm-B)**. This validates the entire
pipeline against ANOTHER AGENT's published number on the gauge it priced -- the strongest
control available, and it is a genuinely independent code path from its analysis (my fastgauge
is a fresh implementation, verified against the shipped `KmStats` at 0.00e+00).

Sign-blind: the literal signed-vs-sign-blind nested-R2 test is only defined for a PERTURBATION
design, so I run it on a single-swap `scissor` sweep (in-band boards + qwerty30m). If the
sign-blind regressor explains as much, the perturbation design measures DISRUPTION not price and
per the standing rule I do NOT report a perturbation-based scissor price.
"""
import _env  # noqa: F401
import json

import boards
import fastgauge
import numpy as np
import search
from _env import ART


class Obj(search.Objective):
    def scissor(self, p):
        return self.fg.scissor_only(p[:30])


def nested_r2(dg, dms, boards_ix, n_boards):
    """R2 of dms on [board fixed effects + intercept] plus signed dg / |dg| / both.

    Board fixed effects absorb every between-board difference, so this is the within-board
    test the retraction rule specifies."""
    n = len(dms)
    B = np.zeros((n, n_boards))
    B[np.arange(n), boards_ix] = 1.0
    base = np.column_stack([np.ones(n), B[:, 1:]])  # intercept + K-1 dummies

    def fit(X):
        beta, *_ = np.linalg.lstsq(X, dms, rcond=None)
        resid = dms - X @ beta
        ss_tot = ((dms - dms.mean()) ** 2).sum()
        return 1.0 - (resid ** 2).sum() / ss_tot

    return {
        "base": fit(base),
        "signed": fit(np.column_stack([base, dg])),
        "sign_blind": fit(np.column_stack([base, np.abs(dg)])),
        "both": fit(np.column_stack([base, dg, np.abs(dg)])),
    }


def main():
    fs, w1, w2 = _env.verify_evaluators(boards.FIELD)
    fg = fastgauge.FastGauges()
    obj = Obj(fs, fg)

    # ---------- P1: recover PRICEBAND-1's sfb unconstrained level (253.9006 = arm-B) ----------
    print("== P1: my frontier machinery on gauge=sfb at PRICEBAND-1's INERT caps ==")
    print("   PRICEBAND-1 published: every inert cap lands on 253.9006 at kmstats-sfb 2.5391 (= arm-B)")
    sfb_km = lambda q: fg.sfb_km(q[:30])  # noqa: E731  (its convention)
    p1 = {}
    for c in (3.0, 3.5, 5.0, 8.0, float("inf")):
        rng = np.random.default_rng(20260801)
        best, bp, vals = search.cap_min_ms(obj, rng, c, 32, gauge=sfb_km)
        p1[str(c)] = {"best_ms": float(best), "sfb_kmstats": float(sfb_km(bp)),
                      "n_feasible": int(len(vals)),
                      "matches_arm_B_layout": bool(
                          np.array_equal(bp[:30], fs.perm(boards.FIELD["arm-B"])[:30]))}
        print(f"   cap {c:<5} ms={best:.4f}  sfb_km={sfb_km(bp):.4f}  "
              f"armB-layout={p1[str(c)]['matches_arm_B_layout']}  feas={len(vals)}/32")
    armb_ms = fs.ms_per_char(boards.FIELD["arm-B"])
    levels = [v["best_ms"] for v in p1.values()]
    p1_gap = max(abs(x - 253.9006) for x in levels)
    print(f"   arm-B's own ms/char (shipped card path) = {armb_ms:.4f}")
    print(f"   worst |inert level - PRICEBAND's 253.9006| = {p1_gap:.4f}   "
          f"=> {'PASS' if p1_gap < 0.05 else 'FAIL'}")

    # ---------- LITERAL sign-blind, on a single-swap scissor perturbation sweep ----------
    print("\n== LITERAL sign-blind falsifier on a single-swap `scissor` sweep ==")
    scis = lambda q: fg.scissor_only(q[:30])  # noqa: E731
    samples = {}
    # (a) the 13 optimized boards, in-band
    for label, names in (("optimized_in_band", boards.OPTIMIZED),
                         ("qwerty30m", [boards.OFF_FRONTIER])):
        dgs, dmss, bix = [], [], []
        for bi, n in enumerate(names):
            p = fs.perm(boards.FIELD[n])
            base_ms, base_g = obj.ms(p), float(scis(p))
            P = search.swap_perms(p)
            for q in P:
                dgs.append(float(scis(q)) - base_g)
                dmss.append(obj.ms(q) - base_ms)
                bix.append(bi)
        dg = np.array(dgs); dms = np.array(dmss); bix = np.array(bix)
        # Only swaps that MOVE the gauge carry directional information; the 91/435 zero-change
        # swaps are pure disruption and are reported separately rather than silently included.
        moved = np.abs(dg) > 1e-12
        r2_all = nested_r2(dg, dms, bix, len(names))
        r2_mv = nested_r2(dg[moved], dms[moved], bix[moved], len(names))
        samples[label] = {
            "n": int(len(dg)), "n_gauge_moved": int(moved.sum()),
            "n_gauge_zero_change": int((~moved).sum()),
            "r2_all_swaps": r2_all, "r2_moved_only": r2_mv,
            "winner_all": "SIGN-BLIND" if r2_all["sign_blind"] > r2_all["signed"] else "SIGNED",
            "winner_moved": "SIGN-BLIND" if r2_mv["sign_blind"] > r2_mv["signed"] else "SIGNED",
            "sign_blind_pct_of_signed_moved": 100.0 * r2_mv["sign_blind"] / r2_mv["signed"]
            if r2_mv["signed"] else None,
        }
        s = samples[label]
        print(f"  {label:<20} n={s['n']:>5} (moved {s['n_gauge_moved']}, "
              f"zero-change {s['n_gauge_zero_change']})")
        for k, r2 in (("all swaps", r2_all), ("moved only", r2_mv)):
            print(f"     {k:<11} R2 signed={r2['signed']:.4f}  sign-blind={r2['sign_blind']:.4f}"
                  f"  both={r2['both']:.4f}  -> "
                  f"{'SIGN-BLIND' if r2['sign_blind'] > r2['signed'] else 'SIGNED'}"
                  f" ({100.0*r2['sign_blind']/r2['signed']:.1f}%)")

    out = {"p1_sfb_recovery": p1, "p1_arm_b_ms_shipped_path": armb_ms,
           "p1_worst_gap_vs_priceband_253_9006": p1_gap,
           "p1_verdict": "PASS" if p1_gap < 0.05 else "FAIL",
           "sign_blind": samples,
           "fasteval_worst": w1, "fastgauge_worst": w2}
    with open(ART + "/s06_control.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(ART + "/s06_DONE", "w") as f:
        f.write("0\n")
    print("\nwrote s06_control.json")


if __name__ == "__main__":
    main()
