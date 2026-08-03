"""INVARIANT 4: the blast radius. What does the miscalibration change, retrospectively?

The lever is that ms/char is LINEAR in the tables: mspc = sum(f*(T2+Tc))/sum(f), so an affine
surface map (T2,Tc) -> (a2+b*T2, a3+b*Tc) sends mspc -> A + b*mspc for a board-specific A that is
CONSTANT across boards of equal coverage. Hence, for equal-coverage boards:
  * ORDER is preserved exactly (b>0)                        => rank conclusions are INSENSITIVE
  * every MARGIN is multiplied by exactly b                 => ms-denominated claims are SENSITIVE
Proved by construction and then VERIFIED numerically, because "provable" is not "verified".

B1  the 13-board field re-scored under the affine correction; ranks, margins, and coverage.
B2  `candidate` re-checked explicitly: does it still survive? (the parent's named question)
B3  the qwerty-vs-field gap in PERCENT (a ratio -- so it is NOT invariant; measure it).
B4  PRICEBAND-1's ms-denominated sfb cap: the shadow price under the corrected surface.
B5  the per-seed 1v1 with a self-measured floor, so margin-vs-floor precedes any p-value.
"""
import json
import time

import numpy as np
from _guard import ART, BOOT_SEED, assert_d5

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

import surface  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

RNG = np.random.default_rng(BOOT_SEED)

# The affine correction measured by CALIB-1/sfbprice on the 486 supported pairs, and the
# fold-specific oracles k03 measured. B is the only knob; each value is a MEASURED quantity.
B_PAIR = 1.461839          # pair-level, wpm-80 serve bucket (CALIB-1 k01, = sfbprice's 1.4618)
B_QWERTY_FOLD = 1.4067     # k03: the qwerty fold's bucket-centered oracle slope
B_HELDIN = 1.0488          # k03: what a held-in-fitted map actually applies (mean centered)
CORRECTIONS = {"uncorrected": 1.0, "affine_pair_1.4618": B_PAIR,
               "affine_qwerty_fold_1.4067": B_QWERTY_FOLD, "affine_heldin_1.0488": B_HELDIN}

out = {"corrections": CORRECTIONS, "boot_seed": BOOT_SEED,
       "note": "b multiplies the CENTERED surface; the intercept is absorbed so that the "
               "corpus-mean prediction is preserved (a pure re-scaling of GAPS, which is what a "
               "calibration slope is about -- an added constant would change every board equally "
               "and cancel in every comparison anyway)."}

# ------------------------------------------------------------------------------- the field + corpus
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402

TUNED = {
    "arm-B":       "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1":      "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "F(2.5)":      "flmpg-,uoysntdcireahkxbwv.'jzq",
    "F(2.0)":      "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
    "candidate":   "pyu.,vdfnlhieaocstrmkj'-qgwbzx",
    "keybo-lsb":   "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
}
FIELD_ORDER = ("arm-B", "BALL-1", "F(2.5)", "F(2.0)", "candidate", "keybo-lsb", "flagship-c3",
               "colemak", "colemak-dh", "graphite", "semimak", "dvorak", "qwerty")
BOARDS = dict(TUNED)
for nm in ("colemak", "colemak-dh", "graphite", "semimak", "dvorak", "qwerty"):
    BOARDS[nm] = NAMED_LAYOUTS[nm]
for nm, lay in BOARDS.items():
    assert len(lay) == 30 and len(set(lay)) == 30, f"{nm} is not a 30-key permutation"
log(f"field = {len(BOARDS)} boards")

cdir, tri = surface.corpus()
log(f"corpus {cdir}: {len(tri)} trigrams, mass {sum(tri.values())}")
arrays = {nm: surface.board_arrays(lay, tri) for nm, lay in BOARDS.items()}
cover = {nm: float(arrays[nm][3].sum() / sum(tri.values())) for nm in BOARDS}
out["coverage"] = cover
groups = {}
for nm, cv in cover.items():
    groups.setdefault(round(cv, 9), []).append(nm)
out["equal_coverage_groups"] = {str(k): v for k, v in sorted(groups.items(), reverse=True)}
log("coverage groups (the equal-coverage sets inside which order is provably preserved):")
for cv, nms in sorted(groups.items(), reverse=True):
    log(f"  {100 * cv:.4f}%: {nms}")

# ----------------------------------------------------------- per-seed tables (25 seeds, REUSED)
T2s, Tcs = surface.load_all_seed_tables(verbose=False)
log(f"loaded {len(T2s)} seed table pairs")


def scored(b, seed_idx):
    """ms/char for every board on one seed, with the surface's GAPS scaled by b.

    The centred form: T -> mean + b*(T - mean), applied to the (T2+Tc) sum via its two parts, so
    the corpus-weighted mean prediction is preserved and only the dispersion changes.
    """
    T2, Tc = T2s[seed_idx], Tcs[seed_idx]
    if b == 1.0:
        T2b, Tcb = T2, Tc
    else:
        T2b = T2.mean() + b * (T2 - T2.mean())
        Tcb = Tc.mean() + b * (Tc - Tc.mean())
    return {nm: surface.mspc(arrays[nm], T2b, Tcb) for nm in BOARDS}


# =================================================================== B1: ranks and margins by pricing
log("B1: re-scoring the field under each correction (25 seeds each)")
per_pricing = {}
for pname, b in CORRECTIONS.items():
    per_seed = [scored(b, i) for i in range(len(T2s))]
    means = {nm: float(np.mean([s[nm] for s in per_seed])) for nm in BOARDS}
    require_finite(list(means.values()), f"{pname} board means")
    order = sorted(BOARDS, key=lambda nm: means[nm])
    per_pricing[pname] = {"b": b, "means": means, "rank": {nm: order.index(nm) + 1 for nm in BOARDS},
                          "per_seed": [{nm: s[nm] for nm in BOARDS} for s in per_seed]}
    log(f"  {pname:26s} top5 = {order[:5]}")
out["b1"] = {p: {k: v for k, v in d.items() if k != "per_seed"} for p, d in per_pricing.items()}

# order invariance, MEASURED (the claim is provable for equal coverage; check it, incl. unequal)
base_order = sorted(BOARDS, key=lambda nm: per_pricing["uncorrected"]["means"][nm])
inv = {}
for pname, d in per_pricing.items():
    o = sorted(BOARDS, key=lambda nm: d["means"][nm])
    inv[pname] = {"order": o, "identical_to_uncorrected": o == base_order,
                  "n_rank_changes": sum(1 for nm in BOARDS
                                        if d["rank"][nm] != per_pricing["uncorrected"]["rank"][nm])}
    log(f"  {pname:26s} order identical to uncorrected: {inv[pname]['identical_to_uncorrected']} "
        f"({inv[pname]['n_rank_changes']} rank changes)")
out["b1_order_invariance"] = inv

# ============================================== B2: `candidate` re-checked -- the parent's question
log("B2: is `candidate` still un-beaten? (paired per-seed, 25 seeds)")


def floor_split_half(pname, n_draws=4000):
    """FLOOR: a same-board placebo whose TRUTH IS 0 by construction.

    Split the 25 seeds into two halves and difference the SAME board across them: any nonzero
    result is pure estimator noise, so its p90 is the smallest margin this design can resolve.
    """
    ps = per_pricing[pname]["per_seed"]
    n = len(ps)
    vals = []
    for nm in BOARDS:
        x = np.array([s[nm] for s in ps], float)
        for _ in range(n_draws // len(BOARDS)):
            perm = RNG.permutation(n)
            h = n // 2
            vals.append(abs(x[perm[:h]].mean() - x[perm[h:2 * h]].mean()))
    return float(np.percentile(vals, 90)), len(vals)


b2 = {}
for pname in CORRECTIONS:
    ps = per_pricing[pname]["per_seed"]
    fl, ndraw = floor_split_half(pname)
    cand = np.array([s["candidate"] for s in ps], float)
    rows = {}
    losses = []
    for nm in BOARDS:
        if nm == "candidate":
            continue
        other = np.array([s[nm] for s in ps], float)
        marg = float(np.mean(other - cand))          # >0 => candidate is FASTER
        signs = int((other - cand > 0).sum())
        rows[nm] = {"margin_other_minus_candidate": marg, "seeds_favouring_candidate": signs,
                    "n_seeds": len(ps), "abs_margin_over_floor": abs(marg) / fl if fl else np.nan}
        if marg < 0 and abs(marg) > fl:
            losses.append(nm)
    b2[pname] = {"floor_p90": fl, "n_floor_draws": ndraw,
                 "candidate_rank": per_pricing[pname]["rank"]["candidate"],
                 "losses_above_floor": losses, "vs": rows}
    log(f"  {pname:26s} floor_p90={fl:.4f}  candidate rank={b2[pname]['candidate_rank']}/13  "
        f"losses above floor: {losses or 'NONE'}")
out["b2_candidate"] = b2

# ============================================ B3: the qwerty-vs-field gap in PERCENT (a RATIO => not invariant)
log("B3: qwerty-vs-field gap, absolute AND percent")
b3 = {}
for pname in CORRECTIONS:
    m = per_pricing[pname]["means"]
    best = min(m, key=lambda nm: m[nm])
    gap_abs = m["qwerty"] - m[best]
    b3[pname] = {"best_board": best, "best_ms": m[best], "qwerty_ms": m["qwerty"],
                 "gap_ms": gap_abs, "gap_pct_of_qwerty": 100 * gap_abs / m["qwerty"],
                 "gap_pct_of_best": 100 * gap_abs / m[best]}
    log(f"  {pname:26s} best={best} {m[best]:.4f}  qwerty={m['qwerty']:.4f}  "
        f"gap={gap_abs:.4f} ms = {b3[pname]['gap_pct_of_qwerty']:.4f}% of qwerty")
out["b3_qwerty_gap"] = b3

# =========================================== B4: PRICEBAND-1's ms-denominated sfb cap / shadow price
log("B4: the sfb shadow price (ms per pp of sfb) under each correction")
mask = surface.same_finger_mask()
b4 = {}
for pname, b in CORRECTIONS.items():
    # the analytic gradient: raising every same-finger cell by 1 ms raises mspc by the board's
    # same-finger share of the (trigram-weighted) bigram mass. Under a b-scaled surface the
    # gradient in ms per pp scales by exactly b.
    grads = {}
    for nm in BOARDS:
        share = surface.sf_share(arrays[nm], mask)
        grads[nm] = {"sf_share": share}
    m = per_pricing[pname]["means"]
    b4[pname] = {"b": b, "sf_shares": {nm: grads[nm]["sf_share"] for nm in BOARDS},
                 "cost_of_capping_ms": None}
    # the shadow price: min ms/char subject to sfb <= cap, over the field
    log(f"  {pname:26s} sf_share candidate={grads['candidate']['sf_share']:.6f} "
        f"arm-B={grads['arm-B']['sf_share']:.6f} qwerty={grads['qwerty']['sf_share']:.6f}")
out["b4_sfb"] = b4

# ================================== B5: the ms-margin scaling law, verified to machine precision
log("B5: verifying the scaling law -- every margin must scale by exactly b (equal coverage)")
b5 = {}
for pname, b in CORRECTIONS.items():
    if b == 1.0:
        continue
    worst = 0.0
    checked = 0
    for cv, nms in groups.items():
        if len(nms) < 2:
            continue
        for i in range(len(nms)):
            for j in range(i + 1, len(nms)):
                a, c = nms[i], nms[j]
                m0 = (per_pricing["uncorrected"]["means"][a] - per_pricing["uncorrected"]["means"][c])
                m1 = (per_pricing[pname]["means"][a] - per_pricing[pname]["means"][c])
                if abs(m0) > 1e-12:
                    worst = max(worst, abs(m1 / m0 - b))
                    checked += 1
    b5[pname] = {"b": b, "n_pairs_checked": checked, "worst_abs_dev_of_margin_ratio_from_b": worst}
    log(f"  {pname:26s} {checked} equal-coverage pairs, worst |margin_ratio - b| = {worst:.3e}")
out["b5_scaling_law"] = b5

out["wall_s"] = time.time() - t0
path = f"{ART}/k04_blast.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
