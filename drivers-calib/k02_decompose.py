"""INVARIANT 1: DECOMPOSE the raw-vs-model gap, with each mechanism's share QUANTIFIED.

Reads the stroke frame ONCE (the 609 MB-scale read sfbprice paid for) and adds the one thing its
c02 could not carry: PER-PAIR SPLIT-HALF replicates, which is the only way to separate
"the model is wrong" from "the raw target is noisy".

E-CONTROL-2  my per-pair raw medians must reproduce sfbprice's c02 records byte-for-byte.
M1 RELIABILITY  split-half the samples inside each pair (random halves AND participant halves) ->
                the reliability of the raw side, and hence the CEILING on r^2 that ANY model could
                reach. r^2=0.4328 is only a defect to the extent it falls below that ceiling.
M2 EIV          disattenuate the raw contrasts for raw-side noise; re-measure the six ratios.
M3 ESTIMAND     the practice term b (per class), and the AGGREGATION change
                (median-of-medians -> sample-weighted / conditional-mean form).
M4 B-CANCELS    the arithmetic claim, verified numerically: sum_ngram freq*b is layout-INDEPENDENT,
                so b cancels EXACTLY in any layout-vs-layout difference. Decides INVARIANT 4.
"""
import json
import time
from collections import defaultdict

import numpy as np
from _guard import (ART, BI, BOOT_SEED, CLASS_ORDER, MIN_N, SERVE, SFBPRICE_C02, SHIPPED,
                    assert_d5, class_masks, sha)

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

import surface  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import build_cells  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

RNG = np.random.default_rng(BOOT_SEED)
out = {"boot_seed": BOOT_SEED, "serve_bucket": SERVE, "min_n": MIN_N}

G31 = ROW_STAGGERED_31
POS31 = [*G31.slots, G31.space_position]
PIDX = {p: i for i, p in enumerate(POS31)}

log(f"loading {BI}")
rows = load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
assert len(rows) == 2202, f"frame drift: {len(rows)} != 2202"
cells = build_cells(rows, 40, 140, 20, 1)
log(f"  {len(rows)} rows -> {len(cells)} cells")

log("building the seed-mean shipped T2 (seeds 0,1,2)")
T2s, _ = surface.load_all_seed_tables(seeds=(0, 1, 2), verbose=False)
T2 = np.mean(T2s, axis=0)

bmaps = [surface.load_shipped_model(f"bigram_reg31_seed{s}").metadata.extra["training"]
         ["practice_term"]["values"] for s in (0, 1, 2)]
log(f"practice term: {[len(b) for b in bmaps]} b values per seed (log-scale, LOGRAT)")

# ------------------------------------------------------------------- aggregate exactly as sfbprice did
agg_vals = defaultdict(list)          # (a,b) -> [durations]
agg_pids = defaultdict(list)          # (a,b) -> [pid]
agg_bnum, agg_bden = defaultdict(float), defaultdict(float)
agg_ng = defaultdict(lambda: defaultdict(int))   # (a,b) -> {ngram: n_samples}
for c in cells:
    if c.bucket != SERVE:
        continue
    try:
        a = PIDX[tuple(int(v) for v in c.positions[0])]
        b = PIDX[tuple(int(v) for v in c.positions[1])]
    except KeyError:
        continue
    vals = [float(s[1]) for s in c.samples]
    agg_vals[(a, b)].extend(vals)
    agg_pids[(a, b)].extend(int(s[2]) for s in c.samples)
    bs = [bm.get(c.ngram) for bm in bmaps]
    bs = [0.0 if x is None else x for x in bs]
    agg_bnum[(a, b)] += float(np.mean(bs)) * len(vals)
    agg_bden[(a, b)] += len(vals)
    agg_ng[(a, b)][c.ngram] += len(vals)
log(f"aggregated {len(agg_vals)} position pairs at serve bucket {SERVE}")

PAIRS = []
for (a, b), vals in agg_vals.items():
    if len(vals) < MIN_N or a == b or a >= 30 or b >= 30:
        continue
    v = np.asarray(vals, float)
    pids = np.asarray(agg_pids[(a, b)], int)
    bbar = agg_bnum[(a, b)] / agg_bden[(a, b)]
    fa, fb = G31.finger(POS31[a][0]), G31.finger(POS31[b][0])
    PAIRS.append({
        "a": a, "b": b, "raw": float(np.median(v)), "pred": float(T2[a, b]), "n": len(v),
        "bbar": bbar, "pred_b": float(T2[a, b]) * float(np.exp(bbar)),
        "dy": abs(POS31[a][1] - POS31[b][1]),
        "same_hand": G31.hand(POS31[a][0]) == G31.hand(POS31[b][0]),
        "row_b": POS31[b][1], "row_a": POS31[a][1],
        "adjacent": bool(C.is_adjacent(G31, POS31[a], POS31[b])),
        "same_finger": bool(fa == fb),
        "mean": float(v.mean()), "_v": v, "_pids": pids,
    })
log(f"selected {len(PAIRS)} pairs ({sum(p['same_finger'] for p in PAIRS)} same-finger)")

# =========================================================== E-CONTROL-2: reproduce sfbprice's records
src = json.load(open(SFBPRICE_C02))
ref = {(r["a"], r["b"]): r for r in src["pairs_same"] + src["pairs_other"]}
mine = {(p["a"], p["b"]): p for p in PAIRS}
assert set(ref) == set(mine), f"pair-set drift: mine-ref={len(set(mine) - set(ref))}, ref-mine={len(set(ref) - set(mine))}"
w_raw = max(abs(ref[k]["raw"] - mine[k]["raw"]) for k in ref)
w_pred = max(abs(ref[k]["pred"] - mine[k]["pred"]) for k in ref)
w_predb = max(abs(ref[k]["pred_b"] - mine[k]["pred_b"]) for k in ref)
w_n = max(abs(ref[k]["n"] - mine[k]["n"]) for k in ref)
out["e_control_2"] = {"n_pairs": len(ref), "worst_abs_diff_raw": w_raw, "worst_abs_diff_pred": w_pred,
                      "worst_abs_diff_pred_b": w_predb, "worst_abs_diff_n": int(w_n),
                      "sfbprice_c02_sha256": sha(SFBPRICE_C02),
                      "pass": bool(w_raw == 0 and w_pred < 1e-9 and w_n == 0)}
log(f"E-CONTROL-2: {len(ref)} pairs, worst|d raw|={w_raw:.3e} |d pred|={w_pred:.3e} "
    f"|d pred_b|={w_predb:.3e} |d n|={w_n} -> {'PASS' if out['e_control_2']['pass'] else 'FAIL'}")

MASKS = class_masks(PAIRS, {id(p) for p in PAIRS if p["same_finger"]})
raw = np.array([p["raw"] for p in PAIRS], float)
pred = np.array([p["pred"] for p in PAIRS], float)
pred_b = np.array([p["pred_b"] for p in PAIRS], float)
mean_obs = np.array([p["mean"] for p in PAIRS], float)
require_finite(list(raw) + list(pred) + list(pred_b) + list(mean_obs), "pair arrays")


def con(values, mask):
    m = np.asarray(mask, bool)
    return float(np.median(values[m]) - np.median(values[~m]))


# ================================================== M1: RELIABILITY of the raw side -> the r^2 CEILING
log("M1: split-half reliability of the raw side (random halves AND participant halves)")
N_SPLIT = 200


def split_half_medians(mode):
    """Return (n_pairs, N_SPLIT) arrays (m1, m2) of half-sample medians per pair."""
    m1 = np.empty((len(PAIRS), N_SPLIT))
    m2 = np.empty((len(PAIRS), N_SPLIT))
    for i, p in enumerate(PAIRS):
        v, pids = p["_v"], p["_pids"]
        for j in range(N_SPLIT):
            if mode == "random":
                perm = RNG.permutation(len(v))
                h = len(v) // 2
                m1[i, j] = np.median(v[perm[:h]])
                m2[i, j] = np.median(v[perm[h:2 * h]])
            else:  # participant-level: split the PIDs, not the samples
                u = np.unique(pids)
                if len(u) < 2:
                    m1[i, j] = m2[i, j] = np.nan
                    continue
                perm = RNG.permutation(u)
                g1 = set(perm[:len(u) // 2].tolist())
                sel = np.array([q in g1 for q in pids])
                if sel.sum() < 1 or (~sel).sum() < 1:
                    m1[i, j] = m2[i, j] = np.nan
                    continue
                m1[i, j] = np.median(v[sel])
                m2[i, j] = np.median(v[~sel])
    return m1, m2


rel = {}
for mode in ("random", "participant"):
    m1, m2 = split_half_medians(mode)
    ok = np.isfinite(m1) & np.isfinite(m2)
    # Spearman-Brown: reliability of the FULL-sample median from the two half correlations
    rs = []
    for j in range(N_SPLIT):
        s = ok[:, j]
        if s.sum() > 10:
            rs.append(np.corrcoef(m1[s, j], m2[s, j])[0, 1])
    r_half = float(np.mean(rs))
    r_full = 2 * r_half / (1 + r_half)                       # Spearman-Brown
    # noise variance of the full-sample median, from the half-difference
    d = (m1 - m2)[ok]
    var_full_median = float(np.var(d, ddof=1) / 4.0)         # var(m_full) ~ var(d)/4
    rel[mode] = {"r_halves": r_half, "reliability_full_median_spearman_brown": r_full,
                 "n_split": N_SPLIT, "var_noise_full_median": var_full_median,
                 "sd_noise_full_median": float(np.sqrt(var_full_median))}
    log(f"  {mode:12s} r(halves)={r_half:.4f}  reliability(full median)={r_full:.4f}  "
        f"noise sd of a pair's median = {np.sqrt(var_full_median):.3f} ms")
out["m1_reliability"] = rel

var_raw = float(np.var(raw, ddof=1))
r = float(np.corrcoef(pred, raw)[0, 1])
out["m1_r2_ceiling"] = {}
for mode, d in rel.items():
    reliab = d["reliability_full_median_spearman_brown"]
    var_noise = d["var_noise_full_median"]
    # ceiling on r^2 against a NOISY target: r2_max = reliability (= signal share of raw variance)
    r2_ceil_rel = reliab
    r2_ceil_var = max(0.0, 1.0 - var_noise / var_raw)
    out["m1_r2_ceiling"][mode] = {
        "r2_observed": r * r, "r2_ceiling_from_reliability": r2_ceil_rel,
        "r2_ceiling_from_variance": r2_ceil_var,
        "frac_of_ceiling_reliability": (r * r) / r2_ceil_rel if r2_ceil_rel > 0 else float("nan"),
        "frac_of_ceiling_variance": (r * r) / r2_ceil_var if r2_ceil_var > 0 else float("nan"),
        "var_raw": var_raw, "var_noise": var_noise}
    log(f"  {mode:12s} r2 observed {r * r:.4f} vs ceiling {r2_ceil_rel:.4f} (reliability) / "
        f"{r2_ceil_var:.4f} (variance) => {100 * (r * r) / r2_ceil_var:.1f}% of the variance ceiling")

# ===================================================== M2: disattenuate the raw contrasts for EIV
log("M2: EIV -- do the six ratios move once raw-side noise is removed?")
m2res = {}
for mode, d in rel.items():
    var_noise = d["var_noise_full_median"]
    shrink = np.sqrt(max(0.0, 1.0 - var_noise / var_raw))   # deflate raw spread to its signal part
    per_class = {}
    for cname in CLASS_ORDER:
        m = np.asarray(MASKS[cname], bool)
        cr, cm = con(raw, m), con(pred, m)
        cr_clean = cr * shrink        # a noise-free raw contrast is SMALLER by the signal fraction
        per_class[cname] = {"raw": cr, "raw_disattenuated": cr_clean, "model": cm,
                            "ratio": cm / cr, "ratio_vs_disattenuated": cm / cr_clean}
    m2res[mode] = {"raw_spread_signal_fraction": float(shrink), "per_class": per_class,
                   "mean_ratio": float(np.mean([v["ratio"] for v in per_class.values()])),
                   "mean_ratio_disattenuated":
                       float(np.mean([v["ratio_vs_disattenuated"] for v in per_class.values()]))}
    log(f"  {mode:12s} signal fraction of raw spread = {shrink:.4f}; mean ratio "
        f"{m2res[mode]['mean_ratio']:.4f} -> {m2res[mode]['mean_ratio_disattenuated']:.4f}")
out["m2_eiv"] = m2res

# ====================================== M3: ESTIMAND -- practice term, and the aggregation change
log("M3: estimand -- practice term b, and median-of-medians vs the conditional-mean form")
m3 = {"per_class": {}}
for cname in CLASS_ORDER:
    m = np.asarray(MASKS[cname], bool)
    cr, cm, cmb = con(raw, m), con(pred, m), con(pred_b, m)
    cr_mean = con(mean_obs, m)                     # raw side as a MEAN, not a median
    m3["per_class"][cname] = {
        "raw_median": cr, "raw_mean": cr_mean, "model": cm, "model_with_b": cmb,
        "ratio": cm / cr, "ratio_with_b": cmb / cr,
        "ratio_vs_raw_mean": cm / cr_mean if cr_mean else float("nan"),
        "ratio_with_b_vs_raw_mean": cmb / cr_mean if cr_mean else float("nan"),
        "b_gap_closed_frac": (cmb - cm) / (cr - cm) if cr != cm else float("nan"),
    }
    d = m3["per_class"][cname]
    log(f"  {cname:20s} ratio {cm / cr:.4f} | +b {cmb / cr:.4f} | vs raw-MEAN {d['ratio_vs_raw_mean']:.4f} "
        f"| +b & raw-MEAN {d['ratio_with_b_vs_raw_mean']:.4f}")
for k in ("ratio", "ratio_with_b", "ratio_vs_raw_mean", "ratio_with_b_vs_raw_mean"):
    m3[f"mean_{k}"] = float(np.mean([v[k] for v in m3["per_class"].values()]))
m3["mean_b_gap_closed_frac"] = float(np.mean([v["b_gap_closed_frac"]
                                              for v in m3["per_class"].values()]))
log(f"  MEAN over 6 classes: {m3['mean_ratio']:.4f} -> +b {m3['mean_ratio_with_b']:.4f} -> "
    f"+b & raw-mean {m3['mean_ratio_with_b_vs_raw_mean']:.4f}")
out["m3_estimand"] = m3

# ======================= M4: does b CANCEL between layouts? The claim that decides INVARIANT 4.
log("M4: does the practice term cancel between layouts? (arithmetic claim, verified numerically)")
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402

BOARDS = {
    "candidate": "pyu.,vdfnlhieaocstrmkj'-qgwbzx",
    "F(2.0)": "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "qwerty": NAMED_LAYOUTS["qwerty"],
    "colemak-dh": NAMED_LAYOUTS["colemak-dh"],
    "dvorak": NAMED_LAYOUTS["dvorak"],
}
_cdir, tri_freq = surface.corpus()

# b is keyed by NGRAM IDENTITY (a 2-char string). The per-board total is
#   sum_ng freq(ng) * b(ng),
# whose terms depend on the CORPUS only -- the layout never enters. So it is the same constant for
# every board, and it cancels EXACTLY in any board-vs-board difference. Verified numerically below
# by computing the b-total per board, i.e. NOT trusting the argument.
bmap_mean = {}
for k in set().union(*[set(bm) for bm in bmaps]):
    bmap_mean[k] = float(np.mean([bm.get(k, 0.0) for bm in bmaps]))

# the corpus's FIRST-TRANSITION bigram marginal, exactly the weighting the T2 term receives
bi_marginal = defaultdict(float)
for ng, f in tri_freq.items():
    bi_marginal[ng[:2]] += f
tot_mass = sum(bi_marginal.values())

per_board = {}
for name, lay in BOARDS.items():
    slot = {ch: i for i, ch in enumerate(lay)}
    slot[" "] = 30
    num = den = 0.0
    for bg, f in bi_marginal.items():
        if bg[0] in slot and bg[1] in slot:        # the board's covered mass
            num += f * bmap_mean.get(bg, 0.0)
            den += f
    per_board[name] = {"b_total_over_covered_mass": num / den, "covered_mass_frac": den / tot_mass}
    log(f"  {name:12s} freq-weighted b = {num / den:+.8f} log units  "
        f"(covers {100 * den / tot_mass:.3f}% of corpus mass)")

vals = [v["b_total_over_covered_mass"] for v in per_board.values()]
spread = float(max(vals) - min(vals))
out["m4"] = {
    "n_b_keys": len(bmap_mean),
    "b_is_keyed_by": "ngram identity (a 2-character string), NOT a position pair",
    "claim": "sum_ngram freq(ngram)*b(ngram) depends only on the corpus, so it is layout-"
             "independent and cancels EXACTLY in any layout-vs-layout difference",
    "per_board": per_board,
    "spread_across_boards": spread,
    "corpus_dir": str(_cdir),
    "note": "the residual spread is COVERAGE, not the layout: boards differ only in WHICH corpus "
            "bigrams they can type (a board missing a character drops those rows). On equal "
            "coverage the constant is identical.",
}
log(f"  spread of the freq-weighted b across {len(BOARDS)} boards = {spread:.3e} log units "
    f"(pure coverage effect; the b VALUES are layout-independent by construction)")

out["wall_s"] = time.time() - t0
for p in PAIRS:                                   # drop the bulk sample arrays before serialising
    p.pop("_v", None); p.pop("_pids", None)
out["pairs"] = PAIRS
path = f"{ART}/k02_decompose.json"
json.dump(out, open(path, "w"), indent=1)
log(f"wrote {path}  ({out['wall_s']:.1f}s)")
