"""E3 + INVARIANT 0: reproduce PICK2-1's contrast, then test its four rival explanations.

E3            reproduce +63.00 / +41.03 / 0.651x / CI95 [53,73] with my own code.
H-PRACTICE    restore the practice term b to the MODEL side (the registered decisive falsifier).
H-SHRINK      the same raw-vs-model contrast on >=3 OTHER partitions of the same pairs.
H-AGG         sample-weighted / matched-support / within-pair-paired forms of the contrast.
"""
import json
import time
from collections import defaultdict

import numpy as np
from _guard import BI, MIN_N, SERVE, ART, assert_d5

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

import surface  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import build_cells  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

# pick2 used ROW_STAGGERED_31 for the finger map and the position index; the shipped T2 table is
# built on ROW_STAGGERED_30 + space, whose first 30 slots are IDENTICAL (geometry.py: K31 APPENDS
# the quote slot). Positions 0..29 therefore mean the same thing in both, and pick2's `a >= 30`
# filter drops index 30 = the quote slot under K31 indexing. Reproduced exactly, then checked.
G31 = ROW_STAGGERED_31
POS31 = [*G31.slots, G31.space_position]          # 31 slots + space = 32 entries
PIDX = {p: i for i, p in enumerate(POS31)}
log(f"K31 index: {len(POS31)} entries (30 letters, 30=quote slot, 31=space)")

log(f"loading {BI}")
rows = load_strokes(BI, ngram_len=2, wpm_threshold=0, min_samples=1)
cells = build_cells(rows, 40, 140, 20, 1)
log(f"  {len(rows)} rows -> {len(cells)} cells")
assert len(rows) == 2202, f"frame drift: {len(rows)} bigram rows != 2202"

# ------------------------------------------------------------- the model side: shipped T2, seed-mean
log("building the seed-mean shipped T2 (seeds 0,1,2) on ROW_STAGGERED_30+space")
T2s, _ = surface.load_all_seed_tables(seeds=(0, 1, 2), verbose=False)
T2 = np.mean(T2s, axis=0)

# pick2 built its own T2 via TimeSurface(target_wpm=SERVE+10=90) -- the same 90.0 as production.
from keybo.analysis.timecard import TimeSurface  # noqa: E402
surf90 = TimeSurface({}, target_wpm=float(SERVE + 10), keep_seed_tables=True)
log(f"  pick2's TimeSurface T2 vs mine: max|diff| = {np.abs(surf90._T2 - T2).max():.3e}")

# ------------------------------------------- the practice term b, pooled EXACTLY as the raw samples are
# b is per-BIGRAM-IDENTITY and LOG-scale (LOGRAT). The raw samples for a position pair pool over
# every bigram identity that lands on that pair, so b must be pooled the same way, weighted by the
# sample counts that actually entered the raw median.
bmaps = [surface.load_shipped_model(f"bigram_reg31_seed{s}").metadata.extra["training"]
         ["practice_term"]["values"] for s in (0, 1, 2)]
log(f"practice term: {[len(b) for b in bmaps]} b values per seed (log-scale, LOGRAT)")

# ------------------------------------------------------------------------------- aggregate the cells
agg_vals = defaultdict(list)      # (a,b) -> [durations]
agg_bnum = defaultdict(float)     # (a,b) -> sum over samples of count-weighted b
agg_bden = defaultdict(float)
n_missing_b = 0
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
    bs = [bm.get(c.ngram) for bm in bmaps]
    if any(x is None for x in bs):
        n_missing_b += 1
        bs = [0.0 if x is None else x for x in bs]
    agg_bnum[(a, b)] += float(np.mean(bs)) * len(vals)
    agg_bden[(a, b)] += len(vals)
log(f"aggregated {len(agg_vals)} position pairs at serve bucket {SERVE} "
    f"({n_missing_b} cells had no b entry -> treated as b=0)")

# ------------------------------------------------------------------- pick2's exact selection filter
same, other = [], []
for (a, b), vals in agg_vals.items():
    if len(vals) < MIN_N or a == b:
        continue
    if a >= 30 or b >= 30:                       # skip space- and quote-touching
        continue
    raw = float(np.median(vals))
    pred = float(T2[a, b])                        # T2 is 31x31 on K30+space; 0..29 align
    bbar = agg_bnum[(a, b)] / agg_bden[(a, b)]
    fa, fb = G31.finger(POS31[a][0]), G31.finger(POS31[b][0])
    rec = {"a": a, "b": b, "raw": raw, "pred": pred, "n": len(vals), "bbar": bbar,
           "pred_b": pred * float(np.exp(bbar)),
           "dy": abs(POS31[a][1] - POS31[b][1]),
           "same_hand": G31.hand(POS31[a][0]) == G31.hand(POS31[b][0]),
           "row_b": POS31[b][1], "row_a": POS31[a][1]}
    from keybo.features import classify as C
    rec["adjacent"] = bool(C.is_adjacent(G31, POS31[a], POS31[b]))
    (same if fa == fb else other).append(rec)

for tag, arr in (("SAME-FINGER", same), ("OTHER", other)):
    require_finite([x for r in arr for x in (r["raw"], r["pred"], r["pred_b"])], f"{tag}")
    log(f"  {tag:12s} {len(arr):4d} pairs, {sum(r['n'] for r in arr):8d} raw samples")

RNG = np.random.default_rng(0)     # pick2 used default_rng(0); same for byte-comparability


def contrast(s_arr, o_arr, key, weighted=False, n_boot=4000, boot=True):
    """median(key | same) - median(key | other), with pick2's pair-level bootstrap on RAW."""
    def med(arr, k):
        v = np.array([r[k] for r in arr], float)
        if not weighted:
            return float(np.median(v))
        w = np.array([r["n"] for r in arr], float)
        o = np.argsort(v)
        v, w = v[o], w[o]
        cw = np.cumsum(w) / w.sum()
        return float(v[np.searchsorted(cw, 0.5)])
    ms, mo = med(s_arr, key), med(o_arr, key)
    out = {"median_same": ms, "median_other": mo, "penalty": ms - mo,
           "n_same": len(s_arr), "n_other": len(o_arr)}
    if boot:
        vs = np.array([r[key] for r in s_arr], float)
        vo = np.array([r[key] for r in o_arr], float)
        bs = [float(np.median(RNG.choice(vs, len(vs))) - np.median(RNG.choice(vo, len(vo))))
              for _ in range(n_boot)]
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


R = {}
log("")
log("=== E3: REPRODUCE PICK2-1 =========================================================")
raw_c = contrast(same, other, "raw")
mod_c = contrast(same, other, "pred", boot=False)
R["e3_raw"] = raw_c
R["e3_model"] = mod_c
R["e3_ratio_model_over_raw"] = mod_c["penalty"] / raw_c["penalty"]
log(f"  RAW    same {raw_c['median_same']:.2f}  other {raw_c['median_other']:.2f}  "
    f"penalty {raw_c['penalty']:+.2f} ms   CI95 [{raw_c['ci95'][0]:+.2f}, {raw_c['ci95'][1]:+.2f}]")
log(f"  MODEL  same {mod_c['median_same']:.4f}  other {mod_c['median_other']:.4f}  "
    f"penalty {mod_c['penalty']:+.2f} ms")
log(f"  ratio model/raw = {R['e3_ratio_model_over_raw']:.3f}   "
    f"(pick2 published: raw +63.00, model +41.03, ratio 0.651, CI [53,73])")
R["e3_vs_published"] = {
    "raw_penalty_diff": raw_c["penalty"] - 63.00, "model_penalty_diff": mod_c["penalty"] - 41.03,
    "ratio_diff": R["e3_ratio_model_over_raw"] - 0.651,
    "n_same_pairs_diff": len(same) - 55, "n_other_pairs_diff": len(other) - 431,
    "n_same_samples": sum(r["n"] for r in same), "n_other_samples": sum(r["n"] for r in other),
}
log(f"  vs published: n_same {len(same)} (pub 55), n_other {len(other)} (pub 431), "
    f"samples {sum(r['n'] for r in same)} (pub 230373) / {sum(r['n'] for r in other)} (pub 3637554)")

log("")
log("=== H-PRACTICE: restore b to the MODEL side (THE DECISIVE FALSIFIER) ===============")
modb_c = contrast(same, other, "pred_b", boot=False)
R["hpractice"] = modb_c
bs_med = float(np.median([r["bbar"] for r in same]))
bo_med = float(np.median([r["bbar"] for r in other]))
R["hpractice_bbar"] = {"median_b_same": bs_med, "median_b_other": bo_med,
                       "contrast": bs_med - bo_med, "predicted_needed": 0.1337}
log(f"  b-bar median: same {bs_med:+.4f}  other {bo_med:+.4f}  contrast {bs_med - bo_med:+.4f} "
    f"(prereg predicted ~+0.1337 needed)")
log(f"  MODEL+b  same {modb_c['median_same']:.4f}  other {modb_c['median_other']:.4f}  "
    f"penalty {modb_c['penalty']:+.2f} ms")
lo, hi = raw_c["ci95"]
inside = lo <= modb_c["penalty"] <= hi
closed = (modb_c["penalty"] - mod_c["penalty"]) / (raw_c["penalty"] - mod_c["penalty"])
R["hpractice_verdict"] = {
    "inside_raw_ci95": bool(inside), "raw_ci95": [lo, hi],
    "gap_closed_fraction": float(closed),
    "verdict": ("CONFIRMED -- inside the raw CI95; the 'underprice' is an ESTIMAND MISMATCH"
                if inside else
                "PARTIAL -- closes >=50% but outside the CI" if closed >= 0.5 else
                "REFUTED -- closes <50% of the gap"),
}
log(f"  gap closed: {100 * closed:.1f}%   inside raw CI95 [{lo:+.2f},{hi:+.2f}]: {inside}")
log(f"  => H-PRACTICE {R['hpractice_verdict']['verdict']}")
# second falsifier: does b explain both class LEVELS, not just the contrast?
R["hpractice_levels"] = {
    "same_raw": raw_c["median_same"], "same_model": mod_c["median_same"],
    "same_model_b": modb_c["median_same"],
    "other_raw": raw_c["median_other"], "other_model": mod_c["median_other"],
    "other_model_b": modb_c["median_other"],
    "same_level_closed": float((modb_c["median_same"] - mod_c["median_same"])
                               / (raw_c["median_same"] - mod_c["median_same"])),
    "other_level_closed": float((modb_c["median_other"] - mod_c["median_other"])
                                / (raw_c["median_other"] - mod_c["median_other"])),
}
log(f"  LEVELS falsifier: same-finger level closed "
    f"{100 * R['hpractice_levels']['same_level_closed']:.1f}%, "
    f"other level closed {100 * R['hpractice_levels']['other_level_closed']:.1f}% "
    f"(both must move TOWARD raw for the mechanism to be real)")

log("")
log("=== H-SHRINK: the same contrast on OTHER partitions of the SAME pairs ==============")
ALL = same + other
PARTITIONS = {
    "same-finger (the claim)": lambda r: r in same,
    "same-hand vs alternate": lambda r: r["same_hand"],
    "bottom-row landing vs not": lambda r: r["row_b"] == 1,
    "top-row landing vs not": lambda r: r["row_b"] == 3,
    "dy>=2 vs dy<2": lambda r: r["dy"] >= 2,
    "adjacent-finger vs not": lambda r: r["adjacent"],
}
shrink = {}
for name, pred in PARTITIONS.items():
    A = [r for r in ALL if pred(r)]
    B = [r for r in ALL if not pred(r)]
    if len(A) < 10 or len(B) < 10:
        continue
    rc = contrast(A, B, "raw", boot=False)
    mc = contrast(A, B, "pred", boot=False)
    ratio = mc["penalty"] / rc["penalty"] if abs(rc["penalty"]) > 1e-9 else float("nan")
    shrink[name] = {"n_A": len(A), "n_B": len(B), "raw_penalty": rc["penalty"],
                    "model_penalty": mc["penalty"], "ratio_model_over_raw": ratio}
    log(f"  {name:28s} nA={len(A):4d} raw {rc['penalty']:+8.2f} model {mc['penalty']:+8.2f} "
        f"ratio {ratio:+.3f}")
R["hshrink"] = shrink
others = [v["ratio_model_over_raw"] for k, v in shrink.items() if k != "same-finger (the claim)"
          and np.isfinite(v["ratio_model_over_raw"])]
sf_ratio = shrink["same-finger (the claim)"]["ratio_model_over_raw"]
R["hshrink_verdict"] = {
    "same_finger_ratio": sf_ratio, "other_partition_ratios": others,
    "other_min": float(min(others)), "other_max": float(max(others)),
    "same_finger_inside_other_spread": bool(min(others) <= sf_ratio <= max(others)),
}
log(f"  same-finger ratio {sf_ratio:.3f} vs other partitions "
    f"[{min(others):.3f}, {max(others):.3f}] -> inside spread: "
    f"{R['hshrink_verdict']['same_finger_inside_other_spread']}")

log("")
log("=== H-AGG: three alternative aggregations of the SAME contrast =====================")
agg = {}
agg["sample_weighted"] = {"raw": contrast(same, other, "raw", weighted=True, boot=False),
                          "model": contrast(same, other, "pred", weighted=True, boot=False)}
nmed_same = float(np.median([r["n"] for r in same]))
ms_ = [r for r in same if r["n"] >= nmed_same]
mo_ = [r for r in other if r["n"] >= nmed_same]
agg["matched_support"] = {"threshold_n": nmed_same, "n_same": len(ms_), "n_other": len(mo_),
                          "raw": contrast(ms_, mo_, "raw", boot=False),
                          "model": contrast(ms_, mo_, "pred", boot=False)}
# within-pair paired residual: median over pairs of (raw - model) per pair, per class
ps = float(np.median([r["raw"] - r["pred"] for r in same]))
po = float(np.median([r["raw"] - r["pred"] for r in other]))
agg["within_pair_paired"] = {"median_resid_same": ps, "median_resid_other": po,
                             "resid_contrast": ps - po}
for k in ("sample_weighted", "matched_support"):
    rp, mp = agg[k]["raw"]["penalty"], agg[k]["model"]["penalty"]
    log(f"  {k:20s} raw {rp:+8.2f} model {mp:+8.2f} ratio {mp / rp:+.3f}")
    agg[k]["ratio"] = mp / rp
log(f"  {'within_pair_paired':20s} resid(same) {ps:+8.2f} resid(other) {po:+8.2f} "
    f"contrast {ps - po:+8.2f}  (this IS the raw-minus-model penalty gap, paired)")
R["hagg"] = agg
signs = [np.sign(agg["sample_weighted"]["raw"]["penalty"] - agg["sample_weighted"]["model"]["penalty"]),
         np.sign(agg["matched_support"]["raw"]["penalty"] - agg["matched_support"]["model"]["penalty"]),
         np.sign(ps - po)]
R["hagg_verdict"] = {"gap_signs": [float(x) for x in signs],
                     "stable_sign": bool(len(set(signs)) == 1)}
log(f"  gap sign stable across all three forms: {R['hagg_verdict']['stable_sign']}")

# per-pair heterogeneity of the same-finger residual (prereg gate A5)
res_s = np.array([r["raw"] - r["pred"] for r in same])
R["a5_same_finger_residual_spread"] = {
    "n": len(res_s), "min": float(res_s.min()), "p25": float(np.percentile(res_s, 25)),
    "median": float(np.median(res_s)), "p75": float(np.percentile(res_s, 75)),
    "max": float(res_s.max()), "sd": float(res_s.std(ddof=1)),
    "frac_negative": float((res_s < 0).mean()),
}
log("")
log(f"A5 same-finger per-pair (raw-model) residual: median {np.median(res_s):+.2f} "
    f"[p25 {np.percentile(res_s, 25):+.2f}, p75 {np.percentile(res_s, 75):+.2f}] "
    f"sd {res_s.std(ddof=1):.2f}, {100 * (res_s < 0).mean():.0f}% NEGATIVE")

R["pairs_same"] = same
R["pairs_other"] = other
R["wall_s"] = time.time() - t0
json.dump(R, open(f"{ART}/c02_contrast.json", "w"), indent=1)
log(f"wrote {ART}/c02_contrast.json")
log("ALL-DONE")
