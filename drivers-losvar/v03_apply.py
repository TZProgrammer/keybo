"""LOSVAR-1 v03: negative controls, the floor RE-MEASURED, the four calibration bars for LOS_valid,
and the live pair re-answered.

Order is deliberate and matches the prereg: NC1 (reproduce a published quantity) BEFORE any new
number is trusted; the floor MEASURED on my own pipeline (borrowing no constant — the campaign has
four distinct measured floors and the floor is a property of the COMPARISON DESIGN); then the four
registered calibration bars for the new estimand; then, last, the live pair.

sigma_diff comes from v02's artifact. It is a MEASURED input, not a tuned one: the prereg registers
that it will not be adjusted to make a bar pass, so the bars are evaluated at whatever v02 measured.
"""
from __future__ import annotations

import json

import numpy as np
from v00_common import (ALL_SEEDS, ART, BOOT_SEED, CACHED_SEEDS, SEED_TABLES, SHIPPED_SEEDS,
                        TOURNAMENT_JSON, assert_provenance, dump, load_boards, log, require_finite)

log("D5 provenance:")
PROV = assert_provenance()

from keybo.analysis.los import compute_los, scale_floor_to_n, split_half_floor  # noqa: E402

PRICINGS = ("all", "obs", "common")
out: dict = {"provenance": PROV, "boot_seed": BOOT_SEED, "n_seeds": len(ALL_SEEDS)}

# ============================================================ per-seed ms/char, 3 pricings
# TOURNAMENT-1's published per-seed matrix. LOS-1 verified it against its own from-scratch
# recomputation at worst |Δ| 4.97e-12 over 13 boards × 25 seeds, so it is the right input to reuse
# (a fresh stroke reload costs ~400 s and buys nothing). NC1 below re-checks the pairwise means.
tj = json.loads(TOURNAMENT_JSON.read_text())
boards = load_boards()
BOARD_NAMES = list(tj["boards"])
mspc = {p: {b: require_finite(f"mspc[{p}][{b}]", tj["mspc"][p][b]) for b in BOARD_NAMES}
        for p in PRICINGS}
log(f"loaded per-seed ms/char: {len(BOARD_NAMES)} boards × {len(mspc['all'][BOARD_NAMES[0]])} seeds "
    f"× {len(PRICINGS)} pricings")
for p in PRICINGS:
    for b in BOARD_NAMES:
        assert mspc[p][b].size == len(ALL_SEEDS), f"{p}/{b}: {mspc[p][b].size} seeds"

# ============================================================ NC1: reproduce TOURNAMENT-1's margins
nc1 = {"worst_abs_diff": 0.0, "n_checked": 0, "registered_bar": 1e-9, "per_pricing": {}}
for pricing, recs in tj["pairs"].items():
    if pricing not in mspc:
        continue
    worst = 0.0
    for rec in recs:
        mine = float((mspc[pricing][rec["A"]] - mspc[pricing][rec["B"]]).mean())
        worst = max(worst, abs(mine - float(rec["mean"])))
        nc1["n_checked"] += 1
    nc1["per_pricing"][pricing] = {"n": len(recs), "worst_abs_diff": worst}
    nc1["worst_abs_diff"] = max(nc1["worst_abs_diff"], worst)
nc1["pass"] = bool(nc1["n_checked"] > 0 and nc1["worst_abs_diff"] <= 1e-9)
log(f"NC1 vs TOURNAMENT-1 margins: {nc1['n_checked']} pairs, worst |Δ| = "
    f"{nc1['worst_abs_diff']:.3e} => {'PASS' if nc1['pass'] else 'CHECK'}")
out["NC1"] = nc1

# ============================================================ the floor, RE-MEASURED per pricing
log("measuring the split-half floor on MY pipeline (borrowing no constant)")
floors = {}
for p in PRICINGS:
    panel = np.vstack([mspc[p][b] for b in BOARD_NAMES])
    f = split_half_floor(panel, n_partitions=2000, rng=np.random.default_rng(BOOT_SEED), pct=90.0)
    f["floor_scaled_to_n25"] = scale_floor_to_n(f["floor"], f["half_n"], len(ALL_SEEDS))
    floors[p] = f
    log(f"  pricing {p:7s}: p90 floor {f['floor']:.4f} ms/char (half_n={f['half_n']}), "
        f"p50 {f['p50']:.4f}, scaled-to-n25 {f['floor_scaled_to_n25']:.4f}")
out["floors"] = floors

# ============================================================ sigma_diff from v02
v02 = json.loads((ART / "v02_sigma_and_flips.json").read_text())


def sigma_for(a: str, b: str, variant: str = "scoring_bucket_80", stat: str = "sigma_diff_rms") -> float:
    """v02 keys pairs as 'a|b' with a<b alphabetically; sigma_diff is symmetric in the pair."""
    pairs = v02["sigma_diff"][variant]["pairs"]
    k = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
    return float(pairs[k][stat])


# ============================================================ the four calibration bars, LOS_valid
log("=== the four registered calibration bars, for LOS_valid ===")
bars: dict = {}
rng = np.random.default_rng(BOOT_SEED)
LIVE_A, LIVE_B = "candidate", "flagship-c3"
SIG_LIVE = sigma_for(LIVE_A, LIVE_B)
FLOOR = floors["all"]["floor"]

# --- null-1: board vs itself, at MANY sigma_diff values (must be exactly 0.5 at every one)
n1 = []
for b in BOARD_NAMES:
    ms = mspc["all"][b]
    for s in (0.0, SIG_LIVE, 1.0, 5.0):
        r = compute_los(ms, ms.copy(), floor=FLOOR, a_name=b, b_name=b, sigma_diff=s)
        n1.append({"board": b, "sigma_diff": s, "los_valid": r.los_valid,
                   "los_design": r.los_design})
worst_n1 = max(abs(x["los_valid"] - 0.5) for x in n1)
bars["null_1"] = {"n_cases": len(n1), "worst_abs_dev_from_half": worst_n1,
                  "registered_bar": "LOS_valid = 0.5000 exactly",
                  "pass": bool(worst_n1 == 0.0), "cases": n1[:8]}
log(f"  null-1 board-vs-itself: worst |LOS_valid - 0.5| = {worst_n1:.3e} over {len(n1)} cases "
    f"=> {'PASS' if worst_n1 == 0.0 else 'FAIL'}")

# --- null-2: same-board split-half (truth is 0 by construction)
n2_vals, n2_dec = [], 0
N2 = 2000
for _ in range(N2):
    b = BOARD_NAMES[rng.integers(len(BOARD_NAMES))]
    ms = mspc["all"][b]
    perm = rng.permutation(ms.size)
    h = ms.size // 2
    r = compute_los(ms[perm[:h]], ms[perm[h:2 * h]], floor=FLOOR, sigma_diff=SIG_LIVE)
    n2_vals.append(r.los_valid)
    n2_dec += int(r.los_valid >= 0.95 or r.los_valid <= 0.05)
n2_vals = np.array(n2_vals)
bars["null_2"] = {"n_partitions": N2, "sigma_diff": SIG_LIVE,
                  "median": float(np.median(n2_vals)), "decided_rate": n2_dec / N2,
                  "registered_bar": "median in [0.45,0.55] and decided-rate <= 0.05",
                  "pass": bool(0.45 <= np.median(n2_vals) <= 0.55 and n2_dec / N2 <= 0.05)}
log(f"  null-2 split-half: median {np.median(n2_vals):.4f}, decided-rate {n2_dec / N2:.4f} "
    f"=> {'PASS' if bars['null_2']['pass'] else 'FAIL'}")

# --- null-3: permutation null on the LIVE pair's per-seed margins (sign-flip => truth is 0)
d_live = mspc["all"][LIVE_A] - mspc["all"][LIVE_B]
base = mspc["all"][LIVE_B]
N3 = 20000
n3_vals = np.empty(N3)
for i in range(N3):
    sgn = rng.choice((-1.0, 1.0), size=d_live.size)
    dd = d_live * sgn
    n3_vals[i] = compute_los(base + dd, base, floor=FLOOR, sigma_diff=SIG_LIVE).los_valid
p_gt = float((n3_vals > 0.95).mean())
bars["null_3"] = {"n_draws": N3, "sigma_diff": SIG_LIVE,
                  "median": float(np.median(n3_vals)), "p_los_gt_0.95": p_gt,
                  "p_los_lt_0.05": float((n3_vals < 0.05).mean()),
                  "registered_bar": "P(LOS_valid > 0.95) <= 0.05",
                  "pass": bool(p_gt <= 0.05)}
log(f"  null-3 permutation: median {np.median(n3_vals):.4f}, P(LOS_valid>0.95) = {p_gt:.5f} "
    f"=> {'PASS' if p_gt <= 0.05 else 'FAIL'}")

# --- known-big: every tuned board vs qwerty. THE BAR AT RISK (registered as such).
TUNED = [b for b in BOARD_NAMES if b not in ("qwerty", "dvorak", "colemak", "colemak-dh",
                                             "graphite", "semimak")]
kb = []
for b in TUNED:
    for p in PRICINGS:
        fl = floors[p]["floor"]
        s = sigma_for(b, "qwerty")
        r = compute_los(mspc[p][b], mspc[p]["qwerty"], floor=fl, a_name=b, b_name="qwerty",
                        sigma_diff=s)
        kb.append({"board": b, "pricing": p, "margin": r.mean_margin,
                   "margin_over_floor": r.margin_over_floor, "floor": fl, "sigma_diff": s,
                   "sem": r.sem_margin, "scale_valid": r.scale_valid,
                   "los_design": r.los_design, "los_valid": r.los_valid,
                   "los_typist": r.los_typist, "signs": f"{r.signs_a_faster}/{r.signs_b_faster}"})
min_kb = min(x["los_valid"] for x in kb)
bars["known_big"] = {"n_cases": len(kb), "min_los_valid": min_kb,
                     "min_los_design": min(x["los_design"] for x in kb),
                     "registered_bar": "LOS_valid >= 0.99",
                     "pass": bool(min_kb >= 0.99), "cases": kb}
log(f"  known-big tuned-vs-qwerty: min LOS_valid = {min_kb:.4f} over {len(kb)} board×pricing "
    f"=> {'PASS' if min_kb >= 0.99 else 'FAIL'}  (min LOS_design {bars['known_big']['min_los_design']:.4f})")
out["calibration_bars"] = bars

# ============================================================ NC3: strict-generalization check
nc3 = []
for p in PRICINGS:
    r0 = compute_los(mspc[p][LIVE_A], mspc[p][LIVE_B], floor=floors[p]["floor"], sigma_diff=0.0)
    nc3.append({"pricing": p, "los_design": r0.los_design, "los_valid_at_sigma0": r0.los_valid,
                "abs_diff": abs(r0.los_valid - r0.los_design),
                "bitwise_equal": r0.los_valid == r0.los_design})
out["NC3"] = {"cases": nc3, "all_bitwise_equal": all(c["bitwise_equal"] for c in nc3),
              "registered_bar": "sigma_diff=0 reproduces los_design (<=1e-12; bitwise expected)"}
log(f"NC3 sigma_diff=0 == los_design bitwise on all 3 pricings: "
    f"{out['NC3']['all_bitwise_equal']}")

# ============================================================ THE LIVE PAIR
log("=== THE LIVE PAIR: candidate vs flagship-c3, under MY error model ===")
live: dict = {"pair": f"{LIVE_A} vs {LIVE_B}", "pricings": {}}
for p in PRICINGS:
    fl = floors[p]["floor"]
    row = {}
    for variant in ("scoring_bucket_80", "all_buckets"):
        for stat in ("sigma_diff_rms", "sigma_diff_sd"):
            s = sigma_for(LIVE_A, LIVE_B, variant, stat)
            r = compute_los(mspc[p][LIVE_A], mspc[p][LIVE_B], floor=fl,
                            a_name=LIVE_A, b_name=LIVE_B, sigma_diff=s)
            row[f"{variant}/{stat}"] = r.as_row() | {"sigma_diff": s}
    r0 = compute_los(mspc[p][LIVE_A], mspc[p][LIVE_B], floor=fl, a_name=LIVE_A, b_name=LIVE_B)
    live["pricings"][p] = {"floor": fl, "baseline_no_sigma": r0.as_row(), "with_sigma": row}
    prim = row["scoring_bucket_80/sigma_diff_rms"]
    log(f"  {p:7s}: margin {r0.mean_margin:+.4f} ({r0.margin_over_floor:.2f}× floor {fl:.4f}), "
        f"sem {r0.sem_margin:.4f}, sigma_diff {prim['sigma_diff']:.4f} "
        f"=> LOS_design {r0.los_design:.4f}  LOS_valid {prim['LOS_valid']:.4f}  "
        f"LOS_typist {r0.los_typist:.4f}")

# the registered decision rule: the verdict CHANGES iff LOS_valid < 0.95 under ANY pricing
lv = [live["pricings"][p]["with_sigma"]["scoring_bucket_80/sigma_diff_rms"]["LOS_valid"]
      for p in PRICINGS]
live["registered_decision_rule"] = "verdict CHANGES iff LOS_valid < 0.95 under ANY of the 3 pricings"
live["min_los_valid"] = float(min(lv))
live["verdict_changes"] = bool(min(lv) < 0.95)
log(f"  => min LOS_valid over pricings = {min(lv):.4f}; VERDICT CHANGES = {live['verdict_changes']}")
out["live_pair"] = live

# ============================================================ the full matrix, LOS_valid
log("=== full 78-pair matrix under LOS_valid (pricing 'all') ===")
import itertools  # noqa: E402

matrix = []
for a, b in itertools.combinations(BOARD_NAMES, 2):
    s = sigma_for(a, b)
    r = compute_los(mspc["all"][a], mspc["all"][b], floor=FLOOR, a_name=a, b_name=b, sigma_diff=s)
    matrix.append(r.as_row())
dec_design = sum(1 for m in matrix if m["LOS_design"] >= 0.95 or m["LOS_design"] <= 0.05)
dec_valid = sum(1 for m in matrix if m["LOS_valid"] >= 0.95 or m["LOS_valid"] <= 0.05)
dec_seed = sum(1 for m in matrix if m["LOS_seed"] >= 0.95 or m["LOS_seed"] <= 0.05)
dec_typ = sum(1 for m in matrix if m["LOS_typist"] >= 0.95 or m["LOS_typist"] <= 0.05)
out["matrix"] = {"rows": matrix, "n_pairs": len(matrix),
                 "decided_seed": dec_seed, "decided_design": dec_design,
                 "decided_valid": dec_valid, "decided_typist": dec_typ}
log(f"  decided: LOS_seed {dec_seed}/{len(matrix)}  LOS_design {dec_design}/{len(matrix)}  "
    f"LOS_valid {dec_valid}/{len(matrix)}  LOS_typist {dec_typ}/{len(matrix)}")

dump("v03_apply.json", out)
log("DONE")
