"""INVARIANT A (the correction + its 5 proof gates) and INVARIANT C (the 1v1 with TWO floors).

DELTA is read from c02_contrast.json -- measured, never assumed. Runs the additive arm and the
multiplicative robustness arm. Then the full 13-board pairwise 1v1 at n=25 off the rescued
layout-independent tables, with FLOOR-S (split-half placebo) and FLOOR-D (bootstrap of DELTA
propagated through the corrected margin).
"""
import itertools
import json
import sys
import time

import numpy as np
from _guard import ART, FIELD_ORDER, SEEDS, assert_d5, build_boards

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

import surface  # noqa: E402
from scipy import stats  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

BOARDS = build_boards()
C02 = json.load(open(f"{ART}/c02_contrast.json"))
RNG = np.random.default_rng(20260803)      # fixed, registered

RAW_PEN = C02["e3_raw"]["penalty"]
MOD_PEN = C02["e3_model"]["penalty"]
DELTA = RAW_PEN - MOD_PEN
MULT = RAW_PEN / MOD_PEN if MOD_PEN else float("nan")
log(f"MEASURED from c02: raw {RAW_PEN:+.4f}  model {MOD_PEN:+.4f}  =>  DELTA = {DELTA:+.4f} ms")
log(f"  multiplicative arm factor on same-finger cells = {MULT:.6f}")

MASK = surface.same_finger_mask()
log(f"same-finger cells to be surcharged: {int(MASK.sum())} of {MASK.size}")

log("loading the 25 per-seed table pairs")
T2s, Tcs = surface.load_all_seed_tables(verbose=False)
cdir, tri_freq = surface.corpus(None)
ARR = {nm: surface.board_arrays(BOARDS[nm], tri_freq) for nm in FIELD_ORDER}

# ------------------------------------------------------------------------ GATE A2: only those cells
T2c = [surface.corrected_T2(t, DELTA, MASK, "add") for t in T2s]
d = T2c[0] - T2s[0]
a2 = {"n_cells_changed": int((d != 0).sum()), "n_mask_cells": int(MASK.sum()),
      "all_changed_by_delta": bool(np.allclose(d[MASK], DELTA, atol=0, rtol=0)),
      "max_abs_change_outside_mask": float(np.abs(d[~MASK]).max()),
      "cells_match_mask_exactly": bool(((d != 0) == MASK).all())}
log(f"A2: {a2['n_cells_changed']} cells changed (mask has {a2['n_mask_cells']}); "
    f"exact-mask match {a2['cells_match_mask_exactly']}; all by exactly DELTA "
    f"{a2['all_changed_by_delta']}; max change OUTSIDE mask {a2['max_abs_change_outside_mask']:.3e}")

# ------------------------------------------------------- GATE A3: analytic prediction of the shift
sf_share = {nm: surface.sf_share(ARR[nm], MASK) for nm in FIELD_ORDER}
X_old = {nm: np.array([surface.mspc(ARR[nm], T2s[s], Tcs[s]) for s in range(25)]) for nm in FIELD_ORDER}
X_new = {nm: np.array([surface.mspc(ARR[nm], T2c[s], Tcs[s]) for s in range(25)]) for nm in FIELD_ORDER}
a3 = {}
worst_a3 = 0.0
for nm in FIELD_ORDER:
    require_finite(X_new[nm].tolist(), f"corrected per-seed {nm}")
    observed = float(X_new[nm].mean() - X_old[nm].mean())
    predicted = DELTA * sf_share[nm]
    err = abs(observed - predicted)
    worst_a3 = max(worst_a3, err)
    a3[nm] = {"sf_share": sf_share[nm], "predicted_shift": predicted,
              "observed_shift": observed, "abs_err": err}
    log(f"A3 {nm:12s} sf-share {sf_share[nm]:.6f}  predicted {predicted:+.4f}  "
        f"observed {observed:+.4f}  |err| {err:.3e}")
log(f"A3 WORST |observed - analytic| = {worst_a3:.3e}  (the surcharge is analytically auditable)")

# ------------------------------------- GATE A4: through the SEARCH's own evaluator (from_table)
from keybo.analysis.surfaces import C30M  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.scoring.table_trigram import TableTrigramScorer  # noqa: E402

log("A4: building the search evaluator on the corrected seed-MEAN surface")
surf = TimeSurface(tri_freq, target_wpm=surface.WPM)
T2m_old, Tcm = surf._T2, surf._Tc
T2m_new = surface.corrected_T2(T2m_old, DELTA, MASK, "add")
def make_scorer(T2m):
    sc = TableTrigramScorer.from_table(T2m[:, :, None] + Tcm, surf.tri, chars=C30M, geometry=G30)
    sc._covered = float(sc._f.sum())
    return sc
sc_old, sc_new = make_scorer(T2m_old), make_scorer(T2m_new)
a4 = {}
for nm in FIELD_ORDER:
    lay = BOARDS[nm]
    if set(lay) != set(C30M):
        a4[nm] = {"skipped": "not a C30M permutation (table path refuses it)"}
        continue
    L = Layout(lay, G30)
    ev_old = sc_old.fitness(L) / sc_old._covered
    ev_new = sc_new.fitness(L) / sc_new._covered
    card_old = surf.card(lay).ms_per_char
    a4[nm] = {"evaluator_old": ev_old, "shipped_card": card_old,
              "parity_rel_dev": abs(ev_old - card_old) / card_old,
              "evaluator_new": ev_new, "evaluator_shift": ev_new - ev_old,
              "analytic_shift": DELTA * sf_share[nm]}
    log(f"A4 {nm:12s} evaluator_old={ev_old:.9f} vs shipped card={card_old:.9f} "
        f"rel={a4[nm]['parity_rel_dev']:.3e}  shift {ev_new - ev_old:+.4f} "
        f"(analytic {DELTA * sf_share[nm]:+.4f})")
worst_parity = max(v["parity_rel_dev"] for v in a4.values() if "parity_rel_dev" in v)
log(f"A4 WORST parity rel dev vs shipped card() = {worst_parity:.3e}")

# ------------------------------------------------------------------- A1 is answered in c02+c04
# (re-running the pick2 contrast on the corrected T2 needs the stroke frame; done in c04.)

# ------------------------------------------------------- the multiplicative robustness arm (shape)
T2cm = [surface.corrected_T2(t, MULT, MASK, "mul") for t in T2s]
X_mul = {nm: np.array([surface.mspc(ARR[nm], T2cm[s], Tcs[s]) for s in range(25)])
         for nm in FIELD_ORDER}

# =============================================================== INVARIANT C: the 1v1 machinery ==
PAIRS = list(itertools.combinations(FIELD_ORDER, 2))


def floor_s(X, n_part=2000, half=12):
    """FLOOR-S: split-half same-board placebo. Truth is EXACTLY 0 by construction."""
    vals = []
    for nm in FIELD_ORDER:
        x = X[nm]
        for _ in range(n_part):
            p = RNG.permutation(len(SEEDS))
            vals.append(abs(x[p[:half]].mean() - x[p[half:2 * half]].mean()))
    v = np.array(vals)
    return {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()), "n": len(v),
            "half_n": half}


def run_matrix(X, floor, label):
    rows = []
    for A, B in PAIRS:
        dd = X[A] - X[B]
        n = len(dd)
        mean = float(dd.mean()); sd = float(dd.std(ddof=1)); sem = sd / np.sqrt(n)
        t, p = stats.ttest_rel(X[A], X[B])
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
        pos, neg = int((dd > 0).sum()), int((dd < 0).sum())
        rows.append({"A": A, "B": B, "mean": mean, "sd": sd, "t": float(t), "p_raw": float(p),
                     "ci": [float(ci[0]), float(ci[1])], "signs_pos_neg": [pos, neg],
                     "abs_mean": abs(mean), "over_floor": abs(mean) / floor})
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    m = len(rows); ok = True
    for rank, i in enumerate(order):
        thr = 0.05 / (m - rank)
        rows[i]["holm_thr"] = thr
        rows[i]["holm_reject"] = bool(ok and rows[i]["p_raw"] < thr)
        if not rows[i]["holm_reject"]:
            ok = False
    for r in rows:
        r["above_floor"] = bool(r["abs_mean"] >= floor)
        r["sign_ok"] = bool(max(r["signs_pos_neg"]) >= 20)
        if r["holm_reject"] and r["above_floor"] and r["sign_ok"]:
            r["verdict"] = r["A"] if r["mean"] < 0 else r["B"]; r["kind"] = "WIN"
        elif r["ci"][0] > -floor and r["ci"][1] < floor:
            r["verdict"] = "TIED"; r["kind"] = "TIED"
        else:
            r["verdict"] = "UNRESOLVED"; r["kind"] = "UNRESOLVED"
    nw = sum(1 for r in rows if r["kind"] == "WIN")
    nt = sum(1 for r in rows if r["kind"] == "TIED")
    nu = sum(1 for r in rows if r["kind"] == "UNRESOLVED")
    log(f"  {label:22s} floor={floor:.4f}  WIN={nw} TIED={nt} UNRESOLVED={nu}")
    return rows


log("")
log("=== INVARIANT C: floors ============================================================")
FS_old = floor_s(X_old); FS_new = floor_s(X_new)
log(f"FLOOR-S UNCORRECTED p50={FS_old['p50']:.4f} p90={FS_old['p90']:.4f} p99={FS_old['p99']:.4f} "
    f"max={FS_old['max']:.4f}   (tournament published p90=0.2921)")
log(f"FLOOR-S CORRECTED   p50={FS_new['p50']:.4f} p90={FS_new['p90']:.4f} p99={FS_new['p99']:.4f} "
    f"max={FS_new['max']:.4f}")

# ---- FLOOR-D: bootstrap DELTA itself and propagate through the corrected margin ----
log("FLOOR-D: bootstrapping DELTA over position pairs and propagating it (this is the floor")
log("         that gates the headline; nobody in this campaign has measured it)")
same = C02["pairs_same"]; other = C02["pairs_other"]
rs = np.array([r["raw"] for r in same]); ro = np.array([r["raw"] for r in other])
ps_ = np.array([r["pred"] for r in same]); po_ = np.array([r["pred"] for r in other])
BD = 2000
deltas = []
rng2 = np.random.default_rng(4242)
for _ in range(BD):
    i_s = rng2.integers(0, len(rs), len(rs)); i_o = rng2.integers(0, len(ro), len(ro))
    rp = float(np.median(rs[i_s]) - np.median(ro[i_o]))
    mp = float(np.median(ps_[i_s]) - np.median(po_[i_o]))
    deltas.append(rp - mp)
deltas = np.array(deltas)
log(f"  DELTA bootstrap: point {DELTA:+.4f}  mean {deltas.mean():+.4f}  sd {deltas.std(ddof=1):.4f}  "
    f"CI95 [{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]")

# propagate: the corrected margin for a pair is analytic in DELTA (gate A3 proves it), so
# margin_corrected(A,B; d) = margin_uncorrected(A,B) + d * (sf_share[A] - sf_share[B]).
KEY = ("candidate", "keybo-lsb")
base_margin = float((X_old[KEY[0]] - X_old[KEY[1]]).mean())
dshare = sf_share[KEY[0]] - sf_share[KEY[1]]
marg_boot = base_margin + deltas * dshare
FD = {"delta_point": DELTA, "delta_boot_sd": float(deltas.std(ddof=1)),
      "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
      "headline_pair": list(KEY), "uncorrected_margin": base_margin,
      "d_sf_share": float(dshare),
      "corrected_margin_point": float(base_margin + DELTA * dshare),
      "corrected_margin_boot_sd": float(marg_boot.std(ddof=1)),
      "corrected_margin_ci95": [float(np.percentile(marg_boot, 2.5)),
                                float(np.percentile(marg_boot, 97.5))],
      "floor_D_p90_of_shift": float(np.percentile(np.abs(deltas * dshare), 90)),
      "max_abs_shift_over_boot": float(np.abs(deltas * dshare).max()),
      "n_boot": BD}
log(f"  d(sf-share) candidate-minus-keybo-lsb = {dshare:+.6f}")
log(f"  uncorrected margin {base_margin:+.4f} -> corrected {FD['corrected_margin_point']:+.4f} "
    f"(shift {DELTA * dshare:+.4f})")
log(f"  FLOOR-D = p90|shift| under DELTA's own uncertainty = {FD['floor_D_p90_of_shift']:.4f}")

# ---- THE FLIP POINT: solve for the delta at which candidate loses, rather than test one point --
# Gate A3 proves margin_corrected(A,B; d) = margin_uncorrected(A,B) + d*(sf_share[A]-sf_share[B])
# EXACTLY. So the whole one-parameter family is available in closed form, and the honest question
# is not "does candidate survive at d=DELTA" but "how large must d be before it does not" -- a
# quantity a single point estimate cannot express. Verified numerically below, not just asserted.
def margin_at(A, B, d):
    return float((X_old[A] - X_old[B]).mean()) + d * (sf_share[A] - sf_share[B])


flip = {}
for rival in FIELD_ORDER:
    if rival == "candidate":
        continue
    m0 = margin_at("candidate", rival, 0.0)
    ds = sf_share["candidate"] - sf_share[rival]
    # candidate loses when margin > 0 (margin = candidate - rival; negative = candidate faster)
    if ds <= 0:                     # the surcharge helps candidate against this rival, forever
        flip[rival] = {"m0": m0, "d_share": ds, "flip_delta": None,
                       "note": "candidate has the LOWER same-finger share -> surcharge never flips this"}
        continue
    d_flip = -m0 / ds
    flip[rival] = {"m0": m0, "d_share": ds, "flip_delta": float(d_flip),
                   "flip_multiple_of_DELTA": float(d_flip / DELTA),
                   "note": "delta (ms) at which candidate stops beating this rival"}
log("")
log("=== THE FLIP POINT (closed form; A3 proves the margin is exactly linear in delta) ===")
for rival in FIELD_ORDER:
    if rival == "candidate":
        continue
    f = flip[rival]
    if f["flip_delta"] is None:
        log(f"  vs {rival:12s} m0={f['m0']:+.4f} d_share={f['d_share']:+.6f}  NEVER FLIPS "
            f"(candidate's same-finger share is lower)")
    else:
        log(f"  vs {rival:12s} m0={f['m0']:+.4f} d_share={f['d_share']:+.6f}  flips at delta="
            f"{f['flip_delta']:+.1f} ms = {f['flip_multiple_of_DELTA']:.1f}x the measured DELTA")
# numeric verification of the closed form at 3 deltas, so it is checked and not merely derived
ver = []
for dtest in (10.0, DELTA, 100.0):
    T2t = [surface.corrected_T2(t, dtest, MASK, "add") for t in T2s]
    for A, B in (("candidate", "keybo-lsb"), ("candidate", "arm-B")):
        num = float(np.mean([surface.mspc(ARR[A], T2t[s], Tcs[s])
                             - surface.mspc(ARR[B], T2t[s], Tcs[s]) for s in range(25)]))
        ver.append({"delta": dtest, "pair": f"{A} vs {B}", "numeric": num,
                    "closed_form": margin_at(A, B, dtest),
                    "abs_err": abs(num - margin_at(A, B, dtest))})
log(f"  closed-form check: worst |numeric - analytic| over 6 cases = "
    f"{max(v['abs_err'] for v in ver):.3e}")
flip["_verification"] = ver

log("")
log("=== INVARIANT C: the matrices (margin-vs-floor FIRST, p SECOND) ====================")
M_old = run_matrix(X_old, FS_old["p90"], "UNCORRECTED")
M_new = run_matrix(X_new, FS_new["p90"], "CORRECTED (additive)")
M_mul = run_matrix(X_mul, floor_s(X_mul)["p90"], "CORRECTED (multiplicative)")

def summarize(rows, X, label):
    means = {nm: float(X[nm].mean()) for nm in FIELD_ORDER}
    order = sorted(FIELD_ORDER, key=lambda n: means[n])
    losses = {nm: [] for nm in FIELD_ORDER}
    for r in rows:
        if r["kind"] == "WIN":
            w = r["verdict"]; l = r["B"] if w == r["A"] else r["A"]
            losses[l].append(w)
    log(f"  {label}: rank order = {' < '.join(order[:6])} ...")
    log(f"    candidate: {len(losses['candidate'])} losses "
        f"{losses['candidate'] if losses['candidate'] else '(none)'}; rank "
        f"{order.index('candidate') + 1}/13")
    return {"means": means, "order": order, "losses": losses}

S_old = summarize(M_old, X_old, "UNCORRECTED")
S_new = summarize(M_new, X_new, "CORRECTED-add")
S_mul = summarize(M_mul, X_mul, "CORRECTED-mul")

out = {"delta": DELTA, "mult_factor": MULT, "raw_pen": RAW_PEN, "mod_pen": MOD_PEN,
       "gates": {"A2": a2, "A3": a3, "A3_worst_err": worst_a3, "A4": a4,
                 "A4_worst_parity": worst_parity},
       "sf_share": sf_share,
       "mspc": {"uncorrected": {nm: X_old[nm].tolist() for nm in FIELD_ORDER},
                "corrected_add": {nm: X_new[nm].tolist() for nm in FIELD_ORDER},
                "corrected_mul": {nm: X_mul[nm].tolist() for nm in FIELD_ORDER}},
       "floor_S": {"uncorrected": FS_old, "corrected_add": FS_new},
       "floor_D": FD, "flip_point": flip,
       "matrix": {"uncorrected": M_old, "corrected_add": M_new, "corrected_mul": M_mul},
       "summary": {"uncorrected": S_old, "corrected_add": S_new, "corrected_mul": S_mul},
       "wall_s": time.time() - t0}
json.dump(out, open(f"{ART}/c03_correct.json", "w"), indent=1)
log(f"wrote {ART}/c03_correct.json")
log("ALL-DONE")
