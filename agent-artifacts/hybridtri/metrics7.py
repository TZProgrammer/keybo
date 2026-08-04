"""HYBRIDB-1 §3 — hybrid-B on ALL SEVEN of INTERPFRAME-1's registered bars, not just M1.

⚠ THIS IS THE CORRECTION TO MY BRIEF, registered in the prereg before measuring. The brief says
hybrid-B "clears INTERPFRAME-1's MAXCORR bar" and treats that as clearing the interpretability half.
It clears ONE of the SEVEN bars INTERPFRAME-1 registered. hybrid-B was screened on M1 alone because
EXPLOIT-1's §g screen was model-free and the other six need a trained model. I have the trained
model, so I measure all seven.

Instrument: INTERPFRAME-1's OWN metrics.py, loaded BY PATH (a plain `import metrics` after a
sys.path.insert picks up that directory's `_boot.py`, which shadows mine and pins the wrong
worktree). Weighting grid: flagship-c3, INTERPFRAME-1's own -- a MAXCORR read on qwerty-C30M gives
0.9556 vs the published 0.9813 purely from the grid, so the grid is part of the instrument.

M3's same-property grouping for the new frame is REGISTERED in the prereg §3 (interp.1's three
groups unchanged, PLUS the served frame's two groups verbatim, PLUS the three CROSS groups that are
the whole point of a hybrid). It is defined HERE rather than taken from metrics.py, because
metrics.py's `same_property_groups` dispatches on `"hand_conflict" in names` and would silently
hand hybrid-B interp.1's grouping -- missing every ordinal-vs-one-hot pair, i.e. exactly the
conflicts a hybrid can have and interp.1 cannot.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, SCRATCH, assert_tree, load_by_path, require  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from keybo.analysis.shap_diff import _shap_tables  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_HYBRIDB_FEATURE_NAMES,
    BIGRAM_HYBRIDB_MONOTONE,
    FEATURE_VERSION_HYBRIDB,
    hybridb_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402

M = load_by_path(
    "interpframe_metrics_m7",
    "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/metrics.py",
)
for _n in ("m1_maxcorr", "m2_constfrac", "m3_splitpairs", "m4_monofrac", "sign_agreement"):
    require(M, _n)

WPM = 90.0
SEEDS = (0, 1, 2)
GEO_TRAIN = ROW_STAGGERED_31  # the models are trained here (the K31 tables carry the quote slot)
GEO_SERVE = ROW_STAGGERED_30  # the boards being compared are 30-character
NAMES = list(BIGRAM_HYBRIDB_FEATURE_NAMES)
MONO = dict(zip(NAMES, BIGRAM_HYBRIDB_MONOTONE, strict=True))
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


# ⚠ The TRAIN and SERVE geometries DIFFER, and that is the shipped arrangement (INTERPFRAME-1's
# poc.py asserted slot EQUALITY here first and it fired -- equality is the wrong invariant). The
# right one: G30's grid is a SUBSET of the trained-on grid under the SAME stagger and space pinning,
# so no served cell is off the distribution the model saw.
assert set(GEO_SERVE.slots) <= set(GEO_TRAIN.slots), "every served slot must be one trained on"
assert GEO_SERVE.row_offsets == GEO_TRAIN.row_offsets, "stagger must match or dx/distance shift"
assert GEO_SERVE.space_position == GEO_TRAIN.space_position, "space must be pinned identically"
TRAIN_POS = [*GEO_TRAIN.slots, GEO_TRAIN.space_position]
SERVE_POS = [*GEO_SERVE.slots, GEO_SERVE.space_position]
log(
    f"grids: TRAIN {len(TRAIN_POS)}^2={len(TRAIN_POS) ** 2}  SERVE {len(SERVE_POS)}^2={len(SERVE_POS) ** 2}"
)

# =========================================================================================
# THE REGISTERED SAME-PROPERTY GROUPING for hybrid-B (prereg §3), defined here on purpose.
# =========================================================================================
ROWS = ["bottom", "home", "top"]
FINGERS = ["pinky", "ring", "middle", "index", "lateral"]
HYBRIDB_SAME_PROPERTY = [
    # interp.1's three, UNCHANGED
    {"row_load", "row_arrival", "bottom_bias"},
    {"row_span", "lateral_span", "same_hand_travel"},
    {"hand_conflict", "finger_load", "off_home_column"},
    # the served frame's two, VERBATIM
    set(ROWS),
    set(FINGERS),
    # the CROSS groups: an ordinal and the one-hots it was BUILT TO REPLACE are the same property
    {"bottom_bias", *ROWS},
    {"finger_load", *FINGERS},
    {"off_home_column", "lateral"},
]


def hybridb_groups(names):
    ns = set(names)
    return [g & ns for g in HYBRIDB_SAME_PROPERTY]


# =========================================================================================
# Load the three hybrid-B models, ASSERTING the frame per seed (a filename is not a provenance).
# =========================================================================================
models = []
for s in SEEDS:
    path = f"{SCRATCH}/hybridb_mono_seed{s}.json"
    m = XGBoostTypingModel.load(path, expected_feature_version=FEATURE_VERSION_HYBRIDB)
    if list(m.metadata.feature_names) != NAMES:
        raise SystemExit(f"ABORT columns seed{s}: {m.metadata.feature_names}")
    rec = (m.metadata.extra.get("training") or {}).get("interp_frame") or {}
    want = list(BIGRAM_HYBRIDB_MONOTONE)
    if rec.get("frame") != "hybrid-b" or list(rec.get("monotone_constraints") or ()) != want:
        raise SystemExit(f"ABORT frame record seed{s}: {rec}")
    models.append(m)
    log(f"loaded hybrid-B seed{s}: {m.metadata.feature_version}, {len(NAMES)} cols, mono recorded")

out: dict = {
    "prereg": "agent-artifacts/hybridtri/HYBRIDTRI-preregistration.md @ 5a5d3c3 §3",
    "frame": "hybrid-b",
    "n_columns": len(NAMES),
    "monotone": {n: MONO[n] for n in NAMES},
    "registered_same_property_groups": [sorted(g) for g in HYBRIDB_SAME_PROPERTY],
}

# =========================================================================================
# The corpus-weighted SERVE grid + the attribution (INTERPFRAME-1's own convention).
# =========================================================================================
surface = default_surface(WPM, None)
from keybo.analysis.shap_diff import _char_weight_tables  # noqa: E402

_, LAY_A = _resolve("flagship-c3")
_, LAY_B = _resolve("graphite")
slot = surface._slot_of(LAY_A)
_w3, _w2, _covered = _char_weight_tables(surface, LAY_A)
NPs = len(SERVE_POS)
perm = np.array([slot[c] for c in LAY_A] + [slot[" "]], dtype=np.intp)
wp = np.zeros((NPs, NPs))
np.add.at(wp, (perm[:, None], perm[None, :]), _w2)
W = wp.ravel()
log(f"[grid] weighting on flagship-c3 = {LAY_A}   covered {_covered:,.0f}")

X_SERVE = np.vstack(
    [
        hybridb_features_from_positions(GEO_SERVE, (a, b), wpm=WPM)
        for a in SERVE_POS
        for b in SERVE_POS
    ]
)

# The ATTRIBUTION: the shipped shap_diff on frame="hybridb". Only reachable because the frame is
# registered in FRAMES + _HYBRIDB_BLOCKS; block_map refuses an unknown frame by design.
from keybo.analysis.shap_diff import shap_diff  # noqa: E402

res = shap_diff(
    LAY_A, LAY_B, channel="t2", target_wpm=WPM, frame="hybridb", bigram_models=tuple(models)
)
if res.t2 is None:
    raise SystemExit("ABORT: shap_diff returned no t2 channel")
order = [c.feature for c in res.t2.contributions]
idx = {n: i for i, n in enumerate(order)}
attrib = np.array([res.t2.contributions[idx[n]].ms_per_char for n in NAMES], dtype=np.float64)
log(f"attribution: gap_t2 {res.t2.gap:+.6f} ms/char over {len(NAMES)} columns")
out["gap_t2_ms_per_char"] = float(res.t2.gap)
out["residuals"] = {
    "resid_cell_lmdi": float(res.t2.resid_cell_lmdi),
    "resid_feature_sum": float(res.t2.resid_feature_sum),
    "resid_gap_vs_shipped": float(res.t2.resid_gap_vs_shipped),
}
log(
    f"  INTERNAL resid cell_lmdi {out['residuals']['resid_cell_lmdi']:.3e}  "
    f"feature_sum {out['residuals']['resid_feature_sum']:.3e}  "
    f"EXTERNAL vs own predict-side table {out['residuals']['resid_gap_vs_shipped']:.3e}"
)

# =========================================================================================
# M1 / M1b -- MAXCORR / MEANCORR
# =========================================================================================
m1 = M.m1_maxcorr(X_SERVE, W, NAMES)
out["M1"] = m1
log("")
log(f"M1  MAXCORR  {m1['maxcorr']:.4f}  (bar <= 0.7850)  worst pair {m1['worst_pair']}")
log(
    f"M1b MEANCORR {m1['meancorr']:.4f}   pairs>0.9 {m1['n_pairs_over_0.9']}  "
    f"pairs>0.7 {m1['n_pairs_over_0.7']}  of {m1['n_pairs']}"
)

# =========================================================================================
# M2 -- CONSTFRAC (attribution mass on columns CONSTANT on the weighted grid)
# =========================================================================================
m2 = M.m2_constfrac(X_SERVE, W, NAMES, attrib)
out["M2"] = m2
log(
    f"M2  CONSTFRAC {m2['constfrac']:.6f}  (bar == 0 exactly)  constant cols {m2['constant_columns']}"
)

# =========================================================================================
# M3 -- SPLITPAIRS on the REGISTERED grouping
# =========================================================================================
_saved = M.same_property_groups
M.same_property_groups = hybridb_groups  # the registered grouping, not interp.1's
try:
    m3 = M.m3_splitpairs(NAMES, attrib)
finally:
    M.same_property_groups = _saved
out["M3"] = m3
log(
    f"M3  SPLITPAIRS {m3['splitpairs']}  (bar < 7)   conflict mass "
    f"{m3['conflict_mass_ms_per_char']:.4f} ms/char"
)
for p in m3["pairs"][:6]:
    log(
        f"       {p['a']} {p['ms_a']:+.4f}  vs  {p['b']} {p['ms_b']:+.4f}   conflict {p['conflict_ms']:.4f}"
    )
# ⚠ Reported BOTH ways, because the grouping is a choice and it is decision-bearing: on interp.1's
# grouping (which metrics.py would have picked silently) the CROSS pairs are invisible.
_saved = M.same_property_groups
M.same_property_groups = lambda names: [g & set(names) for g in HYBRIDB_SAME_PROPERTY[:3]]
try:
    m3_interp_only = M.m3_splitpairs(NAMES, attrib)
finally:
    M.same_property_groups = _saved
out["M3_on_interp1_grouping_only"] = m3_interp_only
log(
    f"    (on interp.1's 3 groups only -- what metrics.py would have picked silently: "
    f"{m3_interp_only['splitpairs']} pairs / {m3_interp_only['conflict_mass_ms_per_char']:.4f} ms)"
)

# =========================================================================================
# M4 -- MONOFRAC. The honored verdict needs all THREE checks (INTERPFRAME-1 §5): the booster
# respects the constraint, the SHAP relationship has the right sign, and the column is NOT DEAD
# (the ADJ-2 trap: a constrained column can learn exactly zero magnitude).
# =========================================================================================
log("")
log("M4 prerequisites: (a) booster sweep, (b) SHAP-level sign, (c) non-degeneracy")
X_TRAIN_GRID = np.vstack(
    [
        hybridb_features_from_positions(GEO_TRAIN, (a, b), wpm=WPM)
        for a in TRAIN_POS
        for b in TRAIN_POS
    ]
)


def booster_sweep(models, names, mono_map, grid):
    """Sweep each column alone from min to max, others at the grid MEDIAN.

    Reports the worst VIOLATION and the total RANGE. The range is what catches a constraint honored
    only because the column was never split on (the ADJ-2 trap), which a violation check cannot see.
    """
    base = np.median(grid, axis=0)
    res = {}
    for j, name in enumerate(names):
        vals = np.unique(grid[:, j])
        if len(vals) > 25:
            vals = np.unique(np.quantile(grid[:, j], np.linspace(0, 1, 25)))
        Xs = np.tile(base, (len(vals), 1))
        Xs[:, j] = vals
        pred = np.mean([m.predict(Xs) for m in models], axis=0)
        d = np.diff(pred)
        want = mono_map.get(name, 0)
        viol = float(-d.min()) if want > 0 else (float(d.max()) if want < 0 else 0.0)
        res[name] = {
            "constraint": want,
            "n_grid": int(len(vals)),
            "response_range": float(pred.max() - pred.min()),
            "worst_violation": max(viol, 0.0),
            "monotone_ok": bool(viol <= 1e-9),
        }
    return res


sweep = booster_sweep(models, NAMES, MONO, X_TRAIN_GRID)
out["booster_sweep"] = sweep

tables = _shap_tables(tuple(models), GEO_SERVE, WPM, 2, "hybridb")
shap = np.mean(tables[0], axis=0).reshape(-1, len(tables[5]))
if list(tables[5]) != NAMES:
    raise SystemExit(f"ABORT: shap table columns {tables[5]} != {NAMES}")
if shap.shape[0] != X_SERVE.shape[0]:
    raise SystemExit(f"ABORT: grid mismatch SHAP {shap.shape} vs features {X_SERVE.shape}")
shap_mono = {}
for j, name in enumerate(NAMES):
    col, sv = X_SERVE[:, j], shap[:, j]
    rho = float(spearmanr(col, sv).statistic) if float(np.ptp(col)) > 0 else float("nan")
    shap_mono[name] = {
        "rho_value_vs_own_shap": rho,
        "mean_abs_shap": float(np.abs(sv).mean()),
        "constraint": MONO[name],
    }
out["shap_mono"] = shap_mono

DEAD_ABS_SHAP = 1e-6
honored, detail = {}, {}
for n in NAMES:
    want = MONO[n]
    rho = shap_mono[n]["rho_value_vs_own_shap"]
    ok_booster = sweep[n]["monotone_ok"]
    ok_shap = (rho >= -1e-12) if want > 0 else ((rho <= 1e-12) if want < 0 else None)
    alive = shap_mono[n]["mean_abs_shap"] > DEAD_ABS_SHAP
    # An UNCONSTRAINED column (want == 0) is NOT honored -- there is nothing to honor. That is what
    # makes M4 fall short by construction on this frame, and it is registered in the prereg.
    honored[n] = bool(want != 0 and ok_booster and ok_shap and alive)
    detail[n] = {
        "constraint": want,
        "booster_ok": ok_booster,
        "shap_rho": rho,
        "shap_sign_ok": ok_shap,
        "mean_abs_shap": shap_mono[n]["mean_abs_shap"],
        "alive": alive,
        "honored": honored[n],
    }
out["honored_detail"] = detail
m4 = M.m4_monofrac(NAMES, attrib, honored)
out["M4"] = m4
log("")
log(
    f"{'column':<18} {'con':>4} {'range':>10} {'viol':>10} {'shap rho':>10} {'mean|shap|':>11} {'honored':>8} {'attrib ms':>10}"
)
for n in NAMES:
    d = detail[n]
    log(
        f"{n:<18} {d['constraint']:>+4d} {sweep[n]['response_range']:>10.5f} "
        f"{sweep[n]['worst_violation']:>10.2e} "
        f"{d['shap_rho']:>10.4f} {d['mean_abs_shap']:>11.3e} {str(d['honored']):>8} "
        f"{attrib[NAMES.index(n)]:>+10.4f}"
    )
log("")
log(
    f"M4  MONOFRAC {m4['monofrac']:.4f}  (bar >= 0.90)  "
    f"-- REGISTERED to fail: 8 of 18 columns carry no constraint"
)
# The magnitude that is NOT predetermined: how much mass the unconstrained one-hots attract.
unc = [n for n in NAMES if MONO[n] == 0]
total_abs = float(np.abs(attrib).sum())
mass_unc = float(sum(abs(attrib[NAMES.index(n)]) for n in unc))
out["unconstrained_mass"] = {
    "columns": unc,
    "abs_mass_ms": mass_unc,
    "share_of_total_abs": mass_unc / total_abs if total_abs else None,
    "total_abs_ms": total_abs,
}
log(
    f"    the number that was NOT predetermined: the 8 UNCONSTRAINED one-hots attract "
    f"{mass_unc:.4f} of {total_abs:.4f} ms |attrib| = "
    f"{100 * mass_unc / total_abs:.1f}%"
)

# =========================================================================================
# M5 -- SIGNSTAB: cross-corpus sign agreement + rho.  M6 -- SEEDSTAB: 3-seed unanimity.
# =========================================================================================
res_iweb = shap_diff(
    LAY_A,
    LAY_B,
    channel="t2",
    target_wpm=WPM,
    frame="hybridb",
    bigram_models=tuple(models),
    corpus="iweb",
)
order2 = [c.feature for c in res_iweb.t2.contributions]
idx2 = {n: i for i, n in enumerate(order2)}
attrib_iweb = np.array([res_iweb.t2.contributions[idx2[n]].ms_per_char for n in NAMES])
m5 = M.sign_agreement(attrib, attrib_iweb)
out["M5"] = m5
log("")
log(
    f"M5  SIGNSTAB {m5['sign_agree_frac']:.4f} (bar >= 0.9000)   rho {m5['rho']:.4f} "
    f"(bar >= 0.8737)   live_either {m5['n_live_either']} live_both {m5['n_live_both']}"
)

per_seed = []
for m in models:
    r = shap_diff(LAY_A, LAY_B, channel="t2", target_wpm=WPM, frame="hybridb", bigram_models=(m,))
    o = [c.feature for c in r.t2.contributions]
    ix = {n: j for j, n in enumerate(o)}
    per_seed.append(np.array([r.t2.contributions[ix[n]].ms_per_char for n in NAMES]))
signs = np.sign(np.vstack(per_seed))
live = (np.abs(np.vstack(per_seed)) >= 1e-4).all(axis=0)
unanimous = int(((signs == signs[0:1]).all(axis=0) & live).sum())
n_live = int(live.sum())
pairwise = [
    float(spearmanr(per_seed[i], per_seed[j]).statistic)
    for i in range(len(per_seed))
    for j in range(i + 1, len(per_seed))
]
out["M6"] = {
    "unanimity": unanimous / n_live if n_live else None,
    "n_unanimous": unanimous,
    "n_live_all_seeds": n_live,
    "mean_pairwise_rho": float(np.mean(pairwise)),
    "pairwise_rho": pairwise,
}
log(
    f"M6  SEEDSTAB unanimity {out['M6']['unanimity']:.4f} (bar >= 0.8000)  "
    f"{unanimous}/{n_live} live cols   mean pairwise rho {out['M6']['mean_pairwise_rho']:.4f}"
)

# =========================================================================================
# THE SEVEN-BAR SCORECARD, against interp.1's published values
# =========================================================================================
INTERP1 = {
    "M1_MAXCORR": 0.7037,
    "M1b_MEANCORR": 0.1572,
    "M2_CONSTFRAC": 0.0000,
    "M3_SPLITPAIRS": 2,
    "M4_MONOFRAC": 1.0000,
    "M5_SIGNSTAB": 1.0000,
    "M5_rho": 0.9394,
    "M6_unanimity": 1.0000,
}
SERVED = {
    "M1_MAXCORR": 0.9813,
    "M1b_MEANCORR": 0.1137,
    "M2_CONSTFRAC": 0.0579,
    "M3_SPLITPAIRS": 7,
    "M4_MONOFRAC": 0.0000,
    "M5_SIGNSTAB": 0.9000,
    "M5_rho": 0.8737,
    "M6_unanimity": 0.8000,
}
MINE = {
    "M1_MAXCORR": m1["maxcorr"],
    "M1b_MEANCORR": m1["meancorr"],
    "M2_CONSTFRAC": m2["constfrac"],
    "M3_SPLITPAIRS": m3["splitpairs"],
    "M4_MONOFRAC": m4["monofrac"],
    "M5_SIGNSTAB": m5["sign_agree_frac"],
    "M5_rho": m5["rho"],
    "M6_unanimity": out["M6"]["unanimity"],
}
BARS = {
    "M1_MAXCORR": ("<=", 0.7850),
    "M1b_MEANCORR": (None, None),
    "M2_CONSTFRAC": ("==", 0.0),
    "M3_SPLITPAIRS": ("<", 7),
    "M4_MONOFRAC": (">=", 0.90),
    "M5_SIGNSTAB": (">=", 0.9000),
    "M5_rho": (">=", 0.8737),
    "M6_unanimity": (">=", 0.8000),
}


def passes(key):
    op, bar = BARS[key]
    v = MINE[key]
    if op is None:
        return None
    if op == "<=":
        return bool(v <= bar)
    if op == "<":
        return bool(v < bar)
    if op == ">=":
        return bool(v >= bar)
    if op == "==":
        return bool(abs(v - bar) <= 1e-12)
    raise AssertionError(op)


log("")
log("=" * 100)
log("THE SEVEN-BAR SCORECARD — hybrid-B measured on ALL of them, not screened on M1")
log("=" * 100)
log(f"{'metric':<16} {'bar':>14} {'served':>10} {'interp.1':>10} {'hybrid-B':>10}  verdict")
scorecard = {}
n_pass = n_fail = 0
for k in BARS:
    op, bar = BARS[k]
    p = passes(k)
    scorecard[k] = {
        "bar": f"{op} {bar}" if op else "reported",
        "served": SERVED[k],
        "interp1": INTERP1[k],
        "hybridb": MINE[k],
        "passes": p,
    }
    if p is True:
        n_pass += 1
    elif p is False:
        n_fail += 1
    tag = "PASS" if p is True else ("** FAIL **" if p is False else "reported")
    log(
        f"{k:<16} {(f'{op} {bar}' if op else '-'):>14} {SERVED[k]:>10.4f} {INTERP1[k]:>10.4f} "
        f"{MINE[k]:>10.4f}  {tag}"
    )
out["scorecard"] = scorecard
out["n_bars_passed"] = n_pass
out["n_bars_failed"] = n_fail
log("")
log(f"hybrid-B passes {n_pass} of {n_pass + n_fail} barred metrics (interp.1 passed all 7).")

with open(f"{ARTIFACTS}/metrics7.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/metrics7.json")
