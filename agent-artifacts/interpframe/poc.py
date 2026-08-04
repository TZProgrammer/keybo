"""INTERPFRAME-1 — the PROOF-OF-CONCEPT model, its monotone verification, and M1-M6.

Trains three seeded models on the interp frame from the SAME stroke table the shipped k31 bigram
models were trained on, then:

  §5(a) BOOSTER-LEVEL monotonicity — sweep each column alone, assert the response is monotone.
  §5(b) SHAP-LEVEL monotonicity on HELD-OUT-style data — Spearman(value, own SHAP) has the sign
        the constraint claims.
  §5(c) NON-DEGENERACY — the ADJ-2 PINKY-MONO trap: a constrained column that learned ZERO
        magnitude is not counted toward M4.
  §7 NC4 MONOTONE PLACEBO — constrain a column the WRONG way and confirm (a)/(b) DETECT it. A
        verification that cannot fail is not a verification.

Then the full metric set on the same weighted grid the baseline used, and the side-by-side
attribution on flagship-c3 vs graphite through BOTH frames.

Writes models to /tmp (regenerable) and every NUMBER to the repo (durable).
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import attrib as A  # noqa: E402
import metrics as M  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from keybo.analysis.shap_diff import _shap_tables, default_models, shap_diff  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_INTERP_FEATURE_NAMES,
    BIGRAM_INTERP_MONOTONE,
    FEATURE_VERSION_INTERP,
    interp_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

WPM = 90.0
SEEDS = (0, 1, 2)
STROKES = "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv"
SCRATCH = "/tmp/interpframe_wk/models"
os.makedirs(SCRATCH, exist_ok=True)
NAMES = list(BIGRAM_INTERP_FEATURE_NAMES)
MONO = dict(zip(NAMES, BIGRAM_INTERP_MONOTONE, strict=True))

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} bigram rows; layouts {sorted({r.layout for r in rows})}")
GEO = ROW_STAGGERED_31

out: dict = {
    "strokes": STROKES,
    "n_rows": len(rows),
    "seeds": list(SEEDS),
    "wpm": WPM,
    "frame": NAMES,
    "monotone": dict(MONO),
    "stamp": FEATURE_VERSION_INTERP,
}

# =========================================================================================
# TRAIN — three seeds, monotone ON; and three seeds monotone OFF (the §5d cost arm)
# =========================================================================================
def train_arm(monotone: bool, tag: str):
    """Train (or REUSE) the three seeded models for one arm.

    Reuse is by design, not a shortcut: training is ~90 s per arm and this driver is expected to
    be re-run as the metric code evolves. The reuse path VERIFIES the artifact's recorded
    constraint set against what this arm asked for, so a stale file trained under the other
    setting cannot be silently adopted — which is the only way reuse could corrupt a result.
    """
    from keybo.models.xgboost_model import XGBoostTypingModel

    models = []
    for s in SEEDS:
        path = f"{SCRATCH}/{tag}_seed{s}.json"
        if os.path.exists(path):
            # The stamp is passed EXPLICITLY: `load()` defaults to expecting FEATURE_VERSION and
            # hard-errors on a mismatch, which is precisely the guard that makes the four
            # populations disjoint. Naming the expected version here is the sanctioned way to load
            # a non-served population, and it means a file stamped anything else still refuses.
            m = XGBoostTypingModel.load(path, expected_feature_version=FEATURE_VERSION_INTERP)
            rec = (m.metadata.extra.get("training") or {}).get("interp_frame") or {}
            if bool(rec.get("monotone_constraints")) == monotone and list(
                m.metadata.feature_names
            ) == NAMES:
                models.append(m)
                log(f"  REUSED {tag} seed{s} from {path}")
                continue
            log(f"  {path} exists but records the WRONG arm -- retraining")
        m = train_bigram_model(
            rows,
            target_wpm=WPM,
            geometry=GEO,
            interp=True,
            monotone=monotone,
            random_state=s,
            n_jobs=8,
        )
        assert m.metadata.feature_version == FEATURE_VERSION_INTERP
        assert list(m.metadata.feature_names) == NAMES
        tag_rec = m.metadata.extra["training"]["interp_frame"]
        assert tag_rec["frame"] == "interp"
        assert bool(tag_rec["monotone_constraints"]) == monotone, "the artifact must record it"
        m.save(f"{SCRATCH}/{tag}_seed{s}.json")
        models.append(m)
        log(f"  trained {tag} seed{s}  (constraints recorded: {bool(tag_rec['monotone_constraints'])})")
    return models


log("training interp MONO arm (3 seeds)")
mono_models = train_arm(True, "interp_mono")
log("training interp NOMONO arm (3 seeds)")
nomono_models = train_arm(False, "interp_nomono")

# =========================================================================================
# §5(a) BOOSTER-LEVEL monotonicity — the parameter is not the property
# =========================================================================================
log("§5(a) booster-level monotonicity sweep")
# ⚠ TWO DIFFERENT GRIDS, named apart because conflating them cost a debugging round: the model is
# TRAINED on ROW_STAGGERED_31 (the K31 tables carry the quote slot) and SERVED on
# ROW_STAGGERED_30 (the boards being compared are 30-character). 32^2 = 1024 cells vs 31^2 = 961.
# The booster sweep below is about the MODEL, so it uses the TRAIN grid; every attribution and
# metric is about the SERVED comparison, so those use the SERVE grid. One name for both silently
# produced a 1024-vs-961 mismatch.
train_positions = [*GEO.slots, GEO.space_position]
X_grid = np.vstack(
    [
        interp_features_from_positions(GEO, (a, b), wpm=WPM)
        for a in train_positions
        for b in train_positions
    ]
)


def booster_sweep(models, names, mono_map):
    """Sweep each column alone from its min to its max, others held at the grid MEDIAN.

    Reports the worst VIOLATION (a step in the forbidden direction) and the total RANGE the
    response covers — the range is what catches a constraint that is honored because the column
    was simply never split on (the ADJ-2 trap), which a violation check alone cannot see.
    """
    base = np.median(X_grid, axis=0)
    res = {}
    for j, name in enumerate(names):
        vals = np.unique(X_grid[:, j])
        if len(vals) > 25:
            vals = np.quantile(X_grid[:, j], np.linspace(0, 1, 25))
            vals = np.unique(vals)
        Xs = np.tile(base, (len(vals), 1))
        Xs[:, j] = vals
        pred = np.mean([m.predict(Xs) for m in models], axis=0)
        d = np.diff(pred)
        want = mono_map.get(name, 0)
        # a violation is a step in the direction the constraint forbids
        viol = float(-d.min()) if want > 0 else (float(d.max()) if want < 0 else 0.0)
        res[name] = {
            "constraint": want,
            "n_grid": int(len(vals)),
            "response_range": float(pred.max() - pred.min()),
            "worst_violation": max(viol, 0.0),
            "monotone_ok": bool(viol <= 1e-9),
        }
    return res


sweep_mono = booster_sweep(mono_models, NAMES, MONO)
sweep_nomono = booster_sweep(nomono_models, NAMES, MONO)
out["sweep_mono"] = sweep_mono
out["sweep_nomono"] = sweep_nomono
print()
print(f"{'column':<20} {'con':>4} | {'MONO arm':>28} | {'NOMONO arm':>28}")
print(f"{'':<20} {'':>4} | {'range':>10} {'violation':>10} {'ok':>5} | {'range':>10} {'violation':>10} {'ok':>5}")
for n in NAMES:
    a, b = sweep_mono[n], sweep_nomono[n]
    print(
        f"{n:<20} {MONO[n]:>+4d} | {a['response_range']:>10.5f} {a['worst_violation']:>10.2e} "
        f"{str(a['monotone_ok']):>5} | {b['response_range']:>10.5f} {b['worst_violation']:>10.2e} "
        f"{str(b['monotone_ok']):>5}"
    )

# =========================================================================================
# §5(b) SHAP-LEVEL monotonicity on the weighted serve grid
# =========================================================================================
log("§5(b) SHAP-level monotonicity")
surface = default_surface(WPM, None)
GEO_SURF = surface.geometry
# ⚠ The TRAIN and SERVE geometries are DIFFERENT, and that is the shipped arrangement, not a bug:
# `k31_train.py` trains on ROW_STAGGERED_31 (the K31 tables carry the quote slot at (6,2)) while
# `TimeSurface` builds its position tables on ROW_STAGGERED_30, because the boards being compared
# are 30-character. Features are pure functions of POSITIONS, so a G31-trained model scores G30
# positions correctly — this is exactly what the three shipped `bigram_reg31` artifacts do.
# My first version asserted slot EQUALITY here and it fired; the equality was the wrong invariant.
# The right one is that G30's grid is a SUBSET of the trained-on grid under the SAME stagger and
# space pinning, so no served cell is off the distribution the model saw.
assert set(GEO_SURF.slots) <= set(GEO.slots), "every served slot must be one the model trained on"
assert GEO_SURF.row_offsets == GEO.row_offsets, "stagger must match or dx/distance change meaning"
assert GEO_SURF.space_position == GEO.space_position, "space must be pinned identically"
serve_positions = [*GEO_SURF.slots, GEO_SURF.space_position]
log(
    f"grids: TRAIN {len(train_positions)}^2 = {len(train_positions) ** 2} cells, "
    f"SERVE {len(serve_positions)}^2 = {len(serve_positions) ** 2} cells"
)
_, LAY_A = _resolve("flagship-c3")
_, LAY_B = _resolve("graphite")
w2_char, covered = A.char_bigram_weight(surface, LAY_A)


def shap_monotone(models, frame="interp"):
    """Spearman(feature value, its OWN SHAP value) over the serve grid, per column.

    The sign must match the constraint. Computed on the 3-seed mean SHAP table, and reported with
    each column's mean |SHAP| so a near-zero rho on a DEAD column is distinguishable from a real
    violation on a live one.
    """
    tables = _shap_tables(models, GEO_SURF, WPM, 2, frame)
    shap = np.mean(tables[0], axis=0).reshape(-1, len(tables[5]))
    names = list(tables[5])
    # The SERVE grid, matching the SHAP table `_shap_tables` just built on GEO_SURF -- not the
    # train grid (see the note at §5a). Asserted below rather than trusted.
    Xg = np.vstack(
        [
            (interp_features_from_positions if frame == "interp" else _served_feat)(
                GEO_SURF, (a, b), wpm=WPM
            )
            for a in serve_positions
            for b in serve_positions
        ]
    )
    assert Xg.shape[0] == shap.shape[0], (
        f"grid mismatch: features {Xg.shape} vs SHAP {shap.shape} -- the feature matrix and the "
        f"SHAP table must be built on the SAME geometry"
    )
    res = {}
    for j, name in enumerate(names):
        col, sv = Xg[:, j], shap[:, j]
        rho = float(spearmanr(col, sv).statistic) if float(np.ptp(col)) > 0 else float("nan")
        res[name] = {
            "rho_value_vs_own_shap": rho,
            "mean_abs_shap": float(np.abs(sv).mean()),
            "constraint": MONO.get(name, 0),
        }
    return res


def _served_feat(g, pos, wpm):
    from keybo.features import bigram_features_from_positions

    return bigram_features_from_positions(g, pos, wpm=wpm)


shap_mono = shap_monotone(mono_models)
shap_nomono = shap_monotone(nomono_models)
out["shap_mono"] = shap_mono
out["shap_nomono"] = shap_nomono

# =========================================================================================
# §5(c) NON-DEGENERACY + the HONORED verdict (feeding M4)
# =========================================================================================
DEAD_ABS_SHAP = 1e-6  # a column with less mean |SHAP| than this learned nothing


def honored_map(sweep, shapm):
    """A constraint counts as HONORED only if all three hold: the booster respects it, the SHAP
    relationship has the right sign, AND the column is not dead. The third is the ADJ-2 trap."""
    out_map, detail = {}, {}
    for n in NAMES:
        want = MONO[n]
        ok_booster = sweep[n]["monotone_ok"]
        rho = shapm[n]["rho_value_vs_own_shap"]
        ok_shap = (rho >= -1e-12) if want > 0 else (rho <= 1e-12)
        if np.isnan(rho):
            ok_shap = False
        alive = shapm[n]["mean_abs_shap"] > DEAD_ABS_SHAP
        out_map[n] = bool(ok_booster and ok_shap and alive)
        detail[n] = {
            "booster_monotone": bool(ok_booster),
            "shap_sign_ok": bool(ok_shap),
            "rho": rho,
            "alive": bool(alive),
            "mean_abs_shap": shapm[n]["mean_abs_shap"],
            "honored": out_map[n],
        }
    return out_map, detail


honored, honored_detail = honored_map(sweep_mono, shap_mono)
out["honored_detail"] = honored_detail
print()
print(f"{'column':<20} {'con':>4} {'booster':>8} {'rho(v,shap)':>12} {'mean|SHAP|':>11} {'alive':>6} {'HONORED':>8}")
for n in NAMES:
    d = honored_detail[n]
    print(
        f"{n:<20} {MONO[n]:>+4d} {str(d['booster_monotone']):>8} {d['rho']:>12.4f} "
        f"{d['mean_abs_shap']:>11.3e} {str(d['alive']):>6} {str(d['honored']):>8}"
    )
n_honored = sum(honored.values())
log(f"§5(a-c): {n_honored}/{len(NAMES)} constraints VERIFIED honored")

# =========================================================================================
# §7 NC4 — THE MONOTONE PLACEBO. A verification that cannot fail is not a verification.
# =========================================================================================
log("§7 NC4 monotone PLACEBO: constrain row_load the WRONG way")
placebo_constraints = list(BIGRAM_INTERP_MONOTONE)
J_PLACEBO = NAMES.index("row_load")
placebo_constraints[J_PLACEBO] = -1  # claim "more off-home = FASTER", the opposite mechanism
placebo_models = []
for s in SEEDS[:1]:
    m = train_bigram_model(
        rows,
        target_wpm=WPM,
        geometry=GEO,
        interp=True,
        monotone=False,  # supply the constraint tuple by hand instead
        monotone_constraints=tuple(placebo_constraints),
        random_state=s,
        n_jobs=8,
    )
    placebo_models.append(m)
log("  placebo trained")
placebo_mono = dict(MONO)
placebo_mono["row_load"] = -1
sw_placebo = booster_sweep(placebo_models, NAMES, placebo_mono)
sh_placebo = shap_monotone(placebo_models)
# Under the TRUE mechanism (+1), the placebo model must be DETECTED as violating.
detected_booster = not booster_sweep(placebo_models, NAMES, MONO)["row_load"]["monotone_ok"]
rho_placebo = sh_placebo["row_load"]["rho_value_vs_own_shap"]
detected_shap = rho_placebo < 0
out["placebo"] = {
    "column": "row_load",
    "claimed_constraint": -1,
    "true_mechanism": +1,
    "detected_by_booster_sweep": bool(detected_booster),
    "detected_by_shap_sign": bool(detected_shap),
    "rho_value_vs_own_shap": rho_placebo,
    "response_range": sw_placebo["row_load"]["response_range"],
    "honest_arm_rho": shap_mono["row_load"]["rho_value_vs_own_shap"],
}
print(
    f"  PLACEBO row_load: booster-sweep detects violation vs +1 = {detected_booster}; "
    f"SHAP rho = {rho_placebo:+.4f} (honest arm: {shap_mono['row_load']['rho_value_vs_own_shap']:+.4f}) "
    f"=> shap detects = {detected_shap}"
)
assert detected_booster or detected_shap, "NC4 FAILED: the verification cannot detect a wrong sign"

# =========================================================================================
# THE ATTRIBUTIONS — side by side, both frames, same pair, same weights
# =========================================================================================
log("attributions: interp frame")
r_interp = A.t2_attribution(
    mono_models, GEO_SURF, surface, LAY_A, LAY_B, w2_char, covered, WPM, "interp"
)
r_interp_nm = A.t2_attribution(
    nomono_models, GEO_SURF, surface, LAY_A, LAY_B, w2_char, covered, WPM, "interp"
)
log(f"  interp MONO gap {r_interp['gap']:+.4f}  reconciles {r_interp['reconciles']}")
log(f"  interp NOMONO gap {r_interp_nm['gap']:+.4f}  reconciles {r_interp_nm['reconciles']}")

# NC3 SHUFFLE on the interp frame
r_shuf = A.t2_attribution(
    mono_models, GEO_SURF, surface, LAY_A, LAY_B, w2_char, covered, WPM, "interp", shuffle_seed=0
)
out["nc3_shuffle"] = {
    "reconciles": r_shuf["reconciles"],
    "resid_cell_lmdi": r_shuf["resid_cell_lmdi"],
    "resid_feature_sum": r_shuf["resid_feature_sum"],
    "resid_gap_vs_own": r_shuf["resid_gap_vs_own"],
    "honest_resid_cell_lmdi": r_interp["resid_cell_lmdi"],
}
log(
    f"  NC3 shuffle: reconciles {r_shuf['reconciles']} (must be False), "
    f"cell {r_shuf['resid_cell_lmdi']:.3e} vs honest {r_interp['resid_cell_lmdi']:.3e}, "
    f"gap tie {r_shuf['resid_gap_vs_own']:.3e} (must stay small)"
)
assert not r_shuf["reconciles"], "NC3 FAILED: a shuffled attribution must not reconcile"

# --- per-seed (M6) and per-corpus (M5) on the interp frame ------------------------------
seed_attribs = []
for s, m in enumerate(mono_models):
    r = A.t2_attribution([m], GEO_SURF, surface, LAY_A, LAY_B, w2_char, covered, WPM, "interp")
    seed_attribs.append(r["ms"])
    log(f"  interp seed{s} gap {r['gap']:+.4f} reconciles {r['reconciles']}")

surface_iweb = default_surface(WPM, "iweb")
w2_iweb, covered_iweb = A.char_bigram_weight(surface_iweb, LAY_A)
r_iweb = A.t2_attribution(
    mono_models, GEO_SURF, surface_iweb, LAY_A, LAY_B, w2_iweb, covered_iweb, WPM, "interp"
)
log(f"  interp iweb gap {r_iweb['gap']:+.4f} reconciles {r_iweb['reconciles']}")

# =========================================================================================
# THE METRICS
# =========================================================================================
log("metrics")
slot_a = surface._slot_of(LAY_A)
perm = np.array([slot_a[c] for c in LAY_A] + [slot_a[" "]], dtype=np.intp)
n_pos = len(serve_positions)
w2_pos = np.zeros((n_pos, n_pos))
np.add.at(w2_pos, (perm[:, None], perm[None, :]), w2_char)
assert abs(w2_pos.sum() - w2_char.sum()) < 1e-6

X_in = np.vstack(
    [
        interp_features_from_positions(GEO_SURF, (a, b), wpm=WPM)
        for a in serve_positions
        for b in serve_positions
    ]
)
assert X_in.shape == (n_pos * n_pos, len(NAMES))
ms = r_interp["ms"]
rec = {
    "n_columns": len(NAMES),
    "columns": NAMES,
    "gap_t2": r_interp["gap"],
    "M1": M.m1_maxcorr(X_in, w2_pos.ravel(), NAMES),
    "M2": M.m2_constfrac(X_in, w2_pos.ravel(), NAMES, ms),
    "M3": M.m3_splitpairs(NAMES, ms),
    "M4": M.m4_monofrac(NAMES, ms, honored),
    "M5": M.sign_agreement(ms, r_iweb["ms"]),
    "attribution": dict(zip(NAMES, [float(v) for v in ms], strict=True)),
    "attribution_nomono": dict(
        zip(NAMES, [float(v) for v in r_interp_nm["ms"]], strict=True)
    ),
    "residuals": {k: v for k, v in r_interp.items() if k.startswith("resid") or k == "reconciles"},
}
agree, rhos = [], []
for i in range(len(seed_attribs)):
    for j in range(i + 1, len(seed_attribs)):
        sg = M.sign_agreement(seed_attribs[i], seed_attribs[j])
        agree.append(sg["sign_agree_frac"])
        rhos.append(sg["rho"])
unanimous = np.array(
    [len({np.sign(sa[k]) for sa in seed_attribs if abs(sa[k]) >= 1e-4}) <= 1 for k in range(len(NAMES))]
)
rec["M6"] = {
    "unanimous_sign_frac": float(unanimous.mean()),
    "mean_pairwise_sign_agree": float(np.mean(agree)),
    "mean_pairwise_rho": float(np.mean(rhos)),
    "n_seeds": len(seed_attribs),
}
out["interp"] = rec

# --- the SERVED side-by-side, through the shipped tool -----------------------------------
d_served = shap_diff(LAY_A, LAY_B, name_a="flagship-c3", name_b="graphite", surface=surface, channel="t2")
assert d_served.reconciles()
out["served_t2"] = {
    "gap_t2": d_served.gap_t2,
    "attribution": {c.feature: c.ms_per_char for c in d_served.t2.contributions},
    "blocks": {b.block: b.ms_per_char for b in d_served.t2.blocks()},
}
out["interp_blocks"] = None
from keybo.analysis.shap_diff import block_map  # noqa: E402

spec = block_map(NAMES)
blk: dict[str, float] = {}
for name, v in zip(NAMES, ms, strict=True):
    blk[spec[name][0]] = blk.get(spec[name][0], 0.0) + float(v)
out["interp_blocks"] = blk

with open(f"{ARTIFACTS}/poc.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/poc.json")

# =========================================================================================
# THE HEADLINE TABLES
# =========================================================================================
base = json.load(open(f"{ARTIFACTS}/baseline.json"))
sb = base["frames"]["served_bigram"]
print()
print("=" * 82)
print("INTERPRETABILITY: served bigram frame  vs  interp frame  (same pair, same weights)")
print("=" * 82)
print(f"{'metric':<32} {'served (20c)':>14} {'interp (10c)':>14} {'bar':>18} {'verdict':>8}")


def verdict(ok):
    return "PASS" if ok else "FAIL"


m1_bar = sb["M1"]["maxcorr"] / 1.25
rows_out = [
    ("M1  MAXCORR", sb["M1"]["maxcorr"], rec["M1"]["maxcorr"], f"<= {m1_bar:.4f}", rec["M1"]["maxcorr"] <= m1_bar),
    ("M1b MEANCORR (reported)", sb["M1"]["meancorr"], rec["M1"]["meancorr"], "reported", None),
    ("M2  CONSTFRAC", sb["M2"]["constfrac"], rec["M2"]["constfrac"], "== 0 exactly", rec["M2"]["constfrac"] == 0.0),
    ("M3  SPLITPAIRS", sb["M3"]["splitpairs"], rec["M3"]["splitpairs"], f"< {sb['M3']['splitpairs']}", rec["M3"]["splitpairs"] < sb["M3"]["splitpairs"]),
    ("M4  MONOFRAC", sb["M4"]["monofrac"], rec["M4"]["monofrac"], ">= 0.90", rec["M4"]["monofrac"] >= 0.90),
    ("M5  SIGNSTAB", sb["M5"]["sign_agree_frac"], rec["M5"]["sign_agree_frac"], f">= {sb['M5']['sign_agree_frac']:.4f}", rec["M5"]["sign_agree_frac"] >= sb["M5"]["sign_agree_frac"]),
    ("M5  rho", sb["M5"]["rho"], rec["M5"]["rho"], ">= 0.8737", rec["M5"]["rho"] >= 0.8737),
    ("M6  unanimous sign", sb["M6"]["unanimous_sign_frac"], rec["M6"]["unanimous_sign_frac"], f">= {sb['M6']['unanimous_sign_frac']:.4f}", rec["M6"]["unanimous_sign_frac"] >= sb["M6"]["unanimous_sign_frac"]),
    ("M6  mean pairwise rho", sb["M6"]["mean_pairwise_rho"], rec["M6"]["mean_pairwise_rho"], "reported", None),
]
for label, a, b, bar, ok in rows_out:
    v = "" if ok is None else verdict(ok)
    print(f"{label:<32} {a:>14.4f} {b:>14.4f} {bar:>18} {v:>8}")
out["verdicts"] = {label: ok for label, _a, _b, _bar, ok in rows_out if ok is not None}

print()
print("=" * 82)
print("SIDE-BY-SIDE ATTRIBUTION — flagship-c3 -> graphite, T2 channel")
print("=" * 82)
print(f"  SERVED frame gap_t2 {d_served.gap_t2:+.4f} ms/char        INTERP frame gap_t2 {r_interp['gap']:+.4f} ms/char")
print()
sv = sorted(out["served_t2"]["attribution"].items(), key=lambda kv: -abs(kv[1]))
iv = sorted(rec["attribution"].items(), key=lambda kv: -abs(kv[1]))
print(f"  {'SERVED (20 columns)':<34}   {'INTERP (10 columns)':<34}")
for i in range(max(len(sv), len(iv))):
    left = f"{sv[i][0]:<20} {sv[i][1]:>+9.4f}" if i < len(sv) else ""
    right = f"{iv[i][0]:<20} {iv[i][1]:>+9.4f}" if i < len(iv) else ""
    print(f"  {left:<34}   {right:<34}")
print(f"  {'SUM':<20} {sum(v for _, v in sv):>+9.4f}        {'SUM':<20} {sum(v for _, v in iv):>+9.4f}")
log("done")
