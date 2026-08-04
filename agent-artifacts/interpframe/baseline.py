"""INTERPFRAME-1 — the CURRENT frame's interpretability baseline (the number to beat).

MEASURED, not borrowed. Five floor-confusions in this project, the newest being that a floor must
match the comparison's DATA VOLUME and not just its design — so every baseline number here is
computed by the SAME code that will score the new frame, on the SAME corpus weighting, at the
SAME wpm, from the SHIPPED models.

Scores the served 20-column BIGRAM frame (the POC's comparison target) and, for context, the
46-column TRIGRAM frame — the latter is where the bg1_/bg2_ split lives, and it is where the
worst M3 score is expected.

Runs AFTER the prereg commit (538dcad) and after the negative control passed 11/11.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import attrib as A  # noqa: E402
import metrics as M  # noqa: E402
from keybo.analysis.shap_diff import (  # noqa: E402
    _char_weight_tables,
    _shap_tables,
    default_models,
    shap_diff,
)
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.features import (  # noqa: E402
    bigram_features_from_positions,
    trigram_features_from_positions,
)

WPM = 90.0
PAIR = ("flagship-c3", "graphite")
_, LAY_A = _resolve(PAIR[0])
_, LAY_B = _resolve(PAIR[1])

print(f"[base] pair {PAIR[0]} -> {PAIR[1]}  wpm={WPM:g}")
surface = default_surface(WPM, None)
geometry = surface.geometry
positions = [*geometry.slots, geometry.space_position]
n_pos = len(positions)
print(f"[base] geometry: {len(geometry.slots)} slots + space = {n_pos} positions")

# --- the corpus weight over POSITION cells, for layout_a ----------------------------------
# The attribution is a frequency-weighted sum over CHARACTER n-grams; the correlation matrix is a
# property of the POSITION grid. The bridge is layout_a's own permutation: cell (i,j) of the
# position grid carries the mass of every character bigram that lands there on THAT board. Using
# an unweighted grid instead would score the frame on cells the corpus never types.
slot_a = surface._slot_of(LAY_A)
chars = LAY_A
w3, w2, covered = _char_weight_tables(surface, chars)
perm = np.array([slot_a[c] for c in chars] + [slot_a[" "]], dtype=np.intp)

w2_pos = np.zeros((n_pos, n_pos))
np.add.at(w2_pos, (perm[:, None], perm[None, :]), w2)
w3_pos = np.zeros((n_pos, n_pos, n_pos))
np.add.at(w3_pos, (perm[:, None, None], perm[None, :, None], perm[None, None, :]), w3)
print(f"[base] covered trigram mass {covered:,}  w2_pos sum {w2_pos.sum():,.0f}")
assert abs(w2_pos.sum() - w2.sum()) < 1e-6, "position-space weight must preserve total mass"
assert abs(w3_pos.sum() - w3.sum()) < 1e-6

out: dict = {"pair": list(PAIR), "wpm": WPM, "covered_mass": int(covered), "frames": {}}


def score_frame(label, names, X, w_flat, attrib, seed_attribs, corpus_attrib):
    """Every metric for one frame, from one implementation (metrics.py)."""
    rec = {"n_columns": len(names), "columns": list(names)}
    rec["M1"] = M.m1_maxcorr(X, w_flat, list(names))
    rec["M2"] = M.m2_constfrac(X, w_flat, list(names), attrib)
    rec["M3"] = M.m3_splitpairs(list(names), attrib)
    # M4 is 0 by construction for the served frames: no shipped model carries any constraint.
    rec["M4"] = M.m4_monofrac(list(names), attrib, honored={})
    if corpus_attrib is not None:
        rec["M5"] = M.sign_agreement(attrib, corpus_attrib)
    if seed_attribs is not None and len(seed_attribs) > 1:
        agree, rhos = [], []
        for i in range(len(seed_attribs)):
            for j in range(i + 1, len(seed_attribs)):
                s = M.sign_agreement(seed_attribs[i], seed_attribs[j])
                agree.append(s["sign_agree_frac"])
                rhos.append(s["rho"])
        unanimous = np.array(
            [
                len({np.sign(sa[k]) for sa in seed_attribs if abs(sa[k]) >= 1e-4}) <= 1
                for k in range(len(names))
            ]
        )
        rec["M6"] = {
            "unanimous_sign_frac": float(unanimous.mean()),
            "mean_pairwise_sign_agree": float(np.mean(agree)),
            "mean_pairwise_rho": float(np.mean(rhos)),
            "n_seeds": len(seed_attribs),
        }
    return rec


# =========================================================================================
# THE SERVED BIGRAM FRAME (the POC's comparison target)
# =========================================================================================
print()
print("=" * 78)
print("SERVED BIGRAM FRAME (20 columns)")
print("=" * 78)

bi_models = default_models("bigram")
X_bi = np.vstack(
    [bigram_features_from_positions(geometry, (a, b), wpm=WPM) for a in positions for b in positions]
)
names_bi = list(bi_models[0].metadata.feature_names)
print(f"[base] X_bi {X_bi.shape}  names {len(names_bi)}")

diff = shap_diff(LAY_A, LAY_B, name_a=PAIR[0], name_b=PAIR[1], surface=surface, channel="both")
assert diff.reconciles(), "the shipped decomposition must reconcile before it is a baseline"
attrib_bi = np.array([c.ms_per_char for c in diff.t2.contributions])
assert [c.feature for c in diff.t2.contributions] == names_bi, "column order must match"
print(f"[base] gap_t2 {diff.gap_t2:+.4f}  gap_tcond {diff.gap_tcond:+.4f}  reconciles True")

# --- CONTROL: the helper must REPRODUCE the shipped tool on the shipped models --------------
# Before any per-seed number is trusted, `attrib.t2_attribution` (which supplies its OWN anchor,
# so a single seed can be attributed at all) is checked against `shap_diff`'s production output on
# the SAME models. If the helper's 3-seed run does not match the shipped tool column for column,
# its single-seed runs mean nothing.
w2_char, covered_char = A.char_bigram_weight(surface, LAY_A)
assert covered_char == covered, "the helper and the driver must share one weight table"
helper3 = A.t2_attribution(
    list(bi_models), geometry, surface, LAY_A, LAY_B, w2_char, covered_char, WPM, "served"
)
assert helper3["names"] == names_bi
d_helper = float(np.abs(helper3["ms"] - attrib_bi).max())
print(f"[base] CONTROL helper-vs-shipped max |diff| per column: {d_helper:.3e}")
print(f"[base]         helper gap {helper3['gap']:+.6f}  shipped gap_t2 {diff.gap_t2:+.6f}")
assert d_helper < 1e-9, f"helper must reproduce the shipped attribution exactly, got {d_helper:.3e}"
out["control_helper_vs_shipped_max_abs_diff"] = d_helper

# --- M6: per-seed attributions, each anchored on its OWN table ------------------------------
# ⚠ A single seed CANNOT be run through `shap_diff` directly: its table differs from the shipped
# 3-seed MEAN by a MODEL difference (~ms, not float32 noise), so the shipped external bar refuses
# it -- correctly. Measured here: the refusal is real, and the fix is a per-seed anchor, not a
# relaxed bar.
seed_attribs_bi = []
for s, m in enumerate(bi_models):
    r = A.t2_attribution(
        [m], geometry, surface, LAY_A, LAY_B, w2_char, covered_char, WPM, "served"
    )
    seed_attribs_bi.append(r["ms"])
    print(
        f"[base]   seed{s} gap {r['gap']:+.4f}  reconciles {r['reconciles']}  "
        f"cell {r['resid_cell_lmdi']:.2e} gap_vs_own {r['resid_gap_vs_own']:.2e}"
    )

# --- M5: the SECOND corpus (iweb), the campaign's other frozen board ----------------------
surface_iweb = default_surface(WPM, "iweb")
d_iweb = shap_diff(LAY_A, LAY_B, surface=surface_iweb, corpus="iweb", channel="t2")
assert d_iweb.reconciles(), "the iweb decomposition must reconcile"
corpus_attrib_bi = np.array([c.ms_per_char for c in d_iweb.t2.contributions])
print(f"[base]   iweb gap_t2 {d_iweb.gap_t2:+.4f}")

out["frames"]["served_bigram"] = score_frame(
    "served_bigram", names_bi, X_bi, w2_pos.ravel(), attrib_bi, seed_attribs_bi, corpus_attrib_bi
)
out["frames"]["served_bigram"]["gap_t2"] = diff.gap_t2
out["frames"]["served_bigram"]["gap_total"] = diff.gap_total
out["frames"]["served_bigram"]["attribution"] = dict(
    zip(names_bi, [float(v) for v in attrib_bi], strict=True)
)

# =========================================================================================
# THE SERVED TRIGRAM FRAME (context: where the bg1_/bg2_ split lives)
# =========================================================================================
print()
print("=" * 78)
print("SERVED TRIGRAM FRAME (46 columns) — context only, the POC is bigram")
print("=" * 78)

tri_models = default_models("trigram")
names_tri = list(tri_models[0].metadata.feature_names)
X_tri = np.vstack(
    [
        trigram_features_from_positions(geometry, (a, b, c), wpm=WPM)
        for a in positions
        for b in positions
        for c in positions
    ]
)
print(f"[base] X_tri {X_tri.shape}")
attrib_tri = np.array([c.ms_per_char for c in diff.tcond.contributions])
assert [c.feature for c in diff.tcond.contributions] == names_tri
corpus_attrib_tri = None
d_iweb_tc = shap_diff(LAY_A, LAY_B, surface=surface_iweb, corpus="iweb", channel="tcond")
assert d_iweb_tc.reconciles()
corpus_attrib_tri = np.array([c.ms_per_char for c in d_iweb_tc.tcond.contributions])

out["frames"]["served_trigram"] = score_frame(
    "served_trigram", names_tri, X_tri, w3_pos.ravel(), attrib_tri, None, corpus_attrib_tri
)
out["frames"]["served_trigram"]["gap_tcond"] = diff.gap_tcond
out["frames"]["served_trigram"]["attribution"] = dict(
    zip(names_tri, [float(v) for v in attrib_tri], strict=True)
)

# =========================================================================================
# THE INTERP FRAME's STRUCTURAL metrics (M1 only — no model yet, so no attribution)
# =========================================================================================
print()
print("=" * 78)
print("INTERP FRAME (10 columns) — M1 is STRUCTURAL (geometry only, no model needed)")
print("=" * 78)

from keybo.features import BIGRAM_INTERP_FEATURE_NAMES, interp_features_from_positions  # noqa: E402

X_in = np.vstack(
    [interp_features_from_positions(geometry, (a, b), wpm=WPM) for a in positions for b in positions]
)
print(f"[base] X_interp {X_in.shape}")
m1_in = M.m1_maxcorr(X_in, w2_pos.ravel(), list(BIGRAM_INTERP_FEATURE_NAMES))
out["frames"]["interp_structural"] = {
    "n_columns": len(BIGRAM_INTERP_FEATURE_NAMES),
    "columns": list(BIGRAM_INTERP_FEATURE_NAMES),
    "M1": m1_in,
    "note": "M1 is a property of the FEATURE MATRIX alone, so it needs no model. M2-M6 need "
    "an attribution and are computed in poc.py once the POC model exists.",
}

# =========================================================================================
# REPORT
# =========================================================================================
print()
print("=" * 78)
print("BASELINE SUMMARY (the numbers the new frame must beat)")
print("=" * 78)
sb = out["frames"]["served_bigram"]
st = out["frames"]["served_trigram"]
si = out["frames"]["interp_structural"]
print(f"{'metric':<34} {'served bigram':>16} {'served trigram':>16} {'interp':>14}")
print(
    f"{'M1  MAXCORR (max |r|)':<34} {sb['M1']['maxcorr']:>16.4f} "
    f"{st['M1']['maxcorr']:>16.4f} {si['M1']['maxcorr']:>14.4f}"
)
print(
    f"{'M1b MEANCORR (mean |r|)':<34} {sb['M1']['meancorr']:>16.4f} "
    f"{st['M1']['meancorr']:>16.4f} {si['M1']['meancorr']:>14.4f}"
)
print(
    f"{'    pairs |r| > 0.9':<34} {sb['M1']['n_pairs_over_0.9']:>16d} "
    f"{st['M1']['n_pairs_over_0.9']:>16d} {si['M1']['n_pairs_over_0.9']:>14d}"
)
print(
    f"{'    pairs |r| > 0.7':<34} {sb['M1']['n_pairs_over_0.7']:>16d} "
    f"{st['M1']['n_pairs_over_0.7']:>16d} {si['M1']['n_pairs_over_0.7']:>14d}"
)
print(f"{'    worst pair (bigram)':<34}   {sb['M1']['worst_pair']}")
print(f"{'    worst pair (trigram)':<34}   {st['M1']['worst_pair']}")
print(f"{'    worst pair (interp)':<34}   {si['M1']['worst_pair']}")
print(
    f"{'M2  CONSTFRAC':<34} {sb['M2']['constfrac']:>16.4f} {st['M2']['constfrac']:>16.4f} "
    f"{'(needs model)':>14}"
)
print(f"{'    constant columns (bigram)':<34}   {sb['M2']['constant_columns']}")
print(f"{'    constant columns (trigram)':<34}   {st['M2']['constant_columns']}")
print(
    f"{'M3  SPLITPAIRS':<34} {sb['M3']['splitpairs']:>16d} {st['M3']['splitpairs']:>16d} "
    f"{'(needs model)':>14}"
)
print(
    f"{'    conflict mass (ms/char)':<34} {sb['M3']['conflict_mass_ms_per_char']:>16.4f} "
    f"{st['M3']['conflict_mass_ms_per_char']:>16.4f}"
)
print(
    f"{'M4  MONOFRAC':<34} {sb['M4']['monofrac']:>16.4f} {st['M4']['monofrac']:>16.4f} "
    f"{'(needs model)':>14}"
)
print(
    f"{'M5  SIGNSTAB (vs iweb)':<34} {sb['M5']['sign_agree_frac']:>16.4f} "
    f"{st['M5']['sign_agree_frac']:>16.4f}"
)
print(f"{'    rho (vs iweb)':<34} {sb['M5']['rho']:>16.4f} {st['M5']['rho']:>16.4f}")
print(f"{'M6  SEEDSTAB unanimous':<34} {sb['M6']['unanimous_sign_frac']:>16.4f}")
print(f"{'    mean pairwise rho':<34} {sb['M6']['mean_pairwise_rho']:>16.4f}")

print()
print("TOP M3 CONFLICTS (same property, opposite signs) — served TRIGRAM frame:")
for p in st["M3"]["pairs"][:8]:
    print(f"  {p['a']:<20} {p['ms_a']:+8.4f}   vs   {p['b']:<20} {p['ms_b']:+8.4f}")
print("TOP M3 CONFLICTS — served BIGRAM frame:")
for p in sb["M3"]["pairs"][:8]:
    print(f"  {p['a']:<20} {p['ms_a']:+8.4f}   vs   {p['b']:<20} {p['ms_b']:+8.4f}")

with open(f"{ARTIFACTS}/baseline.json", "w") as fh:
    json.dump(out, fh, indent=1)
print()
print(f"[base] wrote {ARTIFACTS}/baseline.json")
