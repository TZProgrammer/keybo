"""EXPLOIT-1 §4 — the MECHANISM test. Does exploitation (if any) work the way I registered?

Three per-board, model-free-ish metrics with directional predictions registered at da56139 BEFORE
any number existed. All are computed for BOTH arms' winners AND for the named layouts + the C30M
start, so the metric's own DYNAMIC RANGE is known before a difference between two boards is read.

The registered caveat matters as much as the prediction: if M1 is ~93% for every board, it has no
dynamic range and cannot discriminate -- that is an UNINFORMATIVE METRIC, not a null result.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import ARTIFACTS, SCRATCH, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_INTERP_FEATURE_NAMES,
    FEATURE_VERSION_INTERP,
    bigram_features_from_positions,
    interp_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402

WPM = 90.0
K31_SEEDS = (0, 1, 2)
CHARS = SF.C30M
GEO = ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


models = []
for s in K31_SEEDS:
    m = XGBoostTypingModel.load(f"{SCRATCH}/interp_mono_seed{s}.json",
                                expected_feature_version=FEATURE_VERSION_INTERP)
    assert "wpm" not in m.metadata.feature_names
    models.append(m)

surface = default_surface(WPM, None)
T2_SERVED = surface._T2.copy()
vec_i = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
T2_INTERP = np.mean([m.predict_ms(vec_i, wpm=WPM).reshape(NP, NP) for m in models], axis=0)

# group id per cell, on the interp frame
_, inv, cnt = np.unique(vec_i, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
GRP = inv.reshape(NP, NP)
COLLAPSED = (cnt[inv] > 1).reshape(NP, NP)   # True where the cell shares its row with another

# corpus bigram weight per POSITION-PAIR is layout-dependent, so it is built per board below
tri = {k: v for k, v in load_frequencies(str(production_corpus_dir(None) / "trigrams.txt")).items()
       if len(k) == 3}
IDX = {c: i for i, c in enumerate(CHARS)}
IDX[" "] = NP - 1
F3 = np.zeros((NP, NP, NP))
for ng, f in tri.items():
    try:
        F3[IDX[ng[0]], IDX[ng[1]], IDX[ng[2]]] += f
    except KeyError:
        continue
F2_CHAR = F3.sum(axis=2)      # char-index space; permuted into position space per board

# The truth's within-group dispersion (M3's basis): sd of T2_SERVED inside each interp group,
# weighted by nothing -- a property of the surfaces alone.
grp_sd = np.zeros(len(cnt))
flat_s = T2_SERVED.ravel()
for g in range(len(cnt)):
    if cnt[g] > 1:
        grp_sd[g] = flat_s[inv == g].std()
GRP_SD = grp_sd[inv].reshape(NP, NP)

DELTA = T2_SERVED - T2_INTERP     # >0 where the PROXY UNDERPRICES the truth


def metrics(lay30: str) -> dict:
    """M1/M2/M3 for one board. All are corpus-mass-weighted over the cells the board USES."""
    slot = {pos: i for i, pos in enumerate(GEO.slots)}
    perm = np.empty(NP, dtype=np.intp)
    for i, ch in enumerate(lay30):
        perm[IDX[ch]] = slot[GEO.slots[i]]
    perm[NP - 1] = NP - 1
    # mass on each POSITION pair for this board
    W = np.zeros((NP, NP))
    np.add.at(W, (perm[:, None], perm[None, :]), F2_CHAR)
    tot = W.sum()
    return {
        # M1 -- share of corpus mass in cells that are featurewise COLLAPSED under interp.1
        "M1_collapsed_mass_share": float(W[COLLAPSED].sum() / tot),
        # M2 -- mass-weighted surface OPTIMISM: how much the proxy underprices this board
        "M2_optimism_ms": float((W * DELTA).sum() / tot),
        # M3 -- mass-weighted within-group dispersion of the TRUTH that the proxy is blind to
        "M3_within_group_sd_ms": float((W * GRP_SD).sum() / tot),
        "n_distinct_groups_used": int(len(np.unique(GRP[W > 0]))),
    }


ex = json.load(open(f"{ARTIFACTS}/exploit.json"))
boards = {}
for ch in ("G", "B"):
    v = ex["verdict"][ch]
    boards[f"{ch}-INTERP-optimal"] = v["interp_board"]
    boards[f"{ch}-SERVED-optimal"] = v["served_board"]
boards["qwerty-C30M (start)"] = CHARS
# Named boards for DYNAMIC RANGE only -- those on the C30M charset. A different charset covers
# different corpus rows, so it is skipped rather than printed in the same column.
for name, lay in NAMED_LAYOUTS.items():
    if sorted(lay) == sorted(CHARS):
        boards[f"named:{name}"] = lay

out = {"prereg": "EXPLOIT-preregistration.md @ da56139 §4", "boards": {}}
log(f"{'board':<24} {'M1 collapsed mass':>18} {'M2 optimism ms':>15} {'M3 wg-sd ms':>13} {'groups':>7}")
for label, lay in boards.items():
    r = metrics(lay)
    out["boards"][label] = {**r, "layout": lay}
    log(f"{label:<24} {r['M1_collapsed_mass_share']:>17.4%} {r['M2_optimism_ms']:>15.4f} "
        f"{r['M3_within_group_sd_ms']:>13.4f} {r['n_distinct_groups_used']:>7d}")

# --- the registered directional tests -------------------------------------------------------
out["tests"] = {}
for ch in ("G", "B"):
    i, s = out["boards"][f"{ch}-INTERP-optimal"], out["boards"][f"{ch}-SERVED-optimal"]
    rng_m1 = max(b["M1_collapsed_mass_share"] for b in out["boards"].values()) - \
             min(b["M1_collapsed_mass_share"] for b in out["boards"].values())
    out["tests"][ch] = {
        "M1_interp": i["M1_collapsed_mass_share"], "M1_served": s["M1_collapsed_mass_share"],
        "M1_delta": i["M1_collapsed_mass_share"] - s["M1_collapsed_mass_share"],
        "M1_prediction_holds": i["M1_collapsed_mass_share"] > s["M1_collapsed_mass_share"],
        "M1_dynamic_range_over_all_boards": rng_m1,
        "M2_interp": i["M2_optimism_ms"], "M2_served": s["M2_optimism_ms"],
        "M2_delta": i["M2_optimism_ms"] - s["M2_optimism_ms"],
        "M2_prediction_holds": i["M2_optimism_ms"] > s["M2_optimism_ms"],
        "M3_interp": i["M3_within_group_sd_ms"], "M3_served": s["M3_within_group_sd_ms"],
        "M3_delta": i["M3_within_group_sd_ms"] - s["M3_within_group_sd_ms"],
    }
    t = out["tests"][ch]
    log("")
    log(f"### channel {ch} mechanism ###")
    log(f"  M1 collapsed mass: interp {t['M1_interp']:.4%} vs served {t['M1_served']:.4%}  "
        f"(delta {t['M1_delta']:+.4%})  PREDICTION {'HOLDS' if t['M1_prediction_holds'] else 'FAILS'}")
    log(f"     M1 dynamic range across ALL boards measured: {rng_m1:.4%}"
        f"{'   <-- NO RANGE: metric is UNINFORMATIVE' if rng_m1 < 0.01 else ''}")
    log(f"  M2 optimism: interp {t['M2_interp']:+.4f} vs served {t['M2_served']:+.4f} ms  "
        f"(delta {t['M2_delta']:+.4f})  PREDICTION {'HOLDS' if t['M2_prediction_holds'] else 'FAILS'}")
    log(f"  M3 within-group sd: interp {t['M3_interp']:.4f} vs served {t['M3_served']:.4f} ms "
        f"(delta {t['M3_delta']:+.4f})")

with open(f"{ARTIFACTS}/mechanism.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/mechanism.json")
