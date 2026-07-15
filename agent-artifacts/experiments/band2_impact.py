"""BAND-2 stage B (rule efcb335): K31-frame calibration + flagship-board impact.

Cross-fit (leave-one-layout-out, seed 0) the winning correction family (C-ISO)
and the rank-safe affine reference on the K31 production frame's SERVE band
(80-100, wpm 90), pooling out-of-fold (prediction, observation) pairs — every
pair predicted by a model that never saw its layout. Corrections are fit on
GEOMETRY-ONLY predictions (no practice term): that is exactly what the
production tables serve, so the fitted map is the table-vs-reality gap.

Impact: apply the corrections to the production timecard tables (T2 + Tcond at
wpm 90) and recompute the flagship board's time-saved%% vs qwerty30m, with
per-fold spread and an explicit ORDER-preservation check for the nonlinear
isotonic map. Messaging changes from these numbers remain user-gated.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import (  # noqa: E402
    bigram_features_from_positions,
    trigram_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model, train_trigram_model  # noqa: E402
from keybo.training.validate import build_cells, leave_one_layout_out  # noqa: E402

T0 = time.time()
OUT = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/band2_impact.json")
SERVE_BUCKET, WPM_LO, WPM_HI, W, MIN_CELL = 80, 40, 140, 20, 10
REG_LOLO = dict(
    colsample_bytree=0.7, gamma=0.957, learning_rate=0.05, max_depth=3,
    min_child_weight=4, n_estimators=300, reg_alpha=0.141, reg_lambda=0.011,
    subsample=0.7, verbosity=0, n_jobs=24,
)
CAND4 = dict(
    n_estimators=427, max_depth=5, learning_rate=0.10903767015375725,
    min_child_weight=6, subsample=0.6086566147198375,
    colsample_bytree=0.9893815206317236, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    verbosity=0, n_jobs=24,
)
SURFACES = {
    "bigram": dict(path="/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv",
                   ngram_len=2, trainer=train_bigram_model, params=REG_LOLO),
    "trigram": dict(path="/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv",
                    ngram_len=3, trainer=train_trigram_model, params=CAND4),
}
BOARD = {
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "p16-balance": "frlwg'uyoksntdc.ieahvxmpb,-jqz",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
}
REF = "qwertyuiopasdfghjkl'zxcvbnm,.-"  # qwerty30m


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def predict_cells_geometry_only(model, cells) -> np.ndarray:
    featurize = (trigram_features_from_positions if len(cells[0].positions) == 3
                 else bigram_features_from_positions)
    X = np.vstack([featurize(ROW_STAGGERED_31, c.positions, wpm=c.wpm) for c in cells])
    return model.to_ms(model.predict(X), X)


def wls_affine(p, o, w):
    pm = (w * p).sum() / w.sum()
    om = (w * o).sum() / w.sum()
    den = (w * (p - pm) ** 2).sum()
    b = float((w * (p - pm) * (o - om)).sum() / den) if den > 0 else 1.0
    return om - max(b, 1e-3) * pm, max(b, 1e-3)


def main() -> None:
    from sklearn.isotonic import IsotonicRegression

    git = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                          "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    result = {"experiment": "BAND-2 stage B (rule efcb335)", "git": git,
              "status": "running", "surfaces": {}, "impact": {}}

    def checkpoint():
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1, default=float))
        os.replace(tmp, OUT)

    fits = {}  # surface -> dict(iso=fn, aff=(a,b), per_fold_aff={fold: (a,b)}, ...)
    for sname, spec in SURFACES.items():
        sha = hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
        rows = load_strokes(spec["path"], ngram_len=spec["ngram_len"],
                            wpm_threshold=0, min_samples=1)
        layouts = sorted({r.layout for r in rows})
        log(f"=== {sname}: {len(rows)} rows (sha {sha[:8]}), folds {layouts}")
        oof_p, oof_o, oof_w, oof_fold = [], [], [], []
        for holdout in layouts:
            train_rows, test_rows = leave_one_layout_out(rows, holdout)
            cells = [c for c in build_cells(test_rows, WPM_LO, WPM_HI, W, MIN_CELL)
                     if c.bucket == SERVE_BUCKET]
            if not cells:
                continue
            m = spec["trainer"](train_rows, target_wpm=90.0, geometry=ROW_STAGGERED_31,
                                progress=False, random_state=0, **spec["params"])
            p = predict_cells_geometry_only(m, cells)
            del m
            oof_p += list(p)
            oof_o += [c.obs for c in cells]
            oof_w += [c.frequency for c in cells]
            oof_fold += [holdout] * len(cells)
            log(f"  {sname}/{holdout}: {len(cells)} serve-band OOF cells")
        p = np.array(oof_p); o = np.array(oof_o); w = np.array(oof_w, dtype=np.float64)
        aff = wls_affine(p, o, w)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, o, sample_weight=w)
        per_fold_aff = {}
        for f in sorted(set(oof_fold)):
            idx = np.array([i for i, x in enumerate(oof_fold) if x == f])
            per_fold_aff[f] = wls_affine(p[idx], o[idx], w[idx])
        slope_before = wls_affine(p, o, w)[1]
        fits[sname] = dict(iso=iso, aff=aff, per_fold_aff=per_fold_aff,
                           pred_range=(float(p.min()), float(p.max())))
        result["surfaces"][sname] = {
            "sha256": sha, "n_oof_cells": len(p),
            "pooled_affine": {"a": aff[0], "b": aff[1]},
            "per_fold_affine": {f: {"a": a, "b": b} for f, (a, b) in per_fold_aff.items()},
            "oof_serve_slope_obs_on_pred": slope_before,
        }
        checkpoint()
        log(f"  {sname} pooled affine a={aff[0]:.2f} b={aff[1]:.3f}; "
            f"per-fold b: { {f: round(v[1], 3) for f, v in per_fold_aff.items()} }")

    # ---- flagship impact -----------------------------------------------------------------
    log("building production timecard surface")
    from keybo.analysis.timecard import default_surface

    surf = default_surface(90.0)
    T2_0, Tc_0 = surf._T2.copy(), surf._Tc.copy()

    def board_saved(label: str) -> dict:
        ref_card = surf.card(REF)
        out = {}
        for name, lay in BOARD.items():
            card = surf.card(lay, ref_total_ms=ref_card.total_ms)
            out[name] = round(card.saved_vs_ref_pct, 4)
        log(f"  {label}: " + " ".join(f"{k} {v:+.2f}" for k, v in out.items()))
        return out

    result["impact"]["uncorrected"] = board_saved("uncorrected")

    def apply_correction(kind: str, aff_bi=None, aff_tri=None):
        if kind == "iso":
            surf._T2 = fits["bigram"]["iso"].predict(T2_0.ravel()).reshape(T2_0.shape)
            surf._Tc = fits["trigram"]["iso"].predict(Tc_0.ravel()).reshape(Tc_0.shape)
        else:
            a2, b2 = aff_bi
            a3, b3 = aff_tri
            surf._T2 = a2 + b2 * T2_0
            surf._Tc = a3 + b3 * Tc_0

    apply_correction("aff", fits["bigram"]["aff"], fits["trigram"]["aff"])
    result["impact"]["affine_pooled"] = board_saved("affine_pooled")
    for f in fits["bigram"]["per_fold_aff"]:
        apply_correction("aff", fits["bigram"]["per_fold_aff"][f],
                         fits["trigram"]["per_fold_aff"][f])
        result["impact"][f"affine_fold_{f}"] = board_saved(f"affine[{f}]")
    apply_correction("iso")
    result["impact"]["isotonic_pooled"] = board_saved("isotonic_pooled")
    surf._T2, surf._Tc = T2_0, Tc_0

    orders = {k: sorted(v, key=v.get, reverse=True) for k, v in result["impact"].items()}
    result["order_preserved"] = {k: orders[k] == orders["uncorrected"] for k in orders}
    result["table_range_note"] = {
        s: {"fit_range_ms": fits[s]["pred_range"]} for s in fits
    }
    result["status"] = "complete"
    result["elapsed_seconds"] = time.time() - T0
    checkpoint()
    log("ALL-DONE")


if __name__ == "__main__":
    main()
