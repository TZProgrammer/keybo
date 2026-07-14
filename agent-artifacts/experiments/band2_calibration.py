"""BAND-2 stage A (rule efcb335): confirm + choose the WPM-conditioned calibration family.

3 seeds x 4 LOLO folds x 2 surfaces (bigram, conditioned trigram). Per fold/seed:
train G on the production recipe, fit each correction family on TRAIN-fold cells
(cross-fit — held-out cells never touch the fit), apply to held-out predictions,
ensemble seeds as mean-of-calibrated-ms. Also the (band x class) mechanism probe.

Families: C-BAND (per-20-band affine, ms) / C-SPLINE (band affines interpolated
in wpm) / C-LIN (affine coefficients linear in wpm, one WLS) / C-LOG (per-band
affine in log-ms) / C-ISO (per-band isotonic regression).

Run: nice -n 10 /tmp/keybo_venv/bin/python agent-artifacts/experiments/band2_calibration.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.features.classify import classify_positions  # noqa: E402
from keybo.training.train import train_bigram_model, train_trigram_model  # noqa: E402
from keybo.training.validate import (  # noqa: E402
    _predict_cells,
    build_cells,
    calibration_slope,
    leave_one_layout_out,
    weighted_mae,
)

T0 = time.time()
OUT = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/band2_calibration.json")
SURFACES = {
    "bigram": dict(
        path="/local/home/zegertho/keybo-e2e/bistrokes_v5.tsv",
        sha="d6cb4c81e8915f384787f872af58a6f2acffb7afa97f5d6c6b2a3804488bb1da",
        ngram_len=2,
        objective=str(Path(__file__).resolve().parents[2] / "data/corpus/bigrams.txt"),
    ),
    "trigram": dict(
        path="/local/home/zegertho/keybo-e2e/tristrokes_cond_v3.tsv",
        sha="1b5d7abdbf409deb48135c24332a8a9e3dfdac75cf8e736638e4426cc17d4596",
        ngram_len=3,
        objective=str(Path(__file__).resolve().parents[2] / "data/corpus/trigrams.txt"),
    ),
}
SEEDS = (0, 1, 2)
WPM_LO, WPM_HI, W, MIN_CELL = 40, 140, 20, 10
BANDS = list(range(WPM_LO, WPM_HI, W))
RANKING_BUCKET = 80
TRAIN_PARAMS = dict(
    colsample_bytree=0.7, gamma=0.957, learning_rate=0.05, max_depth=3,
    min_child_weight=4, n_estimators=300, reg_alpha=0.141, reg_lambda=0.011,
    subsample=0.7, verbosity=0, n_jobs=24,
)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def wlogmae(cells, pred, obs) -> float:
    w = np.array([c.frequency for c in cells], dtype=np.float64)
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 1e-9)
    obs = np.maximum(np.asarray(obs, dtype=np.float64), 1e-9)
    return float((w * np.abs(np.log(pred) - np.log(obs))).sum() / w.sum())


def wls_affine(p, o, w):
    pm = (w * p).sum() / w.sum()
    om = (w * o).sum() / w.sum()
    den = (w * (p - pm) ** 2).sum()
    b = float((w * (p - pm) * (o - om)).sum() / den) if den > 0 else 1.0
    b = max(b, 1e-3)
    return om - b * pm, b


# ---- correction families (fit on train cells; return fn(cells, pred) -> corrected) ---------
def fit_c_band(tc, tp, to, tw):
    coef = {}
    for band in BANDS:
        idx = np.array([i for i, c in enumerate(tc) if c.bucket == band])
        coef[band] = wls_affine(tp[idx], to[idx], tw[idx]) if len(idx) >= 10 else (0.0, 1.0)

    def apply(cells, pred):
        out = pred.copy()
        for i, c in enumerate(cells):
            a, b = coef[c.bucket]
            out[i] = a + b * pred[i]
        return out

    return apply, {str(k): v for k, v in coef.items()}


def fit_c_spline(tc, tp, to, tw):
    _, coef = fit_c_band(tc, tp, to, tw)
    mids = np.array([b + W / 2 for b in BANDS])
    avec = np.array([coef[str(b)][0] for b in BANDS])
    bvec = np.array([coef[str(b)][1] for b in BANDS])

    def apply(cells, pred):
        wpms = np.array([c.wpm for c in cells])
        a = np.interp(wpms, mids, avec)
        b = np.interp(wpms, mids, bvec)
        return a + b * pred

    return apply, coef


def fit_c_lin(tc, tp, to, tw):
    # o ~ a0 + a1*w + (b0 + b1*w)*p  — one weighted least squares
    wv = np.array([c.wpm for c in tc])
    X = np.column_stack([np.ones_like(tp), wv, tp, wv * tp])
    sw = np.sqrt(tw)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], to * sw, rcond=None)

    def apply(cells, pred):
        wv = np.array([c.wpm for c in cells])
        return beta[0] + beta[1] * wv + (beta[2] + beta[3] * wv) * pred

    return apply, {"a0": float(beta[0]), "a1": float(beta[1]),
                   "b0": float(beta[2]), "b1": float(beta[3])}


def fit_c_log(tc, tp, to, tw):
    lp, lo = np.log(np.maximum(tp, 1e-9)), np.log(np.maximum(to, 1e-9))
    coef = {}
    for band in BANDS:
        idx = np.array([i for i, c in enumerate(tc) if c.bucket == band])
        coef[band] = wls_affine(lp[idx], lo[idx], tw[idx]) if len(idx) >= 10 else (0.0, 1.0)

    def apply(cells, pred):
        out = np.log(np.maximum(pred, 1e-9))
        for i, c in enumerate(cells):
            a, b = coef[c.bucket]
            out[i] = a + b * out[i]
        return np.exp(out)

    return apply, {str(k): v for k, v in coef.items()}


def fit_c_iso(tc, tp, to, tw):
    from sklearn.isotonic import IsotonicRegression

    fits = {}
    for band in BANDS:
        idx = np.array([i for i, c in enumerate(tc) if c.bucket == band])
        if len(idx) >= 20:
            fits[band] = IsotonicRegression(out_of_bounds="clip").fit(
                tp[idx], to[idx], sample_weight=tw[idx])

    def apply(cells, pred):
        out = pred.copy()
        by_band = defaultdict(list)
        for i, c in enumerate(cells):
            by_band[c.bucket].append(i)
        for band, idx in by_band.items():
            if band in fits:
                out[idx] = fits[band].predict(pred[idx])
        return out

    return apply, {"bands_fit": sorted(fits)}


FAMILIES = {"C-BAND": fit_c_band, "C-SPLINE": fit_c_spline, "C-LIN": fit_c_lin,
            "C-LOG": fit_c_log, "C-ISO": fit_c_iso}


def metrics(cells, pred, obs):
    by_band = defaultdict(list)
    for i, c in enumerate(cells):
        by_band[c.bucket].append(i)
    freq_order = np.argsort([c.frequency for c in cells])
    dec = np.array_split(freq_order, 10)
    b3 = np.concatenate(dec[:3])
    return {
        "wlogmae": wlogmae(cells, pred, obs),
        "wmae": weighted_mae(cells, pred, obs),
        "bottom3_decile_mae": float(np.abs(pred[b3] - obs[b3]).mean()),
        "band_slopes": {str(b): calibration_slope(pred[idx], obs[idx])
                        for b, idx in sorted(by_band.items())},
    }


def bigram_class(c):
    kind = classify_positions(ROW_STAGGERED_30, c.positions[0], c.positions[1]).name
    return kind  # SAME_FINGER / SAME_HAND / ALTERNATE


def main() -> None:
    result = {"experiment": "BAND-2 stage A (rule efcb335)",
              "git": subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                     "rev-parse", "HEAD"], capture_output=True,
                                    text=True).stdout.strip(),
              "train_params": {k: v for k, v in TRAIN_PARAMS.items() if k != "n_jobs"},
              "status": "running", "surfaces": {}}

    def checkpoint():
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1, default=float))
        os.replace(tmp, OUT)

    checkpoint()
    for sname, spec in SURFACES.items():
        sha = hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
        assert sha == spec["sha"], f"{sname} hash mismatch"
        rows = load_strokes(spec["path"], ngram_len=spec["ngram_len"],
                            wpm_threshold=0, min_samples=1)
        layouts = sorted({r.layout for r in rows})
        obj_freq = {}
        for line in open(spec["objective"]):
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                obj_freq[p[0]] = int(p[1])
        trainer = train_bigram_model if spec["ngram_len"] == 2 else train_trigram_model
        log(f"=== surface {sname}: {len(rows)} rows, folds {layouts} ===")
        sres = {"folds": {}, "coefs_example": None}
        rank = defaultdict(lambda: defaultdict(dict))  # arm -> layout -> seed scores

        for holdout in layouts:
            train_rows, test_rows = leave_one_layout_out(rows, holdout)
            cells = build_cells(test_rows, WPM_LO, WPM_HI, W, MIN_CELL)
            obs = np.array([c.obs for c in cells])
            train_cells = build_cells(train_rows, WPM_LO, WPM_HI, W, MIN_CELL)
            t_obs = np.array([c.obs for c in train_cells])
            t_w = np.array([c.frequency for c in train_cells], dtype=np.float64)
            per_seed = {"G": []}
            per_seed.update({f: [] for f in FAMILIES})
            coefs_seed0 = {}
            for seed in SEEDS:
                m = trainer(train_rows, target_wpm=90.0, geometry=ROW_STAGGERED_30,
                            progress=False, random_state=seed, **TRAIN_PARAMS)
                gp = _predict_cells(m, cells, ROW_STAGGERED_30)
                tp = _predict_cells(m, train_cells, ROW_STAGGERED_30)
                per_seed["G"].append(gp)
                for fam, fit in FAMILIES.items():
                    apply, coef = fit(train_cells, tp, t_obs, t_w)
                    per_seed[fam].append(apply(cells, gp))
                    if seed == 0:
                        coefs_seed0[fam] = coef
                del m
                log(f"  {sname}/{holdout} seed {seed} done")
            fold = {"n_cells": len(cells), "arms": {}}
            for arm, plist in per_seed.items():
                ens = np.mean(plist, axis=0)
                fold["arms"][arm] = metrics(cells, ens, obs)
                fold["arms"][arm]["per_seed_wlogmae"] = [wlogmae(cells, p, obs)
                                                         for p in plist]
                idx90 = [i for i, c in enumerate(cells) if c.bucket == RANKING_BUCKET]
                num_p = num_o = den = 0.0
                for i in idx90:
                    f = obj_freq.get(cells[i].ngram)
                    if f:
                        num_p += f * ens[i]
                        num_o += f * cells[i].obs
                        den += f
                if den:
                    rank[arm][holdout] = {"pred": num_p / den, "obs": num_o / den}
            # mechanism probe (bigram only, ensemble G)
            if sname == "bigram":
                ens_g = np.mean(per_seed["G"], axis=0)
                probe = defaultdict(lambda: defaultdict(list))
                for i, c in enumerate(cells):
                    probe[c.bucket][bigram_class(c)].append(i)
                fold["class_slopes"] = {
                    str(b): {cls: calibration_slope(ens_g[idx], obs[idx])
                             for cls, idx in by_cls.items() if len(idx) >= 12}
                    for b, by_cls in sorted(probe.items())}
            if sres["coefs_example"] is None:
                sres["coefs_example"] = coefs_seed0
            sres["folds"][holdout] = fold
            result["surfaces"][sname] = sres
            checkpoint()
            g = fold["arms"]["G"]["wlogmae"]
            log(f"  {sname}/{holdout}: G {g:.5f} | " + " ".join(
                f"{f} {100*(fold['arms'][f]['wlogmae']-g)/g:+.1f}%" for f in FAMILIES))

        taus = {}
        for arm, per_layout in rank.items():
            ls = sorted(per_layout)
            if len(ls) >= 3:
                taus[arm] = float(kendalltau(
                    [per_layout[la]["pred"] for la in ls],
                    [per_layout[la]["obs"] for la in ls]).statistic)
        sres["tau_heldout_wpm90"] = taus
        result["surfaces"][sname] = sres
        checkpoint()
        log(f"=== {sname} taus: {taus}")

    result["status"] = "complete"
    result["elapsed_seconds"] = time.time() - T0
    checkpoint()
    log("ALL-DONE")


if __name__ == "__main__":
    main()
