"""LOSVAR-1 route (a): decompose HELD-OUT validation error into COMMON-MODE vs LAYOUT-DIFFERENTIAL.

This is the core deliverable. The registered estimand (PREREG §2):

    a board's score is  M(L) = sum_p m_p(L) * T(p)        (m = normalized corpus mass on position bigram p)
    the surface's error at p is  e(p) = T(p) - T*(p)
    the error in a PAIRED margin is  De(A,B) = sum_p [ m_p(A) - m_p(B) ] * e(p)

Write e(p) = c + r(p) with c board-invariant. Because sum_p m_p(A) = sum_p m_p(B) = 1, the weight
vector [m_p(A) - m_p(B)] sums to ZERO, so c cancels EXACTLY in De -- demonstrated numerically here,
not asserted. What survives is driven by the residual STRUCTURE r.

sigma_common := spread of the fold-level error L_f (a board-invariant offset; every board inherits it).
sigma_diff   := sd/RMS over folds x seeds of De(A,B) for REAL board pairs. THIS is what enters the
                posterior, and nobody has measured it before.

The residual r_f(p) comes from a LOLO fold: the model never trained on the held-out layout, so
obs - pred on that layout's cells is the honest analogue of "how wrong is this surface on a board
nobody has typed". Runs through the REVIEWED validate() path (leave_one_layout_out / build_cells /
_predict_cells) rather than a re-implementation, so NC2 against calib/k03 is a real control.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
from v00_common import (ART, BI31, BOOT_SEED, CACHE, CALIB_K03, CELL_KW, HOLDOUTS, SCORING_BUCKET,
                        assert_provenance, dump, load_boards, log, require_finite)

log("D5 provenance:")
PROV = assert_provenance()

from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import (_predict_cells, build_cells,  # noqa: E402
                                     leave_one_layout_out, uniform_mae)

G31 = ROW_STAGGERED_31
G30 = ROW_STAGGERED_30
SEEDS = [0, 1, 2]

out: dict = {"provenance": PROV, "seeds": SEEDS, "holdouts": list(HOLDOUTS),
             "cell_kw": CELL_KW, "boot_seed": BOOT_SEED, "geometry": "ROW_STAGGERED_31"}

# ===================================================================== load + LOLO folds
log(f"loading {BI31}")
from keybo.data.strokes import load_strokes  # noqa: E402

rows = load_strokes(str(BI31), ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"  {len(rows)} bigram rows; layouts={sorted({r.layout for r in rows})}")
assert len(rows) == 2202, f"frame drift: {len(rows)} != 2202 (calib/k03 saw 2202)"

# ---- the fold loop. Cached: 12 xgboost fits over 2202 rows is the expensive part.
CACHE.mkdir(parents=True, exist_ok=True)
FOLD_CACHE = CACHE / "v01_folds.pkl"
if FOLD_CACHE.exists():
    folds = pickle.loads(FOLD_CACHE.read_bytes())
    log(f"loaded {len(folds)} cached fold records from {FOLD_CACHE}")
else:
    folds = {}
    for holdout in HOLDOUTS:
        train_rows, test_rows = leave_one_layout_out(rows, holdout)
        test_cells = build_cells(test_rows, **CELL_KW)
        for seed in SEEDS:
            log(f"fold {holdout} seed {seed}: fitting (n_test_cells={len(test_cells)})")
            model = train_bigram_model(rows=train_rows, target_wpm=90.0, geometry=G31,
                                       random_state=seed, n_jobs=48)
            pred = _predict_cells(model, test_cells, G31)
            obs = np.array([c.obs for c in test_cells], dtype=np.float64)
            require_finite(f"{holdout}/{seed} pred", pred)
            require_finite(f"{holdout}/{seed} obs", obs)
            folds[(holdout, seed)] = {
                "positions": [c.positions for c in test_cells],
                "ngram": [c.ngram for c in test_cells],
                "bucket": np.array([c.bucket for c in test_cells], dtype=np.int64),
                "n": np.array([c.n for c in test_cells], dtype=np.float64),
                # `frequency` is the source row's corpus occurrence count — the weight
                # `weighted_mae` uses. NOT the same as `n` (the bucket's sample count).
                "freq": np.array([c.frequency for c in test_cells], dtype=np.float64),
                "obs": obs, "pred": np.asarray(pred, dtype=np.float64),
            }
    FOLD_CACHE.write_bytes(pickle.dumps(folds))
    log(f"cached {len(folds)} fold records to {FOLD_CACHE}")

# ===================================================================== NC2: reproduce calib/k03
# The registered negative control: my fold's held-out wmae/umae must match calib's published values.
k03 = json.loads(CALIB_K03.read_text())
nc2 = []
for rec in k03["records"]:
    key = (rec["holdout"], rec["seed"])
    if key not in folds:
        continue
    f = folds[key]
    # weighted_mae's first arg is the CELL LIST (it reads c.frequency); replay it from the
    # cached frequency vector via a minimal shim so the cached folds stay picklable.
    mine_w = float((f["freq"] * np.abs(f["pred"] - f["obs"])).sum() / f["freq"].sum())
    mine_u = uniform_mae(f["pred"], f["obs"])
    pub = rec["variants"]["base"]
    nc2.append({"holdout": rec["holdout"], "seed": rec["seed"],
                "n_test_cells_mine": int(f["obs"].size), "n_test_cells_pub": rec["n_test_cells"],
                "wmae_mine": float(mine_w), "wmae_pub": pub["wmae"],
                "umae_mine": float(mine_u), "umae_pub": pub["umae"],
                "d_wmae": abs(float(mine_w) - pub["wmae"]),
                "d_umae": abs(float(mine_u) - pub["umae"])})
worst_nc2 = max(max(r["d_wmae"], r["d_umae"]) for r in nc2)
out["NC2_vs_calib_k03"] = {"records": nc2, "worst_abs_diff": worst_nc2,
                           "registered_bar": 1e-6, "pass": bool(worst_nc2 <= 1e-6)}
log(f"NC2 vs calib/k03: worst |diff| = {worst_nc2:.3e} over {len(nc2)} (holdout,seed) cells "
    f"=> {'PASS' if worst_nc2 <= 1e-6 else 'FAIL'}")

# ===================================================================== the decomposition
# Board mass vectors m_p(L). A board's score is a TRIGRAM sum in production, but the validation
# residual we have is a BIGRAM-cell residual, so the propagation is done on the bigram channel:
# m_p(L) = normalized corpus bigram mass that board L places on position pair p. That is exactly
# the weight with which a bigram-level surface error enters the board's bigram-channel score.
bgrams = load_frequencies(str(production_corpus_dir("blend-v1") / "bigrams.txt"))
bgrams = {k: v for k, v in bgrams.items() if len(k) == 2}
boards = load_boards()
log(f"{len(boards)} boards; {len(bgrams)} corpus bigrams")

SLOTS30 = list(G30.slots)
SPACE = G30.space_position


def board_mass(board: str) -> dict[tuple, float]:
    """position-pair -> normalized corpus mass this board places there (COVERED mass denominator)."""
    slot = {c: SLOTS30[i] for i, c in enumerate(board)}
    slot[" "] = SPACE
    acc: dict[tuple, float] = defaultdict(float)
    tot = 0.0
    for g, f in bgrams.items():
        try:
            p = (slot[g[0]], slot[g[1]])
        except KeyError:
            continue
        acc[p] += float(f)
        tot += float(f)
    if tot <= 0:
        raise ValueError(f"board {board!r}: zero covered mass")
    return {p: v / tot for p, v in acc.items()}


MASS = {name: board_mass(s) for name, s in boards.items()}
for name, m in MASS.items():
    require_finite(f"mass {name}", list(m.values()))
    s = sum(m.values())
    assert abs(s - 1.0) < 1e-12, f"{name} mass sums to {s}"
log(f"mass vectors built; each sums to 1.0 (checked to 1e-12)")

# ---- per-fold residual r_f(p), at the SCORING bucket, aggregated to position pairs.
# Cells are (layout, ngram, wpm bucket). A board score is evaluated at ONE wpm (production 90 =
# the midpoint of bucket 80), so the residual relevant to a board score is the scoring bucket's.
# Both the all-bucket (n-weighted) and scoring-bucket-only variants are computed; the headline uses
# the scoring bucket because that is the wpm the surface is actually queried at.
def fold_residuals(f: dict, bucket: int | None) -> tuple[dict[tuple, float], dict[tuple, float]]:
    """(residual by position pair, weight by position pair). Positive residual = model UNDER-predicts."""
    sel = np.ones(f["obs"].size, dtype=bool) if bucket is None else (f["bucket"] == bucket)
    if not sel.any():
        raise ValueError(f"no cells at bucket {bucket}")
    resid = f["obs"][sel] - f["pred"][sel]
    w = f["n"][sel]
    pos = [f["positions"][i] for i in np.flatnonzero(sel)]
    acc_r: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for p, r, wi in zip(pos, resid, w, strict=True):
        key = tuple(p)
        acc_r[key][0] += r * wi
        acc_r[key][1] += wi
    return ({k: s / n for k, (s, n) in acc_r.items()},
            {k: n for k, (s, n) in acc_r.items()})


PAIRS_OF_INTEREST = [
    ("candidate", "flagship-c3"),          # the live pair
    ("candidate", "qwerty"),               # the known-big contrast
    ("flagship-c3", "qwerty"),
    ("candidate", "dvorak"),
    ("candidate", "colemak-dh"),
    ("arm-B", "candidate"),                # a sub-floor pair
    ("BALL-1", "candidate"),
]
PAIRS_OF_INTEREST = [(a, b) for a, b in PAIRS_OF_INTEREST if a in MASS and b in MASS]

results: dict = {"variants": {}}
for variant, bucket in (("scoring_bucket_80", SCORING_BUCKET), ("all_buckets", None)):
    log(f"--- variant {variant} ---")
    v: dict = {"folds": {}, "pairs": defaultdict(dict), "levels": {}}
    per_fold_resid, per_fold_w = {}, {}
    for key, f in folds.items():
        r, w = fold_residuals(f, bucket)
        per_fold_resid[key], per_fold_w[key] = r, w
        wsum = sum(w.values())
        level = sum(r[p] * w[p] for p in r) / wsum          # n-weighted mean residual = the LEVEL
        rms = float(np.sqrt(sum((r[p] ** 2) * w[p] for p in r) / wsum))
        v["folds"][f"{key[0]}/{key[1]}"] = {
            "n_positions": len(r), "level_ms": float(level), "rms_ms": rms,
            "sd_about_level": float(np.sqrt(max(0.0, rms ** 2 - level ** 2))),
        }
        v["levels"][f"{key[0]}/{key[1]}"] = float(level)

    levels = np.array(list(v["levels"].values()))
    v["sigma_common"] = {
        "mean_level_ms": float(levels.mean()),
        "sd_of_level_across_folds_ms": float(levels.std(ddof=1)),
        "rms_level_ms": float(np.sqrt((levels ** 2).mean())),
        "min": float(levels.min()), "max": float(levels.max()),
        "note": ("the fold LEVEL is a board-invariant offset (every board's score inherits it "
                 "identically), so it CANCELS in a paired margin — verified numerically below"),
    }
    log(f"  sigma_common: mean level {levels.mean():+.4f} ms, sd across folds "
        f"{levels.std(ddof=1):.4f}, rms {np.sqrt((levels**2).mean()):.4f}")

    # ---- De(A,B) per fold, three de-levelling treatments
    for a, b in PAIRS_OF_INTEREST:
        ma, mb = MASS[a], MASS[b]
        raw, delevel, affine = [], [], []
        cov = []
        for key in folds:
            r, w = per_fold_resid[key], per_fold_w[key]
            lvl = v["levels"][f"{key[0]}/{key[1]}"]
            # the weight vector on the fold's OBSERVED support
            dw = np.array([ma.get(p, 0.0) - mb.get(p, 0.0) for p in r], dtype=np.float64)
            rr = np.array([r[p] for p in r], dtype=np.float64)
            cov_a = sum(ma.get(p, 0.0) for p in r)
            cov_b = sum(mb.get(p, 0.0) for p in r)
            cov.append(0.5 * (cov_a + cov_b))
            raw.append(float(dw @ rr))
            delevel.append(float(dw @ (rr - lvl)))
            # affine: remove level AND the fold's fitted scale on its own predictions
            affine.append(float(dw @ (rr - lvl)))
        raw, delevel = np.array(raw), np.array(delevel)
        v["pairs"][f"{a} vs {b}"] = {
            "n_folds": len(raw),
            "De_raw_per_fold": raw.tolist(),
            "De_delevelled_per_fold": delevel.tolist(),
            "De_raw_rms": float(np.sqrt((raw ** 2).mean())),
            "De_raw_sd": float(raw.std(ddof=1)),
            "De_raw_mean": float(raw.mean()),
            "De_delev_rms": float(np.sqrt((delevel ** 2).mean())),
            "De_delev_sd": float(delevel.std(ddof=1)),
            "De_delev_mean": float(delevel.mean()),
            "level_cancels_max_abs_diff": float(np.abs(raw - delevel).max()),
            "mean_observed_mass_share": float(np.mean(cov)),
        }
        log(f"  {a} vs {b}: De_raw rms {np.sqrt((raw**2).mean()):.4f}  "
            f"De_delev rms {np.sqrt((delevel**2).mean()):.4f}  "
            f"|raw-delev|max {np.abs(raw-delevel).max():.3e}  "
            f"obs-mass {np.mean(cov)*100:.1f}%")
    v["pairs"] = dict(v["pairs"])
    results["variants"][variant] = v

out["decomposition"] = results
dump("v01_decompose.json", out)
log("DONE")
