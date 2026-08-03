"""LOSVAR-1 v02: sigma_diff for ALL pairs, the CLOSING-2 layout-count test, and the OBSERVED flip rate.

Three things v01 left open.

A. SIGMA_DIFF, TWO ESTIMANDS, ALL 78 PAIRS. v01 measured De on the fold's OBSERVED support with the
   full-support mass weights. That partial sum does NOT have a zero-sum weight vector (coverage
   differs by board), so the fold's residual LEVEL leaks in — measured at up to 0.2499 ms/char, which
   is why de-levelling is mandatory rather than cosmetic. Two variants are computed and both reported:
     * TRUNCATED+DELEVELLED — dw = m_A|O - m_B|O with the fold level removed. A partial sum of the
       ACTUAL quantity of interest (the error in the full-corpus margin), level artifact subtracted.
     * RENORMALIZED — each board's mass renormalized over O so dw sums to 0 EXACTLY and the level
       cancels algebraically. This is the error of the CO-OBSERVED-subset margin: a different score,
       but the one whose common-mode cancellation is exact. Cross-check on the first.

B. THE CLOSING-2 TEST. Does sigma_diff FALL as the model trains on more LAYOUTS? Registered
   prediction (CLOSING-2, standing): layout diversity binds, not compute. Single-variable design: the
   EVALUATION layout is held FIXED while the training set varies over subsets of the other three, so
   n_train = 1, 2, 3 is compared on the SAME held-out board. Anything else confounds train size with
   eval set.

C. THE GENUINELY OBSERVED FLIP RATE — route (b2), which is what the brief's route (b) was supposed
   to be. PICK2-1's q is model-vs-model (verified: both its operands are (T2+Tc) sums over the same
   shipped surface), so it cannot corroborate a validation-error estimate. Here the model's predicted
   margin between two layouts is compared against the margin computed from HELD-OUT OBSERVED cell
   times on the same position bigrams — model vs DATA, a true wrong-sign frequency. Only the 4
   training layouts have observed data, so this runs on their 6 pairs; that smallness is the binding
   limit and is reported, not hidden.
"""
from __future__ import annotations

import itertools
import json
import pickle
from collections import defaultdict

import numpy as np
from v00_common import (BI31, BOOT_SEED, CACHE, CELL_KW, HOLDOUTS, SCORING_BUCKET,
                        assert_provenance, dump, load_boards, log, require_finite)

log("D5 provenance:")
PROV = assert_provenance()

from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402
from keybo.training.validate import _predict_cells, build_cells  # noqa: E402

G31, G30 = ROW_STAGGERED_31, ROW_STAGGERED_30
SEEDS = [0, 1, 2]
out: dict = {"provenance": PROV, "seeds": SEEDS, "cell_kw": CELL_KW, "boot_seed": BOOT_SEED}

log(f"loading {BI31}")
rows = load_strokes(str(BI31), ngram_len=2, wpm_threshold=0, min_samples=1)
assert len(rows) == 2202, f"frame drift: {len(rows)}"
log(f"  {len(rows)} rows")

# ============================================================ board mass vectors (same as v01)
bgrams = {k: v for k, v in
          load_frequencies(str(production_corpus_dir("blend-v1") / "bigrams.txt")).items()
          if len(k) == 2}
boards = load_boards()
SLOTS30, SPACE = list(G30.slots), G30.space_position


def board_mass(board: str) -> dict[tuple, float]:
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
    return {p: v / tot for p, v in acc.items()}


MASS = {n: board_mass(s) for n, s in boards.items()}
BOARD_NAMES = sorted(MASS)
log(f"{len(MASS)} boards")

#: The four REAL layouts the K31 study collected, as 30-char strings on ROW_STAGGERED_30 slot order
#: (row-major top/home/bottom, left->right). Needed for route (b2): the observed flip rate is read
#: on these boards, and only `qwerty` and `dvorak` are in the tournament field.
#: Taken from the stroke table itself (below) rather than transcribed — a transcribed board is a
#: silent data hazard in this campaign (colemak-dh shipped two different strings across artifacts).
_REAL_LAYOUT_STRINGS = {"qwerty": boards["qwerty"], "dvorak": boards["dvorak"]}
_LAYOUT_MASS = {}
for _la in HOLDOUTS:
    if _la in _REAL_LAYOUT_STRINGS:
        _LAYOUT_MASS[_la] = board_mass(_REAL_LAYOUT_STRINGS[_la])
    else:
        # azerty / qwertz are not in the board field. Recover char->position DIRECTLY from the
        # stroke rows (each row carries both its ngram and its positions), which is the
        # authoritative mapping the model was trained against.
        cp: dict[str, tuple] = {}
        for _r in rows:
            if _r.layout != _la:
                continue
            for _ch, _pos in zip(_r.ngram, _r.positions, strict=True):
                cp.setdefault(_ch, tuple(_pos))
        acc: dict[tuple, float] = defaultdict(float)
        tot = 0.0
        for _g, _f in bgrams.items():
            try:
                _p = (cp[_g[0]], cp[_g[1]])
            except KeyError:
                continue
            acc[_p] += float(_f)
            tot += float(_f)
        if tot <= 0:
            raise ValueError(f"layout {_la}: zero covered mass from stroke-derived char map")
        _LAYOUT_MASS[_la] = {p: v / tot for p, v in acc.items()}
        log(f"  {_la}: char map recovered from stroke rows ({len(cp)} chars), "
            f"covered corpus mass {tot / sum(bgrams.values()) * 100:.1f}%")
for _la, _m in _LAYOUT_MASS.items():
    require_finite(f"real-layout mass {_la}", list(_m.values()))
    assert abs(sum(_m.values()) - 1.0) < 1e-12

# ============================================================ the fold engine, cached
FC = CACHE / "v02_folds.pkl"
folds: dict = pickle.loads(FC.read_bytes()) if FC.exists() else {}
if FC.exists():
    log(f"loaded {len(folds)} cached fold records")


def get_fold(train_layouts: tuple[str, ...], eval_layout: str, seed: int) -> dict:
    """Fit on ``train_layouts`` only, predict ``eval_layout``'s cells. Cached by key."""
    key = (train_layouts, eval_layout, seed)
    if key in folds:
        return folds[key]
    assert eval_layout not in train_layouts, "leakage: eval layout is in the training set"
    tr = [r for r in rows if r.layout in train_layouts]
    te = [r for r in rows if r.layout == eval_layout]
    if not tr or not te:
        raise ValueError(f"empty split for {key}")
    cells = build_cells(te, **CELL_KW)
    log(f"  fit train={'+'.join(train_layouts)} eval={eval_layout} seed={seed} "
        f"(n_train_rows={len(tr)}, n_eval_cells={len(cells)})")
    model = train_bigram_model(rows=tr, target_wpm=90.0, geometry=G31,
                              random_state=seed, n_jobs=48)
    pred = np.asarray(_predict_cells(model, cells, G31), dtype=np.float64)
    obs = np.array([c.obs for c in cells], dtype=np.float64)
    require_finite(f"{key} pred", pred)
    folds[key] = {"positions": [c.positions for c in cells],
                  "bucket": np.array([c.bucket for c in cells], dtype=np.int64),
                  "n": np.array([c.n for c in cells], dtype=np.float64),
                  "obs": obs, "pred": pred}
    return folds[key]


def resid_by_position(f: dict, bucket: int | None) -> tuple[dict, dict]:
    sel = np.ones(f["obs"].size, bool) if bucket is None else (f["bucket"] == bucket)
    if not sel.any():
        raise ValueError(f"no cells at bucket {bucket}")
    acc: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for i in np.flatnonzero(sel):
        p = tuple(f["positions"][i])
        acc[p][0] += (f["obs"][i] - f["pred"][i]) * f["n"][i]
        acc[p][1] += f["n"][i]
    return ({k: s / w for k, (s, w) in acc.items()}, {k: w for k, (s, w) in acc.items()})


def de_for_pair(r: dict, w: dict, a: str, b: str) -> dict:
    """Both De estimands for one fold's residual map and one board pair."""
    ma, mb = MASS[a], MASS[b]
    keys = list(r)
    rr = np.array([r[p] for p in keys], dtype=np.float64)
    wa = np.array([ma.get(p, 0.0) for p in keys], dtype=np.float64)
    wb = np.array([mb.get(p, 0.0) for p in keys], dtype=np.float64)
    ws = np.array([w[p] for p in keys], dtype=np.float64)
    level = float((rr * ws).sum() / ws.sum())
    cov_a, cov_b = float(wa.sum()), float(wb.sum())
    dw = wa - wb
    # TRUNCATED + DELEVELLED: partial sum of the real quantity, level artifact removed
    trunc = float(dw @ (rr - level))
    # RENORMALIZED: dw sums to 0 exactly => the level cancels algebraically (checked below)
    dwn = wa / cov_a - wb / cov_b
    renorm = float(dwn @ rr)
    renorm_delev = float(dwn @ (rr - level))
    return {"trunc_delev": trunc, "raw": float(dw @ rr), "renorm": renorm,
            "renorm_delev": renorm_delev, "level": level,
            "dwn_sum": float(dwn.sum()), "cov_a": cov_a, "cov_b": cov_b}


# ============================================================ A: sigma_diff, all 78 pairs
log("=== A: sigma_diff over all pairs (LOLO, n_train=3) ===")
LOLO = [(tuple(sorted(set(HOLDOUTS) - {h})), h) for h in HOLDOUTS]
resid_cache: dict = {}
for tr, ev in LOLO:
    for s in SEEDS:
        f = get_fold(tr, ev, s)
        for vname, bkt in (("scoring_bucket_80", SCORING_BUCKET), ("all_buckets", None)):
            resid_cache[(tr, ev, s, vname)] = resid_by_position(f, bkt)
CACHE.mkdir(parents=True, exist_ok=True)
FC.write_bytes(pickle.dumps(folds))

sigma: dict = {}
for vname in ("scoring_bucket_80", "all_buckets"):
    per_pair = {}
    max_renorm_level_leak = 0.0
    for a, b in itertools.combinations(BOARD_NAMES, 2):
        vals = defaultdict(list)
        for tr, ev in LOLO:
            for s in SEEDS:
                r, w = resid_cache[(tr, ev, s, vname)]
                d = de_for_pair(r, w, a, b)
                for k in ("trunc_delev", "raw", "renorm", "renorm_delev"):
                    vals[k].append(d[k])
                max_renorm_level_leak = max(max_renorm_level_leak,
                                           abs(d["renorm"] - d["renorm_delev"]))
        td = np.array(vals["trunc_delev"])
        rn = np.array(vals["renorm"])
        rw = np.array(vals["raw"])
        per_pair[f"{a}|{b}"] = {
            "sigma_diff_rms": float(np.sqrt((td ** 2).mean())),      # PRIMARY (conservative)
            "sigma_diff_sd": float(td.std(ddof=1)),
            "sigma_diff_mean": float(td.mean()),                     # a BIAS estimate
            "renorm_rms": float(np.sqrt((rn ** 2).mean())),
            "renorm_sd": float(rn.std(ddof=1)),
            "raw_rms": float(np.sqrt((rw ** 2).mean())),
            "level_leak_max": float(np.abs(rw - td).max()),
            "per_fold_trunc_delev": td.tolist(),
            "n_fold_seed": int(td.size), "n_independent_folds": len(LOLO),
        }
    sigma[vname] = {"pairs": per_pair,
                    "renorm_level_cancels_max_abs": max_renorm_level_leak}
    log(f"  {vname}: {len(per_pair)} pairs; renormalized level-cancellation residual "
        f"max |diff| = {max_renorm_level_leak:.3e}")
out["sigma_diff"] = sigma

# ============================================================ B: the CLOSING-2 layout-count test
log("=== B: CLOSING-2 — does sigma_diff fall with MORE TRAINING LAYOUTS? ===")
KEY_PAIRS = [("candidate", "flagship-c3"), ("candidate", "qwerty"), ("arm-B", "candidate")]
closing: dict = {}
for ev in HOLDOUTS:
    pool = sorted(set(HOLDOUTS) - {ev})
    for k in (1, 2, 3):
        for tr in itertools.combinations(pool, k):
            for s in SEEDS:
                get_fold(tuple(tr), ev, s)
FC.write_bytes(pickle.dumps(folds))

for vname, bkt in (("scoring_bucket_80", SCORING_BUCKET), ("all_buckets", None)):
    byk: dict = {}
    for k in (1, 2, 3):
        acc = {f"{a}|{b}": [] for a, b in KEY_PAIRS}
        levels, rmss = [], []
        for ev in HOLDOUTS:
            pool = sorted(set(HOLDOUTS) - {ev})
            for tr in itertools.combinations(pool, k):
                for s in SEEDS:
                    r, w = resid_by_position(get_fold(tuple(tr), ev, s), bkt)
                    ws = np.array(list(w.values()))
                    rr = np.array([r[p] for p in r])
                    levels.append(float((rr * ws).sum() / ws.sum()))
                    rmss.append(float(np.sqrt((rr ** 2 * ws).sum() / ws.sum())))
                    for a, b in KEY_PAIRS:
                        acc[f"{a}|{b}"].append(de_for_pair(r, w, a, b)["trunc_delev"])
        byk[k] = {
            "n_configs": len(levels),
            "level_mean": float(np.mean(levels)), "level_sd": float(np.std(levels, ddof=1)),
            "position_resid_rms_mean": float(np.mean(rmss)),
            "sigma_diff_rms": {p: float(np.sqrt((np.array(v) ** 2).mean())) for p, v in acc.items()},
        }
        log(f"  {vname} n_train={k}: {len(levels)} configs, position-resid rms "
            f"{np.mean(rmss):.3f}, sigma_diff(live pair) "
            f"{byk[k]['sigma_diff_rms']['candidate|flagship-c3']:.4f}")
    closing[vname] = byk
out["closing2_layout_count"] = closing

# ============================================================ C: the OBSERVED flip rate (route b2)
log("=== C: route (b2) — the GENUINELY OBSERVED flip rate (model vs held-out DATA) ===")
# For each ORDERED pair of the 4 real layouts, compare the sign of the model's predicted margin
# against the sign of the OBSERVED margin, both computed on the position bigrams the two layouts
# CO-OBSERVE, using the fold model that held out each layout (so neither side is in-sample for the
# board it scores). The corpus-mass weighting is the same one a board score uses.
obs_flip: dict = {"pairs": {}, "note": (
    "model-vs-DATA. Each layout is scored by the fold model that HELD IT OUT, so both operands are "
    "out-of-sample. Restricted to position bigrams BOTH layouts observed, weighted by the corpus "
    "mass each layout places there. This is a true wrong-sign frequency; PICK2-1's q is not.")}
for a, b in itertools.combinations(HOLDOUTS, 2):
    per_seed = []
    for s in SEEDS:
        fa = get_fold(tuple(sorted(set(HOLDOUTS) - {a})), a, s)
        fb = get_fold(tuple(sorted(set(HOLDOUTS) - {b})), b, s)

        def tables(f):
            sel = f["bucket"] == SCORING_BUCKET
            o, p, w = defaultdict(float), defaultdict(float), defaultdict(float)
            for i in np.flatnonzero(sel):
                key = tuple(f["positions"][i])
                o[key] += f["obs"][i] * f["n"][i]
                p[key] += f["pred"][i] * f["n"][i]
                w[key] += f["n"][i]
            return ({k: o[k] / w[k] for k in w}, {k: p[k] / w[k] for k in w})

        oa, pa = tables(fa)
        ob, pb = tables(fb)
        co = sorted(set(oa) & set(ob))
        if not co:
            continue
        wa = np.array([_LAYOUT_MASS[a].get(p, 0.0) for p in co])
        wb = np.array([_LAYOUT_MASS[b].get(p, 0.0) for p in co])
        wa_n, wb_n = wa / wa.sum(), wb / wb.sum()
        obs_margin = float((wa_n * np.array([oa[p] for p in co])).sum()
                           - (wb_n * np.array([ob[p] for p in co])).sum())
        pred_margin = float((wa_n * np.array([pa[p] for p in co])).sum()
                            - (wb_n * np.array([pb[p] for p in co])).sum())
        per_seed.append({"seed": s, "n_co": len(co),
                         "obs_margin": obs_margin, "pred_margin": pred_margin,
                         "flip": bool(np.sign(obs_margin) != np.sign(pred_margin)),
                         "abs_err": abs(obs_margin - pred_margin)})
    if per_seed:
        obs_flip["pairs"][f"{a}|{b}"] = {
            "per_seed": per_seed,
            "flip_rate": float(np.mean([r["flip"] for r in per_seed])),
            "mean_pred_margin": float(np.mean([r["pred_margin"] for r in per_seed])),
            "mean_obs_margin": float(np.mean([r["obs_margin"] for r in per_seed])),
            "mean_abs_err": float(np.mean([r["abs_err"] for r in per_seed])),
        }
        r = obs_flip["pairs"][f"{a}|{b}"]
        log(f"  {a} vs {b}: pred {r['mean_pred_margin']:+8.3f}  obs {r['mean_obs_margin']:+8.3f}  "
            f"|err| {r['mean_abs_err']:7.3f}  flip_rate {r['flip_rate']:.2f}")
allf = [s["flip"] for p in obs_flip["pairs"].values() for s in p["per_seed"]]
allerr = [s["abs_err"] for p in obs_flip["pairs"].values() for s in p["per_seed"]]
obs_flip["overall_flip_rate"] = float(np.mean(allf)) if allf else float("nan")
obs_flip["n_pair_seed"] = len(allf)
obs_flip["margin_abs_err_rms"] = float(np.sqrt(np.mean(np.square(allerr)))) if allerr else float("nan")
log(f"  OVERALL observed flip rate {obs_flip['overall_flip_rate']:.3f} over {len(allf)} pair-seeds; "
    f"margin |err| rms {obs_flip['margin_abs_err_rms']:.4f} ms/char")
out["observed_flip_rate_route_b2"] = obs_flip

dump("v02_sigma_and_flips.json", out)
log("DONE")
