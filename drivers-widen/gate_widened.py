"""ARM-1b: does WIDENING the AALTO fit set change the LOLO accuracy-gate verdict?

Reuses GATE-1's machinery (gate_accuracy.py) — leave-one-layout-out over the 4 AALTO layouts,
FIXED precomputed surfaces, bucket-centered Spearman rho, split-half ceiling — but with the
GATE-3-AMENDMENT corrections BAKED IN:

  1. VERDICT ON PAIRED PER-FOLD DELTAS, not a mean-of-ratios. Ceiling cancellation is exact
     PER FOLD but a mean-of-ratios across folds with different ceilings is a ceiling-WEIGHTED
     mean and CAN reorder. So the primary statistic is the per-fold delta (blend - ms/char) in
     RAW rho, and we report: sign per fold, and the ceiling-divided ratio per fold separately.
  2. NO `rho/ceiling if ceiling>0 else nan` idiom (that silently overwrote 36 rhos in GATE-2).
     We gate on np.isfinite(ceiling) and abs(ceiling)>0.05, keep RAW rho always, and print the
     finite-rho count per objective so a silent all-nan can't masquerade as "no result".

Runs the SAME gate on two tables and diffs the verdict:
  --table shipped  : /local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv  (POSITIVE CONTROL:
                     must reproduce GATE-1 cell counts 12596/2590/860/235 and ms/char win 4/4)
  --table widened  : drivers-widen/tables/tristrokes31_widened.tsv (the widened frame)
"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")
from keybo.data.strokes import load_strokes
from keybo.training.validate import _bucket_centered, build_cells, split_half_ceiling
from keybo.verdicts import reweighting_margin_bound, require_finite

SURF = Path("/local/home/zegertho/repos/keybo/data/surfaces")
POOLS = ("AALTO", "COMMUNITY", "POOL")
WEIGHTINGS = {
    "ms/char (shipped)": {"AALTO": 1.0},
    "drop-pool 50/50": {"AALTO": 0.5, "COMMUNITY": 0.5},
    "registered (c)": {"AALTO": 0.5411334196884872, "COMMUNITY": 0.39767911636312825,
                       "POOL": 0.06118746394838459},
}
SHIPPED_TSV = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"

def load_surface(name: str) -> np.ndarray:
    with gzip.open(SURF / f"{name}_TRI_PS_FREQ_PRIOR.standardized.npy.gz", "rb") as f:
        return np.load(f, allow_pickle=False)

def slot_of(position, slots):
    try: return slots.index(position)
    except ValueError: return None

def predict(surface, cells, slots):
    out = np.full(len(cells), np.nan)
    for idx, cell in enumerate(cells):
        pos = cell.positions
        if len(pos) != 3: continue
        ijk = [slot_of(p, slots) for p in pos]
        if any(v is None for v in ijk): continue
        out[idx] = surface[ijk[0], ijk[1], ijk[2]]
    return out

def run_gate(tsv_path: str, tag: str = "", ceiling_cache: dict | None = None) -> dict:
    from keybo.geometry import ROW_STAGGERED_31
    slots = list(ROW_STAGGERED_31.slots)
    print(f"[{tag}] loading {tsv_path} ...", flush=True)
    rows = load_strokes(tsv_path, ngram_len=3, wpm_threshold=0, min_samples=10)
    layouts = sorted({r.layout for r in rows})
    surfaces = {p: load_surface(p) for p in POOLS}
    print(f"[{tag}] loaded {len(rows)} rows, layouts {layouts}", flush=True)

    # per-fold RAW rho and ceiling, per objective
    raw_rho: dict[str, dict[str, float]] = {lab: {} for lab in WEIGHTINGS}
    ceilings: dict[str, float] = {}
    n_cells: dict[str, int] = {}
    n_scoreable: dict[str, int] = {}
    n_pids: dict[str, int] = {}
    for held in layouts:
        fold_rows = [r for r in rows if r.layout == held]
        cells = build_cells(fold_rows, wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
        n_cells[held] = len(cells)
        if not cells: continue
        obs = np.array([c.obs for c in cells])
        # Ceiling cache: qwerty is IDENTICAL across shipped/widened (only non-qwerty widened),
        # so a (tag-independent) fold-keyed cache avoids recomputing the 27M-sample qwerty
        # ceiling twice. n_boot=10 matches GATE-1's gate_accuracy.py exactly (positive control).
        ck = held
        if ceiling_cache is not None and ck in ceiling_cache:
            ceiling = ceiling_cache[ck]
            print(f"[{tag}] {held}: ceiling {ceiling:.4f} (cached)", flush=True)
        else:
            print(f"[{tag}] {held}: computing split_half_ceiling ({len(cells)} cells) ...", flush=True)
            ceiling = split_half_ceiling(fold_rows, wpm_lo=40, wpm_hi=140, bucket_width=20,
                                         min_cell_samples=10, n_boot=10, seed=0)
            print(f"[{tag}] {held}: ceiling {ceiling:.4f}", flush=True)
            if ceiling_cache is not None and held == "qwerty":
                ceiling_cache[ck] = float(ceiling)  # qwerty only — the identical, expensive one
        ceilings[held] = float(ceiling)
        preds = {p: predict(surfaces[p], cells, slots) for p in POOLS}
        ok = np.ones(len(cells), dtype=bool)
        for p in POOLS: ok &= np.isfinite(preds[p])
        ok &= np.isfinite(obs)
        n_scoreable[held] = int(ok.sum())
        n_pids[held] = len({s[2] for c, keep in zip(cells, ok) if keep for s in c.samples})
        if int(ok.sum()) < 30: continue
        kept = [c for c, keep in zip(cells, ok, strict=True) if keep]
        obs_c = _bucket_centered(kept, obs[ok])
        for label, weights in WEIGHTINGS.items():
            total = sum(weights.values())
            blended = np.zeros(int(ok.sum()))
            for p, w in weights.items():
                blended += (w / total) * preds[p][ok]
            pred_c = _bucket_centered(kept, blended)
            rho = float(spearmanr(pred_c, obs_c).statistic)  # RAW rho, always kept
            raw_rho[label][held] = rho
    return {"layouts": layouts, "raw_rho": raw_rho, "ceilings": ceilings,
            "n_cells": n_cells, "n_scoreable": n_scoreable, "n_pids": n_pids}

def report(tag: str, g: dict) -> dict:
    folds = sorted(g["ceilings"])
    print(f"\n########## TABLE: {tag} ##########")
    print(f"folds: {folds}")
    print(f"{'layout':10s}{'n_cells':>9s}{'scoreable':>10s}{'n_pids':>8s}{'ceiling':>9s}")
    for f in folds:
        print(f"{f:10s}{g['n_cells'].get(f,0):9d}{g['n_scoreable'].get(f,0):10d}"
              f"{g['n_pids'].get(f,0):8d}{g['ceilings'][f]:9.4f}")

    # RAW rho matrix + finite counts (guard against silent all-nan)
    print(f"\n=== RAW bucket-centered rho per fold ===")
    print(f"{'objective':22s} " + " ".join(f"{f:>10s}" for f in folds) + f" {'#finite':>8s}")
    for lab in WEIGHTINGS:
        vals = [g["raw_rho"][lab].get(f, float('nan')) for f in folds]
        nfin = sum(np.isfinite(v) for v in vals)
        print(f"{lab:22s} " + " ".join(f"{v:+10.4f}" for v in vals) + f" {nfin:8d}")

    # rho/ceiling per fold (reported, NOT the verdict basis)
    print(f"\n=== rho / ceiling per fold (reported; NOT verdict basis) ===")
    print(f"{'objective':22s} " + " ".join(f"{f:>10s}" for f in folds))
    for lab in WEIGHTINGS:
        vals = []
        for f in folds:
            r = g["raw_rho"][lab].get(f, float('nan')); c = g["ceilings"][f]
            vals.append(r/c if (np.isfinite(c) and abs(c) > 0.05) else float('nan'))
        print(f"{lab:22s} " + " ".join(f"{v:+10.4f}" for v in vals))

    # PRIMARY: PAIRED PER-FOLD DELTAS (blend - ms/char) in RAW rho
    print(f"\n=== PAIRED PER-FOLD DELTAS in RAW rho (blend - ms/char) — PRIMARY VERDICT ===")
    base = g["raw_rho"]["ms/char (shipped)"]
    bound = reweighting_margin_bound([(1 + g["ceilings"][f]) / 2 for f in folds])
    print(f"reweighting margin bound (from ceilings {[round(g['ceilings'][f],4) for f in folds]}): {bound:.4f}")
    verdicts = {}
    for lab in WEIGHTINGS:
        if lab.startswith("ms/char"): continue
        deltas = {f: g["raw_rho"][lab].get(f, float('nan')) - base.get(f, float('nan')) for f in folds}
        wins = sum(1 for f in folds if np.isfinite(deltas[f]) and deltas[f] > 0)
        # relative per-fold delta vs |ms/char rho| that fold, and does it clear the bound?
        rel = {f: (deltas[f] / abs(base[f]) if np.isfinite(base.get(f, float('nan'))) and base[f] != 0 else float('nan')) for f in folds}
        clears_perfold = {f: bool(np.isfinite(rel[f]) and rel[f] > bound) for f in folds}
        n_clear = sum(clears_perfold.values())
        mean_delta = float(np.mean([deltas[f] for f in folds if np.isfinite(deltas[f])]))
        print(f"\n  {lab}:")
        for f in folds:
            print(f"    {f:9s} ms/char {base.get(f,float('nan')):+.4f}  blend {g['raw_rho'][lab].get(f,float('nan')):+.4f}"
                  f"  delta {deltas[f]:+.4f}  rel {rel[f]:+.2%}  {'CLEARS' if clears_perfold[f] else 'no'}")
        print(f"    => wins {wins}/{len(folds)} folds; clears-bound {n_clear}/{len(folds)} folds; mean raw delta {mean_delta:+.4f}")
        verdicts[lab] = {"deltas": deltas, "wins": wins, "n_folds": len(folds),
                         "clears_bound_folds": n_clear, "mean_raw_delta": mean_delta,
                         "rel": rel, "clears_perfold": clears_perfold}
    return {"folds": folds, "bound": bound, "verdicts": verdicts,
            "raw_rho": g["raw_rho"], "ceilings": g["ceilings"],
            "n_cells": g["n_cells"], "n_scoreable": g["n_scoreable"], "n_pids": g["n_pids"]}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widened", default="drivers-widen/tables/tristrokes31_widened.tsv")
    args = ap.parse_args()
    tables = {"shipped": SHIPPED_TSV, "widened": args.widened}
    out = {}
    ceiling_cache: dict = {}  # shared across tables; qwerty ceiling is identical, cache it once
    for tag, path in tables.items():
        g = run_gate(path, tag=tag, ceiling_cache=ceiling_cache)
        out[tag] = report(tag, g)
    # DIFF the verdict shipped vs widened
    print("\n\n########## SHIPPED vs WIDENED — did the verdict change? ##########")
    for lab in WEIGHTINGS:
        if lab.startswith("ms/char"): continue
        s = out["shipped"]["verdicts"][lab]; w = out["widened"]["verdicts"][lab]
        print(f"{lab}: shipped wins {s['wins']}/{s['n_folds']} (clears {s['clears_bound_folds']}) "
              f"mean-delta {s['mean_raw_delta']:+.4f}  ->  widened wins {w['wins']}/{w['n_folds']} "
              f"(clears {w['clears_bound_folds']}) mean-delta {w['mean_raw_delta']:+.4f}")
    outp = Path("drivers-widen/gate_widened.json")
    outp.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {outp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
