"""ARM 2: within-AALTO geometry contrast on KEYBOARD_TYPE (never attempted).

Question (from the brief): can a within-Aalto covariate contrast SUBSTITUTE for the community
frame's 2 clean participants? I.e. hold out a KEYBOARD_TYPE level and score the 3 objectives.

The shipped tristroke table stores only (wpm, duration, pid, hold) per sample — no covariate —
so we map pid -> KEYBOARD_TYPE via the metadata and PARTITION each layout's cells' samples by
the held-out covariate level, then score ms/char vs blends the SAME way GATE-1 does.

FINGERS is handled separately (arm2_fingers): the process filter fixes FINGERS=="9-10", so it
has zero variance in the fit set — predicted UNINFORMATIVE. This driver quantifies that too.

DESIGN (locked in PREREGISTRATION.md): held-out unit = KEYBOARD_TYPE level (full | laptop).
Score bucket-centered rho on the held-out level's cells against the SAME fixed surfaces.
Use PAIRED PER-FOLD DELTAS across the 2 levels. Report UNINFORMATIVE if a level has <2 pids or
<30 scoreable cells (do not retry, do not drop silently).
"""
from __future__ import annotations
import csv, gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")
from keybo.data.strokes import load_strokes, StrokeRow, iqr_average
from keybo.training.validate import _bucket_centered, build_cells, split_half_ceiling

META = "/local/home/zegertho/keybo-e2e/dataset/Keystrokes/files/metadata_participants.txt"
SHIPPED_TSV = "/local/home/zegertho/keybo-e2e/tristrokes31_cond_v1.tsv"
SURF = Path("/local/home/zegertho/repos/keybo/data/surfaces")
POOLS = ("AALTO", "COMMUNITY", "POOL")
WEIGHTINGS = {
    "ms/char (shipped)": {"AALTO": 1.0},
    "drop-pool 50/50": {"AALTO": 0.5, "COMMUNITY": 0.5},
    "registered (c)": {"AALTO": 0.5411334196884872, "COMMUNITY": 0.39767911636312825,
                       "POOL": 0.06118746394838459},
}
csv.field_size_limit(sys.maxsize)

def pid_covariate(col: str) -> dict[int, str]:
    """pid -> covariate value, over ALL metadata rows that pass the fit filter's structural
    predicates (FINGERS 9-10, KB full/laptop, supported layout, wpm>=40) so it matches the
    shipped table's population exactly."""
    from keybo.data.keystrokes import _LAYOUT_ROWS
    out = {}
    with open(META, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if (r.get("FINGERS") or "").strip() != "9-10": continue
            try:
                if float((r.get("AVG_WPM_15") or "0").strip()) < 40: continue
            except ValueError: continue
            if (r.get("KEYBOARD_TYPE") or "").strip().lower() not in {"full", "laptop"}: continue
            lay = (r.get("LAYOUT") or "").strip().lower()
            if lay not in _LAYOUT_ROWS: continue
            try: pid = int((r.get("PARTICIPANT_ID") or "").strip())
            except ValueError: continue
            out[pid] = (r.get(col) or "").strip().lower()
    return out

def load_surface(name):
    with gzip.open(SURF / f"{name}_TRI_PS_FREQ_PRIOR.standardized.npy.gz", "rb") as f:
        return np.load(f, allow_pickle=False)

def slot_of(position, slots):
    try: return slots.index(position)
    except ValueError: return None

def predict(surface, cells, slots):
    out = np.full(len(cells), np.nan)
    for idx, cell in enumerate(cells):
        if len(cell.positions) != 3: continue
        ijk = [slot_of(p, slots) for p in cell.positions]
        if any(v is None for v in ijk): continue
        out[idx] = surface[ijk[0], ijk[1], ijk[2]]
    return out

def subset_rows_by_pid(rows, pid_keep: set[int]) -> list:
    """Return new StrokeRows keeping only samples whose pid is in pid_keep (drop empty rows)."""
    out = []
    for r in rows:
        s = [smp for smp in r.samples if smp[2] in pid_keep]
        if s:
            out.append(StrokeRow(r.layout, r.positions, r.ngram, len(s), s))
    return out

def score_level(rows_level, slots, surfaces) -> dict:
    """Score all objectives on one covariate level's rows (pooled across AALTO layouts)."""
    cells = build_cells(rows_level, wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
    if not cells:
        return {"n_cells": 0, "n_scoreable": 0, "n_pids": 0, "ceiling": float("nan"), "rho": {}}
    obs = np.array([c.obs for c in cells])
    ceiling = split_half_ceiling(rows_level, wpm_lo=40, wpm_hi=140, bucket_width=20,
                                 min_cell_samples=10, n_boot=50, seed=0)
    preds = {p: predict(surfaces[p], cells, slots) for p in POOLS}
    ok = np.ones(len(cells), dtype=bool)
    for p in POOLS: ok &= np.isfinite(preds[p])
    ok &= np.isfinite(obs)
    n_ok = int(ok.sum())
    n_pids = len({s[2] for c, keep in zip(cells, ok) if keep for s in c.samples})
    rho = {}
    if n_ok >= 30:
        kept = [c for c, keep in zip(cells, ok, strict=True) if keep]
        obs_c = _bucket_centered(kept, obs[ok])
        for label, weights in WEIGHTINGS.items():
            total = sum(weights.values()); blended = np.zeros(n_ok)
            for p, w in weights.items(): blended += (w / total) * preds[p][ok]
            rho[label] = float(spearmanr(_bucket_centered(kept, blended), obs_c).statistic)
    return {"n_cells": len(cells), "n_scoreable": n_ok, "n_pids": n_pids,
            "ceiling": float(ceiling), "rho": rho}

def main() -> int:
    from keybo.geometry import ROW_STAGGERED_31
    slots = list(ROW_STAGGERED_31.slots)
    rows = load_strokes(SHIPPED_TSV, ngram_len=3, wpm_threshold=0, min_samples=10)
    surfaces = {p: load_surface(p) for p in POOLS}

    kb = pid_covariate("KEYBOARD_TYPE")
    # distribution of the covariate over pids actually present in the table
    table_pids = {s[2] for r in rows for s in r.samples}
    dist = Counter(kb.get(p, "MISSING") for p in table_pids)
    print("=== KEYBOARD_TYPE distribution over table pids ===")
    for k, v in dist.most_common(): print(f"  {k:10s} {v}")

    levels = ["full", "laptop"]
    results = {}
    for lvl in levels:
        keep = {p for p in table_pids if kb.get(p) == lvl}
        rows_lvl = subset_rows_by_pid(rows, keep)
        r = score_level(rows_lvl, slots, surfaces)
        results[lvl] = r
        print(f"\n=== held-out KEYBOARD_TYPE == {lvl} ===")
        print(f"  pids {r['n_pids']}, cells {r['n_cells']}, scoreable {r['n_scoreable']}, ceiling {r['ceiling']:.4f}")
        if r["rho"]:
            for lab in WEIGHTINGS:
                print(f"    {lab:22s} rho {r['rho'].get(lab, float('nan')):+.4f}")
        else:
            print("    UNINFORMATIVE: <30 scoreable cells")

    # PAIRED PER-FOLD DELTAS across the 2 levels
    print("\n=== PAIRED PER-LEVEL DELTAS (blend - ms/char), RAW rho ===")
    verdict = {}
    usable = [l for l in levels if results[l]["rho"]]
    if len(usable) < 1:
        print("UNINFORMATIVE: no level had >=30 scoreable cells.")
    else:
        base = {l: results[l]["rho"]["ms/char (shipped)"] for l in usable}
        for lab in WEIGHTINGS:
            if lab.startswith("ms/char"): continue
            deltas = {l: results[l]["rho"][lab] - base[l] for l in usable}
            wins = sum(1 for l in usable if deltas[l] > 0)
            print(f"  {lab}: " + "  ".join(f"{l} d{deltas[l]:+.4f}" for l in usable)
                  + f"  => wins {wins}/{len(usable)}")
            verdict[lab] = {"deltas": deltas, "wins": wins, "n_levels": len(usable)}
    out = {"kb_distribution": dict(dist), "levels": {l: results[l] for l in levels}, "verdict": verdict,
           "interpretation": "full & laptop are both AALTO-source, same ROW_STAGGERED geometry; "
                             "a KEYBOARD_TYPE hold-out is an AALTO frame, NOT a non-AALTO substitute."}
    Path("drivers-widen/arm2_keyboard_type.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote drivers-widen/arm2_keyboard_type.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
