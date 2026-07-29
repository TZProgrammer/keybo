"""NORMGAUGE-1 — turn evidence into weights, by the rule pre-registered in PREREGISTRATION.md.

Runs the four candidate rules and applies the registered decision tree. Nothing here chooses a
weight after seeing a layout: the tree, the falsifier, and the shrinkage form are all fixed in
the prereg commit that precedes this file's results.

 (a) precision / sample-size   -- per-surface-cell training support, measured here
 (b) independence correction   -- POOL's unique variance share, measured at fit level
 (c) held-out predictive skill -- each source's data is ALREADY out-of-sample for the other's
                                  surface (the two sources are disjoint), so no refit is
                                  needed. Registered PRIMARY, with an explicit falsifier.
 (d) equal weights             -- the fallback, reported as a reference either way

Frame: shipped `.standardized`, geometry-only g, baked 90 WPM, blend-v1. MODELLED ONLY.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import numpy as np  # noqa: E402

from keybo.testkit import (  # noqa: E402
    assert_discriminating,
    assert_module_under,
    assert_operands_computed,
)

assert_module_under("keybo", "/tmp/normgauge")

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.scoring import model_norm as MN  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "weight-evidence.json"
E2E = Path("/local/home/zegertho/keybo-e2e")
COMM = Path("/local/home/zegertho/repos/keybo/data/community/processed")

#: The exact per-source training subsets, identified by scissorsupport via EXACT
#: practice-term key-set match (not by count) and re-verified here by label.
AALTO_LABELS = frozenset({"azerty", "dvorak", "qwerty", "qwertz"})
COMM_LABELS = frozenset(
    {
        "colemak@rowStagger#alite",
        "custom-aa426873@rowStagger#vg",
        "custom-d42a1f92@rowStagger#ddn",
        "mtgap-variant@rowStagger#richarddavison",
    }
)
#: The training recipe's own cell filter — so counted support is the support the FIT saw.
CELL_KW = dict(wpm_lo=40, wpm_hi=140, bucket_width=20, min_cell_samples=10)
#: Bootstrap resamples for the held-out CIs, and the seed. Registered before results.
N_BOOT, BOOT_SEED = 2000, 20260728
#: The random pool the (b) regression is measured on. Larger than the 100-layout anchor pool
#: because this is a variance-decomposition, not an anchor.
REG_POOL_N, REG_POOL_SEED = 400, 20260728

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


SLOT_OF = None  # built lazily in support_map()


# ---------------------------------------------------------------------------
# (a) precision: per-surface-cell training support
# ---------------------------------------------------------------------------
def support_map(path: Path, keep: frozenset[str]) -> dict:
    """Per-31^3-cell training support for one source, filtered as the recipe filters.

    Returns counts over the 29,791 flattened cells plus the covered-cell summary the
    registered ESS form consumes. A row maps to exactly one cell because the surface is
    indexed by SLOT INDEX in `ROW_STAGGERED_30.slots` order with space at 30 — the same
    mapping the feature pipeline uses.
    """
    global SLOT_OF
    if SLOT_OF is None:
        from keybo.geometry import ROW_STAGGERED_30

        SLOT_OF = {p: n for n, p in enumerate(ROW_STAGGERED_30.slots)}
        SLOT_OF[ROW_STAGGERED_30.space_position] = 30

    counts = np.zeros(29791, dtype=np.int64)
    rows_per_cell = np.zeros(29791, dtype=np.int64)
    seen_labels: dict[str, int] = {}
    kept = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            layout, pos_str, ngram, _freq, *tokens = parts
            seen_labels[layout] = seen_labels.get(layout, 0) + 1
            if layout not in keep or len(ngram) != 3:
                continue
            n = 0
            for token in tokens:
                comma = token.find(",")
                if comma <= 1:
                    continue
                try:
                    if int(token[1:comma]) >= CELL_KW["wpm_lo"]:
                        n += 1
                except ValueError:
                    continue
            if n < CELL_KW["min_cell_samples"]:
                continue
            try:
                positions = tuple(
                    tuple(t) for t in json.loads(pos_str.replace("(", "[").replace(")", "]"))
                )
                a, b, c = (SLOT_OF[p] for p in positions)
            except (KeyError, ValueError):
                continue
            counts[a * 961 + b * 31 + c] += n
            rows_per_cell[a * 961 + b * 31 + c] += 1
            kept += 1
    covered = counts > 0
    return {
        "path": str(path),
        "labels_kept": sorted(keep),
        "labels_in_file": seen_labels,
        "samples": int(counts.sum()),
        "rows_kept": kept,
        "cells_covered": int(covered.sum()),
        "cells_total": 29791,
        "coverage": float(covered.mean()),
        "median_samples_per_covered_cell": float(np.median(counts[covered])),
        "min_samples_per_covered_cell": int(counts[covered].min()),
        "_counts": counts,
    }


def registered_ess(summary: dict) -> float:
    """The ESS form REGISTERED IN THE PREREG, quoted here verbatim in code.

        ESS_m = (cells covered by m) x sqrt(median samples per covered cell of m)

    Linear in coverage (what makes a surface able to answer at all), sqrt in depth (the
    standard-error scaling of a per-cell mean). A RAW sample-count ratio would give COMMUNITY
    a weight of ~0.001 and thereby DELETE the source the user explicitly asked to include —
    a design answer wearing an evidence costume. This form is fixed before the numbers are
    combined so it cannot be tuned.
    """
    return summary["cells_covered"] * np.sqrt(summary["median_samples_per_covered_cell"])


# ---------------------------------------------------------------------------
# (b) independence: POOL's unique variance share at fit level
# ---------------------------------------------------------------------------
def independence(fits: MN.SurfaceFits) -> dict:
    """Regress POOL's fit on AALTO's and COMMUNITY's over a random pool.

    Fit level, not cell level: fit level is what the optimizer actually sees. The cell-level
    number is reported alongside as a second, independent view of the same structure.
    """
    pool = MN.random_pool(REG_POOL_N, REG_POOL_SEED)
    F = fits.fits_from_permutations(pool)
    assert_discriminating(F[:, 0].tolist(), "AALTO fits over the regression pool")
    A, C, P = F[:, 0], F[:, 1], F[:, 2]
    X = np.column_stack([np.ones_like(A), A, C])
    beta, *_ = np.linalg.lstsq(X, P, rcond=None)
    resid = P - X @ beta
    r2 = 1.0 - (resid**2).sum() / ((P - P.mean()) ** 2).sum()

    flat = {p: S.load_surface(f"{p}_{S.DEFAULT_FAMILY}").ravel() for p in S.POOLS}
    Xc = np.column_stack([np.ones_like(flat["AALTO"]), flat["AALTO"], flat["COMMUNITY"]])
    bc, *_ = np.linalg.lstsq(Xc, flat["POOL"], rcond=None)
    rc = flat["POOL"] - Xc @ bc
    r2c = 1.0 - (rc**2).sum() / ((flat["POOL"] - flat["POOL"].mean()) ** 2).sum()
    return {
        "fit_level": {
            "pool_n": REG_POOL_N,
            "pool_seed": REG_POOL_SEED,
            "intercept": float(beta[0]),
            "beta_aalto": float(beta[1]),
            "beta_community": float(beta[2]),
            "beta_sum": float(beta[1] + beta[2]),
            "r2": float(r2),
            "unique_variance_share": float(1.0 - r2),
            "resid_sd_ms": float(resid.std(ddof=1)),
            "resid_sd_pct_of_pool_sd": float(100 * resid.std(ddof=1) / P.std(ddof=1)),
            "fit_corr": np.corrcoef(F.T).tolist(),
        },
        "cell_level": {
            "intercept": float(bc[0]),
            "beta_aalto": float(bc[1]),
            "beta_community": float(bc[2]),
            "beta_sum": float(bc[1] + bc[2]),
            "r2": float(r2c),
            "unique_variance_share": float(1.0 - r2c),
        },
        "reading": (
            "POOL is a near-symmetric blend of the other two with most of its fit variance "
            "explained by them, so it is NOT an independent third vote. Its registered weight "
            "is its unique variance share (1 - r2) at fit level."
        ),
    }


# ---------------------------------------------------------------------------
# (c) held-out predictive skill — the registered PRIMARY rule
# ---------------------------------------------------------------------------
def held_out(surface_pool: str, source_path: Path, keep: frozenset[str]) -> dict:
    """Score one surface against a source it never trained on, on that source's own cells.

    Uses the campaign's own cell machinery and its own bucket-centered Spearman, and divides
    by the held-out source's split-half reliability ceiling (participant-bisected,
    Spearman-Brown length-corrected) so a noisy-target source is not penalized for its noise.

    The bucket-centering matters: the wpm->duration axis is a model INPUT, so an uncentered
    correlation would award credit for information the model was handed.
    """
    from keybo.data.strokes import load_strokes
    from keybo.training.validate import (
        _centered_spearman,
        build_cells,
        split_half_ceiling,
    )

    # Cache the parse: the AALTO table is 571 MB and takes ~200 s to load, and this driver is
    # re-run while iterating. Keyed by (path, mtime, size, labels) so a changed table invalidates
    # it rather than silently serving stale rows.
    cache_key = f"{source_path.name}-{source_path.stat().st_mtime_ns}-{source_path.stat().st_size}"
    cache_key += "-" + hashlib.sha256(",".join(sorted(keep)).encode()).hexdigest()[:12]
    cache = HERE / "cache" / f"rows-{cache_key}.pkl"
    if cache.exists():
        with open(cache, "rb") as handle:
            rows = pickle.load(handle)
        log(f"    reused cached rows for {source_path.name} ({len(rows)} rows)")
    else:
        rows = [
            r
            for r in load_strokes(
                str(source_path), 3, CELL_KW["wpm_lo"], CELL_KW["min_cell_samples"]
            )
            if r.layout in keep
        ]
        cache.parent.mkdir(exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(rows, handle, protocol=5)
        tmp.rename(cache)
    cells = build_cells(rows, **CELL_KW)
    surface = S.load_surface(f"{surface_pool}_{S.DEFAULT_FAMILY}")

    from keybo.geometry import ROW_STAGGERED_30

    slot_of = {p: n for n, p in enumerate(ROW_STAGGERED_30.slots)}
    slot_of[ROW_STAGGERED_30.space_position] = 30

    usable, pred, obs, per_cell_samples = [], [], [], []
    for cell in cells:
        try:
            a, b, c = (slot_of[p] for p in cell.positions)
        except KeyError:
            continue  # a position off the 31-slot board
        usable.append(cell)
        pred.append(float(surface[a, b, c]))
        obs.append(float(cell.obs))
        per_cell_samples.append(cell.samples)

    assert_operands_computed(pred, f"{surface_pool} predictions on held-out {source_path.name}")
    assert_discriminating(pred, f"{surface_pool} predictions on held-out {source_path.name}")
    pred_a, obs_a = np.array(pred), np.array(obs)
    rho = _centered_spearman(usable, pred_a, obs_a)
    ceiling = split_half_ceiling(rows, **CELL_KW)

    # The point estimate under the PLAIN-MEAN aggregation the bootstrap replicates use. Reported
    # alongside the shipped IQR-mean value (AMENDMENT 1 A1.2): re-running an IQR-mean per cell per
    # resample is unaffordable, so the two aggregations are compared instead of assumed equal.
    obs_mean = np.array([float(np.mean([s[1] for s in c.samples])) for c in usable])
    rho_plain = _centered_spearman(usable, pred_a, obs_mean)

    # CLUSTER bootstrap over PARTICIPANTS that RE-AGGREGATES each cell's value from the drawn
    # participants' own samples (AMENDMENT 1 A1.2). The registered inclusion-only version was a
    # NO-OP on the AALTO side -- 0.999992 of its 24,079 cells survived every resample, so the cell
    # VALUES never moved and the interval collapsed toward zero width, manufacturing significance
    # on the side with the most data. Resampling participants INTO the cell values is what makes
    # participant uncertainty reach the statistic.
    pid_index: dict[int, int] = {}
    for samples in per_cell_samples:
        for sample in samples:
            pid_index.setdefault(sample[2], len(pid_index))
    n_pids = len(pid_index)
    # Flat (cell, pid_slot, duration) arrays so a replicate is vectorized rather than a Python
    # loop over ~24k cells x 2000 replicates.
    flat_cell, flat_pid, flat_dur = [], [], []
    for index, samples in enumerate(per_cell_samples):
        for sample in samples:
            flat_cell.append(index)
            flat_pid.append(pid_index[sample[2]])
            flat_dur.append(sample[1])
    flat_cell = np.array(flat_cell, dtype=np.int64)
    flat_pid = np.array(flat_pid, dtype=np.int64)
    flat_dur = np.array(flat_dur, dtype=np.float64)

    rng = np.random.default_rng(BOOT_SEED)
    boot, survivors = [], []
    n_cells = len(usable)
    for _ in range(N_BOOT):
        # Participant multiplicities: how many times each pid was drawn (0 = excluded).
        multiplicity = np.bincount(rng.integers(0, n_pids, n_pids), minlength=n_pids).astype(
            np.float64
        )
        weight = multiplicity[flat_pid]
        total = np.bincount(flat_cell, weights=weight, minlength=n_cells)
        summed = np.bincount(flat_cell, weights=weight * flat_dur, minlength=n_cells)
        alive = total > 0
        if int(alive.sum()) < 3:
            continue
        keep_idx = np.flatnonzero(alive)
        replicate = summed[keep_idx] / total[keep_idx]
        value = _centered_spearman([usable[n] for n in keep_idx], pred_a[keep_idx], replicate)
        if np.isfinite(value):
            boot.append(value)
            survivors.append(int(alive.sum()))
    boot_a = np.array(boot)
    lo, hi = np.percentile(boot_a, [2.5, 97.5]) if len(boot_a) else (np.nan, np.nan)
    return {
        "surface": surface_pool,
        "held_out_source": source_path.name,
        "held_out_labels": sorted(keep),
        "n_cells": len(usable),
        "n_rows": len(rows),
        "n_participants": n_pids,
        "participants": sorted(pid_index)[:16],
        "median_pids_per_cell": float(np.median([len({s[2] for s in c.samples}) for c in usable])),
        "rho": float(rho),
        "rho_plain_mean_aggregation": float(rho_plain),
        "aggregation_gap": float(rho - rho_plain),
        "ceiling": float(ceiling),
        "rho_over_ceiling": float(rho / ceiling) if ceiling else float("nan"),
        "boot_n": int(len(boot_a)),
        "boot_estimator": "cluster bootstrap over participants, re-aggregating each cell's value "
        "from the drawn participants' samples (plain mean); see AMENDMENT 1 A1.2",
        "boot_median_surviving_cells": float(np.median(survivors)) if survivors else float("nan"),
        "rho_ci95": [float(lo), float(hi)],
        "rho_boot_se": float(boot_a.std(ddof=1)) if len(boot_a) > 1 else float("nan"),
        # The falsifier compares rho/ceiling values, so its SE must be on the rho/ceiling SCALE.
        # Using the rho-scale SE against a rho/ceiling gap is a unit mismatch that would make the
        # falsifier too easy or too hard depending on which side's ceiling is larger -- a
        # borrowed-ruler error in miniature. The ceiling is treated as fixed here (it is estimated
        # from the same data, so this UNDERSTATES the true uncertainty; stated, not hidden).
        "rho_over_ceiling_boot_se": (
            float(boot_a.std(ddof=1) / ceiling) if len(boot_a) > 1 and ceiling else float("nan")
        ),
        "ci_treats_ceiling_as_fixed": (
            "the ceiling is estimated from the same held-out data but is not resampled, so this "
            "interval UNDERSTATES total uncertainty -- conservative for a falsifier that fires on "
            "a WIDE interval, so it cannot rescue branch (c)"
        ),
        "crosses_zero": bool(not np.isfinite(lo) or (lo <= 0.0 <= hi)),
        "note": (
            "the two sources are DISJOINT (aalto pids <200000, community pids 200001-200007), so "
            "this is genuinely out-of-sample without any refit; bucket-centered because the wpm "
            "axis is a model input"
        ),
    }


# ---------------------------------------------------------------------------
# the registered decision tree
# ---------------------------------------------------------------------------
def decide(support: dict, indep: dict, cross: list[dict]) -> dict:
    """Apply PREREGISTRATION.md §2.5 verbatim. No branch is chosen after seeing a layout."""
    w_pool = indep["fit_level"]["unique_variance_share"]

    by_surface = {c["surface"]: c for c in cross}
    a, c = by_surface.get("AALTO"), by_surface.get("COMMUNITY")
    reasons: list[str] = []

    # STEP 2, branch (c): the registered falsifier.
    c_usable = True
    if a is None or c is None:
        c_usable, why = False, "a cross-prediction cell is missing"
    elif a["crosses_zero"] or c["crosses_zero"]:
        c_usable = False
        why = (
            f"a bootstrap CI crosses 0 (AALTO {a['rho_ci95']}, COMMUNITY {c['rho_ci95']}) — "
            f"the skill is not distinguishable from none"
        )
    else:
        gap = abs(a["rho_over_ceiling"] - c["rho_over_ceiling"])
        # SEs on the SAME scale as the gap (rho/ceiling), not on the raw rho scale.
        pooled_se = float(np.hypot(a["rho_over_ceiling_boot_se"], c["rho_over_ceiling_boot_se"]))
        if gap <= pooled_se:
            c_usable = False
            why = (
                f"the two rho/ceiling values are within one pooled bootstrap SE "
                f"(gap {gap:.4f} <= SE {pooled_se:.4f}) — not separable"
            )
        else:
            why = (
                f"rho/ceiling separates by {gap:.4f} > pooled SE {pooled_se:.4f}, and neither "
                f"CI crosses 0"
            )
    reasons.append(f"(c) held-out: {'USABLE' if c_usable else 'REFUTED'} — {why}")

    if c_usable:
        split = {
            "AALTO": max(0.0, a["rho_over_ceiling"]),
            "COMMUNITY": max(0.0, c["rho_over_ceiling"]),
        }
        rule = "held-out predictive skill (rho/ceiling), POOL at its unique variance share"
    else:
        ess = {m: registered_ess(support[m]) for m in ("AALTO", "COMMUNITY")}
        split = dict(ess)
        rule = "precision (registered ESS = coverage x sqrt(median depth)), POOL at its unique variance share"
        reasons.append(
            f"(a) precision: ESS AALTO {ess['AALTO']:.1f} vs COMMUNITY {ess['COMMUNITY']:.1f} "
            f"= {ess['AALTO'] / ess['COMMUNITY']:.3f}x"
        )

    total = sum(split.values())
    if total <= 0:
        raise SystemExit("the AALTO:COMMUNITY split is degenerate; no rule is identifiable")
    scale = 1.0 - w_pool
    weights = {
        "AALTO": scale * split["AALTO"] / total,
        "COMMUNITY": scale * split["COMMUNITY"] / total,
        "POOL": w_pool,
    }
    reasons.append(
        f"(b) independence: POOL's unique variance share is {w_pool:.4f} at fit level "
        f"(R2={indep['fit_level']['r2']:.5f}), so it takes that weight and the other two split "
        f"the remaining {scale:.4f}"
    )
    equal_effective = {
        "AALTO": (1 / 3) + (1 / 3) * indep["fit_level"]["beta_aalto"],
        "COMMUNITY": (1 / 3) + (1 / 3) * indep["fit_level"]["beta_community"],
        "POOL_unique": (1 / 3) * w_pool,
    }
    return {
        "weights": weights,
        "rule": rule,
        "branch_taken": "(c) held-out" if c_usable else "(a) precision + (b) independence",
        "reasons": reasons,
        "equal_weights_are_not_neutral": {
            "effective_source_loadings_under_equal_weights": equal_effective,
            "reading": (
                "under (1/3,1/3,1/3) the AALTO+COMMUNITY consensus is counted ~1.5x because "
                "POOL re-votes it; POOL's own unique signal gets only a third of its already "
                "small share"
            ),
        },
    }


def main() -> int:
    fits = MN.SurfaceFits()
    fits.assert_batch_invariant(S.C30M)
    log("gate: evaluator bit-stable")

    log("(a) scanning per-cell training support")
    support = {
        "AALTO": support_map(E2E / "tristrokes31_cond_v1.tsv", AALTO_LABELS),
        "COMMUNITY": support_map(COMM / "tristrokes_last_community.tsv", COMM_LABELS),
    }
    for name, summary in support.items():
        log(
            f"  {name}: {summary['samples']:,} samples over {summary['cells_covered']:,} cells "
            f"(median {summary['median_samples_per_covered_cell']:.0f}/cell, "
            f"min {summary['min_samples_per_covered_cell']})"
        )
    ratio = support["AALTO"]["samples"] / support["COMMUNITY"]["samples"]
    log(
        f"  whole-surface sample ratio = {ratio:.2f}x  (the brief's 643x is the "
        f"scissor-neighbourhood covered-pair-filtered figure; see PREREGISTRATION.md 2.1)"
    )

    log("(b) measuring POOL's unique variance share")
    indep = independence(fits)
    log(
        f"  fit level: POOL = {indep['fit_level']['beta_aalto']:.6f}*AALTO + "
        f"{indep['fit_level']['beta_community']:.6f}*COMMUNITY, R2="
        f"{indep['fit_level']['r2']:.5f}, unique={indep['fit_level']['unique_variance_share']:.4f}"
    )

    log("(c) held-out cross-prediction (no refit: the sources are disjoint)")
    cross = [
        held_out("AALTO", COMM / "tristrokes_last_community.tsv", COMM_LABELS),
        held_out("COMMUNITY", E2E / "tristrokes31_cond_v1.tsv", AALTO_LABELS),
    ]
    for cell in cross:
        log(
            f"  {cell['surface']} surface -> held-out {cell['held_out_source']}: "
            f"rho={cell['rho']:+.4f} ceiling={cell['ceiling']:.4f} "
            f"rho/ceil={cell['rho_over_ceiling']:+.4f} CI95={cell['rho_ci95']} "
            f"(cells {cell['n_cells']}, pids {cell['n_participants']})"
        )

    decision = decide(support, indep, cross)
    for reason in decision["reasons"]:
        log(f"  {reason}")
    log(
        "DECIDED weights: "
        + "  ".join(f"{MN.GAUGE_OF_POOL[p]}={w:.4f}" for p, w in decision["weights"].items())
    )

    payload = {
        "frame": "standardized",
        "frame_caveat": MN.frame_caveat(),
        "corpus": str(fits.trigram_path),
        "support": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in support.items()
        },
        "support_ratio_whole_surface": ratio,
        "brief_643x_correction": {
            "claim_in_brief": "AALTO 7,669,316 vs COMMUNITY 11,930 = 643x",
            "actual_scope": "scissor-neighbourhood cell groups under a COVERED-PAIR filter "
            "(state/scissorsupport/artifacts/ss2d_support_filtered.json)",
            "asymmetry": "AALTO's count is IDENTICAL in the filtered and unfiltered artifacts; "
            "COMMUNITY loses 92.1% (151,365 -> 11,930)",
            "unfiltered_same_groups": 7669316 / 151365,
            "whole_stroke_table": 18535823 / 401543,
            "measured_here_on_surface_cells": ratio,
            "verdict": "the CONCLUSION (AALTO is far better supported) is confirmed and "
            "strengthened; the CONSTANT 643x is mis-scoped and is not used as a "
            "reliability ratio",
        },
        "independence": indep,
        "held_out": cross,
        "decision": decision,
        "provenance": {
            "cell_kw": CELL_KW,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "numpy": np.__version__,
        },
    }
    OUT.write_text(json.dumps(payload, indent=1))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
