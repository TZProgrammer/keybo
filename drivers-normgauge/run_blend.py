"""NORMGAUGE-1 deliverable 4 — run the combined objective and ask whether it finds anything.

Runs the pre-registered weighting AND the full sensitivity band (equal, three solos, and the
drop-POOL variant) at an IDENTICAL budget and seed set per cell, so a difference between cells
is the weighting and not the search. Then:

 * measures THIS arm's own search-noise sd, stating the (pool x replicate-structure x scale x
   statistic) quadruple -- no borrowed floor;
 * scores every champion and the incumbent field on the shipped 15-gauge frame AND on ms/char;
 * reports CONTESTED axis counts per pair, never a bare n/15 (`sfr` is a permutation invariant
   and `alt`/`imbalance` tie by construction for layouts sharing a hand partition, so those
   axes cannot be contested and must not pad a dominance count);
 * measures spearman(combined, ms/char) over a large pool, which is the test of whether the
   three-model gauge earns its keep against what the campaign already optimizes.

Frame: shipped `.standardized`, geometry-only g, BAKED 90 WPM, blend-v1. MODELLED ONLY.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from keybo.testkit import assert_module_under, assert_operands_computed  # noqa: E402

assert_module_under("keybo", "/tmp/normgauge")

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.scoring import model_norm as MN  # noqa: E402

HERE = Path(__file__).resolve().parent
ANCHORS = HERE / "anchors.json"
WEIGHTS = HERE / "weight-evidence.json"
OUT = HERE / "blend-report.json"
RUNS = HERE / "runs"

#: Identical across every weighting cell, so a cell-to-cell difference is the WEIGHT.
ISLANDS = 40
UNIQUE_EVALS_REQUESTED = 5_000_000
SEEDS = (20260728, 20260901, 20261015)

#: The incumbent field, verbatim from the shipped registry (`keybo.cli.analyze._EXTRA_NAMED` /
#: `keybo.layouts.NAMED_LAYOUTS`) plus the two campaign arms. GENERATED, never retyped: the
#: registry entries are asserted against the shipped dicts in `_field()`.
_ARMS = {
    "arm-A": "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
}
_FROM_REGISTRY = (
    "keybo-lsb",
    "keybo-lsb+lm",
    "keybo-c30m",
    "flagship-c3",
    "archive-1843",
    "archive-1846",
    "lsb-sib",
    "qwerty30m",
    "graphite",
    "semimak",
)
#: Pool for the ms/char correlation. Big, because the brief's +1.0000 was over only 9 layouts.
CORR_POOL_N, CORR_POOL_SEED = 300, 20260901

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def _field() -> dict[str, str]:
    """The incumbent field, pulled from the SHIPPED registries rather than retyped."""
    from keybo.cli.analyze import _EXTRA_NAMED
    from keybo.layouts import NAMED_LAYOUTS

    registry = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
    field = {}
    for name in _FROM_REGISTRY:
        if name not in registry:
            raise SystemExit(f"{name!r} is not in the shipped registry; the field moved")
        field[name] = registry[name]
    field.update(_ARMS)
    unscorable = {n: v for n, v in field.items() if not S.is_c30m(v)}
    if unscorable:
        raise SystemExit(f"field entries are not C30M-scorable: {sorted(unscorable)}")
    return field


# ---------------------------------------------------------------------------
# the blend search
# ---------------------------------------------------------------------------
class BlendSearch:
    """Memetic islands maximizing a weighted blend of normalized gauges.

    Same engine as the anchor search (full 435-swap steepest descent + perturbation), so the
    blend champion and the solo anchors are comparable: any difference between them is the
    OBJECTIVE, not the searcher.
    """

    def __init__(self, fits: MN.SurfaceFits, anchors: MN.Anchors, spec: MN.BlendSpec) -> None:
        self.fits = fits
        self.anchors = anchors
        self.spec = spec
        self.weights = np.array(
            [float(spec.weights.get(p, 0.0)) for p in fits.pools], dtype=np.float64
        )
        self.evaluated: set[bytes] = set()
        self._pairs = np.array(
            [(a, b) for a in range(30) for b in range(a + 1, 30)], dtype=np.int64
        )

    def _scores(self, permutations: list[np.ndarray]) -> np.ndarray:
        """Blended score per layout (HIGHER is better), via the normalized gauges."""
        raw = self.fits.fits_from_permutations(permutations)
        return self.anchors.normalize_array(raw, self.fits.pools) @ self.weights

    def _sweep(self, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        neighbours = np.repeat(permutation[None, :], len(self._pairs), axis=0)
        rows = np.arange(len(self._pairs))
        left, right = self._pairs[:, 0], self._pairs[:, 1]
        neighbours[rows, left], neighbours[rows, right] = permutation[right], permutation[left]
        values = self._scores(list(neighbours))
        for row in neighbours:
            self.evaluated.add(row.tobytes())
        return neighbours, values

    def ascend(self, permutation: np.ndarray) -> tuple[np.ndarray, float]:
        """Steepest ASCENT (the blend is higher-is-better) to a 2-opt local optimum."""
        current = float(self._scores([permutation])[0])
        self.evaluated.add(permutation.tobytes())
        while True:
            neighbours, values = self._sweep(permutation)
            best = int(values.argmax())
            if values[best] <= current:
                return permutation, current
            permutation, current = neighbours[best].copy(), float(values[best])

    def run(self, seed: int, islands: int = ISLANDS, budget: int = UNIQUE_EVALS_REQUESTED) -> dict:
        rng = np.random.default_rng(seed)
        best_score, best_perm = -float("inf"), None
        per_island = []
        for island in range(islands):
            start = np.concatenate([rng.permutation(30), [30]])
            local, score = self.ascend(start)
            island_best, island_perm = score, local
            share = budget // islands
            while len(self.evaluated) < share * (island + 1) and len(self.evaluated) < budget:
                kicked = island_perm.copy()
                for _ in range(rng.integers(3, 7)):
                    a, b = rng.choice(30, size=2, replace=False)
                    kicked[[a, b]] = kicked[[b, a]]
                local, score = self.ascend(kicked)
                if score > island_best:
                    island_best, island_perm = score, local
            per_island.append(island_best)
            if island_best > best_score:
                best_score, best_perm = island_best, island_perm
            if island % 10 == 0 or island == islands - 1:
                log(
                    f"    island {island + 1}/{islands} best={best_score:.6f} "
                    f"unique={len(self.evaluated):,}"
                )
            if len(self.evaluated) >= budget:
                break
        assert best_perm is not None
        return {
            "seed": seed,
            "best_blend": best_score,
            "best_layout": _layout_of(best_perm),
            "unique_evals_achieved": len(self.evaluated),
            "unique_evals_requested": budget,
            "islands": len(per_island),
            "per_island_best": per_island,
        }


def _layout_of(permutation: np.ndarray) -> str:
    out = [""] * 30
    for char_index, char in enumerate(S.C30M):
        out[int(permutation[char_index])] = char
    return "".join(out)


# ---------------------------------------------------------------------------
# the weightings to run
# ---------------------------------------------------------------------------
def weightings() -> dict[str, MN.BlendSpec]:
    """The pre-registered weighting plus the mandatory sensitivity band."""
    band = {
        "equal": MN.equal_weights(),
        "drop-pool": MN.BlendSpec(
            weights={"AALTO": 0.5, "COMMUNITY": 0.5},
            rule="drop POOL entirely (it is a measured union of the other two), 50/50",
        ),
        "solo-AALTO": MN.solo_weights("AALTO"),
        "solo-COMMUNITY": MN.solo_weights("COMMUNITY"),
        "solo-POOL": MN.solo_weights("POOL"),
    }
    # The REGISTERED cell's weights come from the evidence file, never from this module: a
    # weight typed here could not be distinguished from one chosen after seeing a result.
    if WEIGHTS.exists():
        decision = json.loads(WEIGHTS.read_text())["decision"]
        band["registered"] = MN.BlendSpec(
            weights=decision["weights"],
            rule=decision["rule"],
            evidence={"branch": decision["branch_taken"], "reasons": decision["reasons"]},
        )
    return band


# ---------------------------------------------------------------------------
# scoring the champions on the shipped frames
# ---------------------------------------------------------------------------
def gauge_table(layouts: dict[str, str]) -> dict:
    """The shipped 15-gauge frame + ms/char per layout, via the SHIPPED `analyze` path.

    Deliberately calls `keybo.cli.analyze.run` rather than re-deriving the gauges: the frozen
    board's conventions are non-obvious (comfort is divided by the FULL corpus bigram mass;
    scissor/imbalance come from `oxey.pattern_shares`, not from the severity model), and a
    re-derivation is how a second, silently-disagreeing definition gets into a report. A first
    draft of this function DID re-derive them and got two of the four wrong.

    ⚠ `oxey-style` is therefore computed FRESH by current code. Every pre-2026-07-28 ledger
    value is ~0.65-1.45 HIGHER (the nested `bad_redirect` fix landed that day), so an old value
    must never be mixed into this table.
    """
    import argparse
    import contextlib
    import io

    from keybo.cli import analyze as A

    parser = argparse.ArgumentParser()
    A.add_arguments(parser)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args([*layouts.values(), "--json", "--no-model-scores"])
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = A.run(args)
    if rc != 0:
        raise SystemExit(f"analyze returned rc={rc}")
    payload = json.loads(buffer.getvalue())

    by_text = {v: k for k, v in layouts.items()}
    rows = {}
    for key, row in payload["rows"].items():
        name = by_text.get(row["layout"], key)
        entry = {g: row["gauges"].get(g) for g in A.GAUGE_NAMES}
        entry["ms_per_char"] = (row.get("time") or {}).get("ms_per_char")
        entry["layout"] = row["layout"]
        rows[name] = entry
    return {
        "gauges": list(A.GAUGE_NAMES),
        "gauge_frame": payload["gauge_frame"],
        "corpus": payload["corpus"],
        "corpus_provenance": payload["corpus_provenance"],
        "rows": rows,
    }


#: Axes that CANNOT be contested, and why. Counting them in a dominance tally inflates it.
def uncontestable(rows: dict, left: str, right: str, gauges: list[str]) -> dict[str, str]:
    """Which axes are tied BY CONSTRUCTION for this pair, with the reason for each.

    `sfr` is a permutation invariant (same-finger REPEAT mass depends only on the character
    frequencies, not on placement), and `alt`/`imbalance` are determined by the hand PARTITION,
    so two layouts sharing a partition tie on them identically. Reporting a bare n/15 over
    axes that cannot differ is how a dominance claim gets padded.
    """
    reasons = {}
    for gauge in gauges:
        a, b = rows[left].get(gauge), rows[right].get(gauge)
        if a is None or b is None:
            reasons[gauge] = "not scorable for one of the pair"
        elif a == b:
            reasons[gauge] = "exactly tied (invariant or shared hand partition)"
    return reasons


def contested_counts(table: dict, left: str, right: str) -> dict:
    """Per-pair win/loss over the axes that CAN differ, never a bare n/15.

    Direction: every one of these 15 gauges is lower-is-better on the frozen board, so `left`
    wins an axis when its value is strictly smaller.
    """
    rows, gauges = table["rows"], table["gauges"]
    dead = uncontestable(rows, left, right, gauges)
    contested = [g for g in gauges if g not in dead]
    wins = [g for g in contested if rows[left][g] < rows[right][g]]
    losses = [g for g in contested if rows[left][g] > rows[right][g]]
    return {
        "pair": [left, right],
        "n_gauges_total": len(gauges),
        "n_contested": len(contested),
        "uncontestable": dead,
        "wins": wins,
        "losses": losses,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "dominates": len(losses) == 0 and len(wins) > 0,
        "reading": (
            f"{len(wins)}/{len(contested)} CONTESTED axes ("
            f"{len(dead)} of {len(gauges)} cannot be contested for this pair)"
        ),
    }


def search_noise(results: dict[str, list[dict]]) -> dict:
    """THIS arm's own search-noise sd, with the quadruple that scopes it stated.

    A resolution floor is a property of a (pool x replicate-structure x scale x statistic)
    QUADRUPLE, not of a metric or a corpus, and may be quoted for another design only if all
    four match. This arm measures its own rather than borrowing either of the two prior values
    on this engine (0.0492 and 0.0995) -- a borrowed floor has been wrong by 2x here.
    """
    per_cell = {}
    for label, runs in results.items():
        values = [r["best_blend"] for r in runs]
        assert_operands_computed(values, f"{label} best-blend per seed")
        per_cell[label] = {
            "seeds": [r["seed"] for r in runs],
            "best_blend_per_seed": values,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "range": float(max(values) - min(values)),
            "identical_champion": len({r["best_layout"] for r in runs}) == 1,
            "champions": sorted({r["best_layout"] for r in runs}),
        }
    sds = [v["sd"] for v in per_cell.values() if np.isfinite(v["sd"])]
    return {
        "quadruple": {
            "pool": "6 weighting cells x 3 seeds on the blend objective over C30M permutations, "
            "blend-v1 corpus, shipped .standardized surfaces",
            "replicate_structure": "independent RNG seeds of the SAME memetic-island searcher "
            f"at an identical budget ({UNIQUE_EVALS_REQUESTED:,} unique "
            f"evals requested, {ISLANDS} islands)",
            "scale": "normalized blend units (0 = random-pool mean, 1 = per-model optimum)",
            "statistic": "across-seed sd of the best blend found, per weighting cell",
        },
        "per_cell": per_cell,
        "pooled_sd_across_cells": float(np.mean(sds)) if sds else float("nan"),
        "max_sd_across_cells": float(max(sds)) if sds else float("nan"),
        "warning": "this sd is in NORMALIZED units and is NOT comparable to the campaign's "
        "ms/char search-noise values (0.0492 / 0.0995 ms/char) -- different scale, "
        "different statistic, so neither may be substituted for the other",
    }


def main() -> int:
    if not ANCHORS.exists() or not WEIGHTS.exists():
        raise SystemExit(f"need {ANCHORS} and {WEIGHTS} first")
    fits = MN.SurfaceFits()
    fits.assert_batch_invariant(S.C30M)
    anchors = MN.Anchors.read(ANCHORS)
    anchors.assert_direction()
    anchors.assert_matches_surfaces(fits, anchors.provenance["probe_layout"])
    log("gates: evaluator bit-stable, anchors directional and undrifted")

    band = weightings()
    for label, spec in band.items():
        log(f"weighting {label}: {spec.describe()}")

    results: dict[str, list[dict]] = {}
    for label, spec in band.items():
        results[label] = []
        for seed in SEEDS:
            cell = RUNS / f"blend-{label}-{seed}.json"
            if cell.exists():
                results[label].append(json.loads(cell.read_text()))
                log(f"  reusing {cell.name}")
                continue
            log(f"  search {label} seed={seed}")
            payload = BlendSearch(fits, anchors, spec).run(seed)
            payload["weighting"] = label
            payload["weights"] = dict(spec.weights)
            tmp = cell.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.rename(cell)
            results[label].append(payload)
    with open(HERE / "blend-runs.json", "w") as handle:
        json.dump(results, handle, indent=1)

    # -- MY OWN search-noise sd, before any champion is compared to any other ------------
    noise = search_noise(results)
    log(
        f"search noise: pooled across-seed sd = {noise['pooled_sd_across_cells']:.6f} "
        f"normalized units (max {noise['max_sd_across_cells']:.6f})"
    )
    for label, cell in noise["per_cell"].items():
        log(
            f"  {label:16s} sd={cell['sd']:.6f} identical_champion={cell['identical_champion']} "
            f"({len(cell['champions'])} distinct)"
        )

    # -- champions of record, and whether the weighting is load-bearing -------------------
    champions = {}
    for label, runs in results.items():
        best = max(runs, key=lambda r: r["best_blend"])
        champions[label] = best["best_layout"]
    distinct = sorted(set(champions.values()))
    log(f"{len(distinct)} DISTINCT champions across {len(champions)} weighting cells")

    # -- score everything on the shipped 15-gauge frame + ms/char -------------------------
    field = _field()
    scored = dict(field)
    for label, layout in champions.items():
        scored[f"blend:{label}"] = layout
    anchors = MN.Anchors.read(ANCHORS)
    for pool, layout in anchors.provenance["one_provenance"]["layout_of_record"].items():
        scored[f"anchor:{pool}"] = layout
    log(f"scoring {len(scored)} layouts on the shipped 15-gauge frame + ms/char")
    table = gauge_table(scored)

    # -- the normalized gauges for every scored layout ------------------------------------
    fits2 = MN.SurfaceFits()
    order = list(scored)
    raw = fits2.fits(list(scored.values()))
    norm = anchors.normalize_array(raw, fits2.pools)
    band = weightings()
    gauge_rows = {}
    for index, name in enumerate(order):
        entry = {MN.GAUGE_OF_POOL[p]: float(norm[index, n]) for n, p in enumerate(fits2.pools)}
        entry["fit_ms"] = {p: float(raw[index, n]) for n, p in enumerate(fits2.pools)}
        for label, spec in band.items():
            entry[f"blend:{label}"] = float(
                spec.blend_array(norm[index : index + 1], fits2.pools)[0]
            )
        gauge_rows[name] = entry

    # -- does the combined gauge earn its keep against ms/char? ---------------------------
    from scipy.stats import spearmanr

    pool = MN.random_pool(CORR_POOL_N, CORR_POOL_SEED)
    pool_layouts = {f"r{n:03d}": _layout_of(p) for n, p in enumerate(pool)}
    pool_table = gauge_table(pool_layouts)
    pool_norm = anchors.normalize_array(fits2.fits(list(pool_layouts.values())), fits2.pools)
    ms = np.array([pool_table["rows"][n]["ms_per_char"] for n in pool_layouts])
    assert_operands_computed(ms.tolist(), "ms/char over the correlation pool")
    correlations = {}
    for n, p in enumerate(fits2.pools):
        # ms/char is lower-is-better while the gauges are higher-is-better, so a NEGATIVE
        # spearman here means they AGREE. Reported as agreement to avoid a sign trap.
        rho = spearmanr(pool_norm[:, n], ms).statistic
        correlations[MN.GAUGE_OF_POOL[p]] = {
            "spearman_vs_ms_per_char": float(rho),
            "agreement": float(-rho),
        }
    for label, spec in band.items():
        rho = spearmanr(spec.blend_array(pool_norm, fits2.pools), ms).statistic
        correlations[f"blend:{label}"] = {
            "spearman_vs_ms_per_char": float(rho),
            "agreement": float(-rho),
        }
    log(f"agreement with ms/char over a {CORR_POOL_N}-layout random pool (1.0 = rank-identical):")
    for key, value in correlations.items():
        log(f"  {key:22s} {value['agreement']:+.4f}")

    # -- per-pair CONTESTED counts vs the incumbent field ---------------------------------
    pairs = {}
    registered_champion = champions["registered"]
    registered_key = next(k for k, v in scored.items() if v == registered_champion)
    for name in field:
        pairs[f"{registered_key} vs {name}"] = contested_counts(table, registered_key, name)
    dominated = [k for k, v in pairs.items() if v["dominates"]]
    log(
        f"the registered champion dominates {len(dominated)} of {len(pairs)} field layouts "
        f"on CONTESTED axes"
    )

    payload = {
        "frame": "shipped .standardized, geometry-only g, BAKED 90 WPM",
        "frame_caveat": MN.frame_caveat(),
        "interpretation": MN.interpretation_note(),
        "corpus": table["corpus"],
        "gauge_frame": table["gauge_frame"],
        "oxey_style_freshness": "computed by TODAY's code; every pre-2026-07-28 ledger value is "
        "~0.65-1.45 HIGHER (nested bad_redirect fix) and must not be "
        "mixed in",
        "weightings": {k: {"weights": dict(v.weights), "rule": v.rule} for k, v in band.items()},
        "runs": results,
        "search_noise": noise,
        "champions": champions,
        "n_distinct_champions": len(distinct),
        "distinct_champions": distinct,
        "gauge_table": table,
        "normalized_rows": gauge_rows,
        "ms_per_char_agreement": correlations,
        "ms_per_char_pool": {
            "n": CORR_POOL_N,
            "seed": CORR_POOL_SEED,
            "note": "a RANDOM pool, deliberately larger than the 9-layout "
            "adoption set the +1.0000 aalto/ms-char figure came from",
        },
        "contested_pairs": pairs,
        "dominated_field_layouts": dominated,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
