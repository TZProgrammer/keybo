"""NORMGAUGE-1 — build the 0 and 1 anchors per model on the SHIPPED `.standardized` frame.

The "0" is the MEAN fit of a fixed random pool (n=100, seed 20260728 — the user's n, which
MODELNORM-1 verified sufficient). The "1" is a per-model memetic-island search's best fit, at
an IDENTICAL budget and island structure for every model and every seed.

FREE POSITIVE CONTROL, and it is the reason this arm can size its own search: AALTO's
`.native` and `.standardized` arrays are byte-identical, so MODELNORM-1's AALTO champion is a
10M-unique-eval optimum on EXACTLY this frame. The prereg fixes the bar at +0.05% of
223236317224.4177 and reports the shortfall if missed rather than shipping a weak anchor.

Everything is generated or asserted; no number is retyped. Run:
    PYTHONPATH=/tmp/normgauge/src python drivers-normgauge/build_anchors.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from keybo.testkit import assert_module_under  # noqa: E402

assert_module_under("keybo", "/tmp/normgauge")

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.scoring import model_norm as MN  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "anchors.json"
EVIDENCE = HERE / "anchors-evidence.json"

#: The user's n and a fixed pool seed. n=100 is the USER's number and MODELNORM-1 verified it
#: (n=1000 moves the zero <1 SE, ranking unchanged at 100/1000/10000). Not silently inflated.
POOL_N, POOL_SEED = 100, 20260728
#: Stability check only — reported, never shipped as the anchor.
STABILITY_N = 1000

#: Per-model search budget. IDENTICAL across models and seeds, so no model's anchor can be
#: flattered by a longer search than another's.
ISLANDS = 40
UNIQUE_EVALS_REQUESTED = 5_000_000
SEEDS = (20260728, 20260901, 20261015)

#: The prereg's acceptance bar for the AALTO anchor, and the target it is measured against.
AALTO_TARGET_FIT = 223236317224.4177
AALTO_TARGET_LAYOUT = "lnfdg-,yehcrstmaoiupxqbwv.k'jz"
AALTO_BAR_PCT = 0.05

#: Drift probe: an arbitrary FIXED layout whose fits are recorded so a later run can prove the
#: surfaces/corpus/evaluator did not move under the anchors.
PROBE = S.C30M

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the search: memetic islands over a single model's fit
# ---------------------------------------------------------------------------
_PAIRS = np.array([(a, b) for a in range(30) for b in range(a + 1, 30)], dtype=np.int64)


class SingleModelSearch:
    """Steepest-descent + perturbation islands on ONE surface, counting UNIQUE evaluations.

    Deliberately simple and reproducible: the neighbourhood is the full 435-swap sweep, so a
    step is a true steepest descent, and the fixed 435-row block keeps every matmul at one
    operand shape (the same shape-stability property `SurfaceFits` pins with its tile).
    """

    def __init__(self, fits: MN.SurfaceFits, pool: str) -> None:
        self.fits = fits
        self.column = fits.pools.index(pool)
        self.pool = pool
        self.evaluated: set[bytes] = set()

    def _sweep(self, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All 435 single-swap neighbours and their fits, in one fixed-shape block."""
        neighbours = np.repeat(permutation[None, :], len(_PAIRS), axis=0)
        rows = np.arange(len(_PAIRS))
        left, right = _PAIRS[:, 0], _PAIRS[:, 1]
        neighbours[rows, left], neighbours[rows, right] = (
            permutation[right],
            permutation[left],
        )
        values = self.fits.fits_from_permutations(list(neighbours))[:, self.column]
        for row in neighbours:
            self.evaluated.add(row.tobytes())
        return neighbours, values

    def descend(self, permutation: np.ndarray) -> tuple[np.ndarray, float]:
        """Steepest descent to a 2-opt local optimum."""
        current = float(self.fits.fits_from_permutations([permutation])[0, self.column])
        self.evaluated.add(permutation.tobytes())
        while True:
            neighbours, values = self._sweep(permutation)
            best = int(values.argmin())
            if values[best] >= current:
                return permutation, current
            permutation, current = neighbours[best].copy(), float(values[best])

    def run(self, seed: int, islands: int, unique_budget: int) -> dict:
        """`islands` restarts, each descended then perturbation-restarted until the budget."""
        rng = np.random.default_rng(seed)
        best_fit, best_perm = float("inf"), None
        per_island = []
        for island in range(islands):
            start = np.concatenate([rng.permutation(30), [30]])
            local, fit = self.descend(start)
            island_best, island_perm = fit, local
            # Perturb-and-redescend until this island's share of the budget is spent.
            share = unique_budget // islands
            while (
                len(self.evaluated) < share * (island + 1) and len(self.evaluated) < unique_budget
            ):
                kicked = island_perm.copy()
                for _ in range(rng.integers(3, 7)):
                    a, b = rng.choice(30, size=2, replace=False)
                    kicked[[a, b]] = kicked[[b, a]]
                local, fit = self.descend(kicked)
                if fit < island_best:
                    island_best, island_perm = fit, local
            per_island.append(island_best)
            if island_best < best_fit:
                best_fit, best_perm = island_best, island_perm
            if island % 8 == 0 or island == islands - 1:
                log(
                    f"  {self.pool} seed={seed} island {island + 1}/{islands} "
                    f"best={best_fit:.4f} unique={len(self.evaluated):,}"
                )
            if len(self.evaluated) >= unique_budget:
                log(f"  {self.pool} seed={seed} budget reached at island {island + 1}")
                break
        assert best_perm is not None
        within = sum(1 for v in per_island if (v - best_fit) / abs(best_fit) < 1e-3)
        return {
            "seed": seed,
            "best_fit": best_fit,
            "best_layout": _layout_of(best_perm),
            "unique_evals_achieved": len(self.evaluated),
            "unique_evals_requested": unique_budget,
            "islands": len(per_island),
            "islands_within_0.1pct": within,
            "per_island_best": per_island,
        }


def _layout_of(permutation: np.ndarray) -> str:
    """Inverse of `S.layout_permutation`: the 30-char row-major layout string."""
    out = [""] * 30
    for char_index, char in enumerate(S.C30M):
        out[int(permutation[char_index])] = char
    return "".join(out)


def _gates(fits: MN.SurfaceFits) -> tuple[float, float]:
    """The two gates that must pass before any number is produced. Returns the control fit."""
    fits.assert_batch_invariant(PROBE)
    log("gate: evaluator bit-stable across batch lengths (padded tile)")
    control = fits.fit_of(AALTO_TARGET_LAYOUT)["AALTO"]
    control_rel = (control - AALTO_TARGET_FIT) / AALTO_TARGET_FIT
    log(
        f"control: MODELNORM AALTO champion rescores {control:.4f} vs {AALTO_TARGET_FIT:.4f} "
        f"rel={control_rel:+.3e}"
    )
    if abs(control_rel) > 1e-12:
        raise SystemExit(
            f"POSITIVE CONTROL FAILED: AALTO .native == .standardized is bit-exact, so this "
            f"must reproduce. rel={control_rel:.3e}. Frame, corpus or loader is wrong."
        )
    return control, control_rel


def run_one(pool: str, seed: int) -> int:
    """One (model, seed) search cell -> `runs/anchor-<POOL>-<seed>.json`.

    Split out so the 9 cells run in parallel as separate single-threaded processes. Each cell
    re-runs both gates itself: a cell whose evaluator or frame is wrong must not contribute a
    number just because a sibling process checked.
    """
    fits = MN.SurfaceFits()
    _gates(fits)
    result = SingleModelSearch(fits, pool).run(seed, ISLANDS, UNIQUE_EVALS_REQUESTED)
    target = HERE / "runs" / f"anchor-{pool}-{seed}.json"
    target.parent.mkdir(exist_ok=True)
    # Write via a temp file + rename so a reader can never observe a half-written cell.
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pool": pool, **result}, indent=1))
    tmp.rename(target)
    log(f"wrote {target}")
    return 0


def main() -> int:
    fits = MN.SurfaceFits()
    log(f"corpus {fits.trigram_path}")

    control, control_rel = _gates(fits)

    # -- the zero anchors ----------------------------------------------------------------
    pool = MN.random_pool(POOL_N, POOL_SEED)
    pool_fits = fits.fits_from_permutations(pool)
    zero = {p: float(pool_fits[:, n].mean()) for n, p in enumerate(fits.pools)}
    zero_sd = {p: float(pool_fits[:, n].std(ddof=1)) for n, p in enumerate(fits.pools)}
    zero_se = {p: zero_sd[p] / np.sqrt(POOL_N) for p in fits.pools}
    log(
        f"zero anchors (n={POOL_N}, seed={POOL_SEED}): "
        + "  ".join(f"{p}={zero[p]:.4f}" for p in fits.pools)
    )

    # stability check ONLY — reported, never shipped
    big = fits.fits_from_permutations(MN.random_pool(STABILITY_N, POOL_SEED))
    zero_big = {p: float(big[:, n].mean()) for n, p in enumerate(fits.pools)}
    stability = {
        p: {
            "zero_n1000": zero_big[p],
            "delta_ms": zero_big[p] - zero[p],
            "delta_in_SE": (zero_big[p] - zero[p]) / zero_se[p],
        }
        for p in fits.pools
    }
    for p in fits.pools:
        log(f"  stability {p}: n=1000 moves the zero by {stability[p]['delta_in_SE']:+.3f} SE")

    # -- the one anchors: harvested from the per-cell files run_one() wrote ---------------
    runs: dict[str, list[dict]] = {}
    for p in fits.pools:
        runs[p] = []
        for seed in SEEDS:
            cell = HERE / "runs" / f"anchor-{p}-{seed}.json"
            if not cell.exists():
                raise SystemExit(
                    f"missing search cell {cell}. Run all 9 cells first:\n"
                    f"  for p in AALTO COMMUNITY POOL; do for s in {' '.join(map(str, SEEDS))}; "
                    f"do python {Path(__file__).name} --cell $p $s & done; done; wait"
                )
            payload = json.loads(cell.read_text())
            if payload["pool"] != p or payload["seed"] != seed:
                raise SystemExit(
                    f"{cell} holds {payload['pool']}/{payload['seed']}, not {p}/{seed}"
                )
            if payload["unique_evals_requested"] != UNIQUE_EVALS_REQUESTED:
                raise SystemExit(
                    f"{cell} was run at budget {payload['unique_evals_requested']:,}, but this "
                    f"build expects {UNIQUE_EVALS_REQUESTED:,} — an unequal budget across cells "
                    f"would let one model's anchor be flattered by a longer search"
                )
            runs[p].append(payload)

    # CONSERVATIVE: the SLOWER (higher) of the per-seed bests cannot flatter a model whose
    # search happened to converge better on one seed.
    one = {p: max(r["best_fit"] for r in runs[p]) for p in fits.pools}
    best_seen = {p: min(r["best_fit"] for r in runs[p]) for p in fits.pools}
    layout_of_record = {
        p: next(r["best_layout"] for r in runs[p] if r["best_fit"] == one[p]) for p in fits.pools
    }
    for p in fits.pools:
        spread = one[p] - best_seen[p]
        span = zero[p] - one[p]
        log(
            f"one anchor {p}: {one[p]:.4f} (span {span:.4f} = {100 * span / zero[p]:.4f}% of zero); "
            f"seed spread {spread:.4f} ms = {100 * spread / span:.4f}% of span"
        )

    # -- gate 3: the prereg's AALTO acceptance bar ---------------------------------------
    aalto_gap_pct = 100 * (one["AALTO"] - AALTO_TARGET_FIT) / AALTO_TARGET_FIT
    met = aalto_gap_pct <= AALTO_BAR_PCT
    log(
        f"gate: AALTO anchor is {aalto_gap_pct:+.4f}% vs the 10M-eval target "
        f"(bar +{AALTO_BAR_PCT}%) -> {'MET' if met else 'MISSED — anchors are a LOWER BOUND'}"
    )

    provenance = {
        "frame": "standardized",
        "frame_note": S.FRAME_NOTE,
        "frame_caveat": MN.frame_caveat(),
        "interpretation": MN.interpretation_note(),
        "family": S.DEFAULT_FAMILY,
        "baked_wpm": S.BAKED_WPM,
        "corpus_trigram_path": str(fits.trigram_path),
        "corpus_trigram_sha256": sha256_of(Path(fits.trigram_path)),
        "surface_sha256": {p: sha256_of(S._resolve(f"{p}_{S.DEFAULT_FAMILY}")) for p in fits.pools},
        "zero_statistic": "mean",
        "zero_n": POOL_N,
        "zero_seed": POOL_SEED,
        "zero_sd": zero_sd,
        "zero_se": zero_se,
        "zero_stability_n1000": stability,
        "one_provenance": {
            "kind": "per-model memetic islands (full 435-swap steepest descent + perturbation)",
            "statistic": "the SLOWER of the per-seed bests (conservative: cannot flatter a "
            "model whose search converged better)",
            "unique_evals_requested": UNIQUE_EVALS_REQUESTED,
            "unique_evals_achieved": {
                p: [r["unique_evals_achieved"] for r in runs[p]] for p in fits.pools
            },
            "islands": ISLANDS,
            "seeds": list(SEEDS),
            "layout_of_record": layout_of_record,
            "best_fit_seen": best_seen,
            "seed_spread_ms": {p: one[p] - best_seen[p] for p in fits.pools},
            "aalto_control": {
                "target_fit": AALTO_TARGET_FIT,
                "target_layout": AALTO_TARGET_LAYOUT,
                "target_provenance": "MODELNORM-1's 10M-unique-eval AALTO champion; valid on "
                "this frame because AALTO .native == .standardized "
                "bit-exactly",
                "rescored_here": control,
                "rescore_rel": control_rel,
                "anchor_gap_pct": aalto_gap_pct,
                "bar_pct": AALTO_BAR_PCT,
                "bar_met": bool(met),
            },
            "is_a_lower_bound_not_the_optimum": "an optimizer output bounds the true optimum from one side only, so every "
            "normalized score is an UPPER bound on the true normalized score",
        },
        "probe_layout": PROBE,
        "probe_fits": fits.fit_of(PROBE),
        "evaluator": {
            "tile": MN.TILE,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }

    anchors = MN.Anchors(zero=zero, one=one, provenance=provenance)
    anchors.assert_direction()
    anchors.assert_matches_surfaces(fits, PROBE)
    log("gate: assert_direction PASS (each optimum -> 1.0, pool mean -> 0.0)")
    anchors.write(OUT)
    log(f"wrote {OUT}")

    EVIDENCE.write_text(
        json.dumps(
            {"runs": runs, "pool_fits_shape": list(pool_fits.shape), "zero": zero, "one": one},
            indent=1,
        )
    )
    log(f"wrote {EVIDENCE}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--cell":
        sys.exit(run_one(sys.argv[2], int(sys.argv[3])))
    sys.exit(main())
