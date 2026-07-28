"""SP4 — WHICH cell of the 2x2 actually calibrates the weight? Decided by measurement,
not by argument. And the SCORING EXPERIMENT: does re-weighting change which layout wins?

SP3 produced four defensible ratios (2.34x to 8.30x) depending on marginal/conditional and
linear/tangent. Choosing among them by argument is how a prior gets dressed as a
measurement. But the oxey score has an OBSERVABLE PURPOSE -- it is a scalar meant to track
layout quality -- so the choice is testable: build the scorer under each candidate weight
and measure agreement with the fitted ms/char it is trying to approximate.

Prediction (from SP2's common-factor mechanism): a MARGINAL slope charges scissor for the
shared layout-quality factor that all eleven terms also charge for, so in a SUM the shared
part is counted eleven times. The marginal-calibrated weight should therefore OVERSHOOT and
AGREE WORSE than the conditional one. If that prediction holds, the conditional cell is the
right calibration target and the 7.0x/8.0x marginal headline must not be used as a weight.

Also answers the brief's "does re-weighting change which layout a search picks?" as a
SCORING experiment over EXISTING champions (no new search, per the brief):
  * the six SPEEDTIE-1 speed-tied champions (state/.../speedtie-1/speedtie-summary.json)
  * every C30M-exact registry layout
under shipped w=4.0 vs each candidate re-weight. Reports rank changes, argmin changes, and
the spearman-vs-fitted-ms/char of the whole scorer.

FRAME: g-frame, 90 WPM baked, blend-v1, tau saturated. MODELLED only. No weight is edited;
DEFAULT_OXEY_WEIGHTS is untouched -- candidate weights are passed through the shipped
`OxeyStyleScorer(weights=...)` override, which is a supported public API.
"""

import contextlib
import importlib.util
import io
import json
import random

import numpy as np

spec = importlib.util.spec_from_file_location("c3", "/tmp/scissorprice/probe/collin3.py")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    c3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c3)
assert any("max abs diff = 0" in ln for ln in buf.getvalue().splitlines() if "POSITIVE" in ln)
print("[inherited] share-path positive control: max abs diff = 0")
shares_vec, TERMS = c3.shares_vec, c3.TERMS

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.surfaces import C30M  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS, OxeyStyleScorer  # noqa: E402

NAT = (
    "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
    "old-new-layout-comparison/tri_frequency_old_new_surfaces"
)
ART = "/local/home/zegertho/agent/state/keybo-optimization/artifacts"
OUT = "/local/home/zegertho/agent/state/scissorprice/artifacts"
obj = SF.trigram_objective(SF.default_trigram_path(None))
MASS = obj[3].sum()
REG = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
USABLE = {n: s for n, s in REG.items() if set(s) == set(C30M)}
SRCS = ("AALTO", "COMMUNITY", "POOL")
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}
W0 = {k: v[0] for k, v in DEFAULT_OXEY_WEIGHTS.items()}

# --- candidate weights, from SP3's four cells (mean over the two INDEPENDENT sources) ----
sp3 = json.load(open(f"{OUT}/sp3_ratio_2x2.json"))


def implied_mean(cell, indep_only=True):
    srcs = ("AALTO", "COMMUNITY") if indep_only else SRCS
    return float(np.mean([sp3["per_source"][s][cell]["implied_weight"] for s in srcs]))


CANDIDATES = {
    "shipped": 4.0,
    "marginal_linear": implied_mean("marginal_linear"),
    "marginal_tangent": implied_mean("marginal_tangent"),
    "conditional_linear": implied_mean("conditional_linear"),
    "conditional_tangent": implied_mean("conditional_tangent"),
}
print("\ncandidate scissor weights (sfb-anchored implied, mean of the 2 INDEPENDENT sources):")
for k, v in CANDIDATES.items():
    print(f"  {k:22s} {v:+8.3f}")

# ---------------------------------------------------------------- pools
rng = random.Random(31337)


def neigh(s, k):
    lst = list(s)
    for _ in range(k):
        i, j = rng.randrange(30), rng.randrange(30)
        lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)


pool = []
for _n, s in USABLE.items():
    pool.append(s)
    for _ in range(80):
        pool.append(neigh(s, rng.choice([1, 1, 2, 2, 3, 3, 4, 5])))
n = len(pool)
X = np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def score_with(w_sci, Xm):
    w = dict(W0)
    w["scissor"] = w_sci
    return Xm @ np.array([w[t] for t in TERMS])


# =========== PART 1: which weight makes the SCORER agree best with fitted ms/char? =======
print(f"\n{'='*84}\nPART 1 — CALIBRATION TEST: spearman(oxey score, fitted ms/char), near-optimal pool n={n}")
print("  (the scorer's stated job is to track layout quality; the weight that does that best")
print("   is the calibrated one. Higher = better.)")
part1 = {}
print(f"\n  {'candidate':22s}{'w':>8s}" + "".join(f"{s:>12s}" for s in SRCS) + f"{'mean':>10s}")
Y = {src: np.array([SF.score_fit(lay, SURF[src], obj) for lay in pool]) / MASS for src in SRCS}
for name, wv in CANDIDATES.items():
    sc = score_with(wv, X)
    rows = [spearman(sc, Y[src]) for src in SRCS]
    part1[name] = {"w": wv, "spearman": dict(zip(SRCS, rows)), "mean": float(np.mean(rows))}
    print(f"  {name:22s}{wv:+8.2f}" + "".join(f"{r:+12.5f}" for r in rows) + f"{np.mean(rows):+10.5f}")

# fine sweep to locate the ARGMAX weight — the purely empirical answer
print("\n  fine sweep over w_scissor (argmax = the weight the DATA picks, no ratio algebra):")
# ⚠ TRAP 51. A first pass on grid [0,40] pinned the argmax AT 40.0 on 2 of 3 sources -- the
# signature of a criterion that pays without limit, where the reported "optimum" is really
# the grid ceiling. So sweep WIDE and report explicitly whether the optimum is INTERIOR.
# An argmax at the boundary is NOT a calibrated weight; it means spearman is still rising and
# this criterion cannot identify a level (only a direction).
grid = np.arange(0.0, 400.01, 0.5)
best = {}
for src in SRCS:
    rs = np.array([spearman(score_with(w, X), Y[src]) for w in grid])
    k = int(np.argmax(rs))
    interior = 0 < k < len(grid) - 1
    best[src] = {"argmax_w": float(grid[k]), "spearman": float(rs[k]),
                 "interior": bool(interior), "grid_max": float(grid[-1]),
                 "spearman_at_shipped": float(spearman(score_with(4.0, X), Y[src]))}
    print(f"    {src:10s} argmax w = {grid[k]:6.1f}   rho {rs[k]:+.5f}"
          f"   (at shipped 4.0: {best[src]['spearman_at_shipped']:+.5f}, "
          f"gain {rs[k]-best[src]['spearman_at_shipped']:+.5f})"
          f"   {'INTERIOR' if interior else '** AT GRID BOUNDARY **'}")
    # how flat is it? the range of w within 0.001 spearman of the peak is the real precision
    near = grid[rs >= rs[k] - 0.001]
    best[src]["w_within_0.001_rho"] = [float(near.min()), float(near.max())]
    print(f"               w within 0.001 rho of the peak: [{near.min():.1f}, {near.max():.1f}]"
          f"  <- the criterion's actual resolution on the LEVEL")
# bootstrap the argmax to get a CI on the DATA-CHOSEN weight (coarser grid for speed)
NB = 400
bgrid = np.arange(0.0, 400.01, 2.0)
brng = np.random.default_rng(20260728)
argmax_boot = {src: [] for src in SRCS}
for _b in range(NB):
    ix = brng.choice(np.arange(n), size=n, replace=True)
    Xb = X[ix]
    for src in SRCS:
        yb = Y[src][ix]
        rs = [spearman(score_with(w, Xb), yb) for w in bgrid]
        argmax_boot[src].append(float(bgrid[int(np.argmax(rs))]))
print(f"\n  bootstrap CI95 on the ARGMAX weight ({NB} resamples):")
for src in SRCS:
    a = np.array(argmax_boot[src])
    print(f"    {src:10s} median {np.median(a):5.1f}  CI95 [{np.percentile(a,2.5):.1f},"
          f" {np.percentile(a,97.5):.1f}]")
    best[src]["argmax_ci95"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    best[src]["argmax_median_boot"] = float(np.median(a))

# =========== PART 2: does re-weighting change which layout is picked? ====================
print(f"\n{'='*84}\nPART 2 — SCORING EXPERIMENT over EXISTING champions (no new search)")
st = json.load(open(f"{ART}/speedtie-1/speedtie-summary.json"))

# The six champions are the KEYS of st["layouts"], not values -- an earlier walker that only
# tested values silently harvested ZERO of them and still printed a result (trap 1/19: an
# absent input is not an empty one). Assert the count instead of trusting the traversal.
tie = {k: k for k in st["layouts"] if len(k) == 30 and set(k) == set(C30M)}
assert len(tie) == 6, f"expected 6 speed-tied champions, harvested {len(tie)}"
cands = {f"speedtie:{k}": v for k, v in tie.items()}
cands.update({f"registry:{k}": v for k, v in USABLE.items()})
print(f"  candidates: {len(tie)} speed-tied champions + {len(USABLE)} registry = {len(cands)}")

names = list(cands)
Xc = np.array([[shares_vec(cands[nm])[t] for t in TERMS] for nm in names])
ms = {src: np.array([SF.score_fit(cands[nm], SURF[src], obj) for nm in names]) / MASS
      for src in SRCS}

part2 = {"candidates": {nm: cands[nm] for nm in names}, "per_candidate_weight": {}}
base_rank = None
for name, wv in CANDIDATES.items():
    sc = score_with(wv, Xc)
    order = list(np.argsort(sc))
    rank = {names[i]: r for r, i in enumerate(order)}
    if name == "shipped":
        base_rank = rank
    moved = sum(1 for nm in names if rank[nm] != base_rank[nm])
    part2["per_candidate_weight"][name] = {
        "w": wv,
        "argmin": names[order[0]],
        "top5": [names[i] for i in order[:5]],
        "n_rank_changed_vs_shipped": moved,
        "max_rank_move": max(abs(rank[nm] - base_rank[nm]) for nm in names),
        "spearman_vs_shipped_ranking": spearman(
            [base_rank[nm] for nm in names], [rank[nm] for nm in names]
        ),
        "scores": {nm: float(sc[i]) for i, nm in enumerate(names)},
    }
    d = part2["per_candidate_weight"][name]
    print(f"\n  {name:22s} w={wv:+7.2f}  argmin={d['argmin']}")
    print(f"    top5: {', '.join(d['top5'])}")
    print(f"    vs shipped: {moved}/{len(names)} ranks changed, max move {d['max_rank_move']}, "
          f"spearman {d['spearman_vs_shipped_ranking']:.5f}")

print("\n  scissor share of each candidate (the driver of any re-rank):")
isc = TERMS.index("scissor")
for i in np.argsort(Xc[:, isc]):
    print(f"    {names[i]:34s} scissor {Xc[i,isc]:7.4f}%   msAALTO {ms['AALTO'][i]:9.4f}")

json.dump({"part1_calibration": part1, "part1_argmax": best, "part2_scoring": part2,
           "candidates_weights": CANDIDATES},
          open(f"{OUT}/sp4_calibration_and_scoring.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp4_calibration_and_scoring.json")
