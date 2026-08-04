"""EXPLOIT-1 §7 addendum — the hybrid's INTERPRETABILITY cost, on INTERPFRAME-1's OWN instrument.

hybrid.py found that only hybrid-C (interp.1 + ALL 20 served columns) drives the searchable null
space to exactly 0.0000 ms. But re-importing the served columns re-imports the coupled-column
credit-splitting that interp.1 was BUILT to remove -- so "keeps the floor near zero AND retains the
interpretation columns" needs measuring, not asserting.

Rather than invent a metric, this reuses INTERPFRAME-1's OWN registered instrument
(`agent-artifacts/interpframe/metrics.py`, `m1_maxcorr` / `m2_constfrac`) on the corpus-weighted
serve grid, so the numbers are directly comparable to the seven bars it registered:
MAXCORR served 0.9813 -> interp 0.7037 against a registered bar of 0.7850.

This costs no training and answers the §7 gate honestly in BOTH directions: resolution AND
interpretability. Any hybrid must clear the same bar interp.1 cleared to inherit its claim.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

# INTERPFRAME-1's OWN registered instrument, loaded BY PATH from MY worktree's copy rather than by
# adding its directory to sys.path. Two reasons, both bit me / would have:
#   1. `agent-artifacts/interpframe/_boot.py` SHADOWS my own `_boot` (same module name, and its
#      WORKTREE constant points at keybo-wt-interpframe), so a plain `import metrics` made
#      assert_tree() demand the sibling's checkout and abort. Same shape as the /tmp platform.py
#      shadow-import hazard.
#   2. It must be MY tree's copy of the metric, so the instrument and the frames it measures come
#      from one tree.
import importlib.util as _ilu  # noqa: E402

_mp = "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/interpframe/metrics.py"
_spec = _ilu.spec_from_file_location("interpframe_metrics", _mp)
M = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(M)
print(f"[instr] loaded INTERPFRAME-1's metric from {_mp}")
from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    BIGRAM_INTERP_FEATURE_NAMES,
    bigram_features_from_positions,
    interp_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

WPM = 90.0
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP = len(POS)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


for fn in ("m1_maxcorr", "weighted_corr_matrix"):
    if not hasattr(M, fn):
        raise SystemExit(f"ABORT: interpframe metrics has no {fn!r} on this tree")

# THE WEIGHTING GRID MUST BE INTERPFRAME-1's OWN, or the numbers are not comparable to its bars.
# My first run weighted on the C30M/qwerty board and read served MAXCORR 0.9556 vs the published
# 0.9813 -- not an instrument bug but a DIFFERENT GRID. INTERPFRAME-1 weighted on `flagship-c3`
# (poc.py:223-225). On that grid this code reproduces served 0.9813 / interp 0.7037 and MEANCORR
# 0.1137 / 0.1572 -- all four published values to four decimals. Verified before use.
surface = default_surface(WPM, None)
from keybo.analysis.shap_diff import _char_weight_tables  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402

_, LAY_W = _resolve("flagship-c3")
_slot = surface._slot_of(LAY_W)
_w3, _w2, _covered = _char_weight_tables(surface, LAY_W)
_perm = np.array([_slot[c] for c in LAY_W] + [_slot[" "]], dtype=np.intp)
_wp = np.zeros((NP, NP))
np.add.at(_wp, (_perm[:, None], _perm[None, :]), _w2)
w = _wp.ravel()
print(f"[grid] weighting on INTERPFRAME-1's own board flagship-c3 = {LAY_W}")

SERVED = np.vstack([bigram_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
INTERP = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
sn, inn = list(BIGRAM_FEATURE_NAMES), list(BIGRAM_INTERP_FEATURE_NAMES)
ROWS = [n for n in sn if n in ("bottom", "home", "top")]
FING = [n for n in sn if n in ("index", "middle", "ring", "pinky", "lateral")]


def take(names):
    return SERVED[:, [sn.index(n) for n in names]]


FRAMES = {
    "served (20c)": (SERVED, sn),
    "interp.1 (10c)": (INTERP, inn),
    "hybrid-A (13c)": (np.hstack([INTERP, take(ROWS)]), inn + ROWS),
    "hybrid-B (18c)": (np.hstack([INTERP, take(ROWS + FING)]), inn + ROWS + FING),
    "hybrid-C (30c)": (np.hstack([INTERP, SERVED]), inn + [f"served:{n}" for n in sn]),
}

# INTERPFRAME-1's registered bar for MAXCORR, quoted from the ledger entry (not re-derived):
BAR_MAXCORR = 0.7850
PUB = {"served": 0.9813, "interp": 0.7037}

hyb = json.load(open(f"{ARTIFACTS}/hybrid.json"))
out = {"registered_bar_MAXCORR": BAR_MAXCORR, "published_INTERPFRAME1": PUB, "frames": {}}
log(f"{'frame':<18} {'cols':>5} {'MAXCORR':>9} {'>0.9':>5} {'>0.7':>5} {'MEANCORR':>9} "
    f"{'worst pair':<34} {'nullspace ms':>12}")
for label, (X, names) in FRAMES.items():
    r = M.m1_maxcorr(X, w, names)
    key = {"served (20c)": "served (20c)", "interp.1 (10c)": "interp.1 (10c)",
           "hybrid-A (13c)": "hybrid-A: interp + row one-hots",
           "hybrid-B (18c)": "hybrid-B: interp + row + finger one-hots",
           "hybrid-C (30c)": "hybrid-C: interp + ALL served cols"}[label]
    ns = hyb["frames"][key]["searchable_nullspace_ms"]
    fl = hyb["frames"][key]["floor_wmae_ms"]
    out["frames"][label] = {**r, "n_columns": int(X.shape[1]),
                            "searchable_nullspace_ms": ns, "floor_wmae_ms": fl,
                            "clears_registered_MAXCORR_bar": bool(r["maxcorr"] <= BAR_MAXCORR)}
    log(f"{label:<18} {X.shape[1]:>5d} {r['maxcorr']:>9.4f} {r['n_pairs_over_0.9']:>5d} "
        f"{r['n_pairs_over_0.7']:>5d} {r['meancorr']:>9.4f} "
        f"{str(r['worst_pair']):<34} {ns:>12.4f}")

log("")
log("=" * 100)
log("THE TRADE, MEASURED ON INTERPFRAME-1's OWN BAR (MAXCORR <= 0.7850) -- no training spent")
log("=" * 100)
i = out["frames"]["interp.1 (10c)"]
log(f"  reproduction check: served MAXCORR {out['frames']['served (20c)']['maxcorr']:.4f} "
    f"(published {PUB['served']:.4f}); interp {i['maxcorr']:.4f} (published {PUB['interp']:.4f})")
for label in ("hybrid-A (13c)", "hybrid-B (18c)", "hybrid-C (30c)"):
    h = out["frames"][label]
    log(f"  {label}: MAXCORR {h['maxcorr']:.4f} "
        f"{'CLEARS' if h['clears_registered_MAXCORR_bar'] else 'FAILS'} the 0.7850 bar "
        f"(worst pair {h['worst_pair']}); null space {h['searchable_nullspace_ms']:.4f} ms "
        f"({100 * h['searchable_nullspace_ms'] / i['searchable_nullspace_ms']:.1f}% of interp.1's)")
    out["frames"][label]["verdict"] = (
        "both" if h["clears_registered_MAXCORR_bar"] and h["searchable_nullspace_ms"] < 0.05 * i["searchable_nullspace_ms"]
        else "resolution only" if h["searchable_nullspace_ms"] < 0.05 * i["searchable_nullspace_ms"]
        else "interpretability only" if h["clears_registered_MAXCORR_bar"]
        else "neither")
    log(f"      => achieves: {out['frames'][label]['verdict'].upper()}")

with open(f"{ARTIFACTS}/hybrid_cost.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/hybrid_cost.json")
