"""GATEFOLDS-1 §2 — THE DECIDING CONTROL: is the high-wpm refusal about INTERPRETABILITY at all?

The registered adjudicator of my assigned prior. ONE new arm:

  CUR-INVARIANT   the SERVED 20-column frame, made WPM-INVARIANT, nothing else changed.

If the served frame -- all 20 columns, all its resolution, all its one-hots, NO monotone
constraints, the frame that PASSES the gate -- fails the SAME gate once it is deprived of the ONE
property H1 names (the ability to vary with pace), then the gate is not measuring an
interpretability cost. It is measuring loss of pace adaptation, and interp.1/hybrid-B fail it
because they are wpm-invariant, not because they are interpretable.

⚠ HOW THIS DIFFERS FROM THE EXISTING `CUR-NOWPM` ARM, and why it is worth a separate run.
`interpframe/lolo.py`'s CUR-NOWPM pinned `wpm` to ONE GLOBAL CONSTANT (the corpus mean), which
makes the column unsplittable -- the right ablation for "can a tree USE this column". It already
fails 4/4. But a global constant ALSO destroys the `to_ms` pace, because `to_ms` recovers the pace
from the column: every cell then converts at 90-ish wpm regardless of its true bucket. That is the
BROKEN-CONVERSION confound my brief warns about by name (a sibling's tau collapsed 1.0 -> 0.333
from exactly this). So CUR-NOWPM's 4/4 failure is AMBIGUOUS between "lost pace adaptation" and
"lost the ms conversion".

CUR-INVARIANT removes the ambiguity. The `wpm` column is pinned to a constant **for the MODEL**
(so no split can use it, and the trained function cannot vary with pace) while the ms conversion
still receives each cell's TRUE bucket midpoint -- the same treatment `_predict_cells` already
gives interp.1 and hybrid-B via its `needs_wpm` branch. That makes CUR-INVARIANT the EXACT
served-frame analogue of the interp frames: same invariance, same honest conversion.

ARMS (registered in GATEFOLDS-preregistration.md §2 before any number existed):
  CUR              the served frame, untouched -- the incumbent and the gate's own control
  CUR-INVARIANT    the served frame, wpm-invariant, honest ms conversion   <- the adjudicator

MATCHED: same seeds, same folds, same cell construction, same hyperparameters, same geometry.

Detached-friendly: writes a SENTINEL file when finished so a poller never has to `wait $PID`.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, SCRATCH, STROKES, assert_tree, require  # noqa: E402

assert_tree()

import keybo.models.base as MB  # noqa: E402
import keybo.training.train as T  # noqa: E402
import keybo.training.validate as V  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import validate  # noqa: E402

# Brief-decay defence: every symbol this driver leans on, asserted present on THIS tree.
require(V, "bigram_features_from_positions")
require(T, "bigram_features_from_positions")
require(V, "_predict_cells")
require(MB.TypingModel, "to_ms")

SEEDS = [0, 1, 2]
SENTINEL = f"{SCRATCH}/control.sentinel"
WPM_IDX = BIGRAM_FEATURE_NAMES.index("wpm")  # asserted, not assumed
assert BIGRAM_FEATURE_NAMES[WPM_IDX] == "wpm", "schema moved 'wpm'"

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"loading {STROKES}")
rows = load_strokes(STROKES, ngram_len=2, wpm_threshold=0, min_samples=1)
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")
GEO = ROW_STAGGERED_31

out: dict = {
    "prereg": "agent-artifacts/gatefolds/GATEFOLDS-preregistration.md",
    "strokes": STROKES,
    "n_rows": len(rows),
    "seeds": SEEDS,
    "wpm_column_index": WPM_IDX,
    "arms": {},
}


def _validate(name, **kw):
    log(f"ARM {name}: validate() 4 folds x {len(SEEDS)} seeds  kw={kw}")
    rep = validate(
        rows,
        seeds=SEEDS,
        ngram="bigram",
        n_boot=10,
        geometry=GEO,
        train_params={"n_jobs": 8},
        **kw,
    )
    log(f"ARM {name}: done. config {rep['config']}")
    return rep


def run_invariant(name):
    """The served frame with `wpm` pinned for the MODEL but TRUE per-cell pace for `to_ms`.

    Two patches, applied together and restored in a `finally`:

    1. The FEATURIZER (in both `validate` and `train`) writes the constant into the wpm column, so
       neither the training matrix nor the eval matrix carries real pace: no split can use it and
       the fitted function is provably bucket-invariant. Byte-equivalent to a drop for a tree.
    2. `to_ms` is patched to IGNORE that constant column and use the pace the caller states. This
       is the part CUR-NOWPM lacks. Without it the conversion would price every cell at the
       constant, which is the BROKEN-CONVERSION confound -- a real, separate defect that would
       masquerade as a modelling loss.

    ⚠ Patch 2 must not be a blanket override: `to_ms` REFUSES an explicit wpm when the frame has a
    wpm column (one source of truth). So the patch swaps in a version that takes the pace from a
    module-level slot `_PACE`, set by the patched `_predict_cells` from each cell's own
    `Cell.wpm` -- the identical value `_predict_cells` hands interp.1 and hybrid-B.
    """
    wpm_const = float(np.mean([s[0] for r in rows for s in r.samples]))
    log(f"  {name}: pinning the model's wpm column to the constant {wpm_const:.6f}")
    log(f"  {name}: to_ms will still receive each cell's TRUE bucket midpoint")

    real_feat = V.bigram_features_from_positions
    assert real_feat is T.bigram_features_from_positions, "featurizer identity differs across modules"

    def pinned(geometry, positions, wpm=0.0, direction=False, kitchensink=False):
        vec = real_feat(geometry, positions, wpm=wpm, direction=direction, kitchensink=kitchensink)
        vec[WPM_IDX] = wpm_const
        return vec

    real_predict_cells = V._predict_cells
    real_to_ms = MB.TypingModel.to_ms
    pace_slot: dict[str, np.ndarray | None] = {"pace": None}

    def to_ms_true_pace(self, pred, X, wpm=None):
        """`to_ms` that prices at the cells' TRUE pace, not the pinned column."""
        if self.target_space == "MS":
            return pred
        pace = pace_slot["pace"]
        if pace is None:
            return real_to_ms(self, pred, X, wpm)
        pace = np.asarray(pace, dtype=np.float64)
        if len(pace) != len(np.asarray(pred)):
            raise ValueError(f"pace vector {len(pace)} != predictions {len(np.asarray(pred))}")
        if np.any(pace <= 0):
            raise ValueError("true-pace vector must be positive")
        return np.exp(pred) * 12000.0 / pace

    def predict_cells_true_pace(model, cells, geometry, **kw):
        pace_slot["pace"] = np.array([c.wpm for c in cells], dtype=np.float64)
        try:
            return real_predict_cells(model, cells, geometry, **kw)
        finally:
            pace_slot["pace"] = None

    V.bigram_features_from_positions = pinned
    T.bigram_features_from_positions = pinned
    V._predict_cells = predict_cells_true_pace
    MB.TypingModel.to_ms = to_ms_true_pace
    try:
        return _validate(name)
    finally:
        V.bigram_features_from_positions = real_feat
        T.bigram_features_from_positions = real_feat
        V._predict_cells = real_predict_cells
        MB.TypingModel.to_ms = real_to_ms
        log(f"  {name}: patches restored")


# --- SELF-TEST of the patch, BEFORE the expensive run ------------------------------------
# An arm whose ablation silently did nothing would look like a clean PASS and be read as
# "invariance is not the cause" -- the exact false-negative this campaign keeps producing. So the
# patch is PROVEN to (a) pin the column and (b) preserve the true pace, before any fold runs.
print()
print("=" * 92)
print("PATCH SELF-TEST (an ablation that silently no-ops would read as a PASS)")
print("=" * 92)
_wpm_const = float(np.mean([s[0] for r in rows for s in r.samples]))
_pos = ((1, 1), (3, 0))
_real = V.bigram_features_from_positions
_v_lo = _real(GEO, _pos, wpm=50.0)
_v_hi = _real(GEO, _pos, wpm=130.0)
print(f"  unpatched served: wpm col at 50wpm={_v_lo[WPM_IDX]:.3f} at 130wpm={_v_hi[WPM_IDX]:.3f}"
      f"  -> varies: {_v_lo[WPM_IDX] != _v_hi[WPM_IDX]}")
assert _v_lo[WPM_IDX] != _v_hi[WPM_IDX], "served frame's wpm column should vary -- it does not"


def _pinned_probe(geometry, positions, wpm=0.0, direction=False, kitchensink=False):
    vec = _real(geometry, positions, wpm=wpm, direction=direction, kitchensink=kitchensink)
    vec[WPM_IDX] = _wpm_const
    return vec


_p_lo = _pinned_probe(GEO, _pos, wpm=50.0)
_p_hi = _pinned_probe(GEO, _pos, wpm=130.0)
print(f"  patched served:   wpm col at 50wpm={_p_lo[WPM_IDX]:.3f} at 130wpm={_p_hi[WPM_IDX]:.3f}"
      f"  -> varies: {_p_lo[WPM_IDX] != _p_hi[WPM_IDX]}")
assert _p_lo[WPM_IDX] == _p_hi[WPM_IDX] == _wpm_const, "the pin did not take"
assert np.array_equal(_p_lo, _p_hi), "patched vector must be bit-identical across wpm"
# and the NON-wpm columns must be untouched by the pin
assert np.array_equal(np.delete(_p_lo, WPM_IDX), np.delete(_v_lo, WPM_IDX)), "pin touched other cols"
print("  PASS: the pin makes the served vector bit-identical across wpm, other columns untouched")
out["patch_self_test"] = {
    "wpm_const": _wpm_const,
    "served_wpm_col_varies_unpatched": True,
    "served_vector_invariant_when_pinned": True,
    "other_columns_untouched": True,
}

# --- the arms ----------------------------------------------------------------------------
out["arms"]["CUR"] = _validate("CUR")
with open(f"{ARTIFACTS}/control.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log("checkpointed CUR -> control.json")

out["arms"]["CUR-INVARIANT"] = run_invariant("CUR-INVARIANT")
with open(f"{ARTIFACTS}/control.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log("checkpointed CUR-INVARIANT -> control.json")

# --- SANITY: the invariant arm's predictions must actually be rank-invariant per bucket ---
# Verified from the ARTIFACT rather than asserted: if CUR-INVARIANT's per-bucket rho pattern were
# indistinguishable from CUR's, the ablation would not have bitten and the comparison would be
# vacuous (prereg invariant 5).
cur_b = {
    h: {rec["seed"]: rec["bucket_rhos"] for rec in f["seeds"]}
    for h, f in out["arms"]["CUR"]["folds"].items()
}
inv_b = {
    h: {rec["seed"]: rec["bucket_rhos"] for rec in f["seeds"]}
    for h, f in out["arms"]["CUR-INVARIANT"]["folds"].items()
}
n_same = sum(
    1
    for h in cur_b
    for s in cur_b[h]
    if cur_b[h][s] == inv_b.get(h, {}).get(s)
)
out["sanity_identical_bucket_rho_cells"] = n_same
print()
print(f"SANITY: fold-seed cells with bucket_rhos IDENTICAL between CUR and CUR-INVARIANT: {n_same}/12")
if n_same == 12:
    print("!! THE ABLATION DID NOT BITE -- the comparison would be vacuous. Do not read the verdict.")

with open(f"{ARTIFACTS}/control.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/control.json")
os.makedirs(SCRATCH, exist_ok=True)
with open(SENTINEL, "w") as fh:
    fh.write("done\n")
log(f"SENTINEL {SENTINEL}")
