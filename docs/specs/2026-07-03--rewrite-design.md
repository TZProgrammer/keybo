# keybo Rewrite — Design Spec

**Date:** 2026-07-03
**Status:** Draft for review
**Author:** (rewrite)

## Goal

Rewrite `keybo` (a data-driven keyboard-layout optimizer) from a pile of top-level
scripts into a readable, extensible, maintainable Python package with a large unit-test
suite. Fix known bugs along the way. Keep the modeling philosophy (learned typing-time
models from real keystroke data, XGBoost for now) but put every moving part behind a
clean interface so implementations are plug-and-play.

Non-goals: resurrecting `FreyaScorer` or the paper's hand-tuned cost function (legacy,
pre-fork); generalizing board geometry beyond the 30-key row-staggered board the data
supports; changing the XGBoost modeling approach itself.

## Locked design decisions (from brainstorming)

1. **Scope C** — full rewrite, model architecture in scope.
2. **Two-layer scoring seam.** `IScorer` (keyboard → fitness) sits on top of a lower
   `TypingModel` seam (features → predicted time). XGBoost is one implementation of the
   lower seam; swapping it is a one-file change.
3. **All four workflows are first-class:** D (process raw data), C (train model),
   A (optimize), B (score/compare). The pipeline D→C→A/B runs constantly.
4. **One shared feature pipeline**, imported by processing, training, AND scoring — the
   single guard against train/serve skew (the current repo's worst latent bug).
5. **Fixed geometry** = 30-key, 3×10, row-staggered — but encoded as ONE explicit
   `Geometry` object that features query, not scattered magic numbers.
6. **Model artifacts** = interface-owned `save`/`load` (plug-and-play), with a MANDATORY
   metadata sidecar (feature-pipeline version, feature names/order, target WPM range,
   n-gram order, training date). XGBoost implements this via native `Booster` JSON +
   `.meta.json` sidecar. No pickle. Loading refuses a model whose feature-version does
   not match the current pipeline — loudly, instead of silently producing skewed scores.

## Package structure

```
keybo/
  __init__.py
  geometry.py            # Geometry: key positions, row offsets, finger map, distance
  layout.py              # Layout (was Keyboard): char<->position, swap/undo, render
  features/
    __init__.py
    ngram.py             # NgramFeatures: (ngram, positions) -> feature vector. THE pipeline.
    schema.py            # feature names + FEATURE_VERSION (bumped when features change)
  models/
    __init__.py
    base.py              # TypingModel ABC: predict(), save(), load(), .metadata
    xgboost_model.py     # XGBoostTypingModel implements base via JSON + sidecar
  scoring/
    __init__.py
    base.py              # IScorer ABC
    model_scorer.py      # ModelScorer: sums model-predicted times * ngram freq
  optimize/
    __init__.py
    base.py              # IOptimizer ABC
    annealing.py         # SimulatedAnnealing (init-temp, cooling, stopping criterion)
    local_search.py      # two_opt, three_opt (correct multi-swap undo)
  data/
    __init__.py
    corpus.py            # load ngram frequency files (trigrams.txt etc.)
    keystrokes.py        # process raw keystroke dump -> bistrokes/tristrokes tsv (fixes D)
    strokes.py           # load bistroke/tristroke tsv for training
  training/
    __init__.py
    train.py             # fit a TypingModel from stroke data + eval (CV, leave-one-layout-out)
    tune.py              # hyperparameter search
  cli/
    __init__.py
    __main__.py          # `python -m keybo` dispatch
    optimize.py  score.py  train.py  tune.py  process_data.py
tests/                   # pytest, mirrors package layout
data/corpus/             # trigrams.txt, bigrams.txt, 1-skip.txt (moved here)
models/                  # saved model artifacts (.json + .meta.json)
docs/
  specs/  plans/
legacy/                  # archived: paper scripts, notebook, FreyaScorer, Graph_Utils
pyproject.toml           # deps pinned, console_scripts, pytest/ruff config
README.md
```

## The seams (interfaces)

### Geometry (`geometry.py`)
Owns the physical board. One shipped instance: `ROW_STAGGERED_30`.
- `positions: dict[str, (x, y)]` — signed x (sign=hand, |x|=finger 1..5), y (row 1..3), space=(0,0)
- `row_offsets: {1: 0.5, 2: 0.0, 3: -0.25}`
- `finger(x) -> Finger`, `hand(x) -> -1|0|1`, `stagger_adjusted_dx(a, b)`, `distance(a, b)`
- Features query this; no offset literals anywhere else.

### Layout (`layout.py`) — was `Keyboard`
- Immutable-ish mapping char↔position over a Geometry.
- `swap(k1,k2)`, `undo()` — **undo uses an explicit swap STACK**, so N chained swaps
  undo correctly (fixes the 3-opt corruption bug). `random_swap()`, `pos(c)`, `render()`.

### NgramFeatures (`features/ngram.py`) — THE pipeline
- `bigram_features(layout, bigram) -> np.ndarray`
- `trigram_features(layout, trigram) -> np.ndarray`
- Deterministic, no I/O, no globals. Same code path for training and scoring.
- `schema.py` exports `FEATURE_NAMES` and `FEATURE_VERSION`; any change to features bumps
  the version, which invalidates stale saved models on load.
- **Bug fixes folded in here:** correct SFB definition (same finger = same |x| in same
  hand, incl. index cols 1↔2); scores ALL keys in the layout (no hardcoded 28-char set).

### TypingModel (`models/base.py`)
```
class TypingModel(ABC):
    metadata: ModelMetadata           # feature_version, feature_names, wpm_range, ngram, trained_at
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...          # writes artifact + <path>.meta.json
    @classmethod
    def load(cls, path: str) -> "TypingModel": ...  # validates metadata.feature_version == schema.FEATURE_VERSION
```
`XGBoostTypingModel` implements it (native Booster JSON).

### IScorer (`scoring/base.py`)
- `fitness(layout) -> float`. `ModelScorer` wraps one or two `TypingModel`s + corpus freqs.
- Optional incremental delta-scoring later (only rescore ngrams whose keys moved) — behind
  the same interface, so it's an optimization, not an API change.

### IOptimizer (`optimize/base.py`)
- `optimize(layout, scorer) -> Layout`. `SimulatedAnnealing`, plus `two_opt`/`three_opt`
  local-search functions. **Seeded RNG** for reproducibility.

### CLIs (`keybo/cli/`)
`python -m keybo <cmd>`:
- `optimize` — SA (+ local search) → best layout. (was simulated_annealing.py/two_opt/three_opt)
- `score` — compare named layouts on the objective. (was print_stats.py)
- `train` — fit model(s) from stroke tsv, report CV. (was wpm_conditioned_model.py)
- `tune` — hyperparameter search. (was hyperparameter_tuning.py)
- `process-data` — raw keystroke dump → stroke tsv. (was process_dataset.py, FIXED)

## Bugs fixed as part of the rewrite

| # | Bug | Where it goes |
|---|-----|---------------|
| 1 | 3-opt multi-swap undo corrupts layout | `Layout.undo` swap-stack |
| 2 | Objective blind to 2 of 30 keys | `NgramFeatures` scores all layout keys |
| 3 | SFB feature mislabels index/middle as same-finger | `NgramFeatures` correct SFB |
| 4 | Trigram scorer loads freq from literal `"bigrams_file"` | `data/corpus.py` typed loader |
| 6 | `process_dataset` crashes on `Keyboard(rows, spacebar_pos=)` | rebuilt `data/keystrokes.py` |
| 11 | No RNG seeding | seed plumbed through optimizer CLI |
| 12 | `gpu_hist`/`gpu_id` XGBoost params break on modern ver | `device` param in model |
| 13 | `__main__` pickle fragility | JSON + sidecar in model |
| 14 | SA never converges on a plateau/tie landscape: `delta==0` moves were "accepted" via Metropolis and decremented the convergence counter forever (found in Phase 6) | `optimize/annealing.py`: convergence counter tracks improvements to the global *best* (monotonic, bounded) + a `max_outer` cap |
| — | Duplicated/drifted feature code across 2 files | single `NgramFeatures` |
| — | Dead code (bg_scores, unreachable penalties) | dropped |

## Testing strategy (this is a primary deliverable)

- **pytest**, tests mirror package layout, target broad coverage of every unit.
- **Tests encode CORRECT behavior, not current output** — we're fixing bugs, so we must
  not pin the bugs. Each fixed bug (above) gets a dedicated regression test with a
  **negative control** (confirm it fails against the old broken behavior).
- **Golden-master** only for behavior we deliberately preserve (e.g. SA temperature/stopping
  math, geometry distances) — snapshot + assert stable.
- **Feature pipeline** gets the heaviest coverage (it's the skew risk): hand-computed
  expected vectors for representative bigrams/trigrams (SFB, alternating, scissor, space).
- **Fixtures, not the 136M dump:** tiny synthetic keystroke files for the D and C paths.
- **Fast:** no network, no GPU, sub-second unit tests; a couple slower end-to-end tests
  (tiny SA run is deterministic under a fixed seed) marked `slow`.

## Resolved — old artifacts are disposable

Decision (user, 2026-07-03): the old `bigram_model.pkl` / `trigram_model.pkl` and the
old processed `.tsv`s do NOT need to be preserved or loadable. The only requirement is
that the rewrite can **redo all the work** end-to-end. So:
  - No legacy pickle shim. Old `.pkl`s are archived under `legacy/` and never loaded.
  - `TypingModel.load` only accepts models saved by the new code (JSON + versioned
    sidecar). A feature-version mismatch is a hard, loud failure.
  - "Redo all the work" = the D→C→A/B pipeline is fully runnable: `process-data` (needs
    the raw keystroke dump, supplied by the user — not in the repo), then `train`, then
    `score`/`optimize`. The corpus frequency files (`trigrams.txt`, `bigrams.txt`,
    `1-skip.txt`) ARE in the repo, so only a trained model is needed for `score`/`optimize`.

## Confirmed behavioral change

Bug #2 fix (objective sees all 30 keys) will change the layouts `optimize` produces
versus the old `logfile.txt` bests. This is intended — the old objective was blind to 2
keys. Tests encode the corrected behavior; old outputs are not treated as golden.

## Migration approach

Strangler pattern, package built alongside the old scripts; old scripts moved to `legacy/`
only once the equivalent CLI + tests are green. Frequent commits. A written
implementation plan (bite-sized TDD tasks) follows this spec's approval.
```
