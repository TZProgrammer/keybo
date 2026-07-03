# keybo

A data-driven keyboard layout optimizer. `keybo` learns how long bigrams and trigrams take
to type from a large corpus of real keystroke data, then searches for a 30-key layout that
minimizes predicted typing time over an English corpus.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Dependencies (numpy, scikit-learn, xgboost, tqdm) are pinned in
`pyproject.toml`.

## Workflows

All four are exposed behind `python -m keybo <command>` (or the `keybo` console script). The
usual pipeline is **process-data → train → score / optimize**.

| Command | What it does |
|---------|--------------|
| `process-data` | Turn a raw keystroke dump into a bistroke/tristroke training table (TSV). |
| `train`        | Fit a typing-time model from that table; writes `model.json` + `model.meta.json`. |
| `optimize`     | Search for a layout that minimizes predicted typing time (simulated annealing + 2-opt). |
| `score`        | Compare named layouts on the learned objective. |
| `tune`         | Randomized hyperparameter search for the model. |

Example end-to-end run:

```bash
# 1. Build training tables from the raw keystroke dump (you supply the dump).
keybo process-data --files-dir dataset/Keystrokes/files \
    --metadata dataset/Keystrokes/files/metadata_participants.txt \
    --ngram bigram --output bistrokes.tsv

# 2. Train a bigram typing-time model.
keybo train --strokes bistrokes.tsv --ngram bigram --output models/bigram.json --target-wpm 90

# 3. Compare known layouts, or search for a new one.
keybo score    --model models/bigram.json --bigram-freqs data/corpus/bigrams.txt
keybo optimize --model models/bigram.json --bigram-freqs data/corpus/bigrams.txt --seed 0
```

The raw 136M-keystroke dump is **not** included in the repo (it is large and external); you
point `process-data` at your own copy. The English corpus frequency files it needs for
scoring (`data/corpus/{trigrams,bigrams,1-skip}.txt`) **are** included.

## Architecture

The package is built around plug-and-play seams, so swapping an implementation (a different
model, scorer, or optimizer) is a one-file change.

```
Layout (over a Geometry)
   │  feature pipeline  (keybo.features — ONE shared code path)
   ▼
TypingModel  (features -> predicted time; XGBoost today)   [keybo.models]
   │
   ▼
IScorer      (layout -> fitness = Σ predicted_time · frequency)   [keybo.scoring]
   │
   ▼
IOptimizer   (simulated annealing, 2-opt / 3-opt)   [keybo.optimize]
```

- **geometry** — the physical board: signed-column key positions, finger map, row-stagger
  distances. Ships one instance, `ROW_STAGGERED_30` (the geometry the data supports).
- **layout** — a character→position assignment with swap/undo (stack-based, so multi-swap
  moves undo correctly).
- **features** — the single n-gram feature pipeline, used identically by data processing,
  training, and scoring. This is the guard against train/serve skew. A `FEATURE_VERSION`
  is stamped into each model; loading a model built on a different version is a hard error.
- **models** — `TypingModel` interface + XGBoost implementation. Models save as native
  XGBoost JSON plus a metadata sidecar (no pickle).
- **scoring / optimize / data / training** — the objective, the search, dataset processing,
  and model fitting.

Legacy pre-rewrite code is archived under `legacy/` and is not imported by the package.

## Testing

```bash
pytest -q          # unit tests (fast)
ruff check .       # lint
```

## Known follow-up: optimize performance

`optimize` is correct but currently rescoring the *entire* corpus on every candidate swap
(~25 ms/evaluation on the full bigram corpus), so a full annealing run is slow. The intended
next step is **incremental delta-scoring** — only re-evaluate the n-grams whose keys moved —
which fits behind the existing `IScorer` interface without an API change. Until then, use
`--max-outer` to cap a run, or a smaller corpus.

## Design docs

- `docs/specs/2026-07-03--rewrite-design.md` — architecture and decisions.
- `docs/plans/2026-07-03-rewrite.md` — the phased, test-driven build plan.
