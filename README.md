# keybo

A data-driven keyboard layout optimizer. `keybo` learns how long bigrams and trigrams take
to type from a large corpus of real keystroke data, then searches for a 30-key layout that
minimizes predicted typing time over an English corpus.

## Quick start

Requires [Nix](https://nixos.org) (flakes enabled) on a Linux host.

```bash
nix develop        # enter the dev shell (creates .venv via uv)
just install       # editable install of the keybo package + dev deps
just doctor        # verify python, the data stack, and the CLI
```

Then see `just --list` for the task recipes (process-data, train, optimize, score, tune,
test, lint, fmt, lock).

Without Nix, any Python ≥ 3.11 environment works too:

```bash
uv venv && source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"             # or: pip install -e ".[dev]"
```

## Workflows

All four are exposed as `just` recipes, and directly behind `python -m keybo <command>` (or
the `keybo` console script). The usual pipeline is **process-data → train → score / optimize**.

| Command | What it does |
|---------|--------------|
| `fetch-data`   | Download + extract the public 136M Keystrokes dataset (~1.5 GB, resumable). |
| `process-data` | Turn that keystroke dump into a bistroke/tristroke training table (TSV). |
| `train`        | Fit a typing-time model from that table; writes `model.json` + `model.meta.json`. |
| `optimize`     | Search for a layout that minimizes predicted typing time (simulated annealing + 2-opt). |
| `score`        | Compare named layouts on the learned objective. |
| `tune`         | Randomized hyperparameter search for the model. |

Example end-to-end run (via `just`):

```bash
# 0. Fetch the raw keystroke dataset (~1.5 GB) into dataset/.
just fetch-data

# 1. Build training tables from the dump.  (`just data` chains steps 0+1.)
just process-data dataset/Keystrokes/files dataset/Keystrokes/files/metadata_participants.txt bigram bistrokes.tsv

# 2. Train a bigram typing-time model.
just train bistrokes.tsv bigram models/bigram.json

# 3. Compare known layouts, or search for a new one.
just score models/bigram.json bigram
just optimize models/bigram.json bigram 0
```

For a **trigram** model instead, process/train with `trigram` and score/optimize with the
`trigram` objective (uses `data/corpus/trigrams.txt` by default):

```bash
just process-data dataset/Keystrokes/files dataset/Keystrokes/files/metadata_participants.txt trigram tristrokes.tsv
just train tristrokes.tsv trigram models/trigram.json
just score models/trigram.json trigram
just optimize models/trigram.json trigram 0
```

The same commands directly (equivalent to the recipes above):

```bash
keybo fetch-data --out-dir dataset
keybo process-data --files-dir dataset/Keystrokes/files \
    --metadata dataset/Keystrokes/files/metadata_participants.txt \
    --ngram bigram --output bistrokes.tsv
keybo train --strokes bistrokes.tsv --ngram bigram --output models/bigram.json --target-wpm 90
keybo score    --model models/bigram.json --ngram bigram
keybo optimize --model models/bigram.json --ngram bigram --seed 0
```
```

`fetch-data` pulls the **136M Keystrokes dataset** (Aalto University, Dhakal et al. 2018) —
the public keystroke dump `process-data` consumes. The download resumes if interrupted and
skips work that's already done. The English corpus frequency files under
`data/corpus/{trigrams,bigrams,1-skip}.txt` (derived from the licensed iWeb corpus, not
freely downloadable) **are already committed** in the repo, so there's nothing to fetch there.

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

The package lives under `src/keybo/`. Legacy pre-rewrite code is archived under `legacy/`
and is not imported by the package.

## Testing

```bash
just test          # unit tests (fast)          — or: pytest -q
just lint          # ruff check + format --check
just fmt           # auto-format + fix
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
