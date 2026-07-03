# keybo

A data-driven keyboard layout optimizer. `keybo` learns how long bigrams and trigrams take
to type from a large corpus of real keystroke data, then searches for a 30-key layout that
minimizes predicted typing time over an English corpus.

> **Status:** under active rewrite. See `docs/specs/2026-07-03--rewrite-design.md` for the
> architecture and `docs/plans/2026-07-03-rewrite.md` for the build plan. Pre-rewrite code
> lives in `legacy/`.

## Install

```bash
pip install -e ".[dev]"
```

## Workflows

All four are exposed behind `python -m keybo <command>`:

| Command | What it does |
|---------|--------------|
| `process-data` | Turn a raw keystroke dump into bistroke/tristroke training tables. |
| `train`        | Fit a typing-time model from those tables. |
| `optimize`     | Search for a layout that minimizes predicted typing time. |
| `score`        | Compare named layouts on the learned objective. |
| `tune`         | Hyperparameter search for the model. |

## Testing

```bash
pytest -q          # unit tests
pytest -q -m slow  # include the slow end-to-end tests
ruff check .       # lint
```

## Architecture (one line each)

- **geometry** — the physical board: key positions, finger map, distances.
- **layout** — a character→position assignment over a geometry (with swap/undo).
- **features** — the single n-gram feature pipeline shared by processing, training, scoring.
- **models** — `TypingModel` implementations (features → predicted time); XGBoost today.
- **scoring** — `IScorer` implementations (layout → fitness).
- **optimize** — `IOptimizer` implementations (simulated annealing, 2-opt/3-opt).
- **data / training** — dataset processing and model fitting.
