# keybo task runner. Run `just` or `just --list` to see recipes.
# These are the exact commands that must work identically on the laptop and dev box.

# Default: list recipes
default:
    @just --list

# Install the package + deps into the active venv (editable)
install:
    uv pip install -e ".[dev]"

# Environment sanity check — proves the Nix shell + wheels import
doctor:
    python -c "import sys; print('python', sys.version.split()[0])"
    python -c "import numpy, scipy, sklearn, xgboost; print('data stack OK · xgboost', xgboost.__version__)"
    python -c "import keybo; print('keybo', keybo.__version__)"
    keybo --help >/dev/null && echo "keybo CLI OK"

# Run the test suite
test:
    pytest -q

# Include the slow end-to-end tests
test-all:
    pytest -q -m slow

# Lint + format check
lint:
    ruff check .
    ruff format --check .

# Auto-format
fmt:
    ruff format .
    ruff check --fix .

# Pin Python deps (run once to generate uv.lock, then commit it)
lock:
    uv lock

# --- workflows (fetch-data → process-data → train → score / optimize) -------------

# Download + extract the public 136M Keystrokes dataset (~1.5 GB, resumable) into dataset/.
fetch-data out="dataset":
    keybo fetch-data --out-dir {{out}}

# Turn a raw keystroke dump into a bistroke/tristroke table.
# e.g. `just process-data dataset/Keystrokes/files dataset/Keystrokes/files/metadata_participants.txt bigram bistrokes.tsv`
process-data files_dir metadata ngram="bigram" out="bistrokes.tsv":
    keybo process-data --files-dir {{files_dir}} --metadata {{metadata}} --ngram {{ngram}} --output {{out}}

# Convenience: fetch the dataset then process it to bistrokes.tsv in one go.
data ngram="bigram" out="bistrokes.tsv":
    keybo fetch-data --out-dir dataset
    keybo process-data --files-dir dataset/Keystrokes/files --metadata dataset/Keystrokes/files/metadata_participants.txt --ngram {{ngram}} --output {{out}}

# Fit a typing-time model from a stroke table.
# e.g. `just train bistrokes.tsv bigram models/bigram.json`
train strokes ngram="bigram" out="models/bigram.json" target_wpm="90":
    keybo train --strokes {{strokes}} --ngram {{ngram}} --output {{out}} --target-wpm {{target_wpm}}

# Hyperparameter search; writes best params JSON.
tune strokes ngram="bigram" out="best_hyperparams.json":
    keybo tune --strokes {{strokes}} --ngram {{ngram}} --output {{out}}

# Compare named layouts on the learned objective (bigram or trigram model).
# e.g. `just score models/bigram.json bigram`  ·  `just score models/trigram.json trigram`
score model="models/bigram.json" ngram="bigram":
    keybo score --model {{model}} --ngram {{ngram}}

# Search for a layout that minimizes predicted typing time (bigram or trigram model).
# e.g. `just optimize models/bigram.json bigram 0`  ·  `just optimize models/trigram.json trigram 0`
optimize model="models/bigram.json" ngram="bigram" seed="0":
    keybo optimize --model {{model}} --ngram {{ngram}} --seed {{seed}}
