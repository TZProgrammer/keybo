# Optional frames and gates on mainline

Everything below is **on mainline and OFF by default**. The default path is byte-identical to what
shipped before these merges — verified, not asserted: the same 4,096-row bigram+trigram enumeration
digests to `4e849c04418e84d48acca3a8eec2963d5dab47a8cf3a4cd341c0f9ebb915d772` before and after, with
`FEATURE_VERSION` and both frame widths unchanged.

Each of these was **measured and returned a null**. They live here so the measurement is repeatable and
the code is not lost, **not because taking the path is recommended.** The ledger entry for each says so.

## Feature frames — `direction=` / `kitchensink=`

| frame | flag | bigram | trigram | version stamp |
|---|---|---|---|---|
| served (default) | — | 20 | 46 | `2026-07-05.3` |
| order-aware rolls | `direction=True` | 22 | 52 | `2026-07-05.3+direction.1` |
| all external features | `kitchensink=True` | 27 | 69 | `2026-07-05.3+kitchensink.1` |

`kitchensink` implies `direction`. The flag is threaded through
`features.ngram` → `training.train.build_training_matrix` → `training.validate.validate`, and inside
`validate` it reaches **both** the fold model and `_predict_cells` together — training and evaluation
must agree on the frame, or a model is scored on a matrix it was not fitted for.

**Why additive and not in place.** `inwards`/`outwards` are columns 17/18 of the served frame, and all
six `data/models/k31/` models carry `FEATURE_VERSION`. `models/base.py` errors on a version *mismatch*,
not on a column whose *meaning* changed — so redefining a served column would leave every model loading
fine while scoring a frame that no longer matches its training data. Silent train/serve skew. Each
widened frame therefore gets its own stamp, and a model trained on one can never be confused for a model
trained on another.

**What they measured.** Five consecutive nulls (NGRAM-FE, ARM-M, direction ×2, kitchen-sink). The last
used 3.1× the feature weight of the direction round (7.15% of gain) and still produced a negative
trigram transfer delta with 0 of 4 folds sign-consistent. Ledger: `DIRECTION-1`, `REDIRGATE-1`,
`RETRAIN-DIRECTION-1`, `SFGATED-EVAL-1`, `KITCHENSINK-1`.

## High-WPM non-regression gate — `baseline_buckets=`

```python
report = validate(rows, seeds=[0, 1, 2], baseline_buckets=incumbent_bucket_rhos)
require_no_high_wpm_regression_in_report(report, "my arm")   # RAISES on a structural regression
```

`validate()` always emits a `high_wpm_gate` block per fold/seed; passing `baseline_buckets` makes it a
*verdict* rather than a number, and the `require_…` call makes it *binding*. Floor is
`HIGH_WPM_FLOOR = 80`, and every bucket at or above it is checked independently.

**Structural vs noise is load-bearing.** A bucket regressing on *every* seed of a fold refuses (the
`dvorak` 120+ bucket at 3/3 seeds, −0.0326/−0.0306/−0.0316). One regressing on *some* seeds passes, so
seed wobble cannot veto an arm. An ungated fold, or a missing block, also refuses — "not measured" is not
"did not regress".

## Combined normalized objective — `keybo optimize --model-weight`

Absent by default, so the shipped objective stays `ms/char`. Weights registered at
aalto-n 0.5411 / comm-n 0.3977 / pool-n 0.0612 from held-out predictive skill.

**Registered verdict: HOLD.** Not better — *source-relative*. `ms/char` wins 4/4 folds on AALTO-held-out
while the blends win 9/9 on COMMUNITY-held-out, and the blend regresses every wpm bucket, worst at the
fastest. If it is ever adopted, use the two-source 50/50 form: the held-out skill gap is 1.45 SE
(p ≈ 0.15) and rests on the noisier of the two estimates. Ledger: `NORMGAUGE-1`, `NORMGAUGE-GATE-1/-3`.

## Corpora — `--corpus` / `KEYBO_CORPUS`

`blend-v1` is the production default; `iweb` remains resolvable by name. `production_corpus_dir()`
resolves relative to the **imported module's** tree, so a worktree run needs
`PYTHONPATH=<worktree>/src` or it silently scores the main clone's `data/`. `tests/data/` now fails loudly
with that remedy in the message rather than producing three confusing path mismatches.
