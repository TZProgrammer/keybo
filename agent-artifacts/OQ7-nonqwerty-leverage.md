# OQ-7 — How do we leverage the non-QWERTY data given heavy class imbalance?

**Status: 🔴 open — experiment matrix defined; blocked on the same prerequisites as OQ-5
(layout labels + harness), plus one cheap unblocked step: measure the actual imbalance.**

## Best current answer

**Default lean: keep non-QWERTY primarily as VALIDATION signal, and if used in training,
prefer per-sample weighting over resampling.** Reasoning:

- The non-QWERTY rows are the only off-QWERTY ground truth we have. Their scarcity makes them
  worth more as *test* (proving transfer, OQ-5) than as upsampled *train* mass. Burn them as
  training data and nothing is left to validate with.
- 🟡 Resampling-with-replacement adds no information — it reweights while *inflating apparent
  confidence*: a handful of Dvorak typists' idiosyncrasies get duplicated until the model (and
  the metrics) treat them as population truths. Fine for quick experiments; distrust for
  final numbers.
- 🟡 Per-sample weights (`sample_weight` in XGBoost, inverse layout frequency) achieve the
  same balancing intent without duplication and without lying to the variance estimates.
- 🟠 There may also be a *labeling* subtlety: participants are bucketed by self-reported
  LAYOUT, and the current pipeline trusts it. A "Dvorak" participant hunt-and-pecking on a
  QWERTY-labeled machine (or vice versa) contaminates the minority class far more than the
  majority. Worth a sanity screen: within-participant consistency of per-position timings.

**Interaction with OQ-1 (ordering matters):** upsampling changes the effective frequency
distribution a freq-feature model sees. Decide OQ-1 first; run OQ-7's matrix after.

## Unblocked first step (do this now, cheap)

Measure the real qualifying-participant distribution (the dump isn't on this box; run after
`just fetch-data`):

    /tmp/keybo_venv/bin/python - <<'EOF'
    from collections import Counter
    from keybo.data.keystrokes import load_participant_metadata
    md = load_participant_metadata('dataset/Keystrokes/files/metadata_participants.txt')
    c = Counter(r['LAYOUT'] for r in md.values())
    print(c.most_common(), 'total:', sum(c.values()))
    EOF

Everything downstream (bucket sizes, whether Dvorak has enough rows to hold out with power)
depends on these numbers. Do not design around a guessed split.

## Definitive close (experiment matrix)

Prereqs: layout labels + OQ-5 harness. Then a 4-arm comparison, judged on the OQ-8 matrix:

| Arm | Training set | Balancing |
|---|---|---|
| A | QWERTY only | — (non-QWERTY purely held out) |
| B | all layouts | none (natural imbalance) |
| C | all layouts | inverse-frequency sample weights |
| D | all layouts | resample-with-replacement to parity |

Judge each arm on: held-out-layout Spearman ρ (rotate the holdout; for B–D hold out a
*participant-level* slice of each layout to avoid leakage), worst-cell metrics, and QWERTY
regression (did balancing hurt the majority?). Pre-registered decision rule: pick the arm
with the best minority-layout ranking that does NOT degrade QWERTY ρ by more than the
bootstrap CI. If A ≈ B ≈ C ≈ D on minority ranking, the non-QWERTY data adds nothing to
training — keep it as pure validation (and say so in the README).
