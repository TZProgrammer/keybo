#!/usr/bin/env bash
# GATEFOLDS-1 INVARIANT 4 — the mutation battery for the assertions this arm ADDS.
#
# An assertion that stays GREEN when the code it checks is broken tests nothing. So every new
# assertion gets a mutation aimed at IT specifically, and a SURVIVOR is a finding, not a pass.
#
# The four correctness points live in mutate.sh (purge __pycache__ before AND after, `python -B`,
# assert the mutation actually changed the file, restore in a trap, rc straight from pytest never
# from a pipe tail). ⚠ The purge is the one that mattered: a `.bak` restored in the same second at
# the same byte size satisfies CPython's (source_mtime, source_size) .pyc check, so a sibling's
# first run reported 3 SURVIVORS that were ALL FALSE -- MUTATED BYTECODE against RESTORED SOURCE.
#
# Mutations target the SOURCE the tests assert about (src/keybo/...), never the tests themselves:
# mutating a test to make it fail proves nothing.

set -u
cd "$(dirname "$0")" || exit 2
M=./mutate.sh
T=tests/features/test_frame_pace_invariance.py
SCHEMA=src/keybo/features/schema.py
NGRAM=src/keybo/features/ngram.py

mkdir -p /tmp/gatefolds_wk
echo "=== GATEFOLDS-1 mutation battery ==="
echo

# --- 1. Break the INVARIANCE of interp.1 by giving its frame a wpm column ----------------
# If the tests cannot tell a 10-column wpm-free frame from an 11-column one carrying pace, the
# whole mechanism claim is untestable.
$M M1 "$SCHEMA" \
  'BIGRAM_INTERP_FEATURE_NAMES = [*_BIGRAM_INTERP_NAMES]' \
  'BIGRAM_INTERP_FEATURE_NAMES = [*_BIGRAM_INTERP_NAMES, "wpm"]' \
  "$T"

# --- 2. Break hybrid-B's invariance the same way -----------------------------------------
$M M2 "$SCHEMA" \
  '    *_HYBRIDB_FINGER_ONEHOTS,
]' \
  '    *_HYBRIDB_FINGER_ONEHOTS,
    "wpm",
]' \
  "$T"

# --- 3. Remove wpm from the SERVED frame --------------------------------------------------
# The opposite polarity: the tests must also catch a served frame that STOPPED carrying pace,
# otherwise the "served frame does vary" assertions are vacuous.
$M M3 "$SCHEMA" \
  'BIGRAM_FEATURE_NAMES = [*_BIGRAM_PLACEMENT_NAMES, "wpm"]' \
  'BIGRAM_FEATURE_NAMES = [*_BIGRAM_PLACEMENT_NAMES]' \
  "$T"

# --- 4. Strip wpm from the interp-wpm VARIANT's name list ---------------------------------
# Targets test_the_interp_wpm_variant_does_vary_with_pace specifically.
$M M4 "$SCHEMA" \
  '_BIGRAM_INTERP_WPM_NAMES = [*_BIGRAM_INTERP_NAMES, "wpm"]' \
  '_BIGRAM_INTERP_WPM_NAMES = [*_BIGRAM_INTERP_NAMES]' \
  "$T"

# --- 5. Make the interp BUILDER leak pace into an existing column ------------------------
# The subtlest and most important mutation: the NAME LIST stays 10 wide (so a schema-only test
# still passes) while the BUILDER writes pace into a column that is already there. Only a test that
# actually EVALUATES the featurizer across wpm can catch this. If M5 survives, my tests are
# checking names rather than behaviour.
#
# `del wpm` is deleted first so the value is still in scope, then the pace is folded into a real
# column. Kept tiny (1e-9) on purpose: a mutation that shifted a column by 10.0 could be caught by
# some unrelated golden-value test, which would not prove MY assertion works.
$M M5 "$NGRAM" \
  '    del wpm  # not a column in this frame (see schema); accepted for call-shape parity only
    row = interp_row_from_positions(geometry, positions[0], positions[1])' \
  '    row = interp_row_from_positions(geometry, positions[0], positions[1])
    row["row_load"] = row["row_load"] + wpm * 1e-9' \
  "$T"

# --- 6. Make the SERVED builder ignore pace ----------------------------------------------
# The mirror of M5 on the other polarity: the served frame keeps its 20 names and its wpm column,
# but the column stops carrying pace. Catching this is what makes
# test_the_served_frame_does_vary_with_pace non-vacuous.
#
# ⚠ `row["wpm"] = float(wpm)` occurs FOUR times in ngram.py (bigram_model_row, this one, the
# interp-wpm row, the trigram row) and mutate.sh replaces the FIRST occurrence only -- which would
# hit `bigram_model_row`, a site my tests never call. So the pattern is anchored on the trailing
# `np.array(...)` lines UNIQUE to `bigram_features_from_positions`, the function the tests use.
$M M6 "$NGRAM" \
  '    row["wpm"] = float(wpm)
    return np.array(
        [row[name] for name in _bigram_column_names(direction, kitchensink)], dtype=np.float64
    )' \
  '    row["wpm"] = 0.0
    return np.array(
        [row[name] for name in _bigram_column_names(direction, kitchensink)], dtype=np.float64
    )' \
  "$T"

# --- 7. Break the to_ms rank-preservation demonstration ----------------------------------
# test_to_ms_pace_factor_is_rank_preserving_within_a_bucket computes the conversion inline, so no
# src mutation can reach it -- which means the test can only be VACUOUS or CORRECT, never
# silently broken by a code change. Recorded here as NOT-MUTATED with the reason, rather than
# omitted (a battery that quietly skips an assertion reads as full coverage).
echo "M7 NOT-MUTATED (by design) :: to_ms rank-preservation is demonstrated on inline values, so no src/ mutation can reach it; its non-vacuity is enforced by the paired 'values DO change' assertion in the same test"

echo
echo "=== done. SURVIVED lines above are findings. Logs: /tmp/gatefolds_wk/mut_*.log ==="
