#!/usr/bin/env bash
# HYBRIDB-1 INVARIANT 3 — the mutation battery for every assertion this arm added.
#
# Each line mutates ONE source string in a way that BREAKS a claim a test asserts, then runs the
# tests that should catch it. A SURVIVED line means the assertion tests nothing and must be fixed.
set -u
cd /local/home/zegertho/repos/keybo-wt-hybridtri || exit 2
M=agent-artifacts/hybridtri/mutate.sh
SCHEMA=src/keybo/features/schema.py
NGRAM=src/keybo/features/ngram.py
SHAP=src/keybo/analysis/shap_diff.py
TRAIN=src/keybo/training/train.py
T=tests/features/test_hybridb_frame.py
ALLT="tests/features/test_hybridb_frame.py tests/analysis/test_shap_diff_interp_frame.py tests/training/test_train_interp_frame.py"

# --- the frame's IDENTITY -----------------------------------------------------------------
bash $M M01 "$SCHEMA" '_HYBRIDB_ROW_ONEHOTS = ["bottom", "home", "top"]' \
                      '_HYBRIDB_ROW_ONEHOTS = ["bottom", "home"]' $T
bash $M M02 "$SCHEMA" '_HYBRIDB_FINGER_ONEHOTS = ["pinky", "ring", "middle", "index", "lateral"]' \
                      '_HYBRIDB_FINGER_ONEHOTS = ["pinky", "ring", "middle", "index"]' $T
bash $M M03 "$SCHEMA" '_HYBRIDB_ROW_ONEHOTS = ["bottom", "home", "top"]' \
                      '_HYBRIDB_ROW_ONEHOTS = ["bottom", "home", "top", "lsb"]' $T
# column ORDER: the monotone tuple is applied POSITIONALLY, so a reorder mis-constrains
bash $M M04 "$SCHEMA" '    *_BIGRAM_INTERP_NAMES,
    *_HYBRIDB_ROW_ONEHOTS,
    *_HYBRIDB_FINGER_ONEHOTS,' \
                      '    *_HYBRIDB_ROW_ONEHOTS,
    *_HYBRIDB_FINGER_ONEHOTS,
    *_BIGRAM_INTERP_NAMES,' $T
# a wpm column smuggled in -- would break the "no wpm" claim AND the stated-pace contract
bash $M M05 "$SCHEMA" '    *_HYBRIDB_FINGER_ONEHOTS,
]' '    *_HYBRIDB_FINGER_ONEHOTS,
    "wpm",
]' $T

# --- the MONOTONE tuple -------------------------------------------------------------------
bash $M M06 "$SCHEMA" '    *BIGRAM_INTERP_MONOTONE,
    *((0,) * (len(_HYBRIDB_ROW_ONEHOTS) + len(_HYBRIDB_FINGER_ONEHOTS))),' \
                      '    *((0,) * len(_BIGRAM_INTERP_NAMES)),
    *((0,) * (len(_HYBRIDB_ROW_ONEHOTS) + len(_HYBRIDB_FINGER_ONEHOTS))),' $T
bash $M M07 "$SCHEMA" '    *((0,) * (len(_HYBRIDB_ROW_ONEHOTS) + len(_HYBRIDB_FINGER_ONEHOTS))),' \
                      '    *((1,) * (len(_HYBRIDB_ROW_ONEHOTS) + len(_HYBRIDB_FINGER_ONEHOTS))),' $T
# a LENGTH mismatch: xgboost would map the tuple positionally and constrain the WRONG columns
bash $M M08 "$SCHEMA" '    *((0,) * (len(_HYBRIDB_ROW_ONEHOTS) + len(_HYBRIDB_FINGER_ONEHOTS))),' \
                      '    *((0,) * (len(_HYBRIDB_ROW_ONEHOTS) + len(_HYBRIDB_FINGER_ONEHOTS) - 1)),' $T

# --- the STAMP ----------------------------------------------------------------------------
bash $M M09 "$SCHEMA" 'FEATURE_VERSION_HYBRIDB = f"{FEATURE_VERSION}+hybrid-b.1"' \
                      'FEATURE_VERSION_HYBRIDB = FEATURE_VERSION' $T
bash $M M10 "$SCHEMA" 'FEATURE_VERSION_HYBRIDB = f"{FEATURE_VERSION}+hybrid-b.1"' \
                      'FEATURE_VERSION_HYBRIDB = FEATURE_VERSION_INTERP' $T
# the served stamp edited IN PLACE -- the one-way door the whole opt-in design exists to avoid
bash $M M11 "$SCHEMA" 'FEATURE_VERSION = "2026-07-05.3"' 'FEATURE_VERSION = "2026-07-05.4"' $ALLT

# --- the FEATURIZER -----------------------------------------------------------------------
# re-derive instead of reuse: silently drops the interp half
bash $M M12 "$NGRAM" '    row = interp_row_from_positions(geometry, a, b)
    placement = _placement_row_from_positions(geometry, a, b)' \
                     '    row = {}
    placement = _placement_row_from_positions(geometry, a, b)' $T
# the KeyError guard turned into a silent zero-fill -- a fabricated column reading as measured
bash $M M13 "$NGRAM" '            row[name] = placement[name]' \
                     '            row[name] = placement.get(name, 0.0)' $T
# wpm no longer ignored -- would make the frame pace-dependent without a wpm column
bash $M M14 "$NGRAM" '    del wpm  # not a column in this frame (see schema); accepted for call-shape parity only
    row = hybridb_row_from_positions(geometry, positions[0], positions[1])
    return np.array([row[name] for name in BIGRAM_HYBRIDB_FEATURE_NAMES], dtype=np.float64)' \
                     '    row = hybridb_row_from_positions(geometry, positions[0], positions[1])
    row["hand_conflict"] = row["hand_conflict"] + wpm
    return np.array([row[name] for name in BIGRAM_HYBRIDB_FEATURE_NAMES], dtype=np.float64)' $T

# --- the REGISTRY -------------------------------------------------------------------------
# the near-miss fallback the registry exists to prevent
bash $M M15 "$NGRAM" '    if interp is False or interp not in _REPLACEMENT_FRAMES:
        raise ValueError(' \
                     '    if False:
        raise ValueError(' $T
bash $M M16 "$NGRAM" '    return _REPLACEMENT_FRAMES[interp]' \
                     '    return _REPLACEMENT_FRAMES.get(interp, _REPLACEMENT_FRAMES[True])' $T
# hybridb resolving to the WRONG builder -- 18 names with a 10-column vector
bash $M M17 "$NGRAM" '    "hybridb": (
        hybridb_features_from_positions,' \
                     '    "hybridb": (
        interp_features_from_positions,' $T
# hybridb resolving to the wrong STAMP: the train/serve skew the stamp exists to prevent
bash $M M18 "$NGRAM" '        BIGRAM_HYBRIDB_MONOTONE,
        FEATURE_VERSION_HYBRIDB,' \
                     '        BIGRAM_HYBRIDB_MONOTONE,
        FEATURE_VERSION_INTERP,' $T
# hybridb resolving to the wrong NAME LIST
bash $M M19 "$NGRAM" '        hybridb_features_from_positions,
        BIGRAM_HYBRIDB_FEATURE_NAMES,' \
                     '        hybridb_features_from_positions,
        BIGRAM_INTERP_FEATURE_NAMES,' $T
# the flag list silently widened
bash $M M20 "$NGRAM" 'REPLACEMENT_FRAME_FLAGS = tuple(_REPLACEMENT_FRAMES)' \
                     'REPLACEMENT_FRAME_FLAGS = (*tuple(_REPLACEMENT_FRAMES), "hybrid-b")' $T

# --- the BLOCK PARTITION ------------------------------------------------------------------
# the whole point of (b): put the one-hots in their OWN block, so a block sum stops being
# invariant to ordinal<->one-hot credit leakage
bash $M M21 "$SHAP" '    **{n: ("ROWCOST", "onehot") for n in ("bottom", "home", "top")},' \
                    '    **{n: ("ROW", "") for n in ("bottom", "home", "top")},' $T
bash $M M22 "$SHAP" '    **{n: ("CONTACT", "onehot") for n in ("pinky", "ring", "middle", "index", "lateral")},' \
                    '    **{n: ("FINGER", "") for n in ("pinky", "ring", "middle", "index", "lateral")},' $T
# the partition made INCOMPLETE -- block_map must refuse, not report a partial table
bash $M M23 "$SHAP" \
  '    **{n: ("SPAN", "") for n in ("row_span", "lateral_span", "same_hand_travel")},
    "roll_inward": ("DIRECTION", ""),
}' \
  '    "roll_inward": ("DIRECTION", ""),
}' $T
# the sub-block label flattened to a constant -- a field whose subject cannot vary
bash $M M24 "$SHAP" '    **{n: ("ROWCOST", "ordinal") for n in ("row_load", "row_arrival", "bottom_bias")},' \
                    '    **{n: ("ROWCOST", "onehot") for n in ("row_load", "row_arrival", "bottom_bias")},' $T
# the refusal removed: an unknown frame silently reported instead of refused
bash $M M25 "$SHAP" '    raise ValueError(
        f"no block partition registered for this {len(names)}-column frame; add one to "' \
                    '    return {n: ("ROW", "") for n in names}
    raise ValueError(
        f"no block partition registered for this {len(names)}-column frame; add one to "' \
                    tests/analysis/test_shap_diff_interp_frame.py
# FRAMES no longer carrying hybridb
bash $M M26 "$SHAP" 'FRAMES = ("served", "interp", "interp-wpm", "hybridb")' \
                    'FRAMES = ("served", "interp", "interp-wpm")' $T
# the frame-name -> flag mapping pointing hybridb at the interp frame
bash $M M27 "$SHAP" '    "hybridb": "hybridb",' '    "hybridb": True,' $ALLT

# --- the TRAIN-path refusals ---------------------------------------------------------------
bash $M M28 "$TRAIN" '    if interp is not False and interp not in REPLACEMENT_FRAME_FLAGS:' \
                     '    if False:' $ALLT
bash $M M29 "$TRAIN" '        if ngram != "bigram":' '        if False:' \
                     tests/training/test_train_interp_frame.py
bash $M M30 "$TRAIN" '        if direction or kitchensink:' '        if False:' \
                     tests/training/test_train_interp_frame.py
# the monotone tuple never reaching xgboost -- "present is not effective"
bash $M M31 "$TRAIN" '            params = {**params, "monotone_constraints": tuple(constraints)}' \
                     '            params = {**params}' tests/training/test_train_interp_frame.py
# the length assertion dropped: a short tuple would constrain the WRONG columns silently
bash $M M32 "$TRAIN" '            assert len(constraints) == len(names), "one constraint per column"' \
                     '            constraints = constraints[:1]' tests/training/test_train_interp_frame.py

# --- re-runs of the FIRST battery's four survivors, now that each has a dedicated test -------
# M13 is an EQUIVALENT MUTANT (the KeyError branch is unreachable from real data -- all 18 schema
# names are present in one of the two row builders). It is expected to SURVIVE and is covered
# constructively instead, the route FRAMEDIAG-1 took for its own unreachable state. Re-run so the
# equivalence is recorded rather than assumed.
bash $M M13b "$NGRAM" '            row[name] = placement[name]' \
                      '            row[name] = placement.get(name, 0.0)' $T
# M16/M27/M28 were REAL gaps and now have dedicated tests.
bash $M M16b "$NGRAM" '    return _REPLACEMENT_FRAMES[interp]' \
                      '    return _REPLACEMENT_FRAMES.get(interp, _REPLACEMENT_FRAMES[True])' $T
bash $M M27b "$SHAP" '    "hybridb": "hybridb",' '    "hybridb": True,' $T
bash $M M28b "$TRAIN" '    if interp is not False and interp not in REPLACEMENT_FRAME_FLAGS:' \
                      '    if interp and interp not in REPLACEMENT_FRAME_FLAGS:' $T
