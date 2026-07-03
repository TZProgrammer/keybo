"""Tests for keystroke-dump processing (the data pipeline, workflow D).

Regression for bug #6: the old code built its per-participant keyboard with
``Keyboard(rows, spacebar_pos=(0,-1))``, a signature its own Keyboard class did not accept,
so the whole pipeline raised TypeError. Here the character map builds via the new API and
the pipeline runs end-to-end on synthetic records.
"""

import pytest

from keybo.data.keystrokes import (
    NGRAM_SPECS,
    aggregate_occurrences,
    build_char_map,
    compute_session_wpm,
    extract_occurrences,
    load_participant_metadata,
    mark_correct_flags,
    process_dataset,
    write_ngram_tsv,
)


def rec(letter, press, release=None):
    return {"LETTER": letter, "PRESS_TIME": str(press), "RELEASE_TIME": str(release or press)}


# --- char map (regression #6) ---------------------------------------------------------


def test_regression_bug6_build_char_map_does_not_raise():
    # The exact thing the old pipeline crashed on: constructing a per-layout keyboard.
    cmap = build_char_map("qwerty")
    assert cmap  # non-empty


def test_char_map_positions_match_geometry():
    cmap = build_char_map("qwerty")
    assert cmap["q"] == (-5, 3)  # top-left
    assert cmap["p"] == (5, 3)  # top-right
    assert cmap["f"] == (-2, 2)  # home index
    assert cmap[" "] == (0, 0)  # space/thumb


def test_char_map_supports_known_layouts():
    for name in ("qwerty", "azerty", "dvorak", "qwertz"):
        cmap = build_char_map(name)
        assert cmap[" "] == (0, 0)


def test_unknown_layout_raises():
    with pytest.raises(ValueError):
        build_char_map("colemak")


# --- WPM -------------------------------------------------------------------------------


def test_compute_session_wpm():
    # 25 correct chars over 60000 ms (1 min) -> (25/5)/1 = 5 wpm.
    assert compute_session_wpm(first_press_ms=0, last_press_ms=60000, n_correct=25) == 5


def test_compute_session_wpm_guards_tiny_duration():
    # Zero elapsed time must not divide by zero.
    wpm = compute_session_wpm(first_press_ms=1000, last_press_ms=1000, n_correct=5)
    assert wpm >= 0


# --- correctness marking ---------------------------------------------------------------


def test_mark_correct_all_true_when_typed_matches_expected():
    flags = mark_correct_flags(typed="the", expected="the")
    assert flags == [True, True, True]


def test_mark_correct_flags_mismatches():
    flags = mark_correct_flags(typed="thx", expected="the")
    assert flags[0] is True and flags[1] is True and flags[2] is False


# --- n-gram extraction -----------------------------------------------------------------


def make_session(word, start=1000, step=100):
    """A session that typed `word` correctly, one key per `step` ms."""
    records = []
    t = start
    for ch in word:
        records.append({**rec(ch, t, t + 50), "SENTENCE": word})
        t += step
    return records


def test_extract_bigrams_from_clean_session():
    cmap = build_char_map("qwerty")
    records = make_session("the")
    n, skip = NGRAM_SPECS["bigram"]
    occ = extract_occurrences(records, cmap, n=n, skip=skip, time_mode="full")
    ngrams = [o.ngram for o in occ]
    assert ngrams == ["th", "he"]


def test_extract_trigrams_from_clean_session():
    cmap = build_char_map("qwerty")
    records = make_session("the")
    n, skip = NGRAM_SPECS["trigram"]
    occ = extract_occurrences(records, cmap, n=n, skip=skip, time_mode="full")
    assert [o.ngram for o in occ] == ["the"]


def test_extract_skipgrams_skips_the_middle_key():
    cmap = build_char_map("qwerty")
    records = make_session("the")
    n, skip = NGRAM_SPECS["skipgram"]
    occ = extract_occurrences(records, cmap, n=n, skip=skip, time_mode="full")
    # skipgram over "the" = 't' and 'e'
    assert [o.ngram for o in occ] == ["te"]


def test_occurrence_records_positions_and_duration():
    cmap = build_char_map("qwerty")
    records = make_session("th", start=1000, step=100)  # t press@1000, h press@1100
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    o = occ[0]
    assert o.positions == (cmap["t"], cmap["h"])
    assert o.wpm >= 0
    assert o.duration == 100  # 1100 - 1000


# --- filters ---------------------------------------------------------------------------


def test_banned_key_window_is_dropped():
    cmap = build_char_map("qwerty")
    records = [
        {**rec("t", 1000), "SENTENCE": "tae"},
        {**rec("SHIFT", 1100), "SENTENCE": "tae"},
        {**rec("e", 1200), "SENTENCE": "tae"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    # Any window containing SHIFT is dropped, so no bigram survives.
    assert occ == []


def test_multi_char_letter_is_ignored():
    cmap = build_char_map("qwerty")
    records = [
        {**rec("the whole sentence", 1000), "SENTENCE": "the"},
        {**rec("t", 1100), "SENTENCE": "the"},
        {**rec("h", 1200), "SENTENCE": "the"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    assert [o.ngram for o in occ] == ["th"]


def test_key_not_in_layout_is_dropped():
    cmap = build_char_map("qwerty")  # no 'ö'
    records = [
        {**rec("a", 1000), "SENTENCE": "aö"},
        {**rec("ö", 1100), "SENTENCE": "aö"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    assert occ == []


# --- aggregation + tsv -----------------------------------------------------------------


def test_aggregate_counts_frequency_and_collects_samples():
    cmap = build_char_map("qwerty")
    occ = extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full")
    occ += extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full")
    agg = aggregate_occurrences(occ)
    # 'th' seen twice.
    key = next(k for k in agg if k[1] == "th")
    assert agg[key]["frequency"] == 2
    assert len(agg[key]["occurrences"]) == 2


def test_write_ngram_tsv_roundtrips(tmp_path):
    cmap = build_char_map("qwerty")
    occ = extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full")
    agg = aggregate_occurrences(occ)
    out = tmp_path / "bistrokes.tsv"
    write_ngram_tsv(agg, str(out))
    text = out.read_text()
    assert "th" in text
    # Format: positions<tab>ngram<tab>freq<tab>(wpm, dur)...
    first = text.splitlines()[0].split("\t")
    assert first[1] in ("th", "he")
    assert int(first[2]) >= 1


# --- file-level orchestration (the CLI-facing path) -----------------------------------

_METADATA_HEADER = "PARTICIPANT_ID\tFINGERS\tAVG_WPM_15\tKEYBOARD_TYPE\tLAYOUT"
_KEYSTROKE_HEADER = "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tPRESS_TIME\tRELEASE_TIME\tLETTER"


def _make_dataset(tmp_path):
    files = tmp_path / "files"
    files.mkdir()
    meta = files / "metadata_participants.txt"
    meta.write_text(
        _METADATA_HEADER
        + "\n"
        + "111\t9-10\t90\tfull\tqwerty\n"  # kept
        + "222\t1-2\t90\tfull\tqwerty\n"  # dropped: not a touch typist
        + "\n"
    )
    # Participant 111 typed "the" correctly in one session.
    lines = [_KEYSTROKE_HEADER]
    t = 1000
    for ch in "the":
        lines.append(f"111\ts1\tthe\t{t}\t{t + 50}\t{ch}")
        t += 100
    (files / "111_keystrokes.txt").write_text("\n".join(lines) + "\n")
    return str(files), str(meta)


def test_load_participant_metadata_filters(tmp_path):
    _, meta = _make_dataset(tmp_path)
    md = load_participant_metadata(meta)
    assert "111" in md  # touch typist kept
    assert "222" not in md  # non-touch-typist dropped


def test_process_dataset_end_to_end(tmp_path):
    files_dir, meta = _make_dataset(tmp_path)
    agg = process_dataset(files_dir, meta, ngram="bigram", time_mode="full")
    ngrams = {ngram for _, ngram in agg}
    assert ngrams == {"th", "he"}


def test_process_dataset_rejects_unknown_ngram(tmp_path):
    files_dir, meta = _make_dataset(tmp_path)
    with pytest.raises(ValueError):
        process_dataset(files_dir, meta, ngram="quadgram")
