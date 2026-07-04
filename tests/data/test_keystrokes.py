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
    r = {"LETTER": letter, "PRESS_TIME": str(press)}
    if release is not None:
        r["RELEASE_TIME"] = str(release)
    else:
        r["RELEASE_TIME"] = str(press)
    return r


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


def test_regression_mistype_in_middle_does_not_splice_a_bigram():
    """A mistyped key must break the n-gram window, not be collapsed out so the surrounding
    correct keys splice into a bogus 'adjacent' bigram with an inflated duration.

    Expected 'ab', typed 'a z b' where z is wrong and there is a long pause before the
    correction. The old code emitted 'ab' with a duration spanning the deleted z.
    """
    cmap = build_char_map("qwerty")
    records = [
        {**rec("a", 1000, 1050), "SENTENCE": "ab"},
        {**rec("z", 1100, 1150), "SENTENCE": "ab"},  # wrong key
        {**rec("b", 5100, 5150), "SENTENCE": "ab"},  # 4s later, after correcting
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    # 'a' and 'b' were NOT typed consecutively (z between them) -> no 'ab' bigram.
    assert [o.ngram for o in occ] == []


def test_extract_keeps_bigrams_within_clean_runs_around_a_mistype():
    """A mistype breaks the window at that point, but clean runs on either side still
    yield their bigrams (with genuine adjacent durations)."""
    cmap = build_char_map("qwerty")
    # expected 'that', typed 't h x a t' -> x wrong. Clean runs: 'th' (before x), 'at' (after).
    records = [
        {**rec("t", 1000, 1050), "SENTENCE": "that"},
        {**rec("h", 1100, 1150), "SENTENCE": "that"},
        {**rec("x", 1200, 1250), "SENTENCE": "that"},  # wrong
        {**rec("a", 3000, 3050), "SENTENCE": "that"},
        {**rec("t", 3100, 3150), "SENTENCE": "that"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    ngrams = [o.ngram for o in occ]
    assert "th" in ngrams  # clean run before the error
    assert "at" in ngrams  # clean run after the error
    assert "ha" not in ngrams  # would splice across the deleted x
    # And the surviving bigrams have honest, short durations (not the 1900ms cross-gap).
    for o in occ:
        assert o.duration < 500


def test_regression_no_splice_across_backspace_on_the_cli_path():
    """A control key (BKSP/SHIFT — multi-char LETTER) between two correct chars must break
    the window even though group_sessions pre-drops such rows.

    The first contiguity fix only caught gaps created by flagged-incorrect single chars; a
    control key was dropped BEFORE extraction, so 'a <BKSP> b' still spliced into
    ('ab', ~4000ms) on the CLI path. Original indices must be assigned before any filtering.
    """
    from keybo.data.keystrokes import group_sessions

    cmap = build_char_map("qwerty")
    rows = [
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "a",
            "PRESS_TIME": "1000",
            "RELEASE_TIME": "1050",
            "SENTENCE": "ab",
        },
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "BKSP",
            "PRESS_TIME": "1200",
            "RELEASE_TIME": "1250",
            "SENTENCE": "ab",
        },
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "b",
            "PRESS_TIME": "5000",
            "RELEASE_TIME": "5050",
            "SENTENCE": "ab",
        },
    ]
    all_occ = []
    for recs in group_sessions(rows).values():
        all_occ += extract_occurrences(recs, cmap, n=2, skip=0, time_mode="full")
    assert [o.ngram for o in all_occ] == []  # 'a' and 'b' were separated by a backspace


def test_regression_no_splice_across_arrow_keys_on_the_cli_path():
    """Same as above for navigation keys (e.g. LEFT), which are also multi-char rows."""
    from keybo.data.keystrokes import group_sessions

    cmap = build_char_map("qwerty")
    rows = [
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "t",
            "PRESS_TIME": "1000",
            "RELEASE_TIME": "1050",
            "SENTENCE": "th",
        },
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "LEFT",
            "PRESS_TIME": "1100",
            "RELEASE_TIME": "1150",
            "SENTENCE": "th",
        },
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "h",
            "PRESS_TIME": "2000",
            "RELEASE_TIME": "2050",
            "SENTENCE": "th",
        },
    ]
    all_occ = []
    for recs in group_sessions(rows).values():
        all_occ += extract_occurrences(recs, cmap, n=2, skip=0, time_mode="full")
    assert [o.ngram for o in all_occ] == []


def test_regression_no_splice_across_empty_letter_rows_on_the_cli_path():
    """Delta-audit SUSPECTED: a row with an empty LETTER field must still occupy a stream
    index (creating a contiguity gap), not vanish in group_sessions and let its neighbours
    splice. Same class as the control-key bypass."""
    from keybo.data.keystrokes import group_sessions

    cmap = build_char_map("qwerty")
    rows = [
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "a",
            "PRESS_TIME": "1000",
            "RELEASE_TIME": "1050",
            "SENTENCE": "ab",
        },
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "",
            "PRESS_TIME": "1200",
            "RELEASE_TIME": "1250",
            "SENTENCE": "ab",
        },
        {
            "TEST_SECTION_ID": "s1",
            "LETTER": "b",
            "PRESS_TIME": "5000",
            "RELEASE_TIME": "5050",
            "SENTENCE": "ab",
        },
    ]
    all_occ = []
    for recs in group_sessions(rows).values():
        all_occ += extract_occurrences(recs, cmap, n=2, skip=0, time_mode="full")
    assert [o.ngram for o in all_occ] == []


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
    occ = extract_occurrences(
        make_session("the"), cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=7
    )
    occ += extract_occurrences(
        make_session("the"), cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=7
    )
    agg = aggregate_occurrences(occ)
    # 'th' seen twice — key is now (layout, positions, ngram).
    key = next(k for k in agg if k[2] == "th")
    assert key[0] == "qwerty"
    assert agg[key]["frequency"] == 2
    assert len(agg[key]["occurrences"]) == 2
    # Each sample is a 4-tuple (wpm, duration, pid, hold).
    sample = agg[key]["occurrences"][0]
    assert len(sample) == 4
    assert sample[2] == 7  # pid


def test_write_ngram_tsv_layout_first_column(tmp_path):
    """The TSV's first field must be the layout name (new schema)."""
    cmap = build_char_map("qwerty")
    occ = extract_occurrences(
        make_session("the"), cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=1
    )
    agg = aggregate_occurrences(occ)
    out = tmp_path / "bistrokes.tsv"
    write_ngram_tsv(agg, str(out))
    text = out.read_text()
    first_line = text.splitlines()[0].split("\t")
    # layout<TAB>(positions)<TAB>ngram<TAB>freq<TAB>samples...
    assert first_line[0] == "qwerty"
    assert first_line[1].startswith("(")  # positions
    assert first_line[2] in ("th", "he")  # ngram
    assert int(first_line[3]) >= 1  # frequency


def test_write_ngram_tsv_samples_are_4_tuples(tmp_path):
    """Each sample in the TSV must be a (wpm, duration, pid, hold) 4-tuple."""
    cmap = build_char_map("qwerty")
    records = [
        {**rec("t", 1000, 1060), "SENTENCE": "th"},
        {**rec("h", 1100, 1160), "SENTENCE": "th"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", layout="dvorak", pid=99)
    agg = aggregate_occurrences(occ)
    out = tmp_path / "test.tsv"
    write_ngram_tsv(agg, str(out))
    line = out.read_text().splitlines()[0].split("\t")
    # field[4] onward are samples
    sample_str = line[4]
    sample = eval(sample_str)  # noqa: S307
    assert len(sample) == 4
    assert sample[2] == 99  # pid
    assert sample[3] == 60  # hold = 1060 - 1000


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
    ngrams = {ngram for _, _, ngram in agg}
    assert ngrams == {"th", "he"}


def test_process_dataset_rejects_unknown_ngram(tmp_path):
    files_dir, meta = _make_dataset(tmp_path)
    with pytest.raises(ValueError):
        process_dataset(files_dir, meta, ngram="quadgram")


# --- real-dump robustness: csv.DictReader None fields (short rows) ---------------------


def _short_row_file(tmp_path):
    """A keystroke file containing a SHORT row (fewer columns than the header).

    csv.DictReader fills the missing fields with None -- not "" -- which is what the real
    136M dump produced on first contact (laptop run, 2026-07-04): TypeError at len(None).
    """
    p = tmp_path / "111_keystrokes.txt"
    p.write_text(
        "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tPRESS_TIME\tRELEASE_TIME\tLETTER\n"
        "111\ts1\tab\t1000\t1050\ta\n"
        "111\ts1\n"  # short row -> SENTENCE/PRESS_TIME/RELEASE_TIME/LETTER all None
        "111\ts1\tab\t1100\t1150\tb\n"
    )
    return str(p)


def test_regression_short_rows_do_not_crash_processing(tmp_path):
    from keybo.data.keystrokes import process_keystroke_file

    cmap = build_char_map("qwerty")
    occ = process_keystroke_file(_short_row_file(tmp_path), cmap, n=2, skip=0, time_mode="full")
    # Must not raise; the None row occupies a stream index, so 'a'..'b' must NOT splice
    # across it either (same contiguity rule as control keys).
    assert [o.ngram for o in occ] == []


def test_regression_none_press_time_on_correct_row_does_not_crash(tmp_path):
    """float(None) raises TypeError, which the old except (ValueError, KeyError) missed."""
    cmap = build_char_map("qwerty")
    records = [
        {"LETTER": "a", "PRESS_TIME": None, "RELEASE_TIME": None, "SENTENCE": "ab"},
        {"LETTER": "b", "PRESS_TIME": "1100", "RELEASE_TIME": "1150", "SENTENCE": "ab"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    assert occ == []  # unusable timing -> no occurrences, but no crash


def test_regression_none_sentence_does_not_crash():
    cmap = build_char_map("qwerty")
    records = [
        {"LETTER": "a", "PRESS_TIME": "1000", "RELEASE_TIME": "1050", "SENTENCE": None},
        {"LETTER": "b", "PRESS_TIME": "1100", "RELEASE_TIME": "1150", "SENTENCE": None},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full")
    assert occ == []  # nothing aligns against an absent sentence, but no crash


def test_regression_short_metadata_rows_do_not_crash(tmp_path):
    """A short metadata row gives FINGERS/AVG_WPM_15/... = None; .strip() crashed."""
    p = tmp_path / "metadata_participants.txt"
    p.write_text(
        "PARTICIPANT_ID\tFINGERS\tAVG_WPM_15\tKEYBOARD_TYPE\tLAYOUT\n"
        "111\t9-10\t90\tfull\tqwerty\n"
        "222\n"  # short row -> all fields None
        "333\t9-10\t85\tlaptop\tqwerty\n"
    )
    md = load_participant_metadata(str(p))
    assert set(md) == {"111", "333"}  # short row skipped, not fatal


# --- schema: layout, pid, hold fields -------------------------------------------------


def test_schema_layout_and_pid_stamped_on_occurrence():
    """extract_occurrences stamps layout and pid on each Occurrence."""
    cmap = build_char_map("qwerty")
    records = make_session("th", start=1000, step=100)
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=42)
    assert len(occ) == 1
    assert occ[0].layout == "qwerty"
    assert occ[0].pid == 42


def test_schema_hold_computed_from_first_key_release_minus_press():
    """hold = RELEASE_TIME - PRESS_TIME of the first key in the window, int ms."""
    cmap = build_char_map("qwerty")
    # first key: press=1000, release=1080 -> hold=80
    records = [
        {**rec("t", 1000, 1080), "SENTENCE": "th"},
        {**rec("h", 1100, 1180), "SENTENCE": "th"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=1)
    assert occ[0].hold == 80


def test_schema_hold_minus1_when_release_missing():
    """hold = -1 when the first key has no RELEASE_TIME."""
    cmap = build_char_map("qwerty")
    records = [
        {"LETTER": "t", "PRESS_TIME": "1000", "SENTENCE": "th"},  # no RELEASE_TIME key
        {**rec("h", 1100, 1180), "SENTENCE": "th"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=1)
    assert occ[0].hold == -1


def test_schema_hold_minus1_when_release_unparseable():
    """hold = -1 when RELEASE_TIME is not a valid number."""
    cmap = build_char_map("qwerty")
    records = [
        {"LETTER": "t", "PRESS_TIME": "1000", "RELEASE_TIME": "bad", "SENTENCE": "th"},
        {**rec("h", 1100, 1180), "SENTENCE": "th"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=1)
    assert occ[0].hold == -1


def test_schema_hold_minus1_when_release_less_than_press():
    """hold = -1 when release < press (data error)."""
    cmap = build_char_map("qwerty")
    records = [
        {**rec("t", 1000, 900), "SENTENCE": "th"},  # release 900 < press 1000
        {**rec("h", 1100, 1180), "SENTENCE": "th"},
    ]
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", layout="qwerty", pid=1)
    assert occ[0].hold == -1


# --- rejection counters -----------------------------------------------------------------


def test_counters_session_total_incremented():
    """Each call to extract_occurrences increments session_total once."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters["session_total"] == 1
    extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters["session_total"] == 2


def test_counters_session_no_single_char_rows():
    """A session with no single-char rows increments session_no_single_char_rows."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    records = [
        {"LETTER": "BKSP", "PRESS_TIME": "1000", "RELEASE_TIME": "1050", "SENTENCE": "a"},
        {"LETTER": "SHIFT", "PRESS_TIME": "1100", "RELEASE_TIME": "1150", "SENTENCE": "a"},
    ]
    extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("session_no_single_char_rows") == 1
    assert counters.get("session_total") == 1


def test_counters_session_no_correct_chars():
    """A session where all typed chars are wrong increments session_no_correct_chars."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    records = [
        {"LETTER": "x", "PRESS_TIME": "1000", "RELEASE_TIME": "1050", "SENTENCE": "ab"},
        {"LETTER": "y", "PRESS_TIME": "1100", "RELEASE_TIME": "1150", "SENTENCE": "ab"},
    ]
    extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("session_no_correct_chars") == 1


def test_counters_session_bad_time():
    """A session with unparseable PRESS_TIME on the first correct key."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    records = [
        {"LETTER": "a", "PRESS_TIME": None, "RELEASE_TIME": "1050", "SENTENCE": "ab"},
        {"LETTER": "b", "PRESS_TIME": "1100", "RELEASE_TIME": "1150", "SENTENCE": "ab"},
    ]
    extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("session_bad_time") == 1


def test_counters_window_non_contiguous():
    """A window rejected for non-contiguity."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    # 'a' then mistype 'z' then 'b' — 'a' and 'b' are not contiguous.
    records = [
        {**rec("a", 1000, 1050), "SENTENCE": "ab"},
        {**rec("z", 1100, 1150), "SENTENCE": "ab"},
        {**rec("b", 1200, 1250), "SENTENCE": "ab"},
    ]
    extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("window_non_contiguous") == 1
    assert counters.get("window_kept", 0) == 0


def test_counters_window_banned_key():
    """window_banned_key counter is wired but unreachable with current BANNED_KEYS (all
    multi-char) since windows are built from single-char correct[] entries. Verify the
    counter doesn't appear in a normal session."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("window_banned_key", 0) == 0


def test_counters_window_multi_char():
    """window_multi_char counter is wired but unreachable since windows are built from
    single-char correct[] entries. Verify the counter doesn't appear in a normal session."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("window_multi_char", 0) == 0


def test_counters_window_off_layout():
    """A window where a letter is not in the char map."""
    cmap = {"a": (0, 0), "b": (1, 0), " ": (0, 0)}  # minimal map, no 'c'
    counters: dict = {}
    # 'a' and 'c' are both correct (match expected), contiguous, single-char, not banned
    # but 'c' is not in cmap -> off_layout
    records = [
        {**rec("a", 1000, 1050), "SENTENCE": "ac"},
        {**rec("c", 1100, 1150), "SENTENCE": "ac"},
    ]
    extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("window_off_layout") == 1
    assert counters.get("window_kept", 0) == 0


def test_counters_window_bad_time():
    """A window where PRESS_TIME is unparseable for the timing computation."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    # 3 chars: first and last have valid times (so session WPM succeeds),
    # but the middle one has bad PRESS_TIME -> window timing fails.
    records = [
        {"LETTER": "t", "PRESS_TIME": "1000", "RELEASE_TIME": "1050", "SENTENCE": "the"},
        {"LETTER": "h", "PRESS_TIME": "bad", "RELEASE_TIME": "1150", "SENTENCE": "the"},
        {"LETTER": "e", "PRESS_TIME": "1200", "RELEASE_TIME": "1250", "SENTENCE": "the"},
    ]
    extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters.get("window_bad_time", 0) >= 1


def test_counters_window_kept_matches_occurrences():
    """window_kept count matches the number of emitted occurrences."""
    cmap = build_char_map("qwerty")
    counters: dict = {}
    records = make_session("the")
    occ = extract_occurrences(records, cmap, n=2, skip=0, time_mode="full", counters=counters)
    assert counters["window_kept"] == len(occ)
    assert counters["window_kept"] == 2  # "th" and "he"


def test_counters_none_when_not_passed():
    """When counters=None (default), no cost — function works as before."""
    cmap = build_char_map("qwerty")
    occ = extract_occurrences(make_session("the"), cmap, n=2, skip=0, time_mode="full")
    assert len(occ) == 2  # still works


def test_regression_double_quote_letter_does_not_corrupt_parsing(tmp_path):
    """A participant typing a double-quote character produces LETTER='\"'. csv's DEFAULT
    dialect treats that as an opening quote and swallows the rest of the line AND the next
    row into one field (verified: LETTER became '\\t222\\n1\\ts1...'), silently corrupting
    every session containing a quote character. Ingest must use QUOTE_NONE.
    (Found by muscle-C measuring the real dump; 14/1.46M rows affected in a 2000-file
    sample under correct parsing — but arbitrarily many under the default dialect.)"""
    from keybo.data.keystrokes import process_keystroke_file

    cmap = build_char_map("qwerty")
    p = tmp_path / "111_keystrokes.txt"
    p.write_text(
        "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tPRESS_TIME\tRELEASE_TIME\tLETTER\tKEYCODE\n"
        '111\ts1\tsay "hi" ok\t1000\t1050\ts\t83\n'
        '111\ts1\tsay "hi" ok\t1100\t1150\ta\t65\n'
        '111\ts1\tsay "hi" ok\t1200\t1250\ty\t89\n'
        '111\ts1\tsay "hi" ok\t1300\t1350\t \t32\n'
        '111\ts1\tsay "hi" ok\t1400\t1450\t"\t222\n'  # the quote char keystroke
        '111\ts1\tsay "hi" ok\t1500\t1550\th\t72\n'
        '111\ts1\tsay "hi" ok\t1600\t1650\ti\t73\n'
    )
    occ = process_keystroke_file(str(p), cmap, n=2, skip=0, time_mode="full")
    ngrams = [o.ngram for o in occ]
    # Under the corrupting default dialect, the '"' row swallowed the 'h' row: 'hi' vanished
    # and phantom multi-line "letters" appeared. With QUOTE_NONE, 'sa'/'ay' survive and 'hi'
    # is intact ('"' itself is not on the 30-key map, so windows touching it drop -- fine).
    assert "sa" in ngrams and "ay" in ngrams
    assert "hi" in ngrams
    assert all(len(g) == 2 for g in ngrams)
