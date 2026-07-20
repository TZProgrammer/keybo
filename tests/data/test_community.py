"""Tests for the community (Kiakl monkeytype) ingest — KIAKL-INGEST + Amendments 3/4.

The load-bearing fact (Amendment 3): the capture's ``key`` field is the QWERTY LABEL of
the physical key pressed (monkeytype layout emulation records the raw browser key), not
the produced character. Decoding must map label -> produced char via the session layout,
and the physical slot is the label's qwerty slot.
"""

from keybo.data.community import (
    QWERTY30,
    SessionRecord,
    _session_wpm,
    decode_event_key,
    dedup_prefix_streams,
    extract_windows,
    main30_from_monkeytype,
)
from keybo.geometry import ROW_STAGGERED_30

COLEMAK_DH = "qwfpbjluy;arstgmneiozxcdvkh,./"


# --- decode_event_key -----------------------------------------------------------------------


def test_qwerty_physical_labels_are_literal_and_complete():
    assert QWERTY30 == "qwertyuiopasdfghjkl;zxcvbnm,./"


def test_decode_identity_on_qwerty():
    for c in QWERTY30:
        assert decode_event_key(c, QWERTY30) == c


def test_decode_maps_label_to_produced_char():
    # colemak-dh: physical qwerty-'k' slot (home row, index 17... let's derive) carries
    # the char at the same slot index in the session layout.
    for label in "ksdfj;":
        idx = QWERTY30.index(label)
        assert decode_event_key(label, COLEMAK_DH) == COLEMAK_DH[idx]


def test_decode_example_from_amendment():
    # qwerty label 'k' is slot 17; colemak-dh has 'e' there (arstgmneio home row).
    assert QWERTY30.index("k") == 17
    assert decode_event_key("k", COLEMAK_DH) == "e"


def test_decode_unshifts_letters_and_punctuation():
    assert decode_event_key("K", COLEMAK_DH) == decode_event_key("k", COLEMAK_DH)
    assert decode_event_key("<", COLEMAK_DH) == decode_event_key(",", COLEMAK_DH)
    assert decode_event_key(">", COLEMAK_DH) == decode_event_key(".", COLEMAK_DH)
    assert decode_event_key("?", COLEMAK_DH) == decode_event_key("/", COLEMAK_DH)
    assert decode_event_key(":", COLEMAK_DH) == decode_event_key(";", COLEMAK_DH)


def test_decode_passes_through_control_keys_and_space():
    assert decode_event_key("Backspace", COLEMAK_DH) == "Backspace"
    assert decode_event_key("Enter", COLEMAK_DH) == "Enter"
    assert decode_event_key(" ", COLEMAK_DH) == " "
    # a char outside the 30-key core (e.g. quote) passes through unchanged
    assert decode_event_key("'", COLEMAK_DH) == "'"


def test_session_wpm_uses_milliseconds_and_all_correct_events():
    events = [("q", 0.0, True)] + [("q", 120.0, True)] * 10
    assert _session_wpm(events) == 110


# --- extract_windows uses PHYSICAL slots ------------------------------------------------------


def _session(events, main30=COLEMAK_DH, wpm=80, sid=1, who="tester"):
    return SessionRecord(sid, who, f"x@rowStagger#{who}", main30, wpm, events)


def test_windows_carry_physical_slots_and_decoded_ngram():
    # Typist on colemak-dh presses physical qwerty-'k' then qwerty-'d': produced 'e','s'.
    events = [("k", 0.0, True), ("d", 150.0, True)]
    cells = extract_windows([_session(events)], {"tester": 200001}, n=2, time_mode="full")
    assert len(cells) == 1
    (label, positions, ngram), samples = next(iter(cells.items()))
    # ngram is the DECODED text
    assert ngram == "es"
    # positions are the PHYSICAL slots of qwerty 'k' and 'd'
    k_slot = ROW_STAGGERED_30.slots[QWERTY30.index("k")]
    d_slot = ROW_STAGGERED_30.slots[QWERTY30.index("d")]
    assert positions == (k_slot, d_slot)
    assert samples == [(80, 150, 200001, -1)]


def test_windows_on_qwerty_are_unchanged_by_the_decode():
    events = [("t", 0.0, True), ("h", 120.0, True)]
    cells = extract_windows(
        [_session(events, main30=QWERTY30)], {"tester": 200001}, n=2, time_mode="full"
    )
    (_, positions, ngram), _ = next(iter(cells.items()))
    assert ngram == "th"
    assert positions == (
        ROW_STAGGERED_30.slots[QWERTY30.index("t")],
        ROW_STAGGERED_30.slots[QWERTY30.index("h")],
    )


def test_shifted_press_is_recovered_not_window_breaking():
    events = [("K", 0.0, True), ("d", 150.0, True)]
    cells = extract_windows([_session(events)], {"tester": 200001}, n=2, time_mode="full")
    assert len(cells) == 1
    (_, _, ngram), _ = next(iter(cells.items()))
    assert ngram == "es"


def test_control_key_still_breaks_windows():
    events = [("k", 0.0, True), ("Backspace", 100.0, True), ("d", 150.0, True)]
    cells = extract_windows([_session(events)], {"tester": 200001}, n=2, time_mode="full")
    assert len(cells) == 0  # Backspace not in cmap -> both windows rejected


# --- prefix-stream dedup (Amendment 4b) -------------------------------------------------------


def test_prefix_duplicate_dropped_keeps_longer():
    ev = [("k", 0.0, True), ("d", 150.0, True), ("f", 130.0, True)]
    short = _session(ev[:2], sid=100)
    long = _session(ev, sid=101)
    kept = dedup_prefix_streams([short, long])
    assert kept == [long]


def test_exact_duplicate_dropped():
    ev = [("k", 0.0, True), ("d", 150.0, True)]
    a = _session(ev, sid=100)
    b = _session(list(ev), sid=101)
    kept = dedup_prefix_streams([a, b])
    assert len(kept) == 1


def test_same_text_different_timings_kept():
    # Legitimate repeat of the same monkeytype test: same keys, different intervals.
    a = _session([("k", 0.0, True), ("d", 150.0, True)], sid=100)
    b = _session([("k", 0.0, True), ("d", 149.0, True)], sid=101)
    kept = dedup_prefix_streams([a, b])
    assert len(kept) == 2


# --- layout-string parsing (regression pins) --------------------------------------------------


def test_main30_slicing_pins():
    s = "`1234567890-=wlypkzxou;[]\\crstbfneia'jvdgqmh,./~REST"
    assert main30_from_monkeytype(s) == "wlypkzxou;crstbfneiajvdgqmh,./"
