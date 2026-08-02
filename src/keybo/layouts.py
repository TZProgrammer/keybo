"""Named reference layouts (the 30 main-block keys), for comparison in `keybo score`.

Each string is the 30 characters in canonical slot order (top row left-to-right, then home,
then bottom), matching :data:`keybo.geometry.ROW_STAGGERED_30`.
"""

NAMED_LAYOUTS: dict[str, str] = {
    "qwerty": "qwertyuiopasdfghjkl;zxcvbnm,./",
    "dvorak": "',.pyfgcrlaoeuidhtns;qjkxbmwvz",
    "colemak": "qwfpgjluy;arstdhneiozxcvbkm,./",
    # Colemak-DH (the "curl-DH" mod). Pinned because it circulated as TWO different strings
    # across campaign artifacts under this one name, and the wrong one is 0.867 ms/char off --
    # board-sized, and one copy-paste from being quoted as a result. The other string is not a
    # rival convention, it is MALFORMED: 31 characters with a duplicated `z`. Nothing pinned
    # this layout, so neither copy could be checked against anything.
    "colemak-dh": "qwfpbjluy;arstgmneiozxcdvkh,./",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
}

BASELINE = "qwerty"


def _validate() -> None:
    """Reject a malformed layout string at import time.

    A named layout is a PERMUTATION of its charset: exactly 30 slots, no repeats. The 31-char
    duplicated-`z` colemak-dh variant that circulated in artifacts satisfies neither, and no
    check existed to say so -- it was scored, and produced a plausible number. A malformed
    board must fail loudly at the source rather than silently yield a board-sized error.
    """
    for name, keys in NAMED_LAYOUTS.items():
        if len(keys) != 30:
            raise ValueError(f"layout {name!r} has {len(keys)} keys, expected 30: {keys!r}")
        if len(set(keys)) != 30:
            dupes = sorted({c for c in keys if keys.count(c) > 1})
            raise ValueError(f"layout {name!r} repeats {dupes} -- a layout is a permutation")


_validate()
