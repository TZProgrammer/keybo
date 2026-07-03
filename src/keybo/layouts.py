"""Named reference layouts (the 30 main-block keys), for comparison in `keybo score`.

Each string is the 30 characters in canonical slot order (top row left-to-right, then home,
then bottom), matching :data:`keybo.geometry.ROW_STAGGERED_30`.
"""

NAMED_LAYOUTS: dict[str, str] = {
    "qwerty": "qwertyuiopasdfghjkl;zxcvbnm,./",
    "dvorak": "',.pyfgcrlaoeuidhtns;qjkxbmwvz",
    "colemak": "qwfpgjluy;arstdhneiozxcvbkm,./",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
}

BASELINE = "qwerty"
