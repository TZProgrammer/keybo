"""lmscissor: is the +0.3628 itself robust?

The badscissor spec ships an explicit caveat: "the 8-layout ordering is fragile under BOTH
weightings — dropping any single finger pair reorders it ... Do NOT use mid-board bad-scissor
differences to pick a winner." This delta IS a mid-board difference between adjacent ranks, so
the caveat applies directly. Two cheap tests:

  1. LEAVE-ONE-FINGER-PAIR-OUT on the delta (the spec's own robustness instrument).
  2. CROSS-CORPUS: does the delta hold on iWeb as well as blend-v1?
  3. And the reverse direction: leave-one-finger-pair-out on the dy=2 (unpriced) advantage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import _DEX, bad_scissor  # noqa: E402
from keybo.data.corpus import load_frequencies, resolve_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}
PAIRS = (
    "index-middle",
    "index-ring",
    "index-pinky",
    "middle-ring",
    "middle-pinky",
    "ring-pinky",
)


def kind(x):
    return G.finger(x).value.split("-")[1]


def scores(bigrams, drop_pair=None):
    """(shipped bad-scissor share, dy2-total share) per layout, optionally dropping a finger pair."""
    out = {}
    for label, spec in LAYOUTS.items():
        lay = Layout(spec, G)
        num_bs = num_dy2 = 0.0
        den = 0
        for bg, freq in bigrams.items():
            if len(bg) != 2 or " " in bg:
                continue
            if not all(lay.has_key(c) for c in bg):
                continue
            den += freq
            a, b = lay.pos(bg[0]), lay.pos(bg[1])
            if not C.same_hand(G, a, b) or C.same_finger(G, a, b):
                continue
            dy = abs(a[1] - b[1])
            if dy == 0:
                continue
            ka, kb = kind(a[0]), kind(b[0])
            pair = "-".join(sorted((ka, kb), key=lambda k: -_DEX[k]))
            if drop_pair is not None and pair == drop_pair:
                continue
            if bad_scissor(G, a, b):
                num_bs += freq
            if dy == 2:
                num_dy2 += freq
        out[label] = (100.0 * num_bs / den, 100.0 * num_dy2 / den)
    return out


for corpus_name in ("blend-v1", "iweb"):
    cdir = resolve_corpus_dir(corpus_name)
    bigrams = load_frequencies(str(cdir / "bigrams.txt"))
    print(f"\n{'='*100}\nCORPUS {corpus_name}  ({cdir})\n{'='*100}")

    full = scores(bigrams)
    d_bs = full["keybo-lsb+lm"][0] - full["keybo-lsb"][0]
    d_dy2 = full["keybo-lsb+lm"][1] - full["keybo-lsb"][1]
    print(
        f"  FULL:  bad-scissor {full['keybo-lsb'][0]:.4f} -> {full['keybo-lsb+lm'][0]:.4f} "
        f"(delta {d_bs:+.4f}, winner {'keybo-lsb' if d_bs>0 else 'keybo-lsb+lm'})"
    )
    print(
        f"         dy2 total   {full['keybo-lsb'][1]:.4f} -> {full['keybo-lsb+lm'][1]:.4f} "
        f"(delta {d_dy2:+.4f}, winner {'keybo-lsb' if d_dy2>0 else 'keybo-lsb+lm'})"
    )

    print(f"\n  LEAVE-ONE-FINGER-PAIR-OUT (the spec's own robustness instrument):")
    print(f"  {'dropped pair':<16}{'bad-sc delta':>14}{'winner':>16}{'  |  dy2 delta':>16}{'winner':>16}")
    flips_bs = flips_dy2 = 0
    for p in PAIRS:
        s = scores(bigrams, drop_pair=p)
        db = s["keybo-lsb+lm"][0] - s["keybo-lsb"][0]
        dd = s["keybo-lsb+lm"][1] - s["keybo-lsb"][1]
        wb = "keybo-lsb" if db > 0 else ("keybo-lsb+lm" if db < 0 else "tie")
        wd = "keybo-lsb" if dd > 0 else ("keybo-lsb+lm" if dd < 0 else "tie")
        if (db > 0) != (d_bs > 0):
            flips_bs += 1
        if (dd > 0) != (d_dy2 > 0):
            flips_dy2 += 1
        print(f"  {p:<16}{db:>+14.4f}{wb:>16}{dd:>+16.4f}{wd:>16}")
    print(f"\n  => bad-scissor verdict sign flips: {flips_bs} of {len(PAIRS)}")
    print(f"  => dy2       verdict sign flips: {flips_dy2} of {len(PAIRS)}")

print("\ndone")
