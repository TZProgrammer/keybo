"""`northwest` as a named case: how close is keycraft's #1 to our candidates, really?

The user flagged ``northwest`` as resembling ours. It is keycraft's top-ranked layout (+32.79)
and it is NOT scorable by our model: it is a 36-key board (3x12 grid + ``s`` and ``'`` on the
thumbs) whose inner 3x10 has only 29 keys, so no 30-char keybo string exists for it.

A plain Hamming distance over 30-char strings is therefore undefined. What IS well-defined, and
is what "resembles ours" actually means, is **where each letter sits**. This module compares
layouts by LETTER POSITION on the 3x10 grid (a-z only), reports the agreement, and splits it
into the high-frequency core versus the periphery — because "similar" driven by ``q``/``z``
agreeing is not the same claim as ``nrts``/``aei`` agreeing.

Frequency split is derived from the corpus, not asserted: the core is the top-9 letters by
unigram mass, computed here.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from keybo.data.corpus import production_corpus_dir
from keybo.testkit import assert_module_under

REPO = "/tmp/kcscore"
OURS = {
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "arm-H": "flmpg-,uoysntcdireahkxvwb.'jzq",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}


def letter_slots(key30: str) -> dict[str, int]:
    """{letter -> slot index 0..29} for a 30-char row-major string ('~' = hole)."""
    return {ch: i for i, ch in enumerate(key30) if ch.isalpha()}


def core_letters(n: int = 9) -> list[str]:
    """Top-n letters by unigram mass, derived from the production corpus's bigram table."""
    d = Path(production_corpus_dir())
    mass: Counter[str] = Counter()
    for line in (d / "bigrams.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        try:
            f = int(float(parts[-1]))
        except ValueError:
            continue
        for ch in parts[0]:
            if ch.isalpha():
                mass[ch] += f
    return [c for c, _ in mass.most_common(n)]


def compare(a30: str, b30: str, core: set[str]) -> dict:
    sa, sb = letter_slots(a30), letter_slots(b30)
    shared = sorted(set(sa) & set(sb))
    same = [c for c in shared if sa[c] == sb[c]]
    return {
        "letters_compared": len(shared),
        "same_slot": len(same),
        "same_slot_letters": "".join(sorted(same)),
        "hamming_letters": len(shared) - len(same),
        "core_same": sum(1 for c in same if c in core),
        "core_total": sum(1 for c in shared if c in core),
        "periph_same": sum(1 for c in same if c not in core),
        "periph_total": sum(1 for c in shared if c not in core),
    }


def main() -> int:
    assert_module_under("keybo", REPO)
    parsed = {r["name"]: r for r in json.loads(Path("/tmp/kc_layouts.json").read_text())}
    nw = parsed["northwest"]
    core = core_letters(9)
    print(f"corpus-derived high-frequency core (top 9 letters by unigram mass): {''.join(core)}")
    print()
    print("northwest (keycraft #1, +32.79) — a 36-KEY board, NOT scorable in keybo's 30-key frame:")
    for row in nw["klf_rows"]:
        print(f"    {row}")
    print(f"    thumbs: {nw['thumb_row']}   geometry: {nw['geometry']}")
    print(f"    inner 3x10 projection: {nw['key30']!r}  ({nw['n_filled']}/30 keys)")
    print(f"    {nw['projection']}")
    print()
    print("LETTER-POSITION agreement vs each of our candidates")
    print("(a-z only; 's' is on northwest's THUMB so it is not in the 30-key comparison at all)")
    print()
    hdr = f"{'ours':14s} {'cmp':>4s} {'same':>5s} {'hamm':>5s} {'core':>7s} {'periph':>8s}  same-slot letters"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, s in OURS.items():
        c = compare(nw["key30"], s, set(core))
        rows.append((name, c))
        print(
            f"{name:14s} {c['letters_compared']:4d} {c['same_slot']:5d} {c['hamming_letters']:5d} "
            f"{c['core_same']:3d}/{c['core_total']:<3d} {c['periph_same']:4d}/{c['periph_total']:<3d}  "
            f"{c['same_slot_letters']}"
        )
    best = max(rows, key=lambda kv: kv[1]["same_slot"])
    print()
    print(
        f"closest of ours to northwest by letter position: {best[0]} "
        f"({best[1]['same_slot']}/{best[1]['letters_compared']} letters on the same slot)"
    )
    print()
    # Is the resemblance in the core or the periphery? Compare against the null: if agreement
    # were uniform, core and periphery would agree at the same RATE.
    print("core-vs-periphery agreement RATE (the 'is it the high-frequency core?' question):")
    for name, c in rows:
        cr = c["core_same"] / c["core_total"] if c["core_total"] else 0
        pr = c["periph_same"] / c["periph_total"] if c["periph_total"] else 0
        verdict = "core-driven" if cr > pr else ("periphery-driven" if pr > cr else "even")
        print(f"  {name:14s} core={cr:5.1%}  periphery={pr:5.1%}   -> {verdict}")
    print()
    print("For reference, the same comparison AMONG our own candidates (so the numbers above")
    print("can be read against what 'a related layout' looks like in our own family):")
    ours_names = list(OURS)
    for i, a in enumerate(ours_names):
        for b in ours_names[i + 1 :]:
            if {a, b} <= {"arm-B", "BALL-1", "arm-H"} or {a, b} <= {"keybo-lsb", "keybo-lsb+lm"}:
                c = compare(OURS[a], OURS[b], set(core))
                print(f"  {a:14s} vs {b:14s} same={c['same_slot']:2d}/{c['letters_compared']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
