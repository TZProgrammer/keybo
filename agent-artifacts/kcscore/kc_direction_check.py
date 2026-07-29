"""Independent derivation: is keybo's gauge frame able to SEE stroke direction at all?

Why this exists here, in the scoring arm: keycraft's objective is direction-sensitive by
construction — its only positive weight is ``FLW`` (+8.00) and it reports ``2RL-IN`` vs
``2RL-OUT`` and ``3RL-IN`` vs ``3RL-OUT`` separately. So "do keycraft's layouts reach their
speed the same way ours do?" cannot be answered honestly without knowing whether OUR gauges
can even measure that axis.

The parent asserted the answer (kmstats direction-blind; shipped ``oxey`` inroll/outroll also
blind) and then explicitly told all arms to re-derive its constants rather than trust them.
This is that derivation, from scratch.

**Instrument: CORPUS REVERSAL** — reverse every ngram (``"the" -> "eht"``), hold the layout
fixed. A metric that distinguishes ``a->b`` from ``b->a`` MUST move. A left-right MIRROR is the
wrong instrument: it maps the finger ordering onto itself and cannot move a direction metric by
construction, so it yields a false "blind" verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.data.corpus import production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.oxey import OxeyStyleScorer
from keybo.testkit import assert_module_under

REPO = "/tmp/kcscore"


def read_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        gram, cnt = parts[0], parts[-1]
        try:
            out[gram] = out.get(gram, 0) + int(float(cnt))
        except ValueError:
            continue
    return out


def reverse(counts: dict[str, int]) -> dict[str, int]:
    """Reverse every ngram, summing collisions (a palindrome maps to itself)."""
    out: dict[str, int] = {}
    for gram, cnt in counts.items():
        out[gram[::-1]] = out.get(gram[::-1], 0) + cnt
    return out


def main() -> int:
    assert_module_under("keybo", REPO)
    d = Path(production_corpus_dir())
    big = read_counts(d / "bigrams.txt")
    skip = read_counts(d / "1-skip31.txt")
    tri = read_counts(d / "trigrams.txt")
    print(f"corpus {d.name}: {len(big)} bigrams, {len(skip)} skipgrams, {len(tri)} trigrams")

    lay = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"  # keybo-lsb

    fwd = KmStats(big, skip, tri).stats(lay)
    rev = KmStats(reverse(big), reverse(skip), reverse(tri)).stats(lay)
    print("\n== 11 kmstats gauges under CORPUS REVERSAL (layout fixed) ==")
    moved = 0
    for name in STAT_NAMES:
        a, b = fwd[name], rev[name]
        delta = abs(a - b)
        if delta > 0:
            moved += 1
        print(f"  {name:10s} {a:12.6f} -> {b:12.6f}   delta={delta:.2e}")
    print(f"  => {moved}/{len(STAT_NAMES)} moved")

    print("\n== shipped oxey inroll/outroll under CORPUS REVERSAL ==")
    try:
        sf = OxeyStyleScorer(big, skip, tri)
        sr = OxeyStyleScorer(reverse(big), reverse(skip), reverse(tri))
        cf = sf.pattern_shares(Layout(lay, ROW_STAGGERED_30))
        cr = sr.pattern_shares(Layout(lay, ROW_STAGGERED_30))
        keys = sorted(set(cf) | set(cr))
        omoved = 0
        for k in keys:
            a, b = cf.get(k), cr.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                delta = abs(a - b)
                if delta > 0:
                    omoved += 1
                flag = "  <-- MOVES" if delta > 0 else ""
                print(f"  {k:16s} {a:12.6f} -> {b:12.6f}   delta={delta:.2e}{flag}")
        print(f"  => {omoved}/{len(keys)} components moved")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not decompose OxeyStyleScorer: {type(exc).__name__}: {exc}")

    print("\n== is_inwards: does it read stroke ORDER or geometry? ==")
    src = Path(REPO) / "src/keybo/scoring/oxey.py"
    text = src.read_text()
    for i, line in enumerate(text.splitlines(), 1):
        if "def is_inwards" in line or "inwards" in line and "def " in line:
            print(f"  {src.name}:{i}: {line.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
