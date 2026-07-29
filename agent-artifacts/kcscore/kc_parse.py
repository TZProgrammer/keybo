"""Parse keycraft's 150 published layouts + its ranking into a machine-readable table.

Source, both pulled 2026-07-28/29:

* ranking  — https://rbscholtus.github.io/keycraft/ (one table, 150 rows, sorted by keycraft's
  own ``Score`` descending; ``northwest`` +32.79 at #1 down to ``qwerty`` -225.90 at #150);
* layouts  — https://github.com/rbscholtus/keycraft ``data/layouts/*.klf``, the AUTHORITATIVE
  machine-readable source the site itself renders (BSD-3-Clause). 150 files, 1:1 with the
  ranking's names, verified here.

Why the ``.klf`` files and not the rendered pages: the site draws each board as box-drawing
ASCII whose lattice differs per geometry (``rowstag`` / ``colstag`` / ``anglemod`` / ``ortho``
all stagger differently), so a positional scrape of the art silently mis-columns most boards.
The ``.klf`` is the same data before rendering. Nothing is retyped by hand anywhere in this
module; ``--selftest`` round-trips the parse back against the rendered art for the two board
styles as an independent check that we read the same board the site shows.

**The structural fact this exists to surface**: a keycraft board is NOT keybo's 30-key frame.
Every one of the 150 is 3 rows x 12 columns plus a 6-slot thumb row (``~`` = no key), and the
top-ranked layouts put LETTERS on a thumb (``northwest`` has ``s``; ``north`` has ``s``). So
most cannot be expressed as a keybo 30-char string without dropping keys. This module reports
the full board and an explicit projection verdict; it never silently truncates.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

KLF_DIR = Path("/tmp/kc_klf")
PAGE_DIR = Path("/tmp/kc_pages")
INDEX_JSON = Path("/tmp/kc_index.json")
OUT_JSON = Path("/tmp/kc_layouts.json")

EMPTY = "~"
# keybo's scored charset: a 30-key layout is C30M iff its keys are exactly this set.
C30M = set("qwertyuiopasdfghjkl'zxcvbnm,.-")


def parse_klf(text: str) -> dict:
    """Parse a ``.klf`` into {geometry, rows: 3x12, thumbs: 6}, or raise.

    Format (verified uniform across all 150): optional ``#`` comment lines, then a geometry
    keyword line, then 3 whitespace-separated rows of 12 tokens, then 1 row of 6 thumb tokens.
    ``~`` marks a slot with no key.
    """
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) != 5:
        raise ValueError(f"expected geometry + 4 rows, got {len(lines)} content lines")
    geometry = lines[0].strip()
    rows = [ln.split() for ln in lines[1:4]]
    thumbs = lines[4].split()
    if [len(r) for r in rows] != [12, 12, 12]:
        raise ValueError(f"row widths {[len(r) for r in rows]} != [12,12,12]")
    if len(thumbs) != 6:
        raise ValueError(f"thumb row has {len(thumbs)} slots, expected 6")
    for tok in [t for r in rows for t in r] + thumbs:
        if len(tok) != 1:
            raise ValueError(f"multi-char token {tok!r}")
    return {"geometry": geometry, "rows": rows, "thumbs": thumbs}


def board_charset(parsed: dict) -> str:
    keys = [t for r in parsed["rows"] for t in r if t != EMPTY]
    keys += [t for t in parsed["thumbs"] if t != EMPTY]
    return "".join(sorted(set(keys)))


def project_30(parsed: dict) -> dict:
    """Project a keycraft board onto keybo's 30-char row-major string, stating what is lost.

    keybo's frame is 3 rows x 10 columns (row-major top/home/bottom), no thumb keys. keycraft's
    is 3 x 12 + thumbs. The projection keeps the inner 10 columns of each row (dropping the two
    OUTER columns, index 0 and 11) and drops thumbs. That is faithful ONLY when the dropped
    slots are all empty; otherwise the layout genuinely does not fit keybo's frame and we say
    so instead of scoring a mutilated board.
    """
    rows = parsed["rows"]
    thumbs = parsed["thumbs"]

    dropped_outer = [rows[r][c] for r in range(3) for c in (0, 11) if rows[r][c] != EMPTY]
    thumb_keys = [t for t in thumbs if t != EMPTY]
    # `_` is keycraft's spacebar glyph; a space thumb is not a dropped character.
    thumb_real = [t for t in thumb_keys if t != "_"]
    letters_on_thumb = [t for t in thumb_real if t.isalpha()]

    cells = [rows[r][c] for r in range(3) for c in range(1, 11)]
    n_filled = sum(1 for c in cells if c != EMPTY)
    key30 = "".join(cells)  # keeps '~' visible for any hole
    charset = set(cells) - {EMPTY}

    lost = []
    if dropped_outer:
        lost.append(f"outer columns hold {''.join(sorted(dropped_outer))}")
    if letters_on_thumb:
        lost.append(f"LETTERS on thumb: {''.join(letters_on_thumb)}")
    elif thumb_real:
        lost.append(f"non-space thumb keys: {''.join(thumb_real)}")
    if n_filled != 30:
        lost.append(f"inner 3x10 has {n_filled}/30 keys")

    return {
        "key30": key30,
        "n_filled": n_filled,
        "charset30": "".join(sorted(charset)),
        "is_c30m_perm": n_filled == 30 and charset == C30M,
        "scorable_30": n_filled == 30,
        "dropped_outer": "".join(sorted(dropped_outer)),
        "thumb_row": " ".join(thumbs),
        "thumb_keys_nonspace": "".join(thumb_real),
        "letters_on_thumb": "".join(letters_on_thumb),
        "faithful": not lost,
        "projection": "; ".join(lost)
        if lost
        else "exact 30 keys in keybo's frame; nothing dropped",
    }


def _rendered_keys(name: str) -> set[str] | None:
    """The multiset of key glyphs the SITE draws for this layout, from its ASCII board.

    Position-free on purpose: we only assert that the rendered board shows the same KEYS as the
    ``.klf`` we parsed, which is enough to catch reading the wrong file or a stale page without
    re-deriving each geometry's lattice.
    """
    page = PAGE_DIR / f"{name}.html"
    if not page.exists():
        return None
    text = page.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<pre class="keycraft-view">\n(.*?)</pre>', text, re.S)
    if m is None:
        return None
    lines = html.unescape(m.group(1)).split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Board"))
    end = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Hand"))
    box = set("╭╮╰╯├┤┬┴┼─│ \xa0")
    seen = set()
    for ln in lines[start + 1 : end]:
        for ch in ln:
            if ch not in box:
                seen.add(ch)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="also cross-check every parsed board against the site's rendered art",
    )
    args = ap.parse_args()

    idx = json.loads(INDEX_JSON.read_text())
    hdr = idx["hdr"]
    out: list[dict] = []
    failures: list[tuple[str, str]] = []

    for rec in idx["recs"]:
        name = rec["vals"][1]
        klf = KLF_DIR / f"{name}.klf"
        if not klf.exists():
            failures.append((name, "no .klf"))
            continue
        try:
            parsed = parse_klf(klf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — report, never guess
            failures.append((name, f"klf parse: {exc}"))
            continue

        if args.selftest:
            drawn = _rendered_keys(name)
            if drawn is None:
                failures.append((name, "no rendered page to cross-check"))
                continue
            ours = set(board_charset(parsed)) - {EMPTY}
            # The art draws '_' for the spacebar too; compare the non-space key sets.
            if (ours - {"_"}) != (drawn - {"_"}):
                failures.append((name, f"klf vs rendered key-set mismatch: {sorted(ours ^ drawn)}"))
                continue

        proj = project_30(parsed)
        out.append(
            {
                "rank": int(rec["vals"][0]),
                "name": name,
                "kc_score": float(rec["vals"][3].replace("+", "")),
                "geometry": parsed["geometry"],
                "url": "https://rbscholtus.github.io/keycraft/" + rec["href"],
                "klf_rows": [" ".join(r) for r in parsed["rows"]],
                "board_charset": board_charset(parsed),
                "kc_stats": {hdr[i]: rec["vals"][i] for i in range(len(hdr))},
                **proj,
            }
        )

    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"parsed {len(out)}/150 (selftest={args.selftest}); failures={len(failures)}")
    for n, why in failures:
        print("  FAIL", n, why)
    print(
        f"faithful 30-key projections: {sum(r['faithful'] for r in out)}/{len(out)}; "
        f"exact C30M permutations: {sum(r['is_c30m_perm'] for r in out)}; "
        f"30 keys present at all: {sum(r['scorable_30'] for r in out)}"
    )
    top30 = out[:30]
    print(
        f"TOP 30: faithful={sum(r['faithful'] for r in top30)} "
        f"c30m={sum(r['is_c30m_perm'] for r in top30)} "
        f"letters_on_thumb={sum(bool(r['letters_on_thumb']) for r in top30)}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
