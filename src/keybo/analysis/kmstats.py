"""Native keymeow-class layout statistics on a shared corpus (KAN-1, rule b330ab4).

Reimplements the keymeow ``base`` metric set (km_metrics_src/metrics/base.py +
keycat ``LayoutTotals::percentage``) for our 30-slot ANSI board, so the analyzer
can report the community's trigram/bigram statistics on the SAME corpus as every
other gauge — the cross-tool corpus artifact the campaign kept tripping over.

Semantics pinned against kmrun (the keymeow scoring harness) by golden tests:
* board: ANSI, row staggers (0, 0.25, 0.75), x = stagger + col, y = row;
  fingers by column (col 4 -> LI, col 5 -> RI, else Finger(col-major)).
* percentage denominators are LAYOUT-RESTRICTED totals: only n-grams fully on
  the 30 keys count (keycat ``totals``), so layouts with different charsets
  divide by different masses — exactly keymeow's convention.
* skipgram table = our 1-skip corpus file (trigram-marginalized a_c counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

#: column -> keymeow Finger enum value (LP=0 LR LM LI, LT/RT unused, RI=6 RM RR RP=9)
_COL_FINGER = [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]
_STAGGER = (0.0, 0.25, 0.75)


@dataclass(frozen=True)
class _Key:
    finger: int
    x: float
    row: int

    @property
    def hand(self) -> int:
        return 0 if self.finger <= 4 else 1

    @property
    def kind(self) -> int:  # FingerKind: pinky 0, ring 1, middle 2, index 3
        return self.finger if self.finger <= 4 else 9 - self.finger


_KEYS = [
    _Key(_COL_FINGER[col], _STAGGER[row] + col, row) for row in range(3) for col in range(10)
]  # slot order = our row-major layout strings


def _direction(a: _Key, b: _Key) -> int:
    """keymeow bistroke direction: 1 inward, -1 outward, 0 none."""
    if a.finger == b.finger or a.hand != b.hand:
        return 0
    return 1 if a.kind < b.kind else -1


def _distance(a: _Key, b: _Key) -> float:
    return sqrt((a.x - b.x) ** 2 + float(a.row - b.row) ** 2)


def _is_lsb(a: _Key, b: _Key) -> bool:
    return a.hand == b.hand and abs(a.kind - b.kind) == 1 and abs(a.x - b.x) >= 2


def _is_roll(a: _Key, b: _Key, c: _Key) -> bool:
    return a.hand != c.hand and a.finger != b.finger and b.finger != c.finger


def _is_redirect(a: _Key, b: _Key, c: _Key) -> bool:
    return (
        a.hand == b.hand
        and b.hand == c.hand
        and a.finger != b.finger
        and b.finger != c.finger
        and _direction(a, b) != _direction(b, c)
    )


#: metric short-name -> (ngram kind, per-ngram value fn(keys) or None for "not matched")
_BIGRAM_METRICS = {
    "sfr": lambda a, b: 1.0 if a is b else 0.0,
    "sfb": lambda a, b: 1.0 if (a is not b and a.finger == b.finger) else 0.0,
    "sfb-dist": lambda a, b: _distance(a, b) if (a is not b and a.finger == b.finger) else 0.0,
    "lsb": lambda a, b: 1.0 if _is_lsb(a, b) else 0.0,
    "lsb-dist": lambda a, b: abs(a.x - b.x) if _is_lsb(a, b) else 0.0,
}
_SKIPGRAM_METRICS = {
    "sfs": lambda a, b: 1.0 if (a is not b and a.finger == b.finger) else 0.0,
    "sfs-dist": lambda a, b: _distance(a, b) if (a is not b and a.finger == b.finger) else 0.0,
}


def _trigram_value(short: str, a: _Key, b: _Key, c: _Key) -> float:
    if short == "alt":
        return 1.0 if (a.hand != b.hand and a.hand == c.hand) else 0.0
    if short == "roll":
        return 1.0 if _is_roll(a, b, c) else 0.0
    if short == "sr-roll":
        return 1.0 if (_is_roll(a, b, c) and a.row == b.row and b.row == c.row) else 0.0
    if short == "redir":
        return 1.0 if _is_redirect(a, b, c) else 0.0
    raise KeyError(short)


_TRIGRAM_METRICS = ("alt", "roll", "sr-roll", "redir")

STAT_NAMES = (
    "sfr",
    "sfb",
    "sfs",
    "sfb-dist",
    "sfs-dist",
    "lsb",
    "lsb-dist",
    "alt",
    "roll",
    "sr-roll",
    "redir",
)


class KmStats:
    """keymeow-class statistics for 30-char layouts over one corpus.

    ``bigrams`` / ``skipgrams`` / ``trigrams`` are char-ngram -> count mappings
    (our corpus files). Scores are percentages of the layout-restricted mass per
    n-gram kind, matching keycat's ``percentage``.
    """

    def __init__(
        self, bigrams: dict[str, int], skipgrams: dict[str, int], trigrams: dict[str, int]
    ):
        self.bi = {k: v for k, v in bigrams.items() if len(k) == 2}
        self.sk = {k: v for k, v in skipgrams.items() if len(k) == 2}
        self.tri = {k: v for k, v in trigrams.items() if len(k) == 3}

    def stats(self, lay30: str) -> dict[str, float]:
        key_of = {ch: _KEYS[i] for i, ch in enumerate(lay30)}
        sums = dict.fromkeys(STAT_NAMES, 0.0)
        bi_total = 0
        for ng, f in self.bi.items():
            a, b = key_of.get(ng[0]), key_of.get(ng[1])
            if a is None or b is None:
                continue
            bi_total += f
            for short, fn in _BIGRAM_METRICS.items():
                v = fn(a, b)
                if v:
                    sums[short] += v * f
        sk_total = 0
        for ng, f in self.sk.items():
            a, b = key_of.get(ng[0]), key_of.get(ng[1])
            if a is None or b is None:
                continue
            sk_total += f
            for short, fn in _SKIPGRAM_METRICS.items():
                v = fn(a, b)
                if v:
                    sums[short] += v * f
        tri_total = 0
        for ng, f in self.tri.items():
            a, b, c = key_of.get(ng[0]), key_of.get(ng[1]), key_of.get(ng[2])
            if a is None or b is None or c is None:
                continue
            tri_total += f
            for short in _TRIGRAM_METRICS:
                v = _trigram_value(short, a, b, c)
                if v:
                    sums[short] += v * f
        out = {}
        for short in STAT_NAMES:
            if short in _BIGRAM_METRICS:
                denom = bi_total
            elif short in _SKIPGRAM_METRICS:
                denom = sk_total
            else:
                denom = tri_total
            out[short] = 100.0 * sums[short] / denom if denom else 0.0
        return out
