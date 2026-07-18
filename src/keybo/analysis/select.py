"""SELECT-1: flagship-selection instruments beyond the raw gauges.

The plateau finding (fine speed ranks are estimator-generated at the top) means
"the best layout" cannot be picked by reading the speed column one more time.
These instruments measure what still separates plateau candidates:

* ``switching_costs`` — adoption friction vs qwerty (keys unchanged, zxcv
  shortcut block, same-finger/same-hand retention);
* ``hand_balance_pct`` — corpus letter mass on the left hand;
* :class:`RawSupport` — the share of a layout's corpus mass that lands on
  RAW-OBSERVED position n-grams in the K31 study (serve band, production
  cell convention). High support = the layout's predicted gain rides on
  measurement; low support = it rides on model extrapolation. This
  operationalizes the divergence-RCA finding that fitted headlines can have
  no raw observational support.
* :meth:`~keybo.analysis.timecard.TimeSurface.seed_totals` (timecard) — the
  per-seed estimator spread backing a plateau gate.

Construction registered before any candidate was scored (KAN-PRIME-1 /
SELECT-1 ledger entry); primes live in :mod:`keybo.analysis.community`.
"""

from __future__ import annotations

from pathlib import Path

QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
#: genkey column->finger convention (0..7, thumbs excluded), used for retention
_COL_FINGER = (0, 1, 2, 3, 3, 4, 4, 5, 6, 7)


def switching_costs(lay30: str, ref30: str = QWERTY30M) -> dict[str, int]:
    """Adoption-friction counts vs a reference layout (default qwerty30m)."""
    ref_slot = {ch: i for i, ch in enumerate(ref30)}
    unchanged = sum(a == b for a, b in zip(lay30, ref30, strict=False))
    same_finger = same_hand = 0
    for slot, ch in enumerate(lay30):
        rs = ref_slot[ch]
        if _COL_FINGER[slot % 10] == _COL_FINGER[rs % 10]:
            same_finger += 1
        if (slot % 10 >= 5) == (rs % 10 >= 5):
            same_hand += 1
    zxcv = sum(lay30[ref_slot[ch]] == ch for ch in "zxcv")
    return {
        "unchanged_keys": unchanged,
        "same_finger_keys": same_finger,
        "same_hand_keys": same_hand,
        "zxcv_preserved": zxcv,
    }


def hand_balance_pct(lay30: str, letter_freqs: dict[str, float]) -> float:
    """Percent of letter mass typed by the LEFT hand (columns 0-4)."""
    left = total = 0.0
    for slot, ch in enumerate(lay30):
        f = letter_freqs.get(ch, 0.0)
        total += f
        if slot % 10 < 5:
            left += f
    return 100.0 * left / total if total else float("nan")


def _norm_pos(seq) -> tuple:
    return tuple(tuple(int(v) for v in p) for p in seq)


class RawSupport:
    """Share of corpus mass on raw-observed position n-grams (K31 study).

    ``serve`` sets use the production validation convention (wpm band
    [lo, hi), width w, min ``min_cell`` samples, serve bucket only);
    ``any`` sets require just one raw sample in any wpm cell.
    """

    def __init__(
        self,
        bi_serve: set,
        bi_any: set,
        tri_serve: set,
        tri_any: set,
        positions31: list,
    ):
        self.bi_serve, self.bi_any = bi_serve, bi_any
        self.tri_serve, self.tri_any = tri_serve, tri_any
        self.positions = [tuple(int(v) for v in p) for p in positions31]

    @classmethod
    def from_tsvs(
        cls,
        bi_tsv: str | Path,
        tri_tsv: str | Path,
        geometry=None,
        wpm_lo: int = 40,
        wpm_hi: int = 140,
        w: int = 20,
        min_cell: int = 10,
        serve_bucket: int = 80,
    ) -> RawSupport:
        from keybo.data.strokes import load_strokes
        from keybo.geometry import ROW_STAGGERED_31
        from keybo.training.validate import build_cells

        geometry = geometry or ROW_STAGGERED_31
        sets: dict[str, set] = {}
        for tag, path, n in (("bi", bi_tsv, 2), ("tri", tri_tsv, 3)):
            rows = load_strokes(str(path), ngram_len=n, wpm_threshold=0, min_samples=1)
            cells = build_cells(rows, wpm_lo, wpm_hi, w, 1)
            sets[f"{tag}_any"] = {_norm_pos(c.positions) for c in cells}
            # Cell.frequency is the CORPUS occurrence count; the registered
            # >=min_cell convention counts raw SAMPLES in the bucket (Cell.n).
            sets[f"{tag}_serve"] = {
                _norm_pos(c.positions)
                for c in cells
                if c.bucket == serve_bucket and c.n >= min_cell
            }
        return cls(
            sets["bi_serve"],
            sets["bi_any"],
            sets["tri_serve"],
            sets["tri_any"],
            [*geometry.slots, geometry.space_position],
        )

    def support(
        self,
        lay30: str,
        bigram_freqs: dict[str, float],
        trigram_freqs: dict[str, float],
    ) -> dict[str, float]:
        slot_of = {ch: i for i, ch in enumerate(lay30)}
        slot_of[" "] = 30
        out = {}
        for tag, freqs, serve, anyset in (
            ("bi", bigram_freqs, self.bi_serve, self.bi_any),
            ("tri", trigram_freqs, self.tri_serve, self.tri_any),
        ):
            tot = cov_serve = cov_any = 0.0
            for ng, f in freqs.items():
                try:
                    pos = tuple(self.positions[slot_of[c]] for c in ng)
                except KeyError:
                    continue
                tot += f
                if pos in serve:
                    cov_serve += f
                if pos in anyset:
                    cov_any += f
            out[f"{tag}_serve_pct"] = 100.0 * cov_serve / tot if tot else float("nan")
            out[f"{tag}_any_pct"] = 100.0 * cov_any / tot if tot else float("nan")
        return out
