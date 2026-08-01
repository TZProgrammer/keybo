"""Predicted-typing-time card: the analyzer's primary gauge (KAN-1, rule b330ab4).

Evaluates a layout on the measured-keystroke time surface — the K31-trained
production models (bigram REG-LOLO + conditioned trigram CAND4, seed-averaged)
at a target WPM — and attributes the total to keys, fingers and bigrams so a
reader can see WHERE a layout spends its time, not just the total.

The surface predicts time for TRIGRAMS as ``T2[a,b] + Tcond[a,b,c]`` (the
bigram table plus the conditioned trigram increment), summed over the corpus.
This is byte-identical to the P16/P17 campaign objective (gate G4 pins it to
runs/p17_coopt.json). Corpus n-grams containing characters off a layout are
skipped, and the coverage share is reported — a layout whose charset misses
corpus mass is flagged rather than silently flattered.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30, Finger
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer

_MODELS = Path(__file__).resolve().parents[3] / "data" / "models" / "k31"
_SEEDS = (0, 1, 2)


def _load_gz_model(stem: str) -> XGBoostTypingModel:
    """Inflate a vendored model + sidecar into a temp dir and load it."""
    with tempfile.TemporaryDirectory() as td:
        for suffix in (".json", ".meta.json"):
            with (
                gzip.open(_MODELS / f"{stem}{suffix}.gz", "rb") as src,
                open(Path(td) / f"{stem}{suffix}", "wb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        return XGBoostTypingModel.load(str(Path(td) / f"{stem}.json"))


@dataclass
class TimeCard:
    """One layout's time report. All times in model-predicted milliseconds.

    Two comparisons against the reference are available and they are **not**
    interchangeable — mixing them is the defect this docstring exists to prevent:

    * ``saved_vs_ref_pct`` — the rankable one. Compares ``ms_per_char``, so both layouts
      are measured per CHARACTER TYPED and a charset difference cannot flatter either.
    * ``raw_total_saved_vs_ref_pct`` — raw corpus TOTALS, retained only to reconcile
      frozen artifacts. Valid **only** between layouts of equal coverage: a layout that
      can type more of the corpus accumulates a larger total for that reason alone, so
      this number charges it for the extra coverage and can report a layout as slower
      while it is in fact faster per character.

    ``coverage_pct`` is the disclosure that says whether the two can differ at all: they
    are algebraically the same number when coverage is equal.
    """

    total_ms: float
    ms_per_char: float
    saved_vs_ref_pct: float | None
    raw_total_saved_vs_ref_pct: float | None
    coverage_pct: float
    per_key_ms: dict[str, float]  # char -> summed time of ngrams ENDING on it
    per_finger_ms: dict[str, float]
    top_bigrams: list[tuple[str, float]]  # (bigram, ms) costliest first


class TimeSurface:
    """The K31 production time surface over one trigram corpus."""

    def __init__(
        self,
        trigram_freqs: dict[str, int],
        target_wpm: float = 90.0,
        geometry=ROW_STAGGERED_30,
        keep_seed_tables: bool = False,
    ):
        self.geometry = geometry
        bi_models = [_load_gz_model(f"bigram_reg31_seed{s}") for s in _SEEDS]
        tri_models = [_load_gz_model(f"trigram_cond31_seed{s}") for s in _SEEDS]
        from keybo.models.base import reject_calibrated_trigram_model

        for m in tri_models:
            reject_calibrated_trigram_model(m, "TimeSurface")
        positions = [*geometry.slots, geometry.space_position]
        self._n = len(positions)
        # The charset is a placeholder: this path reads only the position table and
        # supplies no corpus rows. Keep its size aligned for both K30 and K31 geometry.
        placeholder = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(geometry.slots)]
        T2s = [
            TableBigramScorer(m, {}, target_wpm=target_wpm, chars=placeholder, geometry=geometry)._T
            for m in bi_models
        ]
        self._T2 = np.mean(T2s, axis=0)
        vecs = np.vstack(
            [
                trigram_features_from_positions(geometry, (a, b, c), wpm=target_wpm)
                for a in positions
                for b in positions
                for c in positions
            ]
        )
        Tcs = [m.predict_ms(vecs).reshape(self._n, self._n, self._n) for m in tri_models]
        self._Tc = np.mean(Tcs, axis=0)
        # per-seed tables back the SELECT-1 estimator-stability instrument
        self._T2s, self._Tcs = (T2s, Tcs) if keep_seed_tables else (None, None)
        self.tri = {k: v for k, v in trigram_freqs.items() if len(k) == 3}
        self.total_mass = sum(self.tri.values())

    def seed_totals(self, lay30: str) -> list[float]:
        """Per-seed corpus totals (ms) — the estimator spread behind ``card().total_ms``
        (which uses the seed-MEAN tables). Requires ``keep_seed_tables=True``."""
        if self._T2s is None:
            raise ValueError("TimeSurface built without keep_seed_tables=True")
        slot_of = {ch: i for i, ch in enumerate(lay30)}
        slot_of[" "] = self._n - 1
        totals = []
        for T2, Tc in zip(self._T2s, self._Tcs, strict=False):
            total = 0.0
            for ng, f in self.tri.items():
                try:
                    a, b, c = slot_of[ng[0]], slot_of[ng[1]], slot_of[ng[2]]
                except KeyError:
                    continue
                total += (T2[a, b] + Tc[a, b, c]) * f
            totals.append(float(total))
        return totals

    def card(
        self,
        lay30: str,
        ref_total_ms: float | None = None,
        ref_ms_per_char: float | None = None,
    ) -> TimeCard:
        """Score one layout. ``ref_*`` supply the reference for the saved-percentages.

        ``ref_ms_per_char`` drives the rankable ``saved_vs_ref_pct``; ``ref_total_ms``
        drives the frozen-artifact ``raw_total_saved_vs_ref_pct``.

        Passing ``ref_total_ms`` alone yields both, and is exact: the reference's
        per-character rate is recovered as ``ref_total_ms / covered`` — correct whenever
        the reference covers the same mass as this layout, which is the only case where
        a bare total is a well-defined reference at all. When coverage differs, the two
        saved-percentages are *supposed* to differ, so the caller must say which
        reference rate it means; ``ref_ms_per_char`` is how. Every in-tree caller that
        can face a mixed-charset cohort passes it (:mod:`keybo.cli.analyze`), and
        :meth:`cards` does it for you.
        """
        slot_of = {ch: i for i, ch in enumerate(lay30)}
        slot_of[" "] = self._n - 1
        positions = (*self.geometry.slots, self.geometry.space_position)
        total = 0.0
        covered = 0
        per_key = dict.fromkeys((*lay30, " "), 0.0)
        per_finger = dict.fromkeys((finger.name for finger in Finger), 0.0)
        big: dict[str, float] = {}
        T2, Tc = self._T2, self._Tc
        for ng, f in self.tri.items():
            try:
                a, b, c = slot_of[ng[0]], slot_of[ng[1]], slot_of[ng[2]]
            except KeyError:
                continue
            covered += f
            t2 = T2[a, b] * f
            t3 = Tc[a, b, c] * f
            total += t2 + t3
            # attribute: the a->b transition to key b, the trigram increment to key c
            per_key[ng[1]] += t2
            per_finger[self.geometry.finger(positions[b][0]).name] += t2
            per_key[ng[2]] += t3
            per_finger[self.geometry.finger(positions[c][0]).name] += t3
            big[ng[:2]] = big.get(ng[:2], 0.0) + t2
        chars = max(covered, 1)
        ms_per_char = total / chars
        # The RAW-TOTALS comparison: kept for frozen-artifact reconciliation only. It
        # divides two totals accumulated over DIFFERENT corpus subsets whenever the
        # charsets differ, so it is not comparable across charsets -- see TimeCard.
        raw_saved = None
        if ref_total_ms is not None and ref_total_ms > 0:
            raw_saved = 100.0 * (ref_total_ms - total) / ref_total_ms
        # The RANKABLE comparison: per CHARACTER TYPED, so differing coverage cancels.
        # Falling back to the reference's implied rate over THIS layout's covered mass
        # reproduces the raw-total number exactly at equal coverage, which is what keeps
        # every frozen same-charset artifact byte-identical.
        ref_rate = ref_ms_per_char
        if ref_rate is None and ref_total_ms is not None:
            ref_rate = ref_total_ms / chars
        saved = None
        if ref_rate is not None and ref_rate > 0:
            saved = 100.0 * (ref_rate - ms_per_char) / ref_rate
        return TimeCard(
            total_ms=total,
            ms_per_char=ms_per_char,
            saved_vs_ref_pct=saved,
            raw_total_saved_vs_ref_pct=raw_saved,
            coverage_pct=100.0 * covered / max(self.total_mass, 1),
            per_key_ms=per_key,
            per_finger_ms=per_finger,
            top_bigrams=sorted(big.items(), key=lambda kv: -kv[1])[:12],
        )

    def cards(self, layouts: dict[str, str], ref_lay30: str) -> dict[str, TimeCard]:
        """Score a whole cohort against one reference — the mixed-charset-safe entry point.

        Wires BOTH reference quantities through for you, so the rankable saved-percent is
        the per-character comparison even when the cohort spans charsets. Prefer this over
        calling :meth:`card` per layout and threading the reference by hand: that is how
        the raw-total convention came to be reported as the headline in the first place.
        """
        ref = self.card(ref_lay30)
        return {
            name: self.card(lay30, ref_total_ms=ref.total_ms, ref_ms_per_char=ref.ms_per_char)
            for name, lay30 in layouts.items()
        }


@lru_cache(maxsize=4)
def default_surface(target_wpm: float = 90.0, corpus: str | None = None) -> TimeSurface:
    """The surface over the production trigram corpus (cached — model load is the slow part).

    ``corpus`` is a name or path for :func:`keybo.data.corpus.production_corpus_dir`, and is
    part of the cache key so two corpora in one process cannot serve each other's surface.
    (The fitted timing model underneath is a baked artifact; the corpus only sets the
    frequency weighting over it — swapping corpora does not re-fit anything.)
    """
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    tri = load_frequencies(str(production_corpus_dir(corpus) / "trigrams.txt"))
    return TimeSurface(tri, target_wpm=target_wpm)
