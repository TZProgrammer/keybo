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
from keybo.layout import Layout
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.scoring.table_trigram import TableTrigramScorer

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
    """One layout's time report. All times in model-predicted milliseconds."""

    total_ms: float
    ms_per_char: float
    saved_vs_ref_pct: float | None
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

    def triple_ms_table(self) -> np.ndarray:
        """The gauge as ONE ``(n, n, n)`` millisecond table: ``T2[a,b] + Tcond[a,b,c]``.

        This is exactly the per-trigram quantity :meth:`card` accumulates, so it is what a
        SEARCH on this ruler has to optimize. Exposing it — rather than leaving it implicit in
        ``card``'s loop — is what lets the objective and the report share one definition.
        ``card`` is ~50 ms per layout, so every previous attempt to search the reported gauge
        re-implemented this sum by hand, and the plausible-looking re-implementations are
        ~1.5e-2 wrong: they weight the bigram term by ``bigrams.txt`` instead of by this
        corpus's first-two-character marginal, and they use one model seed instead of the mean
        over :data:`_SEEDS`.

        Returned as a fresh array (the broadcast allocates): ``default_surface`` is
        ``lru_cache``d for the process, so handing out a view of ``_T2``/``_Tc`` would let one
        caller's in-place edit change every later gauge number in the same run.
        """
        return self._T2[:, :, None] + self._Tc
    def _slot_of(self, lay30: str) -> dict[str, int]:
        """``char -> slot index``, REFUSING a layout that does not fill the geometry exactly.

        The guard exists because building this map from whatever it was handed fails
        SILENTLY and PLAUSIBLY. A driver passed the literal 6-character string ``"qwerty"``;
        every corpus n-gram touching one of the 24 missing keys hit the ``except KeyError:
        continue`` in the scoring loops and was skipped, so ~95% of corpus mass vanished while
        ``card()`` still returned a well-formed :class:`TimeCard` with a plausible
        ``ms_per_char`` (the denominator is ``covered``, so the per-char RATE barely moves —
        only ``total_ms`` collapses). It surfaced solely as a 19.6x ``total_ms`` discrepancy
        between two runs. ``coverage_pct`` did report it, but nothing forced a caller to look.

        The required size is ``len(self.geometry.slots)`` and NOT the literal 30: this class
        supports ROW_STAGGERED_31, and the K31 tests legitimately pass a 31-character layout.
        Duplicates are rejected for the same reason as short strings — two characters mapping
        to one slot silently drops the shadowed key's mass — and are counted separately in the
        message, since a 30-character string with a typo'd repeat is the likelier authoring
        mistake and it is invisible in the length alone.
        """
        want = len(self.geometry.slots)
        distinct = len(set(lay30))
        if len(lay30) != want or distinct != want:
            duplicates = len(lay30) - distinct
            raise ValueError(
                f"layout must be exactly {want} DISTINCT characters for this geometry, got "
                f"{len(lay30)} characters with {distinct} distinct ({duplicates} duplicate) "
                f"-- layout={lay30!r}. A short or repeating layout does not fail here: unknown "
                f"characters are silently skipped as uncovered corpus mass, so the returned "
                f"time is computed over a fraction of the corpus and still looks plausible."
            )
        slot_of = {ch: i for i, ch in enumerate(lay30)}
        slot_of[" "] = self._n - 1
        return slot_of

    def seed_totals(self, lay30: str) -> list[float]:
        """Per-seed corpus totals (ms) — the estimator spread behind ``card().total_ms``
        (which uses the seed-MEAN tables). Requires ``keep_seed_tables=True``."""
        if self._T2s is None:
            raise ValueError("TimeSurface built without keep_seed_tables=True")
        slot_of = self._slot_of(lay30)
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

    def card(self, lay30: str, ref_total_ms: float | None = None) -> TimeCard:
        """One layout's time report. ``lay30`` must fill the geometry exactly — see
        :meth:`_slot_of` for why a short or repeating layout is refused rather than scored."""
        slot_of = self._slot_of(lay30)
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
        saved = None
        if ref_total_ms is not None and ref_total_ms > 0:
            saved = 100.0 * (ref_total_ms - total) / ref_total_ms
        return TimeCard(
            total_ms=total,
            ms_per_char=total / chars,
            saved_vs_ref_pct=saved,
            coverage_pct=100.0 * covered / max(self.total_mass, 1),
            per_key_ms=per_key,
            per_finger_ms=per_finger,
            top_bigrams=sorted(big.items(), key=lambda kv: -kv[1])[:12],
        )


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


class GaugeTrigramScorer(TableTrigramScorer):
    """The reported ms/char gauge as a search objective: table-speed, parity-gated.

    A thin subclass of the shipped trigram-table evaluator that adds the gauge's own
    normalization. ``fitness`` is the corpus TOTAL (what the optimizer minimizes) and
    :meth:`ms_per_char` divides by the COVERED mass, which is the denominator
    :meth:`TimeCard.ms_per_char` uses — dividing by the corpus total instead would scale every
    number by ~1.13 (coverage is 88.7% on C30M) while preserving ranking, so no comparison
    test would catch it.
    """

    _covered: float

    def ms_per_char(self, layout: Layout) -> float:
        """Total ms over the corpus mass this board actually covers — ``analyze``'s ms/char."""
        return self.fitness(layout) / self._covered

    def parity_rel_dev(self, layout: Layout) -> float:
        """Relative deviation of this objective from ``analyze``'s own number for ``layout``.

        The runtime half of the parity gate: it re-derives the total through the ~50 ms
        :meth:`TimeSurface.card` path and compares. Cheap enough to run once before a search
        and worth recording in a result file, because "this objective IS the published gauge"
        is the claim the whole opt-in path rests on — and the naive reading of it is 1.5e-2
        off while looking entirely plausible.
        """
        mine = self.fitness(layout)
        theirs = self._surface.card("".join(layout.chars)).total_ms
        return abs(mine - theirs) / abs(theirs)


def gauge_search_scorer(
    chars: str,
    target_wpm: float = 90.0,
    corpus: str | None = None,
) -> GaugeTrigramScorer:
    """The REPORTED gauge (``analyze``'s ms/char) as a fast, searchable objective.

    ``optimize``'s default objective is the bigram table, but every published number is this
    gauge; the two rank layouts INVERTED (spearman 0.672, selection tax 4.97 resolution floors
    at the argmin) because the cubic term carries most of the gauge's variance (sd 0.803 vs
    0.274). Searching the ruler the report grades on is therefore not a tuning knob — it is
    what makes a search result and its published score the same measurement.

    Built by handing :meth:`TimeSurface.triple_ms_table` to the reviewed
    :class:`~keybo.scoring.table_trigram.TableTrigramScorer` evaluator, so it reconciles to
    :meth:`TimeSurface.card` structurally rather than coincidentally (measured worst relative
    deviation 1.2e-14 over six boards) and evaluates in ~0.2 ms instead of ~50 ms.

    ``chars`` fixes the assignable charset, exactly as the bigram table's does: the kept corpus
    rows depend on it, so a board off this charset is refused rather than silently scored
    against different corpus mass.
    """
    surface = default_surface(target_wpm, corpus)
    scorer = GaugeTrigramScorer.from_table(
        surface.triple_ms_table(), surface.tri, chars=chars, geometry=surface.geometry
    )
    scorer._covered = float(scorer._f.sum())
    scorer._surface = surface
    return scorer
