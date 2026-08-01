"""The gauge-as-objective path IS the reported gauge — the property every number here rests on.

`TimeSurface.card` is the published ruler but costs ~50 ms per layout, so any search or
board-comparison over it has to go through the table form. The failure mode that motivates these
tests is not a crash: a plausible hand-rolled version of this sum is ~1.5e-2 off (≈11 resolution
floors on the 0.135 ms/char seed floor) and still ranks boards in nearly the right order, so it
passes every comparison test anyone would think to write. Two agent drivers re-derived it wrong.

So the tests pin the two independent ways it can be wrong:

* STRUCTURE — the table path and `card`'s own loop must agree to float64 noise on real boards
  (`test_table_matches_card_*`). Catches a mis-built table, a wrong seed set, a wrong corpus
  weighting.
* NORMALIZATION — `ms_per_char` must divide by the COVERED mass, which is `card`'s denominator.
  Dividing by the corpus total instead scales every number by ~1.13 while preserving ranking, so
  no ordering test would catch it (`test_ms_per_char_uses_covered_mass`).

The seed-table accessor gets its own test because the seed FLOOR the campaign quotes is an
estimator spread over those three tables: a per-seed check is only evidence about the floor if
each per-seed ruler is the same objective on a different table, not a differently-built one.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis import surfaces as SF
from keybo.analysis.timecard import (
    TimeSurface,
    default_surface,
    gauge_scorer_from_surface,
    gauge_search_scorer,
)
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.table_trigram import TableTrigramScorer

#: Two boards with PUBLISHED gauge figures (PREREGISTRATIONS.md:9423/9426, the ARMG-1 board
#: comparison). Reconciling to a published number is a strictly stronger check than internal
#: consistency: it is the only one that can fail when the code is self-consistent but is not
#: the ruler the campaign reported on.
ARM_B = "flmpg-yuo,sntdcireahkxbwv'.jzq"
BALL_1 = "flmpg-yuo,sntcdireahkxbwv'.jzq"
PUBLISHED = {ARM_B: 253.900579, BALL_1: 253.966426}


@pytest.fixture(scope="module")
def scorer():
    return gauge_search_scorer(chars=SF.C30M)


@pytest.mark.parametrize("lay30", [ARM_B, BALL_1, SF.C30M])
def test_table_matches_card_structurally(scorer, lay30):
    """The fast table objective reproduces `card`'s own total to float64 noise."""
    assert scorer.parity_rel_dev(Layout(lay30, ROW_STAGGERED_30)) < 1e-12


@pytest.mark.parametrize(("lay30", "published"), sorted(PUBLISHED.items()))
def test_reproduces_published_ms_per_char(scorer, lay30, published):
    """The objective returns the LEDGER's number, to the precision the ledger printed it at."""
    assert scorer.ms_per_char(Layout(lay30, ROW_STAGGERED_30)) == pytest.approx(published, abs=1e-5)


def test_ms_per_char_uses_covered_mass(scorer):
    """The denominator is the covered mass, not the corpus total.

    Pinned separately from the parity test because getting this wrong is invisible to any
    ordering comparison: it multiplies every board by the same ~1.13 (coverage is 88.7% on
    C30M), so every ranking, every gap RATIO and every flip verdict is unchanged while every
    absolute number — the ones quoted against a 0.135 ms/char floor — is wrong.
    """
    layout = Layout(ARM_B, ROW_STAGGERED_30)
    surface = default_surface(90.0, None)
    card = surface.card(ARM_B)
    assert scorer.ms_per_char(layout) == pytest.approx(card.ms_per_char, rel=1e-12)
    # and it is genuinely a DIFFERENT number from the corpus-total normalization
    assert scorer.ms_per_char(layout) != pytest.approx(
        scorer.fitness(layout) / surface.total_mass, rel=1e-6
    )


def test_from_table_refuses_wrong_shape():
    """A table of the wrong shape is refused, not silently broadcast into a different objective."""
    with pytest.raises(ValueError, match="table must be"):
        TableTrigramScorer.from_table(np.zeros((3, 3, 3)), {"the": 1}, chars=SF.C30M)


def test_from_table_refuses_charset_geometry_mismatch():
    """A charset that does not fill the geometry is refused at construction."""
    n = len(ROW_STAGGERED_30.slots) + 1
    with pytest.raises(ValueError, match="charset has"):
        TableTrigramScorer.from_table(np.zeros((n, n, n)), {"the": 1}, chars="abc")


def test_triple_ms_table_is_a_fresh_array():
    """Each call allocates: `default_surface` is process-cached, so a view would let one
    caller's in-place edit change every later gauge number in the same run."""
    surface = default_surface(90.0, None)
    first = surface.triple_ms_table()
    first[0, 0, 0] += 1234.0
    assert surface.triple_ms_table()[0, 0, 0] != pytest.approx(first[0, 0, 0])


def test_seed_tables_are_three_rulers_built_the_same_way():
    """Per-seed rulers exist, differ from each other, and average to the seed-mean ruler.

    The averaging assertion is the load-bearing one: it is what makes a per-seed disagreement
    evidence about the seed floor rather than evidence that the per-seed path was built
    differently from the path the headline numbers come from.
    """
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    surface = TimeSurface(tri, target_wpm=90.0, keep_seed_tables=True)
    tables = surface.seed_tables()
    assert len(tables) == 3
    assert not np.allclose(tables[0], tables[1])
    np.testing.assert_allclose(np.mean(tables, axis=0), surface.triple_ms_table(), rtol=1e-12)

    # and each per-seed scorer is the same objective on a different table
    layout = Layout(ARM_B, ROW_STAGGERED_30)
    per_seed = [
        gauge_scorer_from_surface(surface, SF.C30M, table=t).ms_per_char(layout) for t in tables
    ]
    mean_ruler = gauge_scorer_from_surface(surface, SF.C30M).ms_per_char(layout)
    assert np.mean(per_seed) == pytest.approx(mean_ruler, rel=1e-12)


def test_seed_tables_requires_keep_seed_tables():
    """The accessor refuses rather than fabricating a spread from the mean table alone."""
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    with pytest.raises(ValueError, match="keep_seed_tables"):
        TimeSurface(tri, target_wpm=90.0).seed_tables()


def test_refuses_off_charset_board(scorer):
    """A board off the table's charset is refused: it covers different corpus rows, so its
    number would be a different denominator's mean printed in the same column."""
    with pytest.raises(ValueError, match="charset"):
        scorer.ms_per_char(Layout("qwertyuiopasdfghjkl;zxcvbnm,./", ROW_STAGGERED_30))
