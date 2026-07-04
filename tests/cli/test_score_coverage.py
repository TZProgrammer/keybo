"""`keybo score` coverage-intersection scoring (fable-audit finding 4).

Named layouts carry different punctuation, so the per-layout `has_key` filter in the
scorer skips a *different* subset of the corpus for each — qwerty scores its ';' bigrams,
graphite its '-' bigrams. The printed "improvement over qwerty" then mixes a coverage
artifact into the geometry effect. The fix restricts the corpus to the n-grams typable on
EVERY compared layout, so all layouts are scored on an identical, comparable subset.
"""

from keybo.cli.__main__ import main
from keybo.cli.score import common_ngrams
from keybo.data.corpus import load_frequencies
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from tests.cli.test_cli import _train_tiny_model


def test_common_ngrams_is_the_intersection_over_layouts():
    """A bigram typable on one layout but not the other is in NEITHER common set.

    qwerty carries ';' (not '-'); graphite carries '-' (not ';'). So of {th, a;, a-}
    only 'th' is typable on both -> the common subset is exactly {'th'}.
    """
    qwerty = Layout(NAMED_LAYOUTS["qwerty"], ROW_STAGGERED_30)
    graphite = Layout(NAMED_LAYOUTS["graphite"], ROW_STAGGERED_30)

    common = common_ngrams({"th": 10, "a;": 5, "a-": 5}, [qwerty, graphite])

    assert set(common) == {"th"}
    assert "a;" not in common  # typable on qwerty only -> excluded
    assert "a-" not in common  # typable on graphite only -> excluded


def test_score_prints_coverage_line_and_scores_only_the_intersection(tmp_path, capsys):
    """End-to-end: comparing qwerty vs graphite on {th, a;, a-} scores only 'th' (1 bigram,
    50% of the 20-unit corpus weight), and announces that common subset."""
    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t10\na;\t5\na-\t5\n")

    rc = main(
        ["score", "--model", model_path, "--bigram-freqs", str(corpus), "--layouts", "graphite"]
    )
    out = capsys.readouterr().out

    assert rc == 0
    # Coverage line exists and reports the true common-subset size + corpus-weight share.
    assert "1-bigram common subset" in out
    assert "50.0% of corpus weight" in out
    # Both compared layouts are still reported.
    assert "qwerty" in out
    assert "graphite" in out


def test_named_layouts_are_all_scored_on_an_identical_bigram_set(corpus_dir):
    """Property (the guard the fable audit asked for): with the real corpus, every named
    layout scores EXACTLY the common subset -- the per-layout scored-bigram sets are equal.

    Non-vacuous: the common subset is a strict subset of the full corpus, so the layouts
    genuinely differ in what they can type; equality only holds because we intersect first.
    """
    freqs = load_frequencies(str(corpus_dir / "bigrams.txt"))
    layouts = {name: Layout(chars, ROW_STAGGERED_30) for name, chars in NAMED_LAYOUTS.items()}

    common = common_ngrams(freqs, layouts.values())

    # What each layout's scorer would actually sum: the common n-grams typable on it.
    scored_per_layout = {
        name: frozenset(ng for ng in common if all(lay.has_key(c) for c in ng))
        for name, lay in layouts.items()
    }
    assert len(set(scored_per_layout.values())) == 1  # all layouts -> one identical set
    assert next(iter(scored_per_layout.values())) == frozenset(common)
    # The bug is real: some corpus bigrams are NOT common to all layouts.
    assert 0 < len(common) < len(freqs)


def test_unknown_layout_name_is_rejected(tmp_path):
    """A typo'd --layouts name must error, not silently vanish from the comparison
    (found in the final self-audit: 'dvorka' produced baseline-only output, rc 0)."""
    import pytest

    from keybo.cli.__main__ import main
    from tests.cli.test_cli import _train_tiny_model

    model_path = _train_tiny_model(tmp_path / "bg.json")
    corpus = tmp_path / "bg.txt"
    corpus.write_text("th\t10\n")
    with pytest.raises(SystemExit):
        main(
            [
                "score",
                "--model",
                str(model_path),
                "--bigram-freqs",
                str(corpus),
                "--layouts",
                "dvorka",
            ]
        )
