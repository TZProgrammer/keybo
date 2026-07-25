"""Tests for the multi-source corpus generator (data/corpus/blend-v1 and its builder).

The generator's job is to make the corpus REPRODUCIBLE, so the properties tested here are
the ones that reproducibility rests on: determinism, the exact production file format, the
declared total, weight arithmetic, charset handling, and a round-trip through the real
``load_frequencies`` the gauges use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from keybo.data.build_corpus import (
    CORPUS_CHARSET,
    DECLARED_TOTAL,
    DEFAULT_WEIGHTS,
    Source,
    apportion,
    blend_tables,
    build_blend,
    count_ngrams,
    load_anchor,
    marginalize_skipgrams,
    normalize,
    repo_latex_source,
    repo_prose_source,
    strip_latex,
    strip_markdown,
    strip_python,
    strip_roff,
    write_build,
    write_table,
)
from keybo.data.corpus import load_frequencies

REPO = Path(__file__).resolve().parents[2]
BLEND = REPO / "data" / "corpus" / "blend-v1"


def _text_source(name: str, register: str, text: str) -> Source:
    """A Source over one in-memory string — lets the blend logic be tested hermetically."""
    return Source(
        name=name,
        register=register,
        extraction=f"literal test text for {name}",
        reader=lambda: iter([(name, text)]),
    )


# --------------------------------------------------------------------------- normalization


def test_normalize_keeps_charset_and_collapses_everything_else():
    assert normalize("Foo(bar)") == "Foo bar "  # out-of-charset -> boundary, not deletion
    assert normalize("a\n\t  b") == "a b"
    assert normalize("Case-Preserved.") == "Case-Preserved."


def test_normalize_output_is_always_within_the_corpus_charset():
    out = normalize("héllo — wörld 123 #!@ Ünïcode tabs\there")
    assert set(out) <= CORPUS_CHARSET


def test_normalize_never_invents_a_bigram_across_a_removed_character():
    # "ob" would be a lie: 'o' and 'b' are not adjacent in the source text.
    bigrams, _ = count_ngrams(normalize("foo(bar"))
    assert "ob" not in bigrams
    assert "o " in bigrams


# --------------------------------------------------------------------------- extraction rules


def test_strip_markdown_drops_code_urls_and_tables():
    out = strip_markdown("prose\n```\ncode_here\n```\n| a | b |\n`inline` https://x.example/y\n")
    assert "code_here" not in out
    assert "inline" not in out
    assert "x.example" not in out
    assert "prose" in out


def test_strip_python_drops_docstrings_and_comments_keeping_code():
    out = strip_python('def f():\n    """A docstring sentence."""\n    total = 1  # a comment\n')
    assert "docstring" not in out
    assert "comment" not in out
    assert "total" in out and "def f" in out


def test_strip_latex_drops_commands_comments_and_math_environments():
    out = strip_latex("Real text \\emph{here} % a comment\n\\begin{equation}x^2\\end{equation}\n")
    assert "Real text" in out
    assert "emph" not in out
    assert "comment" not in out
    assert "x^2" not in out


def test_strip_roff_keeps_prose_and_drops_requests_and_escapes():
    out = strip_roff('.\\" a comment\n.SH NAME\nls \\-list \\fBcontents\\fR\n.TH LS "1" "x"\n')
    assert "NAME" in out and "list" in out and "contents" in out
    assert "comment" not in out
    assert ".SH" not in out and "\\fB" not in out


# --------------------------------------------------------------------------- counting


def test_count_ngrams_is_a_sliding_window_including_space():
    bigrams, trigrams = count_ngrams("ab ab")
    assert bigrams == {"ab": 2, "b ": 1, " a": 1}
    assert trigrams == {"ab ": 1, "b a": 1, " ab": 1}


def test_marginalize_skipgrams_sums_over_the_middle_character():
    assert marginalize_skipgrams({"abc": 3, "axc": 4, "def": 5}) == {"ac": 7, "df": 5}


def test_marginalize_matches_the_committed_production_skipgram_table():
    """``1-skip31.txt`` IS the trigram marginalization — the convention we reproduce."""
    trigrams = load_frequencies(str(REPO / "data" / "corpus" / "trigrams.txt"))
    committed = load_frequencies(str(REPO / "data" / "corpus" / "1-skip31.txt"))
    assert dict(marginalize_skipgrams(trigrams)) == committed


# --------------------------------------------------------------------------- weight arithmetic


def test_apportion_sums_to_exactly_the_requested_total():
    for total in (1, 7, 1000, DECLARED_TOTAL):
        got = apportion({"a": 1.0, "b": 2.0, "c": 0.333}, total)
        assert sum(got.values()) == total


def test_apportion_is_deterministic_and_breaks_ties_on_the_ngram():
    shares = {"aa": 1.0, "bb": 1.0, "cc": 1.0}
    first = apportion(shares, 10)
    assert first == apportion(dict(reversed(list(shares.items()))), 10)
    # 10 units over 3 equal shares: the leftover goes to the lexicographically first keys.
    assert first == {"aa": 4, "bb": 3, "cc": 3}


def test_apportion_drops_zero_and_subunit_entries_and_rejects_bad_totals():
    assert apportion({"a": 1.0, "b": 0.0}, 100) == {"a": 100}
    assert "c" not in apportion({"a": 1.0, "b": 1.0, "c": 1e-9}, 10)
    assert apportion({}, 100) == {}
    with pytest.raises(ValueError):
        apportion({"a": 1.0}, 0)


def test_blend_weights_set_influence_independently_of_component_size():
    """A component's share must come from its WEIGHT, not from how much text it had."""
    big = dict.fromkeys(("aa",), 1_000_000)  # huge mass, one n-gram
    small = {"bb": 1}  # tiny mass, one n-gram
    blended = blend_tables({"x": big, "y": small}, {"x": 0.75, "y": 0.25}, total=1_000_000)
    assert blended == {"aa": 750_000, "bb": 250_000}


def test_blend_renormalizes_over_present_components_only():
    tables = {"x": {"aa": 5}, "y": {"bb": 5}}
    # A weight naming an absent register must not steal mass from the present ones.
    blended = blend_tables(tables, {"x": 0.2, "y": 0.2, "absent": 0.6}, total=1000)
    assert blended == {"aa": 500, "bb": 500}


def test_blend_rejects_a_weight_set_that_selects_nothing():
    with pytest.raises(ValueError):
        blend_tables({"x": {"aa": 1}}, {"other": 1.0}, total=100)


def test_reweighting_shifts_mass_toward_the_upweighted_register():
    # Marker bigrams are exclusive to one register: "id" only in the code text, "th" only
    # in the prose text, so each one's blended count must track its register's weight.
    code = _text_source("c", "code", "for idx in range(n): sum += idx")
    prose = _text_source("p", "prose", "the quick brown fox")
    code_heavy = build_blend(
        [code, prose], {"code": 0.9, "prose": 0.1}, anchor_dir=None, total=10**6
    )
    prose_heavy = build_blend(
        [code, prose], {"code": 0.1, "prose": 0.9}, anchor_dir=None, total=10**6
    )
    assert prose_heavy.tables["bigrams"]["th"] > code_heavy.tables["bigrams"]["th"]
    assert code_heavy.tables["bigrams"]["id"] > prose_heavy.tables["bigrams"]["id"]


# --------------------------------------------------------------------------- build + manifest


def test_build_is_deterministic_for_identical_inputs():
    sources = [
        _text_source("p", "prose", "the quick brown fox jumps over the lazy dog"),
        _text_source("c", "code", "for index in range(count): total += index"),
    ]
    first = build_blend(sources, DEFAULT_WEIGHTS, anchor_dir=None, total=10**6)
    second = build_blend(sources, DEFAULT_WEIGHTS, anchor_dir=None, total=10**6)
    assert first.tables == second.tables
    assert first.manifest["weights_effective"] == second.manifest["weights_effective"]


def test_repo_sources_ignore_transient_directories(tmp_path):
    """A rerun must not change because the tests happened to have been run.

    Regression: the prose sources originally walked the whole tree, so a ``.venv/`` created
    by the test run added 15 vendored ``LICENSE.md`` files (and ``.pytest_cache/README.md``)
    to the "repo prose" register — the build was NOT reproducible, and the register was
    contaminated with third-party licence boilerplate.
    """
    (tmp_path / "REAL.md").write_text("genuine repository prose about layouts\n")
    (tmp_path / "paper.tex").write_text("genuine academic prose\n")
    clean = build_blend(
        [repo_prose_source(tmp_path), repo_latex_source(tmp_path)], {"prose": 1.0}, anchor_dir=None
    )

    for junk_dir in (".venv/lib/site-packages/numpy", ".pytest_cache", "build", "__pycache__"):
        junk = tmp_path / junk_dir
        junk.mkdir(parents=True)
        (junk / "LICENSE.md").write_text("Copyright zzz. Redistribution in binary form zzz.\n")
        (junk / "junk.tex").write_text("\\documentclass{zzz} vendored zzz\n")
    polluted = build_blend(
        [repo_prose_source(tmp_path), repo_latex_source(tmp_path)], {"prose": 1.0}, anchor_dir=None
    )

    assert polluted.tables == clean.tables
    assert [e["sha256"] for e in polluted.manifest["sources"]] == [
        e["sha256"] for e in clean.manifest["sources"]
    ]
    assert all(e["units"] == 1 for e in clean.manifest["sources"])


def test_repo_sources_exclude_the_corpus_directory_itself():
    """Counting the corpus files as prose would feed the generator its own output."""
    from keybo.data.build_corpus import _repo_files

    assert not [p for p in _repo_files(REPO, "*.md") if "data/corpus" in p.as_posix()]
    assert any(p.name == "README.md" for p in _repo_files(REPO, "*.md"))


def test_build_without_anchor_declares_itself_reproducible():
    result = build_blend(
        [_text_source("p", "prose", "the quick brown fox")], {"prose": 1.0}, anchor_dir=None
    )
    assert result.manifest["reproducible_without_anchor"] is True
    assert all(entry["reproducible"] for entry in result.manifest["sources"])


def test_build_with_anchor_marks_the_anchor_not_reproducible_and_records_its_hash():
    result = build_blend(
        [_text_source("p", "prose", "the quick brown fox")],
        {"prose": 0.5, "anchor": 0.5},
        anchor_dir=REPO / "data" / "corpus",
    )
    anchor = next(e for e in result.manifest["sources"] if e["register"] == "anchor")
    assert anchor["reproducible"] is False
    assert "NOT REPRODUCIBLE" in anchor["extraction"]
    assert set(anchor["files"]) == {"bigrams.txt", "trigrams.txt"}
    assert all(len(f["sha256"]) == 64 for f in anchor["files"].values())
    assert result.manifest["reproducible_without_anchor"] is False


def test_manifest_records_every_source_with_bytes_hash_and_extraction_rule():
    result = build_blend(
        [_text_source("p", "prose", "the quick brown fox jumps")], {"prose": 1.0}, anchor_dir=None
    )
    entry = result.manifest["sources"][0]
    assert entry["raw_bytes"] == len(b"the quick brown fox jumps")
    assert len(entry["sha256"]) == 64
    assert entry["extraction"]
    assert entry["ngrams"]["bigram_tokens"] > 0


def test_anchor_skipgrams_are_re_derived_not_read_from_the_unreproducible_pass():
    """The anchor must use ONE skipgram convention — its own trigram marginalization."""
    anchor = load_anchor(REPO / "data" / "corpus")
    assert anchor["skipgrams"] == dict(marginalize_skipgrams(anchor["trigrams"]))
    assert anchor["skipgrams"] != load_frequencies(str(REPO / "data" / "corpus" / "1-skip.txt"))


def test_build_raises_for_a_required_source_that_yields_nothing():
    empty = _text_source("empty", "prose", "")
    with pytest.raises(ValueError, match="no n-grams"):
        build_blend([empty], {"prose": 1.0}, anchor_dir=None)


def test_optional_source_that_yields_nothing_is_skipped_not_fatal():
    optional = Source(
        name="missing",
        register="reference",
        extraction="nothing",
        reader=lambda: iter([]),
        optional=True,
    )
    result = build_blend(
        [_text_source("p", "prose", "the quick brown fox"), optional],
        {"prose": 1.0, "reference": 0.5},
        anchor_dir=None,
    )
    assert [e["name"] for e in result.manifest["sources"]] == ["p"]
    assert "reference" not in result.manifest["weights_effective"]


# --------------------------------------------------------------------------- file format


def test_write_table_emits_the_production_format_sorted_deterministically(tmp_path):
    path = tmp_path / "bigrams.txt"
    write_table(path, {"ab": 1, "cd": 5, "  ": 5})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["  \t5", "cd\t5", "ab\t1"]  # -count, then ngram; space is a real char


def test_written_tables_round_trip_through_the_production_loader(tmp_path):
    result = build_blend(
        [_text_source("p", "prose", "the quick brown fox jumps over the lazy dog")],
        {"prose": 1.0},
        anchor_dir=None,
        total=10**6,
    )
    write_build(result, tmp_path)
    for name in ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"):
        loaded = load_frequencies(str(tmp_path / name))
        assert loaded, name
        assert sum(loaded.values()) == 10**6, name
    assert load_frequencies(str(tmp_path / "1-skip.txt")) == load_frequencies(
        str(tmp_path / "1-skip31.txt")
    )
    assert json.loads((tmp_path / "manifest.json").read_text())["declared_total"] == 10**6


def test_space_containing_ngrams_survive_the_write_read_round_trip(tmp_path):
    """A leading/trailing space is DATA, not whitespace to be stripped."""
    write_table(tmp_path / "t.txt", {"e ": 7, " e": 3, "ee": 1})
    assert load_frequencies(str(tmp_path / "t.txt")) == {"e ": 7, " e": 3, "ee": 1}


# --------------------------------------------------------------------------- the committed blend


@pytest.mark.skipif(not BLEND.is_dir(), reason="blend-v1 not built")
class TestCommittedBlend:
    """The committed ``data/corpus/blend-v1`` must satisfy every published claim."""

    @pytest.mark.parametrize(
        ("filename", "length"),
        [("bigrams.txt", 2), ("trigrams.txt", 3), ("1-skip.txt", 2), ("1-skip31.txt", 2)],
    )
    def test_table_is_loadable_correct_length_and_sums_to_the_declared_total(
        self, filename, length
    ):
        table = load_frequencies(str(BLEND / filename))
        assert table
        assert all(len(gram) == length for gram in table)
        assert all(count > 0 for count in table.values())
        assert sum(table.values()) == DECLARED_TOTAL

    def test_declared_total_makes_the_table_sum_to_one_and_to_one_hundred(self):
        """The stated requirement: a frequency list that sums to 1 / 100."""
        total = sum(load_frequencies(str(BLEND / "bigrams.txt")).values())
        assert total / 1e9 == 1.0
        assert total / 1e7 == 100.0

    @pytest.mark.parametrize(
        "filename", ["bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"]
    )
    def test_no_ngram_leaves_the_production_charset(self, filename):
        chars = set("".join(load_frequencies(str(BLEND / filename))))
        assert chars <= CORPUS_CHARSET

    def test_skipgram_names_carry_identical_content(self):
        assert load_frequencies(str(BLEND / "1-skip.txt")) == load_frequencies(
            str(BLEND / "1-skip31.txt")
        )

    def test_manifest_matches_the_tables_it_describes(self):
        manifest = json.loads((BLEND / "manifest.json").read_text())
        assert manifest["schema"] == "keybo-corpus-manifest/1"
        assert manifest["declared_total"] == DECLARED_TOTAL
        assert manifest["case_preserved"] is True
        for kind, info in manifest["outputs"].items():
            for filename in info["files"]:
                table = load_frequencies(str(BLEND / filename))
                assert len(table) == info["types"], (kind, filename)
                assert sum(table.values()) == info["total"], (kind, filename)

    def test_manifest_documents_multiple_registers_and_the_anchor_honestly(self):
        manifest = json.loads((BLEND / "manifest.json").read_text())
        registers = {entry["register"] for entry in manifest["sources"]}
        assert len(registers) >= 3, "the point of the blend is many different corpora"
        assert abs(sum(manifest["weights_effective"].values()) - 1.0) < 1e-12
        anchors = [e for e in manifest["sources"] if not e["reproducible"]]
        assert len(anchors) == 1 and anchors[0]["register"] == "anchor"
        assert manifest["reproducible_without_anchor"] is False

    def test_blend_is_genuinely_multi_source_not_a_relabelled_iweb_copy(self):
        """If the blend equalled the anchor, nothing would have been blended."""
        blend = load_frequencies(str(BLEND / "bigrams.txt"))
        iweb = load_frequencies(str(REPO / "data" / "corpus" / "bigrams.txt"))
        iweb_mass = sum(iweb.values())
        shared = set(blend) & set(iweb)
        assert shared, "the two tables must be comparable at all"
        divergence = sum(
            abs(blend[gram] / DECLARED_TOTAL - iweb[gram] / iweb_mass) for gram in shared
        )
        assert divergence > 0.01, f"blend is ~identical to iWeb (L1 {divergence:.5f})"

    def test_the_production_corpus_is_left_untouched(self):
        """Switching production is the user's decision — the blend only ADDS files."""
        bigrams = load_frequencies(str(REPO / "data" / "corpus" / "bigrams.txt"))
        assert sum(bigrams.values()) == 515_596_120  # the committed iWeb total
        assert bigrams["th"] == 9_709_171  # the paper's published top-bigram count
