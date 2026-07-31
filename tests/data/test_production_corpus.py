"""Tests for the single production-corpus resolver (CORPUS-SWAP-1).

Before this change the production corpus path was hardcoded at eight call sites, so
"which corpus does keybo score on?" had eight answers and no override. The swap to
``blend-v1`` as the default is only safe if three things hold, and each gets a test here:

1. **The default is blend-v1** — one place decides, and every site follows it.
2. **iWeb stays reachable BY NAME.** Every frozen board in the campaign was computed on
   iWeb; if the swap made those numbers unreproducible it would have destroyed the audit
   trail. ``KEYBO_CORPUS=iweb`` (or ``--corpus iweb``) must still resolve.
3. **The resolved corpus is IDENTIFIED in output.** A silent default change is how a
   future agent stitches two corpora into one table (campaign trap #13). The identity is
   a content hash, not just a name, so a *modified* table cannot masquerade as a known one.

The iWeb ``1-skip.txt``/``1-skip31.txt`` distinction is load-bearing and tested: they are
DIFFERENT tables in iWeb (3474 vs 4087 keys) and the same table in blend-v1, so a
comparison that does not pin ``1-skip31.txt`` confounds the corpus change with a
skipgram-convention change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import keybo.data.corpus as _corpus_module
from keybo.data.corpus import (
    CORPUS_ENV_VAR,
    IWEB,
    PRODUCTION_DEFAULT,
    corpus_identity,
    known_corpora,
    production_corpus_dir,
    resolve_corpus_dir,
)

#: The tree whose ``data/`` these tests assert against — derived from the IMPORTED MODULE, not from
#: this test file.
#:
#: Those two differ, and silently. ``corpus._repo_root()`` is ``Path(corpus.__file__).parents[3]``, so
#: it follows wherever ``keybo`` was imported FROM; deriving REPO from ``__file__`` here follows the
#: TEST instead. In the main clone they coincide and everything passes. In a git worktree run without
#: the worktree's ``src/`` on ``PYTHONPATH``, the editable install resolves ``keybo`` to the MAIN
#: CLONE's ``src/`` while the tests sit in the worktree — so three of these tests failed with
#: ``/local/.../repos/keybo/data/corpus != /tmp/<worktree>/data/corpus``, which reads exactly like a
#: broken resolver and is not one. **The resolver is correct** (verified: with the worktree on
#: PYTHONPATH it returns the worktree, and 19/19 pass).
#:
#: This is the wrong-tree hazard ``keybo.testkit.assert_module_under`` exists for, so the guard below
#: asserts the two trees agree rather than leaving a confusing failure. Keying REPO off the module
#: makes the assertions true of the code actually under test — which is the only tree whose ``data/``
#: the resolver will ever read.
REPO = Path(_corpus_module.__file__).resolve().parents[3]


def test_the_tests_and_the_module_under_test_come_from_the_SAME_TREE() -> None:
    """Fail LOUDLY and specifically if the harness imported keybo from a different checkout.

    Without this, a worktree run produces three path-mismatch failures that look like a resolver bug.
    The remedy is ``PYTHONPATH=<worktree>/src``, and the message says so.
    """
    tests_tree = Path(__file__).resolve().parents[2]
    if tests_tree != REPO:
        raise AssertionError(
            f"harness/tree mismatch: tests live in {tests_tree} but keybo was imported from {REPO}. "
            f"An editable install resolves `keybo` to the main clone's src/, so a worktree run scores "
            f"the wrong tree's data/. Re-run with PYTHONPATH={tests_tree}/src."
        )


# --------------------------------------------------------------- 1. the default is blend-v1


def test_production_default_is_blend_v1():
    """The swap itself: one constant names the production corpus."""
    assert PRODUCTION_DEFAULT == "blend-v1"


def test_production_corpus_dir_resolves_to_the_committed_blend_directory(monkeypatch):
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    assert production_corpus_dir() == REPO / "data" / "corpus" / "blend-v1"


def test_the_default_corpus_directory_actually_holds_all_four_tables(monkeypatch):
    """A name that resolves to an incomplete directory is worse than no default."""
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    resolved = production_corpus_dir()
    for table in ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"):
        assert (resolved / table).is_file(), f"{resolved} is missing {table}"


# ------------------------------------------------------- 2. iWeb stays reachable BY NAME


def test_iweb_is_still_resolvable_by_name():
    """The campaign's whole audit trail is on iWeb; losing this loses reproducibility."""
    assert resolve_corpus_dir(IWEB) == REPO / "data" / "corpus"


def test_iweb_and_the_default_are_different_directories():
    assert resolve_corpus_dir(IWEB) != production_corpus_dir()


def test_env_var_selects_iweb(monkeypatch):
    monkeypatch.setenv(CORPUS_ENV_VAR, IWEB)
    assert production_corpus_dir() == REPO / "data" / "corpus"


def test_explicit_argument_beats_the_env_var(monkeypatch):
    """A CLI flag must win over an inherited environment, or a shell export silently
    overrides what the user asked for on the command line."""
    monkeypatch.setenv(CORPUS_ENV_VAR, IWEB)
    assert production_corpus_dir("blend-v1") == REPO / "data" / "corpus" / "blend-v1"


def test_an_explicit_path_is_honoured_verbatim(tmp_path):
    """Any directory holding the tables can be named, so a new blend needs no code change."""
    for table in ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"):
        (tmp_path / table).write_text("th\t100\n")
    assert resolve_corpus_dir(str(tmp_path)) == tmp_path


def test_unknown_name_fails_loudly_and_lists_the_known_names():
    """Never a silent fallback to the default: a typo must not score the wrong corpus."""
    with pytest.raises(SystemExit) as excinfo:
        resolve_corpus_dir("blend-v2-typo")
    message = str(excinfo.value)
    assert "blend-v2-typo" in message
    for name in known_corpora():
        assert name in message


def test_a_directory_missing_a_table_fails_loudly(tmp_path):
    """A half-populated directory must not score three gauges and silently skip the rest."""
    (tmp_path / "bigrams.txt").write_text("th\t100\n")
    with pytest.raises(SystemExit) as excinfo:
        resolve_corpus_dir(str(tmp_path))
    assert "trigrams.txt" in str(excinfo.value)


def test_known_corpora_contains_both_the_default_and_iweb():
    assert PRODUCTION_DEFAULT in known_corpora()
    assert IWEB in known_corpora()


# --------------------------------------------- 3. the resolved corpus is identified in output


def test_corpus_identity_names_the_corpus_and_hashes_every_table():
    identity = corpus_identity(production_corpus_dir())
    assert identity["corpus"] == PRODUCTION_DEFAULT
    assert identity["path"].endswith("data/corpus/blend-v1")
    for table in ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"):
        assert len(identity["sha256"][table]) == 64


def test_identity_distinguishes_the_two_corpora():
    """The point of a content hash: two corpora cannot report the same identity."""
    blend = corpus_identity(production_corpus_dir())
    iweb = corpus_identity(resolve_corpus_dir(IWEB))
    assert blend["corpus"] != iweb["corpus"]
    assert blend["sha256"]["trigrams.txt"] != iweb["sha256"]["trigrams.txt"]


def test_identity_reports_an_unknown_directory_as_a_path_not_a_name(tmp_path):
    for table in ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"):
        (tmp_path / table).write_text("th\t100\n")
    identity = corpus_identity(tmp_path)
    assert identity["corpus"] == "custom"


def test_identity_carries_the_manifest_total_when_the_corpus_has_one():
    """blend-v1 ships a manifest; its declared total belongs in the provenance block."""
    identity = corpus_identity(production_corpus_dir())
    assert identity["declared_total"] == 1_000_000_000


def test_identity_has_no_manifest_total_for_iweb():
    """iWeb ships no manifest — report None, never a borrowed number from another corpus."""
    assert corpus_identity(resolve_corpus_dir(IWEB))["declared_total"] is None


def test_identity_is_json_serializable():
    """It is embedded in `analyze --json`; a non-serializable value would break that."""
    json.dumps(corpus_identity(production_corpus_dir()))


# ----------------------------------------- the skipgram convention this comparison rests on


def test_iweb_1skip_and_1skip31_are_different_tables():
    """Load-bearing: iWeb's two skipgram files are NOT interchangeable, so any
    iWeb-vs-blend comparison must pin 1-skip31 or it confounds two changes at once."""
    from keybo.data.corpus import load_frequencies

    iweb = resolve_corpus_dir(IWEB)
    assert load_frequencies(str(iweb / "1-skip.txt")) != load_frequencies(
        str(iweb / "1-skip31.txt")
    )


def test_blend_1skip_and_1skip31_are_the_same_table():
    """In blend-v1 both names carry the marginalized skipgrams (build_corpus emits both),
    so blend-v1 is a drop-in wherever either name is loaded."""
    from keybo.data.corpus import load_frequencies

    blend = production_corpus_dir()
    assert load_frequencies(str(blend / "1-skip.txt")) == load_frequencies(
        str(blend / "1-skip31.txt")
    )
