"""`keybo analyze` under the corpus swap (CORPUS-SWAP-1).

These are the end-to-end guards on the three properties that make swapping the production
default from iWeb to ``blend-v1`` safe, plus the two conventions the comparison rests on.
The per-corpus *values* are pinned in ``test_analyze_allgauge.py`` (which now names iWeb);
what is pinned HERE is the wiring — that the default moved, that iWeb is still reachable,
that output says which corpus it used, and that the two are actually different numbers.

Every expectation is a fact re-derived on this branch, not a value copied from a brief.
"""

from __future__ import annotations

import json

import pytest

from keybo.cli.__main__ import main
from keybo.data.corpus import CORPUS_ENV_VAR, IWEB, PRODUCTION_DEFAULT

#: The layout every control here uses (the campaign incumbent).
KEYBO_LSB = "keybo-lsb"

#: Fast flags: the gauges below are corpus-sensitive but need neither the 6-model time
#: card nor the fitted surfaces, and those are the slow parts.
FAST = ("--no-time", "--no-model-scores")


def _json(capsys, argv: list[str]) -> dict:
    rc = main(argv)
    assert rc == 0, f"analyze {argv} returned {rc}"
    return json.loads(capsys.readouterr().out)


# ------------------------------------------------------------------ 1. the default moved


def test_analyze_defaults_to_blend_v1_and_says_so(capsys, monkeypatch):
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    out = _json(capsys, ["analyze", KEYBO_LSB, *FAST, "--json"])
    assert out["corpus"] == PRODUCTION_DEFAULT


def test_the_two_corpora_give_DIFFERENT_gauge_values(capsys, monkeypatch):
    """The swap must actually change the numbers, or none of this matters.

    Guards the failure where a resolver is added, everything passes, and the default
    silently never took effect — the corpus-swap analogue of a flag that is present but
    inert.
    """
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    blend = _json(capsys, ["analyze", KEYBO_LSB, *FAST, "--json"])
    iweb = _json(capsys, ["analyze", "--corpus", IWEB, KEYBO_LSB, *FAST, "--json"])

    b = blend["rows"][KEYBO_LSB]["gauges"]
    i = iweb["rows"][KEYBO_LSB]["gauges"]
    assert b != i
    # and it is not one lone cell moving: the corpus reweights every corpus-sensitive gauge
    moved = [name for name in i if b[name] != i[name]]
    assert len(moved) >= 13, f"only {len(moved)} of {len(i)} gauges moved: {moved}"


# --------------------------------------------------- 2. iWeb is still reachable BY NAME


def test_iweb_is_reachable_by_flag_and_reproduces_a_frozen_board_value(capsys):
    """The audit-trail guard: a frozen iWeb number must still be obtainable after the swap.

    ``sfb`` for keybo-lsb is asserted EXACTLY against wscissor-allgauge's frozen value —
    the same literal ``test_analyze_allgauge.py`` uses, restated here so this file fails on
    its own if ``--corpus iweb`` ever stops resolving to the frozen corpus.
    """
    out = _json(capsys, ["analyze", "--corpus", IWEB, KEYBO_LSB, *FAST, "--json"])
    assert out["corpus"] == IWEB
    assert out["rows"][KEYBO_LSB]["gauges"]["sfb"] == 1.0784319931923778


def test_env_var_reaches_iweb_too(capsys, monkeypatch):
    monkeypatch.setenv(CORPUS_ENV_VAR, IWEB)
    out = _json(capsys, ["analyze", KEYBO_LSB, *FAST, "--json"])
    assert out["corpus"] == IWEB
    assert out["rows"][KEYBO_LSB]["gauges"]["sfb"] == 1.0784319931923778


def test_the_flag_beats_the_env_var(capsys, monkeypatch):
    """A stale ``KEYBO_CORPUS`` export must not override what the user typed."""
    monkeypatch.setenv(CORPUS_ENV_VAR, IWEB)
    out = _json(capsys, ["analyze", "--corpus", PRODUCTION_DEFAULT, KEYBO_LSB, *FAST, "--json"])
    assert out["corpus"] == PRODUCTION_DEFAULT


def test_an_unknown_corpus_name_exits_instead_of_scoring_the_default(capsys):
    """A typo must never silently produce numbers from a different corpus."""
    with pytest.raises(SystemExit, match="unknown corpus"):
        main(["analyze", "--corpus", "blend-v9-typo", KEYBO_LSB, *FAST, "--json"])


# ------------------------------------------- 3. the corpus is IDENTIFIED in every output


def test_json_carries_a_content_hash_not_just_a_name(capsys, monkeypatch):
    """Trap #13's guard: a name can be wrong, a sha256 of the table cannot.

    A modified table would keep its directory name and its numbers would change silently;
    the hash is what makes the provenance block a fact rather than a label.
    """
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    out = _json(capsys, ["analyze", KEYBO_LSB, *FAST, "--json"])
    prov = out["corpus_provenance"]
    assert prov["corpus"] == PRODUCTION_DEFAULT
    for table in ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt"):
        assert len(prov["sha256"][table]) == 64
    # blend-v1 ships a manifest declaring 1e9 per table; iWeb ships none.
    assert prov["declared_total"] == 1_000_000_000
    assert prov["skipgram_table"] == "1-skip31.txt"


def test_the_two_corpora_report_different_hashes(capsys):
    blend = _json(capsys, ["analyze", "--corpus", PRODUCTION_DEFAULT, KEYBO_LSB, *FAST, "--json"])
    iweb = _json(capsys, ["analyze", "--corpus", IWEB, KEYBO_LSB, *FAST, "--json"])
    b, i = blend["corpus_provenance"], iweb["corpus_provenance"]
    assert b["sha256"]["trigrams.txt"] != i["sha256"]["trigrams.txt"]
    assert i["declared_total"] is None, "iWeb ships no manifest — must not borrow a total"


def test_the_text_report_names_the_corpus_on_its_first_line(capsys, monkeypatch):
    """A reader of the text report must not have to know what the default is."""
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    assert main(["analyze", KEYBO_LSB, *FAST]) == 0
    text = capsys.readouterr().out
    assert PRODUCTION_DEFAULT in text.splitlines()[0]


def test_the_text_report_states_that_community_scores_do_not_move_with_the_corpus(
    capsys, monkeypatch
):
    """Anyone reading "we swapped the corpus" may assume every column moved. Four did not."""
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    assert main(["analyze", KEYBO_LSB, *FAST]) == 0
    text = capsys.readouterr().out
    assert "corpus-INVARIANT" in text
    assert "not re-fit" in text


# --------------------------------- the two conventions this comparison rests on (trap 13)


@pytest.mark.slow
def test_the_community_gauges_are_corpus_INVARIANT(capsys):
    """They run on their own vendored corpora, so they must be BIT-IDENTICAL across corpora.

    Load-bearing for the swap's write-up: reporting these as "changed by the swap" would be
    wrong, and a future change that accidentally wired them to the shared corpus would make
    every frozen community number unreproducible. Asserted with ``==``, not approx.
    """
    blend = _json(
        capsys, ["analyze", "--corpus", PRODUCTION_DEFAULT, KEYBO_LSB, "--no-time", "--json"]
    )
    iweb = _json(capsys, ["analyze", "--corpus", IWEB, KEYBO_LSB, "--no-time", "--json"])
    for block in ("community", "community_primed"):
        assert blend["rows"][KEYBO_LSB][block] == iweb["rows"][KEYBO_LSB][block], block


def test_both_corpora_are_scored_on_the_same_skipgram_table(capsys):
    """The comparison must not confound the corpus with the skipgram convention.

    iWeb's ``1-skip.txt`` and ``1-skip31.txt`` are DIFFERENT tables (3474 vs 4087 keys)
    while blend-v1 writes the marginalization under both names. If the two ends of a
    comparison read different names, the measured difference is partly a convention change.
    """
    blend = _json(capsys, ["analyze", "--corpus", PRODUCTION_DEFAULT, KEYBO_LSB, *FAST, "--json"])
    iweb = _json(capsys, ["analyze", "--corpus", IWEB, KEYBO_LSB, *FAST, "--json"])
    assert blend["skipgram_table"] == iweb["skipgram_table"] == "1-skip31.txt"


def test_dvorak_charset_dependent_cells_stay_NA_under_the_new_default(capsys, monkeypatch):
    """dvorak is not C30M; the swap must not turn an N/A into a silently-wrong number."""
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    out = _json(capsys, ["analyze", "dvorak", "--no-time", "--json"])
    row = next(r for r in out["rows"].values() if r["layout"].startswith("',.py"))
    assert row["community"]["oxeylyzer1"] is None
    assert row["community"]["oxeylyzer2"] is None
    assert row["community"]["wfd"] is None
    assert row["model_scores"]["available"] is False
    # charset-agnostic gauges still produce numbers
    assert row["gauges"]["sfb"] > 0.0


def test_the_model_surfaces_are_not_refit_by_a_corpus_change(capsys):
    """The swap changes the objective's WEIGHTING, never the fitted timing model.

    The surfaces are baked at 90 WPM, so the labels that say so must survive the swap —
    otherwise a reader can reasonably assume "we swapped the corpus" implies a refit.
    """
    out = _json(capsys, ["analyze", KEYBO_LSB, "--no-time", "--json"])
    scores = out["rows"][KEYBO_LSB]["model_scores"]
    assert scores["baked_wpm"] == 90.0
    assert scores["wpm_matches_request"] is True
