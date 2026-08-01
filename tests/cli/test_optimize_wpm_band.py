"""`keybo optimize --wpm-band`: the band objective's CLI wiring and its refusals.

The gates matter more than the happy path here. A band objective that silently degenerates to
the incumbent (an un-normalized minimax, which collapses to the band's lowest pace) is worse
than one that errors, because it reports a wider objective while optimizing the narrow one.
"""

from __future__ import annotations

import argparse
import json

import pytest

from keybo.cli import optimize as OPT
from keybo.layouts import NAMED_LAYOUTS
from tests.cli.test_cli import _train_tiny_model

QWERTY = NAMED_LAYOUTS["qwerty"]


def _args(model_path, **over):
    """A minimal `optimize` namespace; `max_outer=1` keeps the search a smoke test."""
    base = dict(
        model=model_path,
        ngram="bigram",
        bigram_freqs="data/corpus/bigrams.txt",
        trigram_freqs="data/corpus/trigrams.txt",
        corpus=None,
        target_wpm=90.0,
        start=QWERTY,
        seed=0,
        alpha=0.999,
        max_outer=1,
        no_local_search=True,
        attempts=1,
        out=None,
        comfort_weight=0.0,
        comfort_config=None,
        oxey_weight=0.0,
        finger_load_weight=0.0,
        no_table=False,
        model_weight=None,
        model_anchors=None,
        wpm_band=None,
        wpm_aggregation="mean",
        wpm_reference=None,
        no_progress=True,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_mean_band_runs_and_records_its_own_objective(tmp_path):
    """The result file must name the band — `target_wpm` alone cannot describe it."""
    model = _train_tiny_model(tmp_path / "bg.json")
    out = tmp_path / "res.json"
    rc = OPT.run(_args(model, wpm_band="90,100,110,120", out=str(out)))
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["wpm_band"] == [90.0, 100.0, 110.0, 120.0]
    assert result["wpm_aggregation"] == "mean"
    assert result["objective"] == "mean of total_ms over wpm in {90/100/110/120}"
    assert result["target_wpm_unused_for_band_search"] is True
    assert len(result["per_wpm_total_ms"]) == 4
    # monotone decreasing in wpm: the 1/wpm factor, i.e. the reason minimax needs a reference
    curve = result["per_wpm_total_ms"]
    assert curve == sorted(curve, reverse=True)


def test_minimax_without_reference_is_refused(tmp_path):
    """The degeneracy gate — and it must fire BEFORE the (long) search, not after."""
    model = _train_tiny_model(tmp_path / "bg.json")
    with pytest.raises(SystemExit, match="requires --wpm-reference"):
        OPT.run(_args(model, wpm_band="90,120", wpm_aggregation="minimax"))


def test_minimax_with_reference_runs(tmp_path):
    model = _train_tiny_model(tmp_path / "bg.json")
    out = tmp_path / "res.json"
    rc = OPT.run(
        _args(
            model,
            wpm_band="90,100,110,120",
            wpm_aggregation="minimax",
            wpm_reference="qwerty",
            out=str(out),
        )
    )
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["wpm_aggregation"] == "minimax"
    assert result["wpm_reference"] == QWERTY  # the NAME was resolved to the board
    assert "minimax" in result["objective"]


def test_reference_must_match_start_charset(tmp_path):
    model = _train_tiny_model(tmp_path / "bg.json")
    with pytest.raises(SystemExit, match="permutation of --start"):
        OPT.run(
            _args(
                model,
                wpm_band="90,120",
                wpm_aggregation="minimax",
                wpm_reference="abcdefghij",
            )
        )


def test_reference_without_minimax_is_refused(tmp_path):
    model = _train_tiny_model(tmp_path / "bg.json")
    with pytest.raises(SystemExit, match="only applies to"):
        OPT.run(_args(model, wpm_band="90,120", wpm_reference="qwerty"))


def test_band_refuses_trigram_and_preference_terms(tmp_path):
    model = _train_tiny_model(tmp_path / "bg.json")
    with pytest.raises(SystemExit, match="bigram objective only"):
        OPT.run(_args(model, wpm_band="90,120", ngram="trigram"))
    with pytest.raises(SystemExit, match="cannot be combined"):
        OPT.run(_args(model, wpm_band="90,120", comfort_weight=1.0))


def test_malformed_and_empty_band_are_refused(tmp_path):
    model = _train_tiny_model(tmp_path / "bg.json")
    with pytest.raises(SystemExit, match="comma-separated numbers"):
        OPT.run(_args(model, wpm_band="90,fast"))
    with pytest.raises(SystemExit, match="is empty"):
        OPT.run(_args(model, wpm_band=","))


def test_out_of_stamped_range_band_warns_without_refusing(tmp_path, capsys):
    """Warn, not refuse — mirrors build_scorer's --target-wpm behaviour.

    Also the regression test for the crash this path had: the warning uses `sys.stderr`, and
    `sys` was not imported in this module, so the first out-of-range band raised NameError
    instead of warning.
    """
    model = _train_tiny_model(tmp_path / "bg.json")
    rc = OPT.run(_args(model, wpm_band="90,120,200"))
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "200" in err


def test_single_pace_band_reproduces_the_incumbent_layout(tmp_path):
    """A length-1 band is the shipped single-point objective, so the SEARCH must agree.

    Same seed, same start, same everything: the band path and the shipped path differ only in
    which scorer object computes an identical number, so the winning board must be identical.
    Without this, the control arm and the band arms would not be comparable.
    """
    model = _train_tiny_model(tmp_path / "bg.json")
    a, b = tmp_path / "band.json", tmp_path / "point.json"
    OPT.run(_args(model, wpm_band="90", out=str(a), max_outer=3))
    OPT.run(_args(model, target_wpm=90.0, out=str(b), max_outer=3))
    assert json.loads(a.read_text())["layout"] == json.loads(b.read_text())["layout"]
