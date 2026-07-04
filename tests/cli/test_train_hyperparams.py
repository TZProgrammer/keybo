"""CLI tests for `keybo train --hyperparams <json>` (restores the tune->train loop).

`keybo tune` writes a best_hyperparams.json; these tests drive `keybo train` consuming it,
covering the flag-overrides-json precedence rule, metadata recording, and how bad params
surface (xgboost 3.x: an unknown param KEY warns but trains; an invalid VALUE raises).
"""

import json

import pytest
import xgboost as xgb

from keybo.cli.__main__ import main


def _write_bistrokes(path):
    """A tiny bistrokes TSV the train command consumes (mirrors test_cli's fixture)."""
    lines = []
    for bg in ["th", "he", "an", "in", "er", "re", "on", "at"]:
        lines.append(f"((-1, 3), (1, 2))\t{bg}\t5\t(90, 120)\t(85, 130)\t(92, 118)")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _num_rounds(model_path):
    """The number of boosted rounds baked into the saved booster artifact."""
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster.num_boosted_rounds()


def _train_argv(strokes, out_model, params_path, *extra):
    return [
        "train",
        "--strokes",
        strokes,
        "--ngram",
        "bigram",
        "--output",
        str(out_model),
        "--min-samples",
        "1",
        "--hyperparams",
        str(params_path),
        *extra,
    ]


def test_train_hyperparams_json_round_trip(tmp_path):
    """A params json from `tune` is honored: n_estimators=7 -> 7 boosted rounds."""
    strokes = _write_bistrokes(tmp_path / "bistrokes.tsv")
    params = tmp_path / "best_hyperparams.json"
    params.write_text(json.dumps({"n_estimators": 7, "max_depth": 2}))
    out_model = tmp_path / "bg.json"

    rc = main(_train_argv(strokes, out_model, params))
    assert rc == 0
    assert out_model.exists()
    assert _num_rounds(str(out_model)) == 7


def test_train_explicit_flag_overrides_hyperparams_json(tmp_path):
    """An explicit --n-estimators wins over the json file's value (7 -> 3)."""
    strokes = _write_bistrokes(tmp_path / "bistrokes.tsv")
    params = tmp_path / "best_hyperparams.json"
    params.write_text(json.dumps({"n_estimators": 7, "max_depth": 2}))
    out_model = tmp_path / "bg.json"

    rc = main(_train_argv(strokes, out_model, params, "--n-estimators", "3"))
    assert rc == 0
    assert _num_rounds(str(out_model)) == 3


def test_train_hyperparams_recorded_in_metadata(tmp_path):
    """The resolved hyperparams actually used are recorded in the model metadata sidecar."""
    strokes = _write_bistrokes(tmp_path / "bistrokes.tsv")
    params = tmp_path / "best_hyperparams.json"
    params.write_text(json.dumps({"n_estimators": 6, "max_depth": 2}))
    out_model = tmp_path / "bg.json"

    rc = main(_train_argv(strokes, out_model, params))
    assert rc == 0
    meta = json.loads(out_model.with_suffix(".meta.json").read_text())
    assert meta["extra"]["hyperparams"]["n_estimators"] == 6
    assert meta["extra"]["hyperparams"]["max_depth"] == 2


def test_train_hyperparams_bad_value_fails_loudly(tmp_path):
    """An invalid VALUE for a known xgboost param propagates loudly (no silent bad model)."""
    strokes = _write_bistrokes(tmp_path / "bistrokes.tsv")
    params = tmp_path / "best_hyperparams.json"
    # xgboost rejects a negative max_depth at fit time with an XGBoostError.
    params.write_text(json.dumps({"n_estimators": 4, "max_depth": -3}))
    out_model = tmp_path / "bg.json"

    with pytest.raises(xgb.core.XGBoostError):
        main(_train_argv(strokes, out_model, params))


def test_train_hyperparams_unknown_key_is_silently_ignored(tmp_path, recwarn):
    """DOCUMENTED behavior of the keybo CLI surface: an unknown param KEY is not fatal, and
    on this path is entirely SILENT. Bare xgboost 3.x would emit a UserWarning ("Parameters:
    {...} are not used"), but XGBoostTypingModel hardcodes verbosity=0, which suppresses that
    warning at the C++ level. So a stray/typo'd key in a tune-output file is swallowed with no
    warning and no error; the known params still take effect (n_estimators=4 -> 4 rounds).

    An invalid VALUE, by contrast, DOES raise -- see the bad_value test above. (If provenance
    of a mistyped key ever matters, the fix would be to validate keys in the CLI, not here.)
    """
    strokes = _write_bistrokes(tmp_path / "bistrokes.tsv")
    params = tmp_path / "best_hyperparams.json"
    params.write_text(json.dumps({"n_estimators": 4, "totally_bogus_param": 123}))
    out_model = tmp_path / "bg.json"

    rc = main(_train_argv(strokes, out_model, params))
    assert rc == 0
    # Known params still applied despite the bogus key riding along.
    assert _num_rounds(str(out_model)) == 4
    # No warning surfaces through the CLI (verbosity=0 mutes xgboost's "not used" notice).
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
