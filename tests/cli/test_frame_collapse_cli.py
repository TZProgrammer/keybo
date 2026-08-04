"""Tests for `keybo frame-collapse` (:mod:`keybo.cli.frame_collapse`).

The fast paths only (no ``--floor``): the floor path loads six XGBoost models, and the numbers it
produces are already pinned against the library in
``tests/analysis/test_frame_collapse.py``. What is tested here is the SURFACE — dispatch, the frame
registry, the extension point, and the four refusals — since a CLI that silently accepts an
incoherent combination is how a wrong number gets published.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from keybo.cli.__main__ import build_parser, main


def _run(capsys, *argv) -> str:
    assert main(["frame-collapse", *argv]) == 0
    return capsys.readouterr().out


def test_frame_collapse_is_registered_as_a_subcommand():
    """Dispatch, not just import: the module could exist and be unreachable from the CLI."""
    parser = build_parser()
    args = parser.parse_args(["frame-collapse"])
    assert args.command == "frame-collapse"
    # and the house-style contract: add_arguments gave it the two shared flags
    assert hasattr(args, "json")
    assert hasattr(args, "tol")


def test_default_invocation_reports_the_three_ledger_frames(capsys):
    out = _run(capsys)
    assert "961 cells" in out
    assert "31 positions incl. space" in out
    for frame in ("served", "interp", "interp-wpm"):
        assert frame in out
    assert "765" in out and "378" in out


def test_json_output_carries_the_published_counts(capsys):
    payload = json.loads(_run(capsys, "--json"))
    assert payload["target"] is None  # no --floor
    frames = payload["frames"]
    assert frames["served"]["distinct_feature_rows"] == 765
    assert frames["served"]["n_columns"] == 20
    assert frames["interp"]["distinct_feature_rows"] == 378
    assert frames["interp"]["n_columns"] == 10
    assert frames["interp-wpm"]["distinct_feature_rows"] == 378
    assert frames["interp-wpm"]["n_columns"] == 11
    assert frames["served"]["floor_wmae"] is None  # floors need --floor
    assert frames["served"]["includes_space"] is True
    assert frames["served"]["n_cells"] == 961


def test_no_space_flag_selects_the_other_961_cell_space(capsys):
    """The 765-vs-775 distinction, reachable from the command line."""
    with_space = json.loads(_run(capsys, "--frame", "served", "--json"))
    no_space = json.loads(
        _run(capsys, "--frame", "served", "--geometry", "k31", "--no-space", "--json")
    )
    assert with_space["frames"]["served"]["n_cells"] == 961
    assert no_space["frames"]["served"]["n_cells"] == 961
    assert with_space["frames"]["served"]["distinct_feature_rows"] == 765
    assert no_space["frames"]["served"]["distinct_feature_rows"] == 775
    assert no_space["frames"]["served"]["includes_space"] is False


def test_trigram_frame_runs_at_its_own_order_without_being_told(capsys):
    """``--order`` defaults to the built-in frame's own order — a trigram frame is order 3."""
    payload = json.loads(_run(capsys, "--frame", "trigram", "--json"))
    r = payload["frames"]["trigram"]
    assert r["order"] == 3
    assert r["n_cells"] == 29791
    assert r["n_columns"] == 46
    assert r["distinct_feature_rows"] == 28006


def test_mixed_order_frames_are_refused_rather_than_silently_sharing_a_cell_space():
    with pytest.raises(SystemExit, match="mixed order"):
        main(["frame-collapse", "--frame", "served", "--frame", "trigram"])


def test_unknown_frame_names_the_builtins_rather_than_failing_obscurely():
    with pytest.raises(SystemExit, match="unknown frame"):
        main(["frame-collapse", "--frame", "nosuchframe"])
    with pytest.raises(SystemExit, match="has no attribute"):
        main(["frame-collapse", "--frame", "keybo.features:no_such_featurizer"])
    with pytest.raises(SystemExit, match="cannot import"):
        main(["frame-collapse", "--frame", "keybo.nosuchmodule:x"])


def test_module_colon_callable_extension_point_diagnoses_a_frame_not_in_the_registry(capsys):
    """INVARIANT 2's real bar: a frame the command has never heard of.

    Uses a callable that exists on an importable module and is NOT one of the built-in frames, so
    this exercises the path a caller inventing a new frame would take.
    """
    payload = json.loads(
        _run(capsys, "--frame", "tests.cli.test_frame_collapse_cli:constant_frame", "--json")
    )
    r = payload["frames"]["tests.cli.test_frame_collapse_cli:constant_frame"]
    assert r["n_cells"] == 961
    assert r["distinct_feature_rows"] == 1  # everything collapses
    assert r["collapsed_cells"] == 961
    assert r["resolution"] == pytest.approx(1 / 961)


def constant_frame(_geometry, _cell):
    """A deliberately terrible frame, used by the extension-point test above."""
    return np.zeros(3)


def test_tolerance_sweep_reports_flatness_and_the_counts(capsys):
    out = _run(capsys, "--frame", "served", "--frame", "interp", "--tolerance-sweep")
    assert "FLAT across this sweep: served, interp" in out
    assert "765" in out and "378" in out
    assert "exact" in out


def test_tolerance_sweep_flags_a_rise_as_real_and_not_as_a_bug(capsys):
    """A coarse sweep on the served frame DOES rise; the output must say so and not call it a bug."""
    out = _run(
        capsys, "--frame", "served", "--tolerance-sweep", "--tols", "0", "0.5", "0.75", "1.0"
    )
    assert "a coarser tolerance produced MORE rows than a finer one" in out
    assert "0.5->0.75" in out
    assert "This is REAL, not a bug" in out
    assert "!! BUG" not in out  # the exact-count invariant is NOT violated
    assert "TOLERANCE-SENSITIVE" in out


def test_sweep_formatter_shouts_when_the_exact_count_is_exceeded():
    """The ``!! BUG`` branch is UNREACHABLE from real data, so it is tested on the formatter directly.

    ⚠ Every other assertion about this branch is ``"!! BUG" not in out``, which a hard-coded-False
    verdict satisfies trivially (mutation M20 caught exactly that in the library). Rather than fake
    coverage, the impossible state is CONSTRUCTED and fed to the formatter — the same route
    INTERPFRAME-1 took for its per-channel gauge clause, and its limitation is documented here: this
    tests the formatter's rendering of the state, not that the state can arise.
    """
    from dataclasses import replace

    from keybo.analysis.frame_collapse import frame_collapse as fc
    from keybo.cli.frame_collapse import _format_sweep
    from keybo.geometry import Geometry

    base = fc(constant_frame, Geometry(slots=((-1, 2), (1, 2))), order=1, include_space=False)
    out = _format_sweep(
        {
            "impossible": [
                replace(base, tol=0.0, distinct_feature_rows=3),
                replace(base, tol=1e-6, distinct_feature_rows=9),
            ]
        }
    )
    assert "!! BUG: exceeded the EXACT count on impossible" in out
    assert "that is impossible" in out
    # a legal sweep through the SAME formatter must not print it
    assert "!! BUG" not in _format_sweep(
        {
            "ok": [
                replace(base, tol=0.0, distinct_feature_rows=9),
                replace(base, tol=1e-6, distinct_feature_rows=3),
            ]
        }
    )


def test_floor_refuses_a_cell_space_the_surface_cannot_target():
    """Two refusals that prevent a floor against a mis-aligned target table."""
    with pytest.raises(SystemExit, match="incompatible with --no-space"):
        main(["frame-collapse", "--frame", "served", "--floor", "--no-space"])
    with pytest.raises(SystemExit, match="needs the cell space the surface is built on"):
        main(["frame-collapse", "--frame", "served", "--floor", "--geometry", "k31"])
