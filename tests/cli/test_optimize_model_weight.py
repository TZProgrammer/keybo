"""`keybo optimize --model-weight` — the normalized-gauge objective wired into the real CLI.

These are the tests that would catch the objective being bolted on as a driver-only path: they
drive `keybo.cli.optimize.run` itself, so the argument parsing, the gates, the search loop, the
postflight and the result file are all the shipped ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from keybo.analysis import surfaces as S
from keybo.cli import optimize as O
from keybo.scoring import model_norm as MN

REPO = Path(__file__).resolve().parents[2]
ANCHORS = REPO / "drivers-normgauge" / "anchors.json"
MODEL = REPO / "data" / "models" / "k31" / "bigram_reg31_seed0.json.gz"


def _args(**overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    O.add_arguments(parser)
    base = [
        "--model",
        str(MODEL),
        "--start",
        S.C30M,
        "--max-outer",
        "3",
        "--no-progress",
        "--no-local-search",
        "--seed",
        "1",
    ]
    args = parser.parse_args(base)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# argument parsing: refuse the ambiguous cases rather than guessing
# ---------------------------------------------------------------------------
def test_gauge_names_map_to_pools():
    assert O._parse_model_weights(["aalto-n=1", "comm-n=2", "pool-n=3"]) == {
        "AALTO": 1.0,
        "COMMUNITY": 2.0,
        "POOL": 3.0,
    }


def test_unknown_gauge_is_refused_not_ignored():
    """A typo'd gauge name must not silently leave that gauge at zero weight."""
    with pytest.raises(SystemExit, match="unknown --model-weight gauge"):
        O._parse_model_weights(["aalto=1"])


def test_repeated_gauge_is_refused_rather_than_last_wins():
    with pytest.raises(SystemExit, match="given more than once"):
        O._parse_model_weights(["aalto-n=1", "aalto-n=2"])


def test_non_numeric_and_missing_values_are_refused():
    with pytest.raises(SystemExit, match="is not a number"):
        O._parse_model_weights(["aalto-n=high"])
    with pytest.raises(SystemExit, match="expected GAUGE=W"):
        O._parse_model_weights(["aalto-n"])


def test_model_weight_requires_anchors():
    """Without anchors there is no scale, so a weight has no meaning — refuse, don't default."""
    with pytest.raises(SystemExit, match="requires --model-anchors"):
        O.run(_args(model_weight=["aalto-n=1"], model_anchors=None))


def test_model_weight_refuses_a_non_c30m_start():
    if not ANCHORS.exists():
        pytest.skip("anchors.json has not been built in this checkout")
    with pytest.raises(SystemExit, match="C30M start layout"):
        O.run(
            _args(
                model_weight=["aalto-n=1"],
                model_anchors=str(ANCHORS),
                start="qwertyuiopasdfghjkl;zxcvbnm,./",
            )
        )


def test_model_weight_refuses_to_be_mixed_with_ms_equivalent_terms():
    """comfort/oxey/finger-load are ms-equivalent sums; adding them to a 0-1 blend is nonsense."""
    with pytest.raises(SystemExit, match="cannot be combined"):
        O.run(
            _args(
                model_weight=["aalto-n=1"],
                model_anchors=str(ANCHORS),
                comfort_weight=1.0,
            )
        )


# ---------------------------------------------------------------------------
# end to end through the shipped CLI
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_optimize_runs_the_blend_and_records_its_objective(tmp_path, capsys):
    if not ANCHORS.exists():
        pytest.skip("anchors.json has not been built in this checkout")
    if not MODEL.exists():
        pytest.skip("the k31 bigram model is not vendored in this checkout")
    out = tmp_path / "result.json"
    rc = O.run(
        _args(
            model_weight=["aalto-n=0.5411", "comm-n=0.3977", "pool-n=0.0612"],
            model_anchors=str(ANCHORS),
            out=str(out),
        )
    )
    assert rc == 0
    printed = capsys.readouterr().out
    # The report must carry the frame caveat and the interpretation, not just a number.
    assert "normalized model blend" in printed
    assert "union" in printed  # the POOL-is-a-union caveat
    assert "NOT near 0" in printed  # the qwerty trap, stated

    result = json.loads(out.read_text())
    assert result["objective"] == "normalized model blend"
    assert result["model_weights"] == {"AALTO": 0.5411, "COMMUNITY": 0.3977, "POOL": 0.0612}
    assert result["model_anchors"] == str(ANCHORS)
    assert set(result["normalized_gauges"]) == set(MN.GAUGE_NAMES)
    # fitness is the NEGATED blend: the optimizer minimizes, the gauge is higher-is-better.
    assert result["fitness"] == pytest.approx(-result["blend_higher_is_better"])
    assert S.is_c30m(result["layout"])


@pytest.mark.slow
def test_the_search_actually_improves_the_blend(capsys):
    """A search that returned its start layout would pass every test above and be useless.

    ⚠ Runs WITH the 2-opt polish, and that is load-bearing rather than incidental: measured on
    this objective, `--no-local-search` at max_outer 60 AND 300 both return 0.523429 (barely
    above qwerty's 0.522878), while enabling the polish reaches 0.941646. On the normalized
    blend the 2-opt pass does nearly all the work, so a version of this test that disabled it
    would have "proved" the objective was not being optimized when it was.
    """
    if not ANCHORS.exists() or not MODEL.exists():
        pytest.skip("anchors.json or the k31 model is not available in this checkout")
    anchors = MN.Anchors.read(ANCHORS)
    fits = MN.SurfaceFits()
    spec = MN.BlendSpec(weights={"AALTO": 0.5411, "COMMUNITY": 0.3977, "POOL": 0.0612})
    start_blend = spec.blend(anchors.normalize_many(fits.fit_of(S.C30M)))

    rc = O.run(
        _args(
            model_weight=["aalto-n=0.5411", "comm-n=0.3977", "pool-n=0.0612"],
            model_anchors=str(ANCHORS),
            max_outer=60,
            no_local_search=False,
        )
    )
    assert rc == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("blend (higher"))
    found = float(line.split(":")[1])
    assert found > start_blend + 0.2, (
        f"the search returned a blend of {found:.6f} from a start of {start_blend:.6f} — it is "
        f"not actually optimizing the objective"
    )
