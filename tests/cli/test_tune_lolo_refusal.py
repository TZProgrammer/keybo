"""`keybo tune --objective lolo` must refuse BEFORE it writes ``--output``.

A params file is indistinguishable from a real one once it is on disk, and this command's own
final line calls it "Best hyperparameters" — so a refusal that happened *after* the write would
leave behind a recommendation nobody selected, with a shell exit code as the only trace. Both
refusals it can raise are un-actionable in exactly that way:

* ``ObjectiveNotEvaluated`` — no fold produced a finite rho/ceiling, so every candidate tied at
  ``-inf`` and the tau gate alone picked the winner. On the shipped community strokes this is the
  DEFAULT experience (every layout has one participant, so ``split_half_ceiling`` bisects nothing).
* ``MarginTooSmall`` — the top two candidates are closer than the scoring rule can resolve.

Both are caught by one ``except`` that raises ``SystemExit`` upstream of the ``open(...)`` — added
in ledger ``824039e`` (the minimum-margin rule) and therefore NOT part of ``cb907aa``'s own diff;
it is the CLI half of the same guard family, reached through the ``--objective lolo`` path
``cb907aa``'s tau-gate change lives in. Nothing pinned the write ordering: the library-level
refusals are covered by ``tests/training/test_tune_unevaluated_objective.py``, but no test drove
the CLI and asserted the FILE. That is the assertion that distinguishes "refused" from "refused,
and left the artifact".

These tests run the real end-to-end command (load → LOLO harness → gates → write), which is what
makes them worth the ~7s each: a mocked ``tune_lolo`` would not exercise the write path at all.
The fixture is a lawful frame — durations linear in key distance, single WPM bucket so every
held-out layout yields cells at the CLI's hardwired ``min_cell_samples=10`` — with the PARTICIPANT
count as the only knob, because that is what decides whether the ceiling (and so the objective) is
evaluable at all.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from keybo.cli.__main__ import main

from ..training.test_validate import _POSITIONS, _distance


def _write_strokes(path, *, n_participants: int, samples_per_participant: int, seed: int = 3):
    """A geometry-lawful bistrokes TSV with a controllable participant count.

    WPM is drawn from a narrow 82-88 band on purpose: the CLI exposes no ``--min-cell-samples``
    and ``validate`` refuses a holdout with no cells at its floor of 10, so the samples have to
    concentrate in ONE bucket rather than spread across the default band.
    """
    rng = np.random.default_rng(seed)
    lines = []
    for layout, ngrams in _POSITIONS.items():
        for ngram, positions in ngrams.items():
            samples = []
            for pid in range(1, n_participants + 1):
                for _ in range(samples_per_participant):
                    wpm = int(rng.integers(82, 88))
                    duration = int(60 + 25 * _distance(positions) + rng.normal(0, 4))
                    samples.append(f"({wpm}, {duration}, {pid}, 50)")
            lines.append(f"{layout}\t{positions}\t{ngram}\t100\t" + "\t".join(samples))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def unevaluable_strokes(tmp_path_factory):
    """One participant per layout — the ceiling is nan, so the objective cannot be evaluated."""
    path = tmp_path_factory.mktemp("lolo_unevaluable") / "bi.tsv"
    return _write_strokes(path, n_participants=1, samples_per_participant=14)


@pytest.fixture(scope="module")
def healthy_strokes(tmp_path_factory):
    """Six participants per layout — the objective IS evaluable, so the MARGIN gate is what bites."""
    path = tmp_path_factory.mktemp("lolo_healthy") / "bi.tsv"
    return _write_strokes(path, n_participants=6, samples_per_participant=6)


def _argv(strokes, output, *extra):
    return [
        "tune",
        "--strokes",
        strokes,
        "--ngram",
        "bigram",
        "--output",
        str(output),
        "--min-samples",
        "1",
        "--objective",
        "lolo",
        "--n-iter",
        "2",
        "--lolo-seeds",
        "0",
        *extra,
    ]


def _run(argv):
    """Run the command with warnings silenced; saturation warnings are not what these test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main(argv)


# --- ObjectiveNotEvaluated: no output file may survive the refusal -----------------------


def test_an_unevaluated_objective_refuses_and_writes_NO_output_file(
    unevaluable_strokes, tmp_path
) -> None:
    """The core contract. The assertion that matters is the LAST one: the file does not exist.

    Without it, moving the ``except`` clause below the ``open(...)`` would still pass a test that
    only checked for a nonzero exit — and the leftover file is the whole problem, because it reads
    exactly like a real recommendation.
    """
    output = tmp_path / "best_hyperparams.json"
    with pytest.raises(SystemExit) as exc:
        _run(_argv(unevaluable_strokes, output))
    assert "never evaluated" in str(exc.value)
    assert not output.exists(), "a params file from an unevaluated objective must NOT be written"


def test_the_refusal_message_names_the_command_and_stays_actionable(unevaluable_strokes, tmp_path):
    """A bare traceback would leave the operator guessing which stage refused and why.

    The message must carry the command (so a CI log line is attributable), the objective's own
    vocabulary, and the diagnosis naming participants — the fixable cause.
    """
    output = tmp_path / "best.json"
    with pytest.raises(SystemExit) as exc:
        _run(_argv(unevaluable_strokes, output))
    message = str(exc.value)
    assert "keybo tune --objective lolo" in message, "must name the command that refused"
    assert "rho/ceiling" in message, "must name the objective that was not evaluated"
    assert "PARTICIPANTS" in message, "must name the fixable cause, not just the symptom"
    assert not output.exists()


def test_the_refusal_leaves_NO_partial_file_behind_not_even_an_empty_one(
    unevaluable_strokes, tmp_path
) -> None:
    """Distinguishes "not written" from "opened then abandoned".

    ``open(path, "w")`` truncates on open, so a refusal raised between the open and the
    ``json.dump`` would leave a zero-byte file — which is worse than no file: it looks like a
    completed run whose contents were lost, and a downstream reader gets a JSON parse error rather
    than a clear absence.
    """
    output = tmp_path / "nested" / "best.json"
    with pytest.raises(SystemExit):
        _run(_argv(unevaluable_strokes, output))
    assert not output.exists()
    # the parent dir IS created up front by ensure_writable_output (fail-fast), and that is fine
    # — an empty directory is not mistakable for a recommendation.
    assert output.parent.is_dir(), "the fail-fast path check still runs first"
    assert list(output.parent.iterdir()) == []


def test_the_refusal_can_be_downgraded_and_THEN_the_file_is_written(
    unevaluable_strokes, tmp_path
) -> None:
    """The escape hatch must be the ONLY way to get a file out of an unevaluated objective.

    This is the other half of the contract: if the default refusal were removed, this test would
    still pass while the one above failed — so the pair is what pins "refuse unless asked".
    ``--min-margin 0`` is needed too, because with every candidate at ``-inf`` the margin gate has
    no finite pair to compare and must not be what decides this test.
    """
    output = tmp_path / "best.json"
    assert (
        _run(
            _argv(
                unevaluable_strokes,
                output,
                "--allow-unevaluated-objective",
                "--min-margin",
                "0",
            )
        )
        == 0
    )
    assert output.exists(), "the explicit downgrade must actually produce the file"
    assert isinstance(json.loads(output.read_text()), dict)


# --- MarginTooSmall: the same contract, on the other refusal ----------------------------


def test_a_selection_inside_the_resolvable_margin_refuses_and_writes_NO_output(
    healthy_strokes, tmp_path
) -> None:
    """The second exception in the same ``except`` clause, driven end to end.

    On this lawful fixture the two candidates genuinely tie (both reach the ceiling), so the
    margin gate fires on real data rather than a mock — a near-tie is the NORMAL outcome of a
    small sweep, not a contrived one.
    """
    output = tmp_path / "best.json"
    with pytest.raises(SystemExit) as exc:
        _run(_argv(healthy_strokes, output))
    message = str(exc.value)
    assert "keybo tune --objective lolo" in message
    assert "lolo hyperparameter selection" in message
    assert "margin" in message
    assert not output.exists(), "a params file from an unresolvable margin must NOT be written"


def test_disabling_the_margin_gate_writes_the_file_and_exits_zero(
    healthy_strokes, tmp_path
) -> None:
    """Proves the refusal above is the margin gate and not the fixture being unusable.

    Without this, a fixture that failed for some unrelated reason would make the refusal test
    pass vacuously — the classic way a "the guard fired" test stops testing the guard.
    """
    output = tmp_path / "best.json"
    assert _run(_argv(healthy_strokes, output, "--min-margin", "0")) == 0
    params = json.loads(output.read_text())
    assert {"n_estimators", "max_depth", "learning_rate"} <= set(params)


def test_downgrading_the_margin_refusal_also_writes_the_file(healthy_strokes, tmp_path) -> None:
    """``--allow-unresolvable-margin`` must warn-and-proceed, not silently keep refusing.

    A flag documented as a downgrade that still refused would be a worse failure than no flag: the
    operator would conclude the data was at fault.
    """
    output = tmp_path / "best.json"
    assert _run(_argv(healthy_strokes, output, "--allow-unresolvable-margin")) == 0
    assert output.exists()


def test_an_EXISTING_output_file_is_not_clobbered_by_a_refusal(healthy_strokes, tmp_path) -> None:
    """The worst version of a post-write refusal: destroying a GOOD params file with a bad run.

    A rerun that refuses must leave the previous, legitimately-selected file intact. Since the
    write is truncating, this is only true while the refusal stays upstream of the ``open`` — the
    same invariant as above, but with a consequence that is not recoverable by rerunning.
    """
    output = tmp_path / "best.json"
    previous = {"n_estimators": 111, "max_depth": 3, "note": "a real earlier selection"}
    output.write_text(json.dumps(previous))
    with pytest.raises(SystemExit):
        _run(_argv(healthy_strokes, output))
    assert json.loads(output.read_text()) == previous, "a refused rerun destroyed a good file"
