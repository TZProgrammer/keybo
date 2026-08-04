"""Tests for the per-feature layout-pair attribution (SHAPDIFF-1).

The point of this file is the IDENTITIES, not coverage of the call graph. A ranked feature
table that does not sum to the gap it claims to decompose is worse than no table, so every
identity gets its own named test with the residual asserted as a NUMBER:

* ``sum_i attrib_i == ms_B - ms_A`` per bigram cell (the LMDI identity);
* ``sum_i contribution_i == gap_t2`` after corpus weighting (the headline reconciliation);
* ``gap_t2 + gap_tcond == gap_total`` (the channel split, i.e. the over-claim guard);
* this module's gap == the shipped ``TimeSurface.card`` gap (the tie to what ``analyze`` prints).

And the two NEGATIVE CONTROLS, which are the tests that give the identities their meaning: an
identity that also holds when the weighting is wrong, or when the attribution is destroyed by
shuffling, is arithmetic rather than evidence. Both are asserted to FAIL.

The real k31 artifacts are used deliberately (they are vendored, ~50 ms to load and cached
per session): a synthetic stand-in would test the algebra while leaving the actual claim —
that this decomposition reconciles to the SHIPPED gauge — unchecked.
"""

import json

import numpy as np
import pytest

from keybo.analysis.shap_diff import WEIGHTINGS, format_report, shap_diff
from keybo.analysis.timecard import default_surface
from keybo.cli.__main__ import main
from keybo.features.schema import BIGRAM_FEATURE_NAMES

FLAGSHIP_C3 = "pyou'vgdnmheai.cstrlkjz,-wfbxq"
GRAPHITE = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"

#: Bars from the SHAPDIFF-1 registration. The ms-space identities are exact algebra (float64
#: rounding); the log-space cross-checks compare two independent xgboost code paths on a
#: float32 booster and cannot beat ~1e-6.
REL_TOL = 1e-9
LOG_TOL = 1e-5


@pytest.fixture(scope="module")
def diff():
    """The flagship-c3 -> graphite decomposition on the production corpus."""
    return shap_diff(FLAGSHIP_C3, GRAPHITE, name_a="flagship-c3", name_b="graphite")


def test_cell_lmdi_identity_is_exact(diff):
    """Per bigram cell, the feature attributions sum to that cell's exact ms difference."""
    assert diff.resid_cell_lmdi <= REL_TOL, f"per-cell LMDI residual {diff.resid_cell_lmdi:.3e}"


def test_weighted_contributions_sum_to_the_t2_gap(diff):
    """THE headline reconciliation: the ranked table sums to the channel it decomposes."""
    assert diff.resid_feature_sum <= REL_TOL, f"feature-sum residual {diff.resid_feature_sum:.3e}"
    # asserted independently of the stored residual, so a bug in the residual itself is caught
    total = sum(c.ms_per_char for c in diff.contributions)
    assert total == pytest.approx(diff.gap_t2, rel=REL_TOL)


def test_channel_split_is_exact_and_the_gap_is_not_all_bigram(diff):
    """``gap_t2 + gap_tcond == gap_total`` — the guard against over-claiming.

    The second assertion is the substantive one: the bigram channel must NOT be reported as
    the whole gap. If a future model change makes ``gap_tcond`` vanish this test fails loudly
    rather than letting the report silently start claiming the full gap.
    """
    assert diff.resid_channel_split <= REL_TOL
    assert diff.gap_t2 + diff.gap_tcond == pytest.approx(diff.gap_total, rel=REL_TOL)
    assert abs(diff.gap_t2) < abs(diff.gap_total), "the T2 channel is not the whole gap"


def test_reconciles_against_the_shipped_gauge(diff):
    """This module's gap must equal ``TimeSurface.card``'s — the tie to what ``analyze`` prints."""
    surface = default_surface(90.0, None)
    card_gap = surface.card(GRAPHITE).ms_per_char - surface.card(FLAGSHIP_C3).ms_per_char
    assert diff.gap_total == pytest.approx(card_gap, abs=1e-3)
    assert diff.resid_vs_card_gap <= 1e-3
    assert diff.reconciles()


def test_treeshap_additivity_holds_in_log_space(diff):
    """The TreeSHAP walk and the ordinary prediction agree to booster precision."""
    assert diff.resid_additivity <= LOG_TOL, f"additivity {diff.resid_additivity:.3e}"
    assert diff.resid_log_vs_predict <= LOG_TOL
    # a nonzero value is expected (float32 booster); an EXACT zero would mean the check is
    # comparing something to itself, which is the degenerate-control failure mode
    assert diff.resid_additivity > 0.0


def test_equal_charsets_make_common_support_a_noop(diff):
    """These two boards are permutations of one charset, so coverage cannot confound the gap."""
    assert set(FLAGSHIP_C3) == set(GRAPHITE)
    assert diff.common_support_is_noop
    assert diff.covered_mass_a == diff.covered_mass_b == diff.covered_mass_common
    assert diff.coverage_cost == pytest.approx(0.0, abs=1e-9)
    assert diff.ms_per_char_a == pytest.approx(diff.ms_per_char_own_a, rel=REL_TOL)


def test_wpm_contribution_is_not_a_frame_mismatch(diff):
    """``wpm`` is constant across cells, so any contribution must come from SHAP, not the frame.

    Registered as a self-test: both boards are featurized at the same scoring WPM, so if the
    two matrices ever differed in that column the run would be void. The column's SHAP value
    still varies by cell (a tree can split on a constant feature's interaction path), so the
    assertion is on the FEATURE VALUES, not on the contribution being zero.
    """
    from keybo.features import bigram_features_from_positions
    from keybo.geometry import ROW_STAGGERED_30

    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]
    col = BIGRAM_FEATURE_NAMES.index("wpm")
    values = np.array(
        [
            bigram_features_from_positions(geom, (a, b), wpm=90.0)[col]
            for a in positions
            for b in positions
        ]
    )
    assert np.ptp(values) == 0.0 and values[0] == 90.0


def test_feature_names_are_real_names_not_indices(diff):
    """The output must be readable as insight: schema names, all 20, no ``f0..f19``."""
    assert [c.feature for c in diff.contributions] == list(BIGRAM_FEATURE_NAMES)


def test_sign_convention_and_favours_agree(diff):
    """``favours == "a"`` iff the contribution makes B slower — the report's whole sign story."""
    for c in diff.contributions:
        expected = "a" if c.ms_per_char > 0 else ("b" if c.ms_per_char < 0 else "tie")
        assert c.favours == expected


def test_top_bigrams_partition_their_feature(diff):
    """Per-bigram detail must be a partition of the feature's contribution, not a re-scoring."""
    leader = diff.ranked()[0]
    everything = diff.top_bigrams(leader.feature, k=10**6)
    assert sum(v for _, v in everything) == pytest.approx(leader.ms_per_char, rel=1e-9)
    top = diff.top_bigrams(leader.feature, k=5)
    assert len(top) == 5
    assert [abs(v) for _, v in top] == sorted((abs(v) for _, v in top), reverse=True)


# --- the negative controls: these are what make the identities above mean something --------


def test_control_bigram_table_weighting_is_caught_by_the_gauge_tie_not_by_r3():
    """Weighting by ``bigrams.txt`` instead of the trigram marginal MUST break reconciliation —
    and it is **R5**, the tie to the shipped gauge, that catches it.

    This test pins down which bar does the work, because the obvious guess is wrong and the
    distinction matters. R3 (``sum_i contribution_i == gap_t2``) is INVARIANT to the weighting:
    both sides are built from the same weight table, so a wrong weight yields a
    self-consistent decomposition of the WRONG quantity — R3 stays at ~1e-16 and would license
    a bogus table. What actually detects the error is R5: ``gap_total`` moves from +3.1934 to
    +3.2491, i.e. 5.6e-2 ms/char away from what ``TimeSurface.card`` reports.

    So an implementation carrying only the self-consistency bar would pass this control and be
    unpinned from the gauge entirely. That is why :meth:`ShapDiff.reconciles` includes R5.
    """
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    freqs = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, weighting="bigram-table", control_bigram_freqs=freqs)
    assert not control.reconciles()
    # the bar that fires
    assert control.resid_vs_card_gap > 1e-3
    # and the bar that does NOT: self-consistency survives a wrong weighting
    assert control.resid_feature_sum <= REL_TOL


def test_control_shuffled_shap_is_caught_by_the_additivity_bars():
    """Permuting the per-cell SHAP deltas MUST break the identity.

    This is the control that distinguishes an ATTRIBUTION from arithmetic: if the totals still
    reconciled after the per-cell vectors were shuffled between cells, the reconciliation would
    be testing addition rather than which feature did what. Here it is R2/R3 that fire (the
    shuffled deltas no longer sum to each cell's own ms change) while R5 is untouched — the
    mirror image of the weighting control, which is why BOTH controls are needed: neither bar
    alone covers both failure modes.
    """
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, shuffle_seed=0)
    assert not control.reconciles()
    assert control.resid_cell_lmdi > REL_TOL
    assert control.resid_feature_sum > REL_TOL
    # the gap itself is unchanged: shuffling redistributes attribution, not the total
    assert control.resid_vs_card_gap <= 1e-3


def test_bigram_table_weighting_requires_the_control_table():
    with pytest.raises(ValueError, match="negative control"):
        shap_diff(FLAGSHIP_C3, GRAPHITE, weighting="bigram-table")


def test_unknown_weighting_is_refused():
    with pytest.raises(ValueError, match="weighting must be one of"):
        shap_diff(FLAGSHIP_C3, GRAPHITE, weighting="bigrams")
    assert WEIGHTINGS == ("trigram-marginal", "bigram-table")


# --- generality + guards ------------------------------------------------------------------


def test_works_for_an_unrelated_pair():
    """The tool is for ANY pair, not the one it was written for."""
    other = shap_diff(QWERTY30M, GRAPHITE, name_a="qwerty30m", name_b="graphite")
    assert other.reconciles()
    assert other.gap_total < 0, "graphite should be faster than qwerty on this surface"
    assert sum(c.ms_per_char for c in other.contributions) == pytest.approx(
        other.gap_t2, rel=REL_TOL
    )


def test_short_or_repeating_layout_is_refused():
    """Reuses ``TimeSurface._slot_of``'s guard: a partial layout would score a corpus fraction."""
    with pytest.raises(ValueError, match="DISTINCT characters"):
        shap_diff("qwerty", GRAPHITE)


def test_calibrated_model_is_refused():
    """A first-finger-calibrated model must be REFUSED, not silently mis-attributed.

    The deltas are a per-POSITION multiplicative factor applied outside the 20-column feature
    path, so SHAP contributions could not sum to the served table. The three shipped
    ``bigram_reg31`` artifacts carry none; this asserts the guard rather than the luck.
    """
    from keybo.analysis.shap_diff import _bigram_shap_tables
    from keybo.analysis.timecard import _SEEDS, _load_gz_model
    from keybo.geometry import ROW_STAGGERED_30

    models = [_load_gz_model(f"bigram_reg31_seed{s}") for s in _SEEDS]
    for model in models:
        assert not (model.metadata.extra["training"].get("calibration") or {}).get("deltas_ms")

    victim = models[0]
    original = victim.metadata.extra["training"].get("calibration")
    victim.metadata.extra["training"]["calibration"] = {"deltas_ms": {"pinky_first": [1.0, 2.0]}}
    try:
        with pytest.raises(NotImplementedError, match="first-finger calibration"):
            _bigram_shap_tables([victim], ROW_STAGGERED_30, 90.0)
    finally:
        victim.metadata.extra["training"]["calibration"] = original


def test_report_puts_reconciliation_before_the_table(diff):
    """A reader must not meet the feature ranking before the residuals that license it."""
    text = format_report(diff)
    assert text.index("RECONCILIATION") < text.index("PER-FEATURE CONTRIBUTIONS")
    assert "RECONCILES: True" in text
    assert "Tcond trigram channel" in text
    assert "bottom" in text


def test_report_flags_a_failed_reconciliation():
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, shuffle_seed=1)
    text = format_report(control)
    assert "RECONCILIATION FAILED" in text
    assert text.index("RECONCILIATION FAILED") < text.index("PER-FEATURE CONTRIBUTIONS")


# --- CLI ----------------------------------------------------------------------------------


def test_cli_end_to_end_writes_json(tmp_path, capsys):
    out = tmp_path / "diff.json"
    rc = main(["shap-diff", "flagship-c3", "graphite", "--json", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "RECONCILES: True" in printed

    payload = json.loads(out.read_text())
    assert payload["residuals"]["reconciles"] is True
    assert payload["gap"]["sign_convention"] == "positive = layout_a is faster"
    contributions = payload["contributions"]
    assert len(contributions) == len(BIGRAM_FEATURE_NAMES)
    assert sum(c["ms_per_char"] for c in contributions) == pytest.approx(
        payload["gap"]["t2_bigram_channel"], rel=REL_TOL
    )
    # the channel split must be present and honest in the artifact, not only in stdout
    assert payload["gap"]["t2_bigram_channel"] + payload["gap"]["tcond_trigram_channel"] == (
        pytest.approx(payload["gap"]["total"], rel=REL_TOL)
    )
    assert payload["coverage"]["common_support_is_noop"] is True


def test_to_dict_is_json_serializable_with_no_numpy_scalars(diff):
    """The artifact must round-trip through ``json``: numpy scalars are not serializable.

    Several residuals are computed with numpy, so ``reconciles()``'s ``and`` chain returned an
    ``np.bool_`` — falsy-correct, silently un-serializable, and it only surfaced when the CLI
    tried to write the file. Asserted here at the dataclass level so the guard does not depend
    on the CLI test running.
    """
    payload = diff.to_dict()
    assert isinstance(payload["residuals"]["reconciles"], bool)
    json.loads(json.dumps(payload))  # raises TypeError on any numpy scalar left in the tree


def test_cli_control_exits_zero_only_when_the_control_fails(capsys):
    """The control's expectation is machine-checked: rc=0 means it correctly FAILED."""
    rc = main(["shap-diff", "flagship-c3", "graphite", "--control", "shuffle"])
    assert rc == 0
    assert "failed reconciliation, as required" in capsys.readouterr().out


def test_cli_refuses_identical_layouts(capsys):
    rc = main(["shap-diff", "graphite", "graphite"])
    assert rc == 1
    assert "same layout" in capsys.readouterr().out
