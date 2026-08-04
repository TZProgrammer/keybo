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


# =========================================================================================
# SHAPDIFF-TCOND: the conditioned-trigram channel.
#
# Same discipline as above — the IDENTITIES as NUMBERS, and the controls that give them
# meaning. Two things are new and are what these tests exist for:
#
# 1. The Tcond channel's correct weight is the trigram frequency DIRECTLY, not the
#    first-two-character marginal the T2 channel needs. The `tcond-marginal` control commits
#    exactly the symmetry error ("T2 needed a marginal, so Tcond needs one too") and MUST fail.
# 2. The primary reporting unit is a BLOCK. `bg1_*`/`bg2_*` are the same 19 placement features
#    on two overlapping key pairs, so a per-column split is not unique — the block sums are the
#    claim that survives, and they are asserted to partition the channel gap exactly.
# =========================================================================================

from keybo.analysis.shap_diff import TCOND_WEIGHTINGS, block_map  # noqa: E402
from keybo.features.schema import TRIGRAM_FEATURE_NAMES  # noqa: E402

#: The Tcond channel's own external anchor, measured from the SHIPPED `_Tc` table before the
#: decomposition existed (SHAPDIFF-TCOND prereg §0 F4) and registered there as the bar.
SHIPPED_GAP_TCOND = 2.195340
GAUGE_TOL = 1e-3


@pytest.fixture(scope="module")
def both():
    """flagship-c3 -> graphite with BOTH channels decomposed (the default)."""
    return shap_diff(FLAGSHIP_C3, GRAPHITE, name_a="flagship-c3", name_b="graphite")


def test_tcond_cell_lmdi_identity_is_exact(both):
    """Per TRIGRAM cell, the 46 feature attributions sum to that cell's exact ms difference."""
    assert both.tcond.resid_cell_lmdi <= REL_TOL, (
        f"per-cell LMDI residual {both.tcond.resid_cell_lmdi:.3e}"
    )


def test_tcond_contributions_sum_to_the_tcond_gap(both):
    """THE headline reconciliation for this channel, asserted twice over.

    The stored residual AND an independent re-summation, so a bug in the residual computation
    itself cannot pass.
    """
    tcond = both.tcond
    assert tcond.resid_feature_sum <= REL_TOL, f"feature-sum residual {tcond.resid_feature_sum:.3e}"
    assert sum(c.ms_per_char for c in tcond.contributions) == pytest.approx(tcond.gap, rel=REL_TOL)
    assert len(tcond.contributions) == 46


def test_tcond_gap_ties_to_the_shipped_trigram_table(both):
    """The EXTERNAL bar: the decomposed gap must equal the shipped ``_Tc`` contraction.

    This is the bar the wrong-weighting control breaks and the internal bars cannot see. The
    literal 2.195340 is the number registered in the prereg BEFORE the decomposition existed,
    so this test would fail if the channel silently started decomposing something else.
    """
    assert both.tcond.resid_gap_vs_shipped <= GAUGE_TOL
    assert both.tcond.gap == pytest.approx(SHIPPED_GAP_TCOND, abs=GAUGE_TOL)
    assert both.tcond.shipped_gap == pytest.approx(SHIPPED_GAP_TCOND, abs=GAUGE_TOL)


def test_both_channels_together_decompose_the_whole_gap(both):
    """The point of the arm: T2 + Tcond covers 100% of the gap, with nothing left over.

    SHAPDIFF-1 could only reach 31.3% because a per-bigram frame is structurally blind to the
    conditioned-trigram increment.
    """
    assert both.t2 is not None and both.tcond is not None
    assert both.decomposed_share_pct == pytest.approx(100.0, abs=1e-6)
    assert both.undecomposed_ms_per_char == pytest.approx(0.0, abs=1e-9)
    assert both.t2.gap + both.tcond.gap == pytest.approx(both.gap_total, rel=REL_TOL)
    # and the tie to what `analyze` prints survives adding the second channel
    assert both.resid_vs_card_gap <= GAUGE_TOL
    assert both.reconciles()


def test_tcond_is_the_majority_channel_on_this_pair(both):
    """The substantive split, pinned as a number: Tcond carries ~68.7% and T2 ~31.3%.

    Asserted as an ORDERING plus loose bounds rather than exact values, so a model refit moves
    it without failing — but a change that silently reassigned the majority to the bigram
    channel (or collapsed one channel) fails loudly.
    """
    assert both.gap_tcond > both.gap_t2 > 0
    assert 60.0 < 100.0 * both.gap_tcond / both.gap_total < 75.0
    assert both.gap_t2 == pytest.approx(0.9981, abs=1e-3)
    assert both.gap_tcond == pytest.approx(2.1953, abs=1e-3)


def test_blocks_partition_the_channel_gap_exactly(both):
    """BLOCKS are the primary table, so they must be an exact partition — of BOTH channels.

    A block sum is invariant to how SHAP split credit among correlated columns *within* the
    block, which is the whole reason it is reported first. That guarantee is worthless if the
    blocks do not cover every column exactly once.
    """
    for channel in (both.t2, both.tcond):
        blocks = channel.blocks()
        covered = [col for b in blocks for col in b.columns]
        assert sorted(covered) == sorted(channel.feature_names), "blocks must partition the frame"
        assert len(covered) == len(set(covered)), "no column may appear in two blocks"
        assert sum(b.ms_per_char for b in blocks) == pytest.approx(channel.gap, rel=REL_TOL)
        for b in blocks:
            # `parts` is a partition of its own block
            assert sum(v for _, v in b.parts) == pytest.approx(b.ms_per_char, rel=1e-9)


def test_block_map_covers_the_served_trigram_frame_and_refuses_others():
    """Every served column has a registered block, and an unknown frame is REFUSED.

    Refusing matters: bucketing unknown columns into an implicit remainder would leave the
    primary table silently incomplete while every identity still closed.
    """
    spec = block_map(TRIGRAM_FEATURE_NAMES)
    assert set(spec) == set(TRIGRAM_FEATURE_NAMES)
    assert {b for b, _ in spec.values()} == {"TRI_LEVEL", "SKIPGRAM", "BG1", "BG2", "WPM"}
    assert spec["redirect"][0] == "TRI_LEVEL"
    assert spec["sg_distance"][0] == "SKIPGRAM"
    assert spec["bg1_bottom"] == ("BG1", "row")
    assert spec["bg2_dx"] == ("BG2", "geometry")
    assert len([n for n, (b, _) in spec.items() if b == "BG1"]) == 19
    assert len([n for n, (b, _) in spec.items() if b == "BG2"]) == 19
    with pytest.raises(ValueError, match="no block partition registered"):
        block_map(["some", "unknown", "frame"])


def test_bg1_and_bg2_are_the_two_constituent_transitions(both):
    """``bg1_*`` is the (a,b) transition and ``bg2_*`` the (b,c) one — asserted, not assumed.

    This is load-bearing for the report's wording, and the natural misreading is that ``bg2_``
    re-describes the FIRST bigram. Since a bigram's placement one-hots describe the SECOND key
    of its pair, ``bg1_bottom`` is the trigram's MIDDLE key and ``bg2_bottom`` its THIRD key —
    so a "bg2_bottom" finding is a statement about where the trigram LANDS, and the two blocks
    are not interchangeable. Byte-compared against the bigram frame rather than read off a
    docstring.
    """
    from keybo.features import bigram_features_from_positions, trigram_features_from_positions
    from keybo.geometry import ROW_STAGGERED_30

    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]
    shared = [c for c in BIGRAM_FEATURE_NAMES if c != "wpm"]
    rng = np.random.default_rng(0)
    for _ in range(40):
        i, j, k = (int(x) for x in rng.integers(0, len(positions), 3))
        a, b, c = positions[i], positions[j], positions[k]
        tri = trigram_features_from_positions(geom, (a, b, c), wpm=90.0)
        for prefix, pair in (("bg1", (a, b)), ("bg2", (b, c))):
            got = np.array([tri[TRIGRAM_FEATURE_NAMES.index(f"{prefix}_{n}")] for n in shared])
            want = bigram_features_from_positions(geom, pair, wpm=90.0)
            want = np.array([want[BIGRAM_FEATURE_NAMES.index(n)] for n in shared])
            assert np.array_equal(got, want), f"{prefix}_* is not the {pair} transition"


def test_tcond_feature_names_are_the_served_trigram_frame(both):
    """46 real schema names, in order — not ``f0..f45``, and not the bigram frame."""
    assert [c.feature for c in both.tcond.contributions] == list(TRIGRAM_FEATURE_NAMES)
    assert len(TRIGRAM_FEATURE_NAMES) == 46


def test_top_ngrams_are_trigrams_and_partition_their_feature(both):
    """Per-cell detail must be a partition of the column's contribution, and be TRIGRAMS."""
    leader = both.tcond.ranked()[0]
    everything = both.tcond.top_ngrams(leader.feature, k=10**7)
    assert sum(v for _, v in everything) == pytest.approx(leader.ms_per_char, rel=1e-9)
    top = both.tcond.top_ngrams(leader.feature, k=5)
    assert len(top) == 5
    assert all(len(ng) == 3 for ng, _ in top), "the Tcond channel's n-grams are TRIGRAMS"
    assert [abs(v) for _, v in top] == sorted((abs(v) for _, v in top), reverse=True)
    assert both.tcond.order == 3 and both.t2.order == 2


# --- the Tcond negative controls, with the which-bar-fires pairing ------------------------


def test_control_tcond_marginal_weighting_is_caught_by_the_external_tie_only():
    """Weighting Tcond by the first-two-char MARGINAL must break the EXTERNAL bar only.

    This is the trap a symmetry-seeking implementer falls into: SHAPDIFF-1 established that the
    T2 channel's weight is the trigram table's first-two-character marginal, and the natural
    inference is that Tcond needs one too. It does NOT — ``card()`` indexes ``Tcond`` by all
    three looped characters, so its weight is the trigram frequency directly.

    The pairing asserted here is the evidence the two bar families are not redundant: the
    INTERNAL sums-back bar still reads ~1e-14 under the wrong weight (both sides share the
    weight table, so it self-consistently decomposes the WRONG quantity), while the EXTERNAL
    tie to the shipped table fires at ~1.6 ms/char — three orders of magnitude past the bar.
    And it is not a cosmetic difference: under this weighting the answer INVERTS (BG1 becomes
    the leading block and BG2 goes negative).
    """
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="tcond", tcond_weighting="tcond-marginal")
    assert not control.reconciles()
    # the bar that FIRES
    assert control.tcond.resid_gap_vs_shipped > 1.0
    assert control.tcond.gap == pytest.approx(0.6210, abs=1e-3)  # not 2.1953
    # the bars that do NOT: self-consistency survives a wrong weighting
    assert control.tcond.resid_cell_lmdi <= REL_TOL
    assert control.tcond.resid_feature_sum <= REL_TOL
    # the external reference itself must NOT move with the control, or the tie is vacuous
    assert control.tcond.shipped_gap == pytest.approx(SHIPPED_GAP_TCOND, abs=GAUGE_TOL)


def test_control_shuffled_tcond_shap_is_caught_by_the_internal_bars_only():
    """Permuting the per-cell SHAP deltas must break the INTERNAL bars only — the mirror image.

    Together with the weighting control this is what proves neither family alone suffices:
    here the attribution is destroyed while the channel gap is untouched, so the external tie
    stays at ~1e-7 and only the internal identities fire.
    """
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="tcond", shuffle_seed=0)
    assert not control.reconciles()
    # the bars that FIRE
    assert control.tcond.resid_cell_lmdi > REL_TOL
    assert control.tcond.resid_feature_sum > REL_TOL
    # the bar that does NOT: shuffling redistributes attribution, not the total
    assert control.tcond.resid_gap_vs_shipped <= GAUGE_TOL
    assert control.tcond.gap == pytest.approx(SHIPPED_GAP_TCOND, abs=GAUGE_TOL)


def test_no_residual_is_exactly_zero_where_a_nonzero_is_expected(both):
    """A residual that CANNOT be non-zero is not evidence — SHAPDIFF-1's scar, asserted.

    Its first log-space control was a tautology (``base := p - sum(shap)``, then assert
    ``base + sum(shap) == p``); it printed EXACTLY 0.000e+00 and could never fail. The
    float32-booster cross-checks compare two INDEPENDENT xgboost code paths, so they must be
    small but strictly positive in both channels.
    """
    for channel in (both.t2, both.tcond):
        assert channel.resid_additivity > 0.0, f"{channel.channel}: walk-vs-predict is exactly 0"
        assert channel.resid_log_vs_predict > 0.0
        assert channel.resid_additivity <= LOG_TOL
        assert channel.resid_table_vs_shipped > 0.0


def test_wpm_is_constant_on_the_trigram_frame_too(both):
    """``wpm`` is one column at a fixed scoring WPM, so its board-to-board DELTA is a SHAP
    interaction artifact — not a frame mismatch.

    Registered as a self-test: if the two matrices ever differed in that column the run would
    be void. Asserted on the FEATURE VALUES, not on the contribution being zero (it is not:
    the trigram channel's ``wpm`` carries -0.0273, the same non-uniqueness symptom SHAPDIFF-1
    measured at -0.0922 on the bigram frame).
    """
    from keybo.features import trigram_features_from_positions
    from keybo.geometry import ROW_STAGGERED_30

    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]
    col = TRIGRAM_FEATURE_NAMES.index("wpm")
    values = np.array(
        [
            trigram_features_from_positions(geom, (a, b, c), wpm=90.0)[col]
            for a in positions[:6]
            for b in positions[:6]
            for c in positions[:6]
        ]
    )
    assert np.ptp(values) == 0.0 and values[0] == 90.0


# --- channel selection, guards, and the SHAPDIFF-1 API -----------------------------------


def test_single_channel_run_reports_its_own_partiality():
    """A ``--channel t2`` run must NOT claim the whole gap — the over-claim guard.

    SHAPDIFF-1's number was 31.3%, and the danger is a caller reading a T2-only table as an
    explanation of the full gap. The undecomposed remainder is a named quantity for that reason.
    """
    only_t2 = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="t2")
    assert only_t2.tcond is None and only_t2.t2 is not None
    assert only_t2.decomposed_share_pct == pytest.approx(31.3, abs=0.5)
    assert only_t2.undecomposed_ms_per_char == pytest.approx(only_t2.gap_tcond, rel=REL_TOL)
    assert only_t2.reconciles()
    # gap_total is still the FULL gap, so the partiality is visible rather than hidden
    assert only_t2.gap_total == pytest.approx(3.1934, abs=1e-3)

    only_tc = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="tcond")
    assert only_tc.t2 is None and only_tc.tcond is not None
    assert only_tc.decomposed_share_pct == pytest.approx(68.7, abs=0.5)
    assert only_tc.reconciles()
    # and the SHAPDIFF-1 accessors REFUSE rather than silently returning trigram numbers
    with pytest.raises(ValueError, match="no T2 attribution"):
        _ = only_tc.contributions


def test_shapdiff1_accessors_still_read_the_t2_channel(both, diff):
    """The bigram-era API must keep meaning the BIGRAM channel, byte for byte.

    A caller written against SHAPDIFF-1 must not silently start reading the trigram frame
    because the default channel widened.
    """
    assert [c.feature for c in both.contributions] == list(BIGRAM_FEATURE_NAMES)
    assert both.resid_cell_lmdi == both.t2.resid_cell_lmdi
    assert both.resid_feature_sum == both.t2.resid_feature_sum
    assert both.weighting == "trigram-marginal"
    assert both.top_bigrams("bottom", 3) == both.t2.top_ngrams("bottom", 3)
    # and adding the second channel did not perturb the first one's numbers
    assert both.gap_t2 == pytest.approx(diff.gap_t2, rel=1e-12)
    for new, old in zip(both.contributions, diff.contributions, strict=True):
        assert new.feature == old.feature
        assert new.ms_per_char == pytest.approx(old.ms_per_char, rel=1e-12)


def test_channel_and_weighting_arguments_are_validated():
    """A control aimed at a channel this run does not decompose is REFUSED, not a silent no-op.

    ``--channel t2 --tcond_weighting tcond-marginal`` would otherwise run a "control" that
    touches nothing and then report itself as correctly failing for an unrelated reason.
    """
    assert TCOND_WEIGHTINGS == ("trigram-direct", "tcond-marginal")
    with pytest.raises(ValueError, match="channel must be one of"):
        shap_diff(FLAGSHIP_C3, GRAPHITE, channel="t3")
    with pytest.raises(ValueError, match="tcond_weighting must be one of"):
        shap_diff(FLAGSHIP_C3, GRAPHITE, tcond_weighting="trigram")
    with pytest.raises(ValueError, match="Tcond-channel control"):
        shap_diff(FLAGSHIP_C3, GRAPHITE, channel="t2", tcond_weighting="tcond-marginal")
    with pytest.raises(ValueError, match="T2-channel control"):
        shap_diff(
            FLAGSHIP_C3,
            GRAPHITE,
            channel="tcond",
            weighting="bigram-table",
            control_bigram_freqs={"th": 1},
        )


def test_tcond_works_for_an_unrelated_pair():
    """The channel is for ANY pair, not the one it was written for."""
    other = shap_diff(QWERTY30M, GRAPHITE, name_a="qwerty30m", name_b="graphite", channel="tcond")
    assert other.reconciles()
    assert other.gap_tcond < 0, "graphite should beat qwerty on the trigram channel too"
    assert sum(c.ms_per_char for c in other.tcond.contributions) == pytest.approx(
        other.tcond.gap, rel=REL_TOL
    )


def test_calibrated_trigram_model_is_refused():
    """A calibrated TRIGRAM model must be REFUSED via the shipped guard, not mis-attributed."""
    from keybo.analysis.shap_diff import _shap_tables, default_models
    from keybo.geometry import ROW_STAGGERED_30

    victim = default_models("trigram")[0]
    original = victim.metadata.extra["training"].get("calibration")
    assert not (original or {}).get("deltas_ms"), "shipped artifact must carry no deltas"
    victim.metadata.extra["training"]["calibration"] = {"deltas_ms": {"pinky_first": [1.0]}}
    try:
        with pytest.raises(NotImplementedError, match="calibration deltas"):
            _shap_tables([victim], ROW_STAGGERED_30, 90.0, 3)
    finally:
        victim.metadata.extra["training"]["calibration"] = original


def test_report_puts_blocks_before_columns(both):
    """A reader must not meet the 46-column table before the blocks that license reading it."""
    text = format_report(both)
    assert text.index("RECONCILIATION") < text.index("BLOCK CONTRIBUTIONS")
    assert text.index("BLOCK CONTRIBUTIONS") < text.index("PER-FEATURE CONTRIBUTIONS")
    assert "CHANNEL TCOND" in text and "CHANNEL T2" in text
    assert "DECOMPOSED SHARE: 100.0%" in text
    assert "bg2_bottom" in text
    # --no-columns keeps the primary table and drops the subordinate one
    blocks_only = format_report(both, columns=False)
    assert "BLOCK CONTRIBUTIONS" in blocks_only
    assert "PER-FEATURE CONTRIBUTIONS" not in blocks_only


def test_cli_both_channels_end_to_end_writes_json(tmp_path, capsys):
    out = tmp_path / "diff.json"
    rc = main(["shap-diff", "flagship-c3", "graphite", "--json", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "RECONCILES: True" in printed

    payload = json.loads(out.read_text())
    assert payload["channel"] == "both"
    assert payload["residuals"]["reconciles"] is True
    assert set(payload["channels"]) == {"t2", "tcond"}
    tcond = payload["channels"]["tcond"]
    assert len(tcond["contributions"]) == 46
    assert sum(c["ms_per_char"] for c in tcond["contributions"]) == pytest.approx(
        tcond["gap_decomposed"], rel=REL_TOL
    )
    assert sum(b["ms_per_char"] for b in tcond["blocks"]) == pytest.approx(
        tcond["gap_decomposed"], rel=REL_TOL
    )
    assert payload["gap"]["decomposed_share_pct"] == pytest.approx(100.0, abs=1e-6)
    assert payload["gap"]["undecomposed_ms_per_char"] == pytest.approx(0.0, abs=1e-9)
    # the SHAPDIFF-1 artifact keys survive alongside the new per-channel tree
    assert len(payload["contributions"]) == len(BIGRAM_FEATURE_NAMES)
    json.loads(json.dumps(payload))  # no numpy scalars anywhere in the tree


def test_cli_tcond_control_exits_zero_only_when_the_control_fails(capsys):
    """The control's expectation is machine-checked: rc=0 means it correctly FAILED."""
    rc = main(["shap-diff", "flagship-c3", "graphite", "--control", "tcond-marginal"])
    assert rc == 0
    assert "failed reconciliation, as required" in capsys.readouterr().out


def test_cli_channel_flag_reports_the_undecomposed_remainder(capsys):
    rc = main(["shap-diff", "flagship-c3", "graphite", "--channel", "t2", "--no-columns"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "NOT decomposed in this run" in printed
    assert "DECOMPOSED SHARE: 31.3%" in printed
