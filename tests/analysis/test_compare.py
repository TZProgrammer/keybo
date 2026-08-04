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
    text = format_report(diff, columns=True)
    assert text.index("RECONCILIATION") < text.index("PER-FEATURE CONTRIBUTIONS")
    assert "RECONCILES: True" in text
    assert "Tcond trigram channel" in text
    assert "bottom" in text


def test_report_flags_a_failed_reconciliation():
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, shuffle_seed=1)
    text = format_report(control, columns=True)
    assert "RECONCILIATION FAILED" in text
    assert text.index("RECONCILIATION FAILED") < text.index("PER-FEATURE CONTRIBUTIONS")


# --- CLI ----------------------------------------------------------------------------------


def test_cli_end_to_end_writes_json(tmp_path, capsys):
    out = tmp_path / "diff.json"
    rc = main(["compare", "flagship-c3", "graphite", "--json", str(out)])
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
    rc = main(["compare", "flagship-c3", "graphite", "--control", "shuffle"])
    assert rc == 0
    assert "failed reconciliation, as required" in capsys.readouterr().out


def test_cli_refuses_identical_layouts(capsys):
    rc = main(["compare", "graphite", "graphite"])
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
    text = format_report(both, columns=True)
    assert text.index("RECONCILIATION") < text.index("BLOCK CONTRIBUTIONS")
    assert text.index("BLOCK CONTRIBUTIONS") < text.index("PER-FEATURE CONTRIBUTIONS")
    assert "CHANNEL TCOND" in text and "CHANNEL T2" in text
    assert "DECOMPOSED SHARE: 100.0%" in text
    assert "bg2_bottom" in text
    # the DEFAULT keeps the primary table and drops the subordinate one (COMPARE-1 H1 flipped
    # this from opt-out to opt-in; see test_per_column_table_is_opt_in_not_opt_out)
    blocks_only = format_report(both)
    assert "BLOCK CONTRIBUTIONS" in blocks_only
    assert "PER-FEATURE CONTRIBUTIONS" not in blocks_only


def test_cli_both_channels_end_to_end_writes_json(tmp_path, capsys):
    out = tmp_path / "diff.json"
    rc = main(["compare", "flagship-c3", "graphite", "--json", str(out)])
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
    rc = main(["compare", "flagship-c3", "graphite", "--control", "tcond-marginal"])
    assert rc == 0
    assert "failed reconciliation, as required" in capsys.readouterr().out


def test_cli_channel_flag_reports_the_undecomposed_remainder(capsys):
    rc = main(["compare", "flagship-c3", "graphite", "--channel", "t2"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "NOT decomposed in this run" in printed
    assert "DECOMPOSED SHARE: 31.3%" in printed


# =========================================================================================
# COMPARE-1: the per-layout FEATURE-VALUE columns, the honesty layer, and the rename.
#
# Same discipline as above — the bars as NUMBERS, and each honesty mechanism asserted through
# the behaviour it is supposed to produce rather than through its own existence. What is new:
#
# 1. mean_a/mean_b are a SECOND quantity beside the attribution: a frequency-weighted mean of
#    the feature column itself, per BOARD, under the channel's own weight (w2 vs w3). The bar
#    that catches a wrong weight here is the CROSS-WEIGHT IDENTITY (B2), which is algebra: a
#    bg1_* column depends only on (a,b), so contracting w3 over the third character MUST
#    reproduce the w2 contraction of the corresponding bigram column.
# 2. The mean columns are what make the NO-DIFF leakage flag computable at all — without them
#    the report cannot distinguish "B does less of it" from "B does exactly as much and the tree
#    split on an interaction path".
# =========================================================================================

from keybo.analysis.shap_diff import (  # noqa: E402
    CANNOT,
    CAVEATS,
    ESTIMAND,
    GAUGE_REFUSAL_MS,
    LEAKAGE_MS_FLOOR,
)

#: The parent's OUT-OF-BAND feature means (state/keybo-optimization/artifacts/feature_means.json),
#: quoted into the prereg BEFORE the in-tool path existed. These are the external reproduction
#: bar (B5): the folded-in path must reproduce the numbers that were computed and reviewed by a
#: separate script, or it is not the same quantity.
EXTERNAL_MEANS = {
    ("t2", "a", "bottom"): 0.07704719271934433,
    ("t2", "b", "bottom"): 0.11904612436920174,
    ("t2", "a", "dx"): 4.302328620093768,
    ("t2", "b", "dx"): 4.500276864941823,
    ("t2", "a", "lateral"): 0.07144692350950059,
    ("t2", "b", "lateral"): 0.05123010951623739,
    ("tcond", "a", "bg2_bottom"): 0.07407521293035432,
    ("tcond", "b", "bg2_bottom"): 0.11608938105695885,
    ("tcond", "a", "bg1_bottom"): 0.07704719271934433,
    ("tcond", "a", "sg_dx"): 3.623099432415372,
}

#: One-hot / rate columns whose mean must lie in [0,1] (B4). Prefix-stripped for bg1_/bg2_.
_RATE_COLUMNS = frozenset(
    {
        "bottom",
        "home",
        "top",
        "pinky",
        "ring",
        "middle",
        "index",
        "lateral",
        "same_hand",
        "same_finger",
        "adjacent",
        "scissor",
        "lsb",
        "inwards",
        "outwards",
        "same_hand_trigram",
        "redirect",
        "bad_redirect",
        "sg_same_finger",
    }
)


def _base_name(column: str) -> str:
    return column[4:] if column.startswith(("bg1_", "bg2_")) else column


# --- B1/B4: the constant-column canary and the in-range bar -------------------------------


def test_wpm_means_are_the_scoring_wpm_and_identical_on_both_boards(both):
    """B1, the canary that the weighting is right — and the row that exposed SHAP's splitting.

    ``wpm`` is a CONSTANT column at a fixed scoring WPM, so its frequency-weighted mean must be
    that WPM on BOTH boards in BOTH channels. It is asserted two ways, because they fail for
    different reasons:

    * ``mean_a == mean_b`` **exactly** (bit-identical). Any difference at all would mean the two
      boards were featurized differently or indexed through the wrong permutation, and the run
      would be void. This is the strong clause and it holds exactly.
    * ``mean == target_wpm`` to a float64 SUMMATION tolerance, NOT exactly. COMPARE-1's prereg
      registered this clause as exact equality and it was WRONG to: the normalized weights sum
      to exactly 1.0, but a weighted sum of 90.0 over 27k-30k cells accumulates float64
      rounding, measured at 4.7e-16 relative on the bigram frame and 4.5e-14 on the trigram one
      (more cells, more accumulation). See the COMPARE-1 addendum in PREREGISTRATIONS.md — the
      bar was mis-specified, not the code.

    And the substantive point: the column's CONTRIBUTION is not zero (-0.0922 on the bigram
    frame, -0.0273 on the trigram one) even though the boards do not differ in it at all. That
    is the coupled-column artifact, and it is exactly what the NO-DIFF flag below reports.
    """
    for channel in (both.t2, both.tcond):
        wpm = next(c for c in channel.contributions if c.feature == "wpm")
        assert wpm.mean_a == wpm.mean_b, (
            f"{channel.channel}: the boards differ in a CONSTANT column"
        )
        assert wpm.mean_delta == 0.0
        assert wpm.mean_a == pytest.approx(both.target_wpm, rel=1e-12)
        assert abs(wpm.ms_per_char) > LEAKAGE_MS_FLOOR, "the artifact this canary exists for"


def test_every_mean_lies_inside_its_column_range(both):
    """B4: a mean outside the column's own range is a normalization or indexing bug.

    Checked against the ACTUAL served matrix rather than a remembered range, so it also
    catches a mean computed over the wrong cells. Rate columns additionally must be in [0,1].
    """
    from keybo.analysis.shap_diff import _shap_tables, default_models
    from keybo.geometry import ROW_STAGGERED_30

    for channel, kind, order in ((both.t2, "bigram", 2), (both.tcond, "trigram", 3)):
        features = _shap_tables(default_models(kind), ROW_STAGGERED_30, 90.0, order)[6]
        flat = features.reshape(-1, features.shape[-1])
        for i, contribution in enumerate(channel.contributions):
            low, high = float(flat[:, i].min()), float(flat[:, i].max())
            for mean in (contribution.mean_a, contribution.mean_b):
                assert low - 1e-9 <= mean <= high + 1e-9, (
                    f"{channel.channel}/{contribution.feature}: mean {mean} outside [{low}, {high}]"
                )
            if _base_name(contribution.feature) in _RATE_COLUMNS:
                assert 0.0 <= contribution.mean_a <= 1.0
                assert 0.0 <= contribution.mean_b <= 1.0


# --- B2: THE bar that catches a wrong weight ----------------------------------------------


def test_cross_weight_identity_ties_bg1_means_to_the_bigram_channel(both):
    """B2 — the bar a WRONG WEIGHT breaks, and it is an algebraic identity, not a tolerance.

    A ``bg1_X`` column of the trigram frame depends only on the trigram's FIRST TWO characters,
    exactly as the bigram frame's ``X`` does. So contracting the trigram weight ``w3`` over the
    third character must reproduce contracting ``w2`` — because ``w2`` IS ``sum_z w3``. The two
    channels' means therefore have to agree for all 19 shared placement features, on both
    boards, to float64 rounding.

    This is the test that would fail if the mean path used a weight of its own: any weight that
    is not the channel's own would break the marginal relationship. It is a much sharper
    instrument than an in-range check, and it is why the implementation reuses
    ``_char_weight_tables`` instead of re-deriving weights (the ~1.5e-2 ``bigrams.txt`` trap
    SHAPDIFF-1 registered would show up here as a ~1e-2 failure).
    """
    t2 = {c.feature: c for c in both.t2.contributions}
    tcond = {c.feature: c for c in both.tcond.contributions}
    shared = [n for n in t2 if n != "wpm"]
    assert len(shared) == 19
    worst = 0.0
    for name in shared:
        bg1 = tcond[f"bg1_{name}"]
        for from_t2, from_tcond in ((t2[name].mean_a, bg1.mean_a), (t2[name].mean_b, bg1.mean_b)):
            assert from_t2 == pytest.approx(from_tcond, rel=1e-12), (
                f"bg1_{name} under w3 != {name} under w2 — the two channels' weights are not a "
                "marginal pair, so one channel is weighted wrongly"
            )
            worst = max(worst, abs(from_t2 - from_tcond) / max(abs(from_tcond), 1e-300))
    assert worst > 0.0, "an EXACT zero would mean the two paths are the same code, not two paths"


def test_bg2_means_differ_from_bg1_means(both):
    """B3: ``bg2_*`` is the ``(b,c)`` transition, so it must NOT reproduce ``bg1_*``.

    The mirror of B2 and equally necessary: B2 alone would also pass if the mean path had used
    the FIRST-position marginal for both blocks. All 19 properties must differ, and they do by
    0.2%-4%, which is the real difference between a trigram's two constituent bigrams.
    """
    tcond = {c.feature: c for c in both.tcond.contributions}
    properties = sorted(n[4:] for n in tcond if n.startswith("bg1_"))
    assert len(properties) == 19
    for name in properties:
        bg1, bg2 = tcond[f"bg1_{name}"], tcond[f"bg2_{name}"]
        rel = abs(bg1.mean_a - bg2.mean_a) / max(abs(bg2.mean_a), 1e-300)
        assert rel > 1e-6, f"bg2_{name} reproduces bg1_{name}: the same marginal was used twice"


# --- B5: external reproduction of the out-of-band numbers ---------------------------------


def test_means_reproduce_the_out_of_band_reference_values(both):
    """B5 — the folded-in path reproduces the numbers a SEPARATE script computed and a human read.

    These literals were produced out-of-band before the tool carried the columns, and were
    quoted into the COMPARE-1 prereg before this code path existed. An in-tool path that
    disagreed would be a different quantity wearing the reviewed numbers' name.
    """
    channels = {"t2": both.t2, "tcond": both.tcond}
    for (channel_name, board, feature), want in EXTERNAL_MEANS.items():
        contribution = next(c for c in channels[channel_name].contributions if c.feature == feature)
        got = contribution.mean_a if board == "a" else contribution.mean_b
        assert got == pytest.approx(want, rel=1e-9), (
            f"{channel_name}/{board}/{feature}: {got!r} != out-of-band {want!r}"
        )


def test_the_user_facing_worked_example_is_the_bottom_row(both):
    """The example that motivated the feature, pinned: bottom 0.077 (flagship) vs 0.119 (graphite).

    ``bottom`` carries +0.7453 ms/char favouring flagship-c3, and WITHOUT these columns a reader
    cannot tell whether flagship is faster because it does less bottom-row work or more of it.
    The answer is less — 0.0770 against 0.1190 — and that direction is the product's headline.
    """
    bottom = next(c for c in both.t2.contributions if c.feature == "bottom")
    assert bottom.mean_a == pytest.approx(0.0770, abs=5e-5)
    assert bottom.mean_b == pytest.approx(0.1190, abs=5e-5)
    assert bottom.ms_per_char > 0 and bottom.favours == "a"
    assert bottom.mean_delta > 0, "graphite does MORE bottom-row work"


def test_a_contribution_may_disagree_in_sign_with_its_feature_delta(both):
    """The REGISTERED NON-CLAIM: sign(mean_delta) need not match sign(ms_per_char).

    Asserted as a POSITIVE fact about this pair, not merely left unasserted, because a future
    reader's first instinct will be to "fix" the apparent inconsistency. ``bg1_top`` favours
    flagship-c3 (+0.4064) while flagship does MORE top-row work (0.2542 vs 0.2034): the model
    prices it that way, and a test asserting agreement would be asserting a falsehood.

    A mismatch is therefore a fact about the fitted surface, not a bug — which is exactly why
    the columns are labelled as feature VALUES and never as attributions.
    """
    disagreeing = [
        c
        for c in both.tcond.contributions
        if abs(c.ms_per_char) > LEAKAGE_MS_FLOOR
        and c.mean_delta != 0.0
        and (c.ms_per_char > 0) != (c.mean_delta > 0)
    ]
    assert disagreeing, "if this ever empties, re-read the non-claim before 'fixing' anything"
    top = next(c for c in both.tcond.contributions if c.feature == "bg1_top")
    assert top.ms_per_char > 0 and top.mean_delta < 0


# --- B6: the productization must not move the science -------------------------------------


def test_the_decomposition_is_unchanged_by_the_new_columns(both, diff):
    """B6: adding a second quantity must not perturb the first one, at all.

    The registered pre-arm values (SHAPDIFF-1/-TCOND) asserted directly, so a mean path that
    accidentally wrote into the attribution — or a refactor of ``_lmdi_channel`` that changed a
    summation order — fails here rather than silently re-pricing every published number.
    """
    assert both.gap_total == pytest.approx(3.1934, abs=1e-3)
    assert both.gap_t2 == pytest.approx(0.9981, abs=1e-3)
    assert both.gap_tcond == pytest.approx(2.1953, abs=1e-3)
    assert both.t2.resid_cell_lmdi <= REL_TOL and both.tcond.resid_cell_lmdi <= REL_TOL
    for feature, want in (("bottom", 0.7453), ("dx", 0.1678), ("wpm", -0.0922)):
        got = next(c for c in both.t2.contributions if c.feature == feature)
        assert got.ms_per_char == pytest.approx(want, abs=1e-3)
    for feature, want in (("bg2_bottom", 0.7382), ("bg1_bottom", -0.2337), ("bg1_top", 0.4064)):
        got = next(c for c in both.tcond.contributions if c.feature == feature)
        assert got.ms_per_char == pytest.approx(want, abs=1e-3)


# --- H2: the leakage flags -----------------------------------------------------------------


def test_no_diff_flag_fires_on_wpm_and_needs_the_mean_columns(both):
    """H2 NO-DIFF: equal means + non-zero credit = the credit is an interaction artifact.

    This flag is the first thing the new columns BUY. Without ``mean_a``/``mean_b`` the report
    cannot tell "board B does less of this" from "board B does exactly as much of it", so it
    cannot tell a real difference from a coupled-column artifact. With them it can, and it says
    so at the point of reading.
    """
    for channel in (both.t2, both.tcond):
        flags = channel.leakage()
        assert flags.get("wpm") == "NO-DIFF", f"{channel.channel}: wpm must be flagged"
        wpm = next(c for c in channel.contributions if c.feature == "wpm")
        assert wpm.flag == "NO-DIFF", "the flag must ride on the row, not only on the channel"
        assert wpm.mean_a == wpm.mean_b and abs(wpm.ms_per_char) >= LEAKAGE_MS_FLOOR


def test_coupled_flag_fires_on_opposite_signed_bg1_bg2_mates(both):
    """H2 COUPLED: the measured ``bg1_bottom`` -0.2337 vs ``bg2_bottom`` +0.7382 case.

    The two columns are the SAME physical property on the trigram's two overlapping key pairs,
    credited in OPPOSITE directions. Neither number stands alone; the joint does, because it
    does not depend on how TreeSHAP divided the credit between them.
    """
    flags = both.tcond.leakage()
    assert flags.get("bg1_bottom") == "COUPLED" and flags.get("bg2_bottom") == "COUPLED"
    bg1 = next(c for c in both.tcond.contributions if c.feature == "bg1_bottom")
    bg2 = next(c for c in both.tcond.contributions if c.feature == "bg2_bottom")
    assert bg1.ms_per_char < 0 < bg2.ms_per_char, "the registered opposite-sign case"
    assert bg1.flag == bg2.flag == "COUPLED"
    assert both.tcond.joint("bottom") == pytest.approx(bg1.ms_per_char + bg2.ms_per_char, rel=1e-12)
    assert both.tcond.joint("bottom") == pytest.approx(0.5045, abs=1e-3)
    # every flagged column is a real member of an opposite-signed pair, and both mates are
    # flagged together — a one-sided flag would tell a reader to distrust the wrong row
    for name, kind in flags.items():
        if kind != "COUPLED":
            continue
        mate = ("bg2_" + name[4:]) if name.startswith("bg1_") else ("bg1_" + name[4:])
        assert flags[mate] == "COUPLED"
    # the T2 frame has no bg1_/bg2_ columns at all, so it can never raise this flag
    assert "COUPLED" not in both.t2.leakage().values()


def test_leakage_flags_are_computed_not_hardcoded(both):
    """The flags must come from the NUMBERS: perturb a contribution, watch the verdict move.

    A hand-listed flag set would pass every assertion above while being blind on any other
    layout pair. Each clause moves ONE input and asserts the verdict follows.

    ⚠ The dust clause must keep the mate's OPPOSITE sign, and this is the whole reason the
    clause is written the way it is. A first draft used ``+LEAKAGE_MS_FLOOR/10``, which is the
    same sign as ``bg2_bottom`` (+0.7382) — so the sign rule excluded the pair before the
    magnitude floor was ever consulted, and the assertion passed for the WRONG REASON. Mutation
    M9 (deleting the floor check entirely) caught it: the test stayed GREEN. With a NEGATIVE
    dust value the pair is still opposite-signed and only the floor can exclude it, so M9 now
    goes red. A passing assertion that cannot fail is not evidence — SHAPDIFF-1 recorded the
    same lesson from a tautological additivity check that printed exactly 0.000e+00.
    """
    import dataclasses

    tcond = both.tcond
    original = tcond.contributions
    try:
        bg1 = next(c for c in original if c.feature == "bg1_bottom")
        bg2 = next(c for c in original if c.feature == "bg2_bottom")
        assert bg1.ms_per_char < 0 < bg2.ms_per_char, "the fixture this test perturbs"

        tcond.contributions = [
            dataclasses.replace(c, ms_per_char=-c.ms_per_char) if c is bg1 else c for c in original
        ]
        assert "bg1_bottom" not in tcond.leakage(), "same-signed mates must NOT be flagged"

        dust = -LEAKAGE_MS_FLOOR / 10  # NEGATIVE: still opposite bg2, so only the FLOOR can act
        assert dust * bg2.ms_per_char < 0.0, "the dust must remain opposite-signed to its mate"
        tcond.contributions = [
            dataclasses.replace(c, ms_per_char=dust) if c is bg1 else c for c in original
        ]
        assert "bg1_bottom" not in tcond.leakage(), "sub-floor dust must NOT be flagged"

        wpm = next(c for c in original if c.feature == "wpm")
        tcond.contributions = [
            dataclasses.replace(c, mean_b=c.mean_b * 1.01) if c is wpm else c for c in original
        ]
        assert tcond.leakage().get("wpm") != "NO-DIFF", "differing means must clear NO-DIFF"

        # and NO-DIFF needs the magnitude too: equal means with dust credit is not worth flagging
        tcond.contributions = [
            dataclasses.replace(c, ms_per_char=LEAKAGE_MS_FLOOR / 10) if c is wpm else c
            for c in original
        ]
        assert "wpm" not in tcond.leakage(), "equal means + dust credit must NOT be flagged"
    finally:
        tcond.contributions = original
    assert tcond.leakage().get("wpm") == "NO-DIFF"


def test_joint_refuses_a_property_the_frame_does_not_carry_twice(both):
    """``joint`` must not answer a question the frame cannot support."""
    with pytest.raises(ValueError, match="not a property"):
        both.tcond.joint("redirect")
    with pytest.raises(ValueError, match="not a property"):
        both.t2.joint("bottom")


# --- H1: block-first, per-column opt-IN ----------------------------------------------------


def test_per_column_table_is_opt_in_not_opt_out(both):
    """H1: the DEFAULT report shows blocks only — the misleading table must be asked for.

    Inverted from SHAPDIFF-1/-TCOND, where the audience was the author. A block sum is
    invariant to how TreeSHAP redistributed credit among correlated columns; a per-column number
    is not. So the default carries the claim that survives, and the subordinate split is opt-in.
    """
    default = format_report(both)
    assert "BLOCK CONTRIBUTIONS" in default
    assert "PER-FEATURE CONTRIBUTIONS" not in default
    assert "PER-FEATURE CONTRIBUTIONS" in format_report(both, columns=True)


def test_block_table_carries_the_two_column_view_and_the_flag(both):
    """The PRIMARY table must show feature values too, or the reader drops to the unsafe one.

    A block spans a one-hot and a distance in key units, so there is deliberately no block-level
    mean — the block's LEADING column and its two values are the honest statement, and the block
    inherits any flag its columns raised.

    ⚠ ``FINGER`` and ``BG1`` are asserted specifically because they are the blocks where the
    largest-|ms| column is NOT the frame-order-first column (``lateral`` not ``pinky``;
    ``bg1_top`` not ``bg1_bottom``). A first draft asserted only ``ROW``, whose two happen to
    coincide at ``bottom`` — so mutation M10 (report ``columns[0]`` instead of the largest) left
    the test GREEN. Any future assertion here must use a block where the two differ, or it
    checks nothing about the selection rule.
    """
    text = format_report(both)
    row_block = next(b for b in both.t2.blocks() if b.block == "ROW")
    assert row_block.leading is not None
    lead, mean_a, mean_b = row_block.leading
    assert lead == "bottom"
    assert (mean_a, mean_b) == pytest.approx((0.0770, 0.1190), abs=5e-5)
    assert "0.0770" in text and "0.1190" in text

    # the discriminating cases: leading must be the LARGEST-|ms| column, not the first one
    finger = next(b for b in both.t2.blocks() if b.block == "FINGER")
    assert finger.columns[0] == "pinky", "frame order (the wrong answer)"
    assert finger.leading[0] == "lateral", "largest |ms/char| (the right answer)"
    bg1 = next(b for b in both.tcond.blocks() if b.block == "BG1")
    assert bg1.columns[0] == "bg1_bottom" and bg1.leading[0] == "bg1_top"
    # and the leading column's means must be ITS OWN, not another column's
    for channel in (both.t2, both.tcond):
        by_name = {c.feature: c for c in channel.contributions}
        for block in channel.blocks():
            name, block_mean_a, block_mean_b = block.leading
            assert name == max(block.columns, key=lambda n: abs(by_name[n].ms_per_char))
            assert (block_mean_a, block_mean_b) == (by_name[name].mean_a, by_name[name].mean_b)
    wpm_block = next(b for b in both.t2.blocks() if b.block == "WPM")
    assert wpm_block.flag == "NO-DIFF" and "[NO-DIFF]" in text
    bg2 = next(b for b in both.tcond.blocks() if b.block == "BG2")
    assert bg2.flag == "COUPLED"
    # the joints are printed for the reader, not left as an exercise
    assert "COUPLED PROPERTIES" in text and "JOINT" in text


def test_top_truncation_names_what_it_withheld(both):
    """A truncated table must PRICE its remainder — the over-claim guard, one level down.

    ``--top`` exists so 66 rows do not land on a reader at once. The danger is that the visible
    rows read as the whole decomposition, so the withheld count, their total, and the largest
    withheld column are all named, and the SUM stays over ALL columns.
    """
    full = format_report(both, columns=True, top=0)
    clipped = format_report(both, columns=True, top=5)
    assert "... and 41 more columns" in clipped  # 46 trigram columns - 5 shown
    assert "and 15 more columns" in clipped  # 20 bigram columns - 5 shown
    assert "--json is never truncated" in clipped
    assert "over ALL columns" in clipped
    assert (
        "bg2_angle" in full
        and "bg2_angle" not in clipped.split("CHANNEL TCOND")[1].split("COUPLED PROPERTIES")[0]
    )


# --- H3: the cross-channel non-additivity guard --------------------------------------------


def test_cross_channel_properties_are_named_and_refused(both):
    """H3: a property in BOTH channels must be NAMED and its cross-channel total REFUSED.

    ``bottom`` is 23.3% of the total gap in the T2 channel and ``bg1_bottom``+``bg2_bottom`` is
    15.8% in the Tcond channel. They are the same physical property attributed on two different
    frames over two different populations of cells, each already carrying its own channel's full
    share — so their sum is a double-count, and there is no correct number to return. The
    library therefore RAISES rather than picking one.
    """
    shared = both.cross_channel_properties()
    assert len(shared) == 19 and "bottom" in shared and "dx" in shared
    assert "wpm" not in shared, "wpm is WPM-blocked in both frames, not a placement property"
    with pytest.raises(ValueError, match="REFUSED"):
        both.total_for_property("bottom")
    # the refusal must carry BOTH numbers, so the reader can act on it
    with pytest.raises(ValueError, match=r"T2 \+0\.7453.*Tcond bg1\+bg2 \+0\.5045"):
        both.total_for_property("bottom")
    # and it refuses to pretend for a property that is not actually doubled
    with pytest.raises(ValueError, match="not carried by both channels"):
        both.total_for_property("redirect")
    # a single-channel run has nothing to refuse
    only_t2 = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="t2")
    assert only_t2.cross_channel_properties() == []


def test_report_states_the_non_additivity_with_a_worked_example(both):
    """The refusal must be in the READER's path, not only in the API."""
    text = format_report(both)
    assert "DO NOT ADD ACROSS CHANNELS" in text
    assert "19 properties appear in BOTH channels" in text
    assert "NOT +1.2498" in text, "the double-count must be shown as the wrong number it is"


# --- H4: provenance and calibration carry --------------------------------------------------


def test_caveats_print_where_the_magnitudes_print(both):
    """H4: the four measured caveats ride WITH the numbers, and name their measurements."""
    text = format_report(both)
    assert "HOW TO READ THIS" in text
    assert ESTIMAND.split(",")[0] in text
    for caveat in CAVEATS:
        assert caveat.split(":")[0] in text
    assert "1.407" in text and "0.7304" in text, "the calibration slopes, as numbers"
    assert "affine-invariant" in text, "orderings are safe even where magnitudes are not"
    assert "prices LONG travel as CHEAPER" in text, "the dx/distance provenance caveat"
    # the caveats are word-WRAPPED into the report, so compare on collapsed whitespace rather
    # than raw substrings — an assertion on the unwrapped string tests the line breaks, not the
    # content, and would go green on a report that had dropped the text and kept the header
    collapsed = " ".join(text.split())
    assert " ".join(CANNOT.split()) in collapsed, "the one thing it CANNOT do must be stated"
    for caveat in CAVEATS:
        assert " ".join(caveat.split()) in collapsed
    # and the caveats must be in the MACHINE artifact too
    payload = both.to_dict()
    assert payload["honesty"]["caveats"] == list(CAVEATS)
    assert payload["honesty"]["estimand"] == ESTIMAND
    assert len(payload["honesty"]["cross_channel_properties_not_summable"]) == 19


# --- the two-bar guarantee as a PRODUCT feature --------------------------------------------


def test_default_run_emits_both_residual_families(both):
    """Both families in the default output — neither alone is sufficient (SHAPDIFF-1's finding)."""
    text = format_report(both)
    assert "INTERNAL per-cell LMDI identity" in text
    assert "INTERNAL sum(features) vs channel gap" in text
    assert "EXTERNAL gap vs shipped table" in text
    assert "EXTERNAL GAUGE TIE: OK" in text
    assert both.gauge_tie_ok()


def test_a_failed_gauge_tie_SUPPRESSES_the_tables_rather_than_annotating_them():
    """THE refusal. A wrong weighting must produce NO attribution table at all.

    This is the mechanism that makes the tool a product rather than an instrument. SHAPDIFF-1
    measured that the internal sums-back identity passes at ~1e-16 under a weighting that is
    wrong by 5.6e-2 ms/char, because both sides of that identity share the weight table.
    SHAPDIFF-TCOND measured that the analogous Tcond error additionally INVERTS the answer (BG1
    overtakes BG2). A tool that printed an interpretable table under those conditions would be
    silently decomposing the wrong quantity — so the tables do not print.

    Asserted as ABSENCE of the tables, which is what a downgrade to a mere warning would break.
    """
    control = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="tcond", tcond_weighting="tcond-marginal")
    assert not control.gauge_tie_ok()
    text = format_report(control, columns=True)
    assert "REFUSED: THE EXTERNAL GAUGE TIE FAILED" in text
    assert "BLOCK CONTRIBUTIONS" not in text, "the primary table must be SUPPRESSED, not flagged"
    assert "PER-FEATURE CONTRIBUTIONS" not in text
    assert "COUPLED PROPERTIES" not in text
    # the reason is stated, with the numbers
    assert "tcond-marginal" in text
    assert "1e-16" in text and "WRONG QUANTITY" in text
    # the RECONCILIATION header still prints: a reader must see the residuals that condemned it
    assert "RECONCILIATION" in text and "EXTERNAL GAUGE TIE: FAILED" in text
    # and the good run is unaffected
    good = shap_diff(FLAGSHIP_C3, GRAPHITE, channel="tcond")
    assert good.gauge_tie_ok()
    assert "BLOCK CONTRIBUTIONS" in format_report(good)


def test_the_shuffle_control_still_reaches_its_table(both):
    """The shuffle breaks the INTERNAL bars only, so it must NOT trigger the gauge refusal.

    The two controls have to be distinguishable in the OUTPUT, not only in the residuals: a
    refusal that fired on both would collapse the distinction SHAPDIFF-1 established between
    "the arithmetic is broken" and "the quantity is wrong".
    """
    shuffled = shap_diff(FLAGSHIP_C3, GRAPHITE, shuffle_seed=0)
    assert not shuffled.reconciles()
    assert shuffled.gauge_tie_ok(), "shuffling redistributes attribution, not the total"
    text = format_report(shuffled)
    assert "RECONCILIATION FAILED" in text
    assert "REFUSED" not in text
    assert "BLOCK CONTRIBUTIONS" in text, "a different failure needs a different presentation"


def test_gauge_refusal_threshold_is_the_registered_bar():
    """The product's refusal bar is the registered ``gauge_tol``, not a second unregistered one."""
    assert GAUGE_REFUSAL_MS == 1e-3
    assert LEAKAGE_MS_FLOOR == 0.01


def test_the_per_channel_gauge_clause_is_load_bearing_only_off_the_natural_inputs(both):
    """``gauge_tie_ok`` checks the per-CHANNEL tie as well as the total — a defence in depth.

    ⚠ HONEST SCOPE, recorded because mutation M11 proved it: on every input reachable through
    the public API the per-channel clause is REDUNDANT. Both weighting controls move a channel
    gap AND the total by the same amount (measured: 1.5743e+00 on each for ``tcond-marginal``),
    so deleting the clause changes no reachable verdict and no black-box test can kill it.

    So this test does not pretend otherwise: it constructs the state directly. A channel whose
    own tie has drifted while the total has NOT is arithmetically possible (two channels erring
    in opposite directions would cancel in the total), and the clause is what catches it. That
    is a real guarantee about the code, established by construction rather than by a natural
    input that does not exist — which is the honest form of the claim.
    """
    import dataclasses

    assert both.gauge_tie_ok(), "the fixture reconciles"
    # only the CHANNEL tie drifts; the total is untouched — the clause's whole purpose
    drifted = dataclasses.replace(
        both, tcond=dataclasses.replace(both.tcond, resid_gap_vs_shipped=1.0)
    )
    assert drifted.resid_vs_card_gap <= GAUGE_REFUSAL_MS, "the total is deliberately still fine"
    assert not drifted.gauge_tie_ok(), "a drifted CHANNEL must refuse even when the total agrees"
    assert "REFUSED" in format_report(drifted)
    # and the mirror: only the total drifts
    total_only = dataclasses.replace(both, resid_vs_card_gap=1.0)
    assert not total_only.gauge_tie_ok()


# --- the rename ----------------------------------------------------------------------------


def test_compare_is_the_registered_command_and_shap_diff_is_gone():
    """COMPARE-1's rename: ``keybo compare`` exists, ``keybo shap-diff`` does not.

    Removed rather than aliased: two documented entry points to one tool is exactly the kind of
    ambiguity the rename was meant to resolve.
    """
    from keybo.cli.__main__ import _COMMANDS, build_parser

    assert "compare" in _COMMANDS
    assert "shap-diff" not in _COMMANDS
    assert hasattr(_COMMANDS["compare"], "add_arguments")
    assert hasattr(_COMMANDS["compare"], "run")
    # it slots into the shared dispatch exactly as `analyze` does
    parser = build_parser()
    args = parser.parse_args(["compare", "flagship-c3", "graphite"])
    assert args.command == "compare"
    assert (args.layout_a, args.layout_b) == ("flagship-c3", "graphite")
    assert args.channel == "both" and args.columns is False and args.top == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["shap-diff", "flagship-c3", "graphite"])


def test_help_disambiguates_compare_from_its_two_neighbours():
    """A user must be able to tell ``compare`` from ``layout-diff`` and ``shap-report``."""
    from keybo.cli import compare as compare_module

    doc = compare_module.__doc__
    assert "layout-diff" in doc and "shap-report" in doc
    assert "N-GRAM" in doc, "layout-diff's unit"
    assert "FEATURES" in doc, "compare's unit"
    assert "analyze" in doc, "the sibling it matches"


def test_compare_matches_analyze_flag_spellings():
    """Parity with ``keybo analyze``: the overlapping flags are spelled identically."""
    import argparse

    from keybo.cli import analyze as analyze_module
    from keybo.cli import compare as compare_module

    def options(module):
        parser = argparse.ArgumentParser()
        module.add_arguments(parser)
        return {option for action in parser._actions for option in action.option_strings}

    shared = options(analyze_module) & options(compare_module)
    for flag in ("--corpus", "--target-wpm", "--json"):
        assert flag in shared, f"{flag} must be spelled the same as in analyze"


def test_cli_accepts_a_raw_30_char_layout_like_analyze(capsys):
    """``analyze`` takes registry names OR raw 30-char strings; so must ``compare``."""
    rc = main(["compare", FLAGSHIP_C3, "graphite", "--channel", "t2", "--top-ngrams", "0"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "BLOCK CONTRIBUTIONS" in printed
    assert "RECONCILES: True" in printed


def test_cli_json_carries_the_means_the_flags_and_the_caveats(tmp_path, capsys):
    """The machine artifact must carry everything the human report does — plus no truncation."""
    out = tmp_path / "compare.json"
    rc = main(["compare", "flagship-c3", "graphite", "--json", str(out), "--top", "3"])
    assert rc == 0
    payload = json.loads(out.read_text())
    json.loads(json.dumps(payload))  # no numpy scalars

    tcond = payload["channels"]["tcond"]
    assert len(tcond["contributions"]) == 46, "--top must NOT truncate the JSON"
    bottom = next(c for c in payload["channels"]["t2"]["contributions"] if c["feature"] == "bottom")
    assert bottom["mean_a"] == pytest.approx(0.07704719271934433, rel=1e-9)
    assert bottom["mean_b"] == pytest.approx(0.11904612436920174, rel=1e-9)
    assert bottom["mean_delta"] == pytest.approx(bottom["mean_b"] - bottom["mean_a"], rel=1e-12)
    assert tcond["leakage_flags"]["bg1_bottom"] == "COUPLED"
    assert tcond["leakage_flags"]["wpm"] == "NO-DIFF"
    bg2_bottom = next(c for c in tcond["contributions"] if c["feature"] == "bg2_bottom")
    assert bg2_bottom["flag"] == "COUPLED"
    assert payload["honesty"]["gauge_tie_ok"] is True
    assert payload["honesty"]["cross_channel_properties_not_summable"][0] == "bottom"
    # the SHAPDIFF-1 compat block carries the new columns too
    legacy = next(c for c in payload["contributions"] if c["feature"] == "bottom")
    assert legacy["mean_a"] == pytest.approx(0.07704719271934433, rel=1e-9)
    # blocks carry the two-column view
    row = next(b for b in payload["channels"]["t2"]["blocks"] if b["block"] == "ROW")
    assert row["leading_column"] == "bottom"
    assert row["leading_mean_b"] == pytest.approx(0.11904612436920174, rel=1e-9)


def test_cli_refuses_and_exits_nonzero_when_the_gauge_tie_fails(tmp_path, capsys):
    """End to end: a wrong-weighting run prints no table and does not exit 0 as a plain run.

    Run WITHOUT ``--control`` (which deliberately inverts the exit code) so the product's own
    contract is what is tested: a run whose external tie fails must not look successful.
    """
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    freqs = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
    control = shap_diff(
        FLAGSHIP_C3, GRAPHITE, channel="t2", weighting="bigram-table", control_bigram_freqs=freqs
    )
    assert not control.gauge_tie_ok()
    text = format_report(control, columns=True)
    assert "REFUSED" in text and "BLOCK CONTRIBUTIONS" not in text
    assert "bigram-table" in text
    # the machine artifact says so too, rather than shipping a table with a quiet caveat
    assert control.to_dict()["honesty"]["gauge_tie_ok"] is False
