"""`keybo analyze` on the FULL campaign gauge frame (ALLGAUGE-1).

Every value here is a POSITIVE CONTROL against a frozen campaign board, not a
self-consistency check: each expectation was produced by a different driver, on a
different day, and is asserted EXACTLY (``==``), not approximately. Trap #3 of the
campaign's tooling-traps file is the reason every metric gets its own control rather
than a sample — three gauges reproducing bit-for-bit does not imply the fourth will.

Two frozen frames are used, and the distinction is load-bearing (trap #13):

* ``wscissor-allgauge`` — the 19-gauge frame. Its 4 community gauges are **primed**
  (``score_primed``), and its 15 corpus-sensitive gauges are computed on
  ``data/corpus/{bigrams,1-skip31,trigrams}.txt``.
* ``board-blend-reselect`` — same 15 corpus-sensitive gauges, same convention, but its
  community block is **raw** ``score()``. It supplies an independent second layout
  (``flagship-c3``) so the controls are not all from one artifact.

Both are checked in as literals below rather than read from another agent's state
directory, so the suite is hermetic.
"""

from __future__ import annotations

import json

import pytest

from keybo.cli.__main__ import main

#: flagship-c3, from artifacts/reselect/board-blend-reselect.json (`layouts`).
FLAGSHIP_C3 = "pyou'vgdnmheai.cstrlkjz,-wfbxq"
#: keybo-lsb, the campaign incumbent (registry name `keybo-lsb`).
KEYBO_LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"

#: FROZEN: wscissor-allgauge.json -> corpora.iweb.scores["keybo-lsb"].
#: The 15 corpus-sensitive gauges on data/corpus + 1-skip31.
FROZEN_KEYBO_LSB = {
    "sfr": 2.8187069323648957,
    "sfb": 1.0784319931923778,
    "sfs": 7.18827075953739,
    "sfb-dist": 1.1959980987412342,
    "sfs-dist": 8.389842389989848,
    "lsb": 0.7582746240682448,
    "lsb-dist": 1.5633282827639192,
    "alt": 43.61506539321432,
    "roll": 43.14440231235716,
    "sr-roll": 14.447992578080159,
    "redir": 3.430775999957276,
    "comfort": 3.587356906409614,
    "scissor": 0.10337818345585705,
    "imbalance": 0.9321038278010377,
    "oxey-style": -10.643631222236356,
}

#: FROZEN: wscissor-allgauge.json -> corpora.iweb.scores["archive-1843"].
ARCHIVE_1843 = "pyou,vgdnmheai.cstlrjz'k-fwbxq"
FROZEN_ARCHIVE_1843 = {
    "sfr": 2.8187069323648957,
    "sfb": 1.5151008596808413,
    "sfs": 6.977606271345833,
    "sfb-dist": 1.6753537976046173,
    "sfs-dist": 8.094344802981917,
    "lsb": 0.5773982166723277,
    "lsb-dist": 1.1924902058944642,
    "alt": 43.61506539321432,
    "roll": 42.66025704274908,
    "sr-roll": 14.568257960885262,
    "redir": 2.122830507906545,
    "comfort": 3.521047982285049,
    "scissor": 0.12669673416695082,
    "imbalance": 0.9321038278010377,
    "oxey-style": -10.231210014482816,
}

#: FROZEN: board-blend-reselect.json -> corpus_sensitive["iweb-only"]["flagship-c3"].
FROZEN_FLAGSHIP_C3 = {
    "sfr": 2.8187069323648957,
    "sfb": 1.2407650391505076,
    "sfs": 6.530070526466785,
    "sfb-dist": 1.3553353231830805,
    "sfs-dist": 7.673937510662194,
    "lsb": 0.556700684268725,
    "lsb-dist": 1.1510951410872585,
    "alt": 43.61506539321432,
    "roll": 43.123813811524386,
    "sr-roll": 14.497020828227,
    "redir": 2.307316203091323,
    "comfort": 3.508484999848331,
    "scissor": 0.07719701257558044,
    "imbalance": 0.9321038278010377,
    "oxey-style": -13.807063168667037,
}

#: FROZEN: wscissor-allgauge.json -> invariant_direction_derivation.reference_scores.
#: These are the PRIMED community scores (score_primed), not score(); `wfd` on this board
#: is the APOSTROPHE-PINNED convention (see tests/analysis/test_community_wfd_frames.py).
FROZEN_PRIMED_KEYBO_LSB = {
    "genkey_primed": 1.3192528115010698,
    "oxey1_primed": -478152084.0,
    "oxey2_primed": -5014283671800.0,
    "wfd": -15082741528300.0,
}

#: FROZEN: board-blend-reselect.json -> corpus_invariant["flagship-c3"] (RAW score()).
FROZEN_RAW_FLAGSHIP_C3 = {
    "genkey": 32.702982334916875,
    "oxeylyzer1": 18799648612,
    "oxeylyzer2": -22722669144300,
    "wfd": -17469561624900,
}

#: FROZEN: all-gauge-table.json -> rows[*].speed.surfaces[*].fit, on the
#: standardized (common-bigram) surfaces at the baked 90 wpm.
FROZEN_FITS = {
    "keybo-lsb": {
        "AALTO_TRI_PS_FREQ_PRIOR": 118837087932.74153,
        "COMMUNITY_TRI_PS_FREQ_PRIOR": 120253383690.93106,
        "POOL_TRI_PS_FREQ_PRIOR": 121754240226.69867,
    },
    "qwerty": {
        "AALTO_TRI_PS_FREQ_PRIOR": 122897128096.63051,
        "COMMUNITY_TRI_PS_FREQ_PRIOR": 125737445271.86528,
        "POOL_TRI_PS_FREQ_PRIOR": 125697996875.05296,
    },
}

SCISSOR_FAMILY = ("scissor", "scissor-pinky-share")


def _run(capsys, argv: list[str]) -> dict:
    rc = main(argv)
    assert rc == 0, f"analyze {argv} returned {rc}"
    return json.loads(capsys.readouterr().out)


@pytest.mark.slow
def test_every_frozen_gauge_reproduces_exactly_for_keybo_lsb(capsys):
    """POSITIVE CONTROL 1: all 15 corpus-sensitive gauges, EXACT, vs wscissor-allgauge."""
    out = _run(capsys, ["analyze", "keybo-lsb", "--json"])
    got = out["rows"]["keybo-lsb"]["gauges"]
    mismatched = {k: (v, got[k]) for k, v in FROZEN_KEYBO_LSB.items() if got[k] != v}
    assert not mismatched, f"frozen-board mismatch (exact compare): {mismatched}"


@pytest.mark.slow
def test_every_frozen_gauge_reproduces_exactly_for_archive_1843(capsys):
    """POSITIVE CONTROL 2: a second layout, same frame, all 15 gauges EXACT."""
    out = _run(capsys, ["analyze", ARCHIVE_1843, "--json"])
    row = next(r for r in out["rows"].values() if r["layout"] == ARCHIVE_1843)
    got = row["gauges"]
    mismatched = {k: (v, got[k]) for k, v in FROZEN_ARCHIVE_1843.items() if got[k] != v}
    assert not mismatched, f"frozen-board mismatch (exact compare): {mismatched}"


@pytest.mark.slow
def test_frozen_gauges_reproduce_for_flagship_c3_on_the_second_board(capsys):
    """POSITIVE CONTROL 3: an INDEPENDENT frozen artifact (board-blend-reselect)."""
    out = _run(capsys, ["analyze", FLAGSHIP_C3, "--json"])
    row = next(r for r in out["rows"].values() if r["layout"] == FLAGSHIP_C3)
    got = row["gauges"]
    mismatched = {k: (v, got[k]) for k, v in FROZEN_FLAGSHIP_C3.items() if got[k] != v}
    assert not mismatched, f"second-board mismatch (exact compare): {mismatched}"


@pytest.mark.slow
def test_primed_and_raw_community_scores_both_reported_and_both_pinned(capsys):
    """The two community FRAMES are distinct and BOTH are reported, each labelled.

    Trap #13: the campaign has two live conventions for the same four community
    gauges. Reporting one silently is how a comparison gets stitched across frames.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", FLAGSHIP_C3, "--json"])
    lsb = out["rows"]["keybo-lsb"]
    for key, expected in FROZEN_PRIMED_KEYBO_LSB.items():
        assert lsb["community_primed"][key] == expected, key
    flag = next(r for r in out["rows"].values() if r["layout"] == FLAGSHIP_C3)
    for key, expected in FROZEN_RAW_FLAGSHIP_C3.items():
        assert flag["community"][key] == expected, key


@pytest.mark.slow
def test_per_finger_scissor_is_an_exact_partition_of_aggregate_scissor(capsys):
    """The single best test that the attribution rule is right: the parts SUM to the whole."""
    out = _run(capsys, ["analyze", "keybo-lsb", "graphite", "--json"])
    for name, row in out["rows"].items():
        per_finger = row["scissor_by_finger"]
        aggregate = row["gauges"]["scissor"]
        assert set(per_finger) == {
            "LP",
            "LR",
            "LM",
            "LI",
            "RI",
            "RM",
            "RR",
            "RP",
        }, f"{name}: unexpected finger set {sorted(per_finger)}"
        assert sum(per_finger.values()) == pytest.approx(aggregate, rel=0, abs=1e-12), (
            f"{name}: per-finger scissor {sum(per_finger.values())!r} != aggregate {aggregate!r}"
        )


@pytest.mark.slow
def test_scissor_by_finger_charges_both_fingers_half_each(capsys):
    """The attribution rule is DECLARED: a scissor charges 0.5 to each of its two fingers.

    An exact partition forces a choice; splitting evenly is the only rule that both
    partitions exactly and stays symmetric under the bigram's order. Pinned here so a
    later change to the rule is a test failure, not a silent renormalization.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", "--json"])
    row = out["rows"]["keybo-lsb"]
    assert row["scissor_by_finger_rule"] == "half-to-each-finger"
    # No single finger can hold more than the aggregate.
    aggregate = row["gauges"]["scissor"]
    assert max(row["scissor_by_finger"].values()) <= aggregate


@pytest.mark.slow
def test_qwerty_is_worse_than_flagship_on_every_scissor_family_metric(capsys):
    """Sanity ordering: the scissor family must rank qwerty below a campaign flagship."""
    out = _run(capsys, ["analyze", "qwerty30m", FLAGSHIP_C3, "--json"])
    qwerty = out["rows"]["qwerty30m"]
    flag = next(r for r in out["rows"].values() if r["layout"] == FLAGSHIP_C3)
    assert qwerty["gauges"]["scissor"] > flag["gauges"]["scissor"]
    # every finger's scissor load, summed over the weak fingers, is also worse
    weak = ("LP", "LR", "RR", "RP")
    assert sum(qwerty["scissor_by_finger"][f] for f in weak) > sum(
        flag["scissor_by_finger"][f] for f in weak
    )


@pytest.mark.slow
def test_bad_redirect_family_is_reported_and_bounded_by_the_redirect_family(capsys):
    """`bad_redirects` mass <= total redirect-family mass, for every layout.

    NOT asserted: ``bad_redirects_sfs <= bad_redirects``. The four classes are mutually
    exclusive siblings (``_v1_pattern`` returns one label), and on qwerty the ``_sfs``
    share is the larger of the two — see ``tests/analysis/test_redirects.py``.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", "graphite", "qwerty30m", "--json"])
    for name, row in out["rows"].items():
        red = row["redirects"]
        for cls in ("redirects", "redirects_sfs", "bad_redirects", "bad_redirects_sfs"):
            assert 0.0 <= red[cls] <= red["redirects_family_total"], f"{name}/{cls}"
        assert red["bad_redirects_total"] == pytest.approx(
            red["bad_redirects"] + red["bad_redirects_sfs"], rel=0, abs=1e-12
        ), name


@pytest.mark.slow
def test_kmstats_redir_equals_the_oxeylyzer_redirect_family_exactly(capsys):
    """The DOCUMENTED relationship, asserted — not assumed to be nesting.

    Verified exhaustively over all 30^3 slot triples: `kmstats._is_redirect` and the
    union of oxeylyzer-1's four redirect patterns select the IDENTICAL trigram set, and
    the two finger maps are identical. So on a shared denominator the masses are equal,
    not merely ordered. (Trap #11: a nested pair of legs is one leg — here they are the
    SAME leg, which is stronger and is what this pins.)
    """
    out = _run(capsys, ["analyze", "keybo-lsb", "graphite", "qwerty30m", "--json"])
    for name, row in out["rows"].items():
        assert row["redirects"]["redirects_family_total"] == pytest.approx(
            row["gauges"]["redir"], rel=0, abs=1e-9
        ), f"{name}: redirect family total != kmstats redir"


@pytest.mark.slow
def test_dvorak_renders_na_on_charset_dependent_cells_and_does_not_crash(capsys):
    """dvorak is neither a C30M nor a classic permutation: N/A, never a number, never a crash."""
    out = _run(capsys, ["analyze", "dvorak", "--json"])
    row = next(r for r in out["rows"].values() if r["layout"].startswith("',.py"))
    # charset-dependent: the oxeylyzer boards and the modeled surfaces
    assert row["community"]["oxeylyzer1"] is None
    assert row["community"]["oxeylyzer2"] is None
    assert row["community"]["wfd"] is None
    assert row["community_primed"]["oxey1_primed"] is None
    assert row["model_scores"]["available"] is False
    assert "charset" in row["model_scores"]["reason"].lower()
    # charset-AGNOSTIC gauges still score
    assert row["community"]["genkey"] is not None
    assert row["gauges"]["sfb"] > 0.0
    assert row["gauges"]["scissor"] > 0.0


def test_dvorak_text_report_does_not_crash_and_prints_na(capsys):
    """The TEXT path must survive dvorak too (it used to raise ValueError)."""
    rc = main(["analyze", "dvorak", "--no-model-scores", "--no-time"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "N/A" in text


@pytest.mark.slow
def test_json_and_text_report_agree(capsys):
    """The JSON path is the one that silently drifts — pin them against each other."""
    text_rc = main(["analyze", "keybo-lsb", "--no-time"])
    assert text_rc == 0
    text = capsys.readouterr().out
    out = _run(capsys, ["analyze", "keybo-lsb", "--no-time", "--json"])
    gauges = out["rows"]["keybo-lsb"]["gauges"]
    # every gauge printed in the text report carries the JSON value to 3dp
    for name in ("sfb", "scissor", "imbalance", "comfort"):
        assert f"{gauges[name]:.3f}" in text, f"{name} ({gauges[name]:.3f}) missing from text"
    # per-finger scissor also appears
    per_finger = out["rows"]["keybo-lsb"]["scissor_by_finger"]
    assert f"{per_finger['LP']:.4f}" in text


@pytest.mark.slow
def test_model_scores_reproduce_the_frozen_fits_exactly(capsys):
    """POSITIVE CONTROL: the three model-surface fits, EXACT vs all-gauge-table.json."""
    surfaces = pytest.importorskip("keybo.analysis.surfaces")
    if not surfaces.available_surfaces():
        pytest.skip("no model surfaces vendored or discoverable")
    out = _run(capsys, ["analyze", "keybo-lsb", "--ref", "qwerty30m", "--json"])
    for label, expected in FROZEN_FITS.items():
        key = "keybo-lsb" if label == "keybo-lsb" else "qwerty30m"
        scores = out["rows"][key]["model_scores"]
        assert scores["available"] is True, f"{label}: model scores unavailable"
        assert scores["baked_wpm"] == 90.0
        for surface, fit in expected.items():
            got = scores["surfaces"][surface]["fit"]
            assert got == fit, f"{label}/{surface}: expected {fit!r}, got {got!r}"


@pytest.mark.slow
def test_model_scores_default_to_tri_ps_freq_prior_and_are_labelled(capsys):
    """An unlabelled 'aalto score' is the ambiguity that cost this campaign a retraction."""
    surfaces = pytest.importorskip("keybo.analysis.surfaces")
    if not surfaces.available_surfaces():
        pytest.skip("no model surfaces vendored or discoverable")
    out = _run(capsys, ["analyze", "keybo-lsb", "--json"])
    scores = out["rows"]["keybo-lsb"]["model_scores"]
    assert set(scores["surfaces"]) == {
        "AALTO_TRI_PS_FREQ_PRIOR",
        "COMMUNITY_TRI_PS_FREQ_PRIOR",
        "POOL_TRI_PS_FREQ_PRIOR",
    }
    assert scores["frame"] == "geometry-only (g); the layout-independent b(ngram) term is excluded"
    for name, cell in scores["surfaces"].items():
        assert cell["surface"] == name  # each cell names its own surface


@pytest.mark.slow
def test_model_surface_flag_selects_other_families(capsys):
    """BASE / FREQ_PRIOR live behind a flag, and the label follows the selection."""
    surfaces = pytest.importorskip("keybo.analysis.surfaces")
    if "COMMUNITY_BASE" not in surfaces.available_surfaces():
        pytest.skip("BASE surfaces not available")
    out = _run(capsys, ["analyze", "keybo-lsb", "--model-family", "BASE", "--json"])
    scores = out["rows"]["keybo-lsb"]["model_scores"]
    assert "COMMUNITY_BASE" in scores["surfaces"]
    assert all("BASE" in name for name in scores["surfaces"])


@pytest.mark.slow
def test_target_wpm_does_not_silently_move_the_baked_model_columns(capsys):
    """The surfaces are BAKED at 90 wpm; a different --target-wpm must SAY so, not lie.

    The underlying per-seed models for 7 of the 8 surfaces no longer exist, so the
    columns cannot be re-evaluated at another WPM. Reporting them unchanged and
    unlabelled under `--target-wpm 110` would be a silently-wrong number.
    """
    surfaces = pytest.importorskip("keybo.analysis.surfaces")
    if not surfaces.available_surfaces():
        pytest.skip("no model surfaces vendored or discoverable")
    at90 = _run(capsys, ["analyze", "keybo-lsb", "--target-wpm", "90", "--json"])
    at110 = _run(capsys, ["analyze", "keybo-lsb", "--target-wpm", "110", "--json"])
    a = at90["rows"]["keybo-lsb"]["model_scores"]
    b = at110["rows"]["keybo-lsb"]["model_scores"]
    # the fits are identical (same baked surface) ...
    assert (
        a["surfaces"]["AALTO_TRI_PS_FREQ_PRIOR"]["fit"]
        == (b["surfaces"]["AALTO_TRI_PS_FREQ_PRIOR"]["fit"])
    )
    # ... and the mismatch is DECLARED rather than passed off as evaluated at 110.
    assert a["wpm_matches_request"] is True
    assert b["wpm_matches_request"] is False
    assert "90" in b["wpm_note"]


def test_missing_model_surfaces_degrade_gracefully(capsys, monkeypatch):
    """No surfaces on disk -> a clear message and the rest of the table, NOT a traceback."""
    from keybo.analysis import surfaces as S

    monkeypatch.setattr(S, "_search_dirs", lambda override=None: [])
    S.available_surfaces.cache_clear()
    try:
        rc = main(["analyze", "keybo-lsb", "--no-time", "--json"])
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        scores = out["rows"]["keybo-lsb"]["model_scores"]
        assert scores["available"] is False
        assert "not available" in scores["reason"].lower()
        # the rest of the table survived
        assert out["rows"]["keybo-lsb"]["gauges"]["sfb"] > 0.0
    finally:
        S.available_surfaces.cache_clear()


def test_surface_dir_override_rejects_a_missing_directory():
    with pytest.raises(SystemExit, match="surface"):
        main(["analyze", "keybo-lsb", "--surface-dir", "/nonexistent/surfaces/xyz"])


@pytest.mark.slow
def test_graded_scissor_flat_control_reproduces_the_flat_gauge_exactly(capsys):
    """The graded gauge at all-weights-1.0 MUST equal the flat gauge.

    This is what makes the graded column a strict generalization rather than a rival
    metric: same support, same denominator, and the weighting switched off recovers the
    incumbent number bit-for-bit. If it ever diverges, the graded path has picked up a
    different denominator (trap #9) and every graded number is off by a constant.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", FLAGSHIP_C3, "qwerty30m", "--json"])
    for name, row in out["rows"].items():
        assert row["scissor_graded"]["flat_control"] == row["gauges"]["scissor"], name


@pytest.mark.slow
def test_graded_scissor_is_at_least_the_flat_share_and_wide_at_least_narrow(capsys):
    """Weights are >= 1.0 multipliers, and wide support is a superset of narrow."""
    out = _run(capsys, ["analyze", "keybo-lsb", "qwerty30m", "--json"])
    for name, row in out["rows"].items():
        graded = row["scissor_graded"]
        assert graded["share"] >= graded["flat_control"], name
        assert graded["wide_support_share"] >= graded["flat_control"], name


@pytest.mark.slow
def test_graded_scissor_class_breakdown_is_an_exact_partition(capsys):
    """The tier|direction|adjacency classes sum to the graded share."""
    out = _run(capsys, ["analyze", "keybo-lsb", "qwerty30m", "--json"])
    for name, row in out["rows"].items():
        graded = row["scissor_graded"]
        assert sum(graded["by_class"].values()) == pytest.approx(
            graded["share"], rel=0, abs=1e-12
        ), name
        assert sum(graded["class_masses_unweighted"].values()) == pytest.approx(
            row["gauges"]["scissor"], rel=0, abs=1e-12
        ), name


#: FROZEN: all-gauge-table.json -> rows["graphite"].speed.surfaces[*], fit AND saved%.
#: A third layout, and the only control that also pins the saved-vs-qwerty arithmetic.
FROZEN_GRAPHITE_SURFACES = {
    "AALTO_TRI_PS_FREQ_PRIOR": (120126518391.9347, 2.2544137097470385),
    "COMMUNITY_TRI_PS_FREQ_PRIOR": (120343024464.43086, 4.290226189796353),
    "POOL_TRI_PS_FREQ_PRIOR": (122551005650.32454, 2.503612868116434),
}


@pytest.mark.slow
def test_graphite_model_fits_and_saved_percentages_reproduce_exactly(capsys):
    """POSITIVE CONTROL: fit AND saved% for a third layout.

    ``saved_vs_ref_pct`` is computed here from two fits; pinning it against the frozen
    board checks the ratio arithmetic and the reference layout, not just the numerator
    (trap #9's shape: a bit-exact numerator says nothing about the quotient).

    The **fits** are asserted bit-exactly. The **saved%** is asserted to 1e-12 relative,
    and the distinction is deliberate rather than a weakened control: the two drivers
    reduce the same two bit-identical fits in a different association order, so the last
    ULP of the quotient can differ (AALTO: 2.254413709747037 here vs 2.2544137097470385
    frozen — 6.6e-16 relative). Demanding ``==`` on the quotient would pin this driver's
    float association order, not its value.
    """
    surfaces = pytest.importorskip("keybo.analysis.surfaces")
    if not surfaces.available_surfaces():
        pytest.skip("no model surfaces vendored or discoverable")
    out = _run(capsys, ["analyze", "graphite", "--ref", "qwerty30m", "--json"])
    scores = out["rows"]["graphite"]["model_scores"]
    assert scores["available"] is True
    for name, (fit, saved) in FROZEN_GRAPHITE_SURFACES.items():
        cell = scores["surfaces"][name]
        assert cell["fit"] == fit, f"{name} fit (must be bit-exact)"
        assert cell["saved_vs_ref_pct"] == pytest.approx(saved, rel=1e-12), f"{name} saved%"


#: FROZEN: badscissor-spec.md §5 (flat share) and §5.2 (dy2 subtotal).
SPEC_BAD_SCISSOR = {"flagship-c3": 3.46985, "graphite": 4.66037}
SPEC_BAD_SCISSOR_DY2 = {"flagship-c3": 0.26570}


@pytest.mark.slow
def test_bad_scissor_is_reported_and_matches_the_sibling_specification(capsys):
    """POSITIVE CONTROL: the CLI's bad-scissor share vs the spec's own expected values."""
    out = _run(capsys, ["analyze", "flagship-c3", "graphite", "--no-time", "--json"])
    for label, expected in SPEC_BAD_SCISSOR.items():
        row = out["rows"][label]
        assert row["bad_scissor"]["share"] == pytest.approx(expected, abs=5e-5), label
        assert row["bad_scissor"]["attribution_rule"] == "all-to-descending-weaker-finger"
        assert "space-EXCLUDED" in row["bad_scissor"]["denominator"]


@pytest.mark.slow
def test_bad_scissor_decompositions_are_exact_partitions_in_the_cli(capsys):
    out = _run(capsys, ["analyze", "flagship-c3", "qwerty30m", "dvorak", "--no-time", "--json"])
    for name, row in out["rows"].items():
        bad = row["bad_scissor"]
        assert sum(bad["by_finger"].values()) == pytest.approx(bad["share"], rel=0, abs=1e-9), name
        assert sum(bad["by_cell"].values()) == pytest.approx(bad["share"], rel=0, abs=1e-9), name
        # both index fingers are structurally zero under this attribution rule
        assert bad["by_finger"]["L-index"] == 0.0 and bad["by_finger"]["R-index"] == 0.0, name


@pytest.mark.slow
def test_bad_scissor_dy2_subtotal_matches_the_specification(capsys):
    """The dy2 subtotal is the number that motivates the predicate — pin it end-to-end."""
    out = _run(capsys, ["analyze", "flagship-c3", "--no-time", "--json"])
    cells = out["rows"]["flagship-c3"]["bad_scissor"]["by_cell"]
    dy2 = sum(value for cell, value in cells.items() if cell.endswith("dy2"))
    assert dy2 == pytest.approx(SPEC_BAD_SCISSOR_DY2["flagship-c3"], abs=5e-5)


@pytest.mark.slow
def test_bad_scissor_scores_dvorak_too(capsys):
    """bad-scissor is charset-AGNOSTIC (no oxeylyzer board), so dvorak gets a number.

    Its denominator covers a different bigram set than a C30M layout's, so the value is a
    within-layout diagnostic and not cross-comparable — but it is not N/A.
    """
    out = _run(capsys, ["analyze", "dvorak", "--no-time", "--json"])
    row = next(r for r in out["rows"].values() if r["layout"].startswith("',.py"))
    assert row["bad_scissor"]["share"] == pytest.approx(5.80304, abs=5e-5)


@pytest.mark.slow
def test_bad_scissor_text_report_agrees_with_json(capsys):
    text_rc = main(["analyze", "flagship-c3", "--no-time"])
    assert text_rc == 0
    text = capsys.readouterr().out
    out = _run(capsys, ["analyze", "flagship-c3", "--no-time", "--json"])
    bad = out["rows"]["flagship-c3"]["bad_scissor"]
    assert f"{bad['share']:.4f}" in text
    assert f"{bad['by_finger']['L-pinky']:.4f}" in text


@pytest.mark.slow
def test_the_graded_scissor_orientation_weight_is_labelled_a_PRIOR(capsys):
    """The `down` weight is an orientation term; the served gauge cannot corroborate one.

    Every relational/geometric feature of the served bigram gauge is a function of the
    UNORDERED position pair (direction enters only via the landing-key one-hots), so a
    direction-of-travel effect is not representable. The weight is a declared preference and
    the report must say so, and must also publish the share WITHOUT it so a reader can see
    how much of the number the prior is carrying.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", "qwerty30m", "--no-time", "--json"])
    for name, row in out["rows"].items():
        term = row["scissor_graded"]["orientation_term"]
        assert term["weight"] == 1.5
        assert "PRIOR" in term["status"]
        assert "direction" in term["status"].lower()
        # dropping the prior can only lower the share (it is a >= 1.0 multiplier) and must
        # still sit at or above the flat gauge (the tier weights remain)
        assert (
            row["gauges"]["scissor"] <= term["share_without_it"] <= row["scissor_graded"]["share"]
        ), name
        assert "PRIOR" in row["scissor_graded"]["note"]


@pytest.mark.slow
def test_bad_scissor_carries_no_orientation_term(capsys):
    """bad-scissor is flat and its predicate is order-invariant, so no prior to declare.

    Its "the weaker finger is on the lower row" condition is a property of the unordered
    pair (0 order-dependent pairs of 900, asserted in tests/analysis/test_bad_scissor.py),
    NOT a direction of travel — so unlike the graded scissor column it needs no prior label.
    """
    out = _run(capsys, ["analyze", "keybo-lsb", "--no-time", "--json"])
    bad = out["rows"]["keybo-lsb"]["bad_scissor"]
    assert bad["severity"].startswith("flat")
    assert "orientation_term" not in bad


@pytest.mark.slow
def test_bad_scissor_caveats_reach_the_user_not_just_the_docstring(capsys):
    """The three caveats are user-visible, and the header does not assert the retracted mechanism.

    The mechanistic reading ("the weaker finger strains") is not identified ON THE AALTO SAMPLE:
    there the weak- and strong-descending groups share no bottom-row key, so any property of the
    two key groups is collinear with the label. The number is robust; the mechanism is not — so
    the header names the OPERATION and the caveats travel with it.

    The caveat must also NOT overstate the limit as structural — see
    ``tests/analysis/test_bad_scissor.py`` for the geometric counterexample that makes it
    empirical.
    """
    rc = main(["analyze", "flagship-c3", "--no-time", "--no-model-scores"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "lower key on a non-index finger" in text
    assert "WEAKER finger descends" not in text, "header asserts a mechanism that is not identified"
    # the three caveats, per spec §0
    assert "+0.41" in text and "[+0.23, +0.55]" in text
    assert "not robust" in text.lower()
    assert "96.6%" in text
    assert "not identified" in text
    # the limit is EMPIRICAL, not structural — the text must say so, and must not claim
    # that no amount of data could fix it
    assert "EMPIRICAL, not structural" in text
    assert "no amount of data" not in text
