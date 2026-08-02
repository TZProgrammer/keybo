"""The ``sg_dist`` gauge (BUILDMETRIC-1): frozen value, convention, and distinctness.

``sg_dist`` is the corpus-weighted first-to-third-key span. These tests pin it against the
BUILDMETRIC field table and, following GAUGEAUDIT-1's method, verify it is a GENUINELY DISTINCT
axis on a perturbed set — not a renamed ``sfs``, and not a hidden invariant like ``sfr`` / ``alt``
(which tie layouts while reading as agreement).
"""

from __future__ import annotations

import random

import pytest

from keybo.analysis.kmstats import STAT_NAMES, KmStats
from keybo.analysis.skipgram_span import _foil_spans, sg_dist, sg_dist_from_string
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.oxey import OxeyStyleScorer

KEYBO_LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
KEYBO_C30M = "fyu,.vgdnlhieaocstrmkj'q-bwpxz"
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
FLAGSHIP_C3 = "pyou'vgdnmheai.cstrlkjz,-wfbxq"
GRAPHITE = "bldwz'foujnrtsgyhaeixqmcvkp,.-"


# --- FROZEN VALUE: reconciled byte-for-byte to the BUILDMETRIC field table ------------------
#
# The ``corpora`` fixture loads the committed iWeb corpus (verified identical to the ``iweb``
# named dir). BUILDMETRIC's report table quotes blend-v1 (keybo-lsb 3.836); the value below is
# the SAME metric on the fixture's iWeb table, and the blend-v1 reconciliation is pinned
# separately in :func:`test_reconciles_to_the_buildmetric_blend_v1_field_table`.
FROZEN_IWEB = {
    KEYBO_LSB: 3.9524690209679565,
    QWERTY30M: 3.6672176609523244,
    GRAPHITE: 4.114173974635862,
}


@pytest.mark.parametrize("layout", list(FROZEN_IWEB))
def test_sg_dist_reproduces_the_frozen_value_exactly(corpora, layout):
    """POSITIVE CONTROL: exact, no tolerance — the metric is pinned, not approximately checked."""
    _bigrams, _skipgrams, trigrams = corpora
    assert sg_dist_from_string(layout, trigrams) == FROZEN_IWEB[layout]


def test_reconciles_to_the_buildmetric_blend_v1_field_table():
    """The report's headline numbers (blend-v1) reproduce to 3 decimals: the campaign anchor.

    BUILDMETRIC-1 field table: keybo-lsb 3.836, keybo-c30m 3.794, qwerty30m 3.680,
    flagship-c3 3.968. Read from the vendored blend-v1 corpus so a corpus swap cannot silently
    relabel the anchor.
    """
    trigrams = load_frequencies(str(production_corpus_dir("blend-v1") / "trigrams.txt"))
    expected = {KEYBO_LSB: 3.836, KEYBO_C30M: 3.794, QWERTY30M: 3.680, FLAGSHIP_C3: 3.968}
    for layout, want in expected.items():
        assert sg_dist_from_string(layout, trigrams) == pytest.approx(want, abs=5e-4)


# --- CONVENTION: the exact denominator, stated and pinned (trap #9) --------------------------


def test_denominator_is_the_space_inclusive_layout_restricted_trigram_mass(corpora):
    """sg_dist divides by the SAME mass ms/char does: trigrams on the layout, space INCLUDED.

    Space is load-bearing — the space bar sits far from the letter block, so excluding
    space-containing trigrams (the redirects/kmstats trigram convention) changes the number.
    This test proves the shipped gauge uses the space-inclusive mass by showing the two
    conventions genuinely differ and the shipped one matches the space-inclusive hand-computation.
    """
    _bigrams, _skipgrams, trigrams = corpora
    layout = Layout(KEYBO_LSB, ROW_STAGGERED_30)
    pos = {c: layout.pos(c) for c in KEYBO_LSB}
    pos[" "] = ROW_STAGGERED_30.space_position

    def hand_rolled(include_space: bool) -> float:
        keys = set(KEYBO_LSB) | ({" "} if include_space else set())
        num = 0.0
        den = 0
        for ng, f in trigrams.items():
            if len(ng) == 3 and all(ch in keys for ch in ng):
                num += f * ROW_STAGGERED_30.distance(pos[ng[0]], pos[ng[2]])
                den += f
        return num / den

    with_space = hand_rolled(True)
    without_space = hand_rolled(False)
    assert with_space != without_space, "space-containing trigrams must move the number"
    # The shipped gauge is the space-INCLUSIVE one.
    assert sg_dist(layout, trigrams) == with_space
    assert sg_dist(layout, trigrams) != without_space


def test_matches_the_served_model_feature_definition(corpora):
    """sg_dist is the layout aggregate of the model's served ``sg_distance`` feature.

    The per-trigram quantity MUST be identical to ``keybo.features``' ``sg_distance`` = the
    board distance from first to third key (they are the same geometry) — else the gauge would
    not be the layout-level shadow of what the trigram model already trains on.
    """
    from keybo.features.ngram import _trigram_level_from_positions

    _bigrams, _skipgrams, trigrams = corpora
    layout = Layout(KEYBO_LSB, ROW_STAGGERED_30)
    pos = {c: layout.pos(c) for c in KEYBO_LSB}
    pos[" "] = ROW_STAGGERED_30.space_position
    # spot-check several real trigrams: gauge's per-trigram phi == feature's sg_distance
    for ng in ("the", "and", "hea", "ing", "e t"):
        a, b, c = (pos[ch] for ch in ng)
        feat = _trigram_level_from_positions(ROW_STAGGERED_30, a, b, c)["sg_distance"]
        gauge_phi = ROW_STAGGERED_30.distance(a, c)
        assert feat == gauge_phi


# --- DISTINCTNESS: GAUGEAUDIT-1's method — a perturbed set, not one board --------------------


def _within_hand_permutations(base: str, n: int, seed: int) -> list[str]:
    """``n`` layouts that permute characters WITHIN each hand (the left/right 15 stay put).

    This is the perturbation GAUGEAUDIT-1 showed exposes hand-partition invariants (``sfr`` /
    ``alt`` are constant under it while every geometric gauge moves) — the candidate set a local
    search around a fixed hand partition actually explores, and the honest test bed for
    "is this a new axis or a relabelled one".
    """
    rng = random.Random(seed)
    left = list(range(15))
    right = list(range(15, 30))
    out = []
    for _ in range(n):
        chars = list(base)
        for side in (left, right):
            picked = [chars[i] for i in side]
            rng.shuffle(picked)
            for i, ch in zip(side, picked, strict=True):
                chars[i] = ch
        out.append("".join(chars))
    return out


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def test_sg_dist_varies_under_within_hand_permutation(corpora):
    """It is NOT a hidden invariant. sfr/alt are constant under this perturbation (GAUGEAUDIT-1);
    a real geometric gauge must move, or it ties layouts while reading as agreement."""
    _bigrams, _skipgrams, trigrams = corpora
    boards = _within_hand_permutations(KEYBO_LSB, 40, seed=0)
    values = [sg_dist_from_string(b, trigrams) for b in boards]
    spread = max(values) - min(values)
    assert spread > 1e-6, (
        f"sg_dist is invariant under within-hand permutation (spread {spread:.2e})"
    )


def test_sg_dist_is_not_within_one_percent_of_any_existing_gauge(corpora):
    """DISTINCTNESS (the brief's bar): on a perturbed set, sg_dist tracks no single existing
    gauge to within 1%. Formalized as correlation, because the gauges are percentages and
    sg_dist is key-widths — proportional-tracking, not raw value-proximity, is what "renamed
    gauge" would mean. BUILDMETRIC-1 measured max |corr| = 0.54 (with sfs) on its perturbation
    set; the bar here (|corr| < 0.99 against every gauge) is the falsifiable "not a relabelling".
    """
    bigrams, skipgrams, trigrams = corpora
    kms = KmStats(bigrams, skipgrams, trigrams)
    oxey = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    boards = _within_hand_permutations(KEYBO_LSB, 60, seed=7)

    sgd = [sg_dist_from_string(b, trigrams) for b in boards]
    # every existing corpus-sensitive gauge, on the same boards
    gauge_series: dict[str, list[float]] = {name: [] for name in STAT_NAMES}
    gauge_series["oxey-style"] = []
    for b in boards:
        stats = kms.stats(b)
        for name in STAT_NAMES:
            gauge_series[name].append(stats[name])
        gauge_series["oxey-style"].append(oxey.fitness(Layout(b, ROW_STAGGERED_30)))

    offenders = {}
    for name, series in gauge_series.items():
        if max(series) - min(series) < 1e-9:
            continue  # an invariant gauge (sfr/alt) can't be "the same axis" as a varying one
        c = abs(_corr(sgd, series))
        if c >= 0.99:
            offenders[name] = c
    assert not offenders, f"sg_dist is ~collinear with existing gauge(s): {offenders}"


def test_sg_dist_is_most_correlated_with_sfs_below_the_buildmetric_bar(corpora):
    """Corroborates BUILDMETRIC-1's headline: the closest gauge is a same-finger skip metric
    (sfs/sfs-dist), and even it stays well under collinearity — sg_dist meters the a→c span for
    ALL trigrams where those meter it only for the same-finger case."""
    bigrams, skipgrams, trigrams = corpora
    kms = KmStats(bigrams, skipgrams, trigrams)
    boards = _within_hand_permutations(KEYBO_LSB, 60, seed=11)
    sgd = [sg_dist_from_string(b, trigrams) for b in boards]
    corrs = {}
    for name in STAT_NAMES:
        series = [kms.stats(b)[name] for b in boards]
        if max(series) - min(series) >= 1e-9:
            corrs[name] = abs(_corr(sgd, series))
    top = max(corrs, key=corrs.get)
    assert top in ("sfs", "sfs-dist"), f"closest gauge is {top}, expected a same-finger skip"
    assert corrs[top] < 0.99


# --- FOILS: path_len_sq / max_hop are diagnostics, NEVER shipped gauges ----------------------


def test_foils_are_not_in_the_shipped_gauge_frame():
    """path_len_sq / max_hop are bigram-decomposable (95% / 69%, BUILDMETRIC-1) — the `reach`
    category error's cousins. They must never enter the analyze gauge frame."""
    from keybo.cli.analyze import GAUGE_NAMES

    assert "sg_dist" in GAUGE_NAMES
    assert "path_len_sq" not in GAUGE_NAMES
    assert "max_hop" not in GAUGE_NAMES


def test_foil_spans_compute_and_differ_from_sg_dist(corpora):
    """The foils are available for the distinctness contrast, and are genuinely different
    quantities (a sanity check that the diagnostic is not accidentally aliased to sg_dist)."""
    _bigrams, _skipgrams, trigrams = corpora
    layout = Layout(KEYBO_LSB, ROW_STAGGERED_30)
    foils = _foil_spans(layout, trigrams)
    assert set(foils) == {"path_len_sq", "max_hop"}
    assert foils["path_len_sq"] > 0 and foils["max_hop"] > 0
    assert foils["path_len_sq"] != sg_dist(layout, trigrams)
    assert foils["max_hop"] != sg_dist(layout, trigrams)


# --- EDGE CASES: normalization, empties -------------------------------------------------------


def test_empty_corpus_yields_zero_not_a_crash():
    assert sg_dist(Layout(KEYBO_LSB, ROW_STAGGERED_30), {}) == 0.0


def test_layout_covering_no_corpus_trigram_normalizes_to_zero():
    """A layout that shares no trigram with the corpus scores 0, not ZeroDivisionError."""
    assert sg_dist(Layout(KEYBO_LSB, ROW_STAGGERED_30), {"XYZ": 10}) == 0.0


def test_bigram_and_other_length_entries_are_ignored(corpora):
    """Only length-3 entries participate (the table can carry other n-grams)."""
    _bigrams, _skipgrams, trigrams = corpora
    layout = Layout(KEYBO_LSB, ROW_STAGGERED_30)
    baseline = sg_dist(layout, trigrams)
    polluted = dict(trigrams)
    polluted["th"] = 10**9  # a bigram with a huge count must not move the trigram gauge
    polluted["once"] = 10**9  # nor a 4-gram
    assert sg_dist(layout, polluted) == baseline
