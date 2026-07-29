"""`finger-travel` and `off-home`: the partitions, the guards, and the prereg's own claims.

Structured so that the tests which could pass on a BROKEN metric are separated from the ones
that could not:

* **§1 exactness** — the shares sum to 100. Necessary, and on its own nearly worthless: it
  passes for ``travel(f) = 1`` for every finger. Never cite it alone as evidence the metric works.
* **§2 non-triviality** — the metric discriminates across layouts, charges the RIGHT finger, and a
  mutant that charges the wrong one is caught. This is what makes §1 mean something.
* **§3 the D4 guard** — an undeclared label RAISES. ``bad_scissor._partition`` was fixed for
  exactly this on 2026-07-28: a drifted label was appended, a caller printing a fixed column list
  showed 0.0000 for it, and 0.46584 pp vanished from a 4.11684 total **while every
  exact-partition test still passed**, because they sum ``.values()`` and never the printed
  columns. §1-style tests are structurally blind to it.
* **§4 the prereg's own arguments, as code** — ``docs/finger-travel-preregistration.md`` §1.5
  argues return-to-home is ``2x`` static and so cannot change a share, and §1.2 argues the
  headline degenerates to the static form. Both are asserted here rather than believed.
* **§5 the harness is positive-controlled** — ``keybo.testkit``: the module under test is the one
  in THIS tree (this repo's venv carries an editable ``.pth`` into a different clone), and the
  suite provably fails when the metric is broken.
"""

from __future__ import annotations

import math

import pytest

from keybo.analysis.finger_travel import (
    FINGER_ORDER,
    HOME_POSITION,
    HOME_ROW,
    FingerTravel,
    OffHomeUsage,
    dispersion,
    finger_label,
    letter_mass,
)

#: Registry layouts plus the campaign boards, derived from the shipped registries rather than
#: retyped: two of two hand-transcriptions by a prior arm were wrong.
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.geometry import ROW_STAGGERED_30 as GEOM
from keybo.geometry import ROW_STAGGERED_31 as GEOM31
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.testkit import assert_discriminating, assert_module_under

ALL_LAYOUTS = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
TOLERANCE = 1e-9


def _layout(lay30: str, geometry=GEOM) -> Layout:
    return Layout(lay30, geometry)


def travel_shares_of_bigrams(layout: Layout, bigram_freqs: dict[str, int]) -> dict[str, float]:
    """Headline (lag-1) shares over a hand-built bigram corpus — for lag-1 vs lag-2 contrasts."""
    return FingerTravel(bigram_freqs).shares(layout)


@pytest.fixture(scope="module")
def bigrams(corpora):
    return corpora[0]


# --- §1 exactness (necessary, NOT sufficient — see the module docstring) ------------------


@pytest.mark.parametrize("name", sorted(ALL_LAYOUTS))
def test_travel_shares_are_an_exact_partition_of_100(bigrams, name):
    shares = FingerTravel(bigrams).shares(_layout(ALL_LAYOUTS[name]))
    assert set(shares) == set(FINGER_ORDER)
    assert sum(shares.values()) == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("name", sorted(ALL_LAYOUTS))
@pytest.mark.parametrize("convention", OffHomeUsage.CONVENTIONS)
def test_usage_is_an_exact_partition_and_off_plus_on_equals_usage(bigrams, name, convention):
    """The shared-denominator property: per finger, ``off_home + on_home == usage`` exactly.

    Under ``restricted`` the eight cells sum to 100. Under ``letter-freqs`` they sum to
    ``coverage_pct`` instead, because that convention keeps untypeable corpus mass in the
    denominator — asserted against the published field rather than tolerated as "about 100".
    """
    off_home = OffHomeUsage(bigrams, convention=convention)
    layout = _layout(ALL_LAYOUTS[name])
    usage, off, on = off_home.usage(layout), off_home.off_home(layout), off_home.on_home(layout)
    expected_sum = off_home.coverage_pct(layout)
    assert sum(usage.values()) == pytest.approx(expected_sum, abs=1e-9)
    for label in FINGER_ORDER:
        assert off[label] + on[label] == pytest.approx(usage[label], abs=1e-9), label
    assert sum(off.values()) + sum(on.values()) == pytest.approx(expected_sum, abs=1e-9)
    if convention == "restricted":
        assert expected_sum == pytest.approx(100.0, abs=1e-9), (
            "the restricted denominator drops untypeable mass from BOTH sides, so it must be 100"
        )


@pytest.mark.parametrize("name", sorted(ALL_LAYOUTS))
def test_letter_freqs_usage_sums_to_coverage_and_NOT_to_100(bigrams, name):
    """Pinned so the shortfall reads as a charset gap, not as a partition bug someone "fixes".

    The two conventions must also actually DIFFER — if a future edit collapsed them, every
    reconciliation against ``DislocationScorer`` would silently change meaning.
    """
    layout = _layout(ALL_LAYOUTS[name])
    unrestricted = OffHomeUsage(bigrams, convention="letter-freqs")
    restricted = OffHomeUsage(bigrams, convention="restricted")
    coverage = unrestricted.coverage_pct(layout)
    assert 90.0 < coverage < 100.0, "every board in the field misses some corpus mass"
    assert sum(unrestricted.usage(layout).values()) == pytest.approx(coverage, abs=1e-9)
    assert restricted.coverage_pct(layout) == pytest.approx(100.0, abs=1e-9)
    assert unrestricted.usage(layout)["L-pinky"] != pytest.approx(
        restricted.usage(layout)["L-pinky"], abs=1e-6
    ), "the conventions must remain distinguishable"


def test_an_unknown_denominator_convention_is_REFUSED_not_guessed(bigrams):
    with pytest.raises(ValueError, match="unknown denominator convention"):
        OffHomeUsage(bigrams, convention="unrestricted")


#: The parent's independently-computed first cut, on **blend-v1**, ``_letter_freqs`` convention:
#: ``(pinky usage, on-home, off-home)``. A genuine POSITIVE CONTROL — computed by a different
#: agent, by a different route, before this module existed.
PARENT_FIRST_CUT = {
    "keybo-lsb": (12.82, 5.73, 7.09),
    "graphite": (15.13, 12.38, 2.75),
    "semimak": (17.57, 12.50, 5.06),
    "qwerty30m": (10.82, 7.62, 3.21),
}


@pytest.fixture(scope="module")
def blend_bigrams():
    """blend-v1 bigrams — the CLI's PRODUCTION default, which is NOT the ``corpora`` fixture.

    ⚠ ``tests/conftest.py``'s ``corpora`` reads ``data/corpus/`` (iWeb); ``keybo analyze``
    defaults to ``data/corpus/blend-v1/``. Any test comparing against a number quoted from the
    CLI or from a campaign board must load blend-v1 explicitly — reusing ``corpora`` silently
    compares two different corpora, which is how a reproduction "fails" for the wrong reason.
    (It did here first time round: semimak read 17.23 against an expected 17.57.)
    """
    from keybo.data.corpus import load_frequencies
    from tests.conftest import CORPUS_DIR

    return load_frequencies(str(CORPUS_DIR / "blend-v1" / "bigrams.txt"))


@pytest.mark.parametrize("name", sorted(PARENT_FIRST_CUT))
def test_the_letter_freqs_convention_REPRODUCES_the_independent_first_cut(blend_bigrams, name):
    """Positive control: agreement with numbers this code did not produce.

    Reproduces to <0.005 pp, which identifies the parent's convention as ``_letter_freqs``
    exactly. The default ``restricted`` convention does NOT reproduce it (keybo-lsb reads 8.00 vs
    7.09) — that gap is a denominator choice, and it is why the convention is explicit.
    """
    layout = _layout(ALL_LAYOUTS[name])
    pinky = OffHomeUsage(blend_bigrams, convention="letter-freqs").report(layout)["pinky"]
    expected_usage, expected_on, expected_off = PARENT_FIRST_CUT[name]
    assert pinky["usage"] == pytest.approx(expected_usage, abs=0.01)
    assert pinky["on_home"] == pytest.approx(expected_on, abs=0.01)
    assert pinky["off_home"] == pytest.approx(expected_off, abs=0.01)


def test_the_restricted_convention_does_NOT_reproduce_it_and_that_is_the_whole_point(
    blend_bigrams,
):
    """If both conventions agreed, making the choice explicit would be pointless ceremony."""
    layout = _layout(ALL_LAYOUTS["keybo-lsb"])
    restricted = OffHomeUsage(blend_bigrams, convention="restricted").report(layout)["pinky"]
    assert restricted["off_home"] == pytest.approx(8.0015, abs=0.01)
    assert abs(restricted["off_home"] - PARENT_FIRST_CUT["keybo-lsb"][2]) > 0.5, (
        "the two conventions differ by ~0.9 pp on this board — a real fork, not rounding"
    )


def test_report_cannot_hand_back_shares_without_the_absolute_total(bigrams):
    """Normalizing destroys the level, so the level travels WITH the shares (prereg §1.3).

    Two layouts can share every percentage and differ in total travel; a shares-only table is
    the ``saved_vs_ref_pct`` coverage artifact this ledger already registered.
    """
    report = FingerTravel(bigrams).report(_layout(NAMED_LAYOUTS["qwerty"]))
    assert set(report["shares"]) == set(FINGER_ORDER)
    assert report["total"] > 0.0
    assert "dispersion" in report and "model" in report and "denominator" in report


# --- §2 non-triviality: what makes §1 mean something --------------------------------------


def test_travel_shares_DISCRIMINATE_across_layouts(bigrams):
    """A metric that sums to 100 identically for every layout would pass §1 and be useless."""
    travel = FingerTravel(bigrams)
    for label in ("L-pinky", "R-index", "R-pinky"):
        assert_discriminating(
            [travel.shares(_layout(lay))[label] for lay in ALL_LAYOUTS.values()],
            f"travel share of {label} across {len(ALL_LAYOUTS)} layouts",
        )


def test_travel_totals_DISCRIMINATE_across_layouts(bigrams):
    travel = FingerTravel(bigrams)
    assert_discriminating(
        [travel.total(_layout(lay)) for lay in ALL_LAYOUTS.values()],
        "absolute travel total across layouts",
    )


def test_travel_charges_the_LANDING_finger_on_a_hand_alternating_bigram():
    """The attribution rule, on a corpus of exactly one bigram, so nothing else can explain it.

    ``qp`` on qwerty: ``q`` is L-pinky, ``p`` is R-pinky. Different fingers, so the charge is
    the MODELLED from-home term and it goes to the landing finger (``p``'s R-pinky) alone.
    """
    layout = _layout(NAMED_LAYOUTS["qwerty"])
    charged = FingerTravel({"qp": 1000}).per_finger(layout)
    expected = GEOM.distance((5, HOME_ROW), layout.pos("p"))
    assert charged["R-pinky"] == pytest.approx(1000 * expected)
    assert charged["L-pinky"] == 0.0, "the DEPARTING finger is not charged for someone else's move"
    assert sum(charged.values()) == pytest.approx(1000 * expected)


def test_travel_charges_the_OBSERVED_distance_on_a_same_finger_bigram():
    """``qa`` on qwerty: both L-pinky, so the charge is dist(q, a) — motion the corpus saw."""
    layout = _layout(NAMED_LAYOUTS["qwerty"])
    charged = FingerTravel({"qa": 1000}).per_finger(layout)
    expected = GEOM.distance(layout.pos("q"), layout.pos("a"))
    assert charged["L-pinky"] == pytest.approx(1000 * expected)
    assert sum(charged.values()) == pytest.approx(1000 * expected)
    # and it is genuinely the observed motion, not a from-home coincidence: `a` IS L-pinky home,
    # so dist(q,a) == dist(home,q); use `qz` where the two differ to prove the branch is live.
    charged_qz = FingerTravel({"qz": 1000}).per_finger(layout)
    observed = GEOM.distance(layout.pos("q"), layout.pos("z"))
    from_home = GEOM.distance((-5, HOME_ROW), layout.pos("z"))
    assert observed != pytest.approx(from_home), "test would be vacuous if these agreed"
    assert charged_qz["L-pinky"] == pytest.approx(1000 * observed)


def test_a_mutant_that_charges_the_WRONG_finger_changes_the_shares(bigrams):
    """The mirror-image mutant still sums to 100 — so only a per-cell check can catch it.

    This is the positive control for §1: it demonstrates that summing to 100 is invariant under
    a fatal error, and that the per-finger assertions above are the part doing real work.
    """
    layout = _layout(NAMED_LAYOUTS["qwerty"])
    real = FingerTravel(bigrams).shares(layout)
    mirrored = {
        label: real[
            label.replace("L-", "R-") if label.startswith("L-") else label.replace("R-", "L-")
        ]
        for label in FINGER_ORDER
    }
    assert sum(mirrored.values()) == pytest.approx(100.0, abs=1e-9), (
        "the mutant is share-preserving by construction; that is the point"
    )
    assert mirrored != pytest.approx(real), "mirroring must be detectable per-cell"


def test_off_home_separates_two_layouts_with_the_SAME_pinky_usage(bigrams):
    """The user's claim needs off-home to move when total does not. Construct that case.

    Swap a pinky-column home key with a pinky-column off-home key: total pinky usage is
    unchanged (the same two characters stay on pinky columns) while off-home usage moves.
    """
    off_home = OffHomeUsage(bigrams)
    base = _layout(NAMED_LAYOUTS["qwerty"])
    swapped = _layout(NAMED_LAYOUTS["qwerty"])
    swapped.swap("q", "a")  # both left-pinky column; q is row 3, a is row 2
    assert off_home.usage(swapped)["L-pinky"] == pytest.approx(off_home.usage(base)["L-pinky"])
    assert off_home.off_home(swapped)["L-pinky"] != pytest.approx(
        off_home.off_home(base)["L-pinky"]
    ), "off-home must move where total does not — otherwise the metric adds nothing"


def test_off_fraction_is_NOT_a_partition_and_the_docstring_says_so(bigrams):
    """Pinned so nobody later 'fixes' it into one: each cell has its own denominator.

    Summing it produces a plausible-looking number in the low hundreds, which is exactly the
    kind of wrong constant that survives review because it looks like a percentage.
    """
    fraction = OffHomeUsage(bigrams).off_fraction(_layout(NAMED_LAYOUTS["qwerty"]))
    assert sum(fraction.values()) > 150.0, "if this ever sums to ~100 the metric changed meaning"
    assert "NOT a partition" in (OffHomeUsage.off_fraction.__doc__ or "")


def test_an_unused_finger_reports_zero_off_fraction_rather_than_dividing_by_zero():
    """A one-bigram corpus leaves six fingers with no usage at all."""
    fraction = OffHomeUsage({"qp": 10}).off_fraction(_layout(NAMED_LAYOUTS["qwerty"]))
    assert fraction["L-middle"] == 0.0
    assert all(math.isfinite(value) for value in fraction.values())


# --- §3 the BSAUDIT-1 D4 guard: an undeclared label must RAISE ----------------------------


def test_charging_an_undeclared_finger_label_RAISES_instead_of_appending_it(bigrams):
    """The failure mode §1-style tests are structurally blind to (see the module docstring)."""
    travel = FingerTravel(bigrams)
    charged = dict.fromkeys(FINGER_ORDER, 0.0)
    with pytest.raises(ValueError) as excinfo:
        travel._charge(charged, "R-little", 1.0, "a drifted labeller")
    message = str(excinfo.value)
    assert "R-little" in message, "must name the offending label"
    assert "R-pinky" in message, "must name the declared set so the drift is diagnosable"
    assert sum(charged.values()) == 0.0, "the dict must NOT have grown"


def test_all_eight_labels_are_pinned_exactly_and_in_column_order(bigrams):
    """Set equality AND order: callers print FINGER_ORDER as table columns."""
    for got in (
        FingerTravel(bigrams).shares(_layout(NAMED_LAYOUTS["qwerty"])),
        OffHomeUsage(bigrams).usage(_layout(NAMED_LAYOUTS["qwerty"])),
    ):
        assert list(got) == list(FINGER_ORDER)
    assert len(FINGER_ORDER) == 8, "eight fingers, or every exactness claim above is vacuous"


def test_the_thumb_has_no_travel_cell(bigrams):
    """Space is a fixed key at (0,0): a thumb cell would be identically 0.0 on every layout."""
    assert not any("thumb" in label for label in FINGER_ORDER)
    with pytest.raises(ValueError, match="thumb"):
        finger_label(GEOM, 0)
    # and space-touching bigrams are excluded from the denominator entirely (trap #9)
    assert letter_mass({"a ": 100, " a": 100}, _layout(NAMED_LAYOUTS["qwerty"])) == {}


def test_the_metric_asks_the_geometry_so_K31_column_6_is_a_pinky():
    """A hardcoded ``abs(col) == 5`` pinky test would silently mislabel the K31 quote slot."""
    assert finger_label(GEOM31, 6) == "R-pinky"
    assert finger_label(GEOM, 5) == "R-pinky"
    assert finger_label(GEOM, -1) == "L-index", "index owns columns 1 AND 2"


# --- §4 the prereg's own arguments, asserted rather than believed ------------------------


@pytest.mark.parametrize("name", sorted(ALL_LAYOUTS))
def test_return_to_home_is_exactly_2x_static_and_so_CANNOT_change_a_share(bigrams, name):
    """Prereg §1.5, as code: (c) is (a) in different units, not an independent third option.

    If this fails, the prereg's argument for shipping (a) vs the lag-resolved path as THE
    sensitivity check is wrong, and a third column is owed.
    """
    travel = FingerTravel(bigrams)
    layout = _layout(ALL_LAYOUTS[name])
    static, returned = travel.static_per_finger(layout), travel.return_home_per_finger(layout)
    for label in FINGER_ORDER:
        assert returned[label] == pytest.approx(2.0 * static[label], rel=1e-12), label
    static_total, return_total = sum(static.values()), sum(returned.values())
    for label in FINGER_ORDER:
        assert 100.0 * returned[label] / return_total == pytest.approx(
            100.0 * static[label] / static_total, rel=1e-12
        ), f"{label}: a positive scalar multiple must cancel in the ratio"


@pytest.mark.parametrize("name", sorted(ALL_LAYOUTS))
def test_the_headline_DEGENERATES_to_the_static_form_when_same_finger_travel_is_removed(
    bigrams, name
):
    """Prereg §1.2: the headline is a strict refinement of the existing quantity, not a rival.

    With the same-finger branch suppressed, every charge is ``dist(home, landing)`` weighted by
    second-of-bigram frequency — the static form under a different weighting. The expectation is
    rebuilt here from the corpus by a SECOND, independently written loop, so agreement is a real
    check rather than the implementation agreeing with itself.
    """
    travel = FingerTravel(bigrams)
    layout = _layout(ALL_LAYOUTS[name])
    full = travel.per_finger(layout)
    observed_only = travel.per_finger(layout, same_finger_only=True)
    modelled_only = {label: full[label] - observed_only[label] for label in FINGER_ORDER}

    expected = dict.fromkeys(FINGER_ORDER, 0.0)
    for bigram, freq in bigrams.items():
        if len(bigram) != 2 or " " in bigram:
            continue
        if not all(layout.has_key(character) for character in bigram):
            continue
        first, second = layout.pos(bigram[0]), layout.pos(bigram[1])
        if GEOM.same_finger(first[0], second[0]):
            continue
        label = finger_label(GEOM, second[0])
        expected[label] += freq * GEOM.distance(HOME_POSITION[label], second)

    for label in FINGER_ORDER:
        assert modelled_only[label] == pytest.approx(expected[label], rel=1e-12), label
    assert sum(observed_only.values()) < sum(full.values()), (
        "the observed branch must be a strict subset of the total, or the split is broken"
    )
    assert sum(observed_only.values()) > 0.0, "…and it must be non-empty, or the branch is dead"


@pytest.mark.parametrize("name", sorted(ALL_LAYOUTS))
def test_the_observed_and_modelled_branches_sum_to_the_headline(bigrams, name):
    """The split the report's ``observed_fraction_pct`` publishes must be exact."""
    travel = FingerTravel(bigrams)
    layout = _layout(ALL_LAYOUTS[name])
    report = travel.report(layout)
    observed = sum(travel.per_finger(layout, same_finger_only=True).values())
    assert report["observed_fraction_pct"] == pytest.approx(100.0 * observed / report["total"])
    assert 0.0 < report["observed_fraction_pct"] < 100.0, (
        "both branches must be live, or the metric is secretly one of its own variants"
    )


def _reversed_corpus(table: dict[str, int]) -> dict[str, int]:
    """Reverse every n-gram, layout fixed — the CORRECT instrument for testing direction.

    ⚠ A LEFT-RIGHT MIRROR is **not** a valid direction test: it maps the finger-index ordering
    onto itself and so cannot move a direction metric by construction. (A sibling agent nearly
    published the opposite conclusion off a mirror test.)
    """
    out: dict[str, int] = {}
    for ngram, freq in table.items():
        out[ngram[::-1]] = out.get(ngram[::-1], 0) + freq
    return out


def test_the_11_kmstats_gauges_are_EXACTLY_direction_blind(bigrams, corpora):
    """The control for the test below: the incumbent frame cannot see stroke order at all.

    Independently re-derived here rather than taken on trust — every delta is exactly
    ``0.00e+00`` over all 11 gauges. This is what makes travel's direction-sensitivity
    (next test) a statement about the FRAME's blind spot rather than about a corpus quirk.
    """
    from keybo.analysis.kmstats import KmStats

    _bigrams, skipgrams, trigrams = corpora
    layout = ALL_LAYOUTS["graphite"]
    forward = dict(KmStats(_bigrams, skipgrams, trigrams).stats(layout))
    backward = dict(
        KmStats(
            _reversed_corpus(_bigrams), _reversed_corpus(skipgrams), _reversed_corpus(trigrams)
        ).stats(layout)
    )
    assert len(forward) == 11, "eleven kmstats gauges, or the claim below is mis-scoped"
    for gauge, value in forward.items():
        assert backward[gauge] == pytest.approx(value, abs=1e-12), (
            f"{gauge} moved under corpus reversal — the direction-blindness control is broken"
        )


def test_travel_IS_direction_sensitive_but_ONLY_via_the_MODELLED_branch(bigrams):
    """Travel moves under corpus reversal where all 11 kmstats gauges cannot — with a caveat.

    Two facts, and the second is what stops the first from being an overclaim:

    1. Travel **does** move (≈+2.9% total, ≈4.2 pp on a per-finger share, and it REORDERS 10 of
       15 layouts), so it carries a channel the incumbent frame provably cannot express.
    2. That movement lives **entirely in the MODELLED from-home branch** — the observed
       same-finger branch moves by **exactly zero**, because ``dist(k1, k2)`` is symmetric per
       pair. So the direction-sensitivity is a property of the return-model ASSUMPTION (which
       key is the *landing* key), not an observed physical asymmetry.

    Pinning (2) alongside (1) is deliberate: "travel sees direction where the frame is blind" is
    true and would survive review on its own, while being a claim about an assumption rather than
    about the corpus. That is exactly the shape of the wrong-constant-behind-a-true-conclusion
    failure this campaign has hit six times.
    """
    layout = _layout(ALL_LAYOUTS["graphite"])
    forward, backward = FingerTravel(bigrams), FingerTravel(_reversed_corpus(bigrams))

    observed_forward = sum(forward.per_finger(layout, same_finger_only=True).values())
    observed_backward = sum(backward.per_finger(layout, same_finger_only=True).values())
    assert observed_backward == pytest.approx(observed_forward, rel=1e-12), (
        "the OBSERVED branch must be exactly direction-blind: dist(k1,k2) is symmetric per pair"
    )

    total_forward, total_backward = forward.total(layout), backward.total(layout)
    assert abs(total_backward - total_forward) / total_forward > 0.01, (
        "the total must move by >1% — if it does not, travel adds no direction channel"
    )
    shares_forward, shares_backward = forward.shares(layout), backward.shares(layout)
    assert (
        max(abs(shares_backward[label] - shares_forward[label]) for label in FINGER_ORDER) > 1.0
    ), "and a per-finger share must move by >1 pp, or the channel cannot discriminate"


def test_off_home_IS_exactly_direction_blind_because_it_is_a_unigram_metric(bigrams):
    """The honest limit of the second metric: it cannot see stroke order even in principle."""
    layout = _layout(ALL_LAYOUTS["graphite"])
    forward = OffHomeUsage(bigrams).report(layout)["pinky"]
    backward = OffHomeUsage(_reversed_corpus(bigrams)).report(layout)["pinky"]
    assert backward["off_home"] == pytest.approx(forward["off_home"], abs=1e-12)
    assert backward["usage"] == pytest.approx(forward["usage"], abs=1e-12)


def test_the_gauges_travel_would_be_redundant_with_are_named_in_the_docstring():
    """Guard the redundancy disclosure: travel_total is |r|≈0.97 with sfb-dist, and it says so."""
    import keybo.analysis.finger_travel as module

    doc = module.__doc__ or ""
    assert "sfb-dist" in doc, "the near-collinear incumbent must be named, not left for a reader"


def test_the_slowness_weighted_variant_is_a_SEPARATE_column_that_actually_differs(bigrams):
    """Prereg §1.4: shipped beside the headline, never as it — and it must not be a no-op."""
    travel = FingerTravel(bigrams)
    layout = _layout(NAMED_LAYOUTS["qwerty"])
    weighted, plain = travel.slowness_weighted_shares(layout), travel.shares(layout)
    assert sum(weighted.values()) == pytest.approx(100.0, abs=1e-9)
    assert weighted["L-pinky"] > plain["L-pinky"], "pinky slowness 1.43 must raise its share"
    assert weighted != pytest.approx(plain)


def test_lag2_resolves_a_same_finger_return_that_bigrams_CANNOT_see():
    """The sensitivity variant's whole justification, on a hand-built trigram.

    ``qwq``: the left pinky types ``q``, another finger interposes ``w``, and the pinky comes
    back to ``q``. At lag 1 the returning ``q`` looks like a fresh from-home arrival; at lag 2
    the corpus says the finger was already ON ``q``, so its true motion is **zero**.

    Note which presses a trigram actually charges: only the two TRANSITIONS ``q->w`` and
    ``w->q``. The leading ``q`` is a departure, never a landing — its own arrival belongs to the
    preceding trigram, which a one-trigram corpus does not contain. Getting that wrong is how a
    plausible constant lands in a test, so the expectation below is derived, not asserted.
    """
    layout = _layout(NAMED_LAYOUTS["qwerty"])
    # DERIVE the labels; `w` is column -4, which is the RING finger, not the middle.
    pinky, interposed = (
        finger_label(GEOM, layout.pos("q")[0]),
        finger_label(GEOM, layout.pos("w")[0]),
    )
    assert pinky == "L-pinky" and interposed == "L-ring", "fixture assumption, made explicit"
    from_home_w = GEOM.distance(HOME_POSITION[interposed], layout.pos("w"))
    assert from_home_w > 0.0, "guard: the fixture would be vacuous if w were a home key"

    # Lag 2 sees the pinky was already on `q`, so its return charge is dist(q,q) == 0 — and the
    # ONLY charge in this trigram is the interposed finger's from-home arrival.
    lag2 = FingerTravel({}).lag2_shares(layout, {"qwq": 1000})
    assert lag2[pinky] == pytest.approx(0.0), "the resolved return must cost nothing"
    assert lag2[interposed] == pytest.approx(100.0), "so it is the whole of the trigram's travel"

    # A lag-1 reading of the same text charges the returning pinky from HOME — the resolution the
    # bigram table structurally cannot have. That gap is the variant's whole justification.
    lag1 = travel_shares_of_bigrams(layout, {"qw": 1000, "wq": 1000})
    assert lag1[pinky] > lag2[pinky], (
        "lag 1 must OVERSTATE the returning pinky; if it does not, the lag-2 branch is inert"
    )


def test_dispersion_reports_concentration_and_is_balanced_on_a_flat_vector():
    flat = dict.fromkeys(FINGER_ORDER, 12.5)
    stats = dispersion(flat)
    assert stats["gini"] == pytest.approx(0.0)
    assert stats["lr_ratio"] == pytest.approx(1.0)
    assert stats["max_share"] == pytest.approx(12.5)
    assert stats["pinky_share"] == pytest.approx(25.0)
    lopsided = {**dict.fromkeys(FINGER_ORDER, 0.0), "L-pinky": 100.0}
    assert dispersion(lopsided)["gini"] > 0.8


def test_off_home_keys_names_the_characters_behind_the_number(bigrams):
    """An auditable-by-eye control: qwerty's pinky off-home keys are q/p on r3 and z//` on r1."""
    layout = _layout(NAMED_LAYOUTS["qwerty"])
    keys = OffHomeUsage(bigrams).off_home_keys(layout, ("L-pinky", "R-pinky"))
    assert set(keys) == {"q(r3)", "p(r3)", "z(r1)", "/(r1)"}


# --- §5 the harness itself is positive-controlled ----------------------------------------


def test_the_module_under_test_is_the_one_in_THIS_tree():
    """This repo's venv carries an editable ``.pth`` into a DIFFERENT clone's ``src``.

    Verified today: the venv interpreter resolves ``keybo`` to the shared clone unless
    ``PYTHONPATH`` puts this tree first, so a probe can test the wrong code while every printed
    path looks right. Ask the module where it actually lives.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src"
    assert_module_under("keybo.analysis.finger_travel", root)
    assert_module_under("keybo.geometry", root)


def test_the_module_docstring_keeps_the_descriptor_disclaimer():
    """Guard the wording: a distance is not a time and not a comfort claim."""
    import keybo.analysis.finger_travel as module

    doc = module.__doc__ or ""
    assert "GEOMETRIC DESCRIPTORS" in doc
    assert "NOT times and NOT comfort claims" in doc
    assert "+0.41 ms" in doc, "the registered frequency-beats-geometry caveat must travel along"
