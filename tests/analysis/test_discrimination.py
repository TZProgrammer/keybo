"""A11: the hand-partition invariance is ENFORCED where it can catch a real error.

``sfr``, ``alt`` and ``imbalance`` are EXACTLY constant under within-hand permutation (measured
spread 0.000e+00 over 25 variants of ``keybo-lsb``; the other twelve gauges of the fifteen-gauge
frame move by 1.9 to 87.5 units). ``keybo.verdicts.all_distinct`` was written for exactly this — its
docstring names those three gauges and says "run it before crediting a per-gauge win count" — and it
had ZERO production callers.

THE BOUNDARY WHERE IT CATCHES SOMETHING, which is the part that had to be found rather than
assumed: ``keybo analyze`` prints the fifteen gauges as one row per layout, i.e. as a COMPARISON.
Four of this campaign's own registry boards — ``keybo-lsb``, ``keybo-lsb+lm``, ``flagship-c3``,
``archive-1843`` — share the left-hand charset ``"',-.aehijkopuyz"``, so all four print an identical
``alt``/``imbalance`` pair, and every C30M board prints an identical ``sfr``. On the page those
repeated numbers are indistinguishable from agreement. That is also why
``tests/cli/test_analyze_allgauge.py`` check-ins the same three literals for three frozen boards:
a correct regression pin that nothing distinguished from a discrimination test.

So two things are wired, at the point where the numbers become a comparison:

1. every tie is CLASSIFIED — forced (a declared invariance accounts for it, named) vs coincidental;
2. a declared invariant that VARIES within one hand partition RAISES, because that makes every
   statement scoped to the declared set false while reading as true — and reads as a gauge that got
   BETTER at telling layouts apart, which is the kind of news nobody investigates.
"""

from __future__ import annotations

import json
import random

import pytest

from keybo.analysis.discrimination import (
    CHARSET_INVARIANT,
    HAND_PARTITION_INVARIANT,
    InvariantBroken,
    discrimination_report,
    format_report,
    hand_partition,
    require_declared_invariants_hold,
)
from keybo.analysis.kmstats import KmStats
from keybo.cli.__main__ import main
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.scoring.comfort import ComfortBigramScorer
from keybo.scoring.oxey import OxeyStyleScorer

#: keybo-lsb — the base every within-hand variant below permutes.
KEYBO_LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
#: Its hand partition as ``(left, right)``. Shared by keybo-lsb+lm, flagship-c3 and archive-1843,
#: which is why those four boards MUST tie on the partition invariants.
KEYBO_LSB_PARTITION = (frozenset("',-.aehijkopuyz"), frozenset("bcdfglmnqrstvwx"))
#: Rendered form, as it appears in messages and the JSON block.
KEYBO_LSB_PARTITION_LABEL = "',-.aehijkopuyz|bcdfglmnqrstvwx"
#: A board with a DIFFERENT hand partition, to prove the forced tie is about the partition.
GRAPHITE = "bldwz'foujnrtsgyhaeixqmcvkp,.-"
#: Same LEFT set as qwerty but a different RIGHT set (';','/' vs "'",'-'), so NOT a within-hand
#: permutation of it — the pair that proves a left-set-only partition key is wrong.
QWERTY_CLASSIC = "qwertyuiopasdfghjkl;zxcvbnm,./"
QWERTY_30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"


@pytest.fixture(scope="module")
def gauge_of():
    """``lay30 -> {gauge: value}`` over the full fifteen-gauge frame, as ``analyze`` builds it."""
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    corpus = production_corpus_dir()
    bigrams = load_frequencies(str(corpus / "bigrams.txt"))
    skipgrams = load_frequencies(str(corpus / "1-skip31.txt"))
    trigrams = load_frequencies(str(corpus / "trigrams.txt"))
    stats = KmStats(bigrams, skipgrams, trigrams)
    oxey = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    comfort = ComfortBigramScorer(bigrams, skipgram_freqs=skipgrams)
    mass = sum(v for k, v in bigrams.items() if len(k) == 2)

    def of(lay30: str) -> dict[str, float]:
        layout = Layout(lay30, ROW_STAGGERED_30)
        shares = oxey.pattern_shares(layout)
        gauges = dict(stats.stats(lay30))
        gauges["scissor"] = shares["scissor"]
        gauges["imbalance"] = shares["imbalance"]
        gauges["oxey-style"] = oxey.fitness(layout)
        gauges["comfort"] = comfort.fitness(layout) / mass
        return gauges

    return of


def _within_hand_variants(base: str = KEYBO_LSB, n: int = 12, seed: int = 0) -> list[str]:
    """Permute INSIDE each hand only, so every variant has an IDENTICAL hand partition.

    ⚠ Not a plain shuffle, and the difference is the finding: a full-charset shuffle moves characters
    between hands, so ``alt`` and ``imbalance`` separate and the invariance HIDES (only ``sfr``
    shows). Only within-hand permutation exposes it — and that is exactly the candidate set a local
    search around a fixed hand partition explores, i.e. what the optimizer itself does.
    """
    slots = ROW_STAGGERED_30.slots
    left = [i for i, slot in enumerate(slots) if ROW_STAGGERED_30.hand(slot[0]) < 0]
    right = [i for i, slot in enumerate(slots) if ROW_STAGGERED_30.hand(slot[0]) > 0]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chars = list(base)
        for side in (left, right):
            values = [chars[i] for i in side]
            rng.shuffle(values)
            for slot, value in zip(side, values, strict=True):
                chars[slot] = value
        out.append("".join(chars))
    return out


# ---------------------------------------------------------------------------------------
# 1. The FACT, measured here rather than quoted — and the guard on the guard.
# ---------------------------------------------------------------------------------------


def test_the_variants_really_do_hold_the_hand_partition_constant():
    """Guard the guard: if a variant moved a character across hands, every test below is vacuous."""
    partitions = {hand_partition(v) for v in _within_hand_variants()}
    assert len(partitions) == 1, "variants must share ONE hand partition"
    assert partitions.pop() == KEYBO_LSB_PARTITION, "and it must be keybo-lsb's"
    assert hand_partition(GRAPHITE) != KEYBO_LSB_PARTITION, "the contrast board must differ"


def test_the_partition_is_BOTH_hands_because_a_left_set_alone_is_not_a_partition():
    """REGRESSION: keying the partition on the LEFT set only made a correct measurement RAISE.

    ``qwerty`` and ``qwerty30m`` have identical left sets but different right sets, so they are NOT
    within-hand permutations of each other and ``alt`` legitimately differs by 0.0509. A
    left-set-only key grouped them as one partition, and the enforcement then refused the whole
    ``analyze`` run — eight shipped frozen-board tests caught it. The left set identifies a partition
    only within a FIXED charset, and ``analyze`` explicitly supports mixed-charset comparisons.
    """
    classic, thirty_m = hand_partition(QWERTY_CLASSIC), hand_partition(QWERTY_30M)
    assert classic[0] == thirty_m[0], "the LEFT sets really are identical — that is the trap"
    assert classic[1] != thirty_m[1], "but the RIGHT sets differ, so it is not one partition"
    assert classic != thirty_m, "so the partition key must distinguish them"


def test_the_declared_invariants_are_exactly_constant_under_within_hand_permutation(gauge_of):
    """The invariance itself, pinned. If a future change makes these vary, that is NEWS."""
    rows = [gauge_of(v) for v in _within_hand_variants()]
    for gauge in sorted(HAND_PARTITION_INVARIANT):
        values = [row[gauge] for row in rows]
        assert max(values) - min(values) == 0.0, f"{gauge} moved; re-measure the declared set"


def test_every_OTHER_gauge_does_discriminate_which_is_what_makes_the_test_above_meaningful(
    gauge_of,
):
    """The contrast is the point — otherwise the pin above could pass on a broken corpus."""
    rows = [gauge_of(v) for v in _within_hand_variants()]
    movers = [g for g in rows[0] if g not in HAND_PARTITION_INVARIANT]
    assert movers
    for gauge in movers:
        values = [row[gauge] for row in rows]
        assert max(values) - min(values) > 0.0, f"{gauge} is ALSO invariant — extend the set"


def test_a_FULL_shuffle_HIDES_the_invariance_so_the_probe_must_be_within_hand(gauge_of):
    """Pin the methodological trap, not just the fact.

    Under full-charset shuffles ``alt`` and ``imbalance`` separate, so a full-shuffle distinctness
    probe reports the frame as healthy. This fails if someone "simplifies" the fixture to a plain
    shuffle.
    """
    rng = random.Random(0)
    full = []
    for _ in range(12):
        chars = list(KEYBO_LSB)
        rng.shuffle(chars)
        full.append("".join(chars))
    rows = [gauge_of(v) for v in full]
    for gauge in ("alt", "imbalance"):
        values = [row[gauge] for row in rows]
        assert max(values) - min(values) > 0.0, (
            f"{gauge} SHOULD separate under a full shuffle — that is the trap this pins"
        )
    sfr = [row["sfr"] for row in rows]
    assert max(sfr) - min(sfr) == 0.0, "sfr is invariant under ANY permutation (see below)"


def test_sfr_is_CHARSET_invariant_which_is_a_STRONGER_claim_than_partition_invariant(gauge_of):
    """``sfr`` counts same-KEY repetition, so it does not depend on placement at all.

    Tracked as its own scope because the two claims license different readings of a tie: a partition
    invariant ties only layouts sharing a partition (so an equal value ACROSS partitions would be
    news), while a charset invariant ties every same-charset layout (so the same observation is
    expected). A first draft of this module declared only the partition set and therefore reported
    ``sfr``'s cross-partition tie as UNEXPECTED — telling a reader to go measure a hidden invariant
    that its own docstring documents.
    """
    assert CHARSET_INVARIANT <= HAND_PARTITION_INVARIANT, "the stronger claim must be a subset"
    # Same charset, DIFFERENT hand partition -> still exactly equal.
    assert gauge_of(KEYBO_LSB)["sfr"] == gauge_of(GRAPHITE)["sfr"]
    assert hand_partition(KEYBO_LSB) != hand_partition(GRAPHITE)
    # A different CHARSET is what moves it (classic qwerty carries ';' and '/').
    assert gauge_of(QWERTY_CLASSIC)["sfr"] != gauge_of(KEYBO_LSB)["sfr"]


# ---------------------------------------------------------------------------------------
# 2. The report: a forced tie is labelled forced, with its cause NAMED.
# ---------------------------------------------------------------------------------------


def test_a_partition_forced_tie_is_reported_as_FORCED_not_as_agreement(gauge_of):
    """Two within-hand variants tie on alt/imbalance; the report must say WHY."""
    first, second = _within_hand_variants(n=2)
    layouts = {"a": first, "b": second}
    report = discrimination_report(layouts, {k: gauge_of(v) for k, v in layouts.items()})

    assert report["compared"] is True
    for gauge in ("alt", "imbalance"):
        ties = report["forced_ties"][gauge]
        assert [t["layouts"] for t in ties] == [["a", "b"]]
        assert ties[0]["forced_by"] == "hand-partition"
        assert ties[0]["shared_hand_partition"] == KEYBO_LSB_PARTITION_LABEL
        assert gauge not in report["discriminating"]
    assert report["forced_ties"]["sfr"][0]["forced_by"] == "charset"


def test_a_tie_no_declared_invariance_explains_is_coincidental_NOT_forced(gauge_of):
    """A tie is only "forced" when something actually forces it.

    ``keybo-lsb`` and ``keybo-lsb+lm`` differ by one finger's placement and tie exactly on several
    ordinary gauges. Those are real ties worth printing, but claiming an invariance for them would
    be inventing one — and eight such lines bury the three that matter.
    """
    layouts = {"lsb": KEYBO_LSB, "lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq"}
    report = discrimination_report(layouts, {k: gauge_of(v) for k, v in layouts.items()})

    assert set(report["forced_ties"]) == set(HAND_PARTITION_INVARIANT), (
        "exactly the declared invariants may be called forced"
    )
    assert report["coincidental_ties"], "these two boards do tie on ordinary gauges"
    assert not set(report["coincidental_ties"]) & HAND_PARTITION_INVARIANT
    for gauge, ties in report["coincidental_ties"].items():
        assert all(t["forced_by"] is None for t in ties), gauge


def test_the_text_report_explains_forced_ties_individually_and_summarizes_the_rest(gauge_of):
    """Each forced tie is a specific claim a reader would otherwise get wrong; ties in bulk are not."""
    layouts = {"lsb": KEYBO_LSB, "lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq", "graphite": GRAPHITE}
    report = discrimination_report(layouts, {k: gauge_of(v) for k, v in layouts.items()})
    lines = format_report(report)

    assert lines[0].startswith("== gauge discrimination")
    forced_lines = [ln for ln in lines if "FORCED" in ln]
    assert len(forced_lines) == sum(len(t) for t in report["forced_ties"].values())
    assert any("do not read the tie" in ln for ln in forced_lines)
    # sfr's explanation must cite the CHARSET scope, alt's the partition scope.
    assert any(ln.startswith("sfr:") and "one charset" in ln for ln in forced_lines)
    assert any(ln.startswith("alt:") and "within-hand permutation" in ln for ln in forced_lines)
    assert sum(1 for ln in lines if "NOT structurally forced" in ln) <= 1, "one summary line"


def test_a_single_layout_reports_NOT_COMPARED_rather_than_all_discriminating(gauge_of):
    """ "No comparison was possible" must not read like "every gauge discriminated".

    Same reason ``verdicts.bucket_regression_report`` carries an explicit ``gated`` flag: an
    artifact that merely omits a verdict reads identically whether the check ran and passed or never
    ran at all.
    """
    report = discrimination_report({"only": KEYBO_LSB}, {"only": gauge_of(KEYBO_LSB)})
    assert report["compared"] is False
    assert report["discriminating"] == []
    assert report["forced_ties"] == {} and report["coincidental_ties"] == {}
    assert format_report(report) == [], "nothing to say, so say nothing"


def test_a_non_finite_cell_is_SKIPPED_per_gauge_not_allowed_to_suppress_the_others(gauge_of):
    """``all_distinct`` refuses non-finite operands (correctly) — but only that gauge is lost.

    A charset that cannot support one gauge must not silence the verdict on the other fourteen,
    and the loss must be RECORDED rather than looking like a pass.
    """
    first, second = _within_hand_variants(n=2)
    layouts = {"a": first, "b": second}
    gauges = {k: dict(gauge_of(v)) for k, v in layouts.items()}
    gauges["a"]["comfort"] = float("nan")

    report = discrimination_report(layouts, gauges)
    assert report["skipped"] == {"comfort": ["a"]}
    assert "comfort" not in report["discriminating"]
    assert "alt" in report["forced_ties"], "the other gauges still get a verdict"
    assert any("not compared (non-finite cell)" in ln for ln in format_report(report))


# ---------------------------------------------------------------------------------------
# 3. ENFORCEMENT: a declared invariant that stops holding RAISES.
# ---------------------------------------------------------------------------------------


def test_a_declared_invariant_that_VARIES_within_one_partition_is_REFUSED(gauge_of):
    """THE assertion. A broken invariance must fail loudly, not print a healthier-looking board.

    Simulated by perturbing the value rather than the gauge, because the gauge is currently correct:
    the test has to pin what happens WHEN it breaks, and a test that can only pass while the code is
    right is not a guard. The perturbation is the smallest that matters — one layout's ``alt`` moved
    inside a partition that must force it constant.
    """
    first, second = _within_hand_variants(n=2)
    layouts = {"a": first, "b": second}
    gauges = {k: dict(gauge_of(v)) for k, v in layouts.items()}
    gauges["b"]["alt"] += 1e-9

    with pytest.raises(InvariantBroken) as excinfo:
        require_declared_invariants_hold(layouts, gauges)
    message = str(excinfo.value)
    assert "alt varies by" in message
    assert "share the hand partition" in message
    assert "test_analyze_allgauge" in message, "name the frozen pin this invalidates"
    assert "do not read the new spread as a gauge that got better" in message


def test_the_enforcement_passes_and_RETURNS_the_report_on_the_real_gauges(gauge_of):
    """One pass gives both the enforcement and the explanation — no second traversal to drift."""
    layouts = {"lsb": KEYBO_LSB, "graphite": GRAPHITE}
    report = require_declared_invariants_hold(layouts, {k: gauge_of(v) for k, v in layouts.items()})
    assert report["compared"] is True
    assert report == discrimination_report(layouts, {k: gauge_of(v) for k, v in layouts.items()})


def test_layouts_in_DIFFERENT_partitions_may_differ_on_a_partition_invariant(gauge_of):
    """The guard must be scoped to WITHIN a partition, or it would forbid a legitimate difference.

    ``archive-1846`` has a different hand partition from ``keybo-lsb`` and a genuinely different
    ``alt`` (45.147 vs 45.156). A guard that compared across partitions would refuse that — turning
    a correct measurement into a crash.
    """
    layouts = {"lsb": KEYBO_LSB, "1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx"}
    gauges = {k: gauge_of(v) for k, v in layouts.items()}
    assert hand_partition(layouts["lsb"]) != hand_partition(layouts["1846"])
    assert gauges["lsb"]["alt"] != gauges["1846"]["alt"], "these genuinely differ"
    require_declared_invariants_hold(layouts, gauges)  # must NOT raise


def test_a_MIXED_CHARSET_comparison_is_not_refused(gauge_of):
    """REGRESSION for the same left-set bug, at the enforcement rather than the key.

    ``analyze qwerty qwerty30m`` is a comparison the suite makes in eight shipped frozen-board tests.
    With a left-set-only partition key those two counted as one partition, ``alt``'s legitimate
    0.0509 difference read as a broken invariant, and the guard took down the whole run — a guard
    that refuses correct data is worse than no guard.
    """
    layouts = {"qwerty": QWERTY_CLASSIC, "qwerty30m": QWERTY_30M}
    gauges = {k: gauge_of(v) for k, v in layouts.items()}
    assert gauges["qwerty"]["alt"] != gauges["qwerty30m"]["alt"]
    report = require_declared_invariants_hold(layouts, gauges)  # must NOT raise
    # `sfr` differs too (different charset), so no forced tie is claimed for it either.
    assert "sfr" not in report["forced_ties"]
    assert "sfr" in report["discriminating"]


# ---------------------------------------------------------------------------------------
# 4. WIRED: the guard runs on the shipped `analyze` path — both of them.
# ---------------------------------------------------------------------------------------


def test_analyze_TEXT_labels_the_campaign_boards_forced_tie(capsys):
    """End-to-end: the four partition-sharing registry boards must be called out, not just tied."""
    assert main(["analyze", "keybo-lsb", "keybo-lsb+lm", "flagship-c3", "--no-time"]) == 0
    text = capsys.readouterr().out
    assert "== gauge discrimination" in text
    assert "FORCED, not agreement" in text
    assert "',-.aehijkopuyz" in text, "the shared partition must be named as the cause"
    for gauge in ("alt", "imbalance"):
        assert any(ln.startswith(f"{gauge}:") and "FORCED" in ln for ln in text.splitlines()), gauge


def test_analyze_JSON_carries_the_discrimination_block_so_a_CONSUMER_cannot_miss_it(capsys):
    """The ``--json`` path is the one a downstream consumer reads, and it cannot re-derive this.

    A forced tie and a genuine agreement are the same two floats in ``rows``; the distinction ships
    alongside them or is lost. A guard wired only into the text renderer would be silently bypassed
    here — which is why the enforcement runs before the output branch, not inside it.
    """
    assert main(["analyze", "keybo-lsb", "flagship-c3", "--no-time", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    block = payload["discrimination"]
    assert block["compared"] is True
    assert sorted(block["declared_invariant"]) == sorted(HAND_PARTITION_INVARIANT)
    assert sorted(block["declared_charset_invariant"]) == sorted(CHARSET_INVARIANT)
    for gauge in ("alt", "imbalance", "sfr"):
        assert gauge in block["forced_ties"], gauge
    # And the numbers it explains really are equal in `rows` — the tie it describes is the tie.
    rows = payload["rows"]
    assert rows["keybo-lsb"]["gauges"]["alt"] == rows["flagship-c3"]["gauges"]["alt"]


def test_all_distinct_now_HAS_a_production_caller_which_it_did_not(gauge_of):
    """``all_distinct`` had ZERO production callers; the guard must actually route through it.

    Asserted by observing the call, not by reading the source: "the guard is wired" is exactly the
    kind of claim that stays true in a docstring after the call site is refactored away.
    """
    from keybo.analysis import discrimination as module

    seen: list[str] = []
    real = module.all_distinct

    def spy(values, what, **kwargs):
        seen.append(what)
        return real(values, what, **kwargs)

    module.all_distinct = spy
    try:
        layouts = {"lsb": KEYBO_LSB, "graphite": GRAPHITE}
        require_declared_invariants_hold(layouts, {k: gauge_of(v) for k, v in layouts.items()})
    finally:
        module.all_distinct = real

    assert seen, "the report must route its distinctness question through verdicts.all_distinct"
    assert any("alt" in what for what in seen)
