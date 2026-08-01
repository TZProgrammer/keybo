"""Which gauges can DISCRIMINATE the layouts being compared, and which ties are FORCED.

A gauge column that prints the same number for two layouts reads as *those layouts agree*. For
three of the campaign's fifteen gauges that reading is wrong: ``sfr``, ``alt`` and ``imbalance``
are EXACTLY constant under within-hand permutation (measured spread 0.000e+00), so any two boards
that place the same characters on the same HAND must tie on them no matter how differently they
arrange those characters. The tie is a property of the gauge, not a finding about the layouts.

This is not hypothetical on this campaign's own boards. Four of them — ``keybo-lsb``,
``keybo-lsb+lm``, ``flagship-c3``, ``archive-1843`` — share the left-hand charset
``"',-.aehijkopuyz"``, so all four print an identical ``sfr``/``alt``/``imbalance`` triple; that is
also why ``tests/cli/test_analyze_allgauge.py`` check-ins the same three literals for three
different frozen boards. Those literals are a CORRECT regression pin. What was missing is anything
that distinguished a pin from a discrimination test, or told a reader of the board which of the two
a repeated number is.

``keybo.verdicts.all_distinct`` was written for exactly this — its docstring names ``alt``,
``imbalance`` and ``sfr`` and says "run it before crediting a per-gauge win count" — and it had
ZERO production callers, so nothing ever ran it. This module is that caller, placed where the
numbers become a comparison.

Two things happen here, and the split is deliberate:

* :func:`discrimination_report` explains ties. A tie a declared invariance accounts for is labelled
  FORCED, with the invariance named — :data:`HAND_PARTITION_INVARIANT` for a shared hand partition,
  :data:`CHARSET_INVARIANT` for the stronger claim that holds across all of them. Any other tie is
  labelled coincidental, and is reported in one summary line rather than flagged: two near-identical
  layouts tie on most gauges by construction, so treating every tie as suspicious buries the forced
  ones. Tracking the two scopes separately is what keeps ``sfr``'s expected cross-partition tie from
  being reported as a mystery.
* :func:`require_declared_invariants_hold` REFUSES when a gauge this module declares
  partition-invariant is observed to vary within a single hand partition. That is not a tie
  problem, it is the frame's own claim being false — and every statement scoped to
  :data:`HAND_PARTITION_INVARIANT` (this docstring, the frozen literals, the ledger) would be
  unfounded while still reading as true. Raising follows the house rule that a partition which
  silently stops holding must not be printed (compare
  :meth:`keybo.analysis.bad_scissor.BadScissor` refusing a drifted label set).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.verdicts import all_distinct

#: Gauges EXACTLY constant under within-hand permutation, so a tie between two layouts sharing a
#: hand partition is structurally forced rather than a measurement.
#:
#: Measured, not assumed: over within-hand permutations of ``keybo-lsb`` (per-hand character sets
#: held fixed, verified against :meth:`keybo.geometry.Geometry.hand`) these three have spread
#: 0.000e+00 while the other twelve of the fifteen-gauge frame move by 1.9 to 87.5 units.
#:
#: ⚠ THE PERTURBATION IS LOAD-BEARING. Under FULL charset shuffles characters cross hands, so
#: ``alt`` and ``imbalance`` separate and only ``sfr`` looks invariant — a full-shuffle probe
#: reports the frame as healthy. Only within-hand permutation exposes it, and that is exactly the
#: candidate set a local search around a fixed hand partition explores.
HAND_PARTITION_INVARIANT = frozenset({"sfr", "alt", "imbalance"})

#: Gauges constant under ANY permutation of a fixed charset — a strictly stronger claim than
#: :data:`HAND_PARTITION_INVARIANT`, and a strict subset of it.
#:
#: ``sfr`` is same-KEY repetition (``a is b``): the mass of doubled letters in the corpus, which
#: depends on WHICH characters are on the board and not at all on where any of them sits. Measured:
#: invariant under full-charset shuffles too (where ``alt`` and ``imbalance`` do separate), and it
#: moves only when the charset itself changes — 2.6595771026964927 on every C30M board, 2.664409719
#: on classic-charset qwerty.
#:
#: Tracked separately because the two claims license different readings of a tie. A partition
#: invariant ties only layouts that share a hand partition, so an equal value across DIFFERENT
#: partitions would be news. A charset invariant ties every same-charset layout, so the same
#: observation is expected — and a first draft of this module, declaring only the partition set,
#: labelled ``sfr``'s cross-partition tie "UNEXPECTED" and told a reader to go measure a hidden
#: invariant that is documented one line above.
CHARSET_INVARIANT = frozenset({"sfr"})


class InvariantBroken(ValueError):
    """A gauge declared partition-invariant was observed to VARY within one hand partition.

    Distinct from a tie: a tie is the invariant holding (and being misreadable). This is the
    invariant not holding, which makes :data:`HAND_PARTITION_INVARIANT` — and everything scoped to
    it — a false statement that still reads as true.
    """


def hand_partition(
    lay30: str, geometry: Geometry = ROW_STAGGERED_30
) -> tuple[frozenset[str], frozenset[str]]:
    """``(left characters, right characters)`` — the layout's hand partition.

    ⚠ BOTH sides, not just the left. The left set alone identifies the partition only within a FIXED
    charset, where the right set is its complement — and ``analyze`` explicitly supports
    mixed-charset comparisons. ``qwerty`` and ``qwerty30m`` are the counterexample that shipped in
    the suite: identical left sets (``'abcdefgqrstvwxz'``) but different right sets (``';'``/``'/'``
    vs ``"'"``/``'-'``), so they are NOT within-hand permutations of each other and ``alt``
    legitimately differs by 0.0509. A left-set-only key called them one partition and made a correct
    measurement raise — this function's first draft did exactly that, and eight shipped frozen-board
    tests caught it.

    Hands come from :meth:`keybo.geometry.Geometry.hand` rather than a column-index convention
    re-derived here, so a geometry whose columns are laid out differently cannot make this disagree
    with the gauges it is explaining.
    """
    if len(lay30) != len(geometry.slots):
        raise ValueError(
            f"layout has {len(lay30)} characters but geometry has {len(geometry.slots)} slots"
        )
    sides: tuple[set[str], set[str]] = (set(), set())
    for char, slot in zip(lay30, geometry.slots, strict=True):
        hand = geometry.hand(slot[0])
        if hand:  # a thumb/space slot belongs to neither hand's assignable set
            sides[hand > 0].add(char)
    return frozenset(sides[0]), frozenset(sides[1])


def _partition_label(partition: tuple[frozenset[str], frozenset[str]]) -> str:
    """The partition as ``'left|right'`` for a message — both sides, since both identify it."""
    return f"{''.join(sorted(partition[0]))}|{''.join(sorted(partition[1]))}"


def _tie_groups(names: Sequence[str], values: Mapping[str, float]) -> list[list[str]]:
    """Layout names grouped by EXACTLY equal value, keeping only groups of two or more.

    Exact equality, not a tolerance: the invariance being explained is exact (spread 0.000e+00), so
    a tolerance would fold genuine near-misses in with forced ties and blur the one distinction
    this module exists to draw.
    """
    by_value: dict[float, list[str]] = {}
    for name in names:
        by_value.setdefault(values[name], []).append(name)
    return [group for group in by_value.values() if len(group) > 1]


def discrimination_report(
    layouts: Mapping[str, str],
    gauges: Mapping[str, Mapping[str, float]],
    *,
    geometry: Geometry = ROW_STAGGERED_30,
) -> dict:
    """Which of ``gauges`` separate ``layouts``, and for each tie, whether it is FORCED.

    ``layouts`` is ``name -> 30-char layout``; ``gauges`` is ``name -> {gauge: value}``. Returns a
    serializable dict; never raises on a tie — use :func:`require_declared_invariants_hold` to
    enforce. ``compared`` is explicit so a report that could not run reads differently from one
    that ran and found everything discriminating: with fewer than two layouts there is no
    comparison to make, which is not the same as "every gauge discriminates".

    Non-finite cells are excluded per gauge and listed in ``skipped``, rather than being fed to
    ``all_distinct`` (which refuses non-finite operands, correctly — but a charset that cannot
    support one gauge should not suppress the verdict on the other fourteen).
    """
    names = list(layouts)
    partitions = {name: hand_partition(layouts[name], geometry) for name in names}
    report: dict = {
        "compared": len(names) >= 2,
        "layouts": names,
        "partition_groups": [
            sorted(group)
            for group in _group_by_partition(names, partitions).values()
            if len(group) > 1
        ],
        "declared_invariant": sorted(HAND_PARTITION_INVARIANT),
        "declared_charset_invariant": sorted(CHARSET_INVARIANT),
        "discriminating": [],
        "forced_ties": {},
        "coincidental_ties": {},
        "skipped": {},
    }
    if not report["compared"]:
        return report

    for gauge in _gauge_names(gauges, names):
        finite = [n for n in names if math.isfinite(float(gauges[n][gauge]))]
        if len(finite) < len(names):
            report["skipped"][gauge] = sorted(set(names) - set(finite))
        if len(finite) < 2:
            continue
        values = {n: float(gauges[n][gauge]) for n in finite}
        # THE production call to the guard written for this. `all_distinct` answers "can this gauge
        # tell these layouts apart at all", which is the question a per-gauge comparison assumes
        # the answer to.
        if all_distinct([values[n] for n in finite], f"{gauge} over {len(finite)} layouts"):
            report["discriminating"].append(gauge)
            continue
        for group in _tie_groups(finite, values):
            shared = len({partitions[n] for n in group}) == 1
            same_charset = len({frozenset(layouts[n]) for n in group}) == 1
            report[_tie_kind(gauge, shared, same_charset)].setdefault(gauge, []).append(
                {
                    "layouts": sorted(group),
                    "value": values[group[0]],
                    "shared_hand_partition": _partition_label(partitions[group[0]])
                    if shared
                    else None,
                    "shared_charset": same_charset,
                    "forced_by": _forced_by(gauge, shared, same_charset),
                }
            )
    return report


def _forced_by(gauge: str, shared_partition: bool, shared_charset: bool) -> str | None:
    """Which declared invariance makes this tie STRUCTURAL, or ``None`` if none does.

    Scope order matters: the charset claim is stronger, so it is checked first and explains a tie
    between layouts that do NOT share a hand partition (which the partition claim cannot). But it is
    still SCOPED — ``sfr`` is constant across placements of one charset, not across charsets (2.6596
    on every C30M board, 2.6644 on classic-charset qwerty), so ``shared_charset`` gates it.
    """
    if shared_charset and gauge in CHARSET_INVARIANT:
        return "charset"
    if shared_partition and gauge in HAND_PARTITION_INVARIANT:
        return "hand-partition"
    return None


def _tie_kind(gauge: str, shared_partition: bool, shared_charset: bool) -> str:
    """``forced_ties`` when a declared invariance explains the tie, else ``coincidental_ties``.

    Deliberately NOT called "unexpected": two near-identical layouts genuinely tie on many gauges
    (``keybo-lsb`` and ``keybo-lsb+lm`` differ by ONE finger's placement and agree exactly on
    ``sfb``/``sfs``/``lsb``/``roll``/``redir``), and labelling each of those a possible hidden
    invariant buries the two real ones in eight lines of false alarm — which is what a first draft
    of this module did. A coincidental tie is still worth printing, because a tie is still a column
    that did not separate those layouts; it is just not evidence of a defect.
    """
    return (
        "forced_ties"
        if _forced_by(gauge, shared_partition, shared_charset)
        else "coincidental_ties"
    )


def _gauge_names(gauges: Mapping[str, Mapping[str, float]], names: Sequence[str]) -> list[str]:
    """Gauge names present for EVERY compared layout, in the first layout's order.

    Intersected rather than unioned: a gauge only some rows carry cannot be compared across the
    board, and defaulting the missing rows to 0.0 would manufacture a tie out of an absence.
    """
    common = set(gauges[names[0]])
    for name in names[1:]:
        common &= set(gauges[name])
    return [gauge for gauge in gauges[names[0]] if gauge in common]


_Partition = tuple[frozenset[str], frozenset[str]]


def _group_by_partition(
    names: Sequence[str], partitions: Mapping[str, _Partition]
) -> dict[_Partition, list[str]]:
    """Layout names grouped by IDENTICAL hand partition (both sides — see :func:`hand_partition`).

    Layouts in one group are within-hand permutations of each other, which is exactly the condition
    under which the declared invariants must hold. Layouts in different groups may legitimately
    differ on them, so the enforcement is scoped per group.
    """
    groups: dict[_Partition, list[str]] = {}
    for name in names:
        groups.setdefault(partitions[name], []).append(name)
    return groups


def require_declared_invariants_hold(
    layouts: Mapping[str, str],
    gauges: Mapping[str, Mapping[str, float]],
    *,
    geometry: Geometry = ROW_STAGGERED_30,
) -> dict:
    """Refuse if a :data:`HAND_PARTITION_INVARIANT` gauge VARIES within one hand partition.

    Returns the :func:`discrimination_report` on success, so a caller gets the explanation and the
    enforcement from one pass.

    This is the assertion at the boundary where these numbers become a printed comparison. If
    ``alt`` started separating two boards with the same hand partition, the three duplicate
    literals in ``tests/cli/test_analyze_allgauge.py`` would quietly stop being forced, every
    "these are hand-partition invariants" statement would be false, and the board would look
    healthier than before — a discriminating gauge is what a reader WANTS to see. So it raises:
    news that reads as good news is the one kind that never gets investigated.
    """
    report = discrimination_report(layouts, gauges, geometry=geometry)
    if not report["compared"]:
        return report
    partitions = {name: hand_partition(layouts[name], geometry) for name in layouts}
    for group in _group_by_partition(list(layouts), partitions).values():
        if len(group) < 2:
            continue
        for gauge in sorted(HAND_PARTITION_INVARIANT):
            if any(gauge not in gauges[name] for name in group):
                continue
            values = [float(gauges[name][gauge]) for name in group]
            if not all(math.isfinite(v) for v in values):
                continue
            if max(values) != min(values):
                raise InvariantBroken(
                    f"{gauge} varies by {max(values) - min(values):.6g} across "
                    f"{sorted(group)}, which share the hand partition "
                    f"{_partition_label(partitions[group[0]])!r} (left|right) — but {gauge} is "
                    f"declared EXACTLY "
                    f"constant under within-hand permutation "
                    f"(keybo.analysis.discrimination.HAND_PARTITION_INVARIANT). Either the gauge "
                    f"changed meaning or the declared set is now wrong. Both make every statement "
                    f"scoped to that set false while it still reads as true, including the "
                    f"duplicate frozen literals in tests/cli/test_analyze_allgauge.py, which are "
                    f"only a correct pin for as long as this holds. Re-measure the invariance and "
                    f"update the declared set deliberately; do not read the new spread as a gauge "
                    f"that got better at telling layouts apart."
                )
    return report


def format_report(report: dict) -> list[str]:
    """The report as text lines for ``keybo analyze`` (empty when there is nothing to say).

    FORCED ties get a line each, because each is a specific claim a reader would otherwise get
    wrong. Coincidental ties get ONE summary line for all of them: they are worth knowing (the
    column did not separate those layouts) but they are not defects, and a line per gauge buries
    the forced ones — near-identical variants tie on most gauges by construction.
    """
    if not report["compared"]:
        return []
    lines = ["== gauge discrimination: which columns above can tell these layouts apart =="]
    forced, coincidental = report["forced_ties"], report["coincidental_ties"]
    if not forced and not coincidental:
        lines.append(
            f"all {len(report['discriminating'])} gauges give every layout a distinct value — "
            "no tie to misread"
        )
        return lines
    for gauge, ties in sorted(forced.items()):
        for tie in ties:
            because = (
                f"every layout over one charset ties on it ({gauge} counts same-KEY repetition, so "
                f"it does not depend on placement at all)"
                if tie["forced_by"] == "charset"
                else (
                    f"they are within-hand permutations of each other (hand partition "
                    f"{tie['shared_hand_partition']!r}, left|right) and {gauge} is exactly constant "
                    f"under within-hand permutation"
                )
            )
            lines.append(
                f"{gauge}: {', '.join(tie['layouts'])} tie at {tie['value']:.6g} — FORCED, not "
                f"agreement: {because}. This column cannot separate them, so do not read the tie "
                f"as these layouts being equally good here."
            )
    if coincidental:
        pairs = sum(len(ties) for ties in coincidental.values())
        lines.append(
            f"also tied but NOT structurally forced ({pairs} group(s) over "
            f"{len(coincidental)} gauges: {', '.join(sorted(coincidental))}) — those columns "
            f"happen not to separate the layouts compared; no invariance is claimed. See the "
            f"`coincidental_ties` block in --json for which layouts."
        )
    if report["skipped"]:
        lines.append(
            "not compared (non-finite cell): "
            + "; ".join(f"{g} on {', '.join(ns)}" for g, ns in sorted(report["skipped"].items()))
        )
    return lines
