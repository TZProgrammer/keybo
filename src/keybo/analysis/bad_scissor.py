"""``bad-scissor`` — a same-hand row-travel bigram whose LOWER key is the less-dextrous
finger's.

The flag is defined by **which key is lower**. Do **not** document it as "the weaker finger
strains": that mechanism is **not identified on the Aalto sample**, even though the measured
effect is robust. On that sample the weak- and strong-descending groups share no bottom-row
key, so any property of the two key groups (rarity, neighbour interference, lateral travel) is
collinear with the label and the contrast never holds the physical key fixed.

**This limitation is EMPIRICAL, not structural** — a distinction worth keeping straight,
because the geometry does admit the missing comparisons. Verified here directly: over all
layout character pairs (not just corpus-observed ones) the dy==2 bottom-key sets **do**
intersect on qwerty — ``{',', '.', 'c', 'x'}`` — and a fixed bottom key takes both labels
(``qx``: top ``q`` = pinky, not flagged; ``ex``: top ``e`` = middle, flagged; same bottom key
``x``, same row span). So a corpus supplying those strong-descending observations **could**
identify the mechanism; the Aalto sample simply does not contain them. Do not restate the limit
as an in-principle one — it is a missing-observations problem, and more data can fix it.

What the effect *is* a statement about, per the spec: a few qwerty-era letter placements, not a
structural law. See ``badscissor-spec.md`` §0 and ``state/badscissor/report.md`` §4.7.

Implements the specification in ``state/badscissor/badscissor-spec.md`` (the ``badscissor``
agent, 2026-07-26) exactly. That document derived the predicate, the severity decision, the
denominator and the attribution rule from the Aalto keystroke frame and pinned expected
values before this code existed; ``tests/analysis/test_bad_scissor.py`` asserts them.
Nothing here is re-decided, and nothing is invented.

**Read this as a posture diagnostic, not a speed predictor.** The frequency-controlled,
overlap-restricted effect is **+0.41 ms [+0.23, +0.55]** — bigram *frequency* explains more
variance than any geometric axis. Two further caveats travel with every number below: the
mid-board layout ordering is **not robust** (only "qwerty is worst" and
"lsb-sib < archive-1843" survive every weighting), and **most of the flagged mass sits on a few
bottom keys (``c``/``x``)** — so the measured effect is a statement about a few qwerty-era letter
placements, not a structural law.

⚠ The figure **96.6 %** stood here unscoped and is WITHDRAWN AS A NUMBER (2026-07-28, BSAUDIT-1):
it reproduces in NONE of 24 shipped frames — qwerty measures 7.559 % on iWeb and 10.019 % on
blend-v1 — and the Aalto raw frame it may refer to is not in this repo, so it is unverifiable
against anything shipped. The DIRECTION holds (dvorak 0.000 %, qwerty highest); only the constant
is withdrawn. A constant published without its frame cannot be checked, and this one supported a
conclusion that is true, which is why it went unquestioned.

    bad-scissor fires  <=>  same hand AND different fingers AND different rows
                            AND the WEAKER finger of the pair is on the LOWER row

**Why this and not the incumbent gauges.** ``classify.is_scissor`` (narrow) and the served
objective's wide support both gate on ``abs(dy) == 2``. This predicate gates on *which
finger descends* instead, which makes it a **cross-cut** of both rather than a superset:

============================================  =====
ordered position pairs on ``ROW_STAGGERED_30``  900
narrow ``is_scissor``                            24
wide (same-hand, distinct-finger, dy == 2)       72
``bad-scissor``                                 108
  ...of which dy == 1                            72
  ...of which dy == 2                            36
  ...of which middle-pinky                       12
narrow \\ bad  (excluded: all weak-on-TOP)       12
wide   \\ bad  (excluded: all weak-on-TOP)       36
============================================  =====

The excluded pairs are the ones the spec's fit measured as *not* costly (the
index-key-on-bottom wide class measures -0.0179, i.e. faster than the same-row baseline, at
n=1.64M — "costly" is what was measured, interval time; "strained" would be the mechanistic
reading, which is not identified). So this gauge drops half of the incumbent's own support and
adds 72 single-row descents neither incumbent can see. Because the supports are **not nested**, comparing it against narrow or
wide is a meaningful check rather than trap #11's nested-guard mistake — but per the spec's
§6.4, correlation with them is still not independent corroboration of anything.

**Severity: FLAT** (1.0 per qualifying bigram). Per-pair graded weights are derivable but
rest on 2-5 bigram identities per pair and do not produce a robust board ordering, so the
spec ships flat and this module implements only flat. Notably ``bad-scissor-dist`` is
deliberately absent: the spec tested vertical distance as the severity axis and **refuted**
it (the distance coefficients come out negative), so there is no distance-weighted variant
to wire.

**Denominator (trap #9): layout-restricted, space-EXCLUDED bigram mass** — the
``kmstats``/``sfb``/``lsb`` convention, NOT ``oxey.pattern_shares``'. ``Layout.has_key(" ")``
is True, so the oxey convention silently counts space-touching bigrams in the denominator.
Space is in no bad-scissor pair (``hand(0) == 0``), so choosing wrong leaves the
**numerator bit-identical** and MOVES every share by a plausible ~1.497x constant. The
direction is DEFLATION, not inflation: space-touching bigrams are 33.85% of the mass, so
adding them to the denominator makes every share SMALLER (measured 1.496137-1.499860x across
all 15 registry layouts). That is exactly the failure the campaign's trap #9 describes, and
``test_the_space_including_denominator_moves_every_share_by_about_1_497x`` pins it.

(Corrected 2026-07-28 per BSAUDIT-1: this paragraph said "inflates" and cited a test named
``..._would_inflate_...``, which exists in no file — the real test was RENAMED to ``..._moves_...``
precisely because the direction was wrong, and the docstring kept both the stale name and the
refuted word. A citation is not a proof unless the cited thing exists.)

**Attribution: the whole of a pair's mass to the finger holding the LOWER key** — not to both,
not split. The predicate is an asymmetric statement about one finger: the flag fires *because*
of which finger holds the lower key, so that is the only finger the mass can be attributed to
without inventing a split the predicate does not contain. It also keeps the decomposition an
exact partition.

**Read the per-finger output as "where the flagged mass sits", NOT as "which finger is
strained".** The finger-level causal claim is not identified on the Aalto sample (see the
identification note at the top — empirical, not structural; ``badscissor-spec.md`` §3,
``report.md`` §4.7). An earlier revision of the spec justified this rule with a ``+0.5453`` vs
``-0.1083`` "placebo" contrast; that control is **RETRACTED** — on that sample it swaps the key
set along with the label, so it was never a placebo.

Structural consequence, expected and tested rather than a bug: **both index fingers are always
0.0**, because the index is the most dextrous finger and so never holds the lower key of a
qualifying pair.

⚠ **This is a MEASUREMENT/DIAGNOSIS gauge — not a search objective.** WSCISSOR-GEN-1
(ledger ``44d282b``) showed that optimizing a scissor-severity axis is optimizing the ruler:
champions won 1 of 19 gauges with a negative normalized floor, and an arm with no severity
axis behaved identically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from keybo.features import classify as C
from keybo.geometry import Geometry, Position
from keybo.layout import Layout

#: Dexterity rank, least -> most dextrous. Matches the campaign's ``_DEXTERITY_RANK`` and is
#: the reverse of ``tb_objective_ref._KIND_ORDER``, so "weaker finger" means the same thing
#: across the campaign's artifacts. Inverting this silently moves mass onto the index
#: fingers, which the test suite catches.
_DEX: Mapping[str, int] = {"pinky": 0, "ring": 1, "middle": 2, "index": 3}

#: The attribution rule, named in the output so a reader never has to infer it.
ATTRIBUTION_RULE = "all-to-descending-weaker-finger"

#: Report order for the per-finger columns (both index entries are structurally 0.0).
FINGER_ORDER: tuple[str, ...] = (
    "L-pinky",
    "L-ring",
    "L-middle",
    "L-index",
    "R-index",
    "R-middle",
    "R-ring",
    "R-pinky",
)

#: Rows the direction semantics are defined for (3 top, 2 home, 1 bottom).
_EXPECTED_ROWS = frozenset({1, 2, 3})


def _kind(geometry: Geometry, x: int) -> str:
    """``'right-index' -> 'index'``. Hand-independent."""
    return geometry.finger(x).value.split("-")[1]


def _weak_and_strong(
    geometry: Geometry, a: Position, b: Position
) -> tuple[str, int, int, int, str]:
    """``(weak kind, weak x, weak row, strong row, strong kind)`` for a two-finger pair."""
    ka, kb = _kind(geometry, a[0]), _kind(geometry, b[0])
    if _DEX[ka] <= _DEX[kb]:
        return ka, a[0], a[1], b[1], kb
    return kb, b[0], b[1], a[1], ka


def bad_scissor(geometry: Geometry, a: Position, b: Position) -> bool:
    """Whether the ordered bigram ``a -> b`` is a bad scissor.

    A property of the two POSITIONS, so it is symmetric: ``bad_scissor(g, a, b) ==
    bad_scissor(g, b, a)`` for all 900 pairs (asserted exhaustively). Direction of travel is
    priced by the corpus supplying both orderings, not by the flag.
    """
    if not C.same_hand(geometry, a, b):
        return False
    if C.same_finger(geometry, a, b):  # index cols 1&2, K31 pinky cols 5&6
        return False
    if a[1] == b[1]:  # no row travel
        return False
    _weak, _weak_x, weak_y, strong_y, _strong = _weak_and_strong(geometry, a, b)
    return weak_y < strong_y  # rows: 3 = top, 2 = home, 1 = bottom


def bad_scissor_finger(geometry: Geometry, a: Position, b: Position) -> str | None:
    """Hand-qualified name of the finger the pair's mass is attributed to, or ``None``.

    Always the descending (weaker) finger, e.g. ``"L-pinky"``. Never an index finger.
    """
    if not bad_scissor(geometry, a, b):
        return None
    weak, weak_x, _weak_y, _strong_y, _strong = _weak_and_strong(geometry, a, b)
    return f"{'L' if weak_x < 0 else 'R'}-{weak}"


def bad_scissor_cell(geometry: Geometry, a: Position, b: Position) -> str | None:
    """The pair's ``"<finger-pair> dy<n>"`` class, e.g. ``"index-pinky dy1"``, or ``None``.

    Finger pairs are named most-dextrous-first so the label matches the specification's
    tables (``index-pinky``, not ``pinky-index``).
    """
    if not bad_scissor(geometry, a, b):
        return None
    ka, kb = _kind(geometry, a[0]), _kind(geometry, b[0])
    first, second = sorted((ka, kb), key=lambda kind: -_DEX[kind])
    return f"{first}-{second} dy{abs(a[1] - b[1])}"


class BadScissor:
    """The ``bad-scissor`` share, and its two exact decompositions, over one bigram corpus."""

    def __init__(self, bigram_freqs: Mapping[str, int]) -> None:
        self._bg = {bg: freq for bg, freq in bigram_freqs.items() if len(bg) == 2}

    # -- gauges ---------------------------------------------------------------------------

    def share(self, layout: Layout, *, exclude_space: bool = True) -> float:
        """Bad-scissor mass as a percent of layout-restricted, space-excluded bigram mass.

        ``exclude_space=False`` selects the ``oxey`` denominator, which is **wrong for this
        gauge** and exists only so the trap-#9 regression test can measure the ~1.497x
        inflation it causes. Production callers leave it alone.
        """
        return self.share_of(layout, bad_scissor, exclude_space=exclude_space)

    def share_of(
        self,
        layout: Layout,
        predicate: Callable[[Geometry, Position, Position], bool],
        *,
        exclude_space: bool = True,
    ) -> float:
        """This gauge's scoring loop and denominator, run on an arbitrary pair predicate.

        Exposed so the ``sfb`` positive control can drive OUR denominator over a predicate
        whose value is independently known from :mod:`keybo.analysis.kmstats`. Because
        ``sfb``'s support is disjoint from bad-scissor's, any disagreement isolates the
        denominator rather than the predicate — which is the whole point of trap #9.
        """
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        numerator = 0.0
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            if predicate(geometry, a, b):
                numerator += freq
        return 100.0 * numerator / denominator if denominator else 0.0

    def by_finger(self, layout: Layout, *, exclude_space: bool = True) -> dict[str, float]:
        """Share attributed to each finger — an exact partition of :meth:`share`.

        Both index entries are always 0.0 (see the module docstring).
        """
        return self._partition(layout, bad_scissor_finger, FINGER_ORDER, exclude_space)

    def by_cell(self, layout: Layout, *, exclude_space: bool = True) -> dict[str, float]:
        """Share per ``"<finger-pair> dy<n>"`` class — a second exact partition.

        The ``dy2`` subtotal is the number that motivates the predicate, and ``dy == 2`` is
        the *only* thing the incumbent gauges see. It is USUALLY a small share of the priced
        mass but NOT always under a tenth: on blend-v1 (the CLI default) it exceeds 10% for
        4 of 15 registry layouts, peaking at 12.908% on ``qwerty30m``; on iWeb only 1 of 15
        does. Corrected 2026-07-28 per BSAUDIT-1, which had said "under a tenth" without
        naming a corpus — and the complement of this ledger's own registered 87.1-99.4% dy1
        range is 12.9%, so the old claim was contradicted by a published number.
        """
        return self._partition(layout, bad_scissor_cell, (), exclude_space)

    # -- internals ------------------------------------------------------------------------

    @staticmethod
    def _counts(layout: Layout, bigram: str, exclude_space: bool) -> bool:
        """Whether a bigram is inside the denominator (and so eligible for the numerator)."""
        if exclude_space and " " in bigram:
            return False
        return all(layout.has_key(character) for character in bigram)

    def _partition(
        self,
        layout: Layout,
        classifier: Callable[[Geometry, Position, Position], str | None],
        preset_keys: tuple[str, ...],
        exclude_space: bool,
    ) -> dict[str, float]:
        self._check_geometry(layout.geometry)
        geometry = layout.geometry
        charged = dict.fromkeys(preset_keys, 0.0)
        denominator = 0.0
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram, exclude_space):
                continue
            denominator += freq
            a, b = layout.pos(bigram[0]), layout.pos(bigram[1])
            key = classifier(geometry, a, b)
            if key is not None:
                if preset_keys and key not in charged:
                    # REFUSE an unclassified key rather than appending it (BSAUDIT-1 D4). The old
                    # `charged.get(key, 0.0)` silently grew the dict, and a caller that PRINTS a
                    # fixed column list then shows a partition that no longer sums to `share`:
                    # a drifted `R-pinky` label printed 0.0000 while its real 0.4658 sat
                    # unprinted, so 0.46584 pp vanished from a 4.11684 total. Every
                    # exact-partition test still passed, because they sum `.values()` and never
                    # the printed columns — so this cannot be caught downstream, only here.
                    raise ValueError(
                        f"{classifier.__name__} returned {key!r}, which is not one of this "
                        f"partition's declared keys {sorted(charged)}. Either the classifier's "
                        f"label set drifted from the caller's column list, or a new class was "
                        f"added without extending the presets — both silently break the "
                        f"partition when it is printed."
                    )
                charged[key] = charged.get(key, 0.0) + freq
        if not denominator:
            return dict.fromkeys(preset_keys, 0.0)
        return {key: 100.0 * value / denominator for key, value in charged.items()}

    @staticmethod
    def _check_geometry(geometry: Geometry) -> None:
        """Refuse a board this gauge's row semantics are not defined for.

        "The lower key belongs to the less-dextrous finger" is well defined on any board, but
        the spec's expected values, its dy census and its severity evidence are all derived on
        the three-row block. A four-row board would score silently against a support the
        specification never examined, so refuse instead — the same stance
        :mod:`keybo.scoring.scissor_severity` takes.
        """
        rows = {y for _x, y in geometry.slots}
        if not rows <= _EXPECTED_ROWS:
            raise ValueError(
                "bad-scissor is defined for the three-row board "
                f"(rows {sorted(_EXPECTED_ROWS)}); got rows {sorted(rows)}."
            )
