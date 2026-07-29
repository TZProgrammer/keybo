"""``finger-travel`` and ``off-home`` — where each finger's MOTION and its OFF-HOME use sit.

Implements ``docs/finger-travel-preregistration.md`` exactly (the FT round, 2026-07-28). Both
metrics were defined, and seven predictions registered, in that document **before any layout was
measured**; nothing here is re-decided.

Two DIFFERENT quantities, shipped as separate columns and never summed together:

* :class:`FingerTravel` — "a finger that moves around more will have a higher percentage".
  Per-finger **path length**, as an exact partition of 100%.
* :class:`OffHomeUsage` — "pinky being used a lot is mostly fine, as long as it stays on the
  home row". Per-finger usage split by row.

⚠ **BOTH ARE GEOMETRIC DESCRIPTORS — NOT times and NOT comfort claims.** A distance in key units
is not a millisecond and not a strain. The campaign has a registered failure of exactly that
shape: :mod:`keybo.analysis.bad_scissor`'s **+0.41 ms [+0.23, +0.55]** effect had bigram
*frequency* explaining more variance than any geometric axis, and its per-finger split had to be
relabelled "where the mass sits" rather than "which finger is strained". Read these the same way.

Why the headline travel definition is a *path* and not the existing displacement
-------------------------------------------------------------------------------
:class:`keybo.scoring.DislocationScorer.per_finger_dislocation` already computes a per-finger
diagnostic, but it answers a different question: it is ``sum freq(letter) * dist(letter, home)``,
i.e. **how far off home a finger's keys SIT**, weighted by use. A finger with three off-home keys
it shuttles between and a finger with one off-home key it hits three times score the same. The
user asked how much a finger **MOVES**, so:

    travel(f) = sum over bigrams xy of w(xy) * ( same_finger(x,y) ? dist(pos x, pos y)
                                                                 : [finger(y)==f] dist(home f, pos y) )

The same-finger branch is the finger's **observed** motion — the corpus says where it was. The
different-finger branch is **modelled**: absent evidence of where that finger last was, it is
taken to be at home. That branch is an assumption and is named as one; :data:`TRAVEL_MODEL` states
it in every emitted record.

**Lag-1 is the honest maximum, not a shortcut.** ``data/corpus/**`` ships bigram, trigram and
1-skip *count tables* and no raw text, so a finger's true unbounded path is not computable from
the shipped corpus at all. Lag-1 (bigrams) is the headline; :meth:`FingerTravel.lag2_shares` adds
the trigram-resolvable ``a?a`` returns as a registered sensitivity check.

Three properties the prereg fixes, each because getting it wrong has already cost this repo a
wrong number
------------------------------------------------------------------------------------------------
**The shares are an exact partition by CONSTRUCTION, not by arithmetic luck.** Every charged unit
of mass goes to exactly one finger — the shared finger, or the finger of the bigram's second
character — so the eight shares sum to 100 identically. The denominator is the metric's own total,
which is what makes the user's "sum to 100%" requirement hold.

**The absolute total is reported beside the shares, always.** Normalizing destroys the level: two
layouts can have identical shares and very different total travel. Reporting shares alone is the
same coverage artifact this ledger registered for ``saved_vs_ref_pct``, so :meth:`FingerTravel.report`
emits ``total`` in the same record as ``shares`` and callers cannot get one without the other.

**Not slowness-scaled.** The headline is pure distance. Mixing distance with a per-finger cost
weight yields neither a travel measure nor a time measure — and this campaign has a registered
failure of that shape (``oxey-style`` at R2=0.9937 on {sfb,lsb,scissor,imbalance,redir,alt}: a
re-weighted restatement of other legs). A slowness-weighted variant exists as a *separate*
column, :meth:`FingerTravel.slowness_weighted_shares`, flagged as a preference.

Trap #9 (the denominator)
-------------------------
``bad_scissor``'s ~1.497x space-denominator trap cannot bite a travel *share* the same way — the
denominator is the numerator's own sum, so both move together. It can still move the shares,
because space-touching bigrams contribute travel asymmetrically. Registered decision: **space is
EXCLUDED and the thumb is not a ninth cell.** The thumb has no home entry and space is a fixed key
at ``(0, 0)``, so a thumb cell would be identically 0.0 on every layout — no information, and it
would dilute the eight cells that carry some.
"""

from __future__ import annotations

from collections.abc import Mapping

from keybo.geometry import Geometry, Position
from keybo.layout import Layout

#: The eight typing fingers, in board order. Same labels as
#: :data:`keybo.analysis.bad_scissor.FINGER_ORDER` so the two modules' columns line up.
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

#: The home row. All eight home positions sit on it (:data:`HOME_POSITION`).
HOME_ROW = 2

#: Home position per finger, ``(column, row)``. The same table as
#: ``keybo.scoring.utilization._FINGER_HOME``, re-declared here keyed by the hand-qualified
#: label this module reports, so a reader never has to map between two key conventions.
HOME_POSITION: Mapping[str, Position] = {
    "L-pinky": (-5, HOME_ROW),
    "L-ring": (-4, HOME_ROW),
    "L-middle": (-3, HOME_ROW),
    "L-index": (-2, HOME_ROW),
    "R-index": (2, HOME_ROW),
    "R-middle": (3, HOME_ROW),
    "R-ring": (4, HOME_ROW),
    "R-pinky": (5, HOME_ROW),
}

#: The travel model, named in every emitted record so the modelled branch is never mistaken
#: for an observation.
TRAVEL_MODEL = (
    "lag1-resolved-path: same-finger bigrams charge the OBSERVED dist(k1,k2); different-finger "
    "bigrams charge the MODELLED dist(home, k2) because the corpus cannot say where that finger "
    "last was"
)

#: Slowness multipliers for the SEPARATE preference column. Imported rather than restated so the
#: two modules cannot drift apart.
from keybo.scoring.utilization import DEFAULT_SLOWNESS  # noqa: E402  (documented re-export)


def finger_label(geometry: Geometry, x: int) -> str:
    """``-5 -> 'L-pinky'``. Asks the geometry, so K31's column 6 resolves to the pinky."""
    if x == 0:
        raise ValueError("the thumb has no travel cell (see the module docstring)")
    kind = geometry.finger(x).value.split("-")[1]
    return f"{'L' if x < 0 else 'R'}-{kind}"


class _PartitionBase:
    """Shared denominator convention and the unknown-label guard both metrics need."""

    def __init__(self, bigram_freqs: Mapping[str, int]) -> None:
        self._bg = {bg: freq for bg, freq in bigram_freqs.items() if len(bg) == 2}

    @staticmethod
    def _counts(layout: Layout, ngram: str) -> bool:
        """Whether an n-gram is in the denominator: on the layout, and space-free.

        Space-EXCLUDED is the ``kmstats``/``sfb``/``lsb``/``bad_scissor`` convention.
        ``Layout.has_key(" ")`` is True, so relying on ``has_key`` alone would silently admit
        space-touching bigrams — the trap #9 shape.
        """
        return " " not in ngram and all(layout.has_key(character) for character in ngram)

    @staticmethod
    def _charge(charged: dict[str, float], key: str, amount: float, what: str) -> None:
        """Add ``amount`` to a DECLARED cell, or raise.

        REFUSES an undeclared label rather than appending it — the ``bad_scissor._partition``
        D4 failure, fixed 2026-07-28: a drifted ``R-pinky`` label was silently appended, a
        caller printing a fixed column list showed 0.0000 for it, and its real 0.4658 sat
        unprinted so 0.46584 pp vanished from a 4.11684 total. **Every exact-partition test
        still passed**, because they sum ``.values()`` and never the printed columns. That
        makes this uncatchable downstream — it has to be refused here.
        """
        if key not in charged:
            raise ValueError(
                f"{what} produced label {key!r}, which is not one of the declared cells "
                f"{sorted(charged)}. Either the labeller drifted from the caller's column list "
                f"or a cell was added without extending the presets — both silently break the "
                f"partition at PRINT time, where no sum-the-values test can see it."
            )
        charged[key] += amount

    @staticmethod
    def _as_shares(charged: Mapping[str, float], total: float) -> dict[str, float]:
        """Normalize to percentages of ``total``, or all-zero when there is no mass."""
        if not total:
            return dict.fromkeys(charged, 0.0)
        return {key: 100.0 * value / total for key, value in charged.items()}


class FingerTravel(_PartitionBase):
    """Per-finger path length over one bigram corpus — an exact partition of 100%.

    ``shares()`` answers the user's question ("a finger that moves around more has a higher
    percentage"); ``total()`` is the LEVEL the shares throw away. Use :meth:`report`, which
    cannot give you one without the other.
    """

    # -- the headline ----------------------------------------------------------------------

    def per_finger(self, layout: Layout, *, same_finger_only: bool = False) -> dict[str, float]:
        """Absolute travel per finger, in key-units x corpus-frequency.

        ``same_finger_only`` drops the modelled from-home branch, leaving only OBSERVED motion.
        Exposed because it isolates the part of the metric that is not an assumption — and
        because the degeneracy test in the prereg (§4.7) needs its complement.
        """
        geometry = layout.geometry
        charged = dict.fromkeys(FINGER_ORDER, 0.0)
        for bigram, freq in self._bg.items():
            if not self._counts(layout, bigram):
                continue
            first, second = layout.pos(bigram[0]), layout.pos(bigram[1])
            if geometry.same_finger(first[0], second[0]):
                # OBSERVED: the corpus says this finger was on `first` and went to `second`.
                self._charge(
                    charged,
                    finger_label(geometry, second[0]),
                    freq * geometry.distance(first, second),
                    "same-finger travel",
                )
            elif not same_finger_only:
                # MODELLED: no evidence where this finger was, so take it at home.
                label = finger_label(geometry, second[0])
                self._charge(
                    charged,
                    label,
                    freq * geometry.distance(HOME_POSITION[label], second),
                    "from-home travel",
                )
        return charged

    def total(self, layout: Layout) -> float:
        """The absolute level the shares normalize away. Always report it with them."""
        return sum(self.per_finger(layout).values())

    def shares(self, layout: Layout) -> dict[str, float]:
        """Per-finger travel as a percent of this layout's own total travel.

        Sums to 100.0 by construction: every charged unit goes to exactly one finger, and the
        denominator is the sum of the same eight cells.
        """
        charged = self.per_finger(layout)
        return self._as_shares(charged, sum(charged.values()))

    def report(self, layout: Layout) -> dict:
        """Shares AND the absolute total AND the dispersion statistics, in one record.

        Deliberately the only convenient accessor: a shares-only table is misleading, because
        two layouts can share every percentage and differ in level — the ``saved_vs_ref_pct``
        coverage artifact this ledger already registered.
        """
        charged = self.per_finger(layout)
        total = sum(charged.values())
        shares = self._as_shares(charged, total)
        return {
            "shares": shares,
            "total": total,
            "model": TRAVEL_MODEL,
            "denominator": "own total travel of the 8 fingers (space-EXCLUDED bigram mass)",
            "dispersion": dispersion(shares),
            "observed_fraction_pct": (
                100.0 * sum(self.per_finger(layout, same_finger_only=True).values()) / total
                if total
                else 0.0
            ),
        }

    # -- registered variants (each a SEPARATE column, never the headline) -------------------

    def static_per_finger(self, layout: Layout) -> dict[str, float]:
        """Definition (a): ``sum freq(letter) * dist(letter, home)`` — static DISPLACEMENT.

        The existing ``per_finger_dislocation`` quantity minus its slowness scaling. Shipped as
        the prereg's sensitivity check: it measures where a finger's keys SIT, not how far it
        MOVES, and it is what the headline degenerates to as same-finger mass vanishes.
        """
        return self._static(layout, factor=1.0)

    def return_home_per_finger(self, layout: Layout) -> dict[str, float]:
        """Definition (c): strict return-to-home, ``2 * dist(key, home)`` per press.

        Exists to make the prereg's §1.5 argument checkable in code rather than asserted: a
        positive scalar multiple cannot change a share, so (c) is (a) in different units and the
        only sensitivity check that matters is (a) vs the headline.
        """
        return self._static(layout, factor=2.0)

    def _static(self, layout: Layout, factor: float) -> dict[str, float]:
        """Letter-mass-weighted distance from home, scaled — the (a)/(c) family."""
        geometry = layout.geometry
        charged = dict.fromkeys(FINGER_ORDER, 0.0)
        for character, mass in letter_mass(self._bg, layout).items():
            position = layout.pos(character)
            label = finger_label(geometry, position[0])
            self._charge(
                charged,
                label,
                factor * mass * geometry.distance(HOME_POSITION[label], position),
                "static displacement",
            )
        return charged

    def slowness_weighted_shares(
        self, layout: Layout, slowness: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """The headline travel, times each finger's slowness — a PREFERENCE, not a measurement.

        Kept strictly separate from :meth:`shares` per the prereg §1.4: a distance times a
        per-finger cost weight is neither a travel measure nor a time measure. Reported as its
        own labelled column or not at all.
        """
        weights = dict(slowness or DEFAULT_SLOWNESS)
        charged = {
            label: value * weights[label.split("-")[1]]
            for label, value in self.per_finger(layout).items()
        }
        return self._as_shares(charged, sum(charged.values()))

    def lag2_shares(self, layout: Layout, trigram_freqs: Mapping[str, int]) -> dict[str, float]:
        """Sensitivity check: resolve one more lag, catching ``a?a`` same-finger returns.

        Bigrams cannot see that a finger left home, typed elsewhere, and came back two presses
        later. Trigrams can, for lag 2. Registered as a variant rather than the headline so the
        headline needs only the bigram table every other bigram gauge here uses.

        Charges, per trigram ``xyz``: the same lag-1 term for ``x->y``, then for ``y->z`` the
        OBSERVED ``dist(pos y, pos z)`` if same finger, else — and this is the added
        resolution — ``dist(pos x, pos z)`` when ``x`` and ``z`` share a finger, because the
        corpus now says where that finger actually was.
        """
        geometry = layout.geometry
        charged = dict.fromkeys(FINGER_ORDER, 0.0)
        for trigram, freq in trigram_freqs.items():
            if len(trigram) != 3 or not self._counts(layout, trigram):
                continue
            first, second, third = (layout.pos(character) for character in trigram)
            for previous, landing, prior in ((first, second, None), (second, third, first)):
                label = finger_label(geometry, landing[0])
                if geometry.same_finger(previous[0], landing[0]):
                    origin = previous  # observed at lag 1
                elif prior is not None and geometry.same_finger(prior[0], landing[0]):
                    origin = prior  # observed at lag 2 — the resolution bigrams lack
                else:
                    origin = HOME_POSITION[label]  # still modelled
                self._charge(
                    charged, label, freq * geometry.distance(origin, landing), "lag2 travel"
                )
        return self._as_shares(charged, sum(charged.values()))


class OffHomeUsage(_PartitionBase):
    """Per-finger usage split by row: ``on_home`` vs ``off_home`` (the user's pinky claim).

    Tests nothing on its own. The user's claim — "pinky being used a lot is mostly fine, as long
    as it stays on the home row" — has a measurement half (the interesting quantity is off-home
    usage, not total usage) and a COST half (total does not hurt, off-home does). **This class
    implements only the measurement half.** The cost half is an empirical claim about typing
    time; shipping this metric does not ship it.
    """

    def usage(self, layout: Layout) -> dict[str, float]:
        """Percent of layout-restricted, space-excluded LETTER mass on each finger's keys.

        An exact partition of 100% over the eight fingers.
        """
        charged, total = self._by_row(layout, rows=None)
        return self._as_shares(charged, total)

    def off_home(self, layout: Layout) -> dict[str, float]:
        """Percent of the SAME total letter mass sitting on each finger's non-home-row keys.

        An exact partition of the layout's total off-home mass — NOT of 100%. Same denominator
        as :meth:`usage`, so ``off_home(f) + on_home(f) == usage(f)`` exactly.
        """
        charged, total = self._by_row(layout, rows="off")
        return self._as_shares(charged, total)

    def on_home(self, layout: Layout) -> dict[str, float]:
        """Percent of total letter mass on each finger's HOME-ROW keys."""
        charged, total = self._by_row(layout, rows="on")
        return self._as_shares(charged, total)

    def off_fraction(self, layout: Layout) -> dict[str, float]:
        """``100 * off_home(f) / usage(f)`` — what share of a finger's OWN use is off-home.

        ⚠ **NOT a partition. Never sum this.** Each cell has a different denominator (that
        finger's own usage), so the eight values sum to something in the low hundreds and mean
        nothing. ``tests/analysis/test_finger_travel.py`` pins that it is not a partition,
        specifically so a later reader does not "fix" it into one.

        A finger with no usage at all reports 0.0 rather than dividing by zero: it has no
        off-home use, which is the honest reading of an unused finger.
        """
        usage = self.usage(layout)
        off = self.off_home(layout)
        return {
            label: (100.0 * off[label] / usage[label] if usage[label] else 0.0)
            for label in FINGER_ORDER
        }

    def report(self, layout: Layout) -> dict:
        """Every column at once, plus the pinky roll-up the user's claim is about."""
        usage, off, on = self.usage(layout), self.off_home(layout), self.on_home(layout)
        fraction = self.off_fraction(layout)
        pinkies = ("L-pinky", "R-pinky")
        pinky_usage = sum(usage[label] for label in pinkies)
        pinky_off = sum(off[label] for label in pinkies)
        return {
            "usage": usage,
            "on_home": on,
            "off_home": off,
            "off_fraction": fraction,
            "off_fraction_note": "per-finger ratio, NOT a partition — do not sum",
            "home_row": HOME_ROW,
            "denominator": (
                "layout-restricted, space-EXCLUDED letter mass (each character of each bigram)"
            ),
            "pinky": {
                "usage": pinky_usage,
                "on_home": sum(on[label] for label in pinkies),
                "off_home": pinky_off,
                "off_fraction": 100.0 * pinky_off / pinky_usage if pinky_usage else 0.0,
                "off_home_keys": self.off_home_keys(layout, pinkies),
            },
            "total_off_home": sum(off.values()),
        }

    def off_home_keys(self, layout: Layout, labels: tuple[str, ...] = FINGER_ORDER) -> list[str]:
        """The actual characters sitting off the home row on the given fingers, with their row.

        Reported because it is what makes a number auditable by eye: "keybo-lsb is worst" is a
        claim, ``p(r3) l(r3) k(r1) q(r1)`` is the reason.
        """
        geometry = layout.geometry
        wanted = set(labels)
        out = []
        for character in layout.chars:
            position = layout.pos(character)
            if position[1] != HOME_ROW and finger_label(geometry, position[0]) in wanted:
                out.append(f"{character}(r{position[1]})")
        return out

    def _by_row(self, layout: Layout, rows: str | None) -> tuple[dict[str, float], float]:
        """Charge letter mass to fingers, optionally filtered by home/off-home row.

        Returns ``(charged, total)`` where ``total`` is ALWAYS the unfiltered letter mass — that
        shared denominator is what makes ``off_home + on_home == usage`` hold exactly instead of
        approximately.
        """
        geometry = layout.geometry
        charged = dict.fromkeys(FINGER_ORDER, 0.0)
        total = 0.0
        for character, mass in letter_mass(self._bg, layout).items():
            total += mass
            position = layout.pos(character)
            on_home_row = position[1] == HOME_ROW
            if rows == "off" and on_home_row:
                continue
            if rows == "on" and not on_home_row:
                continue
            self._charge(
                charged, finger_label(geometry, position[0]), mass, f"row-{rows or 'all'} usage"
            )
        return charged, total


def letter_mass(bigram_freqs: Mapping[str, int], layout: Layout) -> dict[str, float]:
    """Letter frequency from a bigram table: each character of each bigram gets its frequency.

    The same construction as ``DislocationScorer._letter_freqs``, but **space-excluded and
    layout-restricted** to match this module's denominator: a bigram counts only if BOTH its
    characters are on the layout, so a layout is not flattered by a charset that misses corpus
    mass. (``DislocationScorer`` charges each character independently of its partner, which is a
    different — and for its purpose fine — convention. Stated, not hidden.)
    """
    mass: dict[str, float] = {}
    for bigram, freq in bigram_freqs.items():
        if len(bigram) != 2 or " " in bigram:
            continue
        if not all(layout.has_key(character) for character in bigram):
            continue
        for character in bigram:
            mass[character] = mass.get(character, 0.0) + float(freq)
    return mass


def dispersion(shares: Mapping[str, float]) -> dict[str, float]:
    """Concentration statistics for a per-finger share vector.

    A partition can sum to 100 in a balanced way or a lopsided one, and "which finger pays" is
    the whole reason for a per-finger metric. ``gini`` is the standard concentration index (0 =
    perfectly even across the eight cells, higher = more concentrated); ``lr_ratio`` is left-hand
    total over right-hand total, so 1.0 is balanced.
    """
    values = [shares[label] for label in FINGER_ORDER]
    left = sum(value for label, value in shares.items() if label.startswith("L-"))
    right = sum(value for label, value in shares.items() if label.startswith("R-"))
    n = len(values)
    mean = sum(values) / n
    absolute_differences = sum(abs(a - b) for a in values for b in values)
    return {
        "max_share": max(values),
        "pinky_share": shares["L-pinky"] + shares["R-pinky"],
        "lr_ratio": left / right if right else float("inf"),
        "gini": absolute_differences / (2 * n * n * mean) if mean else 0.0,
    }
