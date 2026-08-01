"""Vectorized `scissor` AND `sfb` gauges over a char-index bigram frequency matrix, so a
435-swap sweep is milliseconds instead of minutes.

`scissor` (`classify.is_scissor`) is a pure PREDICATE ON THE TWO POSITIONS -- same hand,
adjacent fingers, |dy| == 2 -- so it vectorizes exactly like `sfb`: a 30x30 slot-pair
indicator matrix contracted against the char-index bigram matrix. That is the whole reason
PRICEBAND-1's method transfers unchanged: the gauge is just another 0/1 slot-pair mask.

⚠ TWO DENOMINATOR CONVENTIONS EXIST IN THIS REPO AND THEY DIFFER BY 1.4896x. Measured here,
not assumed:

  * `oxey.pattern_shares` divides by `bg_total` = bigram mass whose both chars are ON THE
    BOARD -- and `Layout.has_key(' ')` is **True**, so SPACE-containing bigrams ARE in that
    denominator (913,956,722 vs 613,558,937 letters-only => ratio **1.4895989**).
  * `kmstats` (and PRICEBAND-1's `fastsfb`) divides by the LETTERS-ONLY bigram mass.

Space is in no scissor and no sfb (different hand/finger), so only the DENOMINATOR moves:
  gauge_kmstats = 1.4895989 * gauge_pattern.

`scissor` is reported by the shipped 15-gauge frame (`cli/analyze.py` -> `pattern_shares`),
and every `scissor` number in the ledger (SCISSORPRICE-1's in-domain range [0.0682, 0.5173],
the field values) is in the **pattern_shares** convention. `sfb` in PRICEBAND-1 is in the
**kmstats** convention. So this module serves BOTH, explicitly named, and a price quoted
"per pp" must say which -- a per-pp price scales INVERSELY with the gauge's unit, so the two
conventions differ by 1.4896x in the price too.
"""
import numpy as np

from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30

CHARS = "abcdefghijklmnopqrstuvwxyz',.-"
NC = len(CHARS)
NS = 30


class FastGauges:
    """`scissor` and `sfb` shares over the 30-slot board, in BOTH denominator conventions.

    `*_only` methods are the **pattern_shares** convention (space-inclusive denominator, the
    shipped 15-gauge frame and every ledger scissor number). `*_km` methods are the
    **kmstats** convention (letters-only denominator, PRICEBAND-1's sfb)."""

    def __init__(self, corpus=None):
        d = production_corpus_dir(corpus)
        bi = load_frequencies(str(d / "bigrams.txt"))
        ci = {c: i for i, c in enumerate(CHARS)}
        F = np.zeros((NC, NC))
        for ng, f in bi.items():
            if len(ng) == 2 and ng[0] in ci and ng[1] in ci:
                F[ci[ng[0]], ci[ng[1]]] += f
        self.FB = F
        # letters-only mass: the kmstats convention denominator.
        self.bi_total_km = F.sum()
        # space-inclusive mass: the oxey.pattern_shares convention denominator. Space is on
        # the board (`Layout.has_key(' ')` is True) so its bigrams count in bg_total, even
        # though space is in no scissor and no sfb. MEASURED from the corpus, not assumed.
        tot = 0.0
        for ng, f in bi.items():
            if len(ng) == 2 and all(c in ci or c == " " for c in ng):
                tot += f
        self.bi_total_pattern = tot
        self.km_over_pattern = tot / self.bi_total_km
        self.ci = ci
        g = ROW_STAGGERED_30
        slots = g.slots
        self.SCISSOR = np.zeros((NS, NS))
        self.SFB = np.zeros((NS, NS))
        for i in range(NS):
            for j in range(NS):
                a, b = slots[i], slots[j]
                if C.is_scissor(g, a, b):
                    self.SCISSOR[i, j] = 1.0
                if i != j and g.same_finger(a[0], b[0]):
                    self.SFB[i, j] = 1.0

    def perm(self, lay30):
        slot = {ch: i for i, ch in enumerate(lay30)}
        return np.array([slot[c] for c in CHARS], dtype=np.intp)

    def scissor_only(self, p):
        """`scissor` in the SHIPPED pattern_shares convention (space-inclusive denominator)."""
        return 100 * (self.FB * self.SCISSOR[np.ix_(p, p)]).sum() / self.bi_total_pattern

    def sfb_only(self, p):
        """`sfb` in the SHIPPED pattern_shares convention (space-inclusive denominator)."""
        return 100 * (self.FB * self.SFB[np.ix_(p, p)]).sum() / self.bi_total_pattern

    def scissor_km(self, p):
        """`scissor` in the kmstats convention (letters-only denominator)."""
        return 100 * (self.FB * self.SCISSOR[np.ix_(p, p)]).sum() / self.bi_total_km

    def sfb_km(self, p):
        """`sfb` in the kmstats convention -- PRICEBAND-1's sfb, for cross-comparison."""
        return 100 * (self.FB * self.SFB[np.ix_(p, p)]).sum() / self.bi_total_km


def verify(layouts):
    """Both gauges vs the SHIPPED `pattern_shares`, AND the kmstats forms vs shipped `KmStats`.

    Two independent shipped references, so a denominator mix-up cannot pass silently -- it is
    exactly what fired the first time this ran."""
    from keybo.analysis.kmstats import KmStats
    from keybo.layout import Layout
    from keybo.scoring.oxey import OxeyStyleScorer

    d = production_corpus_dir()
    sc = OxeyStyleScorer(
        load_frequencies(str(d / "bigrams.txt")),
        load_frequencies(str(d / "1-skip31.txt")),
        load_frequencies(str(d / "trigrams.txt")),
    )
    kms = KmStats(
        load_frequencies(str(d / "bigrams.txt")),
        load_frequencies(str(d / "1-skip31.txt")),
        load_frequencies(str(d / "trigrams.txt")),
    )
    fg = FastGauges()
    worst = 0.0
    print("== fastgauge verification vs shipped pattern_shares() AND KmStats.stats() ==")
    print(f"  denominator ratio km/pattern = {fg.km_over_pattern:.7f}")
    print(f"{'layout':<14}{'gauge':<16}{'shipped':>12}{'fast':>12}{'abserr':>11}")
    for n, s in layouts.items():
        L = Layout(s, ROW_STAGGERED_30)
        ref = sc.pattern_shares(L)
        p = fg.perm(s)
        pairs = [
            ("scissor:pattern", ref["scissor"], fg.scissor_only(p)),
            ("sfb:pattern", ref["sfb"], fg.sfb_only(p)),
            ("sfb:kmstats", kms.stats(s)["sfb"], fg.sfb_km(p)),
        ]
        for k, r, v in pairs:
            e = abs(r - v)
            worst = max(worst, e)
            print(f"{n:<14}{k:<16}{r:>12.6f}{v:>12.6f}{e:>11.2e}")
    print(f"worst abs error over {len(layouts)} layouts x 3 checks: {worst:.3e}")
    return worst
