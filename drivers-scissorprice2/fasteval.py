"""Exact O(31^3) re-implementation of timecard.TimeSurface.card().ms_per_char.

total = sum_{trigram abc} f(abc) * ( T2[slot(a),slot(b)] + Tc[slot(a),slot(b),slot(c)] )

Collapse the corpus to a 31x31x31 CHARACTER-index frequency cube once, then any layout is a
permutation of char-index -> slot-index and the total is a fancy-indexed sum. Verified
bit-close against the shipped card() in verify_against_card()."""
import numpy as np
from keybo.analysis.timecard import default_surface
from keybo.geometry import ROW_STAGGERED_30

CHARS = "abcdefghijklmnopqrstuvwxyz',.-"   # the C30M charset; space appended as index 30
NC = len(CHARS) + 1
SPACE_C = NC - 1

class FastSurface:
    def __init__(self, surf=None):
        self.surf = surf if surf is not None else default_surface()
        self.g = ROW_STAGGERED_30
        self.positions = [*self.g.slots, self.g.space_position]
        self.NP = len(self.positions)
        self.T2 = self.surf._T2
        self.Tc = self.surf._Tc
        ci = {c: i for i, c in enumerate(CHARS)}; ci[' '] = SPACE_C
        self.ci = ci
        F = np.zeros((NC, NC, NC))
        covered = 0
        for ng, f in self.surf.tri.items():
            try: a, b, c = ci[ng[0]], ci[ng[1]], ci[ng[2]]
            except KeyError: continue
            F[a, b, c] += f; covered += f
        self.F = F
        self.M = covered
        self.F2 = F.sum(axis=2)          # first-bigram marginal, char space

    def perm(self, lay30: str) -> np.ndarray:
        """char-index -> slot-index. lay30 is row-major over the 30 slots."""
        slot_of = {ch: i for i, ch in enumerate(lay30)}
        p = np.empty(NC, dtype=np.intp)
        for c, i in self.ci.items():
            p[i] = SPACE_C if c == ' ' else slot_of[c]
        p[SPACE_C] = self.NP - 1
        return p

    def parts(self, lay30):
        p = self.perm(lay30)
        t2 = float((self.F2 * self.T2[np.ix_(p, p)]).sum())
        tc = float((self.F * self.Tc[np.ix_(p, p, p)]).sum())
        return t2 / self.M, tc / self.M

    def ms_per_char(self, lay30):
        a, b = self.parts(lay30); return a + b

    def ms_per_char_perm(self, p):
        t2 = float((self.F2 * self.T2[np.ix_(p, p)]).sum())
        tc = float((self.F * self.Tc[np.ix_(p, p, p)]).sum())
        return (t2 + tc) / self.M

def verify_against_card(fs, layouts: dict):
    print("== fasteval verification against shipped TimeSurface.card() ==")
    print(f"{'layout':<14}{'card ms/char':>14}{'fast ms/char':>14}{'abs err':>12}")
    worst = 0.0
    for n, s in layouts.items():
        c = fs.surf.card(s).ms_per_char
        f = fs.ms_per_char(s)
        worst = max(worst, abs(c - f))
        print(f"{n:<14}{c:>14.6f}{f:>14.6f}{abs(c-f):>12.2e}")
    print(f"worst abs error over {len(layouts)} layouts: {worst:.3e} ms/char")
    return worst
