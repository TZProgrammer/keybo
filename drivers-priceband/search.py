"""Constrained/unconstrained local search over the 30-slot board on the FAST evaluators.

Neighborhood = all C(30,2)=435 transpositions, i.e. the same 2-opt class the shipped
`local_search.two_opt` uses and the class the 13-board field is already converged in
(PREREG:10551: arm B is a strict 2-opt local optimum on the gauge). Steepest descent.

`constrained_min_ms` solves   min ms/char(L)  s.t.  |sfb(L) - target| <= tol
in two phases: (A) drive sfb into the band, (B) descend ms/char without leaving it.
That is the estimand the frontier needs: the BEST board at a given sfb, with every other
gauge RE-OPTIMIZED rather than held fixed -- which is what differences out disruption."""
import numpy as np

NS = 30
IJ = np.array([(i, j) for i in range(NS) for j in range(i + 1, NS)], dtype=np.intp)


def swap_perms(p):
    """(435, len(p)) array of p with each transposition of the 30 SLOT-values applied.

    p maps char-index -> slot-index. Transposing two SLOTS i,j means: every char sitting
    on slot i moves to j and vice versa. Vectorized as a relabel of p's values."""
    P = np.repeat(p[None, :], len(IJ), axis=0)
    a = IJ[:, 0]
    b = IJ[:, 1]
    r = np.arange(len(IJ))
    ia = (P == a[:, None])
    ib = (P == b[:, None])
    P[ia] = np.repeat(b, ia.sum(axis=1))
    P[ib] = np.repeat(a, ib.sum(axis=1))
    return P


class Objective:
    """Caches the fast evaluators and exposes vectorized sweeps over the 435 neighbours."""

    def __init__(self, fs, fg):
        self.fs = fs
        self.fg = fg

    def ms(self, p):
        return self.fs.ms_per_char_perm(p)

    def sfb(self, p30):
        return self.fg.sfb_only(p30)

    def to30(self, p):
        """fasteval perm is over 31 char-slots (space last); fastsfb wants the 30 letters."""
        return p[:30]

    def sweep(self, p, want_ms=True, feasible=None):
        """All 435 neighbours. sfb is ~16x cheaper than ms/char, so when a feasibility
        mask is supplied we evaluate ms/char ONLY on the feasible neighbours -- the
        constrained descent's inner loop cost then scales with the band width, not 435."""
        P = swap_perms(p)
        sfbs = np.array([self.sfb(q[:30]) for q in P])
        if not want_ms:
            return P, sfbs, None
        if feasible is None:
            mss = np.array([self.ms(q) for q in P])
        else:
            ok = feasible(sfbs)
            mss = np.full(len(P), np.inf)
            for k in np.where(ok)[0]:
                mss[k] = self.ms(P[k])
        return P, sfbs, mss


def two_opt_ms(obj, p, max_sweeps=200):
    """Unconstrained steepest-descent 2-opt on ms/char."""
    cur = obj.ms(p)
    for _ in range(max_sweeps):
        P, _, mss = obj.sweep(p)
        k = int(np.argmin(mss))
        if mss[k] < cur - 1e-12:
            p, cur = P[k], float(mss[k])
        else:
            return p, cur
    return p, cur


def drive_sfb(obj, p, target, tol, max_sweeps=200):
    """Phase A: steepest descent on (sfb-target)^2 until inside the band (or stuck)."""
    for _ in range(max_sweeps):
        cur = abs(obj.sfb(p[:30]) - target)
        if cur <= tol:
            return p, True
        P, sfbs, _ = obj.sweep(p, want_ms=False)
        d = np.abs(sfbs - target)
        k = int(np.argmin(d))
        if d[k] < cur - 1e-12:
            p = P[k]
        else:
            return p, abs(obj.sfb(p[:30]) - target) <= tol
    return p, abs(obj.sfb(p[:30]) - target) <= tol


def constrained_two_opt(obj, p, target, tol, max_sweeps=400):
    """Phase B: steepest ms/char descent restricted to neighbours that stay in the band."""
    cur = obj.ms(p)
    for _ in range(max_sweeps):
        P, sfbs, mss = obj.sweep(p, feasible=lambda s: np.abs(s - target) <= tol)
        k = int(np.argmin(mss))
        if np.isfinite(mss[k]) and mss[k] < cur - 1e-12:
            p, cur = P[k], float(mss[k])
        else:
            return p, cur
    return p, cur


def constrained_min_ms(obj, rng, target, tol, restarts, start_perms=None):
    """Best-of-`restarts` feasible local minimum of ms/char at sfb ~= target.

    Returns (best_ms, best_perm, best_sfb, n_feasible, all_ms)."""
    best = (np.inf, None, None)
    got = []
    for r in range(restarts):
        if start_perms is not None and r < len(start_perms):
            p = start_perms[r].copy()
        else:
            p = _random_perm(rng)
        p, feas = drive_sfb(obj, p, target, tol)
        if not feas:
            continue
        p, m = constrained_two_opt(obj, p, target, tol)
        s = obj.sfb(p[:30])
        if abs(s - target) > tol:
            continue
        got.append(m)
        if m < best[0]:
            best = (m, p.copy(), s)
    return best[0], best[1], best[2], len(got), np.array(got)


def _random_perm(rng):
    """Random 31-perm: the 30 letters shuffled over the 30 slots, space pinned to slot 30."""
    p = np.empty(31, dtype=np.intp)
    p[:30] = rng.permutation(30)
    p[30] = 30
    return p


def random_perm(rng):
    return _random_perm(rng)


# ---------------------------------------------------------------------------
# INEQUALITY-CAP formulation:  min ms/char(L)  s.t.  gauge(L) <= cap
#
# This is the formulation whose derivative is the SHADOW PRICE of the gauge
# constraint. Unlike a point target |gauge-t|<=tol, the feasible set is a
# half-space, so the descent has a large feasible neighborhood and actually
# converges -- the point-target version stalls in an almost-empty neighborhood
# (measured: it lands ~3 ms/char above the field best, i.e. not near-optimal).
# ---------------------------------------------------------------------------


def cap_two_opt(obj, p, cap, gauge=None, max_sweeps=400):
    """Steepest ms/char descent over neighbours satisfying gauge <= cap. Assumes p feasible."""
    g = gauge or (lambda q: obj.sfb(q[:30]))
    cur = obj.ms(p)
    for _ in range(max_sweeps):
        P = swap_perms(p)
        gs = np.array([g(q) for q in P])
        ok = gs <= cap
        if not ok.any():
            return p, cur
        mss = np.full(len(P), np.inf)
        for k in np.where(ok)[0]:
            mss[k] = obj.ms(P[k])
        k = int(np.argmin(mss))
        if np.isfinite(mss[k]) and mss[k] < cur - 1e-12:
            p, cur = P[k], float(mss[k])
        else:
            return p, cur
    return p, cur


def drive_under_cap(obj, p, cap, gauge=None, max_sweeps=200):
    """Phase A for the cap form: steepest descent on the gauge until gauge <= cap."""
    g = gauge or (lambda q: obj.sfb(q[:30]))
    for _ in range(max_sweeps):
        if g(p) <= cap:
            return p, True
        P = swap_perms(p)
        gs = np.array([g(q) for q in P])
        k = int(np.argmin(gs))
        if gs[k] < g(p) - 1e-12:
            p = P[k]
        else:
            return p, g(p) <= cap
    return p, g(p) <= cap


def cap_min_ms(obj, rng, cap, restarts, gauge=None, starts=None, polish3=False):
    """Best-of-`restarts` local minimum of ms/char subject to gauge <= cap.

    Random starts only unless `starts` is given -- a frontier must spend UNIFORM effort per
    cap, or caps near a seeded board's gauge value get an unfair head start and the frontier
    dips there, biasing the very slope we are estimating."""
    g = gauge or (lambda q: obj.sfb(q[:30]))
    vals = []
    best = (np.inf, None)
    for r in range(restarts):
        p = starts[r].copy() if (starts is not None and r < len(starts)) else _random_perm(rng)
        p, feas = drive_under_cap(obj, p, cap, gauge=g)
        if not feas:
            continue
        p, m = cap_two_opt(obj, p, cap, gauge=g)
        if polish3:
            p, m = cap_three_opt(obj, p, cap, gauge=g)
        if g(p) > cap + 1e-9:
            continue
        vals.append(m)
        if m < best[0]:
            best = (m, p.copy())
    return best[0], best[1], np.array(vals)


CYC = None


def _build_cycles():
    """All ordered 3-cycles of distinct slots (i j k): i->j->k->i. 30*29*28/3 = 8120."""
    out = []
    for i in range(NS):
        for j in range(NS):
            if j == i:
                continue
            for k in range(NS):
                if k == i or k == j:
                    continue
                if i < j and i < k:      # canonical rotation representative
                    out.append((i, j, k))
    return np.array(out, dtype=np.intp)


def cycle_perms(p):
    """(n_cycles, len(p)) array of p with each 3-cycle of SLOT labels applied."""
    global CYC
    if CYC is None:
        CYC = _build_cycles()
    P = np.repeat(p[None, :], len(CYC), axis=0)
    a, b, c = CYC[:, 0], CYC[:, 1], CYC[:, 2]
    ia, ib, ic = (P == a[:, None]), (P == b[:, None]), (P == c[:, None])
    P[ia] = np.repeat(b, ia.sum(axis=1))
    P[ib] = np.repeat(c, ib.sum(axis=1))
    P[ic] = np.repeat(a, ic.sum(axis=1))
    return P


def cap_three_opt(obj, p, cap, gauge=None, max_sweeps=60):
    """3-cycle (3-opt) steepest descent under the cap -- opens moves a transposition cannot
    express, which is exactly the 'richer perturbation class' route, used here as SEARCH."""
    g = gauge or (lambda q: obj.sfb(q[:30]))
    cur = obj.ms(p)
    for _ in range(max_sweeps):
        improved = False
        for P in (swap_perms(p), cycle_perms(p)):
            gs = np.array([g(q) for q in P])
            ok = gs <= cap
            if not ok.any():
                continue
            mss = np.full(len(P), np.inf)
            for k in np.where(ok)[0]:
                mss[k] = obj.ms(P[k])
            k = int(np.argmin(mss))
            if np.isfinite(mss[k]) and mss[k] < cur - 1e-12:
                p, cur = P[k], float(mss[k])
                improved = True
        if not improved:
            return p, cur
    return p, cur
