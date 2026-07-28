"""SP2 — WHICH term absorbs scissor's price under conditioning, and is the conditional or
the marginal the contaminated number?

SP1 established that `scissor` is OUTSIDE the collinear cluster (BKW load 0.0002, VIF 2.16,
bootstrap sign-stable 3/3). But scissor's beta still falls +4.90 -> +1.57 when the other ten
enter. Two rival readings, and they lead to OPPOSITE actions:

  (i) HONEST CONTROL. The marginal +4.90 is confounded (scissor co-varies with something
      that genuinely costs time), the conditional +1.57 removes the confound, and the true
      price is ~+1.6. => the 7.0x should be softened to ~2-4x.
  (ii) SUPPRESSION BY AN UNIDENTIFIED REGRESSOR. The drop is absorbed by a term that is
      itself unidentified (alternate VIF 46, redirect VIF 19.5). Conditioning on an
      unidentified regressor does not "control" for anything -- it launders scissor's
      signal into a coefficient nobody can interpret. => the CONDITIONAL is the
      contaminated number and the marginal is closer to right.
      This is trap 49 run in the OTHER direction, and penaltyaudit itself used exactly this
      logic to license its three SIGN claims ("the conditional beta is negative ... that is
      textbook collinearity suppression at VIF 6.22, not a mechanism"). Consistency demands
      the same test here.

DISCRIMINATOR. Add the ten controls ONE AT A TIME and see which single addition moves
scissor's beta. Then repeat with the ill-conditioned block EXCLUDED. If the drop is carried
by {alternate, redirect} -- the two terms with VIF 46/19.5 -- reading (ii) holds. If it is
carried by well-identified terms, reading (i) holds.

Also computes:
  * the drop's decomposition via each control's own R2 against scissor (a control can only
    absorb what it shares),
  * a "clean conditional": scissor conditioned only on terms whose own VIF < 8 (i.e. the
    identified subframe), which is the estimate that is defensible under EITHER reading,
  * MARGINAL and CONDITIONAL side by side always (brief requirement).

FRAME: g-frame, 90 WPM baked, blend-v1, tau saturated. MODELLED only.
"""

import contextlib
import importlib.util
import io
import json
import random

import numpy as np

spec = importlib.util.spec_from_file_location("c3", "/tmp/scissorprice/probe/collin3.py")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    c3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c3)
assert any("max abs diff = 0" in ln for ln in buf.getvalue().splitlines() if "POSITIVE" in ln)
print("[inherited] share-path positive control: max abs diff = 0")

shares_vec, TERMS = c3.shares_vec, c3.TERMS
SCI = TERMS.index("scissor")

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.surfaces import C30M  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402

NAT = (
    "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
    "old-new-layout-comparison/tri_frequency_old_new_surfaces"
)
OUT = "/local/home/zegertho/agent/state/scissorprice/artifacts"
obj = SF.trigram_objective(SF.default_trigram_path(None))
MASS = obj[3].sum()
REG = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
USABLE = {n: s for n, s in REG.items() if set(s) == set(C30M)}

rng = random.Random(31337)


def neigh(s, k):
    lst = list(s)
    for _ in range(k):
        i, j = rng.randrange(30), rng.randrange(30)
        lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)


pool = []
for _n, s in USABLE.items():
    pool.append(s)
    for _ in range(80):
        pool.append(neigh(s, rng.choice([1, 1, 2, 2, 3, 3, 4, 5])))
n = len(pool)
X = np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
print(f"near-optimal pool n={n}")

SRCS = ("AALTO", "COMMUNITY", "POOL")
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}


def ols(A, y):
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    return co


def beta_scissor(y, ctrl_idx):
    """scissor's coefficient with the given control columns included."""
    cols = [np.ones(n), X[:, SCI]] + [X[:, j] for j in ctrl_idx]
    return float(ols(np.column_stack(cols), y)[1])


# VIF in this pool, to define the "identified subframe"
def vifs():
    Z = (X - X.mean(0)) / X.std(0)
    out = {}
    for j in range(len(TERMS)):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        co = ols(A, Z[:, j])
        resid = Z[:, j] - A @ co
        out[TERMS[j]] = float(1.0 / max(1e-12, 1.0 - (1 - resid.var() / Z[:, j].var())))
    return out


V = vifs()
ILL = [t for t in TERMS if V[t] >= 8.0]
CLEAN = [t for t in TERMS if t != "scissor" and V[t] < 8.0]
print(f"VIF>=8 (ill-conditioned block): {ILL}")
print(f"identified controls (VIF<8, excl scissor): {CLEAN}")

others = [j for j in range(len(TERMS)) if j != SCI]
res = {"n": n, "vif": V, "ill_conditioned": ILL, "clean_controls": CLEAN, "per_source": {}}

for src in SRCS:
    y = np.array([SF.score_fit(lay, SURF[src], obj) for lay in pool]) / MASS
    b_marg = beta_scissor(y, [])
    b_cond = beta_scissor(y, others)
    b_clean = beta_scissor(y, [TERMS.index(t) for t in CLEAN])
    b_illonly = beta_scissor(y, [TERMS.index(t) for t in ILL])

    # one-at-a-time: which SINGLE control moves scissor's beta most?
    one = {}
    for j in others:
        one[TERMS[j]] = beta_scissor(y, [j])
    # leave-one-control-out of the FULL conditional: which control's REMOVAL restores it?
    loco = {}
    for j in others:
        loco[TERMS[j]] = beta_scissor(y, [k for k in others if k != j])
    # correlation of each control with scissor (a control can only absorb what it shares)
    rho = {TERMS[j]: float(np.corrcoef(X[:, SCI], X[:, j])[0, 1]) for j in others}

    res["per_source"][src] = {
        "beta_marginal": b_marg,
        "beta_conditional_all10": b_cond,
        "beta_conditional_identified_only": b_clean,
        "beta_conditional_illblock_only": b_illonly,
        "one_at_a_time": one,
        "leave_one_control_out": loco,
        "rho_with_scissor": rho,
    }

    print(f"\n{'='*74}\n{src}")
    print(f"  MARGINAL (no controls)              beta = {b_marg:+.4f}")
    print(f"  CONDITIONAL on all ten              beta = {b_cond:+.4f}")
    print(f"  CONDITIONAL on IDENTIFIED only      beta = {b_clean:+.4f}   <- VIF<8 subframe")
    print(f"  CONDITIONAL on ILL block only       beta = {b_illonly:+.4f}   <- {ILL}")
    print(f"\n  {'control':14s}{'VIF':>7s}{'rho w/sci':>11s}{'beta|+ctrl':>12s}{'drop':>9s}"
          f"{'beta|all-minus':>15s}{'restore':>9s}")
    for t in sorted(others, key=lambda j: one[TERMS[j]]):
        tt = TERMS[t]
        print(f"  {tt:14s}{V[tt]:7.2f}{rho[tt]:+11.4f}{one[tt]:+12.4f}"
              f"{one[tt]-b_marg:+9.4f}{loco[tt]:+15.4f}{loco[tt]-b_cond:+9.4f}")

json.dump(res, open(f"{OUT}/sp2_absorption.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp2_absorption.json")
