"""SP3 — the four-cell ratio table, including the CONDITIONAL TANGENT cell nobody computed.

Two corrections to the 7.0x point in OPPOSITE directions:
  * CONDITIONING pushes it DOWN  (penaltyaudit's scissor_conditional.json: 2.25-4.40x;
    SP2 confirms the drop is an honest confound, not suppression).
  * SATURATION pushes it UP      (penaltyaudit's scissor_tangent.json: ~8.0-8.3x, because
    curvature is NEGATIVE and the operating share sits BELOW the pool mean, so the tangent
    at the operating point is STEEPER than the whole-support linear slope).
Neither agent combined them. The 2x2 is:

                      LINEAR slope        TANGENT at operating share
    MARGINAL          7.0x (shipped hdln)  ~8x (penaltyaudit, unverified by me)
    CONDITIONAL       2.2-4.4x             ???  <-- THE MISSING CELL

This probe fills all four cells with ONE estimator and ONE pool so they are commensurable,
re-derives penaltyaudit's tangent independently (it asked to be contradicted if wrong), and
adds the piece both of us are missing:

  ** WHICH ESTIMATOR CALIBRATES AN ADDITIVE WEIGHT? **
  The oxey score is sum_j w_j * share_j. A weight in that sum is asked to convert one term's
  share into score units. Whether the MARGINAL or the CONDITIONAL slope is the right
  calibration target is not a matter of taste -- it is empirically testable: build both
  scorers and measure which agrees better with the fitted ms/char it is trying to track.
  A marginal slope double-counts the shared factor (every term charges for it), so the
  prediction is that the marginal-calibrated scorer OVERSHOOTS. Trap 53 also applies: a
  linearized weight's sign/level does not tell you what the CURVE does.

CI: 2000-resample bootstrap over LAYOUTS on every ratio (the dossier gave a CI on the slope
but the ratio is what would be acted on, and a ratio of two noisy slopes is not a ratio of
two CIs -- it must be resampled jointly).

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
SCI, SFB = TERMS.index("scissor"), TERMS.index("sfb")

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
SRCS = ("AALTO", "COMMUNITY", "POOL")
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}

rng = random.Random(31337)


def neigh(s, k):
    lst = list(s)
    for _ in range(k):
        i, j = rng.randrange(30), rng.randrange(30)
        lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)


pool, prov = [], []
for nm, s in USABLE.items():
    pool.append(s)
    prov.append(nm)
    for _ in range(80):
        pool.append(neigh(s, rng.choice([1, 1, 2, 2, 3, 3, 4, 5])))
        prov.append(nm)
n = len(pool)
X = np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
REG_X = np.array([[shares_vec(s)[t] for t in TERMS] for s in USABLE.values()])
print(f"pool n={n};  registry n={len(USABLE)}")

# ------- operating shares: the REGISTRY (real layouts), which is the band of use ---------
op = {t: float(np.mean(REG_X[:, i])) for i, t in enumerate(TERMS)}
print(f"\noperating (registry-mean) shares:  scissor {op['scissor']:.4f}%   sfb {op['sfb']:.4f}%")
print(f"pool-mean shares:                  scissor {X[:,SCI].mean():.4f}%   sfb {X[:,SFB].mean():.4f}%")
print("registry scissor shares: " + ", ".join(
    f"{nm}={REG_X[i,SCI]:.4f}" for i, nm in enumerate(USABLE)))


def ols(A, y):
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    return co


def slopes(Xs, y, j, ctrl):
    """(linear slope, tangent slope at the operating share) for term j.

    linear  : coefficient on share_j
    tangent : d/ds of (c1*s + c2*s^2) at s = operating share  ->  c1 + 2*c2*s_op
    Both with the SAME control set, so the marginal/conditional axis and the
    linear/tangent axis are crossed cleanly.
    """
    base = [np.ones(len(Xs))] + [Xs[:, k] for k in ctrl]
    lin = ols(np.column_stack(base + [Xs[:, j]]), y)[-1]
    q = ols(np.column_stack(base + [Xs[:, j], Xs[:, j] ** 2]), y)
    c1, c2 = q[-2], q[-1]
    tan = c1 + 2.0 * c2 * op[TERMS[j]]
    return float(lin), float(tan), float(c2)


others = [j for j in range(len(TERMS)) if j not in (SCI,)]
oth_sfb = [j for j in range(len(TERMS)) if j not in (SFB,)]

SHIPPED_SFB, SHIPPED_SCI = 12.0, 4.0
res = {"n": n, "operating_shares": op, "per_source": {}}

for src in SRCS:
    y = np.array([SF.score_fit(lay, SURF[src], obj) for lay in pool]) / MASS

    # each term gets its own control set = "the other ten"
    sci_lin_m, sci_tan_m, sci_c2_m = slopes(X, y, SCI, [])
    sfb_lin_m, sfb_tan_m, sfb_c2_m = slopes(X, y, SFB, [])
    sci_lin_c, sci_tan_c, sci_c2_c = slopes(X, y, SCI, others)
    sfb_lin_c, sfb_tan_c, sfb_c2_c = slopes(X, y, SFB, oth_sfb)

    def ratio(sci_slope, sfb_slope):
        return (sci_slope / sfb_slope) * SHIPPED_SFB / SHIPPED_SCI

    def implied(sci_slope, sfb_slope):
        return (sci_slope / sfb_slope) * SHIPPED_SFB

    cells = {
        "marginal_linear": (sci_lin_m, sfb_lin_m),
        "marginal_tangent": (sci_tan_m, sfb_tan_m),
        "conditional_linear": (sci_lin_c, sfb_lin_c),
        "conditional_tangent": (sci_tan_c, sfb_tan_c),
    }

    # ---- bootstrap the RATIO jointly (2000 layout resamples) ----
    NB = 2000
    brng = np.random.default_rng(20260728)
    boot = {k: np.empty(NB) for k in cells}
    boot_imp = {k: np.empty(NB) for k in cells}
    idx = np.arange(n)
    for b in range(NB):
        ix = brng.choice(idx, size=n, replace=True)
        Xb, yb = X[ix], y[ix]
        a1, t1, _ = slopes(Xb, yb, SCI, [])
        a2, t2, _ = slopes(Xb, yb, SFB, [])
        a3, t3, _ = slopes(Xb, yb, SCI, others)
        a4, t4, _ = slopes(Xb, yb, SFB, oth_sfb)
        for k, (s1, s2) in {
            "marginal_linear": (a1, a2),
            "marginal_tangent": (t1, t2),
            "conditional_linear": (a3, a4),
            "conditional_tangent": (t3, t4),
        }.items():
            boot[k][b] = ratio(s1, s2)
            boot_imp[k][b] = implied(s1, s2)

    d = {"curvature_c2": {"scissor_marg": sci_c2_m, "scissor_cond": sci_c2_c,
                          "sfb_marg": sfb_c2_m, "sfb_cond": sfb_c2_c}}
    for k, (s1, s2) in cells.items():
        d[k] = {
            "scissor_slope": s1,
            "sfb_slope": s2,
            "ratio": ratio(s1, s2),
            "implied_weight": implied(s1, s2),
            "ratio_ci95": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))],
            "implied_ci95": [float(np.percentile(boot_imp[k], 2.5)),
                             float(np.percentile(boot_imp[k], 97.5))],
        }
    res["per_source"][src] = d

    print(f"\n{'='*80}\n{src}    (curvature c2: scissor marg {sci_c2_m:+.4f} cond {sci_c2_c:+.4f}"
          f" | sfb marg {sfb_c2_m:+.4f} cond {sfb_c2_c:+.4f})")
    print(f"  {'cell':22s}{'sci slope':>10s}{'sfb slope':>10s}{'ratio':>8s}"
          f"{'ratio CI95':>20s}{'implied w':>11s}{'implied CI95':>20s}")
    for k in ("marginal_linear", "marginal_tangent", "conditional_linear", "conditional_tangent"):
        c = d[k]
        print(f"  {k:22s}{c['scissor_slope']:+10.4f}{c['sfb_slope']:+10.4f}{c['ratio']:8.3f}"
              f"   [{c['ratio_ci95'][0]:6.3f},{c['ratio_ci95'][1]:6.3f}]"
              f"{c['implied_weight']:+11.3f}   [{c['implied_ci95'][0]:+7.3f},{c['implied_ci95'][1]:+7.3f}]")

print(f"\n{'='*80}\nTHE 2x2, cross-source (ratio vs shipped +4.0):")
print(f"  {'cell':24s}{'AALTO':>9s}{'COMM':>9s}{'POOL':>9s}{'min-max':>16s}")
for k in ("marginal_linear", "marginal_tangent", "conditional_linear", "conditional_tangent"):
    v = [res["per_source"][s][k]["ratio"] for s in SRCS]
    print(f"  {k:24s}{v[0]:9.3f}{v[1]:9.3f}{v[2]:9.3f}   {min(v):6.3f}-{max(v):6.3f}x")
print(f"\n  {'cell':24s}{'implied weight (sfb-anchored +12.0)':>40s}")
for k in ("marginal_linear", "marginal_tangent", "conditional_linear", "conditional_tangent"):
    v = [res["per_source"][s][k]["implied_weight"] for s in SRCS]
    print(f"  {k:24s}{v[0]:+13.3f}{v[1]:+13.3f}{v[2]:+13.3f}")

json.dump(res, open(f"{OUT}/sp3_ratio_2x2.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp3_ratio_2x2.json")
