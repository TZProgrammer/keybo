"""SP5 — is the calibration argmax scissor's PRICE or a PATCH for the other ten? Plus the
speed-tied tiebreak, and the SATURATING-FORM question done properly.

SP4 found spearman rises monotonically in w_scissor to an argmax of 33.5-60.0, ABOVE even the
marginal implied +26..+33. That over-shoot is diagnostic: only ONE weight was free while the
other ten stayed at community-taste values that this audit says include three inverted signs.
So the free weight can absorb eleven-term misfit -- a 1-D patch of an 11-D misspecification
(the bundled-attribution shape of ARME-1's "72%").

DISCRIMINATOR — refit ALL ELEVEN weights jointly and read scissor's fitted weight off that.
If scissor's jointly-fitted weight lands near the conditional implied (+9..+18) rather than
near the single-weight argmax (34-60), the argmax was a patch. This is also the ONLY estimate
in which scissor's weight is not asked to stand in for its neighbours' errors -- and it is
literally the same estimand as `conditional_linear`, so agreement is a coherence check
(a JOINT scale-free refit of the score should recover the conditional beta ratio).

Three more things, all forced by earlier results:

 (a) A PLACEBO ON THE PATCH (trap 17/32). If a single free weight can buy spearman by
     absorbing misfit, then freeing some OTHER term's weight should buy some too. Sweep each
     of the eleven one at a time and report the gain. Scissor's gain is only meaningful
     RELATIVE to that placebo distribution -- "freeing w_scissor helps" is not evidence about
     scissor if freeing anything helps as much.
 (b) THE SPEED-TIED TIEBREAK. The six SPEEDTIE-1 champions are speed-indistinguishable
     (0.1760 ms/char = 2.85x the objective's own noise sd), so SPEEDTIE-1's registered rule
     makes the gauge frame the tiebreak. Does re-weighting scissor change the pick WITHIN
     that six?
 (c) THE FORM QUESTION, honestly. SP3 showed 75-97% of scissor's curvature is confounded. So
     compare LINEAR vs SATURATING vs CLAMPED specifications by out-of-sample fit (5-fold CV
     by SOURCE LAYOUT, so neighbours of one registry layout never straddle the split -- an
     in-sample R2 comparison would just reward the extra parameter).

FRAME: g-frame, 90 WPM baked, blend-v1, tau saturated. MODELLED only.
DEFAULT_OXEY_WEIGHTS is NOT edited anywhere; candidate weights go through the shipped
`OxeyStyleScorer(weights=...)` public override or a local weight vector.
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
SFB = TERMS.index("sfb")

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.surfaces import C30M  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS  # noqa: E402

NAT = (
    "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
    "old-new-layout-comparison/tri_frequency_old_new_surfaces"
)
ART = "/local/home/zegertho/agent/state/keybo-optimization/artifacts"
OUT = "/local/home/zegertho/agent/state/scissorprice/artifacts"
obj = SF.trigram_objective(SF.default_trigram_path(None))
MASS = obj[3].sum()
REG = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
USABLE = {n: s for n, s in REG.items() if set(s) == set(C30M)}
SRCS = ("AALTO", "COMMUNITY", "POOL")
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}
W0 = np.array([DEFAULT_OXEY_WEIGHTS[t][0] for t in TERMS])

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
prov = np.array(prov)
Y = {s: np.array([SF.score_fit(lay, SURF[s], obj) for lay in pool]) / MASS for s in SRCS}
print(f"pool n={n}, {len(USABLE)} source layouts")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def ols(A, y):
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    return co


res = {"n": n}

# ================= 1. JOINT REFIT of all eleven, anchored so sfb == +12.0 ================
print(f"\n{'='*84}\n1. JOINT REFIT of all eleven weights (vs the SINGLE-weight argmax)")
print("   A score is scale-free, so the fitted vector is rescaled to put sfb at +12.0 --")
print("   the same anchoring convention the dossier's 'implied weight' uses.")
joint = {}
for src in SRCS:
    co = ols(np.column_stack([np.ones(n), X]), Y[src])[1:]
    scale = 12.0 / co[SFB]
    w = co * scale
    joint[src] = {t: float(w[i]) for i, t in enumerate(TERMS)}
    print(f"\n  {src}: sfb-anchored jointly-fitted weights")
    print("    " + "  ".join(f"{t}={joint[src][t]:+.2f}" for t in TERMS))
res["joint_refit_sfb_anchored"] = joint
print(f"\n  {'src':12s}{'scissor JOINT':>15s}{'cond_linear':>13s}{'single-w argmax':>17s}{'shipped':>9s}")
sp3 = json.load(open(f"{OUT}/sp3_ratio_2x2.json"))
sp4 = json.load(open(f"{OUT}/sp4_calibration_and_scoring.json"))
for src in SRCS:
    print(f"  {src:12s}{joint[src]['scissor']:+15.3f}"
          f"{sp3['per_source'][src]['conditional_linear']['implied_weight']:+13.3f}"
          f"{sp4['part1_argmax'][src]['argmax_w']:+17.1f}{4.0:+9.1f}")

# ================= 2. PLACEBO: free EACH weight one at a time ============================
print(f"\n{'='*84}\n2. PLACEBO (trap 17/32) — free ONE weight at a time; how much spearman does each buy?")
print("   scissor's gain is only readable against this distribution.")
grid = np.arange(-100.0, 400.01, 1.0)
plac = {}
for src in SRCS:
    base = spearman(X @ W0, Y[src])
    row = {}
    for j, t in enumerate(TERMS):
        best_r, best_w = -2.0, None
        for wv in grid:
            w = W0.copy()
            w[j] = wv
            r = spearman(X @ w, Y[src])
            if r > best_r:
                best_r, best_w = r, float(wv)
        row[t] = {"argmax_w": best_w, "rho": float(best_r), "gain": float(best_r - base)}
    plac[src] = {"baseline_rho": float(base), "per_term": row}
    order = sorted(TERMS, key=lambda t: -row[t]["gain"])
    print(f"\n  {src}  (shipped-weights baseline rho = {base:+.5f})")
    print(f"    {'term':14s}{'argmax w':>10s}{'rho':>11s}{'gain':>10s}")
    for t in order:
        star = "  <== scissor" if t == "scissor" else ""
        print(f"    {t:14s}{row[t]['argmax_w']:+10.1f}{row[t]['rho']:+11.5f}"
              f"{row[t]['gain']:+10.5f}{star}")
res["placebo_free_one_weight"] = plac

# ================= 3. FORM: linear vs saturating vs clamped, OUT OF SAMPLE ===============
print(f"\n{'='*84}\n3. FORM of the scissor term — 5-fold CV grouped by SOURCE LAYOUT")
print("   (in-sample R2 would just reward the extra parameter; and neighbours of one registry")
print("    layout must not straddle the split, so folds are by source layout.)")
srclist = sorted(set(prov))
folds = [srclist[i::5] for i in range(5)]
CLAMP = (0.08, 3.06)  # the dossier's valid range for scissor


def build(spec_name, Xm):
    """design matrix for the scissor term under each specification, other ten linear."""
    s = Xm[:, SCI]
    rest = np.delete(Xm, SCI, axis=1)
    if spec_name == "linear":
        col = s[:, None]
    elif spec_name == "quadratic":
        col = np.column_stack([s, s**2])
    elif spec_name == "sqrt":
        col = np.sqrt(np.maximum(s, 0))[:, None]
    elif spec_name == "log1p":
        col = np.log1p(np.maximum(s, 0))[:, None]
    elif spec_name == "clamped_linear":
        col = np.clip(s, *CLAMP)[:, None]
    else:
        raise ValueError(spec_name)
    return np.column_stack([np.ones(len(Xm)), rest, col])


forms = ("linear", "quadratic", "sqrt", "log1p", "clamped_linear")
formres = {}
print(f"\n  {'form':16s}" + "".join(f"{s:>22s}" for s in SRCS))
print(f"  {'':16s}" + "".join(f"{'CV RMSE':>11s}{'CV R2':>11s}" for _ in SRCS))
for f in forms:
    cells = []
    for src in SRCS:
        errs, ys = [], []
        for fold in folds:
            te = np.isin(prov, fold)
            tr = ~te
            A_tr, A_te = build(f, X[tr]), build(f, X[te])
            co = ols(A_tr, Y[src][tr])
            pred = A_te @ co
            errs.append(Y[src][te] - pred)
            ys.append(Y[src][te])
        e = np.concatenate(errs)
        yy = np.concatenate(ys)
        cells.append((float(np.sqrt((e**2).mean())), float(1 - e.var() / yy.var())))
    formres[f] = {s: {"cv_rmse": c[0], "cv_r2": c[1]} for s, c in zip(SRCS, cells)}
    print(f"  {f:16s}" + "".join(f"{c[0]:11.5f}{c[1]:11.5f}" for c in cells))
res["form_cv"] = formres
res["clamp_used"] = CLAMP

# ================= 4. SPEED-TIED TIEBREAK ================================================
print(f"\n{'='*84}\n4. SPEED-TIED TIEBREAK — does re-weighting move the pick WITHIN the six?")
st = json.load(open(f"{ART}/speedtie-1/speedtie-summary.json"))
tie = [k for k in st["layouts"] if len(k) == 30 and set(k) == set(C30M)]
assert len(tie) == 6, f"expected 6, got {len(tie)}"
Xt = np.array([[shares_vec(s)[t] for t in TERMS] for s in tie])
mst = {s: np.array([SF.score_fit(lay, SURF[s], obj) for lay in tie]) / MASS for s in SRCS}
CAND_W = {"shipped": 4.0,
          "conditional_linear": float(np.mean(
              [sp3["per_source"][s]["conditional_linear"]["implied_weight"] for s in ("AALTO", "COMMUNITY")])),
          "conditional_tangent": float(np.mean(
              [sp3["per_source"][s]["conditional_tangent"]["implied_weight"] for s in ("AALTO", "COMMUNITY")])),
          "marginal_linear": float(np.mean(
              [sp3["per_source"][s]["marginal_linear"]["implied_weight"] for s in ("AALTO", "COMMUNITY")])),
          "marginal_tangent": float(np.mean(
              [sp3["per_source"][s]["marginal_tangent"]["implied_weight"] for s in ("AALTO", "COMMUNITY")])),
          "joint_refit": float(np.mean([joint[s]["scissor"] for s in ("AALTO", "COMMUNITY")]))}
tieres = {}
print(f"\n  {'weight':22s}{'w':>9s}   pick (argmin of oxey score among the six)")
for nm, wv in CAND_W.items():
    w = W0.copy()
    w[SCI] = wv
    sc = Xt @ w
    k = int(np.argmin(sc))
    tieres[nm] = {"w": wv, "pick": tie[k],
                  "order": [tie[i] for i in np.argsort(sc)],
                  "scores": {tie[i]: float(sc[i]) for i in range(6)}}
    print(f"  {nm:22s}{wv:+9.2f}   {tie[k]}")
print(f"\n  {'layout':34s}{'scissor%':>10s}" + "".join(f"{s[:4]+' ms':>11s}" for s in SRCS))
for i, lay in enumerate(tie):
    print(f"  {lay:34s}{Xt[i,SCI]:10.4f}" + "".join(f"{mst[s][i]:11.4f}" for s in SRCS))
res["speedtie_tiebreak"] = tieres

json.dump(res, open(f"{OUT}/sp5_joint_placebo_form.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp5_joint_placebo_form.json")
