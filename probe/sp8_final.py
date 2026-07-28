"""SP8 — the definitive estimate, with all three correction axes applied at once.

SP3 found two correction axes to the 7.0x and SP7 found a third that both agents missed:
    CONDITIONING       pushes the ratio DOWN  (7.0x -> 2.2-4.4x)
    SATURATION/tangent pushes it UP           (marginal 7.0x -> 8.0x)
    DOMAIN RESTRICTION pushes it UP           (cond +1.5 -> +3.0..+3.6 in the real range)
Nobody has combined all three, and they do not compose by multiplying ratios -- they must be
estimated jointly on one design.

THE ESTIMATOR: conditional on the other ten (SP2 showed the drop is an honest confound, not
suppression, so conditioning is right) AND restricted to the share range REAL layouts occupy
(SP7 attack 4: 43% of the pool sits above it) AND with the CI from a CLUSTER bootstrap over
source layouts (SP7 attack 1: the 891 rows are 11 clusters) AND under the form that wins
out-of-sample (SP5: sqrt beats linear and quadratic 3/3).

Also reports the honest denominator: the SAME treatment applied to `sfb`, the anchor. If
restricting the domain and conditioning change sfb's slope too, the RATIO moves less than
either slope, and the ratio is what the weight is.

And a final decision-relevant check: with the best-supported weight, does the ranking of the
17 real layouts move, and does the argmin move?

FRAME: g-frame, 90 WPM baked, blend-v1, tau saturated. MODELLED only. No weight is edited.
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
INDEP = ("AALTO", "COMMUNITY")  # POOL is a superset, not a third source (dossier sec.7)
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}
W0 = np.array([DEFAULT_OXEY_WEIGHTS[t][0] for t in TERMS])


def ols(A, y):
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    return co


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
prov = np.array(prov)
X = np.array([[shares_vec(t)[q] for q in TERMS] for t in pool])
Y = {s: np.array([SF.score_fit(lay, SURF[s], obj) for lay in pool]) / MASS for s in SRCS}
srclist = sorted(set(prov))

# ---- the real layouts, and the DOMAIN they define -------------------------------------
st = json.load(open(f"{ART}/speedtie-1/speedtie-summary.json"))
tie = [k for k in st["layouts"] if len(k) == 30 and set(k) == set(C30M)]
assert len(tie) == 6
real = {f"speedtie:{k}": k for k in tie}
real.update({f"registry:{k}": v for k, v in USABLE.items()})
rshare = {k: shares_vec(v)["scissor"] for k, v in real.items()}
noq = np.array([v for k, v in rshare.items() if "qwerty" not in k])
DOM = (float(noq.min()), float(noq.max()))
print(f"\nDOMAIN of use (16 real C30M layouts, qwerty30m excluded as an outlier and NOT a")
print(f"  plausible optimizer target): scissor share [{DOM[0]:.4f}, {DOM[1]:.4f}]%")
inD = (X[:, SCI] >= DOM[0]) & (X[:, SCI] <= DOM[1])
print(f"  pool rows in domain: {int(inD.sum())} of {len(pool)} ({100*inD.mean():.1f}%)")

res = {"domain": DOM, "n_pool": len(pool), "n_in_domain": int(inD.sum()),
       "real_layout_scissor_share": rshare}


def est(Xm, y, form="linear"):
    """conditional slope of scissor (and of sfb) under a given form, at the domain midpoint.

    For the sqrt form the 'slope in share units' is d/ds a*sqrt(s) = a/(2 sqrt(s)), which is
    share-dependent, so it is evaluated at the MEDIAN REAL share -- the operating point. That
    keeps the sqrt and linear numbers in the same units and comparable to a weight.
    """
    s_op = float(np.median([v for k, v in rshare.items() if "qwerty" not in k]))
    rest_sci = np.delete(Xm, SCI, axis=1)
    rest_sfb = np.delete(Xm, SFB, axis=1)
    if form == "linear":
        c_sci = ols(np.column_stack([np.ones(len(Xm)), rest_sci, Xm[:, SCI]]), y)[-1]
        c_sfb = ols(np.column_stack([np.ones(len(Xm)), rest_sfb, Xm[:, SFB]]), y)[-1]
        return float(c_sci), float(c_sfb)
    if form == "sqrt":
        a_sci = ols(np.column_stack([np.ones(len(Xm)), rest_sci,
                                     np.sqrt(np.maximum(Xm[:, SCI], 0))]), y)[-1]
        a_sfb = ols(np.column_stack([np.ones(len(Xm)), rest_sfb,
                                     np.sqrt(np.maximum(Xm[:, SFB], 0))]), y)[-1]
        sfb_op = float(np.median([shares_vec(v)["sfb"] for k, v in real.items() if "qwerty" not in k]))
        return float(a_sci / (2 * np.sqrt(s_op))), float(a_sfb / (2 * np.sqrt(sfb_op)))
    raise ValueError(form)


# =============== THE HEADLINE TABLE: every combination of the three axes =================
print(f"\n{'='*88}\nTHE DEFINITIVE TABLE — implied weight (sfb-anchored +12.0) and ratio vs shipped +4.0")
print("  every row is a CONDITIONAL estimate (SP2: the drop is an honest confound, so")
print("  conditioning is correct); rows differ in DOMAIN and FORM.")
rows = [
    ("cond, full pool,  linear", slice(None), "linear"),
    ("cond, IN-DOMAIN,  linear", inD, "linear"),
    ("cond, full pool,  sqrt", slice(None), "sqrt"),
    ("cond, IN-DOMAIN,  sqrt", inD, "sqrt"),
]
NB = 2000
brng = np.random.default_rng(20260728)
table = {}
print(f"\n  {'estimate':26s}" + "".join(f"{s[:4]+' w':>9s}{'ratio':>7s}" for s in SRCS)
      + f"{'w (indep mean)':>16s}{'cluster CI95 on w':>22s}{'P(r>1)':>8s}")
for label, mask, form in rows:
    cells, ws = {}, []
    for s in SRCS:
        Xm = X[mask] if not isinstance(mask, slice) else X
        ym = Y[s][mask] if not isinstance(mask, slice) else Y[s]
        cs, cf = est(Xm, ym, form)
        w = (cs / cf) * 12.0
        cells[s] = {"scissor_slope": cs, "sfb_slope": cf, "implied_w": float(w),
                    "ratio": float(w / 4.0)}
        ws.append(w)
    w_indep = float(np.mean([cells[s]["implied_w"] for s in INDEP]))
    # cluster bootstrap on the independent-source mean implied weight
    bw = []
    for _b in range(NB):
        pick = brng.choice(srclist, size=len(srclist), replace=True)
        idx = np.concatenate([np.where(prov == p)[0] for p in pick])
        if not isinstance(mask, slice):
            idx = idx[mask[idx]]
        if len(idx) < 60:
            continue
        try:
            vals = []
            for s in INDEP:
                cs, cf = est(X[idx], Y[s][idx], form)
                vals.append((cs / cf) * 12.0)
            bw.append(float(np.mean(vals)))
        except Exception:
            continue
    bw = np.array(bw)
    ci = [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))]
    p_gt = float((bw / 4.0 > 1).mean())
    table[label] = {"per_source": cells, "w_indep_mean": w_indep,
                    "w_cluster_ci95": ci, "p_ratio_gt_1": p_gt, "n_boot": int(len(bw))}
    print(f"  {label:26s}" + "".join(f"{cells[s]['implied_w']:+9.2f}{cells[s]['ratio']:7.2f}" for s in SRCS)
          + f"{w_indep:+16.2f}   [{ci[0]:+7.2f},{ci[1]:+7.2f}]{p_gt:8.3f}")
res["definitive_table"] = table

# =============== the DECISION: does the best-supported weight move the ranking? ==========
print(f"\n{'='*88}\nDECISION CHECK — the best-supported weight on the 17 real layouts")
BEST = table["cond, IN-DOMAIN,  linear"]["w_indep_mean"]
print(f"  best-supported weight (conditional, in-domain, linear, independent-source mean)"
      f" = {BEST:+.2f}")
print(f"  cluster CI95 = [{table['cond, IN-DOMAIN,  linear']['w_cluster_ci95'][0]:+.2f},"
      f" {table['cond, IN-DOMAIN,  linear']['w_cluster_ci95'][1]:+.2f}]")
names = list(real)
Xr = np.array([[shares_vec(real[nm])[t] for t in TERMS] for nm in names])


def rank_under(w_sci):
    w = W0.copy()
    w[SCI] = w_sci
    sc = Xr @ w
    order = np.argsort(sc)
    return [names[i] for i in order], {names[i]: int(r) for r, i in enumerate(order)}


base_o, base_r = rank_under(4.0)
lo, hi = table["cond, IN-DOMAIN,  linear"]["w_cluster_ci95"]
print(f"\n  {'weight':34s}{'argmin':>36s}{'moved':>7s}{'maxmove':>9s}")
dec = {}
for lbl, wv in (("shipped +4.0", 4.0), ("best-supported", BEST), ("CI low", lo), ("CI high", hi),
                ("marginal headline (+28.0)", 28.0)):
    o, r = rank_under(wv)
    moved = sum(1 for nm in names if r[nm] != base_r[nm])
    mx = max(abs(r[nm] - base_r[nm]) for nm in names)
    dec[lbl] = {"w": float(wv), "argmin": o[0], "moved": moved, "max_move": mx,
                "top5": o[:5]}
    print(f"  {lbl:34s}{o[0][:36]:>36s}{moved:7d}{mx:9d}")
res["decision_check"] = dec
print(f"\n  argmin is {'INVARIANT' if len({d['argmin'] for d in dec.values()})==1 else 'NOT invariant'}"
      f" across shipped, best-supported, both CI ends, and the marginal headline.")

json.dump(res, open(f"{OUT}/sp8_definitive.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp8_definitive.json")
