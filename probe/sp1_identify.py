"""SP1 — is `scissor` INSIDE or OUTSIDE penaltyaudit's 5-term collinear cluster?

penaltyaudit answered this with average-linkage hierarchical clustering on `1-|rho|` at
K=6, and reported {scissor, outroll} as its own cluster with leave-one-cluster-out
delta-R2 0.12-0.16 while the 5-term cluster {sfb, onehand, redirect, alternate,
imbalance} sits at 0.002-0.020. But it also records that SINGLE linkage "chained all 11
into one group -- a known pathology, discarded". So the membership verdict rests on a
LINKAGE CHOICE. That is exactly the shape of a result that needs a clustering-FREE check
(traps 25/49): the cluster is a means, not the estimand.

So this probe answers the membership question FOUR ways that do not use clustering at all:

  A. VIF + the BKW variance-decomposition proportions. BKW is the standard
     clustering-free diagnostic: a term is implicated in a near-dependency iff it loads
     heavily on a HIGH-condition-index eigenvector. This attributes collinearity to
     TERMS rather than to groups, and it is what "is X in the collinear set?" means
     without a dendrogram.
  B. Leave-one-TERM-out delta-R2 (the per-term analogue of loco), which is what the
     "priors, not measurements" warning is really about.
  C. Bootstrap stability of the CONDITIONAL beta. A term inside a collinear cluster has
     a conditional beta whose SIGN is unstable across resamples (that is the operational
     content of "unidentified"); a separately-identified term does not.
  D. The scissor-vs-outroll SPLIT specifically. {scissor,outroll} is well-identified as a
     CLUSTER; that does not license a per-term scissor number unless the two are
     separable. Test: partial outroll out of scissor and see whether the price survives.

POOL: near-optimal band (the band of use), built EXACTLY as penaltyaudit's form.py does
so my numbers are comparable to its dossier -- and ALSO on the random pool, so the
band-dependence is visible rather than assumed.

FRAME: g-frame only (geometry; layout-independent b(ngram) excluded), surfaces baked at
90 WPM, corpus blend-v1, tau saturated. MODELLED -- no realized-speed claim.
"""

import contextlib
import importlib.util
import io
import json
import random

import numpy as np

# --- reuse the POSITIVE-CONTROLLED share instrument (never re-implement it: trap 28) ---
spec = importlib.util.spec_from_file_location("c3", "/tmp/scissorprice/probe/collin3.py")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    c3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c3)
_ctrl = [ln for ln in buf.getvalue().splitlines() if "POSITIVE CONTROL" in ln]
for ln in _ctrl:
    print("[inherited]", ln.strip())
assert any("max abs diff = 0" in ln for ln in _ctrl), "share-path control did not pass"

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

# ---------------------------------------------------------------- pools
rng = random.Random(31337)  # same seed as penaltyaudit's form.py/scissor_cond.py


def neigh(s, k):
    lst = list(s)
    for _ in range(k):
        i, j = rng.randrange(30), rng.randrange(30)
        lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)


near = []
for _n, s in USABLE.items():
    near.append(s)
    for _ in range(80):
        near.append(neigh(s, rng.choice([1, 1, 2, 2, 3, 3, 4, 5])))

rng2 = random.Random(20260728)


def rl():
    ch = list(C30M)
    rng2.shuffle(ch)
    return "".join(ch)


rand = [rl() for _ in range(400)]

POOLS = {"near_optimal": near, "random": rand}
print(f"\npools: near_optimal n={len(near)} ({len(USABLE)} C30M-exact registry x 81)  random n={len(rand)}")

SRCS = ("AALTO", "COMMUNITY", "POOL")
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}


def design(pool):
    return np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])


def target(pool, src):
    return np.array([SF.score_fit(lay, SURF[src], obj) for lay in pool]) / MASS


def ols(A, y):
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    return co


def r2(A, y):
    co = ols(A, y)
    resid = y - A @ co
    return 1.0 - resid.var() / y.var()


results = {}
for pname, pool in POOLS.items():
    X = design(pool)
    Z = (X - X.mean(0)) / X.std(0)  # standardized, for VIF/BKW
    n, p = X.shape
    res = {"n": n}

    # ---------------- A. VIF and the BKW variance-decomposition proportions ----------
    # VIF_j = 1/(1-R2_j) from regressing column j on the others.
    vif = {}
    for j in range(p):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        vif[TERMS[j]] = 1.0 / max(1e-12, 1.0 - r2(A, Z[:, j]))

    # BKW: SVD of the standardized (unit-column-scaled) design. Condition indices
    # eta_k = smax/s_k ; variance-decomposition proportion pi_{k,j} = (v_kj^2/s_k^2) /
    # sum_k (v_kj^2/s_k^2).  A near-dependency is "implicated in term j" when eta_k is
    # large (>=30 is Belsley's rule) AND pi_{k,j} is large (>=0.5).
    Zu = Z / np.sqrt((Z**2).sum(0))  # unit column length (BKW's scaling)
    _U, sv, Vt = np.linalg.svd(Zu, full_matrices=False)
    cond_idx = sv.max() / sv
    phi = (Vt.T**2) / (sv**2)  # (p terms, p components)
    pi = phi / phi.sum(axis=1, keepdims=True)  # rows sum to 1 over components
    res["cond_index"] = [float(c) for c in cond_idx]
    res["bkw_pi"] = {TERMS[j]: [float(x) for x in pi[j]] for j in range(p)}
    # "collinearity load" = share of term j's variance sitting on components whose
    # condition index exceeds the threshold. This is the clustering-free membership test.
    for thr in (10.0, 30.0):
        bad = cond_idx >= thr
        res[f"bkw_load_ci{int(thr)}"] = {
            TERMS[j]: float(pi[j][bad].sum()) for j in range(p)
        }
    res["vif"] = {k: float(v) for k, v in vif.items()}

    # ---------------- B/C/D per source ---------------------------------------------
    res["per_source"] = {}
    for src in SRCS:
        y = target(pool, src)
        A_full = np.column_stack([np.ones(n), X])
        r2_full = r2(A_full, y)
        co_full = ols(A_full, y)
        cond_beta = {TERMS[j]: float(co_full[1 + j]) for j in range(p)}

        marg = {}
        for j in range(p):
            marg[TERMS[j]] = float(ols(np.column_stack([np.ones(n), X[:, j]]), y)[1])
        # B. leave-one-TERM-out delta R2
        loto = {}
        for j in range(p):
            A = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
            loto[TERMS[j]] = float(r2_full - r2(A, y))

        # C. bootstrap stability of the conditional beta (layout resample)
        NB = 2000
        bs = np.empty((NB, p))
        idx_all = np.arange(n)
        brng = np.random.default_rng(20260728)
        for b in range(NB):
            ix = brng.choice(idx_all, size=n, replace=True)
            Ab = np.column_stack([np.ones(n), X[ix]])
            bs[b] = ols(Ab, y[ix])[1:]
        frac_pos = {TERMS[j]: float((bs[:, j] > 0).mean()) for j in range(p)}
        ci = {
            TERMS[j]: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
            for j in range(p)
        }

        # D. the scissor-vs-outroll SPLIT: partial outroll out of BOTH scissor and y,
        # then regress. If scissor's price survives residualizing on its own cluster
        # partner, the pair is separable and a per-term scissor number is licensed.
        jo = TERMS.index("outroll")
        Ao = np.column_stack([np.ones(n), X[:, jo]])
        sci_res = X[:, SCI] - Ao @ ols(Ao, X[:, SCI])
        y_res = y - Ao @ ols(Ao, y)
        beta_split = float(ols(np.column_stack([np.ones(n), sci_res]), y_res)[1])
        rho_sci_out = float(np.corrcoef(X[:, SCI], X[:, jo])[0, 1])

        res["per_source"][src] = {
            "r2_full": float(r2_full),
            "cond_beta": cond_beta,
            "marginal_beta": marg,
            "loto_dr2": loto,
            "boot_frac_pos": frac_pos,
            "boot_ci95": ci,
            "scissor_partial_outroll_beta": beta_split,
            "rho_scissor_outroll": rho_sci_out,
        }
    results[pname] = res

# ------------------------------------------------------------------ report
CLUSTER5 = ["sfb", "onehand", "redirect", "alternate", "imbalance"]
for pname in POOLS:
    R = results[pname]
    print(f"\n{'='*78}\nPOOL: {pname}  (n={R['n']})")
    print("  condition indices:", "  ".join(f"{c:.1f}" for c in R["cond_index"]))
    print(
        f"\n  {'term':14s}{'w':>6s}{'VIF':>8s}{'BKW>=10':>9s}{'BKW>=30':>9s}"
        f"{'lotoR2 A':>10s}{'lotoR2 C':>10s}{'lotoR2 P':>10s}  {'in5cluster'}"
    )
    for t in TERMS:
        l5 = "YES" if t in CLUSTER5 else "-"
        lo = [R["per_source"][s]["loto_dr2"][t] for s in SRCS]
        print(
            f"  {t:14s}{c3.DEFAULT_OXEY_WEIGHTS[t][0]:+6.1f}{R['vif'][t]:8.2f}"
            f"{R['bkw_load_ci10'][t]:9.3f}{R['bkw_load_ci30'][t]:9.3f}"
            f"{lo[0]:10.4f}{lo[1]:10.4f}{lo[2]:10.4f}  {l5}"
        )
    print(f"\n  {'src':10s}{'rho(sci,out)':>13s}{'marg':>9s}{'cond':>9s}{'partial-out':>12s}"
          f"{'boot CI95 cond':>26s}{'P(>0)':>8s}")
    for s in SRCS:
        d = R["per_source"][s]
        ci = d["boot_ci95"]["scissor"]
        print(
            f"  {s:10s}{d['rho_scissor_outroll']:+13.4f}{d['marginal_beta']['scissor']:+9.4f}"
            f"{d['cond_beta']['scissor']:+9.4f}{d['scissor_partial_outroll_beta']:+12.4f}"
            f"   [{ci[0]:+9.4f},{ci[1]:+9.4f}]{d['boot_frac_pos']['scissor']:8.3f}"
        )

json.dump(results, open(f"{OUT}/sp1_identification.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp1_identification.json")
