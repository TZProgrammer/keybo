"""SP7 — the adversarial pass on MY OWN result. Four attacks, each able to kill the verdict.

Everything so far rests on one pool construction: 11 registry layouts x 81 perturbations
(1-5 random swaps), n=891. That is penaltyaudit's `form.py` design, reused so my numbers are
comparable to its dossier -- but a shared design is a shared failure mode, and it has a
specific hazard: **the 891 rows are 11 CLUSTERS of near-duplicates, not 891 independent
layouts.** An OLS/bootstrap over rows treats them as independent, which understates every CI
(the effective n is closer to 11 than to 891). penaltyaudit's dossier bootstraps "over
layouts" the same way. So:

  ATTACK 1 — CLUSTER BOOTSTRAP. Resample SOURCE LAYOUTS (11 of them), not rows. If the
    conditional beta's CI now spans zero, no per-term scissor number is supportable at all
    and both of us have been quoting a CI that is too narrow by construction.
  ATTACK 2 — LEAVE-ONE-SOURCE-LAYOUT-OUT. qwerty30m has scissor share 1.5831%, 3.1x the next
    highest (graphite 0.5173) and 11x the lowest. A slope fitted on a pool whose top of range
    is ONE layout's neighbourhood is a one-layout finding (trap 44's shape). Drop each source
    layout and re-read the slope.
  ATTACK 3 — PERTURBATION RADIUS. Does the slope depend on how far the neighbours wander?
    Rebuild at 1-swap only, 2-3 swaps, 4-5 swaps. If the estimate moves monotonically with
    radius the "band of use" is not a band, it is a gradient, and the number is an artifact of
    an arbitrary radius choice.
  ATTACK 4 — DOMAIN COVERAGE / OUT-OF-DOMAIN CHECK (trap 51/52). What share range do REAL
    layouts occupy, and is the pool's mass where the champions actually live? If the slope is
    driven by rows outside the champions' range, it is priced where nobody types -- the exact
    error the brief warns about, one level down.

Every attack reports MARGINAL and CONDITIONAL side by side (brief requirement).

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
ART = "/local/home/zegertho/agent/state/keybo-optimization/artifacts"
OUT = "/local/home/zegertho/agent/state/scissorprice/artifacts"
obj = SF.trigram_objective(SF.default_trigram_path(None))
MASS = obj[3].sum()
REG = {**NAMED_LAYOUTS, **_EXTRA_NAMED}
USABLE = {n: s for n, s in REG.items() if set(s) == set(C30M)}
SRCS = ("AALTO", "COMMUNITY", "POOL")
SURF = {s: np.load(f"{NAT}/{s}_TRI_PS_FREQ_PRIOR.native.npy") for s in SRCS}


def ols(A, y):
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    return co


def build_pool(radii, per=80, seed=31337):
    rng = random.Random(seed)

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
        for _ in range(per):
            pool.append(neigh(s, rng.choice(radii)))
            prov.append(nm)
    X = np.array([[shares_vec(t)[q] for q in TERMS] for t in pool])
    return pool, np.array(prov), X


def betas(X, y):
    """(marginal, conditional) slope of scissor."""
    m = float(ols(np.column_stack([np.ones(len(X)), X[:, SCI]]), y)[1])
    c = float(ols(np.column_stack([np.ones(len(X)), X]), y)[1 + SCI])
    return m, c


def ratio(X, y):
    """(marginal ratio, conditional ratio) vs shipped +4.0, sfb-anchored."""
    mfull = ols(np.column_stack([np.ones(len(X)), X]), y)
    cs, cf = mfull[1 + SCI], mfull[1 + SFB]
    ms = ols(np.column_stack([np.ones(len(X)), X[:, SCI]]), y)[1]
    mf = ols(np.column_stack([np.ones(len(X)), X[:, SFB]]), y)[1]
    return float((ms / mf) * 3.0), float((cs / cf) * 3.0)


BASE_RADII = [1, 1, 2, 2, 3, 3, 4, 5]
pool, prov, X = build_pool(BASE_RADII)
n = len(pool)
Y = {s: np.array([SF.score_fit(lay, SURF[s], obj) for lay in pool]) / MASS for s in SRCS}
srclist = sorted(set(prov))
print(f"base pool n={n}, source layouts {len(srclist)}")
res = {"n": n, "source_layouts": srclist}

# =================== ATTACK 1: CLUSTER BOOTSTRAP over source layouts =====================
print(f"\n{'='*84}\nATTACK 1 — CLUSTER BOOTSTRAP: resample the 11 SOURCE LAYOUTS, not the 891 rows")
print("   (the 891 rows are 11 clusters of near-duplicates; a row bootstrap treats them as")
print("    independent and understates every CI -- mine AND the dossier's)")
NB = 2000
brng = np.random.default_rng(20260728)
a1 = {}
print(f"\n   {'src':10s}{'marg pt':>9s}{'marg CI95 (cluster)':>24s}{'cond pt':>9s}"
      f"{'cond CI95 (cluster)':>24s}{'P(cond>0)':>11s}")
for s in SRCS:
    mb, cb = [], []
    for _b in range(NB):
        pick = brng.choice(srclist, size=len(srclist), replace=True)
        idx = np.concatenate([np.where(prov == p)[0] for p in pick])
        m, c = betas(X[idx], Y[s][idx])
        mb.append(m)
        cb.append(c)
    mb, cb = np.array(mb), np.array(cb)
    m0, c0 = betas(X, Y[s])
    a1[s] = {
        "marg_point": m0, "cond_point": c0,
        "marg_ci95_cluster": [float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))],
        "cond_ci95_cluster": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
        "p_cond_gt0": float((cb > 0).mean()), "p_marg_gt0": float((mb > 0).mean()),
    }
    d = a1[s]
    print(f"   {s:10s}{m0:+9.4f}   [{d['marg_ci95_cluster'][0]:+8.4f},{d['marg_ci95_cluster'][1]:+8.4f}]"
          f"{c0:+9.4f}   [{d['cond_ci95_cluster'][0]:+8.4f},{d['cond_ci95_cluster'][1]:+8.4f}]"
          f"{d['p_cond_gt0']:11.3f}")
# ratio CI under the cluster bootstrap
print(f"\n   RATIO vs shipped +4.0 under the CLUSTER bootstrap:")
print(f"   {'src':10s}{'marg ratio':>12s}{'CI95':>22s}{'cond ratio':>12s}{'CI95':>22s}{'P(>1)':>8s}")
for s in SRCS:
    rm, rc = [], []
    for _b in range(NB):
        pick = brng.choice(srclist, size=len(srclist), replace=True)
        idx = np.concatenate([np.where(prov == p)[0] for p in pick])
        try:
            a, b = ratio(X[idx], Y[s][idx])
        except Exception:
            continue
        rm.append(a)
        rc.append(b)
    rm, rc = np.array(rm), np.array(rc)
    r0m, r0c = ratio(X, Y[s])
    a1[s].update({
        "marg_ratio": r0m, "cond_ratio": r0c,
        "marg_ratio_ci95_cluster": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))],
        "cond_ratio_ci95_cluster": [float(np.percentile(rc, 2.5)), float(np.percentile(rc, 97.5))],
        "p_cond_ratio_gt1": float((rc > 1).mean()),
    })
    d = a1[s]
    print(f"   {s:10s}{r0m:12.3f}   [{d['marg_ratio_ci95_cluster'][0]:7.3f},{d['marg_ratio_ci95_cluster'][1]:7.3f}]"
          f"{r0c:12.3f}   [{d['cond_ratio_ci95_cluster'][0]:7.3f},{d['cond_ratio_ci95_cluster'][1]:7.3f}]"
          f"{d['p_cond_ratio_gt1']:8.3f}")
res["attack1_cluster_bootstrap"] = a1

# =================== ATTACK 2: leave-one-SOURCE-LAYOUT-out ===============================
print(f"\n{'='*84}\nATTACK 2 — LEAVE-ONE-SOURCE-LAYOUT-OUT (is this a qwerty30m finding?)")
isc = X[:, SCI]
print("   scissor share by source layout (mean of its 81 rows):")
for p in sorted(srclist, key=lambda p: -isc[prov == p].mean()):
    m = prov == p
    print(f"     {p:16s} mean {isc[m].mean():7.4f}%  range [{isc[m].min():.4f},{isc[m].max():.4f}]")
a2 = {}
print(f"\n   {'dropped':16s}" + "".join(f"{s[:4]+' marg':>12s}{s[:4]+' cond':>12s}" for s in SRCS))
for p in ["<none>"] + srclist:
    keep = np.ones(n, bool) if p == "<none>" else (prov != p)
    row = {}
    line = f"   {p:16s}"
    for s in SRCS:
        m, c = betas(X[keep], Y[s][keep])
        row[s] = {"marg": m, "cond": c}
        line += f"{m:+12.4f}{c:+12.4f}"
    print(line)
    a2[p] = row
res["attack2_leave_one_source_out"] = a2
for s in SRCS:
    cs = [a2[p][s]["cond"] for p in srclist]
    ms = [a2[p][s]["marg"] for p in srclist]
    print(f"   {s:10s} cond range over drops [{min(cs):+.4f},{max(cs):+.4f}]"
          f"  marg range [{min(ms):+.4f},{max(ms):+.4f}]"
          f"  {'ALL SAME SIGN' if min(cs)>0 else '** SIGN FLIPS **'}")

# =================== ATTACK 3: perturbation radius ======================================
print(f"\n{'='*84}\nATTACK 3 — PERTURBATION RADIUS (is the number an artifact of the radius?)")
a3 = {}
print(f"   {'radii':14s}{'n':>6s}{'sci share mean':>16s}"
      + "".join(f"{s[:4]+' marg':>12s}{s[:4]+' cond':>12s}" for s in SRCS))
for label, radii in (("1 only", [1]), ("2-3", [2, 3]), ("4-5", [4, 5]),
                     ("base 1-5", BASE_RADII), ("8-12 (wide)", [8, 10, 12])):
    pl, pv, Xr = build_pool(radii)
    row = {"n": len(pl), "scissor_share_mean": float(Xr[:, SCI].mean())}
    line = f"   {label:14s}{len(pl):6d}{Xr[:,SCI].mean():16.4f}"
    for s in SRCS:
        yr = np.array([SF.score_fit(lay, SURF[s], obj) for lay in pl]) / MASS
        m, c = betas(Xr, yr)
        row[s] = {"marg": m, "cond": c}
        line += f"{m:+12.4f}{c:+12.4f}"
    print(line)
    a3[label] = row
res["attack3_radius"] = a3

# =================== ATTACK 4: domain coverage ==========================================
print(f"\n{'='*84}\nATTACK 4 — DOMAIN COVERAGE (trap 51/52): is the pool's mass where champions live?")
st = json.load(open(f"{ART}/speedtie-1/speedtie-summary.json"))
tie = [k for k in st["layouts"] if len(k) == 30 and set(k) == set(C30M)]
assert len(tie) == 6
real = {f"speedtie:{k[:12]}": shares_vec(k)["scissor"] for k in tie}
real.update({f"registry:{k}": shares_vec(v)["scissor"] for k, v in USABLE.items()})
rv = np.array(list(real.values()))
print(f"   REAL layouts (17): scissor share range [{rv.min():.4f}, {rv.max():.4f}]%"
      f"  median {np.median(rv):.4f}%")
noq = np.array([v for k, v in real.items() if "qwerty" not in k])
print(f"   excluding qwerty30m (16): [{noq.min():.4f}, {noq.max():.4f}]%  median {np.median(noq):.4f}%")
print(f"   POOL n=891:            [{isc.min():.4f}, {isc.max():.4f}]%  mean {isc.mean():.4f}%")
frac_in = float(((isc >= noq.min()) & (isc <= noq.max())).mean())
print(f"   fraction of pool rows INSIDE the real (ex-qwerty) range: {100*frac_in:.1f}%")
print(f"   fraction of pool rows ABOVE  the real (ex-qwerty) max:   "
      f"{100*float((isc > noq.max()).mean()):.1f}%")
# re-estimate restricted to the champions' actual range
a4 = {"real_range_ex_qwerty": [float(noq.min()), float(noq.max())],
      "pool_range": [float(isc.min()), float(isc.max())],
      "frac_pool_in_real_range": frac_in}
keep = (isc >= noq.min()) & (isc <= noq.max())
print(f"\n   RE-ESTIMATE restricted to the champions' own share range (n={int(keep.sum())}):")
print(f"   {'src':10s}{'marg all':>11s}{'marg in-range':>15s}{'cond all':>11s}{'cond in-range':>15s}")
for s in SRCS:
    m0, c0 = betas(X, Y[s])
    m1, c1 = betas(X[keep], Y[s][keep])
    a4[s] = {"marg_all": m0, "cond_all": c0, "marg_inrange": m1, "cond_inrange": c1}
    print(f"   {s:10s}{m0:+11.4f}{m1:+15.4f}{c0:+11.4f}{c1:+15.4f}")
res["attack4_domain"] = a4

json.dump(res, open(f"{OUT}/sp7_robustness.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp7_robustness.json")
