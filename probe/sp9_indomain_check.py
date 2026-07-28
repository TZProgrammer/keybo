"""SP9 — is the IN-DOMAIN estimate (+32.59) trustworthy, or did the restriction break it?

SP8's headline rests on restricting the pool to the share range real layouts occupy. That
restriction is legitimate in principle (selecting on a REGRESSOR does not bias OLS, and a
weight applied to real layouts wants the LOCAL slope where they live), but it has two specific
hazards that must be checked rather than asserted:

  1. IDENTIFICATION. Restricting scissor's range removes scissor variance. If VIF(scissor)
     and the BKW load rise sharply in the subsample, the in-domain conditional is LESS
     identified than the full-pool one and the primary "scissor is outside the cluster"
     verdict would not carry over to the number I am quoting.
  2. IS IT THE RESTRICTION OR THE SUBSAMPLE? Dropping 43% of rows changes n and composition.
     A same-SIZE placebo (trap 17/32) is required: restrict on a DIFFERENT term's range to the
     same row count and see whether scissor's slope moves anyway. If a random/other-term
     restriction of equal size moves it as much, the in-domain number is a subsample artifact.

Also reports the concavity directly: the slope estimated in successive share BINS. If the
response really is concave, low-share bins must show a STEEPER slope than high-share bins --
which is the mechanism that makes the in-domain number larger, and is checkable independently
of any functional form.

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

NAT = ("/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
       "old-new-layout-comparison/tri_frequency_old_new_surfaces")
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
n = len(pool)

st = json.load(open(f"{ART}/speedtie-1/speedtie-summary.json"))
tie = [k for k in st["layouts"] if len(k) == 30 and set(k) == set(C30M)]
real = {f"speedtie:{k}": k for k in tie}
real.update({f"registry:{k}": v for k, v in USABLE.items()})
noq = np.array([shares_vec(v)["scissor"] for k, v in real.items() if "qwerty" not in k])
DOM = (float(noq.min()), float(noq.max()))
inD = (X[:, SCI] >= DOM[0]) & (X[:, SCI] <= DOM[1])
res = {"domain": DOM, "n_in_domain": int(inD.sum())}


def diag(Xm, label):
    """VIF + BKW load for scissor on a subsample."""
    Z = (Xm - Xm.mean(0)) / Xm.std(0)
    p = Xm.shape[1]
    others = np.delete(Z, SCI, axis=1)
    A = np.column_stack([np.ones(len(Z)), others])
    co = ols(A, Z[:, SCI])
    r2 = 1 - (Z[:, SCI] - A @ co).var() / Z[:, SCI].var()
    vif = 1.0 / max(1e-12, 1 - r2)
    Zu = Z / np.sqrt((Z**2).sum(0))
    _U, sv, Vt = np.linalg.svd(Zu, full_matrices=False)
    ci = sv.max() / sv
    phi = (Vt.T**2) / (sv**2)
    pi = phi / phi.sum(axis=1, keepdims=True)
    load = float(pi[SCI][ci >= 10].sum())
    return {"n": int(len(Xm)), "vif_scissor": float(vif), "max_cond_index": float(ci.max()),
            "bkw_load_scissor_ci10": load,
            "scissor_sd": float(Xm[:, SCI].std())}


print(f"\n{'='*80}\n1. IDENTIFICATION — did restricting the domain break it?")
d_full, d_in = diag(X, "full"), diag(X[inD], "in-domain")
res["identification"] = {"full_pool": d_full, "in_domain": d_in}
print(f"   {'':14s}{'n':>6s}{'VIF(sci)':>10s}{'maxCondIdx':>12s}{'BKWload(sci)':>14s}{'sd(sci)':>9s}")
for lbl, d in (("full pool", d_full), ("IN-DOMAIN", d_in)):
    print(f"   {lbl:14s}{d['n']:6d}{d['vif_scissor']:10.2f}{d['max_cond_index']:12.2f}"
          f"{d['bkw_load_scissor_ci10']:14.5f}{d['scissor_sd']:9.4f}")
verdict = "STILL WELL-IDENTIFIED" if d_in["vif_scissor"] < 8 and d_in["bkw_load_scissor_ci10"] < 0.5 \
    else "** IDENTIFICATION DEGRADED **"
print(f"   => {verdict}")
res["identification"]["verdict"] = verdict

print(f"\n{'='*80}\n2. SAME-SIZE PLACEBO (trap 17/32) — restrict on OTHER terms to the same n")
target_n = int(inD.sum())
plac = {}
print(f"   each placebo keeps ~{target_n} rows by restricting a DIFFERENT term to its central"
      f" range;\n   if scissor's slope moves as much there, the in-domain number is a subsample artifact.")
print(f"\n   {'restriction':22s}{'n':>6s}" + "".join(f"{s[:4]+' cond':>12s}" for s in SRCS))


def cond_sci(Xm, y):
    return float(ols(np.column_stack([np.ones(len(Xm)), Xm]), y)[1 + SCI])


base = {s: cond_sci(X, Y[s]) for s in SRCS}
print(f"   {'<none> (full pool)':22s}{n:6d}" + "".join(f"{base[s]:+12.4f}" for s in SRCS))
plac["<none>"] = {"n": n, **{s: base[s] for s in SRCS}}
row = {"n": target_n}
print(f"   {'scissor IN-DOMAIN':22s}{target_n:6d}"
      + "".join(f"{cond_sci(X[inD], Y[s][inD]):+12.4f}" for s in SRCS))
plac["scissor_in_domain"] = {"n": target_n,
                             **{s: cond_sci(X[inD], Y[s][inD]) for s in SRCS}}
prng = np.random.default_rng(20260728)
for j, t in enumerate(TERMS):
    if j == SCI:
        continue
    # central window on term j sized to the same row count
    order = np.argsort(X[:, j])
    lo = (n - target_n) // 2
    keep = np.zeros(n, bool)
    keep[order[lo:lo + target_n]] = True
    row = {"n": int(keep.sum())}
    line = f"   {'restrict ' + t:22s}{int(keep.sum()):6d}"
    for s in SRCS:
        v = cond_sci(X[keep], Y[s][keep])
        row[s] = v
        line += f"{v:+12.4f}"
    print(line)
    plac[f"restrict_{t}"] = row
# random subsample of the same size
keep = np.zeros(n, bool)
keep[prng.choice(n, size=target_n, replace=False)] = True
row = {"n": target_n}
line = f"   {'RANDOM same-size':22s}{target_n:6d}"
for s in SRCS:
    v = cond_sci(X[keep], Y[s][keep])
    row[s] = v
    line += f"{v:+12.4f}"
print(line)
plac["random_same_size"] = row
res["placebo_same_size"] = plac
for s in SRCS:
    vals = [plac[k][s] for k in plac if k not in ("<none>", "scissor_in_domain")]
    tgt = plac["scissor_in_domain"][s]
    print(f"   {s:10s} placebo range [{min(vals):+.4f},{max(vals):+.4f}]  "
          f"scissor-in-domain {tgt:+.4f}  "
          f"{'OUTSIDE placebo range' if tgt > max(vals) else '** inside placebo range **'}")

print(f"\n{'='*80}\n3. CONCAVITY, form-free — slope estimated within successive share BINS")
edges = [0.0, 0.2, 0.35, 0.55, 0.9, 1.5, 5.0]
print(f"   {'bin (share %)':18s}{'n':>6s}" + "".join(f"{s[:4]+' cond':>12s}" for s in SRCS))
bins = {}
for a, b in zip(edges[:-1], edges[1:]):
    m = (X[:, SCI] >= a) & (X[:, SCI] < b)
    if m.sum() < 60:
        continue
    row = {"n": int(m.sum())}
    line = f"   [{a:.2f},{b:.2f})".ljust(21) + f"{int(m.sum()):6d}"
    for s in SRCS:
        v = cond_sci(X[m], Y[s][m])
        row[s] = v
        line += f"{v:+12.4f}"
    print(line)
    bins[f"[{a},{b})"] = row
res["concavity_bins"] = bins
# ⚠ RESULT, not the expectation this section was written to test. The bins do NOT show a
# monotone decline: they are non-monotone and COMMUNITY goes NEGATIVE in two of them. Within a
# narrow bin the share variance is tiny, so the slope is poorly determined (trap 46's shape --
# too little information per cell to read a pattern). CONCLUSION: the in-domain estimate is
# robust (the same-size placebo and the identification diagnostics establish that), but a
# CONCAVE MECHANISM is NOT established by this test. The sqrt form winning out-of-sample (SP5)
# is consistent with concavity; it is not the same claim.
mono = all(
    bins[k][s] >= bins[kk][s]
    for s in SRCS
    for k, kk in zip(list(bins)[:-1], list(bins)[1:])
)
res["concavity_bins_monotone_declining"] = bool(mono)
res["concavity_verdict"] = (
    "MONOTONE DECLINE CONFIRMED" if mono else
    "NOT ESTABLISHED — bins non-monotone and/or sign-unstable; within-bin share variance is "
    "too small to determine a slope. The in-domain increase stands on the placebo + "
    "identification checks, NOT on a demonstrated concave mechanism."
)
print(f"   => {res['concavity_verdict']}")

json.dump(res, open(f"{OUT}/sp9_indomain_check.json", "w"), indent=1)
print(f"\nwrote {OUT}/sp9_indomain_check.json")
