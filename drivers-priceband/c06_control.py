"""C06 -- POSITIVE CONTROL P1 (mandatory gate) + the LITERAL sign-blind falsifier (6a).

Re-derives the magnitude-matched paired estimator FROM MY OWN swap sweep in MY OWN code path
(not by reading the prior arm's npz) and must recover +0.39 ms/pp on qwerty, CI [+0.09,+0.64].
Also runs the mandated signed-vs-sign-blind nested R2 on the perturbation data, in-band and qwerty.

If P1 fails, my instrument stack is broken and the whole arm stops (pre-registered)."""
import json

import _env
import numpy as np
from boards import FIELD, OFF_FRONTIER, OPTIMIZED

import fastsfb
import search as S

rng = np.random.default_rng(20260801)
fs, w1, w2 = _env.verify_evaluators(FIELD)
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
print(f"evaluators re-verified: {w1:.2e} / {w2:.2e}")

# ---------- my own single-swap sweep over all 14 boards ----------
recs = {}
for b in sorted(FIELD):
    p0 = fs.perm(FIELD[b])
    m0, s0 = obj.ms(p0), obj.sfb(p0[:30])
    P, sfbs, mss = obj.sweep(p0)
    recs[b] = dict(dsfb=sfbs - s0, dms=mss - m0)
print(f"swept {len(recs)} boards x {len(S.IJ)} transpositions")

# cross-check against the prior arm's npz (same quantity, independent code path)
prior = np.load("/local/home/zegertho/agent/state/pair-perturb/artifacts/p01_swaps.npz", allow_pickle=True)
pb, pdsfb, pdms = prior["board"], prior["dsfb"], prior["dms"]
worst_s = worst_m = 0.0
for b in sorted(FIELD):
    m = pb == b
    if not m.any():
        continue
    o = np.argsort(pdsfb[m], kind="stable")
    n = np.argsort(recs[b]["dsfb"], kind="stable")
    worst_s = max(worst_s, float(np.abs(pdsfb[m][o] - recs[b]["dsfb"][n]).max()))
    worst_m = max(worst_m, float(np.abs(pdms[m][o] - recs[b]["dms"][n]).max()))
print(f"cross-check vs prior arm p01_swaps.npz: worst |d dsfb| {worst_s:.2e}   worst |d dms| {worst_m:.2e}")

# ---------- the paired estimator (pre-registered form) ----------
def match(dsfb, dms, tol):
    up = np.where(dsfb > 1e-9)[0]
    dn = np.where(dsfb < -1e-9)[0]
    au, ad = np.abs(dsfb[up]), np.abs(dsfb[dn])
    used = np.zeros(len(up), bool)
    pairs = []
    for di in np.argsort(ad)[::-1]:            # scarce side, big first
        d = np.abs(au - ad[di]); d[used] = np.inf
        uj = int(np.argmin(d))
        if d[uj] <= tol:
            used[uj] = True
            pairs.append((0.5 * (au[uj] + ad[di]), dms[up[uj]], dms[dn[di]]))
    return pairs


def price_wls(pairs):
    if not pairs:
        return np.nan, 0
    x = np.array([p[0] for p in pairs])
    D = np.array([p[1] for p in pairs]) - np.array([p[2] for p in pairs])
    return float((2 * x * D).sum() / ((2 * x) ** 2).sum()), len(pairs)


print("\n== P1 POSITIVE CONTROL: paired price on qwerty30m (target +0.39, CI [+0.09,+0.64]) ==")
q = recs[OFF_FRONTIER]
out = {}
for tol in (0.05, 0.10, 0.20):
    pr, n = price_wls(match(q["dsfb"], q["dms"], tol))
    out[f"qwerty_tol{tol}"] = dict(price=pr, n=n)
    tag = "  <== PRIMARY" if tol == 0.05 else ""
    print(f"   delta={tol:<5} n_pairs={n:>4}  price = {pr:+.4f} ms/char per pp{tag}")

pairs = match(q["dsfb"], q["dms"], 0.05)
x = np.array([p[0] for p in pairs]); D = np.array([p[1] for p in pairs]) - np.array([p[2] for p in pairs])
boot = []
for _ in range(10000):
    i = rng.integers(0, len(x), len(x))
    boot.append((2 * x[i] * D[i]).sum() / ((2 * x[i]) ** 2).sum())
boot = np.array(boot); lo, hi = np.percentile(boot, [2.5, 97.5])
p_q, n_q = price_wls(pairs)
print(f"   pair-bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}]   excludes 0? {'YES' if lo>0 else 'NO'}   frac>0 {np.mean(boot>0):.3f}")
P1 = (abs(p_q - 0.3910) < 0.02) and lo > 0
print(f"   => P1 {'PASS' if P1 else 'FAIL'}: reproduced {p_q:+.4f} vs prior arm's +0.3910 (|diff| {abs(p_q-0.3910):.4f})")

print("\n== in-band paired price (13 optimized), for continuity with the prior arm ==")
allp = []
for b in OPTIMIZED:
    allp += match(recs[b]["dsfb"], recs[b]["dms"], 0.05)
p_in, n_in = price_wls(allp)
print(f"   n_pairs={n_in}  price = {p_in:+.4f} ms/char per pp   (prior arm: -1.0957, n=274)")

# ---------- (6a) the LITERAL mandated sign-blind falsifier ----------
def r2(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())), b


print("\n== (6a) MANDATED SIGN-BLIND FALSIFIER on the perturbation data (nested R2) ==")
sb = {}
for name, blist in (("13 optimized (in-band)", OPTIMIZED), ("qwerty30m (off-frontier)", [OFF_FRONTIER])):
    xs = np.concatenate([recs[b]["dsfb"] for b in blist])
    ys = np.concatenate([recs[b]["dms"] for b in blist])
    fe = [np.concatenate([np.full(len(recs[b]["dsfb"]), 1.0 if b == bb else 0.0) for b in blist]) for bb in blist[1:]]
    one = np.ones(len(ys))
    R_sgn, _ = r2(np.column_stack([one, xs] + fe), ys)
    R_abs, _ = r2(np.column_stack([one, np.abs(xs)] + fe), ys)
    R_both, bb = r2(np.column_stack([one, xs, np.abs(xs)] + fe), ys)
    sb[name] = dict(n=int(len(ys)), r2_signed=float(R_sgn), r2_signblind=float(R_abs), r2_both=float(R_both),
                    beta_signed=float(bb[1]), gamma_disruption=float(bb[2]))
    win = "SIGN-BLIND" if R_abs >= R_sgn else "SIGNED"
    print(f"   {name}: n={len(ys)}")
    print(f"      R2 signed {R_sgn:.4f} | sign-blind {R_abs:.4f} | both {R_both:.4f}   => {win} wins"
          f"   (sign-blind/signed = {100*R_abs/R_sgn:.1f}%)")
    print(f"      two-term: beta(signed price) {bb[1]:+.4f}   gamma(disruption) {bb[2]:+.4f}")

json.dump(dict(evaluator_worst=[w1, w2], crosscheck_vs_prior=dict(dsfb=worst_s, dms=worst_m),
               qwerty=out, qwerty_primary=dict(price=p_q, n=n_q, ci=[float(lo), float(hi)]),
               inband=dict(price=p_in, n=n_in), P1_pass=bool(P1), signblind=sb),
          open(_env.ART + "/c06_control.json", "w"), indent=1)
print("\nwrote c06_control.json")
assert P1, "P1 POSITIVE CONTROL FAILED -- instrument broken, stop per pre-registration"
print("P1 GATE PASSED")
