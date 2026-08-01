"""C08 -- evaluate the frontier against the SIX PRE-REGISTERED CRITERIA and emit the verdict.

Reads c07_frontier.json (R replicates x caps). Computes:
  F_own(c)  = best over that cap's OWN restarts (effort-symmetric => monotonicity is a real falsifier)
  F_pool(c) = best over every board found at ANY cap <= c (monotone by construction; better est of F)
  price over an interval = -(F(c_hi)-F(c_lo))/(c_hi-c_lo)
and the six gates F1..F6."""
import json
import os
import sys

import _env
import numpy as np

TAG = os.environ.get("PB_TAG", "c07")
d = json.load(open(_env.ART + f"/{TAG}_frontier.json"))
runs = d["runs"] if "runs" in d else d
PRICED = d.get("priced", [0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
INERT = d.get("inert", [3.0, 3.5, 5.0, 8.0, 1e9])
CAPS = PRICED + INERT
R = d.get("R", 1 + max(v["replicate"] for v in runs.values()))

def get(r, c):
    return runs.get(f"r{r}_c{c}")

# ---- per-replicate F_own and F_pool ----
own = np.full((R, len(CAPS)), np.nan)
pool = np.full((R, len(CAPS)), np.nan)
nfeas = np.zeros((R, len(CAPS)), int)
sds = np.full((R, len(CAPS)), np.nan)
sfb_at = np.full((R, len(CAPS)), np.nan)
for r in range(R):
    # collect every (sfb, ms) board this replicate ever found, for pooling
    found = []
    for j, c in enumerate(CAPS):
        v = get(r, c)
        if v is None:
            continue
        cand = list(v.get("polished_vals") or []) + list(v.get("vals") or [])
        if cand:
            own[r, j] = min(cand)
        nfeas[r, j] = v.get("n_feasible", 0)
        if v.get("vals") and len(v["vals"]) > 1:
            sds[r, j] = float(np.std(v["vals"], ddof=1))
        if v.get("sfb_at_best") is not None:
            sfb_at[r, j] = v["sfb_at_best"]
        # every board found under cap c also satisfies every LOOSER cap
        if v.get("best_polished") is not None:
            found.append((v.get("sfb_at_best", c), v["best_polished"]))
    for j, c in enumerate(CAPS):
        elig = [m for s, m in found if s is not None and s <= c + 1e-9]
        # a board found at a TIGHTER cap is feasible here too; also its own cap's raw values
        if elig:
            pool[r, j] = min(elig)
        if not np.isnan(own[r, j]):
            pool[r, j] = min(pool[r, j], own[r, j]) if not np.isnan(pool[r, j]) else own[r, j]
    # enforce monotone non-increasing in c for the pooled estimator
    for j in range(len(CAPS) - 2, -1, -1):
        pass  # pooled is already monotone by the eligibility rule above

def slope(F, lo, hi):
    """price over [lo,hi] = -(F(hi)-F(lo))/(hi-lo)  ms/char per pp."""
    i, j = CAPS.index(lo), CAPS.index(hi)
    return -(F[j] - F[i]) / (hi - lo)

print(f"== FRONTIER  (N={d.get('N')} restarts/cap, R={R} replicates, 3-opt top-{d.get('TOP3')}) ==")
print(f"{'cap':>8} | " + " ".join(f"{'r'+str(r):>10}" for r in range(R)) + f" | {'mean':>9}{'sd':>8}{'best':>10}{'feas':>7}")
for j, c in enumerate(CAPS):
    lab = f"{c:.2f}" if c < 1e8 else "inf"
    row = " ".join(f"{own[r,j]:>10.4f}" if not np.isnan(own[r, j]) else f"{'--':>10}" for r in range(R))
    v = own[:, j][~np.isnan(own[:, j])]
    tag = "" if c in PRICED else "  (inert)"
    print(f"{lab:>8} | {row} | {v.mean():>9.4f}{v.std(ddof=1) if len(v)>1 else float('nan'):>8.4f}"
          f"{v.min():>10.4f}{nfeas[:,j].mean():>7.1f}{tag}")

Fown_mean = np.nanmean(own, axis=0)
Fpool_mean = np.nanmean(pool, axis=0)
print(f"\n{'cap':>8}{'F_own(mean)':>13}{'F_pool(mean)':>14}{'sfb@best(mean)':>16}")
for j, c in enumerate(CAPS):
    lab = f"{c:.2f}" if c < 1e8 else "inf"
    print(f"{lab:>8}{Fown_mean[j]:>13.4f}{Fpool_mean[j]:>14.4f}{np.nanmean(sfb_at[:,j]):>16.4f}")

# ---------------- the priced interval + per-interval prices ----------------
print("\n== per-interval SHADOW PRICE (ms/char per pp), F_pool, per replicate ==")
ivs = list(zip(PRICED[:-1], PRICED[1:]))
print(f"{'interval':>14}{'mean price':>12}{'sd':>9}{'CI95 (replicate percentile)':>32}{'excl 0':>8}")
iv_out = {}
for lo, hi in ivs:
    ps = np.array([slope(pool[r], lo, hi) for r in range(R)])
    ps = ps[np.isfinite(ps)]
    lo_ci, hi_ci = (np.percentile(ps, [2.5, 97.5]) if len(ps) > 1 else (np.nan, np.nan))
    iv_out[f"{lo}-{hi}"] = dict(mean=float(ps.mean()), sd=float(ps.std(ddof=1)) if len(ps) > 1 else None,
                                ci=[float(lo_ci), float(hi_ci)], vals=ps.tolist())
    print(f"{f'[{lo},{hi}]':>14}{ps.mean():>12.4f}{(ps.std(ddof=1) if len(ps)>1 else float('nan')):>9.4f}"
          f"{f'[{lo_ci:+.4f}, {hi_ci:+.4f}]':>32}{'YES' if lo_ci>0 else 'no':>8}")

# headline: the interval spanning the FIELD's sfb range (1.0666 .. 2.5391) -> use [1.0, 2.5]
HEAD = (1.0, 2.5)
head = np.array([slope(pool[r], *HEAD) for r in range(R)])
head = head[np.isfinite(head)]
h_lo, h_hi = np.percentile(head, [2.5, 97.5]) if len(head) > 1 else (np.nan, np.nan)
print(f"\nHEADLINE in-band price over cap [{HEAD[0]}, {HEAD[1]}] (spans the field's own sfb range):")
print(f"   mean {head.mean():+.4f}  sd {head.std(ddof=1) if len(head)>1 else float('nan'):.4f}"
      f"  replicate CI95 [{h_lo:+.4f}, {h_hi:+.4f}]  per-replicate {np.round(head,4).tolist()}")

# ---------------- the SIX GATES ----------------
print("\n" + "=" * 78)
print("SIX PRE-REGISTERED CRITERIA")
print("=" * 78)
G = {}

# F1 sign + CI excludes zero
G["F1_sign_CI"] = dict(price=float(head.mean()), ci=[float(h_lo), float(h_hi)],
                       passed=bool(head.mean() > 0 and h_lo > 0))
print(f"F1 SIGN+CI: price {head.mean():+.4f}, CI [{h_lo:+.4f},{h_hi:+.4f}]  => "
      f"{'PASS' if G['F1_sign_CI']['passed'] else 'FAIL'}")

# F2 rise > 3x replicate sd of F at those caps
i_lo, i_hi = CAPS.index(HEAD[0]), CAPS.index(HEAD[1])
rise = float(np.nanmean(pool[:, i_lo]) - np.nanmean(pool[:, i_hi]))
sd_F = float(np.nanmean([np.nanstd(pool[:, i_lo], ddof=1), np.nanstd(pool[:, i_hi], ddof=1)]))
G["F2_exceeds_noise"] = dict(rise=rise, sd_F=sd_F, ratio=rise / sd_F if sd_F else None,
                             passed=bool(sd_F > 0 and rise > 3 * sd_F))
print(f"F2 RISE vs NOISE: rise {rise:+.4f} ms/char over the interval; replicate sd of F {sd_F:.4f}"
      f"  ratio {rise/sd_F if sd_F else float('nan'):.2f}x (need >3x) => {'PASS' if G['F2_exceeds_noise']['passed'] else 'FAIL'}")

# F3 PLACEBO: inert caps must be FLAT
in_caps = [c for c in INERT if c < 1e8]
pl = []
for r in range(R):
    a, b = CAPS.index(in_caps[0]), CAPS.index(in_caps[-1])
    s = -(pool[r, b] - pool[r, a]) / (in_caps[-1] - in_caps[0])
    pl.append(s)
pl = np.array([p for p in pl if np.isfinite(p)])
p_lo, p_hi = np.percentile(pl, [2.5, 97.5]) if len(pl) > 1 else (np.nan, np.nan)
flat = bool(abs(pl.mean()) < abs(head.mean()) / 3 and p_lo <= 0 <= p_hi)
G["F3_placebo"] = dict(inert_interval=[in_caps[0], in_caps[-1]], slope=float(pl.mean()),
                       ci=[float(p_lo), float(p_hi)], inband=float(head.mean()), passed=flat)
print(f"F3 PLACEBO (inert caps [{in_caps[0]},{in_caps[-1]}], constraint cannot bind):")
print(f"   slope {pl.mean():+.4f}  CI [{p_lo:+.4f},{p_hi:+.4f}]   |slope| vs in-band/3 = "
      f"{abs(pl.mean()):.4f} vs {abs(head.mean())/3:.4f}  => {'PASS (flat)' if flat else 'FAIL'}")

# F4 monotonicity of F_own over the priced range
viol = []
for r in range(R):
    for j in range(CAPS.index(PRICED[0]), CAPS.index(PRICED[-1])):
        a, b = own[r, j], own[r, j + 1]
        if np.isfinite(a) and np.isfinite(b) and b > a:      # looser cap gave WORSE value
            viol.append((r, CAPS[j], CAPS[j + 1], float(b - a)))
sd_cell = float(np.nanmean(np.nanstd(own, axis=0, ddof=1)))
worst_v = max([v[3] for v in viol], default=0.0)
G["F4_monotone"] = dict(n_violations=len(viol), worst=worst_v, sd_cell=sd_cell,
                        passed=bool(worst_v <= 2 * sd_cell), violations=viol[:12])
print(f"F4 MONOTONICITY of F_own: {len(viol)} violations, worst {worst_v:+.4f}; per-cap replicate sd {sd_cell:.4f}"
      f"  (need worst <= 2sd = {2*sd_cell:.4f}) => {'PASS' if G['F4_monotone']['passed'] else 'FAIL'}")

# F6 best-of-N saturation: N vs N/2 (recompute own-cap best from first half of the restarts)
half = {}
for j, c in enumerate(CAPS):
    a, b = [], []
    for r in range(R):
        v = get(r, c)
        if not v or not v.get("vals"):
            continue
        vv = v["vals"]
        a.append(min(vv))
        b.append(min(vv[: max(1, len(vv) // 2)]))
    if a:
        half[c] = (float(np.mean(a)), float(np.mean(b)))
worst_gap = max((half[c][1] - half[c][0]) for c in PRICED if c in half)
G["F6_saturation"] = dict(worst_gap=float(worst_gap), rise=rise, third=float(rise / 3),
                          passed=bool(worst_gap < rise / 3),
                          per_cap={str(c): half[c] for c in half})
print(f"F6 SATURATION best-of-N vs N/2 (raw, priced caps): worst gap {worst_gap:+.4f}"
      f"  (need < rise/3 = {rise/3:.4f}) => {'PASS' if G['F6_saturation']['passed'] else 'FAIL'}")
print("   (F5 warm-start stability is a separate run -- c09)")

json.dump(dict(caps=CAPS, priced=PRICED, inert=INERT, R=R, N=d.get("N"),
               F_own=own.tolist(), F_pool=pool.tolist(), n_feasible=nfeas.tolist(),
               sfb_at_best=sfb_at.tolist(), intervals=iv_out,
               headline=dict(interval=list(HEAD), mean=float(head.mean()),
                             ci=[float(h_lo), float(h_hi)], vals=head.tolist()),
               gates=G),
          open(_env.ART + f"/{TAG}_analysis.json", "w"), indent=1)
print(f"\nwrote {TAG}_analysis.json")
