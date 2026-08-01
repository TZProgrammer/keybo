"""C12 -- a CI for the WARM (conservative) frontier price.

c09's warm pass is a single deterministic run over the whole 154-board donor pool, so it has no
sampling distribution -- but F1 requires a CI. Two independent resamplings, both pre-committed here
before looking at either result:

 (A) DONOR BOOTSTRAP (primary): resample the donor pool with replacement (B draws) and recompute the
     whole warm frontier from each resample. This is the right unit: the frontier's value at a cap is
     `min` over donors, so donor-set variation IS the estimator's variability. Cheap because c09
     already recorded, per cap, the re-descended value reached from EVERY donor.
 (B) REPLICATE SPLIT: rebuild the warm frontier using only the donors contributed by replicate r of
     c07 (r = 0..3, disjoint seed blocks) + the 13 field boards. 4 independent warm frontiers =>
     percentile CI across them, exactly the design-matched ruler used for the cold estimate."""
import json
import os
import time

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

fs, w1, w2 = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
print(f"evaluators {w1:.2e}/{w2:.2e}", flush=True)

A = json.load(open(_env.ART + "/c07_analysis.json"))
FR = json.load(open(_env.ART + "/c07_frontier.json"))
runs = FR["runs"]
PRICED, INERT = A["priced"], A["inert"]
HEAD = tuple(A["headline"]["interval"])
CAPS = PRICED + INERT
B = int(os.environ.get("PB_B", "2000"))

# ---------- (B) REPLICATE-SPLIT warm frontiers (4 independent) ----------
# donors, tagged by which c07 replicate produced them
donors_by_rep = {r: [] for r in range(FR["R"])}
for k, v in runs.items():
    r = v["replicate"]
    for q in (v.get("top_perms") or []):
        donors_by_rep[r].append(np.array(q, dtype=np.intp))
    if v.get("perm_at_best"):
        donors_by_rep[r].append(np.array(v["perm_at_best"], dtype=np.intp))
field = [fs.perm(FIELD[b]) for b in OPTIMIZED]

def warm_frontier(donors, caps):
    """(cap -> best ms) after repairing each donor into feasibility and re-descending under the cap."""
    out = {}
    for c in caps:
        best = np.inf
        for q in donors:
            p, feas = S.drive_under_cap(obj, q.copy(), c)
            if not feas:
                continue
            p, m = S.cap_three_opt(obj, p, c)
            if obj.sfb(p[:30]) <= c + 1e-9 and m < best:
                best = float(m)
        out[c] = best
    return out

CAPS_CI = [HEAD[0], 1.5, HEAD[1], 3.0, 8.0]      # enough for the headline price + the placebo
print(f"\n== (B) REPLICATE-SPLIT warm frontiers (R={FR['R']}, caps {CAPS_CI}) ==", flush=True)
rep_F = {}
for r in range(FR["R"]):
    t0 = time.perf_counter()
    dn = donors_by_rep[r] + field
    seen = set(); uq = []
    for q in dn:
        t = tuple(q.tolist())
        if t not in seen:
            seen.add(t); uq.append(q)
    F = warm_frontier(uq, CAPS_CI)
    rep_F[r] = F
    pr = -(F[HEAD[1]] - F[HEAD[0]]) / (HEAD[1] - HEAD[0])
    pl = -(F[8.0] - F[3.0]) / (8.0 - 3.0)
    print(f"   r{r}: {len(uq)} donors  F({HEAD[0]})={F[HEAD[0]]:.4f}  F({HEAD[1]})={F[HEAD[1]]:.4f}"
          f"  price {pr:+.4f}  placebo {pl:+.6f}   {time.perf_counter()-t0:.0f}s", flush=True)
    json.dump({str(k): {str(c): v for c, v in vv.items()} for k, vv in rep_F.items()},
              open(_env.ART + "/c12_repsplit.json", "w"), indent=1)

prices = np.array([-(rep_F[r][HEAD[1]] - rep_F[r][HEAD[0]]) / (HEAD[1] - HEAD[0]) for r in rep_F])
plac = np.array([-(rep_F[r][8.0] - rep_F[r][3.0]) / (8.0 - 3.0) for r in rep_F])
lo, hi = np.percentile(prices, [2.5, 97.5])
print(f"\n   REPLICATE-SPLIT price over cap [{HEAD[0]},{HEAD[1]}]: mean {prices.mean():+.4f}"
      f"  sd {prices.std(ddof=1):.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  excl 0? {'YES' if lo>0 else 'NO'}")
print(f"   REPLICATE-SPLIT placebo (inert 3->8): mean {plac.mean():+.6f}  max|.| {np.abs(plac).max():.6f}")

# ---------- (A) DONOR BOOTSTRAP over c09's recorded per-donor values ----------
W = json.load(open(_env.ART + "/c07_warm.json"))
print(f"\n== (A) DONOR BOOTSTRAP (B={B}) over c09's per-donor re-descended values ==", flush=True)
# c09 stored, per cap, the list of values reached from each feasible donor (order = donor order)
vals = {float(v["cap"]): np.array(v["vals"], float) for v in W.values() if v.get("vals")}
have = [c for c in (HEAD[0], HEAD[1], 3.0, 8.0) if c in vals and len(vals[c])]
rng = np.random.default_rng(4242)
bs = []
for _ in range(B):
    F = {}
    for c in have:
        v = vals[c]
        F[c] = v[rng.integers(0, len(v), len(v))].min()
    bs.append(-(F[HEAD[1]] - F[HEAD[0]]) / (HEAD[1] - HEAD[0]))
bs = np.array(bs)
b_lo, b_hi = np.percentile(bs, [2.5, 97.5])
print(f"   NOTE: this resamples WITHIN each cap independently, so it measures how much the frontier")
print(f"   depends on WHICH donors were available -- it does NOT re-run the search.")
print(f"   price mean {bs.mean():+.4f}  CI95 [{b_lo:+.4f}, {b_hi:+.4f}]  frac>0 {np.mean(bs>0):.4f}")

json.dump(dict(headline_interval=list(HEAD),
               repsplit=dict(prices=prices.tolist(), mean=float(prices.mean()),
                             sd=float(prices.std(ddof=1)), ci=[float(lo), float(hi)],
                             placebo=plac.tolist()),
               donor_bootstrap=dict(B=B, mean=float(bs.mean()), ci=[float(b_lo), float(b_hi)],
                                    frac_pos=float(np.mean(bs > 0))),
               rep_frontiers={str(k): {str(c): v for c, v in vv.items()} for k, vv in rep_F.items()}),
          open(_env.ART + "/c12_warmci.json", "w"), indent=1)
print("\nwrote c12_warmci.json")
