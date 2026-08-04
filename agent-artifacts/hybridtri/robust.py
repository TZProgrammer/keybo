"""HYBRIDB-1 INVARIANT 6 — bootstrap the exploitability verdict's STABILITY, beside the verdict.

Registered in the prereg §5. EXPLOIT-1's G-channel verdict turned out to be a coin flip (EXPLOITABLE
in only 38-49% of resamples) while its B channel was 100% -- and it reported that beside the
registered verdict rather than substituting it. Same discipline here.

Goalpost discipline: THE REGISTERED VERDICT STANDS AS RECORDED regardless of what this says. This
measures how much of the verdict is the particular best-of-12 draw I happened to get.

Also here: the SIGN robustness with no selection at all (all 24 seeds' mean), which is the weakest
and most assumption-free reading of the same data.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

with open(f"{ARTIFACTS}/exploit.json") as fh:
    ex = json.load(fh)

RESAMPLES = 4000
BUDGETS = (3, 6, 12)
rng = np.random.default_rng(20260804)
out = {
    "prereg": "agent-artifacts/hybridtri/HYBRIDTRI-preregistration.md @ 5a5d3c3 §5 (INVARIANT 6)",
    "n_resamples": RESAMPLES,
    "budgets": list(BUDGETS),
    "channels": {},
}

for ch in ("G", "B"):
    hyb = ex["arms"][f"{ch}-HYBRIDB"]
    srv = ex["arms"][f"{ch}-SERVED"]
    own_h = np.array(hyb["own_ms_per_char"])
    tr_h = np.array(hyb["trusted_ms_per_char"])
    own_s = np.array(srv["own_ms_per_char"])
    tr_s = np.array(srv["trusted_ms_per_char"])
    floor = ex["verdict"][ch]["floor"]["p95"]
    n_seeds = len(own_h)

    block = {"registered_gap": ex["verdict"][ch]["gap_ms_per_char"], "floor_p95": floor}
    for n in BUDGETS:
        gaps = np.empty(RESAMPLES)
        for t in range(RESAMPLES):
            # resample WHICH SEEDS fill each arm's block, exactly as EXPLOIT-1's R1 did -- the
            # question is "how much of the verdict is my particular draw", not "is the search noisy"
            ih = rng.choice(n_seeds, size=n, replace=True)
            isv = rng.choice(n_seeds, size=n, replace=True)
            kh = ih[np.argmin(own_h[ih])]  # best-of-n on the arm's OWN objective
            ks = isv[np.argmin(own_s[isv])]
            gaps[t] = tr_h[kh] - tr_s[ks]
        block[f"n{n}"] = {
            "share_EXPLOITABLE": float((gaps > floor).mean()),
            "share_gap_positive": float((gaps > 0).mean()),
            "median_gap": float(np.median(gaps)),
            "median_gap_exceeds_floor": bool(np.median(gaps) > floor),
            "p05_gap": float(np.percentile(gaps, 5)),
            "p95_gap": float(np.percentile(gaps, 95)),
        }
    # the assumption-free reading: no best-of selection at all
    block["no_selection"] = {
        "mean_trusted_hybridb": float(tr_h.mean()),
        "mean_trusted_served": float(tr_s.mean()),
        "mean_gap": float(tr_h.mean() - tr_s.mean()),
        "sign_positive": bool(tr_h.mean() > tr_s.mean()),
    }
    out["channels"][ch] = block

print("=" * 96)
print("BOOTSTRAP STABILITY of the exploitability verdict (INVARIANT 6) -- beside, not instead of")
print("=" * 96)
for ch, b in out["channels"].items():
    print(
        f"\nchannel {ch}: registered gap {b['registered_gap']:+.6f}  floor p95 {b['floor_p95']:.6f}"
    )
    for n in BUDGETS:
        r = b[f"n{n}"]
        print(
            f"  n={n:<3} EXPLOITABLE in {r['share_EXPLOITABLE']:>6.1%} of resamples   "
            f"gap>0 in {r['share_gap_positive']:>6.1%}   median gap {r['median_gap']:+.6f} "
            f"({'above' if r['median_gap_exceeds_floor'] else 'BELOW'} its own floor)   "
            f"[p05 {r['p05_gap']:+.4f}, p95 {r['p95_gap']:+.4f}]"
        )
    ns = b["no_selection"]
    print(
        f"  no selection at all (mean of all 24 seeds): hybrid-B {ns['mean_trusted_hybridb']:.6f} "
        f"vs served {ns['mean_trusted_served']:.6f}  => gap {ns['mean_gap']:+.6f}  "
        f"sign positive: {ns['sign_positive']}"
    )

# EXPLOIT-1's published R1, for the comparison
out["EXPLOIT1_R1_published"] = {
    "G_share_exploitable": [0.382, 0.359, 0.486],
    "B_share_exploitable": [1.0, 1.0, 1.0],
    "note": "EXPLOIT-1's G verdict was a coin flip at every budget; its B was 100%",
}
print()
print("EXPLOIT-1's own R1 for comparison: G exploitable in 38.2/35.9/48.6% of resamples at")
print("n=3/6/12 (a coin flip, and its MEDIAN resample gap sat BELOW its floor); B in 100.0%.")

with open(f"{ARTIFACTS}/robust.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(f"\nwrote {ARTIFACTS}/robust.json")
