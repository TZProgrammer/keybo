"""TASK 1b: the arm that ACTUALLY bears on the campaign -- restart power on the C30M charset.

The T1 pool ran on qwerty's charset (the CLI default --start). But every campaign board
(BALL-1, arm B, keybo-lsb, the F5M family) lives on C30M = "qwertyuiop asdfghjkl' zxcvbnm,.-"
(model_norm.S.C30M) -- a DIFFERENT charset (';/' <-> "'-"). A restart-saturation number measured
on qwerty's charset does not automatically transfer, so this arm re-runs it where the campaign's
comparisons actually live, and asks the question that matters:

    can a bigger restart budget BEAT BALL-1 / arm B on the campaign's own gauge?

Note the objective here is the shipped SPEED objective (bigram table, wpm 90). BALL-1/arm-B were
produced by CONSTRAINED searches on other axes, so beating them on speed alone is not a claim that
they are bad boards -- it is a claim about whether "X is a local optimum / X beats Y by 0.1-0.7"
survives a properly powered speed search.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import numpy as np
import _harness as H
from keybo.scoring import model_norm as MN

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t1b_c30m.json"
FLOOR = 0.135
C30M = MN.S.C30M
N_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 256

REF = {  # the campaign's own boards, from PREREGISTRATIONS.md:9423 (ARMH-1 table)
    "BALL-1":   "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "MID":      "flmpg.yuo,sntcdireahkxbwv'-jzq",
    "HEADLINE": "flmpg-,uoysntcdireahkxvwb.'jzq",
    "arm B":    "flmpg-yuo,sntdcireahkxbwv'.jzq",
}
sc = H.build_search_scorer(start=C30M)          # table charset = C30M
recs, t0 = [], time.perf_counter()
for seed in range(N_POOL):
    r = H.one_attempt(sc, seed=seed, start=C30M)
    r["ms_per_char"] = H.ms_per_char(r["layout"])
    recs.append(r)
    if (seed + 1) % 32 == 0:
        print("  %3d/%d %.0fs" % (seed + 1, N_POOL, time.perf_counter() - t0), flush=True)

fit = np.array([r["fitness"] for r in recs]); mpc = np.array([r["ms_per_char"] for r in recs])
n = len(recs); rng = np.random.default_rng(99)
ref = {k: {"ms_per_char": H.ms_per_char(v), "fitness": sc.fitness(
        __import__("keybo.layout", fromlist=["Layout"]).Layout(v, __import__("keybo.geometry", fromlist=["ROW_STAGGERED_30"]).ROW_STAGGERED_30))}
       for k, v in REF.items()}

def sel_curve(ladder, draws=4000):
    out = {}
    for N in ladder:
        if N > n: continue
        idx = np.arange(n)[None, :] if N == n else np.array(
            [rng.choice(n, size=N, replace=False) for _ in range(draws)])
        pick = idx[np.arange(idx.shape[0]), fit[idx].argmin(axis=1)]
        got = mpc[pick]
        out[N] = {"mean": float(got.mean()), "median": float(np.median(got)),
                  "p10": float(np.percentile(got, 10)), "p90": float(np.percentile(got, 90)),
                  "min": float(got.min()), "max": float(got.max()),
                  "P_beats_BALL1": float((got < ref["BALL-1"]["ms_per_char"]).mean()),
                  "P_beats_armB": float((got < ref["arm B"]["ms_per_char"]).mean()),
                  "P_beats_BALL1_by_floor": float((got < ref["BALL-1"]["ms_per_char"] - FLOOR).mean())}
    return out

ladder = [1, 2, 4, 8, 16, 32, 64, 128, 256]
curve = sel_curve(ladder)
doub = []
for i, N in enumerate(ladder[:-1]):
    if ladder[i+1] not in curve: break
    d = curve[N]["mean"] - curve[ladder[i+1]]["mean"]
    doub.append({"N": N, "2N": ladder[i+1], "delta_ms_per_char": float(d), "below_floor": bool(d < FLOOR)})

best_i = int(fit.argmin())
res = {
  "charset": C30M, "n_pool": n, "floor": FLOOR, "wall_sec": time.perf_counter() - t0,
  "objective": "shipped bigram table speed objective (bigram_reg31_seed0, wpm 90), start=C30M",
  "gauge": "K31 trigram 3-seed-mean ms/char (reproduces the ledger's ARMH-1 numbers to 6 dp)",
  "reference_boards": ref,
  "single_attempt_spread": {"mean": float(mpc.mean()), "sd": float(mpc.std(ddof=1)),
      "min": float(mpc.min()), "max": float(mpc.max()), "range": float(mpc.max()-mpc.min())},
  "curve_selection_consistent": curve, "doubling_deltas": doub,
  "saturation_N_star": next((r["N"] for r in doub if r["below_floor"]), None),
  "best_found": {"layout": recs[best_i]["layout"], "fitness": float(fit[best_i]),
                 "ms_per_char": float(mpc[best_i])},
  "best_on_gauge": {"layout": recs[int(mpc.argmin())]["layout"], "ms_per_char": float(mpc.min())},
  "n_distinct_layouts": len({r["layout"] for r in recs}),
}
res["vs_campaign"] = {
  k: {"ref_ms_per_char": v["ms_per_char"],
      "our_best_on_objective_minus_ref": float(mpc[best_i] - v["ms_per_char"]),
      "our_best_on_gauge_minus_ref": float(mpc.min() - v["ms_per_char"]),
      "n_of_256_single_attempts_beating_it": int((mpc < v["ms_per_char"]).sum()),
      "beaten_by_more_than_floor_on_gauge": bool(mpc.min() < v["ms_per_char"] - FLOOR)}
  for k, v in ref.items()}
json.dump({"meta": res, "runs": recs}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
