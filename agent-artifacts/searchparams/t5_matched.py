"""THE DECISIVE ARM: restart power when the SEARCH OBJECTIVE MATCHES THE REPORTING GAUGE.

T4 showed the shipped default objective (bigram, 1 model) and the campaign gauge (bigram+trigram,
3-seed mean) rank layouts in PERFECTLY INVERTED order over the 6-board head-to-head. So T1/T1b's
"under-powered" answer, while true of the shipped default, answers the wrong question: more
restarts on a mis-specified objective buys nothing on the gauge.

So: build the objective that IS the gauge (up to the /covered normalization, a positive constant)
and re-run restart saturation on IT.

    gauge_total(perm) = sum_tri f * (T2[a,b] + Tc[a,b,c])   with T2, Tc the 3-SEED-MEAN tables
                        from analysis.timecard.TimeSurface -- i.e. EXACTLY analyze's ms/char.

I implement this as an IScorer built from the SHIPPED TimeSurface tables (no re-fitting, no new
model, nothing under src/ touched) and hand it to the SHIPPED SimulatedAnnealing + two_opt.
Then the question "is the search under-powered?" becomes answerable on the ruler that matters.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/tmp/searchparams/agent-artifacts/searchparams")
import numpy as np
import _harness as H
from keybo.analysis.timecard import default_surface
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt
from keybo.scoring.base import IScorer
from keybo.scoring import model_norm as MN

OUT = "/local/home/zegertho/agent/state/searchparams/artifacts/t5_matched.json"
FLOOR = 0.135
C30M = MN.S.C30M
N_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 128
REF = {"BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq", "arm B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
       "MID": "flmpg.yuo,sntcdireahkxbwv'-jzq", "HEADLINE": "flmpg-,uoysntcdireahkxvwb.'jzq"}


class GaugeScorer(IScorer):
    """ms/char itself as a search objective, from the SHIPPED TimeSurface tables.

    Vectorized: the corpus trigrams are frozen into index arrays once, so fitness is three fancy
    indexes and a dot -- fast enough for multi-restart search (the same trick TableBigramScorer
    and TableTrigramScorer use, applied to the seed-MEAN tables the gauge is built on).
    """

    def __init__(self, chars: str, target_wpm: float = 90.0):
        surf = default_surface(target_wpm)
        self._T2, self._Tc = surf._T2, surf._Tc
        self._chars = tuple(chars)
        idx = {c: i for i, c in enumerate(self._chars)}; idx[" "] = len(self._chars)
        I, J, L, F = [], [], [], []
        for ng, f in surf.tri.items():
            if len(ng) != 3: continue
            try: a, b, c = idx[ng[0]], idx[ng[1]], idx[ng[2]]
            except KeyError: continue
            I.append(a); J.append(b); L.append(c); F.append(f)
        self._I = np.array(I); self._J = np.array(J); self._L = np.array(L)
        self._F = np.array(F, dtype=np.float64)
        self._covered = self._F.sum()
        self._slot_index = {p: i for i, p in enumerate(ROW_STAGGERED_30.slots)}
        self._space = len(ROW_STAGGERED_30.slots)

    def permutation(self, layout: Layout) -> np.ndarray:
        p = np.empty(len(self._chars) + 1, dtype=np.intp)
        for i, c in enumerate(self._chars):
            p[i] = self._slot_index[layout.pos(c)]
        p[len(self._chars)] = self._space
        return p

    def fitness_of_permutation(self, p: np.ndarray) -> float:
        a, b, c = p[self._I], p[self._J], p[self._L]
        return float(self._F @ (self._T2[a, b] + self._Tc[a, b, c]))

    def fitness(self, layout: Layout) -> float:
        return self.fitness_of_permutation(self.permutation(layout))

    def ms_per_char(self, layout: Layout) -> float:
        return self.fitness(layout) / self._covered


gs = GaugeScorer(C30M)
# --- PARITY GATE: this scorer must reproduce analyze's ms/char, else it is a different objective ---
parity = []
for lab, l in REF.items():
    mine = gs.ms_per_char(Layout(l, ROW_STAGGERED_30)); ref = H.ms_per_char(l)
    parity.append({"label": lab, "mine": mine, "analyze": ref, "abs_dev": abs(mine - ref)})
maxdev = max(p["abs_dev"] for p in parity)
print("PARITY max abs dev vs analyze ms/char: %.3e" % maxdev, flush=True)
assert maxdev < 1e-9, "GaugeScorer is NOT the gauge -- refusing to report"

recs, t0 = [], time.perf_counter()
for seed in range(N_POOL):
    lay = Layout(C30M, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=seed, alpha=0.999, progress=False)
    t = time.perf_counter()
    best = sa.optimize(lay, gs)
    best = two_opt(best, gs)
    recs.append({"seed": seed, "layout": "".join(best.chars),
                 "ms_per_char": gs.ms_per_char(best), "sec": time.perf_counter() - t})
    if (seed + 1) % 16 == 0:
        print("  %3d/%d %.0fs" % (seed + 1, N_POOL, time.perf_counter() - t0), flush=True)

mpc = np.array([r["ms_per_char"] for r in recs]); n = len(recs)
rng = np.random.default_rng(555)
ref_g = {k: H.ms_per_char(v) for k, v in REF.items()}
ladder = [1, 2, 4, 8, 16, 32, 64, 128]
curve = {}
for N in ladder:
    if N > n: continue
    idx = np.arange(n)[None, :] if N == n else np.array([rng.choice(n, N, replace=False) for _ in range(4000)])
    got = mpc[idx].min(axis=1)   # objective IS the gauge now -> selection is not an oracle
    curve[N] = {"mean": float(got.mean()), "median": float(np.median(got)),
                "p10": float(np.percentile(got, 10)), "p90": float(np.percentile(got, 90)),
                "min": float(got.min()),
                "P_beats_BALL1": float((got < ref_g["BALL-1"]).mean()),
                "P_beats_armB": float((got < ref_g["arm B"]).mean()),
                "P_beats_armB_by_floor": float((got < ref_g["arm B"] - FLOOR).mean())}
doub = []
for i, N in enumerate(ladder[:-1]):
    if ladder[i+1] not in curve: break
    dd = curve[N]["mean"] - curve[ladder[i+1]]["mean"]
    doub.append({"N": N, "2N": ladder[i+1], "delta": float(dd), "below_floor": bool(dd < FLOOR)})
res = {"design": "SEARCH OBJECTIVE == REPORTING GAUGE (3-seed-mean T2+Tc over the trigram corpus), "
                 "shipped SimulatedAnnealing(alpha=0.999)+two_opt, start=C30M, %d restarts" % n,
       "parity_gate": {"max_abs_dev_vs_analyze_ms_per_char": maxdev, "rows": parity,
                       "verdict": "PASS -- this objective IS analyze's ms/char"},
       "n_pool": n, "wall_sec": time.perf_counter() - t0, "floor": FLOOR,
       "sec_per_attempt": float(np.mean([r["sec"] for r in recs])),
       "reference_boards_ms_per_char": ref_g,
       "single_attempt_spread": {"mean": float(mpc.mean()), "sd": float(mpc.std(ddof=1)),
            "min": float(mpc.min()), "max": float(mpc.max()), "range": float(mpc.max()-mpc.min())},
       "curve": curve, "doubling_deltas": doub,
       "saturation_N_star": next((r["N"] for r in doub if r["below_floor"]), None),
       "best_found": {"layout": recs[int(mpc.argmin())]["layout"], "ms_per_char": float(mpc.min())},
       "n_distinct_layouts": len({r["layout"] for r in recs})}
res["vs_campaign"] = {k: {"ref": v, "our_best_minus_ref": float(mpc.min() - v),
        "in_floor_units": float((mpc.min() - v) / FLOOR),
        "n_single_attempts_beating_it": int((mpc < v).sum()),
        "beaten_by_more_than_floor": bool(mpc.min() < v - FLOOR)} for k, v in ref_g.items()}
json.dump({"meta": res, "runs": recs}, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
