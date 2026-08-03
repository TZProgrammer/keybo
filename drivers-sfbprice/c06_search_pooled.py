"""INVARIANT B, PROPERLY POWERED — pooled best-of-K restarts, R replicates per arm.

c04's 12-seed single-restart design is UNDERPOWERED and I measured why: per-seed ms/char sd is
0.8702 for the OLD arm while the surcharge's whole effect on ms/char is 0.2527, i.e. the
SEARCH-SEED NOISE IS 3.4x THE SIGNAL and the spread is 8.1x. Neither arm reached the field optimum
(OLD argmin 254.1639 vs field best 253.9006) and each arm returned 7-8 DISTINCT boards from 12
seeds -- an unconverged search. Reading a +0.3168 pp sfb difference off that is exactly the
"significance without resolution" error TOURNAMENT-1 documented, in a search-noise disguise.

(My 0.8702 also independently REPRODUCES the campaign's registered search-seed floor of 0.883 from
a different code path -- reported as a positive control.)

The fix: each replicate is a POOLED BEST-OF-K (K restarts, keep the argmin), which is what the
campaign's own searches did (28-128 restarts). R replicates then give a measurable spread of the
converged argmin, and the arms are compared replicate-paired on SHARED seed blocks.
"""
import json
import sys
import time

import numpy as np
from _guard import ART, FIELD_ORDER, assert_d5, build_boards

t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

log("D5:"); assert_d5()

import surface  # noqa: E402
from scipy import stats  # noqa: E402
from keybo.analysis.kmstats import KmStats  # noqa: E402
from keybo.analysis.surfaces import C30M  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.optimize.annealing import SimulatedAnnealing  # noqa: E402
from keybo.optimize.local_search import three_opt, two_opt  # noqa: E402
from keybo.scoring.table_trigram import TableTrigramScorer  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8      # restarts pooled per replicate
R = int(sys.argv[2]) if len(sys.argv) > 2 else 6      # replicates per arm

BOARDS = build_boards()
C03 = json.load(open(f"{ART}/c03_correct.json"))
DELTA = C03["delta"]
MASK = surface.same_finger_mask()
CD = production_corpus_dir(None)
tri_freq = {k: v for k, v in load_frequencies(str(CD / "trigrams.txt")).items() if len(k) == 3}
surf = TimeSurface(tri_freq, target_wpm=surface.WPM)
T2m_old, Tcm = surf._T2, surf._Tc
T2m_new = surface.corrected_T2(T2m_old, DELTA, MASK, "add")


def make_scorer(T2m):
    sc = TableTrigramScorer.from_table(T2m[:, :, None] + Tcm, surf.tri, chars=C30M, geometry=G30)
    sc._covered = float(sc._f.sum())
    return sc


SC = {"OLD": make_scorer(T2m_old), "NEW": make_scorer(T2m_new)}
bg = load_frequencies(str(CD / "bigrams.txt"))
sk = load_frequencies(str(CD / "1-skip31.txt"))
KM = KmStats(bg, sk, tri_freq)


def sfb_of(lay): return float(KM.stats(lay)["sfb"])
def ms_of(arm, lay): return SC[arm].fitness(Layout(lay, G30)) / SC[arm]._covered


def one(sc, seed):
    sa = SimulatedAnnealing(seed=seed, alpha=0.999, max_outer=None, progress=False)
    return "".join(three_opt(two_opt(sa.optimize(Layout(C30M, G30), sc), sc), sc).chars)


log(f"pooled design: K={K} restarts per replicate, R={R} replicates per arm, shared seed blocks")
log(f"  => {K * R} searches per arm, {2 * K * R} total")
RES = {"OLD": [], "NEW": []}
for rep in range(R):
    seeds = list(range(100 + rep * K, 100 + (rep + 1) * K))     # SHARED between arms
    for arm in ("OLD", "NEW"):
        ts = time.time()
        cands = [one(SC[arm], s) for s in seeds]
        best = min(cands, key=lambda L: ms_of(arm, L))
        rec = {"rep": rep, "seeds": seeds, "layout": best, "sfb": sfb_of(best),
               "mspc_own": ms_of(arm, best), "mspc_old_surface": ms_of("OLD", best),
               "mspc_new_surface": ms_of("NEW", best),
               "pool_sfb": [sfb_of(c) for c in cands],
               "pool_ms_own": [ms_of(arm, c) for c in cands],
               "n_distinct_in_pool": len(set(cands)), "wall_s": time.time() - ts}
        RES[arm].append(rec)
        log(f"  rep{rep} {arm}  pooled-best sfb {rec['sfb']:.4f}  ms/char(own) "
            f"{rec['mspc_own']:.4f}  ms(OLD surf) {rec['mspc_old_surface']:.4f}  "
            f"{rec['n_distinct_in_pool']}/{K} distinct  [{rec['wall_s']:.0f}s]  {best}")

S = {}
for arm in ("OLD", "NEW"):
    sfbs = np.array([r["sfb"] for r in RES[arm]])
    own = np.array([r["mspc_own"] for r in RES[arm]])
    require_finite(sfbs.tolist(), f"{arm} pooled sfb")
    bi = int(np.argmin(own))
    S[arm] = {"sfb_min": float(sfbs.min()), "sfb_median": float(np.median(sfbs)),
              "sfb_max": float(sfbs.max()), "sfb_mean": float(sfbs.mean()),
              "sfb_sd": float(sfbs.std(ddof=1)),
              "ms_min": float(own.min()), "ms_sd": float(own.std(ddof=1)),
              "argmin_layout": RES[arm][bi]["layout"], "argmin_sfb": RES[arm][bi]["sfb"],
              "argmin_ms_own": RES[arm][bi]["mspc_own"],
              "argmin_ms_old_surface": RES[arm][bi]["mspc_old_surface"]}
    log("")
    log(f"{arm}: pooled sfb min {S[arm]['sfb_min']:.4f} median {S[arm]['sfb_median']:.4f} "
        f"max {S[arm]['sfb_max']:.4f} (sd {S[arm]['sfb_sd']:.4f}); ms sd {S[arm]['ms_sd']:.4f}")
    log(f"  best-of-all: {S[arm]['argmin_layout']} sfb {S[arm]['argmin_sfb']:.4f} "
        f"ms(own) {S[arm]['argmin_ms_own']:.4f} ms(OLD surf) {S[arm]['argmin_ms_old_surface']:.4f}")

# the negative control, now on a converged search
FIELD_OLD = {nm: ms_of("OLD", BOARDS[nm]) for nm in FIELD_ORDER if set(BOARDS[nm]) == set(C30M)}
pb = min(FIELD_OLD.values())
nc = {"field_best": pb, "field_best_board": min(FIELD_OLD, key=FIELD_OLD.get),
      "old_argmin": S["OLD"]["argmin_ms_own"], "gap": S["OLD"]["argmin_ms_own"] - pb,
      "gate_pass": bool(abs(S["OLD"]["argmin_ms_own"] - pb) < 0.5)}
log("")
log(f"NEGATIVE CONTROL (converged): OLD best {nc['old_argmin']:.4f} vs field best {pb:.4f} "
    f"({nc['field_best_board']})  gap {nc['gap']:+.4f}  PASS={nc['gate_pass']}")

d = np.array([r["sfb"] for r in RES["NEW"]]) - np.array([r["sfb"] for r in RES["OLD"]])
t = stats.ttest_rel([r["sfb"] for r in RES["NEW"]], [r["sfb"] for r in RES["OLD"]])
P = {"paired_d_sfb_mean": float(d.mean()), "paired_d_sfb_sd": float(d.std(ddof=1)),
     "signs_pos_neg": [int((d > 0).sum()), int((d < 0).sum())], "p": float(t.pvalue),
     "per_rep_d": d.tolist()}
log(f"PAIRED d(sfb) NEW-OLD over {R} replicates: mean {d.mean():+.4f} sd {d.std(ddof=1):.4f} "
    f"signs +{int((d > 0).sum())}/-{int((d < 0).sum())} p={t.pvalue:.4f}")
log("  (the SEARCH-NOISE floor on sfb is the per-arm sd above; compare BEFORE reading the p)")

# the registered verdict
old_min, new_med, new_max = S["OLD"]["sfb_min"], S["NEW"]["sfb_median"], S["NEW"]["sfb_max"]
verdict = ("ARTIFACT" if (new_med < old_min and new_max < 2.0) else
           "REAL -- corrected search still reaches sfb >= 2.0" if new_max >= 2.0 else "PARTIAL")
log(f"REGISTERED VERDICT: {verdict}   (arm-B 2.5391 / candidate 1.7365 reference)")

out = {"K": K, "R": R, "delta": DELTA, "results": RES, "summary": S,
       "negative_control": nc, "paired": P, "verdict": verdict,
       "search_noise_floor_sfb": {arm: S[arm]["sfb_sd"] for arm in ("OLD", "NEW")},
       "wall_s": time.time() - t0}
json.dump(out, open(f"{ART}/c06_search_pooled.json", "w"), indent=1)
log(f"wrote {ART}/c06_search_pooled.json")
log("ALL-DONE")
