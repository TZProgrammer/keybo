"""INVARIANT B — the paired A/B search: does the CORRECTED price change WHAT THE SEARCH FINDS?

Same code, same 12 seeds, same budget, same start. The ONLY difference between arms is the 96
edited T2 cells. Objective = the REPORTED GAUGE (T2+Tcond over blend-v1, seed-mean, C30M) through
the reviewed TableTrigramScorer.from_table -- NOT the repo's default bigram objective, which ranks
layouts INVERTED to the gauge.

Also runs A1: the pick2 contrast RE-MEASURED on the corrected T2, which is the proof the price is
actually corrected (a number, not an assertion). Uses the cached pair records from c02 so the
609MB stroke frame need not be re-read.
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

BOARDS = build_boards()
C02 = json.load(open(f"{ART}/c02_contrast.json"))
C03 = json.load(open(f"{ART}/c03_correct.json"))
DELTA = C03["delta"]
MASK = surface.same_finger_mask()
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

# =========================================================== A1: is the price ACTUALLY corrected?
log("")
log("=== A1: re-measure the pick2 contrast ON THE CORRECTED T2 (the proof, not an assertion) ===")
T2s3, _ = surface.load_all_seed_tables(seeds=(0, 1, 2), verbose=False)
T2m_old = np.mean(T2s3, axis=0)
T2m_new = surface.corrected_T2(T2m_old, DELTA, MASK, "add")
same, other = C02["pairs_same"], C02["pairs_other"]
a1 = {}
for tag, T in (("uncorrected", T2m_old), ("corrected", T2m_new)):
    ms = float(np.median([T[r["a"], r["b"]] for r in same]))
    mo = float(np.median([T[r["a"], r["b"]] for r in other]))
    a1[tag] = {"model_median_same": ms, "model_median_other": mo, "model_penalty": ms - mo}
    log(f"  {tag:12s} model penalty {ms - mo:+.4f} ms   (same {ms:.4f}, other {mo:.4f})")
a1["raw_penalty_target"] = C02["e3_raw"]["penalty"]
a1["abs_err_vs_raw"] = abs(a1["corrected"]["model_penalty"] - C02["e3_raw"]["penalty"])
a1["gate_pass"] = bool(a1["abs_err_vs_raw"] < 0.01)
log(f"  A1 GATE: corrected penalty {a1['corrected']['model_penalty']:+.4f} vs raw target "
    f"{C02['e3_raw']['penalty']:+.4f}  |err| {a1['abs_err_vs_raw']:.3e}  PASS={a1['gate_pass']}")

# =============================================================================== the two scorers
CD = production_corpus_dir(None)
tri_freq = {k: v for k, v in load_frequencies(str(CD / "trigrams.txt")).items() if len(k) == 3}
surf = TimeSurface(tri_freq, target_wpm=surface.WPM)
assert np.abs(surf._T2 - T2m_old).max() == 0.0, "shipped TimeSurface T2 != my seed-mean rebuild"
Tcm = surf._Tc


def make_scorer(T2m):
    sc = TableTrigramScorer.from_table(T2m[:, :, None] + Tcm, surf.tri, chars=C30M, geometry=G30)
    sc._covered = float(sc._f.sum())
    return sc


SC = {"OLD": make_scorer(T2m_old), "NEW": make_scorer(T2m_new)}
# parity of the OLD arm's objective against the SHIPPED analyzer -- the gate cli/optimize.py runs
par = {nm: abs(SC["OLD"].fitness(Layout(BOARDS[nm], G30)) - surf.card(BOARDS[nm]).total_ms)
       / surf.card(BOARDS[nm]).total_ms
       for nm in FIELD_ORDER if set(BOARDS[nm]) == set(C30M)}
log(f"gauge parity (OLD arm) worst rel dev vs shipped card().total_ms = {max(par.values()):.3e} "
    f"(cli tolerance 1e-12)")
assert max(par.values()) < 1e-12, "gauge parity gate FAILED -- the objective is not the gauge"

bg = load_frequencies(str(CD / "bigrams.txt"))
sk = load_frequencies(str(CD / "1-skip31.txt"))
KM = KmStats(bg, sk, tri_freq)


def sfb_of(lay):
    return float(KM.stats(lay)["sfb"])


def mspc_on(sc, lay):
    return sc.fitness(Layout(lay, G30)) / sc._covered


def one_attempt(sc, seed):
    """SA + 2-opt + 3-opt, exactly cli/optimize.py's _one_attempt/_polish with --three-opt."""
    lay = Layout(C30M, G30)
    sa = SimulatedAnnealing(seed=seed, alpha=0.999, max_outer=None, progress=False)
    best = sa.optimize(lay, sc)
    return three_opt(two_opt(best, sc), sc)


log("")
log(f"=== INVARIANT B: paired A/B search, {N_SEEDS} shared seeds, gauge objective, "
    f"SA+2opt+3opt ===")
RES = {"OLD": [], "NEW": []}
for seed in range(N_SEEDS):
    for arm in ("OLD", "NEW"):
        ts = time.time()
        lay = "".join(one_attempt(SC[arm], seed).chars)
        # score EVERY found board on BOTH surfaces, so the two arms are comparable either way
        rec = {"seed": seed, "layout": lay, "sfb": sfb_of(lay),
               "mspc_own": mspc_on(SC[arm], lay),
               "mspc_old_surface": mspc_on(SC["OLD"], lay),
               "mspc_new_surface": mspc_on(SC["NEW"], lay),
               "wall_s": time.time() - ts}
        RES[arm].append(rec)
        log(f"  seed{seed:<3d} {arm}  sfb {rec['sfb']:.4f}  ms/char(own) {rec['mspc_own']:.4f}  "
            f"ms/char(old surf) {rec['mspc_old_surface']:.4f}  [{rec['wall_s']:.0f}s]  {lay}")

for arm in ("OLD", "NEW"):
    require_finite([r["sfb"] for r in RES[arm]], f"{arm} sfb")
    require_finite([r["mspc_own"] for r in RES[arm]], f"{arm} ms/char")

S = {}
for arm in ("OLD", "NEW"):
    sfbs = np.array([r["sfb"] for r in RES[arm]])
    own = np.array([r["mspc_own"] for r in RES[arm]])
    best_i = int(np.argmin(own))
    S[arm] = {"sfb_min": float(sfbs.min()), "sfb_median": float(np.median(sfbs)),
              "sfb_max": float(sfbs.max()), "sfb_mean": float(sfbs.mean()),
              "sfb_sd": float(sfbs.std(ddof=1)),
              "argmin_layout": RES[arm][best_i]["layout"],
              "argmin_sfb": RES[arm][best_i]["sfb"],
              "argmin_mspc_own": RES[arm][best_i]["mspc_own"],
              "argmin_mspc_old_surface": RES[arm][best_i]["mspc_old_surface"],
              "n": len(sfbs)}
    log("")
    log(f"{arm} arm: sfb min {S[arm]['sfb_min']:.4f} median {S[arm]['sfb_median']:.4f} "
        f"max {S[arm]['sfb_max']:.4f} (sd {S[arm]['sfb_sd']:.4f})")
    log(f"  argmin: {S[arm]['argmin_layout']}  sfb {S[arm]['argmin_sfb']:.4f}  "
        f"ms/char(own) {S[arm]['argmin_mspc_own']:.4f}  "
        f"ms/char(OLD surface) {S[arm]['argmin_mspc_old_surface']:.4f}")

# ---------------- the registered NEGATIVE CONTROL: OLD must land in the known plateau ----------
FIELD_OLD = {nm: mspc_on(SC["OLD"], BOARDS[nm]) for nm in FIELD_ORDER
             if set(BOARDS[nm]) == set(C30M)}
plateau_best = min(FIELD_OLD.values())
nc = {"field_best_on_old_surface": plateau_best,
      "field_best_board": min(FIELD_OLD, key=FIELD_OLD.get),
      "arm_old_argmin": S["OLD"]["argmin_mspc_own"],
      "gap_vs_field_best": S["OLD"]["argmin_mspc_own"] - plateau_best,
      "gate_pass": bool(abs(S["OLD"]["argmin_mspc_own"] - plateau_best) < 0.5)}
log("")
log(f"NEGATIVE CONTROL: ARM-OLD argmin {S['OLD']['argmin_mspc_own']:.4f} vs field best "
    f"{plateau_best:.4f} ({nc['field_best_board']})  gap {nc['gap_vs_field_best']:+.4f}  "
    f"PASS={nc['gate_pass']}  (bar: |gap| < 0.5 = the known ~0.5-wide plateau)")

# ------------------------------------------------------- the registered artifact-vs-real verdict
old_min, new_med = S["OLD"]["sfb_min"], S["NEW"]["sfb_median"]
new_max = S["NEW"]["sfb_max"]
if new_med < old_min and new_max < 2.0:
    verdict = "ARTIFACT -- corrected search abandons high sfb entirely"
elif new_max >= 2.0:
    verdict = "REAL -- corrected search still reaches sfb >= 2.0"
else:
    verdict = "PARTIAL -- a shift, quoted by magnitude"
V = {"verdict": verdict, "d_median_sfb": new_med - S["OLD"]["sfb_median"],
     "old_sfb_min": old_min, "new_sfb_median": new_med, "new_sfb_max": new_max,
     "reference_armB_sfb": sfb_of(BOARDS["arm-B"]),
     "reference_candidate_sfb": sfb_of(BOARDS["candidate"])}
log("")
log(f"REGISTERED VERDICT: {verdict}")
log(f"  d(median sfb) OLD->NEW = {V['d_median_sfb']:+.4f} pp   "
    f"(arm-B 2.5391 / candidate 1.7365 for reference)")

# does either arm's board beat candidate? (the 14th-entry question from the prereg)
beats = {}
for arm in ("OLD", "NEW"):
    lay = S[arm]["argmin_layout"]
    for surf_tag, sc in (("old", SC["OLD"]), ("new", SC["NEW"])):
        beats[f"{arm}_argmin_vs_candidate_on_{surf_tag}"] = {
            "found": mspc_on(sc, lay), "candidate": mspc_on(sc, BOARDS["candidate"]),
            "margin": mspc_on(sc, lay) - mspc_on(sc, BOARDS["candidate"])}
log("")
for k, v in beats.items():
    log(f"  {k:38s} found {v['found']:.4f} vs candidate {v['candidate']:.4f} "
        f"margin {v['margin']:+.4f}")

out = {"a1": a1, "n_seeds": N_SEEDS, "gauge_parity_worst": max(par.values()),
       "results": RES, "summary": S, "negative_control": nc, "verdict": V,
       "vs_candidate": beats, "delta": DELTA, "wall_s": time.time() - t0}
json.dump(out, open(f"{ART}/c04_search.json", "w"), indent=1)
log(f"wrote {ART}/c04_search.json")
log("ALL-DONE")
