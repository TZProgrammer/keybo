"""INVARIANT D1, measured WITHOUT search noise: does the corrected objective PREFER lower sfb?

The search output (c04) answers D1 but is contaminated by search noise -- the campaign's own
search-seed floor is 0.883 ms/char, ~3x the model-seed floor, so a 12-seed argmin is a noisy
estimate of "what the objective wants". This driver answers the same question DETERMINISTICALLY,
three ways that share no search randomness at all:

  P1  PREFERENCE ON A FIXED POOL. Score one pool of boards on both surfaces and ask which board
      each surface picks, and what its sfb is. Same pool both sides => zero search noise.
  P2  THE INDUCED sfb GRADIENT. Over 2-opt neighbours of the field boards, regress d(ms/char) on
      d(sfb) under each surface. The corrected surface must show a MORE POSITIVE slope; the change
      in slope is the re-pricing made visible as a preference, not as an outcome.
  P3  THE SHADOW PRICE, i.e. the constrained optimum: min ms/char s.t. sfb <= cap, over the pool,
      under each surface. This is PRICEBAND-1's estimand and it is what "does the objective want
      high sfb" means precisely -- if the unconstrained optimum's sfb FALLS under correction, the
      objective's preference moved.
"""
import itertools
import json
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
from keybo.scoring.table_trigram import TableTrigramScorer  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

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
CHARS = list(C30M)


def sfb_of(lay):
    return float(KM.stats(lay)["sfb"])


def ms_of(arm, lay):
    return SC[arm].fitness(Layout(lay, G30)) / SC[arm]._covered


# ---------------------------------------------------------------------------- the pool
POOL = {nm: BOARDS[nm] for nm in FIELD_ORDER if set(BOARDS[nm]) == set(C30M)}
try:
    C04 = json.load(open(f"{ART}/c04_search.json"))
    for arm in ("OLD", "NEW"):
        for r in C04["results"][arm]:
            POOL[f"search-{arm}-s{r['seed']}"] = r["layout"]
    log(f"pool: {len(POOL)} boards ({len(FIELD_ORDER)} field C30M + searched boards from c04)")
except FileNotFoundError:
    log(f"pool: {len(POOL)} boards (c04 not yet written; field only)")

rows = {}
for nm, lay in POOL.items():
    rows[nm] = {"layout": lay, "sfb": sfb_of(lay),
                "ms_old": ms_of("OLD", lay), "ms_new": ms_of("NEW", lay)}
require_finite([v["ms_old"] for v in rows.values()], "pool ms_old")
require_finite([v["ms_new"] for v in rows.values()], "pool ms_new")

log("")
log("=== P1: which board does each surface PREFER, on ONE fixed pool (zero search noise) ===")
best_old = min(rows, key=lambda n: rows[n]["ms_old"])
best_new = min(rows, key=lambda n: rows[n]["ms_new"])
p1 = {"n_pool": len(rows),
      "argmin_old": best_old, "argmin_old_sfb": rows[best_old]["sfb"],
      "argmin_old_ms": rows[best_old]["ms_old"],
      "argmin_new": best_new, "argmin_new_sfb": rows[best_new]["sfb"],
      "argmin_new_ms": rows[best_new]["ms_new"],
      "argmin_changed": best_old != best_new,
      "d_argmin_sfb": rows[best_new]["sfb"] - rows[best_old]["sfb"]}
log(f"  OLD surface prefers {best_old:20s} sfb {rows[best_old]['sfb']:.4f}  "
    f"ms {rows[best_old]['ms_old']:.4f}")
log(f"  NEW surface prefers {best_new:20s} sfb {rows[best_new]['sfb']:.4f}  "
    f"ms {rows[best_new]['ms_new']:.4f}")
log(f"  => argmin changed: {p1['argmin_changed']}   d(sfb of the preferred board) "
    f"{p1['d_argmin_sfb']:+.4f} pp")
# top-5 under each
for tag, key in (("OLD", "ms_old"), ("NEW", "ms_new")):
    top = sorted(rows, key=lambda n: rows[n][key])[:5]
    log(f"  {tag} top-5: " + "  ".join(f"{n}(sfb {rows[n]['sfb']:.3f})" for n in top))
p1["top5_old"] = sorted(rows, key=lambda n: rows[n]["ms_old"])[:5]
p1["top5_new"] = sorted(rows, key=lambda n: rows[n]["ms_new"])[:5]
# rank correlation of sfb with each surface's ms/char over the pool
sf = np.array([rows[n]["sfb"] for n in rows])
p1["spearman_sfb_vs_ms_old"] = float(stats.spearmanr(sf, [rows[n]["ms_old"] for n in rows]).statistic)
p1["spearman_sfb_vs_ms_new"] = float(stats.spearmanr(sf, [rows[n]["ms_new"] for n in rows]).statistic)
log(f"  spearman(sfb, ms/char): OLD {p1['spearman_sfb_vs_ms_old']:+.4f}  "
    f"NEW {p1['spearman_sfb_vs_ms_new']:+.4f}  (more positive = surface dislikes sfb more)")

log("")
log("=== P2: the INDUCED sfb GRADIENT over 2-opt neighbourhoods (deterministic) ===")
p2 = {}
for base in ("candidate", "arm-B", "keybo-lsb", "F(2.0)"):
    lay = BOARDS[base]
    d_sfb, d_old, d_new = [], [], []
    b_sfb, b_old, b_new = sfb_of(lay), ms_of("OLD", lay), ms_of("NEW", lay)
    for i, j in itertools.combinations(range(30), 2):
        L = list(lay); L[i], L[j] = L[j], L[i]
        nl = "".join(L)
        d_sfb.append(sfb_of(nl) - b_sfb)
        d_old.append(ms_of("OLD", nl) - b_old)
        d_new.append(ms_of("NEW", nl) - b_new)
    d_sfb = np.array(d_sfb); d_old = np.array(d_old); d_new = np.array(d_new)
    s_old = stats.linregress(d_sfb, d_old)
    s_new = stats.linregress(d_sfb, d_new)
    p2[base] = {"n_neighbours": len(d_sfb),
                "slope_old": float(s_old.slope), "r2_old": float(s_old.rvalue ** 2),
                "slope_new": float(s_new.slope), "r2_new": float(s_new.rvalue ** 2),
                "d_slope": float(s_new.slope - s_old.slope),
                "analytic_d_slope_note": "the surcharge's exact contribution is DELTA * "
                                         "d(sf_share)/d(sfb), i.e. a convention conversion"}
    log(f"  {base:12s} n={len(d_sfb)}  slope OLD {s_old.slope:+.4f} (R2 {s_old.rvalue**2:.3f})  "
        f"slope NEW {s_new.slope:+.4f} (R2 {s_new.rvalue**2:.3f})  d {s_new.slope - s_old.slope:+.4f} "
        f"ms/char per pp")

log("")
log("=== P3: the CONSTRAINED optimum over the pool: min ms/char s.t. sfb <= cap ===")
caps = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0, 99.0]
p3 = {}
for arm, key in (("OLD", "ms_old"), ("NEW", "ms_new")):
    curve = []
    for cap in caps:
        elig = [n for n in rows if rows[n]["sfb"] <= cap]
        if not elig:
            curve.append({"cap": cap, "best": None}); continue
        b = min(elig, key=lambda n: rows[n][key])
        curve.append({"cap": cap, "best": b, "ms": rows[b][key], "sfb": rows[b]["sfb"],
                      "n_eligible": len(elig)})
    p3[arm] = curve
    unc = curve[-1]
    log(f"  {arm}: unconstrained best {unc['best']} (sfb {unc['sfb']:.4f}, ms {unc['ms']:.4f})")
    for c in curve[:-1]:
        if c["best"] is None:
            continue
        log(f"     cap {c['cap']:5.1f}  best {c['best']:20s} sfb {c['sfb']:.4f}  ms {c['ms']:.4f}  "
            f"cost vs unconstrained {c['ms'] - unc['ms']:+.4f}")

out = {"delta": DELTA, "pool": rows, "p1_preference": p1, "p2_gradient": p2,
       "p3_constrained": p3, "wall_s": time.time() - t0}
json.dump(out, open(f"{ART}/c05_objective.json", "w"), indent=1)
log(f"wrote {ART}/c05_objective.json")
log("ALL-DONE")
