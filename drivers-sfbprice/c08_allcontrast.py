"""THE CONSISTENCY ARM: correct EVERY class contrast, not just same-finger.

H-SHRINK measured that the model compresses same-finger's contrast to 0.651x of raw -- but ALSO
compresses five other class contrasts HARDER (0.265-0.408x). Correcting only same-finger to the raw
price while leaving five larger compressions alone is not a defensible objective: it is a targeted
intervention justified by nothing the data singles out.

So this driver asks the question the brief's hypothesis really implies: if you repair the model's
UNDER-CHARGING OF STRUCTURE GENERALLY -- same-finger AND same-hand AND row AND dy AND adjacency --
does the high-sfb cluster survive, and does `candidate` survive? A correction that only helps
candidate when applied selectively to its best axis would be a weak result; this is the stress test
that cannot be accused of being tuned to the answer.

Implementation: an ISOTONIC-style global recalibration of T2 -- fit the affine map that takes the
model's per-pair predictions onto the raw per-pair medians over the 486 supported pairs, then apply
it to ALL 961 cells. That corrects every contrast simultaneously by construction (an affine
expansion of the surface), and its effect on same-finger is a CONSEQUENCE rather than a target.
"""
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
C02 = json.load(open(f"{ART}/c02_contrast.json"))
C03 = json.load(open(f"{ART}/c03_correct.json"))
MASK = surface.same_finger_mask()
same, other = C02["pairs_same"], C02["pairs_other"]
ALL = same + other

CD = production_corpus_dir(None)
tri_freq = {k: v for k, v in load_frequencies(str(CD / "trigrams.txt")).items() if len(k) == 3}
surf = TimeSurface(tri_freq, target_wpm=surface.WPM)
T2_old, Tcm = surf._T2, surf._Tc

# ------------------------------------------------------------------- fit the global affine map
x = np.array([r["pred"] for r in ALL])
y = np.array([r["raw"] for r in ALL])
w = np.array([r["n"] for r in ALL], float)
fit = stats.linregress(x, y)
# also the sample-weighted version, reported so the choice of weighting is visible
Wm = np.polyfit(x, y, 1, w=np.sqrt(w))
log(f"global affine fit over {len(ALL)} supported pairs (raw ~ a + b*model):")
log(f"  UNWEIGHTED  slope {fit.slope:.6f}  intercept {fit.intercept:+.4f}  R2 {fit.rvalue**2:.4f}")
log(f"  WEIGHTED    slope {Wm[0]:.6f}  intercept {Wm[1]:+.4f}")
log(f"  => slope > 1 means the model COMPRESSES the whole surface; the expansion corrects EVERY")
log(f"     contrast at once, and same-finger's correction is a CONSEQUENCE not a target.")

ARMS = {}
for tag, (a_, b_) in (("affine_unweighted", (fit.intercept, fit.slope)),
                      ("affine_weighted", (Wm[1], Wm[0]))):
    ARMS[tag] = {"intercept": float(a_), "slope": float(b_),
                 "T2": a_ + b_ * T2_old}
# the same-finger-only arm, for side-by-side comparison
ARMS["sf_only_additive"] = {"delta": C03["delta"],
                            "T2": surface.corrected_T2(T2_old, C03["delta"], MASK, "add")}
ARMS["uncorrected"] = {"T2": T2_old}


def contrast_on(T):
    ms = float(np.median([T[r["a"], r["b"]] for r in same]))
    mo = float(np.median([T[r["a"], r["b"]] for r in other]))
    return ms - mo


bg = load_frequencies(str(CD / "bigrams.txt"))
sk = load_frequencies(str(CD / "1-skip31.txt"))
KM = KmStats(bg, sk, tri_freq)
RAW = C02["e3_raw"]["penalty"]

log("")
log("=== what each arm does to the SAME-FINGER price, and to the FIVE OTHER contrasts ===")
PARTS = {
    "same-finger": lambda r: r in same,
    "same-hand": lambda r: r["same_hand"],
    "bottom-row landing": lambda r: r["row_b"] == 1,
    "top-row landing": lambda r: r["row_b"] == 3,
    "dy>=2": lambda r: r["dy"] >= 2,
    "adjacent-finger": lambda r: r["adjacent"],
}
sameset = {(r["a"], r["b"]) for r in same}
def in_same(r): return (r["a"], r["b"]) in sameset
PARTS["same-finger"] = in_same

prices = {}
for tag in ("uncorrected", "sf_only_additive", "affine_unweighted", "affine_weighted"):
    T = ARMS[tag]["T2"]
    row = {}
    for pname, pred in PARTS.items():
        A = [r for r in ALL if pred(r)]
        B = [r for r in ALL if not pred(r)]
        rawp = float(np.median([r["raw"] for r in A]) - np.median([r["raw"] for r in B]))
        modp = float(np.median([T[r["a"], r["b"]] for r in A])
                     - np.median([T[r["a"], r["b"]] for r in B]))
        row[pname] = {"raw": rawp, "model": modp, "ratio": modp / rawp if abs(rawp) > 1e-9 else None}
    prices[tag] = row
    log(f"  {tag}:")
    for pname, v in row.items():
        log(f"     {pname:22s} raw {v['raw']:+8.2f}  model {v['model']:+8.2f}  "
            f"ratio {v['ratio']:+.3f}")

log("")
log("=== the 1v1 under each arm: does candidate still win when EVERY contrast is corrected? ===")
T2s, Tcs = surface.load_all_seed_tables(verbose=False)
ARR = {nm: surface.board_arrays(BOARDS[nm], tri_freq) for nm in FIELD_ORDER}
RES = {}
for tag in ("uncorrected", "sf_only_additive", "affine_unweighted", "affine_weighted"):
    if tag == "uncorrected":
        T2list = T2s
    elif tag == "sf_only_additive":
        T2list = [surface.corrected_T2(t, C03["delta"], MASK, "add") for t in T2s]
    else:
        a_, b_ = ARMS[tag]["intercept"], ARMS[tag]["slope"]
        T2list = [a_ + b_ * t for t in T2s]
    X = {nm: np.array([surface.mspc(ARR[nm], T2list[s], Tcs[s]) for s in range(25)])
         for nm in FIELD_ORDER}
    for nm in FIELD_ORDER:
        require_finite(X[nm].tolist(), f"{tag} {nm}")
    means = {nm: float(X[nm].mean()) for nm in FIELD_ORDER}
    order = sorted(FIELD_ORDER, key=lambda n: means[n])
    # floor for THIS arm (the design's own floor; scales with the arm)
    rng = np.random.default_rng(20260803)
    vals = []
    for nm in FIELD_ORDER:
        xx = X[nm]
        for _ in range(2000):
            p = rng.permutation(25)
            vals.append(abs(xx[p[:12]].mean() - xx[p[12:24]].mean()))
    floor = float(np.percentile(vals, 90))
    m_cl = float((X["candidate"] - X["keybo-lsb"]).mean())
    losses = []
    for rival in FIELD_ORDER:
        if rival == "candidate":
            continue
        d = X["candidate"] - X[rival]
        if d.mean() > 0 and abs(d.mean()) >= floor and (d > 0).sum() >= 20:
            losses.append((rival, float(d.mean())))
    RES[tag] = {"means": means, "order": order, "floor": floor,
                "candidate_rank": order.index("candidate") + 1,
                "candidate_losses": losses,
                "margin_vs_keybo_lsb": m_cl,
                "margin_over_floor": abs(m_cl) / floor,
                "armB_minus_candidate": means["arm-B"] - means["candidate"]}
    log(f"  {tag:20s} floor {floor:.4f}  candidate rank {RES[tag]['candidate_rank']}/13  "
        f"losses {losses if losses else '(none)'}")
    log(f"      order: {' < '.join(order[:5])} ...")
    log(f"      candidate vs keybo-lsb margin {m_cl:+.4f} = {abs(m_cl) / floor:.2f}x floor; "
        f"arm-B minus candidate {means['arm-B'] - means['candidate']:+.4f}")

# and: which board does each arm's surface PREFER over the pool (the D1 question per arm)
POOL = {nm: BOARDS[nm] for nm in FIELD_ORDER if set(BOARDS[nm]) == set(C30M)}
C04 = json.load(open(f"{ART}/c04_search.json"))
C06 = json.load(open(f"{ART}/c06_search_pooled.json"))
for src, key in ((C04["results"], "s"), (C06["results"], "r")):
    for arm in ("OLD", "NEW"):
        for r in src[arm]:
            POOL[f"search-{arm}-{key}{r.get('seed', r.get('rep'))}"] = r["layout"]
log("")
log(f"=== D1 per arm: preferred board over a {len(POOL)}-board pool ===")
pref = {}
for tag in ("uncorrected", "sf_only_additive", "affine_unweighted", "affine_weighted"):
    T = ARMS[tag]["T2"]
    sc = TableTrigramScorer.from_table(T[:, :, None] + Tcm, surf.tri, chars=C30M, geometry=G30)
    cov = float(sc._f.sum())
    vals = {nm: sc.fitness(Layout(lay, G30)) / cov for nm, lay in POOL.items()}
    best = min(vals, key=vals.get)
    pref[tag] = {"argmin": best, "argmin_sfb": float(KM.stats(POOL[best])["sfb"]),
                 "argmin_ms": vals[best],
                 "top5": sorted(vals, key=vals.get)[:5],
                 "top5_sfb": [float(KM.stats(POOL[n])["sfb"]) for n in sorted(vals, key=vals.get)[:5]]}
    log(f"  {tag:20s} prefers {best:22s} sfb {pref[tag]['argmin_sfb']:.4f}")
    log(f"      top5: " + "  ".join(f"{n}({s:.3f})" for n, s in zip(pref[tag]['top5'],
                                                                    pref[tag]['top5_sfb'])))

out = {"affine_fit": {"unweighted_slope": float(fit.slope),
                      "unweighted_intercept": float(fit.intercept),
                      "r2": float(fit.rvalue ** 2),
                      "weighted_slope": float(Wm[0]), "weighted_intercept": float(Wm[1]),
                      "n_pairs": len(ALL)},
       "prices_per_arm": prices, "onevone_per_arm": RES, "preference_per_arm": pref,
       "raw_same_finger_penalty": RAW, "wall_s": time.time() - t0}
json.dump(out, open(f"{ART}/c08_allcontrast.json", "w"), indent=1)
log(f"wrote {ART}/c08_allcontrast.json")
log("ALL-DONE")
