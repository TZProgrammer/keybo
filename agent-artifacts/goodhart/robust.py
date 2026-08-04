"""EXPLOIT-1 robustness — does the verdict survive, and is it the STRUCTURE or the LEVEL?

Three questions the headline does not answer, run AFTER the registered rule so they cannot be
mistaken for it. The registered verdict stands as recorded either way (goalpost discipline).

R1 -- BUDGET/SELECTION SENSITIVITY. The G-channel margin is only 0.48x its own floor, so the
     verdict there could plausibly be a lucky pair of best-of-12 draws. Recompute gap-vs-floor at
     n = 3, 6, 12 and ALSO as a full paired bootstrap over which seeds land in each arm's block.
     Reports the FRACTION of resamples that call EXPLOITABLE -- the honest strength of the claim.

R2 -- IS IT THE NULL SPACE OR JUST A WORSE MODEL? This is the confound that would make the whole
     probe uninformative: a proxy can lose a search-then-score comparison simply by being LESS
     ACCURATE, with no null space involved. So I build a CALIBRATED CONTROL SURFACE: take
     T2_interp and replace every cell with its own interp-class mean of the TRUTH
     (groupmean(T2_served)). That surface has:
        - the SAME 378-class null structure as interp.1 (identical collapse, by construction), and
        - ZERO model error otherwise -- it is the BEST POSSIBLE model on that frame (the
          within-group floor INTERPFRAME-1 measured at 2.2399 ms).
     Searching it isolates the structural penalty from the model-quality penalty. If the
     best-possible-on-frame surface ALSO loses to the control, the null space is the cause and
     interp.1 is not merely undertrained. If it does NOT lose, the gap is model error, and the
     collapse story is wrong -- I would have to say so.

R3 -- LEVEL/SCALE IMMUNITY. A search is invariant to an affine transform of its objective, so a
     level or scale offset between the two models CANNOT produce the gap. Demonstrated rather than
     argued: re-run one interp arm on an affinely rescaled surface and confirm the identical board.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import ARTIFACTS, SCRATCH, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import FEATURE_VERSION_INTERP, interp_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402
from keybo.optimize.annealing import SimulatedAnnealing  # noqa: E402
from keybo.optimize.local_search import two_opt  # noqa: E402
from keybo.scoring.base import IScorer  # noqa: E402

WPM, K31_SEEDS = 90.0, (0, 1, 2)
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP = len(POS)
SEEDS_TOTAL = 24
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


models = [XGBoostTypingModel.load(f"{SCRATCH}/interp_mono_seed{s}.json",
                                  expected_feature_version=FEATURE_VERSION_INTERP)
          for s in K31_SEEDS]
surface = default_surface(WPM, None)
T2_SERVED, TCOND = surface._T2.copy(), surface._Tc.copy()
vec_i = np.vstack([interp_features_from_positions(GEO, (a, b), wpm=WPM) for a in POS for b in POS])
T2_INTERP = np.mean([m.predict_ms(vec_i, wpm=WPM).reshape(NP, NP) for m in models], axis=0)
_, inv, cnt = np.unique(vec_i, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()

tri = {k: v for k, v in load_frequencies(str(production_corpus_dir(None) / "trigrams.txt")).items()
       if len(k) == 3}
IDX = {c: i for i, c in enumerate(CHARS)}
IDX[" "] = NP - 1
F3 = np.zeros((NP, NP, NP))
for ng, f in tri.items():
    try:
        F3[IDX[ng[0]], IDX[ng[1]], IDX[ng[2]]] += f
    except KeyError:
        continue
F2 = F3.sum(axis=2)
COVERED = float(F3.sum())
SLOT = {pos: i for i, pos in enumerate(GEO.slots)}


class Arm(IScorer):
    def __init__(self, T2, trigram: bool):
        self._T = (T2[:, :, None] + TCOND) if trigram else T2
        self._F = F3 if trigram else F2
        self._tri = trigram

    def _perm(self, layout):
        p = np.empty(NP, dtype=np.intp)
        for c in CHARS:
            p[IDX[c]] = SLOT[layout.pos(c)]
        p[NP - 1] = NP - 1
        return p

    def fitness(self, layout):
        p = self._perm(layout)
        ix = np.ix_(p, p, p) if self._tri else np.ix_(p, p)
        return float((self._F * self._T[ix]).sum())

    def ms_per_char(self, layout):
        return self.fitness(layout) / COVERED


def search(scorer, seed):
    sa = SimulatedAnnealing(seed=seed, alpha=0.999, progress=False)
    return two_opt(sa.optimize(Layout(CHARS, GEO), scorer), scorer)


ex = json.load(open(f"{ARTIFACTS}/exploit.json"))
out = {"note": "robustness, run AFTER the registered rule; the registered verdict stands as recorded"}

# =============================================================================================
# R1 — budget/selection sensitivity, by paired bootstrap over which seeds fill each block
# =============================================================================================
rng = np.random.default_rng(11071971)
out["R1"] = {}
for ch in ("G", "B"):
    ia, sa_ = ex["arms"][f"{ch}-INTERP"], ex["arms"][f"{ch}-SERVED"]
    own_i, tr_i = np.array(ia["own_ms_per_char"]), np.array(ia["trusted_ms_per_char"])
    own_s, tr_s = np.array(sa_["own_ms_per_char"]), np.array(sa_["trusted_ms_per_char"])
    per_n = {}
    for n in (3, 6, 12):
        gaps, floors, calls = [], [], 0
        for _ in range(4000):
            # a bootstrap draw of WHICH n seeds each arm got, independently -- the real sampling
            # variation a campaign faces when it picks a budget and a seed range
            bi = rng.choice(SEEDS_TOTAL, n, replace=False)
            bs = rng.choice(SEEDS_TOTAL, n, replace=False)
            gi = tr_i[bi[np.argmin(own_i[bi])]]
            gs = tr_s[bs[np.argmin(own_s[bs])]]
            gaps.append(gi - gs)
            # the floor, measured the SAME way at this n: two disjoint n-blocks of the control
            p = rng.permutation(SEEDS_TOTAL)
            a, b = p[:n], p[n:2 * n]
            fa = tr_s[a[np.argmin(own_s[a])]]
            fb = tr_s[b[np.argmin(own_s[b])]]
            floors.append(abs(fa - fb))
        gaps, floors = np.array(gaps), np.array(floors)
        f95 = float(np.percentile(floors, 95))
        calls = float((gaps > f95).mean())
        per_n[str(n)] = {
            "gap_median": float(np.median(gaps)), "gap_p05": float(np.percentile(gaps, 5)),
            "gap_p95": float(np.percentile(gaps, 95)), "floor_p95": f95,
            "frac_resamples_calling_EXPLOITABLE": calls,
            "frac_gap_positive": float((gaps > 0).mean()),
        }
        log(f"[R1/{ch}] n={n:2d}: gap median {np.median(gaps):+.4f} "
            f"[p05 {np.percentile(gaps, 5):+.4f}, p95 {np.percentile(gaps, 95):+.4f}]  "
            f"floor_p95 {f95:.4f}  -> EXPLOITABLE in {100 * calls:.1f}% of resamples  "
            f"(gap>0 in {100 * (gaps > 0).mean():.1f}%)")
    out["R1"][ch] = per_n
    # ALSO: the every-seed view -- no selection at all, all 24 vs all 24
    log(f"[R1/{ch}] ALL-SEEDS mean trusted: interp {tr_i.mean():.6f} vs served {tr_s.mean():.6f} "
        f"(delta {tr_i.mean() - tr_s.mean():+.6f});  worst-case interp {tr_i.max():.6f}")
    out["R1"][ch]["all_seeds"] = {
        "interp_mean": float(tr_i.mean()), "served_mean": float(tr_s.mean()),
        "delta_of_means": float(tr_i.mean() - tr_s.mean()),
        "interp_min": float(tr_i.min()), "served_min": float(tr_s.min()),
        "n_interp_seeds_beating_best_served": int((tr_i < tr_s.min()).sum()),
    }
with open(f"{ARTIFACTS}/robust.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)

# =============================================================================================
# R2 — the CALIBRATED-CONTROL surface: same null structure, ZERO model error
# =============================================================================================
flat_s = T2_SERVED.ravel()
gmean = np.bincount(inv, weights=flat_s, minlength=len(cnt)) / cnt
T2_BEST_ON_FRAME = gmean[inv].reshape(NP, NP)
# by construction: identical collapse structure, and the best any model on the frame could do
_, inv_b, cnt_b = np.unique(T2_BEST_ON_FRAME.ravel().round(9), return_inverse=True, return_counts=True)
log(f"[R2] best-on-frame surface: {len(cnt_b)} distinct values (interp.1 has "
    f"{len(np.unique(T2_INTERP.ravel().round(9)))}; served {len(np.unique(flat_s.round(9)))})")
wg = float((np.abs(flat_s - T2_BEST_ON_FRAME.ravel())).mean())
log(f"[R2] its unweighted within-group MAE vs the truth = {wg:.4f} ms "
    f"(INTERPFRAME-1 published floor_umae 2.7981)")

out["R2"] = {"distinct_values_best_on_frame": int(len(cnt_b)),
             "unweighted_within_group_mae_ms": wg, "channels": {}}
for ch, trig in (("G", True), ("B", False)):
    trusted = Arm(T2_SERVED, trig)
    best_arm = Arm(T2_BEST_ON_FRAME, trig)
    own, tru = [], []
    for s in range(SEEDS_TOTAL):
        b = search(best_arm, s)
        own.append(best_arm.ms_per_char(b))
        tru.append(trusted.ms_per_char(b))
    own, tru = np.array(own), np.array(tru)
    k = int(np.argmin(own[:12]))
    ts = ex["verdict"][ch]["served_trusted_ms_per_char"]
    gap = float(tru[k] - ts)
    f95 = out["R1"][ch]["12"]["floor_p95"]
    out["R2"]["channels"][ch] = {
        "best_on_frame_trusted_ms_per_char": float(tru[k]), "seed": k,
        "served_control_trusted": ts, "gap_ms_per_char": gap,
        "gap_pct": 100.0 * gap / ts, "floor_p95_n12": f95,
        "exploitable": bool(gap > f95 and gap > 0),
        "interp1_gap_for_comparison": ex["verdict"][ch]["gap_ms_per_char"],
        "share_of_interp1_gap": gap / ex["verdict"][ch]["gap_ms_per_char"],
        "board": "".join(search(best_arm, k).chars),
    }
    r = out["R2"]["channels"][ch]
    log(f"[R2/{ch}] BEST-POSSIBLE-on-frame surface: trusted {tru[k]:.6f} vs control {ts:.6f} "
        f"=> gap {gap:+.6f} ({r['gap_pct']:+.4f}%) vs floor {f95:.4f} -> "
        f"{'EXPLOITABLE' if r['exploitable'] else 'EXONERATED'}")
    log(f"        that is {100 * r['share_of_interp1_gap']:.1f}% of interp.1's own "
        f"{r['interp1_gap_for_comparison']:+.4f} gap  => the rest is MODEL ERROR")
with open(f"{ARTIFACTS}/robust.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)

# =============================================================================================
# R3 — affine immunity, demonstrated
# =============================================================================================
a_arm = Arm(T2_INTERP, False)
b_arm = Arm(3.7 * T2_INTERP + 41.0, False)
same = []
for s in (0, 1, 2):
    ba = "".join(search(a_arm, s).chars)
    bb = "".join(search(b_arm, s).chars)
    same.append(ba == bb)
    log(f"[R3] seed {s}: affine-rescaled surface gives the {'SAME' if ba == bb else 'DIFFERENT'} board")
out["R3"] = {"affine_invariant_boards": same, "all_same": all(same),
             "transform": "3.7 * T2_interp + 41.0"}
log(f"[R3] => a LEVEL/SCALE offset between the models CANNOT explain the gap: {all(same)}")

with open(f"{ARTIFACTS}/robust.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/robust.json")
