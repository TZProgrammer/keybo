"""Timing probe ONLY — no scientific number is produced or recorded here.

Scopes the prereg: how long one SA+2-opt search takes, and how long one bigram model trains.
Run BEFORE the prereg so the registered budget is a measured one rather than a guess.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.optimize.annealing import SimulatedAnnealing  # noqa: E402
from keybo.optimize.local_search import two_opt  # noqa: E402
from keybo.scoring.base import IScorer  # noqa: E402

WPM = 90.0
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP_ = len(POS)

t = time.time()
surface = default_surface(WPM, None)
print(f"[t] default_surface {time.time() - t:.1f}s")
T2, TCOND = surface._T2.copy(), surface._Tc.copy()

tri = {
    k: v
    for k, v in load_frequencies(str(production_corpus_dir(None) / "trigrams.txt")).items()
    if len(k) == 3
}
IDX = {c: i for i, c in enumerate(CHARS)}
IDX[" "] = NP_ - 1
F3 = np.zeros((NP_, NP_, NP_))
for ng, f in tri.items():
    try:
        F3[IDX[ng[0]], IDX[ng[1]], IDX[ng[2]]] += f
    except KeyError:
        continue
F2 = F3.sum(axis=2)
COVERED = float(F3.sum())
SLOT = {pos: i for i, pos in enumerate(GEO.slots)}


class BigramArm(IScorer):
    def __init__(self, T2):
        self._T = T2

    def _perm(self, layout):
        p = np.empty(NP_, dtype=np.intp)
        for c in CHARS:
            p[IDX[c]] = SLOT[layout.pos(c)]
        p[NP_ - 1] = NP_ - 1
        return p

    def fitness(self, layout):
        p = self._perm(layout)
        return float((F2 * self._T[np.ix_(p, p)]).sum())

    def ms_per_char(self, layout):
        return self.fitness(layout) / COVERED


class GaugeArm(IScorer):
    def __init__(self, T2):
        self._T = T2[:, :, None] + TCOND

    def _perm(self, layout):
        p = np.empty(NP_, dtype=np.intp)
        for c in CHARS:
            p[IDX[c]] = SLOT[layout.pos(c)]
        p[NP_ - 1] = NP_ - 1
        return p

    def fitness(self, layout):
        p = self._perm(layout)
        return float((F3 * self._T[np.ix_(p, p, p)]).sum())

    def ms_per_char(self, layout):
        return self.fitness(layout) / COVERED


for label, sc in (("B (bigram)", BigramArm(T2)), ("G (gauge)", GaugeArm(T2))):
    t = time.time()
    sa = SimulatedAnnealing(seed=0, alpha=0.999, progress=False)
    best = sa.optimize(Layout(CHARS, GEO), sc)
    t_sa = time.time() - t
    t = time.time()
    two_opt(best, sc)
    print(f"[t] {label}: SA {t_sa:.1f}s + two_opt {time.time() - t:.1f}s")

# one bigram model train, on the LOLO stroke data
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

t = time.time()
rows = load_strokes(
    "/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv", ngram_len=2, wpm_threshold=0, min_samples=1
)
print(f"[t] load_strokes {time.time() - t:.1f}s  ({len(rows)} rows)")
t = time.time()
train_bigram_model(rows, target_wpm=WPM, geometry=ROW_STAGGERED_31, seed=0, n_jobs=8)
print(f"[t] train_bigram_model (served) {time.time() - t:.1f}s")
t = time.time()
train_bigram_model(
    rows, target_wpm=WPM, geometry=ROW_STAGGERED_31, seed=0, n_jobs=8, interp=True, monotone=True
)
print(f"[t] train_bigram_model (interp,mono) {time.time() - t:.1f}s")
