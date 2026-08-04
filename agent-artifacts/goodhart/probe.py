"""Capability + timing probe. Runs BEFORE the preregistration so the budget I register is
one I have measured, not guessed. No comparison numbers are produced here."""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import assert_tree, require  # noqa: E402
assert_tree()

import numpy as np  # noqa: E402
from keybo import features as FT  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402

# Assert every symbol I intend to lean on EXISTS on this tree.
for s in ("bigram_features_from_positions", "interp_features_from_positions",
          "BIGRAM_INTERP_FEATURE_NAMES", "FEATURE_VERSION_INTERP"):
    require(FT, s)
from keybo.analysis import timecard as TC  # noqa: E402
for s in ("default_surface", "gauge_search_scorer", "GaugeTrigramScorer"):
    require(TC, s)
from keybo.training import train as TR  # noqa: E402
require(TR, "train_bigram_model")
from keybo.scoring import table_scorer as TS  # noqa: E402
require(TS, "TableBigramScorer")
from keybo.optimize import annealing as AN, local_search as LS  # noqa: E402
require(AN, "SimulatedAnnealing"); require(LS, "two_opt")

print("[probe] all symbols present")

# --- the served bigram surface: what the DEFAULT objective searches ---------------------
t0 = time.time()
from keybo.models.xgboost_model import XGBoostTypingModel  # noqa: E402
from keybo.cli._scorer import load_freqs  # noqa: E402
from argparse import Namespace  # noqa: E402
from keybo.data.corpus import production_corpus_dir  # noqa: E402
cd = production_corpus_dir(None)
print(f"[probe] production corpus dir = {cd}")
import os
print(f"[probe] corpus files: {sorted(os.listdir(cd))[:8]}")

C30M = TC.default_surface(90.0, None)
print(f"[probe] surface geometry slots={len(C30M.geometry.slots)}  T2 shape={C30M._T2.shape}")
print(f"[probe] surface load {time.time()-t0:.1f}s")

# how many k31 seeds does the served surface average?
print(f"[probe] served surface averages seeds: T2s={'kept' if C30M._T2s is not None else 'not kept'}")

# --- gauge scorer parity, on qwerty ------------------------------------------------------
import keybo.analysis.surfaces as SF  # noqa: E402
print(f"[probe] C30M charset = {SF.C30M!r}")
g = TC.gauge_search_scorer(chars=SF.C30M, target_wpm=90.0, corpus=None)
lay = Layout(SF.C30M, ROW_STAGGERED_30)
t0 = time.time()
dev = g.parity_rel_dev(lay)
print(f"[probe] gauge parity rel dev on C30M start: {dev:.3e}  ({time.time()-t0:.2f}s)")
print(f"[probe] gauge ms/char C30M start = {g.ms_per_char(lay):.6f}")
t0=time.time()
for _ in range(200): g.fitness(lay)
print(f"[probe] gauge fitness: {(time.time()-t0)/200*1000:.3f} ms/eval")
