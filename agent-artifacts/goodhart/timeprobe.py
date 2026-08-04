"""How long does ONE search attempt take on each surface? Sizes the registered budget."""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-goodhart/agent-artifacts/goodhart")
from _boot import assert_tree  # noqa: E402
assert_tree()
import numpy as np  # noqa: E402
from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import gauge_search_scorer  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.optimize.annealing import SimulatedAnnealing, stopping_point  # noqa: E402
from keybo.optimize.local_search import two_opt  # noqa: E402

print(f"[t] stopping_point(30) = {stopping_point(30)} consecutive non-improving iters")
g = gauge_search_scorer(chars=SF.C30M, target_wpm=90.0, corpus=None)
lay = Layout(SF.C30M, ROW_STAGGERED_30)
print(f"[t] start ms/char = {g.ms_per_char(lay):.6f}")
t0 = time.time()
sa = SimulatedAnnealing(seed=0, alpha=0.999, progress=False)
best = sa.optimize(Layout(SF.C30M, ROW_STAGGERED_30), g)
t_sa = time.time() - t0
t0 = time.time()
pol = two_opt(best, g)
t_2opt = time.time() - t0
print(f"[t] SA {t_sa:.1f}s -> ms/char {g.ms_per_char(best):.6f}")
print(f"[t] 2opt {t_2opt:.1f}s -> ms/char {g.ms_per_char(pol):.6f}")
print(f"[t] TOTAL one gauge attempt = {t_sa+t_2opt:.1f}s")
print(f"[t] board: {''.join(pol.chars)}")
