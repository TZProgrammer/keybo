"""ARM A2 — the control on the RULER IT IS REPORTED ON.

SEARCHPARAMS-1 (PREREG:10540): `keybo optimize --ngram bigram` minimizes TableBigramScorer,
which ranks layouts INVERTED vs the reported ms/char gauge (their spearman 0.6715, my 0.246).
The reported gauge is analyze's TimeSurface = mean-over-3-bigram-seeds T2 + mean-over-3-trigram
-seeds Tc, trigram-freq weighted. So build a scorer that IS that gauge and run the SAME shipped
annealer + 2-opt at the SAME defaults, seeds 0-9. Only the search scorer changes.
"""
import sys, json, time
sys.path.insert(0,"/tmp/normopt/src")
import numpy as np
from keybo.analysis.timecard import default_surface
from keybo.scoring.base import IScorer
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.optimize.annealing import SimulatedAnnealing
from keybo.optimize.local_search import two_opt
from keybo.analysis import surfaces as S

surf = default_surface(90.0, None)          # the SHIPPED reported-gauge surface
POS  = (*G.slots, G.space_position)
N    = len(POS)
T2, Tc = surf._T2, surf._Tc
tri  = surf.tri

class ReportedGaugeScorer(IScorer):
    """fitness == analyze's total predicted ms (monotone in ms_per_char at fixed charset:
    coverage depends only on the CHARSET, which the search never changes)."""
    def __init__(self, chars):
        self._chars = tuple(chars)
        idx = {c:i for i,c in enumerate(self._chars)}
        rows=[]
        for ng,f in tri.items():
            if len(ng)!=3: continue
            try: a,b,c = (N-1 if ch==" " else idx[ch] for ch in ng)
            except KeyError: continue
            rows.append((a,b,c,f))
        self._i=np.array([r[0] for r in rows]); self._j=np.array([r[1] for r in rows])
        self._k=np.array([r[2] for r in rows]); self._f=np.array([r[3] for r in rows],dtype=np.float64)
        self._covered=float(self._f.sum())
    def _perm(self, layout):
        pos={ch:n for n,ch in enumerate(layout.chars)}
        p=np.empty(N,dtype=np.int64)
        for n,ch in enumerate(self._chars): p[n]=pos[ch]
        p[N-1]=N-1
        return p
    def fitness(self, layout):
        p=self._perm(layout)
        a,b,c = p[self._i],p[self._j],p[self._k]
        return float((self._f*(T2[a,b]+Tc[a,b,c])).sum())
    def ms_per_char(self, layout):
        return self.fitness(layout)/self._covered

START = S.C30M
sc = ReportedGaugeScorer(START)
# PARITY GATE: my scorer must equal analyze's ms/char on known boards, or it is not the gauge.
import json as _j
V=_j.load(open("/tmp/normopt/runs/verdict.json"))
print("PARITY GATE — my ReportedGaugeScorer vs shipped `keybo analyze` ms/char:")
worst=0.0
for nm in ("keybo-lsb","keybo-c30m","arm-B","graphite","qwerty30m"):
    lay=V["field"][nm]["layout"]; want=V["field"][nm]["ms"]
    got=sc.ms_per_char(Layout(lay,G)); rel=abs(got-want)/want; worst=max(worst,rel)
    print(f"  {nm:12} mine {got:.9f}  analyze {want:.9f}  rel {rel:.3e}")
assert worst < 1e-9, f"PARITY FAILED, worst rel {worst:.3e} — not the reported gauge"
print(f"  PARITY PASS (worst rel {worst:.3e})\n")

out={}
for seed in range(10):
    t0=time.time()
    lay=Layout(START,G)
    sa=SimulatedAnnealing(seed=seed, alpha=0.999, max_outer=None, progress=False)
    best=sa.optimize(lay,sc)
    best=two_opt(best,sc)
    txt="".join(best.chars)
    out[f"A2-s{seed}"]={"layout":txt,"ms_per_char":sc.ms_per_char(best),"seed":seed,
                       "secs":time.time()-t0}
    print(f"  A2-s{seed}: {txt!r}  ms/char {sc.ms_per_char(best):.6f}  ({time.time()-t0:.1f}s)")
json.dump(out, open("/tmp/normopt/runs/armA2.json","w"), indent=1, sort_keys=True)
print("\nwrote armA2.json")
