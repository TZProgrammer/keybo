"""CORRECTION to the in-domain/extrapolation split. My first pass clamped BOTH champion and
reference, and archive-1846 is itself out-of-domain on 9 of 14 gauges — so "100% extrapolation"
included the reference's own extrapolation and is not attributable to the SEARCH.

The right question: how much of arm A's advantage comes from moving FURTHER outside the domain
than the reference already was? Decompose delta into:
  d_within  = price(clamp(champ)) - price(clamp(ref))     <- the part a CLAMPED objective keeps
  d_beyond  = delta - d_within                            <- the part only extrapolation buys
and separately report how much of d_beyond is the reference's own OOD position vs the champ's.
"""
import sys, json
from pathlib import Path
sys.path.append("/local/home/zegertho/agent/state/optevidence/artifacts/drivers")
import numpy as np, evobj as EV
import scipy.stats as st
S=Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMJ="/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
J=json.load(open(S/"judgement.json")); pl=J["per_layout"]
W=json.load(open(ARMJ))["weights"]; shap={x["metric"]:x["shap_share_pct"] for x in W["weights"]}
fe=EV.FastEval(None,ARMJ,with_surface=True)
A="champ-evidence"
print("=== corrected split, computed against EVERY incumbent (reference-robust) ===")
allres={}
for R in ["archive-1846","keybo-lsb","lsb-sib","archive-1843","keybo-lsb+lm"]:
    perms=np.stack([EV.perm_of(pl[A]["layout"]),EV.perm_of(pl[R]["layout"])]); g=fe.gauges(perms)
    tw=tb=0.0; rows=[]
    for c in fe.curves:
        m=c.metric; xa,xr=float(g[m][0]),float(g[m][1])
        p=lambda x: float(c.price(np.array([x]))[0])
        d=p(xa)-p(xr)
        dw=p(float(np.clip(xa,*c.domain)))-p(float(np.clip(xr,*c.domain)))
        db=d-dw
        # attribute d_beyond: champ's own excursion vs the reference's
        champ_ex = p(xa)-p(float(np.clip(xa,*c.domain)))
        ref_ex   = p(xr)-p(float(np.clip(xr,*c.domain)))
        tw+=dw; tb+=db
        rows.append((m,d,dw,db,champ_ex,-ref_ex,shap[m]))
    net=tw+tb
    allres[R]={"net":net,"within":tw,"beyond":tb,"beyond_pct":100*tb/net}
    print("  ref=%-14s net %+8.4f | WITHIN-domain %+8.4f (%5.1f%%) | BEYOND-domain %+8.4f (%5.1f%%)"%(
        R,net,tw,100*tw/net,tb,100*tb/net))
    if R=="archive-1846":
        print("\n  per-gauge, ref=archive-1846:")
        print("  %-11s %8s %9s %9s %11s %11s %7s"%("gauge","delta","WITHIN","BEYOND","champ_excur","ref_excur(-)","shap%"))
        for m,d,dw,db,ce,re_,sh in sorted(rows,key=lambda r:r[1]):
            print("  %-11s %+8.4f %+9.4f %+9.4f %+11.4f %+11.4f %7.2f"%(m,d,dw,db,ce,re_,sh))
        print()
print("=== so: how much of arm A's SCORE advantage survives a CLAMP? ===")
for R,v in allres.items():
    print("  vs %-14s clamped advantage %+.4f of %+.4f raw = %.1f%% survives"%(R,v["within"],v["net"],100*v["within"]/v["net"]))
json.dump(allres,open(S/"extrapolation-split.json","w"),indent=1)
print("\nWROTE extrapolation-split.json")
