import sys, json, numpy as np
sys.path.insert(0,"/tmp/quadgram-wt/src")
A=json.load(open("/tmp/quad_eval_ckpt/arm_A_quad_full.json"))
B=json.load(open("/tmp/quad_eval_ckpt/arm_B_quad_trictx.json"))
FOLDS=["azerty","dvorak","qwerty","qwertz"]; SEEDS=[0,1,2]
def rho_map(rep):
    m={}; bk={}
    for h,fold in rep["folds"].items():
        m[h]={}; bk[h]={}
        for rec in fold["seeds"]:
            m[h][rec["seed"]]=rec["rho"]; bk[h][rec["seed"]]={int(k):v for k,v in rec["bucket_rhos"].items()}
    return m,bk
ra,ba=rho_map(A); rb,bb=rho_map(B)
def mfc(rep):
    fr=[m["rho_frac_ceiling"] for f in rep["folds"].values() for m in f["seeds"] if m["rho_frac_ceiling"] is not None]
    return np.mean(fr) if fr else float("nan")
def mean_key(rep,k):
    v=[m[k] for f in rep["folds"].values() for m in f["seeds"] if k in m and m[k] is not None]
    return np.mean(v) if v else float("nan")
print("=== ARM LEVELS (mean over 4 folds x 3 seeds) ===")
for nm,rep in [("A QUAD-FULL (72col)",A),("B QUAD-TRICTX (46col=trigram-on-last3)",B)]:
    print(f"  {nm:42s} rho/ceil={mfc(rep):.4f}  rho={mean_key(rep,'rho'):.4f}  wmae={mean_key(rep,'wmae'):.4f}  umae={mean_key(rep,'umae'):.4f}")
    print(f"      pooled tau_heldout={[round(p['tau_heldout'],4) for p in rep['pooled']]}  ceilings={ {k:round(v,3) for k,v in rep['ceilings'].items()} }")
print("\n=== A - B PAIRED PER-FOLD DELTAS (MOR-FIX-1) ===")
alld=[]; consist_win=0
for h in FOLDS:
    sd=[ra[h][s]-rb[h][s] for s in SEEDS if ra[h].get(s) is not None and rb[h].get(s) is not None]
    alld+=sd
    signs=[np.sign(d) for d in sd]
    cons = all(x>0 for x in signs) or all(x<0 for x in signs)
    if cons and sd[0]>0: consist_win+=1
    print(f"  {h:8s} mean {np.mean(sd):+.6f}  seeds {[round(d,5) for d in sd]}  {'CONSISTENT' if cons else 'MIXED'} {'A>B' if np.mean(sd)>0 else 'A<B'}")
W=sum(1 for d in alld if d>0); L=sum(1 for d in alld if d<0)
print(f"\n  overall: mean paired delta {np.mean(alld):+.6f}, W/L {W}/{L}, sign-consistent winning folds for A: {consist_win}/4")
print(f"  CRITERION (a): mean>0 AND >=3/4 folds consistent  => {'PASS' if (np.mean(alld)>0 and consist_win>=3) else 'FAIL'}")
# high-wpm A vs B
from keybo.verdicts import HIGH_WPM_FLOOR, HIGH_WPM_TOLERANCE
print(f"\n=== HIGH-WPM GATE (A vs B, floor {HIGH_WPM_FLOOR}, tol {HIGH_WPM_TOLERANCE}) ===")
structural=[]
for h in FOLDS:
    counts={}; n=0
    for s in SEEDS:
        a=ba[h].get(s,{}); b=bb[h].get(s,{})
        if not a or not b: continue
        n+=1
        for bk_ in sorted(b):
            if bk_<HIGH_WPM_FLOOR or bk_ not in a: continue
            if a[bk_]-b[bk_] < -HIGH_WPM_TOLERANCE: counts[bk_]=counts.get(bk_,0)+1
    st=sorted(k for k,v in counts.items() if v==n and n>0)
    if st: structural.append(f"{h}{st}")
    print(f"  {h:8s} n_seeds={n} structural={st} counts={dict(sorted(counts.items()))}")
print(f"  CRITERION (b) high-wpm: {'PASS (no structural regression)' if not structural else 'FAIL '+str(structural)}")
