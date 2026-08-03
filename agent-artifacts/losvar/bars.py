import os,sys,json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v,"48")
WT="/local/home/zegertho/repos/keybo-wt-losvar"; sys.path.insert(0,WT+"/src")
import numpy as np, keybo
assert keybo.__file__.startswith(WT+"/"), keybo.__file__
print("keybo:",keybo.__file__)
from keybo.analysis.los import compute_los, split_half_floor
tj=json.load(open('/local/home/zegertho/agent/state/tournament/artifacts/tournament.json'))
d1=json.load(open('/local/home/zegertho/agent/state/losvar/artifacts/v01_decompose.json'))
B=list(tj['boards']); BOOT=20260803
P=d1['decomposition']['variants']
SIG_SB=P['scoring_bucket_80']['pairs']['candidate vs flagship-c3']['De_delev_rms']
SIG_AB=P['all_buckets']['pairs']['candidate vs flagship-c3']['De_delev_rms']
SIG_QW=P['scoring_bucket_80']['pairs']['candidate vs qwerty']['De_delev_rms']
panel=np.vstack([np.array(tj['mspc']['all'][b],float) for b in B])
FL=split_half_floor(panel,n_partitions=2000,rng=np.random.default_rng(BOOT),pct=90.0)['floor']
print(f"floor(all) {FL:.4f}  sigma_sb {SIG_SB:.4f}  sigma_ab {SIG_AB:.4f}  sigma_qwerty {SIG_QW:.4f}")
out={"floor_all":FL,"sigma_scoring_bucket":SIG_SB,"sigma_all_buckets":SIG_AB,"boot_seed":BOOT}

# ---- null-1: board vs itself, exact 0.5 at every sigma
n1=[]
for b in B:
    ms=np.array(tj['mspc']['all'][b],float)
    for s in (0.0,SIG_SB,SIG_AB,1.0,5.0,50.0):
        r=compute_los(ms,ms.copy(),floor=FL,a_name=b,b_name=b,sigma_diff=s)
        n1.append(abs(r.los_valid-0.5))
w1=max(n1)
out["null_1"]={"n_cases":len(n1),"worst_abs_dev":w1,"bar":"= 0.5000 exactly","pass":bool(w1==0.0)}
print(f"null-1: {len(n1)} cases, worst |LOS_valid-0.5| = {w1:.3e} => {'PASS' if w1==0.0 else 'FAIL'}")

# ---- null-2: same-board split-half (truth is 0 by construction)
for tag,SIG in (("scoring_bucket",SIG_SB),("all_buckets",SIG_AB)):
    rng=np.random.default_rng(BOOT); vals=[]; dec=0; N=2000
    for _ in range(N):
        ms=np.array(tj['mspc']['all'][B[rng.integers(len(B))]],float)
        perm=rng.permutation(ms.size); h=ms.size//2
        r=compute_los(ms[perm[:h]],ms[perm[h:2*h]],floor=FL,sigma_diff=SIG)
        vals.append(r.los_valid); dec+=int(r.los_valid>=0.95 or r.los_valid<=0.05)
    vals=np.array(vals); med=float(np.median(vals)); dr=dec/N
    ok=bool(0.45<=med<=0.55 and dr<=0.05)
    out[f"null_2_{tag}"]={"n":N,"sigma":SIG,"median":med,"decided_rate":dr,
        "bar":"median in [0.45,0.55] and decided-rate <= 0.05","pass":ok}
    print(f"null-2 ({tag}, sigma={SIG:.4f}): median {med:.4f} decided-rate {dr:.4f} => {'PASS' if ok else 'FAIL'}")

# ---- null-3: permutation null on the LIVE pair (sign-flip => truth 0)
d=np.array(tj['mspc']['all']['candidate'],float)-np.array(tj['mspc']['all']['flagship-c3'],float)
base=np.array(tj['mspc']['all']['flagship-c3'],float)
for tag,SIG in (("scoring_bucket",SIG_SB),("all_buckets",SIG_AB)):
    rng=np.random.default_rng(BOOT); N=20000; v=np.empty(N)
    for i in range(N):
        v[i]=compute_los(base+d*rng.choice((-1.0,1.0),size=d.size),base,floor=FL,sigma_diff=SIG).los_valid
    pg=float((v>0.95).mean()); ok=bool(pg<=0.05)
    out[f"null_3_{tag}"]={"n":N,"sigma":SIG,"median":float(np.median(v)),"p_gt_0.95":pg,
        "p_lt_0.05":float((v<0.05).mean()),"bar":"P(LOS_valid>0.95) <= 0.05","pass":ok}
    print(f"null-3 ({tag}, sigma={SIG:.4f}): median {np.median(v):.4f} P(LOS>0.95)={pg:.5f} => {'PASS' if ok else 'FAIL'}")

# ---- known-big: every tuned board vs qwerty, all 3 pricings, using EACH PAIR's own sigma
TUNED=[b for b in B if b not in ("qwerty","dvorak","colemak","colemak-dh","graphite","semimak")]
for tag,vn in (("scoring_bucket","scoring_bucket_80"),("all_buckets","all_buckets")):
    kb=[]
    for pr in ("all","observed","common"):
        pan=np.vstack([np.array(tj['mspc'][pr][b],float) for b in B])
        fl=split_half_floor(pan,n_partitions=2000,rng=np.random.default_rng(BOOT),pct=90.0)['floor']
        for b in TUNED:
            pairs=P[vn]['pairs']; k=f"{b} vs qwerty" if f"{b} vs qwerty" in pairs else f"qwerty vs {b}"
            if k not in pairs: continue
            s=pairs[k]['De_delev_rms']
            r=compute_los(np.array(tj['mspc'][pr][b],float),np.array(tj['mspc'][pr]['qwerty'],float),
                          floor=fl,a_name=b,b_name='qwerty',sigma_diff=s)
            kb.append({"board":b,"pricing":pr,"sigma":s,"margin":r.mean_margin,
                       "m_over_floor":r.margin_over_floor,"LOS_design":r.los_design,"LOS_valid":r.los_valid})
    if kb:
        mn=min(x["LOS_valid"] for x in kb); ok=bool(mn>=0.99)
        out[f"known_big_{tag}"]={"n_cases":len(kb),"min_LOS_valid":mn,
            "min_LOS_design":min(x["LOS_design"] for x in kb),"bar":">= 0.99","pass":ok,"cases":kb}
        worst=min(kb,key=lambda x:x["LOS_valid"])
        print(f"known-big ({tag}): {len(kb)} cases, min LOS_valid {mn:.4f} "
              f"(worst {worst['board']}/{worst['pricing']}, sigma {worst['sigma']:.3f}, "
              f"{worst['m_over_floor']:.1f}x floor) => {'PASS' if ok else 'FAIL'}")
json.dump(out,open('/local/home/zegertho/agent/state/losvar/artifacts/v03_bars.json','w'),indent=1,default=float)
print("\nwrote v03_bars.json")
