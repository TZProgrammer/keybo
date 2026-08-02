"""WHY do some pairs resolve and others not? Test the within-family vs cross-family hypothesis.

Observation driving this: the 5 boards fall into TWO near-clone families by string similarity.
Hypothesis: seed noise is CORRELATED between similar boards, so it cancels in within-family
differences (small sd) but not across families (large sd). If true, the tie-break resolves
easily among near-clones and is hard exactly where adoption needs it (across families).
"""
import json, sys, itertools
MY_WT="/local/home/zegertho/agent/workspaces/seedtb/wt"; sys.path.insert(0,MY_WT+"/src")
import keybo; assert keybo.__file__.startswith(MY_WT)
import numpy as np
from scipy import stats
ART="/local/home/zegertho/agent/state/seedtb/artifacts"
d=json.load(open(f"{ART}/margins_n25.json")); mspc=d["mspc"]
BOARDS={"arm-B":"flmpg-yuo,sntdcireahkxbwv'.jzq","F(2.5)":"flmpg-,uoysntdcireahkxbwv.'jzq",
        "BALL-1":"flmpg-yuo,sntcdireahkxbwv'.jzq","F(2.0)":"pyu.,gdfnlhieaocstrmkj'-qbwzvx",
        "candidate":"pyu.,vdfnlhieaocstrmkj'-qgwbzx"}
SFB={"arm-B":2.5391,"F(2.5)":2.3521,"BALL-1":2.5391,"F(2.0)":1.8677,"candidate":1.7365}
NAMES=list(BOARDS)
def ham(a,b): return sum(1 for x,y in zip(a,b) if x!=y)
print("=== BOARD-TO-BOARD HAMMING DISTANCE (chars in different slots) ===")
print("            " + "".join(f"{n:>11}" for n in NAMES))
for a in NAMES:
    print(f"{a:<12}" + "".join(f"{ham(BOARDS[a],BOARDS[b]):>11}" for b in NAMES))
print("\n=> two near-clone FAMILIES: {arm-B, F(2.5), BALL-1} (all 'flmpg-' prefix, pairwise")
print("   hamming 2-4) and {F(2.0), candidate} ('pyu.,' prefix, hamming 4). Cross-family = 28-30.")

print("\n=== PER-PAIR sd vs FAMILY RELATIONSHIP  (n=25) ===")
FAM={"arm-B":"A","F(2.5)":"A","BALL-1":"A","F(2.0)":"B","candidate":"B"}
print(f"{'pair':<24}{'hamming':>8}{'same fam?':>11}{'sd':>9}{'|mean|':>9}{'d/sd':>7}{'p':>11}{'n80':>7}")
rows=[]
def projn(dsd):
    if dsd<=0: return None
    for k in range(3,100001):
        nc=stats.t.ppf(0.975,k-1)
        if stats.nct.sf(nc,k-1,dsd*np.sqrt(k))+stats.nct.cdf(-nc,k-1,dsd*np.sqrt(k))>=0.80: return k
    return ">100000"
for x,y in itertools.combinations(NAMES,2):
    dd=np.array(mspc[x])-np.array(mspc[y]); m=dd.mean(); s=dd.std(ddof=1); nn=len(dd)
    t=m/(s/np.sqrt(nn)); p=float(2*stats.t.sf(abs(t),nn-1)); h=ham(BOARDS[x],BOARDS[y])
    same=FAM[x]==FAM[y]
    print(f"{x+' vs '+y:<24}{h:>8}{('YES' if same else 'no'):>11}{s:>9.4f}{abs(m):>9.4f}"
          f"{abs(m)/s:>7.2f}{p:>11.2e}{str(projn(abs(m)/s)):>7}")
    rows.append(dict(pair=f"{x} vs {y}",hamming=h,same_family=bool(same),sd=float(s),
                     absmean=float(abs(m)),dsd=float(abs(m)/s),p=p,n80=projn(abs(m)/s)))
w=[r["sd"] for r in rows if r["same_family"]]; c=[r["sd"] for r in rows if not r["same_family"]]
print(f"\nwithin-family sd: mean {np.mean(w):.4f}  (n={len(w)}, range {min(w):.4f}-{max(w):.4f})")
print(f"cross-family  sd: mean {np.mean(c):.4f}  (n={len(c)}, range {min(c):.4f}-{max(c):.4f})")
print(f"RATIO cross/within = {np.mean(c)/np.mean(w):.2f}x")
print(f"Mann-Whitney U on sd by family: p={stats.mannwhitneyu(w,c,alternative='less').pvalue:.4f}")
print(f"\nresolved (Holm-eligible raw p<0.05): within-family {sum(1 for r in rows if r['same_family'] and r['p']<0.05)}/{len(w)}"
      f"   cross-family {sum(1 for r in rows if not r['same_family'] and r['p']<0.05)}/{len(c)}")

print("\n=== THE MECHANISM: per-seed CORRELATION of board totals ===")
M=np.array([mspc[n] for n in NAMES])
C=np.corrcoef(M)
print("            " + "".join(f"{n:>11}" for n in NAMES))
for i,a in enumerate(NAMES):
    print(f"{a:<12}" + "".join(f"{C[i,j]:>11.5f}" for j in range(5)))
print("=> all boards move TOGETHER across seeds (r>0.99): the seed effect is a near-common shift.")
print("   Within-family r is higher still, so the difference cancels more completely.")

print("\n=== WHAT ADOPTION ACTUALLY NEEDS: the cross-family question ===")
A=["arm-B","F(2.5)","BALL-1"]; B=["F(2.0)","candidate"]
fa=np.mean([mspc[n] for n in A],axis=0); fb=np.mean([mspc[n] for n in B],axis=0)
dd=fa-fb; m=dd.mean(); s=dd.std(ddof=1); nn=len(dd); t=m/(s/np.sqrt(nn))
tc=stats.t.ppf(0.975,nn-1)
print(f"family-A mean vs family-B mean: {m:+.4f} ms/char  sd {s:.4f}  t {t:+.3f}  "
      f"p {2*stats.t.sf(abs(t),nn-1):.4f}")
print(f"  95% CI [{m-tc*s/np.sqrt(nn):+.4f}, {m+tc*s/np.sqrt(nn):+.4f}]   signs {int((dd>0).sum())}/{nn} positive")
print(f"  projected n for 80% power: {projn(abs(m)/s)}")
print(f"\nsfb: family A = {[SFB[n] for n in A]}, family B = {[SFB[n] for n in B]}")
print("=> the SPEED frame cannot separate the families, and family B is the LOW-sfb one.")
json.dump({"rows":rows,"within_sd_mean":float(np.mean(w)),"cross_sd_mean":float(np.mean(c)),
           "ratio":float(np.mean(c)/np.mean(w)),"corr":C.tolist(),"names":NAMES,
           "familyA_vs_B":{"mean":float(m),"sd":float(s),"t":float(t),
                           "p":float(2*stats.t.sf(abs(t),nn-1)),
                           "ci":[float(m-tc*s/np.sqrt(nn)),float(m+tc*s/np.sqrt(nn))],
                           "n80":projn(abs(m)/s)}},
          open(f"{ART}/family_structure.json","w"),indent=1)
