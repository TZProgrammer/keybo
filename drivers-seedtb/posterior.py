"""Does the enlarged sample change BESTFINAL-1's ~85/15 flat-prior reading for arm-B?

BESTFINAL-1 computed one-sided p(margin<=0)=0.1495 from n=3 and read it as ~85/15. Recompute
the same quantity at n=15 for all 4 arm-B pairs, and add the honest Bayesian version (a flat
prior on (mu,log sigma) gives a t posterior for mu, so P(mu<0|data) = the one-sided t p-value
mirrored -- they coincide numerically, which is worth stating explicitly).
Also: was the 3-seed spread REPRESENTATIVE? Compare sd(n=3) to sd(n=15) per pair.
"""
import json, sys
MY_WT="/local/home/zegertho/agent/workspaces/seedtb/wt"; sys.path.insert(0, MY_WT+"/src")
import keybo; assert keybo.__file__.startswith(MY_WT)
import numpy as np
from scipy import stats

d=json.load(open("/local/home/zegertho/agent/state/seedtb/artifacts/margins_n15.json"))
mspc=d["mspc"]; NAMES=["arm-B","F(2.5)","BALL-1","F(2.0)","candidate"]
print("=== Is arm-B FASTER? P(mu_margin < 0 | data), flat prior  [margin = arm-B - other] ===")
print("(negative margin = arm-B faster. BESTFINAL-1 read 85/15 for arm-B from n=3.)")
print(f"{'pair':<22}{'n=3 P(armB faster)':>20}{'n=15 P(armB faster)':>21}{'sd(3)':>9}{'sd(15)':>9}{'mean(3)':>10}{'mean(15)':>10}")
out={}
for other in NAMES[1:]:
    a=np.array(mspc["arm-B"]); b=np.array(mspc[other]); dd=a-b
    row=[]
    for k in (3,15):
        x=dd[:k]; m=x.mean(); s=x.std(ddof=1); t=m/(s/np.sqrt(k))
        # P(mu<0|data) under flat prior == cdf of t posterior at 0
        p_faster=float(stats.t.cdf(-m/(s/np.sqrt(k)), df=k-1)) if s>0 else float(m<0)
        p_faster=float(stats.t.cdf(0, df=k-1, loc=m, scale=s/np.sqrt(k)))
        row.append((p_faster,m,s))
    print(f"{'arm-B vs '+other:<22}{row[0][0]:>20.4f}{row[1][0]:>21.4f}"
          f"{row[0][2]:>9.4f}{row[1][2]:>9.4f}{row[0][1]:>+10.4f}{row[1][1]:>+10.4f}")
    out[other]={"p_armB_faster_n3":row[0][0],"p_armB_faster_n15":row[1][0],
                "sd_n3":row[0][2],"sd_n15":row[1][2],"mean_n3":row[0][1],"mean_n15":row[1][1]}

print("\n=== Was the 3-seed sd representative? (ratio sd(15)/sd(3), and the sd's own 95% CI at n=3) ===")
for other in NAMES[1:]:
    s3=out[other]["sd_n3"]; s15=out[other]["sd_n15"]
    lo=s3*np.sqrt(2/stats.chi2.ppf(0.975,2)); hi=s3*np.sqrt(2/stats.chi2.ppf(0.025,2))
    inside = lo<=s15<=hi
    print(f"  arm-B vs {other:<11} sd15/sd3={s15/s3:>5.2f}   n=3 sd CI [{lo:.4f},{hi:.4f}] "
          f"contains sd15? {'yes' if inside else 'NO'}")

print("\n=== SIGN-AGREEMENT at n=15 (the 'do seeds even agree' question, no test) ===")
for i in range(5):
    for j in range(i+1,5):
        x,y=NAMES[i],NAMES[j]
        dd=np.array(mspc[x])-np.array(mspc[y]); pos=int((dd>0).sum())
        print(f"  {x+' vs '+y:<24} {pos:>2}/{len(dd)} positive  "
              f"{'UNANIMOUS' if pos in (0,len(dd)) else ''}")

print("\n=== THE FIVE BOARDS' MEAN ms/char at n=15 vs n=3 (does the ORDER change?) ===")
print(f"{'board':<11}{'mean n=3':>13}{'mean n=15':>13}{'rank n=3':>10}{'rank n=15':>11}")
m3={n:float(np.mean(mspc[n][:3])) for n in NAMES}; m15={n:float(np.mean(mspc[n])) for n in NAMES}
r3={n:i+1 for i,n in enumerate(sorted(NAMES,key=lambda q:m3[q]))}
r15={n:i+1 for i,n in enumerate(sorted(NAMES,key=lambda q:m15[q]))}
for n in sorted(NAMES,key=lambda q:m15[q]):
    print(f"{n:<11}{m3[n]:>13.4f}{m15[n]:>13.4f}{r3[n]:>10}{r15[n]:>11}"
          f"{'   <-- RANK MOVED' if r3[n]!=r15[n] else ''}")
json.dump(out,open("/local/home/zegertho/agent/state/seedtb/artifacts/posterior_n15.json","w"),indent=1)
