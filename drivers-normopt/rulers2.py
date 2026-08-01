"""Both ms rulers side by side. Arm A's SEARCH minimizes the bigram table; the campaign
REPORTS analyze's trigram ms/char. A winner must be judged on the ruler it optimized."""
import sys, json, statistics as st
sys.path.insert(0,"/tmp/normopt/src")
XS = json.load(open("/tmp/normopt/runs/crossscore.json"))
V  = json.load(open("/tmp/normopt/runs/verdict.json"))
FLOOR=0.135
P = V["produced"]
bigram = {k: XS["produced"][k]["ms_per_char"] for k in P}     # bigram-table ms/char (scale ~118)
tri    = {k: P[k]["ms"] for k in P}                            # analyze ms/char  (scale ~256)

print("="*104)
print("THE TWO ms RULERS — arm A's SEARCH minimizes ruler 1; the campaign/floor lives on ruler 2")
print("="*104)
print(f"{'run':7} {'ruler1 bigram-table':>20} {'ruler2 analyze-trigram':>23}")
for a in "ABC":
    ks=sorted([k for k in P if P[k]['arm']==a], key=lambda k:P[k]['seed'])
    for k in ks: print(f"{k:7} {bigram[k]:20.6f} {tri[k]:23.6f}")
    b=[bigram[k] for k in ks]; t=[tri[k] for k in ks]
    print(f"  arm {a}: ruler1 min {min(b):.6f} sd {st.stdev(b):.6f} | ruler2 min {min(t):.6f} sd {st.stdev(t):.6f}")

print("\n--- WINNER ON EACH RULER (best-of-10 per arm) ---")
for name,R in (("ruler1 bigram-table (what arm A's search minimizes)",bigram),
               ("ruler2 analyze-trigram (the campaign's published scale + the 0.135 floor)",tri)):
    print(f"\n{name}")
    best={a:min([k for k in P if P[k]['arm']==a], key=lambda k:R[k]) for a in "ABC"}
    for a in "ABC": print(f"   arm {a} best-on-this-ruler: {best[a]} = {R[best[a]]:.6f}")
    for a in "BC":
        d=R[best[a]]-R[best['A']]
        sdp=st.mean([st.stdev([R[k] for k in P if P[k]['arm']==x]) for x in "ABC"])
        extra = f" | {abs(d)/FLOOR:.2f}x floor" if R is tri else ""
        print(f"   {a} minus A = {d:+.6f} ({'A better' if d>0 else 'A worse'}){extra}  [pooled within-sd {sdp:.6f}, |d|/sd {abs(d)/sdp:.2f}]")

# P2 check: does each arm win on ITS OWN objective?
print("\n--- P2: does each arm win on ITS OWN objective? (registered check) ---")
print(f"{'objective':34} {'armA best':>12} {'armB best':>12} {'armC best':>12}   winner")
for lab,key,hi in (("ruler1 bigram-table ms/char","r1",False),
                   ("ruler2 analyze ms/char","r2",False),
                   ("normgauge blend registered(c)","bl_c",True),
                   ("normgauge blend 50/50","bl_50",True)):
    vals={}
    for a in "ABC":
        ks=[k for k in P if P[k]['arm']==a]
        if key=="r1": v=[bigram[k] for k in ks]
        elif key=="r2": v=[tri[k] for k in ks]
        else: v=[P[k][key] for k in ks]
        vals[a]=max(v) if hi else min(v)
    win=max(vals,key=lambda a:vals[a]) if hi else min(vals,key=lambda a:vals[a])
    print(f"{lab:34} {vals['A']:12.6f} {vals['B']:12.6f} {vals['C']:12.6f}   arm {win}")
