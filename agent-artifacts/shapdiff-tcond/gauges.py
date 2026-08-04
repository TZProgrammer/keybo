"""GAUGE CROSS-CHECK (SHAPDIFF-TCOND §4): the four registered expectations, plus the shipped
gauges recomputed HERE so the comparison does not rest on numbers handed to me.

Nothing in the shap_diff pipeline reads a gauge, so agreement is real corroboration.
"""
import os, json, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import subprocess, pathlib
import numpy as np, keybo
print("keybo.__file__ =", keybo.__file__)
root = pathlib.Path(keybo.__file__).resolve().parents[2]
print("checkout =", root, "branch =", subprocess.run(["git","-C",str(root),"rev-parse","--abbrev-ref","HEAD"],capture_output=True,text=True).stdout.strip())

F="pyou'vgdnmheai.cstrlkjz,-wfbxq"; G="bldwz'foujnrtsgyhaeixqmcvkp,.-"
A=json.load(open(sys.argv[1])); B=json.load(open(sys.argv[2]))
ca={x["feature"]:x["ms_per_char"] for x in A["channels"]["tcond"]["contributions"]}
cb={x["feature"]:x["ms_per_char"] for x in B["channels"]["tcond"]["contributions"]}
gA=A["channels"]["tcond"]["gap_decomposed"]; gB=B["channels"]["tcond"]["gap_decomposed"]

# --- recompute the gauges MYSELF (parent handed me numbers; verify, don't anchor) --------
from keybo.analysis.kmstats import KmStats
from keybo.analysis.redirects import RedirectFamily
from keybo.data.corpus import load_frequencies, production_corpus_dir
d = production_corpus_dir(None)
bi = load_frequencies(str(d/"bigrams.txt")); tri = load_frequencies(str(d/"trigrams.txt"))
sk = load_frequencies(str(d/"1-skip.txt"))
km = KmStats(bi, sk, tri)
sF, sG = km.stats(F), km.stats(G)
print("\n=== gauges RECOMPUTED here (parent's brief numbers in brackets) ===")
brief = {"sfb":(1.654,1.526),"redir":(2.494,3.213),"scissor":(0.089,0.517),"alt":(45.156,44.076),"roll":(41.761,42.466)}
keys = sorted(set(sF) & set(sG))
print(f"{'gauge':<22} {'flagship-c3':>12} {'graphite':>12} {'better':>12}   brief")
for k in keys:
    vF, vG = sF[k], sG[k]
    if not isinstance(vF,(int,float)): continue
    bnote = ""
    for bk,(bF,bG) in brief.items():
        if k==bk or k.startswith(bk):
            bnote = f"  [{bF} / {bG}]" + ("  MATCH" if abs(vF-bF)<0.01 and abs(vG-bG)<0.01 else "  <-- DIFFERS")
    print(f"{k:<22} {vF:>12.4f} {vG:>12.4f} {('flagship' if vF<vG else 'graphite'):>12}{bnote}")
rs = RedirectFamily(tri)
rF, rG = rs.shares(F), rs.shares(G)
print("\n=== the redirect FAMILY split (analysis/redirects.py) ===")
print(f"{'class':<26} {'flagship-c3':>12} {'graphite':>12} {'better':>12}")
for k in sorted(set(rF)&set(rG)):
    print(f"{k:<26} {rF[k]:>12.4f} {rG[k]:>12.4f} {('flagship' if rF[k]<rG[k] else 'graphite'):>12}")

# --- the four registered expectations ---------------------------------------------------
def grp(c, cols): return sum(c[k] for k in cols)
print("\n=== the FOUR REGISTERED EXPECTATIONS (attribution sign: + favours flagship-c3) ===")
tests = [
  ("G1 redirect family FOR flagship (redir 2.494 vs 3.213)",
   ["redirect","bad_redirect"], "flagship"),
  ("G2 SKIPGRAM FOR flagship (sg_dist 3.968 vs 4.031)",
   ["sg_same_finger","sg_dx","sg_dy","sg_distance"], "flagship"),
  ("G3 bg*_same_finger AGAINST flagship (sfb 1.654 vs 1.526)",
   ["bg1_same_finger","bg2_same_finger"], "graphite"),
  ("G4 bg*_scissor FOR flagship (0.089 vs 0.517)",
   ["bg1_scissor","bg2_scissor"], "flagship"),
]
for label, cols, want in tests:
    vA, vB = grp(ca,cols), grp(cb,cols)
    gotA = "flagship" if vA>0 else ("graphite" if vA<0 else "tie")
    gotB = "flagship" if vB>0 else ("graphite" if vB<0 else "tie")
    verdict = "CONFIRMED" if gotA==want and gotB==want else ("DISAGREEMENT" if gotA!=want and gotB!=want else "SPLIT across corpora")
    print(f"\n{label}")
    print(f"   blend-v1 {vA:+.4f} ({100*vA/gA:+.2f}% of gap) -> favours {gotA};  iweb {vB:+.4f} ({100*vB/gB:+.2f}%) -> {gotB}")
    print(f"   expected {want}  ==>  {verdict}")
    for k in cols:
        print(f"      {k:<20} blend-v1 {ca[k]:+.4f}   iweb {cb[k]:+.4f}")
