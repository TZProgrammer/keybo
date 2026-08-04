"""The PRE-REGISTERED corpus-robustness rule, evaluated as registered (SHAPDIFF-TCOND §5).

Rule, fixed in the ledger before any number existed:
  (i)   the SIGN of every BLOCK in the top-3 by |contribution| agrees across corpora, AND
  (ii)  Spearman rho over the 5 BLOCK contributions >= 0.90, AND
  (iii) Spearman rho over all 46 COLUMN contributions >= 0.90.
Block-level pass with column-level fail is a legitimate, informative outcome and is reported
as exactly that -- not laundered into a single "robust".
"""
import os, json, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import subprocess, pathlib
import numpy as np, keybo
print("keybo.__file__ =", keybo.__file__)
root = pathlib.Path(keybo.__file__).resolve().parents[2]
print("checkout =", root, "branch =", subprocess.run(["git","-C",str(root),"rev-parse","--abbrev-ref","HEAD"],capture_output=True,text=True).stdout.strip())
from scipy.stats import spearmanr

A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
for p, nm in ((A,"blend-v1"),(B,"iweb")):
    assert "channels" in p and "tcond" in p["channels"], f"KEY MISSING in {nm}: channels.tcond"
    assert p["channels"]["tcond"]["reconciles"] is True, f"{nm} tcond DID NOT RECONCILE"

for chan in ("tcond","t2"):
    a, b = A["channels"][chan], B["channels"][chan]
    print(f"\n{'='*78}\n=== CHANNEL {chan.upper()}  blend-v1 gap {a['gap_decomposed']:+.4f}   iweb gap {b['gap_decomposed']:+.4f}")
    ab = {x["block"]: x["ms_per_char"] for x in a["blocks"]}
    bb = {x["block"]: x["ms_per_char"] for x in b["blocks"]}
    blocks = sorted(ab, key=lambda k: -abs(ab[k]))
    print(f"\n{'block':<12} {'blend-v1':>10} {'share%':>8} {'iweb':>10} {'share%':>8} {'sign':>6}")
    for k in blocks:
        sg = "AGREE" if (ab[k]>0)==(bb[k]>0) else "FLIP"
        print(f"{k:<12} {ab[k]:>+10.4f} {100*ab[k]/a['gap_decomposed']:>7.1f}% {bb[k]:>+10.4f} {100*bb[k]/b['gap_decomposed']:>7.1f}% {sg:>6}")
    top3 = blocks[:3]
    i_pass = all((ab[k]>0)==(bb[k]>0) for k in top3)
    rho_b = spearmanr([ab[k] for k in blocks], [bb[k] for k in blocks]).statistic
    ac = {x["feature"]: x["ms_per_char"] for x in a["contributions"]}
    bc = {x["feature"]: x["ms_per_char"] for x in b["contributions"]}
    cols = sorted(ac, key=lambda k: -abs(ac[k]))
    rho_c = spearmanr([ac[k] for k in cols], [bc[k] for k in cols]).statistic
    print(f"\n(i)   top-3 BLOCK sign agreement {top3}: {'PASS' if i_pass else 'FAIL'}")
    print(f"(ii)  BLOCK  Spearman rho over {len(blocks)} blocks : {rho_b:.4f}   {'PASS' if rho_b>=0.90 else 'FAIL'} (bar 0.90)")
    print(f"(iii) COLUMN Spearman rho over {len(cols)} columns: {rho_c:.4f}   {'PASS' if rho_c>=0.90 else 'FAIL'} (bar 0.90)")
    verdict = "ROBUST" if (i_pass and rho_b>=0.90 and rho_c>=0.90) else "NOT robust as registered"
    print(f"==> REGISTERED VERDICT for {chan}: {verdict}")
    flip = [k for k in cols if (ac[k]>0)!=(bc[k]>0)]
    print(f"\nsign FLIPPERS among columns ({len(flip)}): ")
    for k in flip:
        print(f"   {k:<20} blend-v1 {ac[k]:+.4f} ({100*ac[k]/a['gap_decomposed']:+.2f}% of gap)  ->  iweb {bc[k]:+.4f} ({100*bc[k]/b['gap_decomposed']:+.2f}%)")
    tot = sum(abs(ac[k]) for k in flip)
    print(f"   flippers' total |share| of the blend-v1 channel gap: {100*tot/abs(a['gap_decomposed']):.2f}%")
    print(f"   largest flipper by |blend-v1 share|: {max(flip, key=lambda k: abs(ac[k])) if flip else '(none)'}")
    print(f"\ntop-8 columns, both corpora:")
    print(f"{'feature':<20} {'blend-v1':>10} {'rank':>5} {'iweb':>10} {'rank':>5}")
    rb = {k:i+1 for i,k in enumerate(cols)}
    rc = {k:i+1 for i,k in enumerate(sorted(bc, key=lambda k:-abs(bc[k])))}
    for k in cols[:8]:
        print(f"{k:<20} {ac[k]:>+10.4f} {rb[k]:>5} {bc[k]:>+10.4f} {rc[k]:>5}")
