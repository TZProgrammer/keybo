"""Structural facts I must verify myself before the dossier rests on them."""
import sys, numpy as np
sys.path.insert(0,'/tmp/scissorprice/probe')
import matched_prices as M
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.features import bigram_features_from_positions
from keybo.features.schema import BIGRAM_FEATURE_NAMES
pairs=[(a,b) for a in M.SLOTS for b in M.SLOTS if a!=b]

print("=== S1. roll <=> same-hand 2-finger AND different ROW (structural) ===")
def roll(ab): return C.is_inwards(G,*ab) or C.is_outwards(G,*ab)
def shb(ab): return M.shb(ab)
bad=[ab for ab in pairs if shb(ab) and (roll(ab) != (ab[0][1]!=ab[1][1]))]
print(f"  counterexamples to 'shb: roll <=> rowspan>0': {len(bad)}  -> {'STRUCTURAL' if not bad else 'NOT structural'}")
print(f"  => matching a roll contrast on rowspan is DISJOINT BY CONSTRUCTION (trap 16, structural kind)")

print("\n=== S2. THEORY-1 D3: is_inwards/is_outwards are SWAP-INVARIANT (order-invariant) ===")
n_in=sum(1 for ab in pairs if C.is_inwards(G,*ab)); n_out=sum(1 for ab in pairs if C.is_outwards(G,*ab))
rev_same_in =sum(1 for a,b in pairs if C.is_inwards(G,a,b) and C.is_inwards(G,b,a))
rev_same_out=sum(1 for a,b in pairs if C.is_outwards(G,a,b) and C.is_outwards(G,b,a))
print(f"  is_inwards fires on {n_in} ordered pairs; {rev_same_in}/{n_in} have their REVERSE in the SAME class")
print(f"  is_outwards fires on {n_out} ordered pairs; {rev_same_out}/{n_out} likewise")
unord_in=len({frozenset((a,b)) for a,b in pairs if C.is_inwards(G,a,b)})
print(f"  {n_in} ordered pairs span only {unord_in} UNORDERED pairs  (THEORY-1 said 108 over 54)")

print("\n=== S3. served BIGRAM feature vector: max |non-landing feature diff| under swap ===")
names=list(BIGRAM_FEATURE_NAMES); print(f"  {len(names)} features: {names}")
LAND_PREFIX=('index','middle','ring','pinky','top','home','bottom','lateral')
land_ix=[i for i,n in enumerate(names) if n in LAND_PREFIX]
nonland_ix=[i for i,n in enumerate(names) if n not in LAND_PREFIX]
print(f"  landing one-hots ({len(land_ix)}): {[names[i] for i in land_ix]}")
print(f"  non-landing ({len(nonland_ix)}): {[names[i] for i in nonland_ix]}")
worst=0.0; worst_which=None
for a,b in pairs:
    fa=np.asarray(bigram_features_from_positions(G,(a,b),90.0),dtype=float)
    fb=np.asarray(bigram_features_from_positions(G,(b,a),90.0),dtype=float)
    d=np.abs(fa[nonland_ix]-fb[nonland_ix])
    if d.max()>worst: worst=d.max(); worst_which=(a,b,[names[nonland_ix[i]] for i in np.argsort(-d)[:2]])
print(f"  MAX |non-landing feature diff| under swap over all {len(pairs)} ordered pairs = {worst:.6e}")
print(f"  => an inroll and its reverse outroll over the SAME key pair are the SAME MODEL INPUT")
print(f"     except through the landing one-hots. ORDER-DEPENDENCE OF THE BIGRAM VECTOR: {'NONE' if worst==0 else 'PRESENT'}")
# and the angle/inwards/outwards columns specifically
for nm in ('angle','inwards','outwards'):
    if nm in names:
        i=names.index(nm)
        w=max(abs(bigram_features_from_positions(G,(a,b),90.)[i]-bigram_features_from_positions(G,(b,a),90.)[i]) for a,b in pairs)
        print(f"     max|{nm}(a,b) - {nm}(b,a)| = {w:.3e}")
