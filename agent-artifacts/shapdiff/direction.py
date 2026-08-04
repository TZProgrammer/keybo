"""Does the MODEL actually price bottom-row landings as slow? (direction check on the
dominant channel) + confirm lateral/dx are landing-key/geometry quantities, + the
frame's blind spots stated as measured facts."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="2"
import keybo, numpy as np
print("keybo:",keybo.__file__)
from keybo.analysis.shap_diff import _bigram_shap_tables
from keybo.analysis.timecard import _SEEDS,_load_gz_model
from keybo.features import BIGRAM_FEATURE_NAMES, bigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
geom=ROW_STAGGERED_30
models=[_load_gz_model(f"bigram_reg31_seed{s}") for s in _SEEDS]
shap_t,p_t,pp_t,ms_t,worst,names=_bigram_shap_tables(models,geom,90.0)
shap=np.mean(shap_t,axis=0); ms=np.mean(ms_t,axis=0)
positions=[*geom.slots,geom.space_position]
X=np.vstack([bigram_features_from_positions(geom,(a,b),wpm=90.0) for a in positions for b in positions])
n=len(positions)
for feat in ("bottom","home","top","lateral","same_finger","scissor","lsb"):
    col=names.index(feat); v=X[:,col].reshape(n,n); s=shap[:,:,col]
    on=v==1.0; off=v==0.0
    print(f"{feat:<12} mean SHAP where=1: {s[on].mean():+.5f} logs | where=0: {s[off].mean():+.5f} "
          f"| cells on: {on.sum():4d} | mean ms on {ms[on].mean():7.2f} off {ms[off].mean():7.2f}")
print("\n=> a POSITIVE mean SHAP where the one-hot is 1 means the model prices that class SLOW.")
# swap-invariance of inwards/outwards, MEASURED not cited
print("\n=== frame blind spots, measured over all 870 ordered slot pairs ===")
pairs=[(i,j) for i in range(30) for j in range(30) if i!=j]
for feat in ("inwards","outwards","same_finger","lsb","scissor","dx","distance","angle","dy"):
    col=names.index(feat); v=X[:,col].reshape(n,n)
    changed=sum(1 for i,j in pairs if v[i,j]!=v[j,i])
    print(f"  {feat:<10} changes under pair REVERSAL in {changed:3d} / 870 ordered pairs")
# hand identity: is any column a function of hand?
print("\n  hand-identity channel: BIGRAM_FEATURE_NAMES =",list(BIGRAM_FEATURE_NAMES))
print("  -> no L/R column; `same_hand` is relational only, finger one-hots are hand-agnostic.")
