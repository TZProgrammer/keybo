"""POSITIVE CONTROL on the ARM G objective itself, BEFORE any search runs.
Every constant I hardcoded must equal what the live code computes, and the objective must
have the properties I registered."""
import json, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, evobj as EV, search as S
import keybo
assert str(Path(keybo.__file__).resolve()).startswith('/tmp/armg/')
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
assert str(Path(fe.corpus_dir).resolve()).startswith('/tmp/armg/')
ARMB="flmpg-yuo,sntdcireahkxbwv'.jzq"
P=json.load(open('/local/home/zegertho/agent/state/optevidence/artifacts/search-noise-placebo.json'))
six=[r['layout'] for r in P['runs']['baseline']]
G=tuple(S.ARMG_DIR)
res={}

# (1) hardcoded ARMG_REF == live arm B gauges?
g=fe.gauges(np.stack([EV.perm_of(ARMB)]))
w=0.0
for k in G: w=max(w,abs(float(g[k][0])-S.ARMG_REF[k]))
print(f"(1) ARMG_REF vs live arm B gauges: worst |diff| = {w:.3e}")
res['ref_worst']=w; assert w<1e-12, "HARDCODED REF DRIFTED"

# (2) hardcoded ARMG_SCALE == live range over the six frozen champions?
g6=fe.gauges(np.stack([EV.perm_of(l) for l in six]))
w2=0.0
for k in G: w2=max(w2,abs(float(g6[k].max()-g6[k].min())-S.ARMG_SCALE[k]))
print(f"(2) ARMG_SCALE vs live six-champion range: worst |diff| = {w2:.3e}")
res['scale_worst']=w2; assert w2<1e-12, "HARDCODED SCALE DRIFTED"

# (3) ARMG_REF_MS == live arm B ms/char?
d3=abs(float(g['_ms_per_char'][0])-S.ARMG_REF_MS)
print(f"(3) ARMG_REF_MS vs live arm B ms/char: |diff| = {d3:.3e}")
res['ref_ms_diff']=d3; assert d3<1e-11

# (4) D(arm B) == 0 EXACTLY (by construction)
D_armB=float(S.armg_deficit(g)[0])
print(f"(4) D(arm B) = {D_armB!r}   (must be exactly 0.0)")
res['D_armB']=D_armB; assert D_armB==0.0

# (5) the FULL objective at arm B == 0 (in band, D=0)
S._EVAL.update({"fe":fe,"arm":"armg","bounds":{}})
f,_=S._objective(np.stack([EV.perm_of(ARMB)]))
print(f"(5) F(arm B) = {float(f[0])!r}   (must be exactly 0.0)")
res['F_armB']=float(f[0]); assert float(f[0])==0.0

# (6) the D values I published in the prereg table reproduce from the live path
INC={"arm-A":"udy.,fgpmliheaocsntr-k'qjwzbvx","keybo-lsb":"pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
 "keybo-lsb+lm":"pyuo,vgdnmhiea.cstrlkj-z'fwbxq","flagship-c3":"pyou'vgdnmheai.cstrlkjz,-wfbxq",
 "graphite":"bldwz'foujnrtsgyhaeixqmcvkp,.-"}
PRE={"arm-A":0.4533,"keybo-lsb":2.1317,"keybo-lsb+lm":1.9092,"flagship-c3":1.4878,"graphite":3.2226}
print("(6) prereg D table vs live:")
w6=0.0
for n,l in INC.items():
    dv=float(S.armg_deficit(fe.gauges(np.stack([EV.perm_of(l)])))[0])
    w6=max(w6,abs(dv-PRE[n])); print(f"      {n:<13} live {dv:.4f}  prereg {PRE[n]:.4f}")
res['prereg_D_worst']=w6; assert w6<5e-5, w6
print(f"    worst |diff| = {w6:.3e} (prereg printed 4dp)")

# (7) THE PENALTY BITES: a fast-but-bad layout must score WORSE than an in-band one.
#     Without this the band is advisory (trap 51).
qw="qwertyuiopasdfghjkl'zxcvbnm,.-"
fq,_=S._objective(np.stack([EV.perm_of(qw)]))
gq=fe.gauges(np.stack([EV.perm_of(qw)]))
print(f"(7) qwerty: ms={float(gq['_ms_per_char'][0]):.4f} (OUT of band) D={float(S.armg_deficit(gq)[0]):.4f} F={float(fq[0]):.1f}")
assert float(fq[0])>1000.0, "penalty did NOT dominate"
res['F_qwerty']=float(fq[0])
# a layout exactly at the band edge pays 0; one EPS past it pays LAMBDA
print(f"    band edge = {S.ARMG_REF_MS+S.ARMG_EPS:.4f}; one EPS past edge costs LAMBDA={S.ARMG_LAMBDA}")

# (8) MUTATION CONTROL on the deficit: flipping one direction must change D(some layout)
sv=S.ARMG_DIR['scissor']; S.ARMG_DIR['scissor']=-sv
dmut=float(S.armg_deficit(fe.gauges(np.stack([EV.perm_of(INC['graphite'])])))[0])
S.ARMG_DIR['scissor']=sv
dok=float(S.armg_deficit(fe.gauges(np.stack([EV.perm_of(INC['graphite'])])))[0])
print(f"(8) mutation control: D(graphite) clean {dok:.4f} vs scissor-sign-flipped {dmut:.4f}")
assert abs(dmut-dok)>1e-6, "MUTATION DID NOT BITE -- the deficit ignores its directions"
res['mutation_delta']=abs(dmut-dok)
json.dump(res,open('/tmp/armg/agent-artifacts/armg/pc_armg_objective.json','w'),indent=1)
print("\nALL 8 ARM G OBJECTIVE CONTROLS PASS")

# (9) THE HARDCODED SIX must equal the artifact's six, and the shipped constant-assert
#     must pass on them. Added after hand-transcription defects were found TWICE.
P2=json.load(open('/local/home/zegertho/agent/state/optevidence/artifacts/search-noise-placebo.json'))
real=[r['layout'] for r in P2['runs']['baseline']]
assert list(S.ARMG_SIX)==real, f"ARMG_SIX != artifact\n{list(S.ARMG_SIX)}\n{real}"
print(f"(9) ARMG_SIX == artifact's six champions: OK")
chk=S.armg_assert_constants(fe, list(S.ARMG_SIX))
print(f"(9b) shipped armg_assert_constants(): {chk}")
res['constants_check']=chk
json.dump(res,open('/tmp/armg/agent-artifacts/armg/pc_armg_objective.json','w'),indent=1)
print("\nALL 9 ARM G OBJECTIVE CONTROLS PASS")
