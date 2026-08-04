"""Pure-math check of the LMDI (log-mean Divisia) identity BEFORE registering it.
No keybo, no models -- just: is  ms_B - ms_A == L * (p_B - p_A)  exact, where
ms = K*exp(p) and L = (ms_B-ms_A)/(p_B-p_A)?  And does it distribute over a
partition of (p_B - p_A) into per-feature deltas?"""
import numpy as np
rng = np.random.default_rng(0)
K = 12000.0 / 90.0
n, F = 200000, 20
# realistic LOGRAT scale: log(ms*wpm/12000) with ms~250/char*... keep p in [0.3,1.2]
pA = rng.uniform(0.3, 1.2, n)
pB = rng.uniform(0.3, 1.2, n)
# force some exact ties and some near-ties (the numerically scary cases)
pB[:100] = pA[:100]
pB[100:200] = pA[100:200] + 1e-15
pB[200:300] = pA[200:300] + 1e-9
msA, msB = K*np.exp(pA), K*np.exp(pB)
dp, dms = pB - pA, msB - msA
L = np.where(dp != 0.0, dms/np.where(dp != 0.0, dp, 1.0), 0.5*(msA+msB))
print("identity  L*dp  vs  dms :  max abs", np.abs(L*dp - dms).max(),
      " max rel", np.abs((L*dp - dms)/np.where(dms!=0,dms,1)).max())
# now a partition: split dp into F random pieces summing EXACTLY to dp (as SHAP does)
w = rng.random((n, F)); w /= w.sum(axis=1, keepdims=True)
dshap = w * dp[:, None]            # sums to dp per row, to float error
attrib = L[:, None] * dshap        # per-feature ms attribution
recon = attrib.sum(axis=1)
print("partition sum vs dms    :  max abs", np.abs(recon - dms).max(),
      " max rel", np.abs((recon-dms)/np.where(dms!=0,dms,1)).max())
# aggregate: frequency-weighted sum over rows
f = rng.integers(1, 10**7, n).astype(float)
agg = (f[:, None]*attrib).sum(axis=0)          # (F,)
direct = (f*dms).sum()
print("AGG  sum_i agg_i vs sum f*dms:", agg.sum(), direct,
      " rel resid", abs(agg.sum()-direct)/abs(direct))
# contrast: what a FIRST-ORDER (linearization at A) decomposition would cost
lin = (msA[:, None]*dshap)
print("first-order (msA) rel resid  :", abs((f[:,None]*lin).sum()-direct)/abs(direct))
lin2 = (0.5*(msA+msB)[:, None]*dshap)
print("midpoint    (mean) rel resid :", abs((f[:,None]*lin2).sum()-direct)/abs(direct))
