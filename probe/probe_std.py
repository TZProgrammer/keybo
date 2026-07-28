import numpy as np
NAT = "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
def nat(n): return np.load(f'{NAT}/{n}.native.npy')
def std(n): return np.load(f'{NAT}/{n}.standardized.npy')
# Hypothesis: standardized(X) = T2_AALTO[:,:,None] + Cond(X), where Cond(X)=native(X)-T2_X[:,:,None]
# Then std(X)-nat(X) = T2_AALTO - T2_X  (constant in c). TEST: is the diff constant along axis 2?
for fam in ('BASE','TRI_PS_FREQ_PRIOR'):
    for pool in ('COMMUNITY','POOL'):
        n=f'{pool}_{fam}'
        D = std(n)-nat(n)
        var_along_c = np.abs(D - D[:,:,:1]).max()
        print(f'{n:30s} max|D - D[:,:,0]| = {var_along_c:.6g}   (0 => D is a pure BIGRAM shift)')
        print(f'{"":30s} implied B_sub - B_own: range [{D.min():.4f},{D.max():.4f}]')
# Per trap 45: use the SHIPPED per-source bigram part, do not recover by difference.
Bc = np.load(f'{NAT}/COMMUNITY_BASE.bigram.seedmean.npy')
print('COMMUNITY_BASE.bigram.seedmean shape', Bc.shape)
A = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1"
T2a = np.load(f'{A}/T2_prod.npy')   # labelled prod == AALTO per THEORY-1
D = std('COMMUNITY_BASE')-nat('COMMUNITY_BASE')
Bsub_minus_Bown = D[:,:,0]
print('max|(B_sub - B_own) + Bc - T2_aalto| =', np.abs(Bsub_minus_Bown + Bc - T2a).max())
print('  -> if ~0, then B_sub == T2_aalto exactly (the shared-component claim)')
print('max|Bc - T2_aalto| =', np.abs(Bc-T2a).max())
