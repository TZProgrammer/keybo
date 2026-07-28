import numpy as np, gzip, io
NAT = "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
import os
print("=== native dir listing ===")
for f in sorted(os.listdir(NAT)): print("  ", f)
def nat(n): return np.load(f'{NAT}/{n}.native.npy')
def std_repo(n):
    p=f'/tmp/penaudit/data/surfaces/{n}.standardized.npy.gz'
    return np.load(io.BytesIO(gzip.open(p,'rb').read())) if os.path.exists(p) else None
def std_nat(n):
    p=f'{NAT}/{n}.standardized.npy'
    return np.load(p) if os.path.exists(p) else None
for fam in ('BASE','FREQ_PRIOR','TRI_PS_FREQ_PRIOR'):
    for pool in ('AALTO','COMMUNITY','POOL'):
        n=f'{pool}_{fam}'
        try: N=nat(n)
        except Exception as e: print(n,'native MISSING'); continue
        SR=std_repo(n); SN=std_nat(n)
        msg=f'{n:32s} native[{N.min():8.2f},{N.max():8.2f}]'
        if SN is not None: msg+=f'  max|nat-stdNAT|={np.abs(N-SN).max():.4g}'
        if SR is not None: msg+=f'  max|nat-stdREPO|={np.abs(N-SR).max():.4g}'
        if SN is not None and SR is not None: msg+=f'  max|stdNAT-stdREPO|={np.abs(SN-SR).max():.4g}'
        print(msg)
