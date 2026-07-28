import numpy as np, gzip, io, os
A = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1"
T2p=np.load(f'{A}/T2_prod.npy'); T2c=np.load(f'{A}/T2_comm.npy'); T2pool=np.load(f'{A}/T2_pool.npy')
Tca=np.load(f'{A}/Tc_aalto.npy'); Tcc=np.load(f'{A}/Tc_comm.npy'); Tcp=np.load(f'{A}/Tc_pool.npy')
print('shapes T2', T2p.shape, 'Tc', Tca.shape)
print('max|T2_prod - T2_comm| =', np.abs(T2p-T2c).max())
print('max|T2_prod - T2_pool| =', np.abs(T2p-T2pool).max())
print('max|T2_comm - T2_pool| =', np.abs(T2c-T2pool).max())
print('max|Tc_aalto - Tc_comm| =', np.abs(Tca-Tcc).max())
print('max|Tc_aalto - Tc_pool| =', np.abs(Tca-Tcp).max())
def L(n):
    p=f'/tmp/penaudit/data/surfaces/{n}_TRI_PS_FREQ_PRIOR.standardized.npy.gz'
    return np.load(io.BytesIO(gzip.open(p,'rb').read()))
S={n:L(n) for n in ('AALTO','COMMUNITY','POOL')}
for n,v in S.items(): print(f'{n}_std shape {v.shape} range [{v.min():.3f},{v.max():.3f}]')
serv = T2p[:,:,None]+Tca
print('max|T2_prod[:,:,None]+Tc_aalto - AALTO_std| =', np.abs(serv-S['AALTO']).max())
print('max|AALTO_std - COMM_std| =', np.abs(S['AALTO']-S['COMMUNITY']).max())
print('max|AALTO_std - POOL_std| =', np.abs(S['AALTO']-S['POOL']).max())
# Bigram-part recovery: the additive decomp S=B[a,b]+C[a,b,c] is not unique; test
# whether the c-MARGINAL structure differs, i.e. delta = S - S.mean(2,keepdims=True)
for n in ('COMMUNITY','POOL'):
    d = (S[n]-S[n].mean(2,keepdims=True)) - (S['AALTO']-S['AALTO'].mean(2,keepdims=True))
    print(f'max|centered({n}) - centered(AALTO)| =', np.abs(d).max())
    print(f'max|Bmarg({n}) - Bmarg(AALTO)| =', np.abs(S[n].mean(2)-S['AALTO'].mean(2)).max())
