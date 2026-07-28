"""Two hazards the brief flags explicitly, tested rather than assumed.

(A) IDENTITY CHECK: is any oxey share definitionally entangled with the fitted-ms target?
    The target is sum_t F[t]*S[slot(t)] over trigrams; a share is sum_t F[t]*1[class(t)]/sum F.
    Both are corpus-weighted sums over the same index set, so if S were a LINEAR function of
    the class indicators the "prediction" would be algebra. Test: regress the surface's own
    cell values on the class indicators and see how much of S is explained.

(B) PAIRED RESOLUTION FLOOR for MY pool (the brief forbids reusing another artifact's floor).
    3 AALTO seed surfaces exist (T2_shipped_seed{0,1,2} + Tc_aalto_seed{0,1,2}), so the
    layout x seed matrix can be decomposed and the PAIRED floor computed for THIS pool.
"""
import io, contextlib, importlib.util, json, numpy as np
spec=importlib.util.spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print([l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
shares_vec=c3.shares_vec; TERMS=c3.TERMS; SLOTS=c3.SLOTS; NS=c3.NS; SPACE=c3.SPACE
M_oh,M_rd,M_br=c3.M_oh,c3.M_rd,c3.M_br
M_sfb,M_alt,M_lsb,M_sci,M_in,M_out=c3.M_sfb,c3.M_alt,c3.M_lsb,c3.M_sci,c3.M_in,c3.M_out
from keybo.analysis import surfaces as SF
NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
TH ="/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1"

print('\n=== (A) IDENTITY CHECK: how much of the SURFACE is a linear function of the oxey classes? ===')
S=np.load(f'{NAT}/AALTO_TRI_PS_FREQ_PRIOR.native.npy')
ii,jj,kk=np.meshgrid(np.arange(NS),np.arange(NS),np.arange(NS),indexing='ij')
ii,jj,kk=ii.ravel(),jj.ravel(),kk.ravel(); sv=S[ii,jj,kk]
cols={'sfb':M_sfb[ii,jj],'alt':M_alt[ii,jj],'lsb':M_lsb[ii,jj],'sci':M_sci[ii,jj],
      'in':M_in[ii,jj],'out':M_out[ii,jj],'oh':M_oh[ii,jj,kk],'rd':M_rd[ii,jj,kk],'br':M_br[ii,jj,kk]}
A=np.column_stack([np.ones_like(sv)]+[c.astype(float) for c in cols.values()])
co,*_=np.linalg.lstsq(A,sv,rcond=None)
r2=1-((sv-A@co)**2).sum()/((sv-sv.mean())**2).sum()
print(f'  regress every one of {len(sv)} surface cells on the 9 oxey class indicators:')
print(f'  R2 = {r2:.4f}   -> {"ALGEBRA (entangled)" if r2>0.95 else "NOT an identity: the classes explain only part of the surface"}')
print(f'  residual RMS = {np.sqrt(((sv-A@co)**2).mean()):.2f} ms  vs surface sd {sv.std():.2f} ms')
print('  per-class fitted ms coefficient (the surface\'s OWN price for membership):')
for n,c in zip(cols,co[1:]): print(f'    {n:5s} {c:+8.3f} ms')

print('\n=== (B) PAIRED RESOLUTION FLOOR for MY pool (trap 37 / the brief: name your pool) ===')
T2s=[np.load(f'{TH}/T2_shipped_seed{s}.npy') for s in range(3)]
Tcs=[np.load(f'{TH}/Tc_aalto_seed{s}.npy') for s in range(3)]
SEED=[T2s[s][:,:,None]+Tcs[s] for s in range(3)]
print(f'  {len(SEED)} AALTO seed surfaces (T2_shipped_seed*+Tc_aalto_seed*); shape {SEED[0].shape}')
obj=SF.trigram_objective(SF.default_trigram_path(None))
pool=json.load(open('/tmp/penaudit/probe/_pool_random.json'))
import random
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
from keybo.analysis.surfaces import C30M
REG={**NAMED_LAYOUTS,**_EXTRA_NAMED}; usable={n:s for n,s in REG.items() if set(s)==set(C30M)}
LAYS=sorted(usable.items())
print(f'  POOL FOR THE FLOOR: the {len(LAYS)} C30M-exact registry layouts = {[n for n,_ in LAYS]}')
# per-char normalisation: total corpus trigram mass
mass=obj[3].sum()
Y=np.array([[SF.score_fit(s,SEED[k],obj)/mass for n,s in LAYS] for k in range(3)])  # (seed, layout)
print(f'  scores in ms/char: seed-mean per layout:')
for i,(n,_) in enumerate(LAYS): print(f'    {n:14s} {Y[:,i].mean():8.4f}  per-seed [{Y[:,i].min():.4f},{Y[:,i].max():.4f}]  spread {Y[:,i].max()-Y[:,i].min():.4f}')
gm=Y.mean(); seed_eff=Y.mean(1)-gm; lay_eff=Y.mean(0)-gm
ss_tot=((Y-gm)**2).sum(); ss_seed=len(LAYS)*(seed_eff**2).sum(); ss_lay=3*(lay_eff**2).sum()
ss_res=ss_tot-ss_seed-ss_lay
print(f'\n  variance decomposition of the {3}x{len(LAYS)} matrix:')
print(f'    SEED main effect   {100*ss_seed/ss_tot:6.2f}% of SS')
print(f'    LAYOUT main effect {100*ss_lay/ss_tot:6.2f}% of SS')
print(f'    interaction/resid  {100*ss_res/ss_tot:6.2f}% of SS')
unpaired=float(np.mean(Y.max(0)-Y.min(0)))
print(f'\n  UNPAIRED floor (mean within-layout per-seed range) = {unpaired:.4f} ms/char')
D=[]
for a in range(len(LAYS)):
    for b in range(a+1,len(LAYS)):
        d=Y[:,a]-Y[:,b]; D.append(d.max()-d.min())
paired=float(np.mean(D))
print(f'  PAIRED floor (mean per-seed range of the DIFFERENCE, n={len(D)} pairs) = {paired:.4f} ms/char')
print(f'  ratio paired/unpaired = {paired/unpaired:.3f}')
print(f'  >> MY POOL IS n={len(LAYS)} (registry C30M-exact). The brief cites 0.2222 at n=8 and')
print(f'     other artifacts at n=10/n=11 -- this is MY pool\'s number, not theirs.')
json.dump(dict(identity_r2=float(r2), class_ms_coeffs={n:float(c) for n,c in zip(cols,co[1:])},
  floor_pool=[n for n,_ in LAYS], floor_pool_n=len(LAYS),
  unpaired_floor_ms_per_char=unpaired, paired_floor_ms_per_char=paired,
  ss_seed_pct=float(100*ss_seed/ss_tot), ss_layout_pct=float(100*ss_lay/ss_tot),
  ss_resid_pct=float(100*ss_res/ss_tot),
  per_layout_ms_per_char={n:float(Y[:,i].mean()) for i,(n,_) in enumerate(LAYS)}),
  open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/floor_and_identity.json','w'),indent=1)
print('\nwrote floor_and_identity.json')
