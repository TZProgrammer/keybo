"""The sibling scissorprice's point 2, answered from the fit I ALREADY have (cheap, no new pool):
the 7.0x uses a LINEAR slope while the form is SATURATING, so the linearized slope is not the
marginal price AT the share real champions carry. Report the TANGENT at the operating share.
Quadratic form used by form.py: y = c0 + c1*(x-xbar) + c2*(x-xbar)^2, so dy/dx = c1 + 2*c2*(x-xbar).
I refit here to recover c1 (functional_form.json stored only c2 as `curvature` and the LINEAR slope)."""
import io, contextlib, importlib.util, random, json, numpy as np
spec=importlib.util.spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print([l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
shares_vec=c3.shares_vec; TERMS=c3.TERMS
from keybo.analysis import surfaces as SF
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
from keybo.analysis.surfaces import C30M
NAT="/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
obj=SF.trigram_objective(SF.default_trigram_path(None)); mass=obj[3].sum()
REG={**NAMED_LAYOUTS,**_EXTRA_NAMED}; usable={n:s for n,s in REG.items() if set(s)==set(C30M)}
rng=random.Random(31337)
def neigh(s,k):
    l=list(s)
    for _ in range(k):
        i,j=rng.randrange(30),rng.randrange(30); l[i],l[j]=l[j],l[i]
    return ''.join(l)
pool=[]
for n,s in usable.items():
    pool.append(s)
    for _ in range(80): pool.append(neigh(s,rng.choice([1,1,2,2,3,3,4,5])))
X=np.array([[shares_vec(s)[t] for t in TERMS] for s in pool])
isc=TERMS.index('scissor'); isf=TERMS.index('sfb')
x=X[:,isc]; xs=X[:,isf]
# the 11 REGISTRY layouts' own scissor share = the true operating point
reg_sh={n:shares_vec(s)['scissor'] for n,s in sorted(usable.items())}
print('\nregistry (n=11) scissor share %:')
for n,v in reg_sh.items(): print(f'   {n:14s} {v:.4f}')
reg=np.array(list(reg_sh.values()))
print(f'   -> registry mean {reg.mean():.4f}  median {np.median(reg):.4f}  range [{reg.min():.4f},{reg.max():.4f}]')
print(f'   -> form-pool mean {x.mean():.4f}; RANDOM-pool mean was 1.7471 (3.1x the registry mean!)')
out={'registry_scissor_share':reg_sh,'form_pool_share_mean':float(x.mean()),
     'per_source':{}}
print(f"\n{'src':10s}{'lin slope':>11s}{'tangent@regmean':>17s}{'tangent@regmed':>16s}{'ratio LIN':>11s}{'ratio TAN':>11s}")
for src in ('AALTO','COMMUNITY','POOL'):
    S=np.load(f'{NAT}/{src}_TRI_PS_FREQ_PRIOR.native.npy')
    y=np.array([SF.score_fit(l,S,obj) for l in pool])/mass
    def fit(v):
        A=np.column_stack([np.ones(len(v)),v]); c,*_=np.linalg.lstsq(A,y,rcond=None); return c[1]
    def quad_tangent(v, at):
        vc=v-v.mean(); A=np.column_stack([np.ones(len(v)),vc,vc**2])
        c,*_=np.linalg.lstsq(A,y,rcond=None)
        return c[1]+2*c[2]*(at-v.mean()), c[1], c[2]
    lin=fit(x); lsf=fit(xs)
    t_mean,c1,c2=quad_tangent(x, reg.mean())
    t_med,_,_ =quad_tangent(x, np.median(reg))
    # sfb anchor: use ITS tangent at ITS registry mean, so the ratio is apples-to-apples
    sfb_reg=np.array([shares_vec(s)['sfb'] for n,s in sorted(usable.items())])
    tsf,_,_=quad_tangent(xs, sfb_reg.mean())
    print(f'{src:10s}{lin:+11.4f}{t_mean:+17.4f}{t_med:+16.4f}{(lin/lsf)*12/4:10.3f}x{(t_mean/tsf)*12/4:10.3f}x')
    out['per_source'][src]=dict(linear_slope=float(lin), quad_c1=float(c1), quad_c2=float(c2),
        tangent_at_registry_mean=float(t_mean), tangent_at_registry_median=float(t_med),
        sfb_linear=float(lsf), sfb_tangent_at_registry_mean=float(tsf),
        ratio_linear=float((lin/lsf)*12/4), ratio_tangent=float((t_mean/tsf)*12/4))
print('\n  ratio TAN = both numerator and denominator evaluated as TANGENTS at the registry mean share.')
print('  Concavity is NEGATIVE (c2 -0.83..-1.28) and the registry mean share (%.4f) is BELOW the' % reg.mean())
print('  form-pool mean (%.4f), so the tangent at the operating point is STEEPER than the linear slope.' % x.mean())
json.dump(out,open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/scissor_tangent.json','w'),indent=1)
print('  wrote scissor_tangent.json')
