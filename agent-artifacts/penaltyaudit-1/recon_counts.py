"""RECONCILE the parent's redirect=3240 against my reported 2700.

Hypothesis to test: in `scoring/oxey.py` (lines 143-146) `bad_redirect` is NESTED INSIDE
`redirect` — the code does `shares["redirect"] += f` and THEN, conditionally,
`shares["bad_redirect"] += f`. So the `redirect` TERM fires on (plain + bad) triples, while my
`pc_tri.py` helper returned a single MUTUALLY EXCLUSIVE label and therefore counted only the
PLAIN subset. If so: term-firing redirect = 2700 + 540 = 3240 = the parent's number.
"""
import numpy as np
from collections import Counter
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.analysis.community import _v1_pattern, FINGERS, SLOT2DOF
POS=list(G.slots)
_L={5:0,4:1,3:2,2:3,1:3}; _R={1:6,2:6,3:7,4:8,5:9}
def fenum(p): return _L[abs(p[0])] if p[0]<0 else _R[abs(p[0])]
for slot,p in enumerate(G.slots): assert fenum(p)==FINGERS[SLOT2DOF[slot]]

def oxey_terms_fired(a,b,c):
    """EXACTLY oxey.py:138-146 — returns the SET of terms incremented (bad IMPLIES redirect)."""
    ha,hb,hc=G.hand(a[0]),G.hand(b[0]),G.hand(c[0])
    if not (ha==hb==hc and ha!=0): return frozenset()
    d1=abs(b[0])-abs(a[0]); d2=abs(c[0])-abs(b[0])
    if d1 and d2 and (d1>0)==(d2>0): return frozenset({'onehand'})
    if d1 and d2:
        s={'redirect'}
        if not any(abs(p[0]) in (1,2) for p in (a,b,c)): s.add('bad_redirect')
        return frozenset(s)
    return frozenset()

def v1_family(a,b,c):
    r=_v1_pattern(fenum(a),fenum(b),fenum(c))
    if r is None: return None
    if r.startswith('onehand'): return 'onehand'
    if r.startswith('bad_redirect'): return 'bad_redirect'
    if r.startswith('redirect'): return 'redirect'
    return 'other'

UNIVERSES={
 'ALL 30^3 (a,b,c free, repeats allowed)': [(a,b,c) for a in POS for b in POS for c in POS],
 'a!=b and b!=c (adjacent-distinct)':      [(a,b,c) for a in POS for b in POS for c in POS if a!=b and b!=c],
 'all three distinct':                     [(a,b,c) for a in POS for b in POS for c in POS if a!=b and b!=c and a!=c],
}
print(f"{'universe':40s}{'N':>8s} | {'oh':>6s}{'rd_TERM':>9s}{'rd_plain':>9s}{'br':>6s} | "
      f"{'v1 oh':>6s}{'v1 rd':>7s}{'v1 br':>7s}{'v1 fam':>8s}")
for nm,U in UNIVERSES.items():
    oh=sum(1 for t in U if 'onehand' in oxey_terms_fired(*t))
    rd=sum(1 for t in U if 'redirect' in oxey_terms_fired(*t))          # the TERM (nested)
    br=sum(1 for t in U if 'bad_redirect' in oxey_terms_fired(*t))
    rdp=rd-br                                                           # plain-only subset
    v_oh=sum(1 for t in U if v1_family(*t)=='onehand')
    v_rd=sum(1 for t in U if v1_family(*t)=='redirect')
    v_br=sum(1 for t in U if v1_family(*t)=='bad_redirect')
    print(f'{nm:40s}{len(U):8d} | {oh:6d}{rd:9d}{rdp:9d}{br:6d} | {v_oh:6d}{v_rd:7d}{v_br:7d}{v_rd+v_br:8d}')
print('\n=== VERDICT on the 2700-vs-3240 discrepancy ===')
U=UNIVERSES['ALL 30^3 (a,b,c free, repeats allowed)']
rd=sum(1 for t in U if 'redirect' in oxey_terms_fired(*t)); br=sum(1 for t in U if 'bad_redirect' in oxey_terms_fired(*t))
print(f'  oxey.py `redirect` TERM fires on            {rd}  <- the parent\'s 3240')
print(f'  of which ALSO fire `bad_redirect` (nested)  {br}')
print(f'  so redirect-but-NOT-bad (exclusive subset)  {rd-br}  <- my reported 2700')
print(f'  => 2700 + 540 = {rd-br+br}. NOT a contradiction: different QUANTITY, same universe (30^3).')
print(f'  bad_redirect IS A SUBSET of redirect in oxey.py, so a bad redirect pays +2.0 AND +4.0 = +6.0.')
v_rd=sum(1 for t in U if v1_family(*t)=='redirect'); v_br=sum(1 for t in U if v1_family(*t)=='bad_redirect')
print(f'\n  _v1_pattern classes are MUTUALLY EXCLUSIVE: redirects*={v_rd}, bad_redirects*={v_br}, family={v_rd+v_br}')
print(f'  comparable family-vs-family: oxey.py {rd} vs v1 {v_rd+v_br}  -> oxey.py-only = {rd-(v_rd+v_br)}  <- my 432 HOLDS')
print(f'  plain-vs-plain:              oxey.py {rd-br} vs v1 {v_rd}    -> oxey.py-only = {(rd-br)-v_rd}')
# does collin3's mask (which passed the 0.0 positive control) use the NESTED structure?
print('\n=== does the SHARE instrument use the nested structure? (it passed PC at 0.0, so it must) ===')
import io, contextlib, importlib.util
spec=importlib.util.spec_from_file_location('c3','/tmp/penaudit/probe/collin3.py')
buf=io.StringIO()
with contextlib.redirect_stdout(buf): c3=importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
print(' ', [l for l in buf.getvalue().splitlines() if 'POSITIVE CONTROL' in l][0])
M_rd,M_br=c3.M_rd,c3.M_br
nested=bool((M_br & ~M_rd).sum()==0)
print(f'  M_br implies M_rd (nested)? {nested}   cells: M_rd={int(M_rd.sum())} M_br={int(M_br.sum())} (31 slots incl space)')
print(f'  -> the SHARE numbers in the dossier are NOT affected; only the reported CLASS COUNT was the exclusive subset.')
