"""What SHOULD each weight be, in the scorer's OWN units?

DEFAULT_OXEY_WEIGHTS is 'signed weight per corpus-share PERCENT' and the total is a
dimensionless score. The measurement-calibrated analogue of a weight is therefore
   w_measured(term) = d(fitted ms/char) / d(share percentage point)
which is exactly the slope table -- so the recommendation can be quoted in ms/char/pt AND
rescaled to oxey's dimensionless convention. The rescaling constant is a CHOICE, so I anchor
it on sfb (the one term everybody agrees about) and report every other weight RELATIVE to it.
That makes the comparison convention-free.
"""
import json, numpy as np
FF=json.load(open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/functional_form.json'))
TERMS=FF['terms']; W=FF['weights']
print('anchor: sfb. oxey pays sfb +12.0, so scale = 12.0 / slope(sfb) per source.\n')
print(f'{"term":13s}{"oxeyW":>7s}'+''.join(f'{s[:5]+" impl":>12s}' for s in ('AALTO','COMMUNITY','POOL'))
      +f'{"impl mean":>11s}{"ratio impl/oxey":>17s}  verdict")'.replace('")','') )
rows={}
for t in TERMS:
    impl=[]
    for src in ('AALTO','COMMUNITY','POOL'):
        d=FF['per_source'][src]
        sc=12.0/d['sfb']['slope_ms_per_char_per_pt']
        impl.append(d[t]['slope_ms_per_char_per_pt']*sc)
    m=float(np.mean(impl)); ratio=m/W[t] if W[t] else float('nan')
    if np.sign(m)!=np.sign(W[t]): verdict='WRONG SIGN'
    elif abs(ratio)>2.0 or abs(ratio)<0.5: verdict='WRONG MAGNITUDE'
    else: verdict='consistent'
    rows[t]=dict(implied_per_source=impl, implied_mean=m, oxey=W[t], ratio=float(ratio), verdict=verdict)
    print(f'{t:13s}{W[t]:+7.1f}'+''.join(f'{v:+12.2f}' for v in impl)+f'{m:+11.2f}{ratio:+17.2f}  {verdict}')
print('\n  (implied = marginal slope rescaled so sfb == +12.0; sfb is +12.0 by construction)')
print('  (a NEGATIVE implied weight means the measurement wants a REWARD; positive = penalty)')
json.dump(rows,open('/local/home/zegertho/agent/state/penaltyaudit/artifacts/implied_weights.json','w'),indent=1)
print('\nwrote implied_weights.json')
