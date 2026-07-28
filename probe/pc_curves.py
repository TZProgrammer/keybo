import json
mine=json.load(open('/tmp/penaudit/out/ec-uniform/curves.json'))
theirs=json.load(open('/local/home/zegertho/agent/state/keybo-optimization/artifacts/theory-1/effect-curves-output/curves.json'))
# rename map: theirs used inroll/outroll, mine uses outer_high/outer_low
REN={'inroll':'outer_high','outroll':'outer_low'}
tw=theirs['wpms']; mw=mine['wpms']
idx_m={w:i for i,w in enumerate(mw)}
worst=0.0; nchk=0
print(f"{'block':32s} {'class':16s} {'wpm':>5s} {'theirs':>12s} {'mine':>12s} {'absdiff':>10s}")
for block in ('class_mean_ms','contrast_vs_alternate_ms','contrast_vs_alternate_pct','shap_of_defining_feature_ms'):
    for cls, vals in theirs[block].items():
        mc = REN.get(cls, cls)
        if mc not in mine[block]: print(f'  !! class {cls}->{mc} MISSING in mine'); continue
        for w, tv in zip(tw, vals):
            if w not in idx_m: continue
            mv = mine[block][mc][idx_m[w]]
            d = abs(tv-mv); worst=max(worst,d); nchk+=1
            if d > 1e-9:
                print(f'{block:32s} {cls:16s} {w:5.0f} {tv:12.6f} {mv:12.6f} {d:10.3g}')
print(f'\nPOSITIVE CONTROL: {nchk} cells compared, max abs diff = {worst:.6g}')
print('n_pairs theirs:', theirs['n_pairs'])
print('n_pairs mine  :', mine['n_pairs'])
print('weighted_by theirs:', theirs['weighted_by'], '| mine:', mine['weighted_by'])
