"""Read back EVERY headline number in report.md from the JSON artifacts, so no figure in the
dossier is a transcription. (My own trap-20 discipline applied to myself.)"""
import json
A='/local/home/zegertho/agent/state/penaltyaudit/artifacts'
ff=json.load(open(f'{A}/functional_form.json'))
bc=json.load(open(f'{A}/band_compare.json'))
idn=json.load(open(f'{A}/identification.json'))
iw=json.load(open(f'{A}/implied_weights.json'))
ca=json.load(open(f'{A}/cluster_attribution.json'))
fi=json.load(open(f'{A}/floor_and_identity.json'))
ok=[];bad=[]
def chk(label, got, want, tol=0.005):
    (ok if abs(got-want)<=tol else bad).append(f'{label}: got {got:.4f} want {want:.4f}')
# section 2
chk('eff dof random', idn['effective_dof'], 5.69, 0.01)
chk('eff dof nearopt', bc['near_optimal_pool']['eff_dof'], 2.50, 0.01)
chk('VIF alternate random', idn['vif']['alternate'], 8.17, 0.01)
chk('VIF alternate nearopt', bc['near_optimal_pool']['vif']['alternate'], 46.34, 0.02)
chk('VIF redirect nearopt', bc['near_optimal_pool']['vif']['redirect'], 19.49, 0.02)
chk('identity R2', fi['identity_r2'], 0.1858, 0.001)
chk('paired floor', fi['paired_floor_ms_per_char'], 0.2453, 0.001)
chk('unpaired floor', fi['unpaired_floor_ms_per_char'], 0.6513, 0.001)
chk('seed SS pct', fi['ss_seed_pct'], 0.97, 0.02)
chk('layout SS pct', fi['ss_layout_pct'], 98.94, 0.02)
chk('onehand cell ms', fi['class_ms_coeffs']['oh'], 18.053, 0.01)
chk('redirect cell ms', fi['class_ms_coeffs']['rd'], 12.099, 0.01)
chk('scissor cell ms', fi['class_ms_coeffs']['sci'], 36.150, 0.01)
# section 3 slopes + valid ranges
EXP={'sfb':(2.00,3.32,2.09,0.95,7.38),'dsfb':(2.06,4.17,2.34,4.05,8.39),
     'lsb':(1.67,3.48,1.85,0.23,4.49),'scissor':(4.90,7.31,4.90,0.08,3.06),
     'inroll':(0.89,1.33,0.90,3.51,12.38),'outroll':(1.35,2.15,1.29,1.66,11.07),
     'onehand':(4.08,5.91,3.76,0.30,2.82),'redirect':(1.45,2.28,1.40,1.24,8.48),
     'bad_redirect':(3.71,5.21,3.46,0.15,2.19),'alternate':(-0.70,-1.06,-0.68,64.96,80.81),
     'imbalance':(0.35,0.54,0.35,0.20,27.93)}
for t,(a,c,p,lo,hi) in EXP.items():
    for src,want in (('AALTO',a),('COMMUNITY',c),('POOL',p)):
        chk(f'slope {t}/{src}', ff['per_source'][src][t]['slope_ms_per_char_per_pt'], want)
    vr=ff['per_source']['AALTO'][t]['valid_range']
    chk(f'range-lo {t}', vr[0], lo); chk(f'range-hi {t}', vr[1], hi)
# implied weights
IMP={'sfb':12.00,'dsfb':13.64,'lsb':11.08,'scissor':28.02,'inroll':5.10,'outroll':7.77,
     'onehand':22.49,'redirect':8.33,'bad_redirect':20.32,'alternate':-3.97,'imbalance':2.02}
for t,w in IMP.items(): chk(f'implied {t}', iw[t]['implied_mean'], w, 0.01)
RAT={'onehand':-15.00,'outroll':-7.77,'inroll':-2.55,'scissor':7.01,'alternate':7.95,
     'bad_redirect':5.08,'redirect':4.16,'lsb':3.69,'dsfb':2.73,'imbalance':1.35}
for t,r in RAT.items(): chk(f'ratio {t}', iw[t]['ratio'], r, 0.01)
# cluster LOCO at K=6
K6=ca['sweep']['6']
grp_sci=[g for g in K6['groups'] if g.startswith('scissor')][0]
grp_br=[g for g in K6['groups'] if g=='bad_redirect'][0]
for src,want in (('AALTO',0.1630),('COMMUNITY',0.1247),('POOL',0.1362)):
    chk(f'LOCO scissor,outroll/{src}', K6['per_source'][src]['loco_dr2'][grp_sci], want, 0.001)
for src,want in (('AALTO',0.0000),('COMMUNITY',0.0012),('POOL',0.0002)):
    chk(f'LOCO bad_redirect/{src}', K6['per_source'][src]['loco_dr2'][grp_br], want, 0.001)
print(f'VERIFIED {len(ok)} / {len(ok)+len(bad)} report figures against the JSON artifacts')
for b in bad: print('  MISMATCH ->', b)
print('groups at K=6:', K6['groups'])
