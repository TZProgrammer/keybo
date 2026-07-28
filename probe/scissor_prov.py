"""Full provenance for the scissor 7.0x claim, plus a CI on the RATIO itself
(the dossier gave a CI on the slope but not on the ratio, and the ratio is what the parent
would act on)."""
import json, numpy as np
A='/local/home/zegertho/agent/state/scissorprice/artifacts'
ff=json.load(open(f'{A}/functional_form.json')); iw=json.load(open(f'{A}/implied_weights.json'))
idn=json.load(open(f'{A}/identification.json')); bc=json.load(open(f'{A}/band_compare.json'))
print('=== ESTIMATOR / POOL / n / CONTROL LEVEL behind scissor 7.0x ===')
print(f'  pool n = {ff["pool_n"]}  (near-optimal: 11 C30M-exact registry layouts x 81, 1-5 random swaps)')
print(f'  corpus = blend-v1 ; target = score_fit(layout, <POOL>_TRI_PS_FREQ_PRIOR.native, blend-v1 trigrams)/mass')
print(f'  units  = ms/char per PERCENTAGE POINT of share ; g-frame only ; baked 90 WPM')
print(f'  CONTROL LEVEL = **NONE (MARGINAL)** -- OLS of ms/char on the single term, no other term partialled out')
for src in ('AALTO','COMMUNITY','POOL'):
    d=ff['per_source'][src]['scissor']
    print(f'  {src:10s} slope {d["slope_ms_per_char_per_pt"]:+.4f}  CI95 [{d["slope_ci95"][0]:+.4f},{d["slope_ci95"][1]:+.4f}]'
          f'  R2lin {d["r2_linear"]:.4f} R2quad {d["r2_quad"]:.4f} gain {d["quad_gain"]:.4f} form {d["form"]}'
          f'  valid [{d["valid_range"][0]:.4f},{d["valid_range"][1]:.4f}]%')
print(f'\n  implied weight per source: '+', '.join(f'{s}={v:+.2f}' for s,v in zip(("A","C","P"),iw["scissor"]["implied_per_source"])))
print(f'  implied MEAN {iw["scissor"]["implied_mean"]:+.4f} vs shipped {iw["scissor"]["oxey"]:+.1f} -> ratio {iw["scissor"]["ratio"]:.4f}')
# CI on the RATIO: propagate the two slope CIs (scissor and the sfb anchor) per source
print('\n=== CI on the RATIO itself (was NOT in the dossier) ===')
print('  ratio = (slope_scissor/slope_sfb)*12.0/4.0 ; anchor sfb==+12.0, shipped scissor +4.0')
rs=[]
for src in ('AALTO','COMMUNITY','POOL'):
    ss=ff['per_source'][src]['scissor']; sf=ff['per_source'][src]['sfb']
    # conservative interval: scissor-CI endpoints over sfb-CI opposite endpoints
    lo=(ss['slope_ci95'][0]/sf['slope_ci95'][1])*12.0/4.0
    hi=(ss['slope_ci95'][1]/sf['slope_ci95'][0])*12.0/4.0
    pt=(ss['slope_ms_per_char_per_pt']/sf['slope_ms_per_char_per_pt'])*12.0/4.0
    rs.append(pt)
    print(f'  {src:10s} ratio {pt:.3f}x   conservative CI95 [{lo:.3f}x, {hi:.3f}x]')
print(f'  cross-source SPREAD of the ratio: {min(rs):.3f}x - {max(rs):.3f}x (mean {np.mean(rs):.3f}x)')
print('\n=== IS SCISSOR IN THE 5-TERM COLLINEAR CLUSTER? ===')
print(f'  VIF(scissor): random pool {idn["vif"]["scissor"]:.2f}  |  near-optimal pool {bc["near_optimal_pool"]["vif"]["scissor"]:.2f}')
print(f'  VIF for comparison, near-opt: alternate {bc["near_optimal_pool"]["vif"]["alternate"]:.2f}, '
      f'redirect {bc["near_optimal_pool"]["vif"]["redirect"]:.2f}, sfb {bc["near_optimal_pool"]["vif"]["sfb"]:.2f}, '
      f'onehand {bc["near_optimal_pool"]["vif"]["onehand"]:.2f}, imbalance {bc["near_optimal_pool"]["vif"]["imbalance"]:.2f}')
print(f'  var-share of oxey score (near-opt): scissor {100*bc["near_optimal_pool"]["var_share"][3]:.1f}%')
