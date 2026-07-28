import json, subprocess, sys
from pathlib import Path
sys.path.insert(0,'/tmp/armg/agent-artifacts/armg/drivers')
import numpy as np, evobj as EV
from keybo.analysis.evidence_scorer import EXPECTED_SIGN
G=("sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll","redir","scissor","imbalance","oxey-style","comfort")
P=json.load(open('/local/home/zegertho/agent/state/optevidence/artifacts/search-noise-placebo.json'))
six=[r['layout'] for r in P['runs']['baseline']]
QW="qwertyuiopasdfghjkl'zxcvbnm,.-"
fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
# 4000 random perms: rank-correlate each gauge with predicted ms/char (ARME-1's test)
rng=np.random.default_rng(20260728)
perms=np.stack([EV._as31 if False else np.concatenate([rng.permutation(30).astype(np.int32),[30]]) for _ in range(4000)])
g=fe.gauges(perms); ms=g['_ms_per_char']
from scipy.stats import spearmanr
print(f"{'gauge':<12} {'EXPECTED':>9} {'rho(g,ms) 4000 rand':>20} {'agrees':>7} | {'qwerty':>10} {'best-of-6':>10} {'qwerty worst?':>14}")
gq=fe.gauges(np.stack([EV.perm_of(QW)])); g6=fe.gauges(np.stack([EV.perm_of(l) for l in six]))
rows={}
for gg in G:
    rho=spearmanr(g[gg],ms).statistic
    exp=EXPECTED_SIGN[gg]
    agree = (np.sign(rho)==np.sign(exp))
    qv=float(gq[gg][0]); best=float(np.min(g6[gg])) if exp>0 else float(np.max(g6[gg]))
    qworst = (qv>best) if exp>0 else (qv<best)
    rows[gg]={"rho_random4000":float(rho),"expected_sign":exp,"rho_agrees":bool(agree),
              "qwerty":qv,"best_of_six":best,"qwerty_is_worse_than_best_of_six":bool(qworst)}
    print(f"{gg:<12} {exp:>+9.1f} {rho:>20.4f} {str(agree):>7} | {qv:>10.4f} {best:>10.4f} {str(qworst):>14}")
n_ag=sum(1 for v in rows.values() if v['rho_agrees']); n_qw=sum(1 for v in rows.values() if v['qwerty_is_worse_than_best_of_six'])
print(f"\nEXPECTED_SIGN agrees with rho on {n_ag}/14 over 4000 random perms")
print(f"qwerty is WORSE than best-of-six on {n_qw}/14 under EXPECTED_SIGN directions")
json.dump(rows,open('/tmp/armg/agent-artifacts/armg/gauge-directions.json','w'),indent=1)
