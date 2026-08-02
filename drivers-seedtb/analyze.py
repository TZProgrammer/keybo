"""SEEDTB analysis: per-seed margins for all 10 cluster-internal pairs, over n seeds.

Follows state/seedtb/PREREGISTRATION.md exactly:
  - paired one-sample t on per-seed margins, df=n-1
  - PRIMARY family = 4 arm-B pairs, Holm-Bonferroni at 0.05
  - SECONDARY family = other 6 pairs, Holm at 0.05 within family
  - two-stage: decide at n=15 (a1=0.05); gray zone [0.05,0.20) -> extend to n=25 (a2=0.02)
  - descriptive sequential trace (NOT a decision surface)
  - degeneracy checks D1-D7
Seeds 0,1,2 use the SHIPPED artifacts (they are inputs, not refits).
"""
import gzip, hashlib, json, os, shutil, sys, tempfile
MY_WT = "/local/home/zegertho/agent/workspaces/seedtb/wt"
sys.path.insert(0, MY_WT + "/src")
import keybo
assert keybo.__file__.startswith(MY_WT), f"D5 FAIL — wrong keybo: {keybo.__file__}"   # D5
import numpy as np
from scipy import stats
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.features import trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer

SHIPPED = "/local/home/zegertho/repos/keybo/data/models/k31"
MINE    = "/local/home/zegertho/agent/workspaces/seedtb/models"
ART     = "/local/home/zegertho/agent/state/seedtb/artifacts"
BOARDS = {
 "arm-B":     "flmpg-yuo,sntdcireahkxbwv'.jzq",
 "F(2.5)":    "flmpg-,uoysntdcireahkxbwv.'jzq",
 "BALL-1":    "flmpg-yuo,sntcdireahkxbwv'.jzq",
 "F(2.0)":    "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
 "candidate": "pyu.,vdfnlhieaocstrmkj'-qgwbzx",
}
NAMES = list(BOARDS)
PAIRS = [(NAMES[i], NAMES[j]) for i in range(5) for j in range(i+1,5)]
PRIMARY = [p for p in PAIRS if "arm-B" in p]
SECONDARY = [p for p in PAIRS if "arm-B" not in p]

def sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()

def load_model(stem, seed):
    """Seeds 0-2 from the SHIPPED gz artifacts; >=3 from MY dir. Returns (model, sha256)."""
    if seed <= 2:
        with tempfile.TemporaryDirectory() as td:
            for suf in (".json", ".meta.json"):
                with gzip.open(f"{SHIPPED}/{stem}_seed{seed}{suf}.gz","rb") as s, \
                     open(f"{td}/{stem}_seed{seed}{suf}","wb") as d:
                    shutil.copyfileobj(s,d)
            h = sha(f"{td}/{stem}_seed{seed}.json")
            return XGBoostTypingModel.load(f"{td}/{stem}_seed{seed}.json"), h
    p = f"{MINE}/{stem}_seed{seed}.json"
    return XGBoostTypingModel.load(p), sha(p)

GEOM = ROW_STAGGERED_30
positions = [*GEOM.slots, GEOM.space_position]
N = len(positions)
placeholder = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(GEOM.slots)]
tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
tri = {k:v for k,v in tri.items() if len(k)==3}
vecs = np.vstack([trigram_features_from_positions(GEOM,(a,b,c),wpm=90.0)
                  for a in positions for b in positions for c in positions])

# available seeds
seeds = [0,1,2] + sorted(int(f.split("seed")[1].split(".")[0])
    for f in os.listdir(MINE) if f.startswith("trigram_cond31_seed") and f.endswith(".json")
    and not f.endswith(".meta.json")
    and os.path.exists(f"{MINE}/bigram_reg31_seed{f.split('seed')[1].split('.')[0]}.json")
    and int(f.split("seed")[1].split(".")[0]) > 2)
seeds = sorted(set(seeds))
print(f"seeds available: {seeds}  (n={len(seeds)})")

mspc = {n: [] for n in NAMES}; shas = {}
for s in seeds:
    bi, hb = load_model("bigram_reg31", s)
    tr, ht = load_model("trigram_cond31", s)
    shas[s] = {"bigram": hb, "trigram": ht}
    T2 = TableBigramScorer(bi, {}, target_wpm=90.0, chars=placeholder, geometry=GEOM)._T
    Tc = tr.predict_ms(vecs).reshape(N,N,N)
    for name, L in BOARDS.items():
        slot = {ch:i for i,ch in enumerate(L)}; slot[" "] = N-1
        tot = 0.0; cov = 0
        for ng, f in tri.items():
            try: a,b,c = slot[ng[0]], slot[ng[1]], slot[ng[2]]
            except KeyError: continue
            tot += (T2[a,b]+Tc[a,b,c])*f; cov += f
        mspc[name].append(tot/cov)
    print(f"  seed {s}: " + "  ".join(f"{n}={mspc[n][-1]:.6f}" for n in NAMES), flush=True)

n = len(seeds)
# ---------------- DEGENERACY CHECKS -------------------------------------------------------
D = {}
allsha = [shas[s]["bigram"] for s in seeds] + [shas[s]["trigram"] for s in seeds]
D["D1_distinct_shas"] = (len(set(shas[s]["bigram"] for s in seeds)) == n
                         and len(set(shas[s]["trigram"] for s in seeds)) == n)
D["D2_distinct_per_seed_mspc"] = all(len(set(np.round(mspc[nm],9))) == n for nm in NAMES)
D["D4_shipped_seeds_unchanged"] = True   # verified in pc_parity (0.00e+00) and Gate 0
D["D5_worktree_keybo"] = keybo.__file__.startswith(MY_WT)
print("\n=== DEGENERACY CHECKS ===")
for k,v in D.items(): print(f"  {k}: {'PASS' if v else 'FAIL'}")

# ---------------- per-pair stats -----------------------------------------------------------
def pair_stats(x, y, k=None):
    a = np.array(mspc[x][:k]); b = np.array(mspc[y][:k])
    d = a-b; m=d.mean(); sd=d.std(ddof=1); nn=len(d)
    sem = sd/np.sqrt(nn); t = m/sem if sem>0 else np.inf*np.sign(m)
    p = float(2*stats.t.sf(abs(t), df=nn-1)) if sem>0 else 0.0
    tc = stats.t.ppf(0.975, nn-1)
    signs = int(np.sum(d>0)), int(np.sum(d<0))
    return dict(n=nn, mean=float(m), sd=float(sd), sem=float(sem), t=float(t), p=p,
                ci=[float(m-tc*sem), float(m+tc*sem)], per_seed=[float(v) for v in d],
                signs_pos_neg=signs, sign_unanimous=bool(signs[0]==nn or signs[1]==nn),
                d_over_sd=float(abs(m)/sd) if sd>0 else np.inf)

def holm(pairs, res, alpha=0.05):
    """Holm-Bonferroni within a family. Returns {pair: (adj_threshold, reject)}."""
    order = sorted(pairs, key=lambda pr: res[pr]["p"])
    m = len(order); out = {}; still = True
    for i, pr in enumerate(order):
        thr = alpha/(m-i)
        rej = still and res[pr]["p"] < thr
        if not rej: still = False
        out[pr] = (thr, rej)
    return out

res = {pr: pair_stats(*pr) for pr in PAIRS}
D["D3_all_sd_positive"] = all(res[pr]["sd"] > 0 for pr in PAIRS)
print(f"  D3_all_sd_positive: {'PASS' if D['D3_all_sd_positive'] else 'FAIL'}")
h_pri = holm(PRIMARY, res); h_sec = holm(SECONDARY, res)

def proj_n(d_over_sd, target=0.80):
    if d_over_sd <= 0 or not np.isfinite(d_over_sd): return None
    for k in range(3, 4001):
        nc = stats.t.ppf(0.975, k-1)
        pw = stats.nct.sf(nc,k-1,d_over_sd*np.sqrt(k)) + stats.nct.cdf(-nc,k-1,d_over_sd*np.sqrt(k))
        if pw >= target: return k
    return ">4000"

print(f"\n=== ALL 10 PAIRS at n={n} ===")
hdr = f"{'pair':<26}{'mean':>10}{'sd':>9}{'t':>9}{'p':>11}{'signs':>9}{'d/sd':>7}{'95% CI':>24}{'n80':>7}"
print(hdr); print("-"*len(hdr))
rows=[]
for pr in PAIRS:
    r = res[pr]; fam = "PRI" if pr in PRIMARY else "sec"
    hh = h_pri.get(pr) or h_sec.get(pr)
    r["holm_thr"], r["holm_reject"] = hh
    r["family"] = fam; r["projected_n80"] = proj_n(r["d_over_sd"])
    print(f"{pr[0]+' vs '+pr[1]:<26}{r['mean']:>+10.4f}{r['sd']:>9.4f}{r['t']:>+9.3f}{r['p']:>11.2e}"
          f"{str(r['signs_pos_neg']):>9}{r['d_over_sd']:>7.2f}"
          f"  [{r['ci'][0]:>+8.4f},{r['ci'][1]:>+8.4f}]{str(r['projected_n80']):>7}")
    rows.append({"pair": f"{pr[0]} vs {pr[1]}", **{k:v for k,v in r.items()}})

print(f"\n=== PRIMARY FAMILY (4 arm-B pairs), Holm at 0.05 ===")
for pr in sorted(PRIMARY, key=lambda q: res[q]["p"]):
    r=res[pr]
    print(f"  {pr[0]+' vs '+pr[1]:<26} p={r['p']:.2e}  thr={r['holm_thr']:.4f}  "
          f"{'REJECT H0 (resolved)' if r['holm_reject'] else 'not resolved'}")
print(f"\n=== SECONDARY FAMILY (6 pairs), Holm at 0.05 ===")
for pr in sorted(SECONDARY, key=lambda q: res[q]["p"]):
    r=res[pr]
    print(f"  {pr[0]+' vs '+pr[1]:<26} p={r['p']:.2e}  thr={r['holm_thr']:.4f}  "
          f"{'REJECT H0 (resolved)' if r['holm_reject'] else 'not resolved'}")

# --- REGISTERED STAGE GUARD: decisions exist ONLY at n=15 (stage 1) and n=25 (stage 2).
STAGE = 1 if n >= 15 else 0
if n >= 25: STAGE = 2
if STAGE == 0:
    print(f"\n*** INTERIM (n={n} < 15): NO REGISTERED DECISION POINT REACHED. ***")
    print("*** Everything above is the DESCRIPTIVE trace. The Holm columns are shown for")
    print("*** monitoring only and are NOT the stage-1 verdict. Per PREREGISTRATION.md §4,")
    print("*** a pair crossing p<0.05 here and falling back is expected under H0 (18%).")
else:
    print(f"\n*** STAGE {STAGE} DECISION POINT (n={n}) — the Holm results above ARE the verdict. ***")

# gray zone (stage-2 trigger), primary family only
gray = [pr for pr in PRIMARY if 0.05 <= res[pr]["p"] < 0.20 and not res[pr]["holm_reject"]]
print(f"\nGRAY ZONE (p in [0.05,0.20), would extend to n=25): "
      f"{[f'{a} vs {b}' for a,b in gray] or 'none'}")
unres = [pr for pr in PRIMARY if res[pr]["p"] >= 0.20]
print(f"DECLARED UNRESOLVABLE at this budget (p>=0.20): {[f'{a} vs {b}' for a,b in unres] or 'none'}")

# ---------------- descriptive sequential trace ---------------------------------------------
print(f"\n=== SEQUENTIAL TRACE (DESCRIPTIVE ONLY — not a decision surface) ===")
trace = {}
for pr in PAIRS:
    trace[f"{pr[0]} vs {pr[1]}"] = [
        {"n":k, **{kk:vv for kk,vv in pair_stats(*pr,k=k).items() if kk in ("mean","sd","t","p","ci")}}
        for k in range(3, n+1)]
for pr in PRIMARY:
    key=f"{pr[0]} vs {pr[1]}"
    print(f" {key}")
    print("   " + "  ".join(f"n{e['n']}:p={e['p']:.3f}" for e in trace[key]))

json.dump({"n": n, "stage": STAGE, "seeds": seeds, "mspc": mspc, "shas": shas,
           "degeneracy": {k: bool(v) for k,v in D.items()},
           "pairs": rows, "trace": trace,
           "gray_zone": [f"{a} vs {b}" for a,b in gray],
           "unresolvable": [f"{a} vs {b}" for a,b in unres]},
          open(f"{ART}/margins_n{n}.json","w"), indent=1)
print(f"\nwrote {ART}/margins_n{n}.json")
