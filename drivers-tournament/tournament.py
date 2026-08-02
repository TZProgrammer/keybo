"""THE TOURNAMENT — complete pairwise round-robin over 13 boards, per TOURNAMENT-1 prereg.

Three pricings (all-cells / observed-only / common-support), paired per-seed margins over a COMMON
seed set, the measured floor (FLOOR-A/B/C), the registered verdict rule, transitivity/Condorcet
analysis, and the degeneracy battery.
"""
import json, os, sys, time, itertools, glob
import numpy as np
sys.path.insert(0, "/local/home/zegertho/agent/workspaces/tournament/wt/drivers-tournament")
from _guard import assert_d5, BOARDS, ART, E2E, SHIPPED

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"D5 OK keybo={assert_d5()}")

from scipy import stats
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.data.corpus import load_frequencies, production_corpus_dir

WPM = 90.0
TABLES = "/local/home/zegertho/agent/workspaces/tournament/tables"
POS = [*G.slots, G.space_position]; N = len(POS)
RNG = np.random.default_rng(20260802)   # fixed, registered
NAMES = list(BOARDS)
SEEDS = list(range(25))

# ---------------------------------------------------------------- tables (seeds 0-2 shipped) ----
import gzip, shutil, tempfile
from pathlib import Path
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.features import trigram_features_from_positions

def _shipped(stem):
    with tempfile.TemporaryDirectory() as td:
        for suf in (".json", ".meta.json"):
            with gzip.open(f"{SHIPPED}/{stem}{suf}.gz","rb") as s, open(Path(td)/f"{stem}{suf}","wb") as d:
                shutil.copyfileobj(s, d)
        return XGBoostTypingModel.load(str(Path(td)/f"{stem}.json"))

_vecs = None
def build(bi, tri):
    global _vecs
    ph = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(G.slots)]
    T2 = np.asarray(TableBigramScorer(bi, {}, target_wpm=WPM, chars=ph, geometry=G)._T, dtype=float)
    if _vecs is None:
        _vecs = np.vstack([trigram_features_from_positions(G,(a,b,c),wpm=WPM)
                           for a in POS for b in POS for c in POS])
    return T2, np.asarray(tri.predict_ms(_vecs).reshape(N,N,N), dtype=float)

T2s, Tcs = [], []
for s in SEEDS:
    if s <= 2:
        a, b = build(_shipped(f"bigram_reg31_seed{s}"), _shipped(f"trigram_cond31_seed{s}"))
    else:
        z = np.load(f"{TABLES}/tables_seed{s}.npz"); a, b = z["T2"], z["Tc"]
    T2s.append(a); Tcs.append(b)
log(f"loaded {len(T2s)} per-seed table pairs")

# ---------------------------------------------------------------- corpus + the three pricings ---
tri_freq = {k: v for k, v in load_frequencies(str(production_corpus_dir(None)/"trigrams.txt")).items()
            if len(k) == 3}
log(f"corpus blend-v1: {len(tri_freq)} trigrams, mass {sum(tri_freq.values())}")

obs = set()
with open(f"{E2E}/tristrokes31_cond_v1.tsv") as f:
    for line in f:
        r = line.split("\t", 2)
        if len(r) > 1: obs.add(r[1])
OBSCELL = np.zeros((N, N, N), dtype=bool)
for i in range(N):
    for j in range(N):
        for k in range(N):
            OBSCELL[i,j,k] = str((POS[i], POS[j], POS[k])) in obs
log(f"observed position-triples: {len(obs)} distinct in TSV; {OBSCELL.sum()} of {N**3} cells in-frame")

SLOT = {nm: ({ch:i for i,ch in enumerate(lay)} | {" ": N-1}) for nm, lay in BOARDS.items()}
# COMMON SUPPORT: trigrams typable by EVERY board (kills the charset-composition confound)
common = [ng for ng in tri_freq if all(all(c in SLOT[nm] for c in ng) for nm in NAMES)]
common_mass = sum(tri_freq[ng] for ng in common)
log(f"COMMON SUPPORT: {len(common)} trigrams, mass {common_mass} "
    f"({100*common_mass/sum(tri_freq.values()):.2f}% of corpus)")

# Precompute per-board index arrays once; then each pricing is a masked weighted mean.
def board_arrays(nm, ngrams):
    sl = SLOT[nm]; A=[];B=[];C=[];F=[]
    for ng in ngrams:
        try: a,b,c = sl[ng[0]], sl[ng[1]], sl[ng[2]]
        except KeyError: continue
        A.append(a);B.append(b);C.append(c);F.append(tri_freq[ng])
    return (np.array(A),np.array(B),np.array(C),np.array(F,dtype=float))

ALL_NG = list(tri_freq)
ARR = {"all": {nm: board_arrays(nm, ALL_NG) for nm in NAMES},
       "common": {nm: board_arrays(nm, common) for nm in NAMES}}
# observed-only = all-cells arrays filtered by OBSCELL
ARR["observed"] = {}
for nm in NAMES:
    a,b,c,f = ARR["all"][nm]; m = OBSCELL[a,b,c]
    ARR["observed"][nm] = (a[m],b[m],c[m],f[m])
for pr in ("all","observed","common"):
    log(f"pricing {pr:9s} mass: " + " ".join(f"{nm}={ARR[pr][nm][3].sum():.3g}" for nm in NAMES[:3]))

def mspc(nm, pricing, si):
    a,b,c,f = ARR[pricing][nm]
    return float(((T2s[si][a,b] + Tcs[si][a,b,c]) * f).sum() / f.sum())

X = {pr: {nm: np.array([mspc(nm, pr, s) for s in range(len(SEEDS))]) for nm in NAMES}
     for pr in ("all","observed","common")}
log("scored 13 boards x 25 seeds x 3 pricings")

# ---------------------------------------------------------------- D-battery ---------------------
D = {}
D["D2_distinct_per_seed"] = all(len(set(np.round(X["all"][nm],12))) == len(SEEDS) for nm in NAMES)
seedtb = json.load(open("/local/home/zegertho/agent/state/seedtb/artifacts/margins_n25.json"))
d4 = {nm: float(np.max(np.abs(np.array(seedtb["mspc"][nm]) - X["all"][nm])))
      for nm in ("arm-B","F(2.5)","BALL-1","F(2.0)","candidate")}
D["D4_vs_seedtb_worst_absdiff"] = max(d4.values()); D["D4_per_board"] = d4
D["D6_rows"] = {"bigram": 2202, "trigram": 16643}
log(f"D4: worst |diff| vs SEEDTB-1's 125 published per-seed values = {D['D4_vs_seedtb_worst_absdiff']:.3e}")

# ---------------------------------------------------------------- FLOOR-A (split-half placebo) --
# Truth is EXACTLY 0 by construction: same board, disjoint seed halves. Any spread = instrument.
def floor_a(pricing, n_part=2000):
    vals = []
    for nm in NAMES:
        x = X[pricing][nm]
        for _ in range(n_part):
            p = RNG.permutation(len(SEEDS)); h1, h2 = p[:12], p[12:24]
            vals.append(abs(x[h1].mean() - x[h2].mean()))
    v = np.array(vals)
    return {"p50": float(np.percentile(v,50)), "p90": float(np.percentile(v,90)),
            "p99": float(np.percentile(v,99)), "max": float(v.max()), "mean": float(v.mean()),
            "n": len(v), "half_n": 12}
FA = {pr: floor_a(pr) for pr in ("all","observed","common")}
for pr,v in FA.items():
    log(f"FLOOR-A {pr:9s} p50={v['p50']:.4f} p90={v['p90']:.4f} p99={v['p99']:.4f} max={v['max']:.4f}")

# ---------------------------------------------------------------- the pairwise tournament -------
def signflip_p(d, n_perm=20000):
    """FLOOR-B: exact-in-distribution sign-flip permutation null (distribution-free)."""
    obs_m = abs(d.mean())
    S = RNG.choice([-1.0,1.0], size=(n_perm, len(d)))
    return float((np.abs((S*d).mean(axis=1)) >= obs_m - 1e-15).mean())

def hamming(a,b): return sum(1 for x,y in zip(BOARDS[a],BOARDS[b]) if x!=y)

PAIRS = list(itertools.combinations(NAMES, 2))
log(f"{len(PAIRS)} unordered pairs")

def run_pricing(pricing):
    floor = FA[pricing]["p90"]
    rows = []
    for A,B in PAIRS:
        d = X[pricing][A] - X[pricing][B]
        n = len(d); mean = float(d.mean()); sd = float(d.std(ddof=1)); sem = sd/np.sqrt(n)
        t, p = stats.ttest_rel(X[pricing][A], X[pricing][B])
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
        pos = int((d > 0).sum()); neg = int((d < 0).sum())
        rows.append({"A":A,"B":B,"pair":f"{A} vs {B}","hamming":hamming(A,B),
                     "mean":mean,"sd":sd,"sem":float(sem),"t":float(t),"p_raw":float(p),
                     "ci":[float(ci[0]),float(ci[1])],"signs_pos_neg":[pos,neg],
                     "perm_p":signflip_p(d),"abs_mean":abs(mean),
                     "d_over_sd": abs(mean)/sd if sd>0 else float("inf")})
    # Holm within this pricing's family of 78
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    m = len(rows); holm_ok = True
    for rank,i in enumerate(order):
        thr = 0.05/(m-rank)
        rows[i]["holm_thr"] = thr
        rows[i]["holm_reject"] = bool(holm_ok and rows[i]["p_raw"] < thr)
        if not rows[i]["holm_reject"]: holm_ok = False
    for r in rows:
        r["bonf_reject"] = bool(r["p_raw"] < 0.05/(3*m))   # conservative across all 3 pricings
        r["above_floor"] = bool(r["abs_mean"] >= floor)
        r["sign_ok"] = bool(max(r["signs_pos_neg"]) >= 20)
        if r["holm_reject"] and r["above_floor"] and r["sign_ok"]:
            r["verdict"] = r["A"] if r["mean"] < 0 else r["B"]
            r["verdict_kind"] = "WIN"
        elif r["ci"][0] > -floor and r["ci"][1] < floor:
            r["verdict"] = "TIED"; r["verdict_kind"] = "TIED"
        else:
            r["verdict"] = "UNRESOLVED"; r["verdict_kind"] = "UNRESOLVED"
        r["pair_class"] = ("NEAR-CLONE" if r["hamming"] <= 6 else
                           "REAL MATCHUP" if r["hamming"] >= 20 else "INTERMEDIATE")
    return rows, floor

RES = {}
for pr in ("all","observed","common"):
    RES[pr], fl = run_pricing(pr)
    nw = sum(1 for r in RES[pr] if r["verdict_kind"]=="WIN")
    nt = sum(1 for r in RES[pr] if r["verdict_kind"]=="TIED")
    nu = sum(1 for r in RES[pr] if r["verdict_kind"]=="UNRESOLVED")
    log(f"pricing {pr:9s} floor={fl:.4f}  WIN={nw} TIED={nt} UNRESOLVED={nu}")

# ---------------------------------------------------------------- flips across pricings ---------
flips = []
for i,(A,B) in enumerate(PAIRS):
    v = {pr: RES[pr][i]["verdict"] for pr in ("all","observed","common")}
    if len(set(v.values())) > 1:
        flips.append({"pair":f"{A} vs {B}","hamming":RES['all'][i]['hamming'],"verdicts":v,
                      "means":{pr: RES[pr][i]["mean"] for pr in v}})
log(f"FLIPPED verdicts across pricings: {len(flips)} of {len(PAIRS)} pairs")

# ---------------------------------------------------------------- D7/D8 + Condorcet ------------
def cycles_from(rows):
    beat = {}   # beat[(A,B)] = True if A beats B
    for r in rows:
        if r["verdict_kind"]=="WIN":
            w = r["verdict"]; l = r["B"] if w==r["A"] else r["A"]
            beat[(w,l)] = True
    cyc = [ (a,b,c) for a,b,c in itertools.permutations(NAMES,3)
            if beat.get((a,b)) and beat.get((b,c)) and beat.get((c,a)) ]
    # D8 antisymmetry
    anti = all(not (beat.get((a,b)) and beat.get((b,a))) for a in NAMES for b in NAMES if a!=b)
    return cyc, anti, beat

CON = {}
for pr in ("all","observed","common"):
    cyc, anti, beat = cycles_from(RES[pr])
    means = {nm: float(X[pr][nm].mean()) for nm in NAMES}
    order = sorted(NAMES, key=lambda n: means[n])
    # is the WIN relation consistent with the total order by mean? (D7)
    consistent = all(means[w] < means[l] for (w,l) in beat)
    CON[pr] = {"n_3cycles": len(cyc)//3 if cyc else 0, "cycles": [list(c) for c in cyc[:12]],
               "antisymmetric_D8": anti, "consistent_with_total_order_D7": consistent,
               "mean_order": order, "means": means, "n_wins": len(beat)}
    log(f"CONDORCET {pr:9s}: 3-cycles={CON[pr]['n_3cycles']} antisym={anti} "
        f"consistent-with-M-order={consistent}  wins={len(beat)}")

# ---------------------------------------------------------------- FLOOR-C verdict stability ----
def verdicts_on(pricing, idx):
    fl = FA[pricing]["p90"]; out = {}
    for A,B in PAIRS:
        d = X[pricing][A][idx] - X[pricing][B][idx]
        n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1)); sem=sd/np.sqrt(n)
        p = stats.ttest_rel(X[pricing][A][idx], X[pricing][B][idx]).pvalue
        sig = p < 0.05/len(PAIRS)      # Bonferroni within the half (same strictness both halves)
        out[(A,B)] = (A if mean<0 else B) if (sig and abs(mean)>=fl) else "NOT-WIN"
    return out
h1, h2 = list(range(12)), list(range(12,24))
FC = {}
for pr in ("all","observed","common"):
    v1, v2 = verdicts_on(pr,h1), verdicts_on(pr,h2)
    dis = [f"{A} vs {B}: H1={v1[(A,B)]} H2={v2[(A,B)]}" for A,B in PAIRS if v1[(A,B)]!=v2[(A,B)]]
    contra = [s for s in dis if "NOT-WIN" not in s]
    FC[pr] = {"n_disagree":len(dis),"n_contradictory":len(contra),
              "contradictory":contra,"disagreements":dis[:25]}
    log(f"FLOOR-C {pr:9s}: {len(dis)}/{len(PAIRS)} half-vs-half disagreements, "
        f"{len(contra)} CONTRADICTORY (each half names a different winner)")

json.dump({"seeds":SEEDS,"boards":BOARDS,"n_pairs":len(PAIRS),
           "pricing_mass":{pr:{nm:float(ARR[pr][nm][3].sum()) for nm in NAMES}
                           for pr in ("all","observed","common")},
           "common_support":{"n_trigrams":len(common),"mass":common_mass},
           "mspc":{pr:{nm:X[pr][nm].tolist() for nm in NAMES} for pr in ("all","observed","common")},
           "floor_A":FA,"degeneracy":D,"pairs":RES,"flips":flips,"condorcet":CON,"floor_C":FC,
           "wall_s":time.time()-t0},
          open(f"{ART}/tournament.json","w"), indent=1)
log("ALL-DONE")
