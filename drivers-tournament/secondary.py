"""SECONDARY AXES + the multi-criterion Condorcet test (the ONLY place a cycle is well-posed).

Per TOURNAMENT-1 section 6: sfb (MEASURED corpus count), lat-span (MEASURED, the informative
lateral gauge), comfort (OPINION -- six hand-chosen weights; reported, NEVER decisive).
Then aggregate {speed, sfb, lat-span} by per-pair majority and enumerate all 286 triples.

APIs read from source, not guessed: kmstats.KmStats(bi,sk,tri).stats(lay30) ->
dict of percentages; lateral_span.LateralSpan(bigrams).share(Layout) -> percent;
comfort.ComfortBigramScorer(bigrams, skipgram_freqs=sk).fitness(Layout) -> ms-equivalents.
"""
import json, sys, time, itertools
import numpy as np
sys.path.insert(0, "/local/home/zegertho/agent/workspaces/tournament/wt/drivers-tournament")
from _guard import assert_d5, BOARDS, ART

t0=time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
log(f"D5 OK keybo={assert_d5()}")

NAMES=list(BOARDS)
from keybo.analysis.kmstats import KmStats
from keybo.analysis.lateral_span import LateralSpan
from keybo.scoring.comfort import ComfortBigramScorer, DEFAULT_COMFORT
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout

CD = production_corpus_dir(None)
bg = load_frequencies(str(CD/"bigrams.txt"))
sk = load_frequencies(str(CD/"1-skip31.txt"))
tri = load_frequencies(str(CD/"trigrams.txt"))
log(f"corpus: bigrams {len(bg)}  1-skip {len(sk)}  trigrams {len(tri)}")
log(f"comfort weights (OPINION): { {k:v for k,(v,_) in DEFAULT_COMFORT.items()} }")

km = KmStats(bg, sk, tri); lsp = LateralSpan(bg); cf = ComfortBigramScorer(bg, skipgram_freqs=sk)
rows={}
for nm,lay in BOARDS.items():
    L = Layout(lay, G); st = km.stats(lay)
    rows[nm] = {"sfb": float(st["sfb"]), "lsb": float(st["lsb"]),
                "lat_span": float(lsp.share(L)), "comfort_OPINION": float(cf.fitness(L))}
log("secondary gauges computed")

T=json.load(open(f"{ART}/tournament.json"))
for nm in NAMES: rows[nm]["speed"]=float(np.mean(T["mspc"]["all"][nm]))

print(f"\n{'board':14s} {'speed':>10s} {'sfb':>8s} {'lat-span':>9s} {'lsb':>7s} {'comfort(OP)':>13s}")
for nm in sorted(NAMES,key=lambda n:rows[n]["speed"]):
    r=rows[nm]
    print(f"{nm:14s} {r['speed']:10.4f} {r['sfb']:8.4f} {r['lat_span']:9.4f} {r['lsb']:7.4f} "
          f"{r['comfort_OPINION']:13.6g}")

# ---- the multi-criterion Condorcet test. All three axes: LOWER IS BETTER. ----
AX=("speed","sfb","lat_span")
def wins(a,b): return sum(1 for ax in AX if rows[a][ax] < rows[b][ax])
beat={}
for a,b in itertools.combinations(NAMES,2):
    wa=wins(a,b)
    if wa>=2: beat[(a,b)]=True
    elif wa<=1: beat[(b,a)]=True
cyc=[(a,b,c) for a,b,c in itertools.permutations(NAMES,3)
     if beat.get((a,b)) and beat.get((b,c)) and beat.get((c,a))]
uniq={frozenset(c) for c in cyc}
n_trip=len(list(itertools.combinations(NAMES,3)))
log(f"MULTI-CRITERION majority over {AX}: {len(uniq)} distinct 3-cycles of {n_trip} triples")
for c in sorted(uniq,key=lambda s:sorted(s))[:20]:
    trip=[x for x in cyc if frozenset(x)==c][0]
    log(f"  CYCLE  {trip[0]} > {trip[1]} > {trip[2]} > {trip[0]}")
cw=[n for n in NAMES if all(beat.get((n,m)) for m in NAMES if m!=n)]
log(f"multi-criterion Condorcet WINNER: {cw if cw else 'NONE'}")
cope={n: sum(1 for m in NAMES if m!=n and beat.get((n,m)))
        - sum(1 for m in NAMES if m!=n and beat.get((m,n))) for n in NAMES}
log("Copeland (3-axis majority): " + ", ".join(f"{n}={cope[n]:+d}" for n in sorted(NAMES,key=lambda x:-cope[x])))

# correlations among axes on THIS field (collider-contaminated; stated as such)
from scipy import stats as SS
cors={}
for a,b in itertools.combinations(("speed","sfb","lat_span","comfort_OPINION"),2):
    v1=[rows[n][a] for n in NAMES]; v2=[rows[n][b] for n in NAMES]
    cors[f"{a}~{b}"]={"pearson":float(np.corrcoef(v1,v2)[0,1]),
                      "spearman":float(SS.spearmanr(v1,v2).statistic),
                      "spearman_p":float(SS.spearmanr(v1,v2).pvalue)}
for k,v in cors.items(): log(f"  corr {k:34s} r={v['pearson']:+.4f} rho={v['spearman']:+.4f} p={v['spearman_p']:.3f}")

json.dump({"rows":rows,"axes":list(AX),
           "comfort_weights_OPINION":{k:v for k,(v,_) in DEFAULT_COMFORT.items()},
           "axis_provenance":{
             "speed":"MEASURED -- model prediction on a measured-keystroke surface",
             "sfb":"MEASURED -- corpus bigram count",
             "lat_span":"MEASURED -- corpus-weighted graded span (r=+0.3137 with speed, LSBNAME-1)",
             "comfort_OPINION":"OPINION -- six hand-chosen weights (sfb 25, scissor 15, "
                               "bottom_row 10, lsb 10, off_home 8, lag2_reuse 5). NOT a "
                               "measurement. Reported; NEVER decisive.",
             "lsb":"MEASURED but ~UNINFORMATIVE (r~0.08 with speed; r=0.9954 with lsb-dist => "
                   "the SAME gauge). EXCLUDED as a decision input per prereg section 6."},
           "multicriterion":{"n_distinct_3cycles":len(uniq),"n_triples":n_trip,
                             "cycles":[sorted(c) for c in sorted(uniq,key=lambda s:sorted(s))],
                             "condorcet_winner":cw,"copeland":cope},
           "axis_correlations_on_this_field_COLLIDER_CONTAMINATED":cors,
           "wall_s":time.time()-t0},
          open(f"{ART}/secondary.json","w"),indent=1)
log("ALL-DONE")
