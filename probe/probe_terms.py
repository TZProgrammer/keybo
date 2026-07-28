"""Per-term matched pricing for the 11 DEFAULT_OXEY_WEIGHTS terms, per source.

Instrument: the NATIVE fitted surfaces (not standardized — standardized substitutes
AALTO's bigram tensor into every source, verified exactly, so cross-source bigram
comparison there is meaningless). All three families where available.

Every contrast is MATCHED: the treatment and reference sets are restricted to share the
control signature so the difference is not a composition artifact.  Every contrast also
reports the disjointness detector from trap 16 and the strata fraction.
"""
import numpy as np, json, os, itertools
from keybo.geometry import ROW_STAGGERED_31, ROW_STAGGERED_30
from keybo.features import classify as C
from keybo.analysis.surfaces import C30M

NAT = "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
G = ROW_STAGGERED_31
SLOTS = G.slots            # 31 slots; index 30 == quote slot (6,2). Space is NOT a slot here.
N = len(SLOTS)
assert N == 31, N

def load(pool, fam, frame):
    p = f'{NAT}/{pool}_{fam}.{frame}.npy'
    if not os.path.exists(p): return None
    a = np.load(p)
    assert a.shape == (31,31,31), a.shape
    return a

# --- the surface's slot axis ---------------------------------------------------------
# surfaces.py: C30M is the char order, slot 30 is SPACE.  So axis index i<30 -> the
# geometry slot i of ROW_STAGGERED_30 is NOT right either: C30M is a CHARACTER order and
# the surface is indexed by SLOT.  Verify by construction: 31 slots where 30 = space.
SPACE = 30
GEO30 = ROW_STAGGERED_30
POS = list(GEO30.slots)          # 30 letter slots
assert len(POS) == 30

def hand(p): return G.hand(p[0])
def finger(p): return G.finger(p[0])
def row(p): return p[1]

# ---------------- landing/control signature -----------------------------------------
def land_sig(b):
    """The landing-key signature THEORY-1 matched on: (finger, row) of the landing key."""
    return (finger(b).name, row(b))

def pair_sig(a, b):
    return (land_sig(a), land_sig(b))

# ---------------- term predicates on BIGRAMS (positions) -----------------------------
def is_sfb(a,b):     return C.same_finger(G,a,b) and a != b
def is_lsb(a,b):     return C.is_lsb(G,a,b)
def is_scissor(a,b): return C.is_scissor(G,a,b)
def is_inroll(a,b):  return C.is_inwards(G,a,b)
def is_outroll(a,b): return C.is_outwards(G,a,b)
def is_alt(a,b):     return C.classify_positions(G,a,b) is C.BigramClass.ALTERNATE
def is_shb(a,b):     return C.classify_positions(G,a,b) is C.BigramClass.SAME_HAND
def is_flat_adj(a,b):
    return C.is_adjacent(G,a,b) and abs(a[1]-b[1]) == 0

# ---------------- the bigram marginal of a surface ----------------------------------
def bigram_marginal(S):
    """Mean over the third slot -> a 31x31 (a,b) table in ms.  This is the (a,b) price
    the optimizer sees averaged over what follows; it is the right object for a BIGRAM
    term because the oxey terms are bigram-share terms."""
    return S.mean(axis=2)

def report_contrast(name, note, Btab, treat, ref, control=pair_sig, nboot=2000, seed=0):
    """Matched difference of means with a STRATUM-clustered bootstrap CI.

    treat/ref: predicates over (a,b) positions. Matching: only strata (control signatures)
    present in BOTH sets contribute; the estimate is the unweighted mean over shared strata
    of (mean_treat - mean_ref) within the stratum.  Bootstrap resamples STRATA.
    """
    tre, rfe = {}, {}
    tkeys, rkeys = set(), set()
    for i,a in enumerate(POS):
        for j,b in enumerate(POS):
            if a == b: continue
            if treat(a,b):
                tre.setdefault(control(a,b), []).append(Btab[i,j]); tkeys |= {a,b}
            if ref(a,b):
                rfe.setdefault(control(a,b), []).append(Btab[i,j]); rkeys |= {a,b}
    shared = sorted(set(tre) & set(rfe))
    if not shared:
        return dict(name=name, note=note, n_strata=0, verdict="NO SHARED STRATUM",
                    key_overlap=len(tkeys & rkeys))
    per = np.array([np.mean(tre[k]) - np.mean(rfe[k]) for k in shared])
    rng = np.random.default_rng(seed)
    boots = np.array([per[rng.integers(0,len(per),len(per))].mean() for _ in range(nboot)])
    return dict(name=name, note=note,
                delta_ms=float(per.mean()),
                ci95=[float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))],
                n_strata=len(shared), frac_pos=float((per>0).mean()),
                n_treat_cells=int(sum(len(v) for v in tre.values())),
                n_ref_cells=int(sum(len(v) for v in rfe.values())),
                key_overlap=len(tkeys & rkeys),
                identified=bool(tkeys & rkeys))

CONTRASTS = [
    ("sfb  vs same-hand 2-finger", "the SFB penalty proper (matched landing sigs)",
        is_sfb, is_shb),
    ("sfb  vs alternate",          "SFB vs the fastest class",
        is_sfb, is_alt),
    ("lsb  vs same-hand non-lsb",  "lateral stretch",
        is_lsb, lambda a,b: is_shb(a,b) and not is_lsb(a,b)),
    ("scissor vs adjacent-finger FLAT", "the scissor (adjacency held)",
        is_scissor, is_flat_adj),
    ("inroll vs outroll",          "roll direction -- STRUCTURALLY UNREPRESENTABLE here",
        is_inroll, is_outroll),
    ("inroll vs same-hand non-roll", "is an inroll cheaper than generic same-hand?",
        is_inroll, lambda a,b: is_shb(a,b) and not is_inroll(a,b) and not is_outroll(a,b)),
    ("outroll vs same-hand non-roll","is an outroll cheaper than generic same-hand?",
        is_outroll, lambda a,b: is_shb(a,b) and not is_inroll(a,b) and not is_outroll(a,b)),
    ("alternate vs same-hand (any)", "hand alternation: the ONLY thing `alternate` can mean",
        is_alt, is_shb),
]

out = {"frame": "NATIVE fitted surfaces (per-source bigram tensor intact); "
                "bigram MARGINAL = S.mean(axis=2) in ms; g-frame only (no b(ngram)); "
                "baked 90 WPM", "contrasts": {}}
for fam in ('BASE','TRI_PS_FREQ_PRIOR'):
    for pool in ('AALTO','COMMUNITY','POOL'):
        S = load(pool, fam, 'native')
        if S is None:
            print(f'-- {pool}_{fam}.native MISSING, skipped'); continue
        B = bigram_marginal(S)
        for nm, note, t, r in CONTRASTS:
            res = report_contrast(nm, note, B, t, r)
            out["contrasts"].setdefault(nm, {})[f'{pool}_{fam}'] = res
print(f"{'contrast':36s} {'source':28s} {'delta_ms':>10s} {'ci95':>22s} {'strata':>7s} {'frac+':>6s} {'ident':>6s}")
for nm, per in out["contrasts"].items():
    for src, r in per.items():
        if r.get('n_strata',0)==0:
            print(f'{nm:36s} {src:28s} {"":>10s} {"NO SHARED STRATUM":>22s} {0:7d}')
            continue
        ci=f"[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}]"
        print(f"{nm:36s} {src:28s} {r['delta_ms']:+10.3f} {ci:>22s} {r['n_strata']:7d} {r['frac_pos']:6.2f} {str(r['identified']):>6s}")
    print()
json.dump(out, open('/local/home/zegertho/agent/state/scissorprice/artifacts/term_contrasts.json','w'), indent=1)
print('wrote term_contrasts.json')
