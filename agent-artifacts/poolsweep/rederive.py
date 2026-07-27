"""Trap 20: re-derive every number the brief quotes, from the shipped machinery."""
import json, numpy as np
from pathlib import Path
import keybo.analysis.surfaces as S, keybo.analysis.evidence_scorer as E, keybo.analysis.evidence_validation as V

SD = "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces"
FM = "/local/home/zegertho/agent/state/keybo-optimization/artifacts/frontier_map.json"

# --- ASSERT native frame (brief constraint), and that native != standardized ---
sA = E.load_target_surface("AALTO_BASE", SD, "native")
sB = E.load_target_surface("COMMUNITY_BASE", SD, "native")
sAs = E.load_target_surface("AALTO_BASE", SD, "standardized")
sBs = E.load_target_surface("COMMUNITY_BASE", SD, "standardized")
assert sA.frame == "native" and sB.frame == "native", "FRAME ASSERT FAILED"
print("FRAME ASSERT: both native. sha A=%s B=%s" % (sA.sha256[:12], sB.sha256[:12]))
print("max|AALTO.native - AALTO.std|      = %.6e   (brief: EXACTLY 0.0)" % np.abs(sA.array-sAs.array).max())
print("max|COMMUNITY.native - COMMUNITY.std| = %.6e (brief: nonzero -> frames differ)" % np.abs(sB.array-sBs.array).max())

obj = S.trigram_objective(S.default_trigram_path(None))
ctx = E.gauge_context(None)
print("corpus = %s  sha256=%s" % (ctx.corpus_name, json.dumps(ctx.identity.get("sha256", {}))[:200]))

def ms(pool, surf): return E.surface_ms_per_trigram(pool, surf, obj)

# --- pools EXACTLY as the shipped _load_pool builds them ---
def random_pool(n, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.permutation(list(S.C30M))) for _ in range(n)]

def archive_pool(n, seed):
    rng = np.random.default_rng(seed)
    d = json.loads(Path(FM).read_text())
    entries = d.get("archive") or []
    cands = [e["layout"] if isinstance(e, dict) else e for e in entries]
    cands = [c for c in cands if S.is_c30m(c)]
    uniq = list(dict.fromkeys(cands))
    print("archive: %d entries -> %d c30m -> %d unique" % (len(entries), len(cands), len(uniq)))
    if len(uniq) > n:
        idx = rng.choice(len(uniq), n, replace=False)
        uniq = [uniq[i] for i in sorted(idx)]
    return uniq

for label, pool in (("random400 seed0", random_pool(400, 0)), ("archive400 seed0", archive_pool(400, 0))):
    yA, yB = ms(pool, sA), ms(pool, sB)
    r = V._spearman(yA, yB)
    print("\n%-18s n=%d  rho(AALTO_BASE,COMMUNITY_BASE) = %+.4f   sqrt = %.4f" % (label, len(pool), r, np.sqrt(max(r,0))))
    for nm, y in (("AALTO", yA), ("COMMUNITY", yB)):
        print("   %-10s ms/trigram mean %.4f  sd %.4f  min %.4f max %.4f  CV %.5f" % (
            nm, y.mean(), y.std(ddof=1), y.min(), y.max(), y.std(ddof=1)/y.mean()))
    # all 6 independent pairs (the MEAN is what EVSCORE-1's `ceiling` column reports)
    names = [n for n in ("AALTO_BASE","AALTO_TRI_PS_FREQ_PRIOR","COMMUNITY_BASE","COMMUNITY_FREQ_PRIOR","COMMUNITY_TRI_PS_FREQ_PRIOR","AALTO_FREQ_PRIOR")
             if (Path(SD)/f"{n}.native.npy").is_file()]
    tg = {n: ms(pool, E.load_target_surface(n, SD, "native")) for n in names}
    ag = V.cross_source_agreement(tg)
    print("   surfaces present: %s" % names)
    print("   ceiling MEAN over %d independent pairs = %+.4f (min %+.4f max %+.4f)" % (len(ag["pairwise"]), ag["mean"], ag["min"], ag["max"]))
    for k, v in ag["pairwise"].items(): print("      %-48s %+.4f" % (k, v))
