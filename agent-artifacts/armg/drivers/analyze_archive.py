"""Did the ARM G search actually FIND in-band low-D layouts, and merely fail to RETURN one?

This is NOT post-hoc threshold tuning. The registered verdict (FAILURE by F1) stands. This
asks a different, mechanistic question of data ALREADY COLLECTED: each run archived its
final population's top 50. If those archives contain layouts that are in-band on my MEASURED
ruler AND have low D, then the free lunch is collectable and the defect is my EPS constant --
not the premise. If they do not, the premise itself is refuted.

The distinction matters: 'my band was mis-set' and 'the headroom is not collectable' are
different findings and only the archive can separate them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np  # noqa: E402
import search as S  # noqa: E402

RUNS=Path("/local/home/zegertho/agent/state/armg/artifacts/runs")
ART=Path("/local/home/zegertho/agent/state/armg/artifacts")
GAUGES=("sfb","sfs","sfb-dist","sfs-dist","lsb","lsb-dist","alt","roll","sr-roll","redir",
        "scissor","imbalance","oxey-style","comfort")
ARMB="flmpg-yuo,sntdcireahkxbwv'.jzq"

def main() -> int:
    with open(ART/"armg-judgement.json") as fh: J=json.load(fh)
    sd_G=J["ruler_MEASURED_NOT_BORROWED"]["sd_G"]; band=2*sd_G
    ARMB_MS=S.ARMG_REF_MS
    with open(RUNS/"armg-summary.json") as fh: summ=json.load(fh)

    # gather every archived layout from every run (armg AND baseline control)
    pool={}
    for r in summ["runs"]:
        if not r["ok"]: continue
        for e in r["top50"]:
            pool.setdefault(e["layout"], set()).add(r["arm"])
    print(f"archive pool: {len(pool)} distinct layouts across 10 runs' top50")

    # score them all through the FAST path (positive-controlled to 1.2e-14 vs shipped)
    import evobj as EV
    fe=EV.FastEval(corpus=None,weights_json=None,with_surface=True)
    assert str(Path(fe.corpus_dir).resolve()).startswith("/tmp/armg/")
    chk=S.armg_assert_constants(fe,list(S.ARMG_SIX))
    print(f"constants re-derived: {chk}")
    lays=sorted(pool)
    g=fe.gauges(np.stack([EV.perm_of(x) for x in lays]))
    ms=g["_ms_per_char"]; D=S.armg_deficit(g)

    edge=ARMB_MS+band
    inband=np.where(ms<=edge)[0]
    print(f"\nMEASURED verdict band edge = {edge:.6f} (arm B + 2*sd_G)")
    print(f"SEARCH band edge (EPS)     = {ARMB_MS+S.ARMG_EPS:.6f}  <-- LOOSER by "
          f"{S.ARMG_EPS-band:.6f}")
    print(f"archived layouts inside the MEASURED band: {len(inband)} of {len(lays)}")

    rows=[]
    for i in inband:
        rows.append({"layout":lays[i],"ms":float(ms[i]),"D":float(D[i]),
                     "found_by":sorted(pool[lays[i]])})
    rows.sort(key=lambda x:x["D"])
    print(f"\n{'rank':<5} {'D':>8} {'ms/char':>11} {'vs armB':>9} {'found_by':<20} layout")
    for k,r in enumerate(rows[:15],1):
        print(f"{k:<5} {r['D']:>8.4f} {r['ms']:>11.4f} {r['ms']-ARMB_MS:>+9.4f} "
              f"{','.join(r['found_by']):<20} {r['layout']}")

    best=rows[0] if rows else None
    # who found the best in-band low-D layout -- armg, the control, or both?
    armg_only=[r for r in rows if r["found_by"]==["armg"]]
    base_only=[r for r in rows if r["found_by"]==["baseline"]]
    both=[r for r in rows if len(r["found_by"])==2]
    print(f"\nin-band archived layouts by discoverer: armg-only {len(armg_only)}, "
          f"baseline-only {len(base_only)}, both {len(both)}")
    if armg_only: print(f"  best armg-only D  = {min(r['D'] for r in armg_only):.4f}")
    if base_only: print(f"  best baseline-only D = {min(r['D'] for r in base_only):.4f}")

    # the decisive question: is D=0 (a true dominator) anywhere in the pool AT ALL?
    zero=[lays[i] for i in range(len(lays)) if D[i]==0.0]
    print(f"\nlayouts with D EXACTLY 0 anywhere in the archive (excl arm B itself): "
          f"{[z for z in zero if z!=ARMB]}")
    print(f"  (arm B itself in pool: {ARMB in pool})")
    print(f"  global min D over the whole {len(lays)}-layout archive = {float(D.min()):.4f} "
          f"at ms {float(ms[int(np.argmin(D))]):.4f}")

    out={"sd_G":sd_G,"measured_band_edge":edge,"search_band_edge":ARMB_MS+S.ARMG_EPS,
         "band_looser_by":S.ARMG_EPS-band,"n_archive":len(lays),
         "n_inband":int(len(inband)),"inband_sorted_by_D":rows,
         "best_inband":best,"n_armg_only":len(armg_only),"n_baseline_only":len(base_only),
         "n_both":len(both),"global_min_D":float(D.min()),
         "zero_D_layouts_excl_armB":[z for z in zero if z!=ARMB],
         "constants_check":chk,
         "why":("Separates 'my EPS was mis-set' from 'the headroom is not collectable'. "
                "Uses ONLY data already collected by the registered runs; the registered "
                "verdict (FAILURE by F1) is unchanged by this analysis."),
         "modelled_only":"g-frame, baked 90 WPM, blend-v1, 1-skip31."}
    with open(ART/"armg-archive-analysis.json","w") as fh: json.dump(out,fh,indent=1)
    print(f"\nWROTE {ART/'armg-archive-analysis.json'}")
    return 0

if __name__=="__main__": sys.exit(main())
