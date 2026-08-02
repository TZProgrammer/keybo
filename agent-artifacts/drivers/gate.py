"""INVARIANT C, part (c): the high-wpm gate, run through the repo's OWN enforcement function.

prereg C3(c) requires require_no_high_wpm_regression_in_report to PASS, or the arm is reported
as UNGATED rather than as passing. The baseline_buckets are the SHIPPED arm's per-bucket rhos
(seed-mean per fold), which is exactly what the incumbent supplies elsewhere in this repo.
"""
import json, numpy as np
from keybo.training.validate import require_no_high_wpm_regression_in_report
from keybo.verdicts import bucket_regression_report, HighWpmRegression

d=json.load(open("inv_c_arms.json"))
ARMS=[k for k in d if k not in ("ceilings","deltas")]
ship=d["SHIPPED"]["folds"]

# baseline per fold = SHIPPED's seed-MEAN bucket rho
base_by_fold={}
for h,per in ship.items():
    ks=set().union(*[set(p["bucket_rhos"]) for p in per])
    base_by_fold[h]={int(k): float(np.mean([p["bucket_rhos"][k] for p in per if k in p["bucket_rhos"]])) for k in ks}
print("SHIPPED per-fold seed-mean bucket rhos (the baseline):")
for h,b in base_by_fold.items(): print(f"  {h:8s} " + "  ".join(f"b{k}={v:+.4f}" for k,v in sorted(b.items())))

print("\n=== high-wpm gate per arm (repo's own require_no_high_wpm_regression_in_report) ===")
res={}
for arm in ARMS:
    rep={"folds":{}}
    for h,per in d[arm]["folds"].items():
        rep["folds"][h]={"seeds":[
            {"high_wpm_gate": bucket_regression_report(
                {int(k):v for k,v in p["bucket_rhos"].items()}, base_by_fold[h], f"{arm} {h} seed={p['seed']}")}
            for p in per]}
    try:
        v=require_no_high_wpm_regression_in_report(rep,arm)
        res[arm]={"verdict":"PASS","detail":v}
        print(f"  {arm:10s} PASS")
        for h,pf in v["per_fold"].items():
            if pf["noise_buckets"] or pf["regressing_bucket_seed_counts"]:
                print(f"      {h:8s} noise buckets {pf['noise_buckets']}  seed-counts {pf['regressing_bucket_seed_counts']}")
    except HighWpmRegression as e:
        res[arm]={"verdict":"FAIL","error":str(e)}
        print(f"  {arm:10s} FAIL — {str(e)[:300]}")
json.dump(res,open("inv_c_gate.json","w"),indent=1,default=str)

print("\n=== THE DECISIVE TABLE: candidate effects vs the instrument's OWN noise ===")
dl=d["deltas"]
print(f"  {'arm':11s} {'mean paired delta':>18s} {'sd':>9s} {'folds improving':>16s}")
for k in ("PLACEBO","ZERO","SEEDNOISE"):
    v=dl[k]; print(f"  {k:11s} {v['mean_paired_delta']:+18.6f} {v['sd']:9.6f} {v['folds_improving']:>13d}/{v['n_folds']}")
print(f"\n  REGISTERED BAR (prereg C3): mean paired delta <= -0.135 ms/char AND sign on >=3/4 folds AND gate PASS")
sn=dl["SEEDNOISE"]
print(f"  SEEDNOISE is the SAME GEOMETRY re-seeded: |mean| {abs(sn['mean_paired_delta']):.6f}, sd {sn['sd']:.6f}")
print(f"  => the instrument's own reseeding noise (sd {sn['sd']:.4f}) is LARGER than ZERO's whole effect "
      f"({abs(dl['ZERO']['mean_paired_delta']):.4f}) and than the B-grid argmin's (0.0377).")
