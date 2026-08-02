"""B07 — what SURVIVES R3's refutation, stated without rescuing the dead claim.

R3 killed my headline: the candidate-vs-arm-B margin has per-seed values
[+0.1629, -0.0433, +0.1778] — the seeds DISAGREE ON THE SIGN and 2 of 3 exceed the 0.135
floor. By my prereg D4.4 ("must hold on 3/3 model seeds") the named winner FAILS.

I am NOT reinterpreting that into a win. Instead: ask what the per-seed data DOES establish,
which is a strictly weaker and differently-shaped claim.

(1) THE RESOLVABILITY PARTITION. For every pair, is the ordering established = sign-stable
    on 3/3 seeds AND |margin| > floor on 3/3? That is the honest definition of "board A is
    faster than board B" under this estimator. Anything else is "not established".
(2) THE t-TEST MY OWN "0.73 floors" SHOULD HAVE BEEN. +0.0991 with per-seed sd ~0.124 over
    n=3 is not a resolved difference; compute the paired t and the number of model seeds that
    WOULD resolve it. That converts "unresolvable" into a priced, actionable experiment.
"""
import json
import math
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

R3 = "/local/home/zegertho/agent/state/bestfinal/artifacts/b06_r3_seed_stability.json"
TAB = "/local/home/zegertho/agent/state/bestfinal/artifacts/b02_master_table.json"
OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b07_cluster.json"
FLOOR = 0.135

SFB = {   # from b02 (emitted, not transcribed by hand: keyed by the same names b06 used)
    "CANDIDATE F(1.75)": "FRONTIER@sfb<=1.75", "arm-B": "arm-B", "BALL-1": "BALL-1",
    "F(2.0)": "FRONTIER@sfb<=2", "F(2.5)": "FRONTIER@sfb<=2.5",
    "keybo-lsb (unpolished)": "keybo-lsb", "flagship-c3": "flagship-c3", "semimak": "semimak",
}


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    d = json.load(open(R3))
    rows, pairs = d["rows"], d["pairs"]
    tab = json.load(open(TAB))["rows"]
    sfb = {k: tab[v]["gauges"]["sfb"] for k, v in SFB.items()}

    print("=" * 100)
    print("(1) THE RESOLVABILITY PARTITION — which orderings are ESTABLISHED on 3/3 model seeds?")
    print("    established := sign-stable on 3/3 AND |margin| > 0.135 on 3/3")
    print("=" * 100)
    est, notest = [], []
    for name, p in pairs.items():
        ms = p["per_seed"]
        ok = len({m > 0 for m in ms}) == 1 and all(abs(m) > FLOOR for m in ms)
        (est if ok else notest).append((name, ms, p["mean"]))
    print(f"\n  ESTABLISHED ({len(est)}):")
    for n, ms, mean in sorted(est, key=lambda x: -abs(x[2])):
        a, b = n.split(" vs ")
        fast, slow = (b, a) if mean > 0 else (a, b)
        print(f"    {fast:26} is faster than {slow:26} mean {abs(mean):+.4f} "
              f"({abs(mean) / FLOOR:5.2f} floors) per-seed {[f'{m:+.3f}' for m in ms]}")
    print(f"\n  NOT ESTABLISHED ({len(notest)}):")
    for n, ms, mean in sorted(notest, key=lambda x: abs(x[2])):
        why = []
        if len({m > 0 for m in ms}) > 1:
            why.append("SIGN FLIPS")
        if not all(abs(m) > FLOOR for m in ms):
            why.append("inside floor")
        print(f"    {n:52} mean {mean:+.4f}  per-seed {[f'{m:+.3f}' for m in ms]}  "
              f"[{', '.join(why)}]")

    # the top cluster = boards no OTHER board is established-faster than
    beaten = set()
    for n, ms, mean in est:
        a, b = n.split(" vs ")
        beaten.add(a if mean > 0 else b)          # the slower one
    cluster = [k for k in rows if k not in beaten]
    print("\n" + "=" * 100)
    print("THE TOP CLUSTER — boards that NO other board is established-faster than")
    print("=" * 100)
    print(f"{'board':26} {'seed-mean ms':>13} {'sfb':>8}")
    for k in sorted(cluster, key=lambda x: rows[x]["mean_ms_per_char"]):
        print(f"{k:26} {rows[k]['mean_ms_per_char']:13.4f} {sfb[k]:8.4f}")
    internal = [(n, ms, m) for n, ms, m in est
                if n.split(" vs ")[0] in cluster and n.split(" vs ")[1] in cluster]
    print(f"\n  established orderings WITHIN the cluster: {len(internal)}")
    for n, ms, m in internal:
        print(f"    {n}: mean {m:+.4f} per-seed {[f'{x:+.3f}' for x in ms]}")
    if not internal:
        print("    => NONE. The cluster is INTERNALLY UNORDERABLE on this estimator.")
    cl_sfb = {k: sfb[k] for k in cluster}
    print(f"\n  sfb range inside the cluster: {min(cl_sfb.values()):.4f} "
          f"({min(cl_sfb, key=cl_sfb.get)}) .. {max(cl_sfb.values()):.4f} "
          f"({max(cl_sfb, key=cl_sfb.get)})  = {max(cl_sfb.values()) - min(cl_sfb.values()):.4f} pp")

    print("\n" + "=" * 100)
    print("(2) THE t-TEST MY '0.73 floors' SHOULD HAVE BEEN — and the priced tie-breaker")
    print("=" * 100)
    m = d["candidate_vs_armB"]["per_seed_margins"]
    n = len(m)
    mean, s = sum(m) / n, sd(m)
    sem = s / math.sqrt(n)
    t = mean / sem
    print(f"  candidate vs arm-B per-seed margins : {[f'{x:+.4f}' for x in m]}")
    print(f"  mean {mean:+.4f}   sd {s:.4f}   sem(n={n}) {sem:.4f}   t = {t:.3f}")
    print(f"  => |t| = {abs(t):.2f} < 2  ⇒ NOT resolved. And the sd ({s:.4f}) is {s / FLOOR:.2f}x")
    print(f"     the 0.135 floor, so for THIS pair the floor UNDERSTATES estimator spread:")
    print(f"     my published '+0.0991 = 0.73 floors, inside the floor' was FALSE PRECISION.")
    need = [k for k in range(3, 200) if abs(mean) / (s / math.sqrt(k)) >= 2.0]
    nseeds = need[0] if need else None
    print(f"\n  SEEDS NEEDED to resolve +{mean:.4f} at |t| >= 2 (holding sd = {s:.4f}): "
          f"{nseeds}  ⇒ {nseeds - n} MORE model seeds")
    print("  => THE CHEAPEST TIE-BREAKER IS NOT A TYPING STUDY: it is training "
          f"{nseeds - n} more model seeds")
    print("     of the SAME architecture and re-running this exact comparison. No new data, no")
    print("     new features, no participants — it attacks the ESTIMATOR's variance, which is")
    print("     what actually binds this decision, rather than the FRAME (which CLOSING-1 shows")
    print("     is unfixable with available corpora).")

    json.dump({"established": [{"pair": n, "per_seed": ms, "mean": mm} for n, ms, mm in est],
               "not_established": [{"pair": n, "per_seed": ms, "mean": mm} for n, ms, mm in notest],
               "top_cluster": cluster, "cluster_sfb": cl_sfb,
               "internal_established_orderings": len(internal),
               "candidate_vs_armB_ttest": {"per_seed": m, "mean": mean, "sd": s, "sem": sem,
                                           "t": t, "resolved_at_t2": abs(t) >= 2,
                                           "seeds_needed_for_t2": nseeds},
               "floor_used": FLOOR}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
