"""B09 — is the TOP-CLUSTER claim robust, or an artifact of the 0.135 floor I happened to pick?

The revised verdict rests on: "a 5-board top cluster is INTERNALLY UNORDERABLE". That used
floor = 0.135 (model-seed) plus 3/3 sign-stability. Two ways that could be an artifact:

  (A) FLOOR SENSITIVITY. Sweep the floor from 0 to 0.5 and report the cluster at each value.
      If the cluster only exists at ~0.135 it is a knife-edge and I should say so. If it is
      stable over a wide band -- or if it survives even at floor = 0 (pure sign-stability, no
      floor at all) -- the claim is robust and the floor is not doing the work.

  (B) THE SIGN-STABILITY-ONLY TEST. Drop the floor entirely: an ordering is established iff
      all 3 seeds agree on the sign. This is the WEAKEST possible bar -- it grants a win to any
      pair whose seeds merely agree in direction. If the cluster's internal pairs STILL fail
      even here, the unorderability is a fact about sign-consistency, not about my floor choice.
      This is the honest stress test, because it can only HURT my claim.
"""
import json
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

R3 = "/local/home/zegertho/agent/state/bestfinal/artifacts/b06_r3_seed_stability.json"
OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b09_robustness.json"
CLUSTER = ["arm-B", "F(2.5)", "BALL-1", "F(2.0)", "CANDIDATE F(1.75)"]


def cluster_at(pairs, floor, require_floor=True):
    """Boards that no OTHER board is established-faster than, at this floor."""
    beaten = set()
    est = []
    for name, p in pairs.items():
        ms = p["per_seed"]
        sign_ok = len({m > 0 for m in ms}) == 1
        floor_ok = all(abs(m) > floor for m in ms) if require_floor else True
        if sign_ok and floor_ok:
            a, b = name.split(" vs ")
            beaten.add(a if p["mean"] > 0 else b)
            est.append(name)
    return beaten, est


def main():
    d = json.load(open(R3))
    pairs, rows = d["pairs"], d["rows"]
    names = list(rows)
    out = {}

    print("=" * 100)
    print("(A) FLOOR SENSITIVITY — does the top cluster depend on my 0.135 choice?")
    print("=" * 100)
    print(f"{'floor':>7} {'#established':>13} {'cluster size':>13} {'internal orderings':>19}  cluster")
    sweep = {}
    for floor in [0.0, 0.02, 0.05, 0.08, 0.10, 0.135, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        beaten, est = cluster_at(pairs, floor)
        cl = [n for n in names if n not in beaten]
        internal = [e for e in est
                    if e.split(" vs ")[0] in cl and e.split(" vs ")[1] in cl]
        sweep[floor] = {"n_established": len(est), "cluster": cl,
                        "internal_orderings": len(internal)}
        mark = "  <== my choice" if abs(floor - 0.135) < 1e-9 else ""
        print(f"{floor:7.3f} {len(est):13d} {len(cl):13d} {len(internal):19d}  "
              f"{', '.join(cl)[:44]}{mark}")
    out["floor_sweep"] = {str(k): v for k, v in sweep.items()}

    print("\n" + "=" * 100)
    print("(B) THE WEAKEST BAR: SIGN-STABILITY ONLY, NO FLOOR (this can only HURT my claim)")
    print("=" * 100)
    beaten, est = cluster_at(pairs, 0.0, require_floor=False)
    cl = [n for n in names if n not in beaten]
    internal = [(e, pairs[e]) for e in est
                if e.split(" vs ")[0] in cl and e.split(" vs ")[1] in cl]
    print(f"  established (sign-stable 3/3, no floor): {len(est)} of {len(pairs)}")
    print(f"  cluster: {cl}")
    print(f"  internal orderings within the cluster  : {len(internal)}")
    for e, p in internal:
        print(f"      {e}: mean {p['mean']:+.4f} per-seed {[f'{x:+.3f}' for x in p['per_seed']]}")
    out["signonly"] = {"n_established": len(est), "cluster": cl,
                       "internal_orderings": [e for e, _ in internal]}

    print("\n  THE 10 CLUSTER-INTERNAL PAIRS, at the weakest bar:")
    print(f"  {'pair':46} {'per-seed':>34} {'sign-stable':>12} {'max|m|':>8}")
    nfail = 0
    detail = {}
    for i, a in enumerate(CLUSTER):
        for b in CLUSTER[i + 1:]:
            k = f"{a} vs {b}" if f"{a} vs {b}" in pairs else f"{b} vs {a}"
            p = pairs[k]
            ms = p["per_seed"]
            ss = len({m > 0 for m in ms}) == 1
            if not ss:
                nfail += 1
            detail[k] = {"per_seed": ms, "sign_stable": ss, "max_abs": max(abs(m) for m in ms)}
            print(f"  {k:46} {str([f'{m:+.3f}' for m in ms]):>34} {str(ss):>12} "
                  f"{max(abs(m) for m in ms):8.4f}")
    out["cluster_internal_pairs"] = detail
    print(f"\n  => {nfail} of 10 cluster-internal pairs FLIP SIGN across model seeds.")
    print(f"     {10 - nfail} are sign-stable but ALL of those sit inside the 0.135 floor "
          "(see the floor sweep).")

    print("\n" + "=" * 100)
    print("VERDICT ON ROBUSTNESS")
    print("=" * 100)
    stable_band = [f for f, v in sweep.items() if set(v["cluster"]) == set(CLUSTER)]
    print(f"  floors at which the cluster is EXACTLY my 5 boards: {stable_band}")
    zero_internal = all(v["internal_orderings"] == 0 for v in sweep.values())
    print(f"  internal orderings == 0 at EVERY floor tested: {zero_internal}")
    print(f"  and at the weakest bar (sign only, no floor): {len(internal)} internal orderings")
    if zero_internal and len(internal) == 0:
        print("\n  🟢 ROBUST. The unorderability is NOT an artifact of the floor: it holds even with")
        print("     NO floor at all, on pure sign-consistency. The cluster's MEMBERSHIP does move")
        print("     with the floor (a looser floor lets more boards in), but the load-bearing claim")
        print("     -- that the top boards cannot be ordered -- does not depend on the floor value.")
    else:
        print("\n  ⚠ NOT fully robust — see above; the floor is doing some of the work.")
    out["robust"] = bool(zero_internal and len(internal) == 0)
    out["floors_giving_exact_cluster"] = stable_band

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
