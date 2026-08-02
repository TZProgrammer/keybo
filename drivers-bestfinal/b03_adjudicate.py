"""B03 — apply the PREREGISTERED decision rule (D2, D3, D4, D5) to the master table.

Nothing here chooses a threshold: every floor, gate and tie-break order comes from
`state/bestfinal/PREREGISTRATION.md`, written before the frontier LEVELS were read.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

T = json.load(open("/local/home/zegertho/agent/state/bestfinal/artifacts/b02_master_table.json"))
OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b03_adjudication.json"
ROWS = T["rows"]
SEED, SEARCH = 0.135, 0.883

# LOWER-is-better for every gauge here EXCEPT these, per the campaign's conventions:
#   roll, sr-roll, alt  -> HIGHER is better (desirable patterns)
#   oxey-style          -> LOWER is better (oxey.py:92 "Positive = penalty") <-- the trap
#                          BESTLAYOUT-1 mis-signed and the campaign propagated 3x
#   comfort             -> LOWER is better (a ms-equivalent penalty sum)
#   imbalance, sfr      -> HAND/CHARSET INVARIANTS on a shared partition (GAUGEAUDIT-1 F1):
#                          NOT layout gauges. EXCLUDED from every ranking below.
HIGHER_BETTER = {"roll", "sr-roll", "alt"}
EXCLUDED_INVARIANT = {"sfr", "imbalance"}
FIELD_13 = [k for k in ROWS if not k.startswith("FRONTIER")]
FRONTIERS = sorted((k for k in ROWS if k.startswith("FRONTIER")),
                   key=lambda k: ROWS[k]["gauges"]["sfb"])
CHAMPION = "arm-B"


def gauge_axes():
    ax = [g for g in T["gauge_names"] if g not in EXCLUDED_INVARIANT]
    return ax


def better(name, a, b):
    """Is a better than b on gauge `name`?  Returns the signed improvement (a over b)."""
    return (a - b) if name in HIGHER_BETTER else (b - a)


def dominance(x, y, axes):
    """Does x dominate y on ms/char + the live gauge axes? (weak dominance, >=1 strict)"""
    wins = ties = losses = 0
    detail = {}
    dm = ROWS[y]["ms_per_char"] - ROWS[x]["ms_per_char"]  # >0 means x faster
    detail["ms_per_char"] = dm
    if dm > 1e-9:
        wins += 1
    elif dm < -1e-9:
        losses += 1
    else:
        ties += 1
    for g in axes:
        d = better(g, ROWS[x]["gauges"][g], ROWS[y]["gauges"][g])
        detail[g] = d
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "ties": ties, "losses": losses,
            "dominates": losses == 0 and wins > 0, "detail": detail}


def main():
    axes = gauge_axes()
    out = {"axes": axes, "excluded_invariant": sorted(EXCLUDED_INVARIANT)}

    print("=" * 100)
    print("D2 — THE FRONTIER / EXISTENCE TEST  (F(c) - F(inf), vs BOTH floors)")
    print("=" * 100)
    finf = ROWS[CHAMPION]["ms_per_char"]
    print(f"F(inf) = {finf:.6f} = {CHAMPION} (recovered from random starts by PRICEBAND-1)\n")
    print(f"{'board':26} {'sfb':>8} {'ms/char':>11} {'d vs arm-B':>11} "
          f"{'seed fl':>9} {'search fl':>10} {'INSIDE?':>18}")
    d2 = {}
    for k in FRONTIERS + [CHAMPION]:
        r = ROWS[k]
        d = r["ms_per_char"] - finf
        ins = ("inside BOTH" if abs(d) < SEED else
               "inside SEARCH only" if abs(d) < SEARCH else "OUTSIDE both")
        d2[k] = {"sfb": r["gauges"]["sfb"], "ms": r["ms_per_char"], "delta": d,
                 "seed_floors": d / SEED, "search_floors": d / SEARCH, "inside": ins}
        print(f"{k:26} {r['gauges']['sfb']:8.4f} {r['ms_per_char']:11.4f} {d:+11.4f} "
              f"{d / SEED:+9.3f} {d / SEARCH:+10.3f} {ins:>18}")
    out["D2"] = d2

    # The prereg's c* question: strictest cap whose cost is inside the applicable floor.
    free_seed = [k for k in FRONTIERS if abs(d2[k]["delta"]) < SEED]
    free_search = [k for k in FRONTIERS if abs(d2[k]["delta"]) < SEARCH]
    best_seed = min(free_seed, key=lambda k: d2[k]["sfb"]) if free_seed else None
    best_search = min(free_search, key=lambda k: d2[k]["sfb"]) if free_search else None
    out["D2_verdict"] = {
        "strictest_cap_free_at_seed_floor": best_seed,
        "sfb_there": d2[best_seed]["sfb"] if best_seed else None,
        "strictest_cap_free_at_search_floor": best_search,
        "sfb_there": d2[best_search]["sfb"] if best_search else None,
        "PASS": bool(best_seed),
    }
    print(f"\n=> D2 {'PASS' if best_seed else 'FAIL'}: the strictest sfb cap achievable at NO "
          f"resolvable speed cost is\n   sfb <= {d2[best_seed]['sfb']:.4f} "
          f"(model-seed floor) / sfb <= {d2[best_search]['sfb']:.4f} (search-seed floor)")
    print(f"   vs the champion's own sfb of {ROWS[CHAMPION]['gauges']['sfb']:.4f} "
          f"=> {ROWS[CHAMPION]['gauges']['sfb'] - d2[best_seed]['sfb']:.4f} pp of sfb is FREE")

    print("\n" + "=" * 100)
    print("D4.2 — THE SPEED-ADMISSIBLE SET (within the applicable floor of the champion)")
    print("=" * 100)
    adm = {}
    for k, r in ROWS.items():
        d = r["ms_per_char"] - finf
        fl = SEARCH if k.startswith("FRONTIER") else SEED
        adm[k] = {"delta": d, "floor_used": fl,
                  "floor_name": "search-seed" if k.startswith("FRONTIER") else "model-seed",
                  "admissible": abs(d) < fl}
    print(f"{'board':26} {'ms/char':>11} {'d vs arm-B':>11} {'floor':>13} {'ADMISSIBLE':>12}")
    for k in sorted(ROWS, key=lambda x: ROWS[x]["ms_per_char"]):
        a = adm[k]
        print(f"{k:26} {ROWS[k]['ms_per_char']:11.4f} {a['delta']:+11.4f} "
              f"{a['floor_name']:>13} {'YES' if a['admissible'] else 'no':>12}")
    out["D4_admissible"] = adm
    A = [k for k in ROWS if adm[k]["admissible"]]
    print(f"\nspeed-admissible set ({len(A)}): {sorted(A)}")

    print("\n" + "=" * 100
          + "\nD4.3 — COMFORT WITHIN THE SPEED-ADMISSIBLE SET (sfb leads: oxey weight 12.0)")
    print("=" * 100)
    print(f"{'board':26} {'sfb':>8} {'sfb-dist':>9} {'sfs':>8} {'lsb':>8} {'scissor':>8} "
          f"{'oxey-style':>11} {'comfort':>9} {'roll':>8}")
    for k in sorted(A, key=lambda x: ROWS[x]["gauges"]["sfb"]):
        g = ROWS[k]["gauges"]
        print(f"{k:26} {g['sfb']:8.4f} {g['sfb-dist']:9.4f} {g['sfs']:8.4f} {g['lsb']:8.4f} "
              f"{g['scissor']:8.4f} {g['oxey-style']:11.4f} {g['comfort']:9.4f} {g['roll']:8.3f}")

    print("\n" + "=" * 100)
    print("DOMINANCE — does the best speed-admissible low-sfb board dominate the field?")
    print(f"axes = ms/char + {len(axes)} live gauges (sfr/imbalance EXCLUDED: "
          "hand/charset invariants, GAUGEAUDIT-1 F1)")
    print("=" * 100)
    cands = sorted(A, key=lambda x: ROWS[x]["gauges"]["sfb"])[:3] + [CHAMPION]
    dom = {}
    for c in dict.fromkeys(cands):
        dom[c] = {}
        print(f"\n--- {c}  (ms {ROWS[c]['ms_per_char']:.4f}, sfb {ROWS[c]['gauges']['sfb']:.4f}) ---")
        print(f"{'vs':26} {'W':>3} {'T':>3} {'L':>3}  {'dominates?':>11}  losing axes")
        ndom = 0
        for y in sorted(FIELD_13):
            if y == c:
                continue
            d = dominance(c, y, axes)
            dom[c][y] = d
            lose = [f"{g}({v:+.3g})" for g, v in d["detail"].items() if v < -1e-9]
            if d["dominates"]:
                ndom += 1
            print(f"{y:26} {d['wins']:3d} {d['ties']:3d} {d['losses']:3d}  "
                  f"{'YES' if d['dominates'] else 'no':>11}  "
                  f"{', '.join(lose[:4]) if lose else '-'}")
        print(f"  => dominates {ndom} of {len(FIELD_13) - (1 if c in FIELD_13 else 0)} field boards")
        dom[c]["_n_dominated"] = ndom
    out["dominance"] = dom

    print("\n" + "=" * 100)
    print("D4.6 — PER-FINGER LOAD (% of total keystroke time; weak fingers first)")
    print("=" * 100)
    fingers = sorted(next(iter(ROWS.values()))["per_finger_pct"])
    order = [f for f in fingers if "pinky" in f] + [f for f in fingers if "ring" in f] + \
            [f for f in fingers if "middle" in f] + [f for f in fingers if "index" in f] + \
            [f for f in fingers if "thumb" in f]
    order += [f for f in fingers if f not in order]
    print(f"{'board':26} " + " ".join(f"{f[:9]:>10}" for f in order))
    for k in sorted(ROWS, key=lambda x: ROWS[x]["ms_per_char"]):
        pf = ROWS[k]["per_finger_pct"]
        print(f"{k:26} " + " ".join(f"{pf[f]:10.3f}" for f in order))
    out["finger_order"] = order

    print("\n" + "=" * 100)
    print("D4.5 — DISTANCE FROM OWN 2-OPT OPTIMUM (the Hamming caveat column)")
    print("=" * 100)
    print(f"{'board':26} {'class':>10} {'n_impr':>7} {'polish gain':>12} {'seed fl':>8} "
          f"{'HAMMING moved':>14} {'at own opt?':>12}")
    for k in sorted(ROWS, key=lambda x: ROWS[x]["ms_per_char"]):
        t = ROWS[k]["twoopt"]
        print(f"{k:26} {ROWS[k]['class']:>10} {t['n_improving_2opt']:7d} "
              f"{t['polish_gain']:12.4f} {t['polish_gain_seed_floors']:8.2f} "
              f"{t['hamming_moved']:14d} {'YES' if t['at_own_2opt_optimum'] else 'no':>12}")

    print("\n" + "=" * 100)
    print("THE THREE FITTED SURFACES (ranks; 1 = fastest) — the frame that must not be "
          "over-trusted (threat #7)")
    print("=" * 100)
    sr = {}
    for s in ("AALTO", "COMMUNITY", "POOL"):
        o = sorted(ROWS, key=lambda k: ROWS[k]["surfaces"][s])
        for i, k in enumerate(o, 1):
            sr.setdefault(k, {})[s] = i
    print(f"{'board':26} {'AALTO':>9} {'COMM':>9} {'POOL':>9}   raw (Gms)")
    for k in sorted(ROWS, key=lambda x: sr[x]["AALTO"]):
        print(f"{k:26} {sr[k]['AALTO']:9d} {sr[k]['COMMUNITY']:9d} {sr[k]['POOL']:9d}   "
              + " ".join(f"{ROWS[k]['surfaces'][s] / 1e9:9.4f}"
                         for s in ("AALTO", "COMMUNITY", "POOL")))
    out["surface_ranks"] = sr

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
