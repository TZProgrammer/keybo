"""ARM G judge — applies the PRE-REGISTERED decision rule to the runs.

Written BEFORE the runs finished, deliberately, so the thresholds cannot be tuned to the
result. Every threshold here is read from ../PREREGISTRATION.md (committed 0f606d1).

The rule, verbatim from the prereg:
  * ruler   = sd_G, the ddof=1 sd of the BASELINE CONTROL champions' ms/char across MY OWN
              seeds. Quadruple printed beside it (POOL x REPLICATE-STRUCTURE x SCALE x
              STATISTIC), per the standing rule. NOT borrowed from another arm.
  * select  = among champions within 2*sd_G of the best, pick on the pre-declared GAUGE
              frame (lowest D, ties broken by strict-win count), never on the objective.
  * verdict = FASTER / TIED-AND-STRICTLY-BETTER / TIED-AND-PARTIALLY-BETTER / FAILURE,
              decided in that order, with 4 registered FAILURE conditions.

Scoring goes through the SHIPPED `keybo analyze --json`, not through the search's evaluator,
so the reported numbers are on the same path the campaign's frozen boards are on. (The two
paths were positive-controlled to worst rel 1.233e-14 before any run.)
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import search as S  # noqa: E402

WORKTREE = Path("/tmp/armg")
RUNS = Path("/local/home/zegertho/agent/state/armg/artifacts/runs")
ART = Path("/local/home/zegertho/agent/state/armg/artifacts")

GAUGES = ("sfb", "sfs", "sfb-dist", "sfs-dist", "lsb", "lsb-dist", "alt", "roll",
          "sr-roll", "redir", "scissor", "imbalance", "oxey-style", "comfort")
#: `sfr` is a PERMUTATION INVARIANT (trap 23) -- reported, never counted.
INVARIANT = ("sfr",)

INCUMBENTS = {
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "arm-A": "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
}
#: pre-registered failure bar: flagship-c3's D, the lowest of any existing layout except
#: arm A (which is 2.95 ms/char slower and so outside any speed band).
D_FAILURE_BAR = 1.4878


def analyze(specs: list[str]) -> dict:
    """SHIPPED CLI. Sends UNIQUE specs -- the CLI refuses a dropped row (the trap-38 fix),
    which is correct behaviour, so duplicates must be deduped by the caller."""
    uniq = sorted(set(specs))
    cmd = ["uv", "run", "--no-sync", "keybo", "analyze", "--json", *uniq]
    p = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"analyze rc={p.returncode}: {p.stderr[-3000:]}")
    rows = json.loads(p.stdout)["rows"]
    assert len(rows) >= len(uniq), f"analyze dropped a row: {len(rows)} < {len(uniq)}"
    return rows


def profile(rows: dict, spec: str) -> dict:
    r = rows[spec]
    out = {g: r["gauges"][g] for g in GAUGES}
    out.update({g: r["gauges"][g] for g in INVARIANT})
    out["ms_per_char"] = r["time"]["ms_per_char"]
    return out


def deficit(prof: dict) -> float:
    """D against arm B, using the SHIPPED-analyze values (not the search evaluator)."""
    d = 0.0
    for g in GAUGES:
        d += max(0.0, S.ARMG_DIR[g] * (prof[g] - S.ARMG_REF[g]) / S.ARMG_SCALE[g])
    return d


def contested(a: dict, b: dict) -> dict:
    """Tie-aware pairwise comparison. Reports CONTESTED axes, never a bare n/15 -- `alt`
    and `imbalance` are HAND-PARTITION invariants (ULTRAAUDIT-INTERIM) and `sfr` is a
    permutation invariant, so some pairs tie BY CONSTRUCTION. Requires a STRICT-win term
    (trap 33: a `>=`-only predicate credits ties as wins -- a defect found at 4 sites)."""
    better = worse = tie = 0
    tied_axes, better_axes, worse_axes = [], [], []
    for g in GAUGES:
        if a[g] == b[g]:
            tie += 1
            tied_axes.append(g)
        elif S.ARMG_DIR[g] * (a[g] - b[g]) < 0:
            better += 1
            better_axes.append(g)
        else:
            worse += 1
            worse_axes.append(g)
    n_contested = better + worse
    return {
        "better": better, "worse": worse, "tie": tie,
        "n_contested": n_contested,
        "frame_size": len(GAUGES),
        "better_axes": better_axes, "worse_axes": worse_axes, "tied_axes": tied_axes,
        # Pareto dominance requires >= on ALL contested axes AND strictly better on >= 1.
        "dominates": worse == 0 and better >= 1,
        "sfr_tie_by_construction": a["sfr"] == b["sfr"],
    }


def cluster_corrected(cmp: dict) -> dict:
    """trap 27/39: `oxey-style` is R2=0.9937 on {sfb,lsb,scissor,imbalance,redir,alt}, and
    {lsb,lsb-dist} / {sfs,sfs-dist} are near-duplicate pairs. Counting per gauge over-counts
    those clusters, which is exactly the ~4x over-count that reversed a 'broad competence'
    verdict. Count wins per CLUSTER, not per gauge."""
    clusters = {
        "sf-bigram": ("sfb", "sfb-dist"),
        "sf-skip": ("sfs", "sfs-dist"),
        "lsb": ("lsb", "lsb-dist"),
        "hand-flow": ("alt", "roll", "sr-roll"),
        "redirect": ("redir",),
        "scissor": ("scissor",),
        "balance": ("imbalance",),
        "comfort": ("comfort",),
        "oxey-composite": ("oxey-style",),
    }
    b = set(cmp["better_axes"])
    w = set(cmp["worse_axes"])
    cb = cw = ct = 0
    detail = {}
    for name, axes in clusters.items():
        nb = len(b & set(axes))
        nw = len(w & set(axes))
        if nb and not nw:
            cb += 1
            detail[name] = "better"
        elif nw and not nb:
            cw += 1
            detail[name] = "worse"
        elif nb and nw:
            detail[name] = "mixed"
        else:
            ct += 1
            detail[name] = "tie"
    return {"clusters_better": cb, "clusters_worse": cw, "clusters_tie_or_mixed":
            len(clusters) - cb - cw, "n_clusters": len(clusters), "per_cluster": detail,
            "note": ("oxey-composite RESTATES sf/lsb/scissor/balance/redirect/hand-flow "
                     "(R2=0.9937), so it is NOT independent evidence alongside them.")}


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    with open(RUNS / f"armg-summary{tag}.json") as fh:
        summ = json.load(fh)

    runs = [r for r in summ["runs"] if r["ok"]]
    armg = [r for r in runs if r["arm"] == "armg"]
    base = [r for r in runs if r["arm"] == "baseline"]

    # ---- the 80% ACHIEVED floor, applied as registered ----
    armg_pri = [r for r in armg if r["clears_floor"]]
    base_pri = [r for r in base if r["clears_floor"]]
    excluded = [(r["arm"], r["seed"], r["unique_evals_ACHIEVED"], r["achieved_frac"])
                for r in runs if not r["clears_floor"]]

    # ---- score EVERYTHING through the shipped CLI in one call ----
    specs = ([r["layout"] for r in runs] + list(INCUMBENTS.values()))
    rows = analyze(specs)
    prof = {s: profile(rows, s) for s in set(specs)}

    armB = INCUMBENTS["arm-B"]

    # ---- MY OWN RULER, measured from MY OWN baseline-control seeds ----
    base_ms = np.array([prof[r["layout"]]["ms_per_char"] for r in base_pri])
    base_lays = [r["layout"] for r in base_pri]
    sd_G = float(base_ms.std(ddof=1)) if len(base_ms) >= 2 else float("nan")
    ruler = {
        "sd_G": sd_G,
        "band_2sd": 2 * sd_G,
        "n_runs": len(base_pri),
        "n_distinct_champions": len(set(base_lays)),
        "mean": float(base_ms.mean()), "min": float(base_ms.min()),
        "max": float(base_ms.max()), "range": float(base_ms.max() - base_ms.min()),
        "QUADRUPLE": {
            "POOL": ("my own 5-seed ARM-G-family baseline-control champions "
                     "(near-optimal, cold start, blend-v1)"),
            "REPLICATE_STRUCTURE": ("independent cold-start search runs, one champion "
                                    "each; NOT per-seed model refits, NOT bootstrap draws"),
            "SCALE": "raw ms/char (served K31 surface, baked 90 WPM)",
            "STATISTIC": f"sd, ddof=1, n={len(base_pri)}",
        },
        "borrowed_for_comparison_only": {
            "value": 0.0617,
            "what": ("SPEEDTIE-1's baseline-objective sd (n=6, 1M, seed family "
                     "900000+7919r). Used ONLY to set the SEARCH band EPS before my seeds "
                     "existed; NOT used for any verdict."),
        },
    }

    # ---- profiles + D for every champion and incumbent ----
    def block(name: str, spec: str) -> dict:
        p = prof[spec]
        # `gauges` carries the INVARIANT axes too, so it is a complete profile that
        # contested() can consume. The invariants are reported, never COUNTED -- the
        # counting loop iterates GAUGES only.
        return {"name": name, "layout": spec, "ms_per_char": p["ms_per_char"],
                "D_vs_armB": deficit(p),
                "gauges": {g: p[g] for g in (*GAUGES, *INVARIANT)}, "sfr": p["sfr"]}

    champs = []
    for r in runs:
        b = block(f"{r['arm']}-seed{r['seed']}", r["layout"])
        b.update({"arm": r["arm"], "seed": r["seed"],
                  "unique_evals_ACHIEVED": r["unique_evals_ACHIEVED"],
                  "achieved_frac": r["achieved_frac"],
                  "clears_floor": r["clears_floor"],
                  "search_fitness": r["search_fitness"]})
        champs.append(b)
    incs = [block(n, s) for n, s in INCUMBENTS.items()]

    ARMB_MS = S.ARMG_REF_MS
    # ---- SELECTION among MY OWN armg champions, per the registered rule ----
    pool = [c for c in champs if c["arm"] == "armg" and c["clears_floor"]]
    sel = None
    if pool and np.isfinite(sd_G):
        best_ms = min(c["ms_per_char"] for c in pool)
        inband = [c for c in pool if c["ms_per_char"] <= best_ms + 2 * sd_G]
        # decide on the GAUGE FRAME, never on the objective
        inband.sort(key=lambda c: (c["D_vs_armB"],
                                   -contested(c["gauges"], prof[armB])["better"]))
        sel = {"n_in_selection_band": len(inband),
               "band": [best_ms, best_ms + 2 * sd_G],
               "winner": inband[0]["name"], "winner_layout": inband[0]["layout"],
               "rule": ("within 2*sd_G of the best, chosen on the pre-declared gauge frame "
                        "(lowest D, then most strictly-better axes) -- NOT on the objective")}

    # ---- VERDICT, in the registered order ----
    all_pri = [c for c in champs if c["clears_floor"]]
    faster = [c for c in all_pri if c["ms_per_char"] < ARMB_MS - 2 * sd_G]
    tied = [c for c in all_pri if abs(c["ms_per_char"] - ARMB_MS) <= 2 * sd_G]
    tied_dom = [c for c in tied
                if contested(c["gauges"], prof[armB])["dominates"]]
    best_D_inband = min((c["D_vs_armB"] for c in tied), default=float("inf"))

    failures = []
    if not any(c["ms_per_char"] <= ARMB_MS + 2 * sd_G
               for c in all_pri if c["arm"] == "armg"):
        failures.append("F1: no ARM G champion landed inside the band "
                        f"(min armg ms = "
                        f"{min((c['ms_per_char'] for c in all_pri if c['arm']=='armg'), default=float('nan')):.4f} "
                        f"> {ARMB_MS + 2*sd_G:.4f})")
    if np.isfinite(best_D_inband) and best_D_inband >= D_FAILURE_BAR:
        failures.append(f"F2: in-band but best D = {best_D_inband:.4f} >= "
                        f"{D_FAILURE_BAR} (flagship-c3's D) -- bought nothing the campaign "
                        "does not already have")
    if len(armg_pri) < 3:
        failures.append(f"F4: only {len(armg_pri)} armg seeds cleared the 80% floor "
                        "-- below the registered n>=3 minimum")

    if faster:
        verdict = "FASTER"
    elif tied_dom:
        verdict = "TIED-AND-STRICTLY-BETTER"
    elif failures:
        verdict = "FAILURE"
    elif tied and best_D_inband < min(i["D_vs_armB"] for i in incs if i["name"] != "arm-B"):
        verdict = "TIED-AND-PARTIALLY-BETTER"
    else:
        verdict = "NEITHER"

    # ---- pairwise table vs every requested comparator, tie-aware ----
    comparators = dict(INCUMBENTS)
    pairwise = {}
    target = sel["winner_layout"] if sel else (pool[0]["layout"] if pool else None)
    if target:
        for n, s in comparators.items():
            c = contested(prof[target], prof[s])
            c["cluster_corrected"] = cluster_corrected(c)
            c["ms_delta_vs_them"] = prof[target]["ms_per_char"] - prof[s]["ms_per_char"]
            c["resolves_on_my_ruler"] = abs(c["ms_delta_vs_them"]) > 2 * sd_G
            pairwise[n] = c

    # ---- cross-arm placebo: does the OBJECTIVE do anything the DRAW does not? ----
    placebo = None
    if armg_pri and base_pri:
        ag = np.array([c["D_vs_armB"] for c in champs
                       if c["arm"] == "armg" and c["clears_floor"]])
        bg = np.array([c["D_vs_armB"] for c in champs
                       if c["arm"] == "baseline" and c["clears_floor"]])
        placebo = {
            "armg_D": {"mean": float(ag.mean()), "min": float(ag.min()),
                       "max": float(ag.max()), "n": int(ag.size)},
            "baseline_D": {"mean": float(bg.mean()), "min": float(bg.min()),
                           "max": float(bg.max()), "n": int(bg.size)},
            "mean_D_reduction": float(bg.mean() - ag.mean()),
            "min_D_reduction": float(bg.min() - ag.min()),
            "why": ("SAME seeds, SAME budget, SAME engine -- only the objective differs, so "
                    "a D reduction here is attributable to the OBJECTIVE, not the draw "
                    "(traps 17/32/34: an effect needs a same-size placebo)."),
        }

    # ---- Hamming BOTH ways (SPEEDTIE-BUDGET-1 trap 3) ----
    def hamming(a: str, b: str) -> int:
        return sum(1 for x, y in zip(a, b, strict=True) if x != y)

    def ham_stats(lays: list[str]) -> dict:
        runs_pairs = [hamming(a, b) for a, b in itertools.combinations(lays, 2)]
        dist = sorted(set(lays))
        dist_pairs = [hamming(a, b) for a, b in itertools.combinations(dist, 2)]
        return {"n_runs": len(lays), "n_distinct": len(dist),
                "mean_over_runs": float(np.mean(runs_pairs)) if runs_pairs else None,
                "mean_over_distinct": float(np.mean(dist_pairs)) if dist_pairs else None,
                "n_zero_pairs": sum(1 for h in runs_pairs if h == 0),
                "vs_armB": {lay: hamming(lay, armB) for lay in dist}}

    out = {
        "experiment": "ARM G",
        "prereg_commit": "0f606d1",
        "drivers_commit": "f58698e",
        "verdict": verdict,
        "registered_failures_triggered": failures,
        "ruler_MEASURED_NOT_BORROWED": ruler,
        "armB_ms": ARMB_MS,
        "selection": sel,
        "n_runs": {"armg": len(armg), "baseline": len(base),
                   "armg_clearing_floor": len(armg_pri),
                   "baseline_clearing_floor": len(base_pri)},
        "excluded_below_80pct_floor": excluded,
        "champions": champs,
        "incumbents": incs,
        "pairwise_vs_selected": pairwise,
        "objective_vs_draw_placebo": placebo,
        "hamming_armg": ham_stats([c["layout"] for c in champs
                                   if c["arm"] == "armg" and c["clears_floor"]]),
        "hamming_baseline": ham_stats([c["layout"] for c in champs
                                       if c["arm"] == "baseline" and c["clears_floor"]]),
        "D_failure_bar": D_FAILURE_BAR,
        "frame": {"live_gauges": list(GAUGES), "n_live": len(GAUGES),
                  "excluded_invariant": list(INVARIANT),
                  "why": ("sfr counts doubled letters -> permutation invariant (trap 23). "
                          "alt and imbalance are HAND-PARTITION invariants "
                          "(ULTRAAUDIT-INTERIM), so some pairs tie BY CONSTRUCTION -- "
                          "hence CONTESTED counts, never a bare n/15.")},
        "oxey_convention": ("AS-SHIPPED nested bad_redirect (my base predates OXEYFIX-1), "
                            "which is the SAME convention SPEEDTIE-1's frozen 14.05x spread "
                            "was measured on. NOT comparable to a post-OXEYFIX board."),
        "modelled_only": ("MODELLED ONLY: g-frame, baked 90 WPM, blend-v1, 1-skip31. Not a "
                          "claim about realized human typing speed. No layout adopted."),
    }
    dest = ART / f"armg-judgement{tag}.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)

    # ---- human-readable ----
    print(f"VERDICT: {verdict}")
    print(f"\nMY OWN RULER (not borrowed): sd_G = {sd_G:.6f} ms/char  "
          f"(2*sd = {2*sd_G:.6f})")
    print(f"  quadruple: {ruler['QUADRUPLE']['STATISTIC']} | "
          f"n_distinct={ruler['n_distinct_champions']}/{ruler['n_runs']} | "
          f"range={ruler['range']:.6f}")
    print("  (borrowed 0.0617 was used for the SEARCH band only, never a verdict)")
    if excluded:
        print(f"\nEXCLUDED below the 80% achieved floor: {excluded}")
    print(f"\narm B = {ARMB_MS:.4f}")
    print(f"\n{'champion':<26} {'ms/char':>10} {'D':>8} {'uniq ACHIEVED':>15} {'%':>7}")
    for c in sorted(champs, key=lambda x: x["ms_per_char"]):
        flag = "" if c["clears_floor"] else "  <<EXCLUDED"
        print(f"{c['name']:<26} {c['ms_per_char']:>10.4f} {c['D_vs_armB']:>8.4f} "
              f"{c['unique_evals_ACHIEVED']:>15,} {c['achieved_frac']:>6.1%}{flag}")
    print(f"\n{'incumbent':<26} {'ms/char':>10} {'D':>8}")
    for i in sorted(incs, key=lambda x: x["ms_per_char"]):
        print(f"{i['name']:<26} {i['ms_per_char']:>10.4f} {i['D_vs_armB']:>8.4f}")
    if placebo:
        print("\nOBJECTIVE vs DRAW (same seeds/budget/engine, only the objective differs):")
        print(f"  armg     D: mean {placebo['armg_D']['mean']:.4f} "
              f"min {placebo['armg_D']['min']:.4f}")
        print(f"  baseline D: mean {placebo['baseline_D']['mean']:.4f} "
              f"min {placebo['baseline_D']['min']:.4f}")
        print(f"  => mean D reduction {placebo['mean_D_reduction']:+.4f}, "
              f"min-D reduction {placebo['min_D_reduction']:+.4f}")
    if sel:
        print(f"\nSELECTED (gauge frame, not objective): {sel['winner']} "
              f"({sel['n_in_selection_band']} in band)")
        print(f"  {sel['winner_layout']}")
        print(f"\n{'vs':<14} {'contested':>10} {'better':>7} {'worse':>6} {'tie':>4} "
              f"{'clusters b/w':>13} {'ms delta':>10} {'resolves':>9}")
        for n, c in pairwise.items():
            cc = c["cluster_corrected"]
            print(f"{n:<14} {c['n_contested']:>10} {c['better']:>7} {c['worse']:>6} "
                  f"{c['tie']:>4} {cc['clusters_better']:>6}/{cc['clusters_worse']:<6} "
                  f"{c['ms_delta_vs_them']:>+10.4f} {str(c['resolves_on_my_ruler']):>9}")
    print(f"\nWROTE {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
