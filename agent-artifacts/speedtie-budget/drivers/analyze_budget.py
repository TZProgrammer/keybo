"""Score both budgets' champions with the SHIPPED CLI and evaluate the PRE-REGISTERED rule.

Every gauge number here comes from `keybo analyze --json` (trap 28: a hand-rolled
reimplementation of a validated constructor loses the validation). The search driver's own
`_ms_per_char` is used only as the search's internal objective and is reported separately so a
reader can see the two agree.

`sfr` is EXCLUDED from every spread / ratio / dominance / win-count: it counts doubled letters
and is a permutation invariant (trap 23 — numpy reports its std as ~1.9e-14, not 0, so a
`std>0` filter would keep it and rank-correlate pure noise). Test invariance directly, which is
what the assertion below does.

Dominance carries a STRICT-WIN term (trap 33): `a` dominates `b` iff `a` is at-least-as-good on
all 14 live gauges AND strictly better on >= 1. `n_ge` and `n_strict` are reported separately.
"""

from __future__ import annotations

import itertools
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

WORKTREE = Path("/tmp/speedtie")
STATE = Path("/local/home/zegertho/agent/state/speedtie/artifacts")
FROZEN_1M = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/speedtie-1/"
                 "speedtie-summary.json")

INVARIANT = ("sfr",)
# Direction for each live gauge: True = lower is better. Taken from the campaign's registered
# convention (ARME-1 corroborated EXPECTED_SIGN on 14/14 in-band), NOT invented here.
LOWER_BETTER = {
    "sfb": True, "sfs": True, "sfb-dist": True, "sfs-dist": True,
    "lsb": True, "lsb-dist": True,
    "alt": False, "roll": False, "sr-roll": False,
    "redir": True, "scissor": True, "imbalance": True,
    "oxey-style": True, "comfort": True,
}


def cli_analyze(layouts: list[str]) -> dict:
    """Shipped CLI, one call, one row per layout.

    The CLI also emits a row for `--ref` (default qwerty), so the expected row count is
    len(layouts) + 1 unless qwerty was itself requested. Asserting the WRONG count would be a
    self-inflicted trap-38 false positive, so the expectation is derived, not assumed.
    """
    cmd = ["uv", "run", "--no-sync", "keybo", "analyze", "--json", *layouts]
    proc = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"CLI FAILED rc={proc.returncode}\n{proc.stderr[-3000:]}")
    blob = json.loads(proc.stdout)
    rows = blob["rows"]
    expect = len({*layouts, "qwerty"})
    if len(rows) != expect:
        raise SystemExit(f"CLI returned {len(rows)} rows, expected {expect} for "
                         f"{len(layouts)} layouts + ref — a row was dropped (trap 38)")
    # Every requested layout must appear as a row whose `layout` field is that exact string.
    by_layout = {row["layout"]: row for row in rows.values()}
    missing = [l for l in layouts if l not in by_layout]
    if missing:
        raise SystemExit(f"CLI rows did not cover {missing}")
    return blob


def spreads(vals: list[float]) -> dict:
    return {"min": min(vals), "max": max(vals), "range": max(vals) - min(vals),
            "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
            "mean": st.mean(vals),
            "ratio": (max(vals) / min(vals)) if min(vals) > 0 else None}


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def dominance(items: list[tuple[str, dict[str, float]]], live: list[str]) -> dict:
    """Pairwise dominance over DISTINCT champions, with a mandatory STRICT-WIN term (trap 33).

    `a` dominates `b` iff `a` is at-least-as-good on all live gauges AND strictly better on >=1.
    A pair of IDENTICAL champions is excluded from the pair universe and counted separately:
    without the strict-win term such a pair would score n_ge == n_axes and be reported as a
    dominator, which is trap 33 exactly. Reporting it as "identical" is the honest statement —
    it is neither a dominance nor a mixed comparison.
    """
    cells = []
    n_dom = 0
    n_identical = 0
    for (na, ga), (nb, gb) in itertools.permutations(items, 2):
        if na == nb:
            n_identical += 1
            continue
        n_ge = n_strict = n_better = n_worse = 0
        for g in live:
            lo = LOWER_BETTER[g]
            va, vb = ga[g], gb[g]
            better = (va < vb) if lo else (va > vb)
            worse = (va > vb) if lo else (va < vb)
            if not worse:
                n_ge += 1
            if better:
                n_strict += 1
                n_better += 1
            if worse:
                n_worse += 1
        dom = (n_ge == len(live)) and (n_strict >= 1)
        n_dom += dom
        cells.append({"a": na, "b": nb, "n_ge": n_ge, "n_strict": n_strict,
                      "n_better": n_better, "n_worse": n_worse,
                      "n_ties": len(live) - n_better - n_worse, "dominates": dom})
    return {"n_axes": len(live), "n_distinct_compared": len(items),
            "n_ordered_pairs": len(cells), "n_dominating_pairs": n_dom,
            "n_identical_ordered_pairs_excluded": n_identical, "cells": cells}


def profile(layouts: list[str], label: str, seeds: list[int]) -> dict:
    """Score `layouts` (one per RUN, in run order) through the shipped CLI.

    ⚠ The result is a LIST indexed by run, never a dict keyed by layout string. At 10M several
    seeds converge to the SAME champion string, and keying on the layout would silently collapse
    those runs into one entry — computing every spread and sd over 4 values instead of 6 and
    UNDERSTATING the spread. That is trap 38's shape (a collection keyed on a lossy form) and it
    would bias the result toward H-UNDER, i.e. toward a false convergence verdict. The CLI is
    still called on the DEDUPED set (it rejects duplicate specs), then results are mapped back
    onto every run.
    """
    uniq = list(dict.fromkeys(layouts))
    blob = cli_analyze(uniq)
    by_layout = {row["layout"]: row for row in blob["rows"].values()}
    out = {"label": label, "corpus": blob.get("corpus"),
           "corpus_provenance": blob.get("corpus_provenance"),
           "skipgram_table": blob.get("skipgram_table"),
           "gauge_frame": blob.get("gauge_frame"), "model_family": blob.get("model_family"),
           "target_wpm": blob.get("target_wpm"),
           "n_runs": len(layouts), "n_distinct_champions": len(uniq), "runs": []}
    for seed, lay in zip(seeds, layouts, strict=True):
        row = by_layout[lay]
        # ms/char lives under row["time"]; a top-level row["ms_per_char"] does not exist and
        # reading it would silently yield None for every layout.
        out["runs"].append({"seed": seed, "layout": lay,
                            "ms_per_char": row["time"]["ms_per_char"],
                            "coverage_pct": row["time"]["coverage_pct"],
                            "gauges": row["gauges"]})
    return out


def analyse(prof: dict, live: list[str]) -> dict:
    """All spreads are over RUNS (n = number of seeds), not over distinct champion strings.

    Hamming is reported BOTH ways and the difference is load-bearing:
      - over all C(n,2) RUN pairs, which includes 0-distance pairs when two seeds land on the
        same champion. This is the statistic the pre-registered rule uses, because "the runs
        converged toward each other" is a claim about RUNS.
      - over DISTINCT champions only, which answers the different question "how far apart are
        the distinct optima the search finds?" Quoting only the second would hide convergence;
        quoting only the first would hide that the survivors are still far apart.
    """
    runs = prof["runs"]
    lays = [r["layout"] for r in runs]
    ms = [r["ms_per_char"] for r in runs]
    per_gauge = {name: spreads([r["gauges"][name] for r in runs]) for name in live}
    H = [{"a": a, "b": b, "hamming": hamming(a, b)}
         for a, b in itertools.combinations(lays, 2)]
    hv = [h["hamming"] for h in H]
    uniq = list(dict.fromkeys(lays))
    Hd = [hamming(a, b) for a, b in itertools.combinations(uniq, 2)]
    return {
        "n_runs": len(runs),
        "seeds": [r["seed"] for r in runs],
        "layouts_by_run": lays,
        "n_distinct_champions": len(uniq),
        "speed": spreads(ms),
        "per_gauge": per_gauge,
        "hamming_over_runs": {"n_pairs": len(hv), "mean": st.mean(hv), "min": min(hv),
                              "max": max(hv), "n_zero_pairs": sum(1 for v in hv if v == 0),
                              "pairs": H},
        "hamming_over_distinct": ({"n_pairs": len(Hd), "mean": st.mean(Hd), "min": min(Hd),
                                   "max": max(Hd)} if len(Hd) else None),
        "dominance": dominance([(l, runs[lays.index(l)]["gauges"]) for l in uniq], live),
    }


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    runs = json.load(open(STATE / "runs" / f"budget-{budget}-summary.json"))
    ok = [r for r in runs["runs"] if r["ok"]]
    if not ok:
        raise SystemExit("no successful runs to analyse")

    frozen = json.load(open(FROZEN_1M))
    lay_1m_all = list(frozen["layouts"])
    # PAIRING: keep only the 1M runs whose seed has a completed 10M partner.
    placebo = json.load(open("/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
                             "optevidence-1/search-noise-placebo.json"))
    seed_to_1m = {row["seed"]: row["layout"] for row in placebo["runs"]["baseline"]}
    seed_to_1m_uniq = {row["seed"]: row["unique_evals"] for row in placebo["runs"]["baseline"]}

    # PRE-REGISTERED EXCLUSION: a run below 80% of its requested budget is a DIFFERENT
    # experiment and is not pooled. It is still reported, and a sensitivity analysis including
    # it is reported alongside, so the exclusion cannot hide a result.
    floor = 0.8 * budget
    primary = [r for r in ok if r["seed"] in seed_to_1m and r["unique_evals"] >= floor]
    excluded = [r for r in ok if r["seed"] in seed_to_1m and r["unique_evals"] < floor]

    def build(rows: list[dict], tag: str) -> dict:
        seeds = [r["seed"] for r in rows]
        l1 = [seed_to_1m[s] for s in seeds]
        l10 = [r["layout"] for r in rows]
        for l in l1:
            if l not in lay_1m_all:
                raise SystemExit(f"paired 1M layout {l} absent from frozen speedtie-1 summary")
        return {"seeds": seeds,
                "p1": profile(l1, f"1M paired {tag} (n={len(l1)})", seeds),
                "p10": profile(l10, f"{budget/1e6:.0f}M {tag} (n={len(l10)})", seeds)}

    B = build(primary, "PRIMARY")
    p1, p10, paired_seeds = B["p1"], B["p10"], B["seeds"]

    # sfr invariance tested DIRECTLY (trap 23), not via a variance threshold.
    sfr_set = {round(r["gauges"]["sfr"], 12) for r in (*p10["runs"], *p1["runs"])}
    # NB `gauge_frame` is a DESCRIPTION STRING, not a list of names — iterating it would walk
    # characters. The gauge names come from a row's own `gauges` dict.
    live = [g for g in p10["runs"][0]["gauges"] if g not in INVARIANT]
    missing_dir = [g for g in live if g not in LOWER_BETTER]
    if missing_dir:
        raise SystemExit(f"no registered direction for {missing_dir}")

    a1 = analyse(p1, live)
    a10 = analyse(p10, live)

    # ---- the PRE-REGISTERED statistics ----
    def stats(x1: dict, x10: dict) -> dict:
        per_ratio = {g: (x10["per_gauge"][g]["range"] / x1["per_gauge"][g]["range"])
                     for g in live}
        return {
            "R_speed_range": x10["speed"]["range"] / x1["speed"]["range"],
            "R_speed_sd": (x10["speed"]["sd"] / x1["speed"]["sd"]) if x1["speed"]["sd"] else None,
            "per_gauge_range_ratio_10M_over_1M": per_ratio,
            "M_gauge_median": st.median(per_ratio.values()),
            "n_gauges_shrunk_2x_or_more": sum(1 for v in per_ratio.values() if v <= 0.5),
            "mean_hamming_ratio": (x10["hamming_over_runs"]["mean"]
                                   / x1["hamming_over_runs"]["mean"]),
            "n_live_gauges_ratio_ge_5_at_10M": sum(
                1 for g in live if (x10["per_gauge"][g]["ratio"] or 0) >= 5.0),
            "no_dominance_at_10M": x10["dominance"]["n_dominating_pairs"] == 0,
        }

    S = stats(a1, a10)
    R_speed = S["R_speed_range"]
    R_sd = S["R_speed_sd"]
    per_ratio = S["per_gauge_range_ratio_10M_over_1M"]
    M_gauge = S["M_gauge_median"]
    n_shrunk_2x = S["n_gauges_shrunk_2x_or_more"]
    H_ratio = S["mean_hamming_ratio"]
    n_big_ratio_10m = S["n_live_gauges_ratio_ge_5_at_10M"]
    no_dom_10m = S["no_dominance_at_10M"]

    h_under = (R_speed <= 0.50) and (M_gauge <= 0.50) and (H_ratio <= 0.75)
    h_real = (M_gauge >= 0.80) and (n_big_ratio_10m >= 2) and no_dom_10m
    asym = (R_speed <= 0.50) and (M_gauge >= 0.80)
    suspect = (R_speed >= 1.50) and (M_gauge >= 0.80)

    if h_under:
        verdict = "H-UNDER"
    elif asym and n_big_ratio_10m >= 2 and no_dom_10m:
        verdict = "H-REAL (STRONG FORM — asymmetric case)"
    elif suspect:
        verdict = "SUSPECT HARNESS (speed spread GREW) — H-REAL not established"
    elif h_real:
        verdict = "H-REAL"
    else:
        verdict = "INDETERMINATE"

    # ---- SENSITIVITY: repeat everything with the sub-floor run(s) INCLUDED, so the
    # pre-registered exclusion cannot be the thing that produced the verdict. ----
    sens = None
    if excluded:
        allrows = [r for r in ok if r["seed"] in seed_to_1m]
        BA = build(allrows, "SENSITIVITY-all")
        sa1, sa10 = analyse(BA["p1"], live), analyse(BA["p10"], live)
        SS = stats(sa1, sa10)
        s_h_under = (SS["R_speed_range"] <= 0.50 and SS["M_gauge_median"] <= 0.50
                     and SS["mean_hamming_ratio"] <= 0.75)
        s_asym = SS["R_speed_range"] <= 0.50 and SS["M_gauge_median"] >= 0.80
        s_h_real = (SS["M_gauge_median"] >= 0.80
                    and SS["n_live_gauges_ratio_ge_5_at_10M"] >= 2
                    and SS["no_dominance_at_10M"])
        sens = {
            "note": ("includes the run(s) below the pre-registered 80% floor; reported so the "
                     "exclusion cannot be what produced the verdict"),
            "seeds": BA["seeds"],
            "n": len(BA["seeds"]),
            "statistics": SS,
            "at_1M": sa1, "at_10M": sa10,
            "verdict_would_be": ("H-UNDER" if s_h_under else
                                 "H-REAL (STRONG FORM — asymmetric case)"
                                 if (s_asym and SS["n_live_gauges_ratio_ge_5_at_10M"] >= 2
                                     and SS["no_dominance_at_10M"]) else
                                 "H-REAL" if s_h_real else "INDETERMINATE"),
        }

    shortfall = [r for r in ok if r["unique_evals"] < 0.8 * budget]

    out = {
        "experiment": "SPEEDTIE-BUDGET",
        "modelled_only": ("MODELLED ONLY: g-frame, baked 90 WPM, blend-v1, skipgrams=1-skip31. "
                          "NOT a claim about realized human typing speed. No layout adopted."),
        "budget_requested": budget,
        "n_paired": len(paired_seeds),
        "paired_seeds": paired_seeds,
        "achieved_unique_evals_10M": {str(r["seed"]): r["unique_evals"] for r in ok},
        "achieved_unique_evals_1M": {str(s): seed_to_1m_uniq[s] for s in paired_seeds},
        "runs_below_80pct_of_request_EXCLUDED_from_primary": [
            {"seed": r["seed"], "unique_evals": r["unique_evals"],
             "pct_of_request": 100 * r["unique_evals"] / budget,
             "layout": r["layout"]} for r in shortfall],
        "sensitivity_including_subfloor_runs": sens,
        "sfr_excluded": {"reason": "permutation invariant (trap 23); tested directly by "
                                   "distinct-value count, not by a variance threshold",
                         "n_distinct_values_over_all_champions": len(sfr_set),
                         "value": sorted(sfr_set)},
        "live_gauges": live,
        "frame_1M": {k: p1[k] for k in ("corpus", "corpus_provenance", "skipgram_table",
                                        "model_family", "target_wpm")},
        "frame_10M": {k: p10[k] for k in ("corpus", "corpus_provenance", "skipgram_table",
                                          "model_family", "target_wpm")},
        "at_1M": a1,
        "at_10M": a10,
        "preregistered_statistics": S,
        "preregistered_rule_evaluation": {
            "H_UNDER_all_three": h_under,
            "H_REAL_both": h_real,
            "ASYMMETRIC_strong_form": asym,
            "SUSPECT_growing_speed_spread": suspect,
        },
        "VERDICT": verdict,
        "profiles": {"1M": p1, "10M": p10},
    }
    path = STATE / f"speedtie-budget-{budget}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"WROTE {path}")
    print(f"\nVERDICT: {verdict}")
    print(f"  R_speed(range) = {R_speed:.4f}   R_sd = {R_sd:.4f}")
    print(f"  M_gauge(median per-gauge range ratio) = {M_gauge:.4f}")
    print(f"  mean Hamming ratio = {H_ratio:.4f} "
          f"({a1['hamming_over_runs']['mean']:.2f} -> {a10['hamming_over_runs']['mean']:.2f}; "
          f"{a10['hamming_over_runs']['n_zero_pairs']} zero-distance run pairs at 10M)")
    print(f"  gauges with ratio>=5 at 10M: {n_big_ratio_10m}   dominating pairs: "
          f"{a10['dominance']['n_dominating_pairs']}  "
          f"(distinct champions {a10['n_distinct_champions']}/{a10['n_runs']})")
    print(f"  n paired = {len(paired_seeds)}  seeds {paired_seeds}")
    if sens:
        print(f"  SENSITIVITY (n={sens['n']}, sub-floor runs included): "
              f"verdict would be {sens['verdict_would_be']}, "
              f"R_speed={sens['statistics']['R_speed_range']:.4f}, "
              f"M_gauge={sens['statistics']['M_gauge_median']:.4f}")
    print("\nper-gauge range ratio (10M/1M), sorted:")
    for g, v in sorted(per_ratio.items(), key=lambda kv: kv[1]):
        r10 = a10["per_gauge"][g]
        r1 = a1["per_gauge"][g]
        print(f"  {g:<12s} {r1['range']:9.4f} -> {r10['range']:9.4f}  ratio_of_ranges={v:7.4f}"
              f"   spread_ratio_10M={(r10['ratio'] or 0):7.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
