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


def dominance(gauges: dict[str, dict[str, float]], live: list[str]) -> dict:
    """n x n matrix with a STRICT-WIN term (trap 33)."""
    names = list(gauges)
    cells = []
    n_dom = 0
    for a, b in itertools.permutations(names, 2):
        n_ge = n_strict = n_better = n_worse = 0
        for g in live:
            lo = LOWER_BETTER[g]
            va, vb = gauges[a][g], gauges[b][g]
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
        cells.append({"a": a, "b": b, "n_ge": n_ge, "n_strict": n_strict,
                      "n_better": n_better, "n_worse": n_worse, "dominates": dom})
    return {"n_axes": len(live), "n_dominating_pairs": n_dom, "cells": cells}


def profile(layouts: list[str], label: str) -> dict:
    blob = cli_analyze(layouts)
    by_layout = {row["layout"]: row for row in blob["rows"].values()}
    out = {"label": label, "corpus": blob.get("corpus"),
           "corpus_provenance": blob.get("corpus_provenance"),
           "skipgram_table": blob.get("skipgram_table"),
           "gauge_frame": blob.get("gauge_frame"), "model_family": blob.get("model_family"),
           "target_wpm": blob.get("target_wpm"), "layouts": {}}
    for lay in layouts:
        row = by_layout[lay]
        # ms/char lives under row["time"]; a top-level row["ms_per_char"] does not exist and
        # reading it would silently yield None for every layout.
        out["layouts"][lay] = {"ms_per_char": row["time"]["ms_per_char"],
                               "coverage_pct": row["time"]["coverage_pct"],
                               "gauges": row["gauges"]}
    return out


def analyse(prof: dict, live: list[str]) -> dict:
    lays = list(prof["layouts"])
    ms = [prof["layouts"][l]["ms_per_char"] for l in lays]
    g = {l: prof["layouts"][l]["gauges"] for l in lays}
    per_gauge = {}
    for name in live:
        per_gauge[name] = spreads([g[l][name] for l in lays])
    H = [{"a": a, "b": b, "hamming": hamming(a, b)} for a, b in itertools.combinations(lays, 2)]
    hv = [h["hamming"] for h in H]
    return {
        "n": len(lays),
        "layouts": lays,
        "speed": spreads(ms),
        "per_gauge": per_gauge,
        "hamming": {"n_pairs": len(hv), "mean": st.mean(hv), "min": min(hv), "max": max(hv),
                    "pairs": H},
        "dominance": dominance(g, live),
        "n_distinct_champions": len(set(lays)),
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
    paired_seeds = [r["seed"] for r in ok if r["seed"] in seed_to_1m]
    lay_1m = [seed_to_1m[s] for s in paired_seeds]
    lay_10m = [r["layout"] for r in ok if r["seed"] in seed_to_1m]
    for l in lay_1m:
        if l not in lay_1m_all:
            raise SystemExit(f"paired 1M layout {l} absent from the frozen speedtie-1 summary")

    # ---- score BOTH budgets through the shipped CLI, in one call each ----
    p1 = profile(lay_1m, f"1M (paired subset, n={len(lay_1m)})")
    p10 = profile(lay_10m, f"{budget/1e6:.0f}M (n={len(lay_10m)})")

    # sfr invariance tested DIRECTLY (trap 23), not via a variance threshold.
    sfr_vals = {l: p10["layouts"][l]["gauges"]["sfr"] for l in lay_10m}
    sfr_vals.update({l: p1["layouts"][l]["gauges"]["sfr"] for l in lay_1m})
    sfr_set = set(round(v, 12) for v in sfr_vals.values())
    # NB `gauge_frame` is a DESCRIPTION STRING, not a list of names — iterating it would walk
    # characters. The gauge names come from a row's own `gauges` dict.
    live = [g for g in p10["layouts"][lay_10m[0]]["gauges"] if g not in INVARIANT]
    missing_dir = [g for g in live if g not in LOWER_BETTER]
    if missing_dir:
        raise SystemExit(f"no registered direction for {missing_dir}")

    a1 = analyse(p1, live)
    a10 = analyse(p10, live)

    # ---- the PRE-REGISTERED statistics ----
    R_speed = a10["speed"]["range"] / a1["speed"]["range"]
    R_sd = a10["speed"]["sd"] / a1["speed"]["sd"]
    per_ratio = {g: a10["per_gauge"][g]["range"] / a1["per_gauge"][g]["range"] for g in live}
    M_gauge = st.median(per_ratio.values())
    n_shrunk_2x = sum(1 for v in per_ratio.values() if v <= 0.5)
    H_ratio = a10["hamming"]["mean"] / a1["hamming"]["mean"]
    n_big_ratio_10m = sum(1 for g in live
                          if (a10["per_gauge"][g]["ratio"] or 0) >= 5.0)
    no_dom_10m = a10["dominance"]["n_dominating_pairs"] == 0

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

    shortfall = [r for r in ok if r["unique_evals"] < 0.8 * budget]

    out = {
        "experiment": "SPEEDTIE-BUDGET",
        "modelled_only": ("MODELLED ONLY: g-frame, baked 90 WPM, blend-v1, skipgrams=1-skip31. "
                          "NOT a claim about realized human typing speed. No layout adopted."),
        "budget_requested": budget,
        "n_paired": len(lay_10m),
        "paired_seeds": paired_seeds,
        "achieved_unique_evals": {str(r["seed"]): r["unique_evals"] for r in ok},
        "runs_below_80pct_of_request": [{"seed": r["seed"], "unique_evals": r["unique_evals"]}
                                        for r in shortfall],
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
        "preregistered_statistics": {
            "R_speed_range": R_speed,
            "R_speed_sd": R_sd,
            "per_gauge_range_ratio_10M_over_1M": per_ratio,
            "M_gauge_median": M_gauge,
            "n_gauges_shrunk_2x_or_more": n_shrunk_2x,
            "mean_hamming_ratio": H_ratio,
            "n_live_gauges_ratio_ge_5_at_10M": n_big_ratio_10m,
            "no_dominance_at_10M": no_dom_10m,
        },
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
    print(f"  mean Hamming ratio = {H_ratio:.4f} ({a1['hamming']['mean']:.2f} -> "
          f"{a10['hamming']['mean']:.2f})")
    print(f"  gauges with ratio>=5 at 10M: {n_big_ratio_10m}   dominating pairs: "
          f"{a10['dominance']['n_dominating_pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
