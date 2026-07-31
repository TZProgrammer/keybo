"""Emit every KITCHEN-SINK table from the results JSON — GENERATED, never hand-transcribed.

A prior agent in this campaign hand-typed a table with 56 wrong cells out of 208. Every number in
the KITCHEN-SINK report comes out of this script, and the report says so.

    PYTHONPATH=src python agent-artifacts/kitchensink_tables.py \
        --results agent-artifacts/kitchensink_results.json --out agent-artifacts/KITCHENSINK_TABLES.md
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _fmt(x, nd=6) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(f) else f"{f:+.{nd}f}"


def table_paired(res: dict, ngram: str, baseline: str) -> str:
    """Per-(fold, seed) paired rho deltas — the MOR-FIX-1 statistic."""
    rows = []
    for r in res[ngram][f"paired_vs_{baseline}"]:
        rows.append(
            [
                r["holdout"],
                str(r["seed"]),
                f"{r['ceiling']:.4f}",
                f"{r['rho_baseline']:.6f}",
                f"{r['rho_kitchensink']:.6f}",
                f"**{_fmt(r['delta'])}**",
                f"{r['wmae_baseline']:.4f}",
                f"{r['wmae_kitchensink']:.4f}",
                _fmt(r["hw_gate_passed"]),
                ",".join(str(b) for b in r["hw_regressing_buckets"]) or "—",
            ]
        )
    return _md_table(
        [
            "fold",
            "seed",
            "ceiling",
            f"rho {baseline}",
            "rho kitchensink",
            "delta",
            f"wmae {baseline}",
            "wmae ks",
            "hw gate",
            "regressing ≥80",
        ],
        rows,
    )


def table_sign(res: dict, ngram: str, baseline: str) -> str:
    sc = res[ngram][f"sign_consistency_vs_{baseline}"]
    rows = [
        [
            fold,
            " ".join(f"{d:+.6f}" for d in v["deltas"]),
            _fmt(v["mean"]),
            f"{v['n_pos']}/{v['n_pos'] + v['n_neg']}",
            f"**{v['verdict']}**",
        ]
        for fold, v in sc.items()
    ]
    return _md_table(["fold", "per-seed deltas", "mean", "pos/nonzero", "verdict"], rows)


def table_highwpm(res: dict, ngram: str) -> str:
    hw = res[ngram]["high_wpm_vs_widened"]
    rows = []
    for key, v in sorted(hw["structural"].items()):
        rows.append([v["fold"], str(v["bucket"]), f"{v['seeds_hit']}/{v['of']}", "**STRUCTURAL**"])
    for key, v in sorted(hw["noise"].items()):
        rows.append([v["fold"], str(v["bucket"]), f"{v['seeds_hit']}/{v['of']}", "noise"])
    if not rows:
        rows = [["—", "—", "—", "no regression at any bucket ≥ 80"]]
    return _md_table(["fold", "bucket", "seeds hit", "class"], rows)


def table_hw_deltas(res: dict, ngram: str) -> str:
    """Every gated bucket's rho delta, per cell — the full evidence behind the gate verdict."""
    cells = res[ngram]["paired_vs_widened"]
    buckets = sorted({int(b) for r in cells for b in r["hw_deltas"]})
    rows = []
    for r in cells:
        rows.append(
            [r["holdout"], str(r["seed"])]
            + [_fmt(r["hw_deltas"].get(str(b), r["hw_deltas"].get(b)), 4) for b in buckets]
        )
    return _md_table(["fold", "seed"] + [f"b{b}" for b in buckets], rows)


def table_importance(res: dict, ngram: str) -> str:
    imp = res[ngram]["importance"]
    rows = []
    for name in imp["new_columns"]:
        frac = imp["mean_col_frac"].get(name, 0.0)
        used = sum(1 for s in imp["per_seed"] if s["by_name"][name] > 0)
        rows.append([f"`{name}`", f"{100 * frac:.4f}%", f"{used}/{len(imp['per_seed'])}"])
    rows.append(
        [
            "**all new columns**",
            f"**{100 * imp['mean_new_columns_frac']:.4f}%**",
            f"{len(imp['used_on_all_seeds'])}/{imp['n_new_columns']} used on all seeds",
        ]
    )
    return _md_table(["new column", "mean share of total gain", "seeds used"], rows)


def table_ranking(res: dict) -> str:
    if "ranking" not in res:
        return "_UNSCOREABLE — the ranking measurement did not run._"
    rows = []
    for ngram, r in res["ranking"].items():
        rows.append(
            [
                ngram,
                str(r["n_layouts"]),
                f"{r['kendall_tau']:+.4f}",
                f"{r['spearman_rho']:+.4f}",
                f"{r['n_positions_moved']}/{r['n_layouts']}",
                f"{r['argmin_widened']} → {r['argmin_kitchensink']}",
                "**yes**" if r["argmin_stable"] else "**NO**",
                f"{r['argmin_margin_pct']:.3f}%",
                f"{r['median_adjacent_gap_pct']:.3f}%",
            ]
        )
    return _md_table(
        [
            "ngram",
            "n",
            "Kendall tau",
            "Spearman rho",
            "moved",
            "argmin (widened → ks)",
            "argmin stable",
            "argmin margin",
            "median adj gap",
        ],
        rows,
    )


def table_three_way(res: dict, ngram: str) -> str:
    """narrow / widened / kitchensink rho side by side, per (fold, seed)."""
    tr = res[ngram]["transfer"]
    rows = []
    for fold in tr["narrow"]["folds"]:
        by = {
            arm: {r["seed"]: r["rho"] for r in tr[arm]["folds"][fold]["seeds"]}
            for arm in ("narrow", "widened", "kitchensink")
        }
        for seed in sorted(by["narrow"]):
            rows.append(
                [
                    fold,
                    str(seed),
                    f"{by['narrow'][seed]:.6f}",
                    f"{by['widened'][seed]:.6f}",
                    f"{by['kitchensink'][seed]:.6f}",
                    _fmt(by["kitchensink"][seed] - by["narrow"][seed]),
                ]
            )
    return _md_table(
        ["fold", "seed", "rho narrow", "rho widened", "rho kitchensink", "ks − narrow"], rows
    )


def summary_line(res: dict, ngram: str, baseline: str) -> str:
    ds = [r["delta"] for r in res[ngram][f"paired_vs_{baseline}"]]
    w, l = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
    sc = res[ngram][f"sign_consistency_vs_{baseline}"]
    consistent = sum(1 for v in sc.values() if v["verdict"] in ("WIN", "LOSS"))
    wins = sum(1 for v in sc.values() if v["verdict"] == "WIN")
    return (
        f"- **{ngram} vs {baseline}:** {w} win / {l} loss of {len(ds)} cells, "
        f"mean paired delta **{np.mean(ds):+.6f}**, "
        f"{consistent}/{len(sc)} folds sign-consistent ({wins} WIN, "
        f"{consistent - wins} LOSS, {len(sc) - consistent} MIXED)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = json.load(open(args.results, encoding="utf-8"))

    parts = [
        "# KITCHEN-SINK — generated tables",
        "",
        f"Machine-generated by `agent-artifacts/kitchensink_tables.py` from `{args.results}`. "
        "No number in the report is hand-transcribed.",
        "",
        "**Scope, travelling with every n below:** "
        f"seeds {res['seeds']}, LOLO folds {res['scope']['lolo_folds']} "
        f"({len(res['scope']['lolo_folds'])} folds — NOT the 15-name scoring catalog), "
        f"{res['scope']['n_bistroke_rows']} bistroke / {res['scope']['n_tristroke_rows']} tristroke "
        f"rows, high-wpm floor {res['high_wpm_floor']}.",
        "",
    ]

    ngrams = [n for n in ("bigram", "trigram") if n in res]
    if ngrams:
        parts += ["## Headline", ""]
        for ngram in ngrams:
            for baseline in ("widened", "narrow"):
                parts.append(summary_line(res, ngram, baseline))
        parts.append("")

    for ngram in ngrams:
        parts += [
            f"## {ngram}",
            "",
            "### Paired per-fold/seed deltas vs the WIDENED incumbent",
            "",
            table_paired(res, ngram, "widened"),
            "",
            "### Sign consistency per fold (vs widened)",
            "",
            table_sign(res, ngram, "widened"),
            "",
            "### Sign consistency per fold (vs narrow — the two-round-back baseline)",
            "",
            table_sign(res, ngram, "narrow"),
            "",
            "### Three-way rho: narrow / widened / kitchen-sink",
            "",
            table_three_way(res, ngram),
            "",
            "### High-wpm non-regression (floor 80): STRUCTURAL vs NOISE",
            "",
            table_highwpm(res, ngram),
            "",
            "### Per-bucket rho deltas, every cell (kitchen-sink − widened)",
            "",
            table_hw_deltas(res, ngram),
            "",
            "### Total-gain importance of the new columns (full-data models)",
            "",
            table_importance(res, ngram),
            "",
        ]

    parts += ["## Ranking over the named-layout catalog (the SEPARATE surface)", "", table_ranking(res), ""]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    print("\n".join(parts))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
