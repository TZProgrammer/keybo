"""Render the RETRAIN-DIRECTION results JSON into GENERATED markdown tables.

Every table in the report comes from here — no hand-transcription (a prior campaign report had
56 wrong cells of 208 from hand-typing). Usage:

    python agent-artifacts/retrain_direction_tables.py agent-artifacts/retrain_direction_results.json
"""

from __future__ import annotations

import json
import sys


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if x == x else "nan"  # x==x guards NaN
    return str(x)


def _u(x, nd=4):
    """Unsigned float format (for magnitudes like wmae, ceiling, fitness)."""
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if x == x else "nan"
    return str(x)


def transfer_table(res: dict, ngram: str) -> str:
    rows = res[ngram]["paired_fold_deltas"]
    lines = [
        f"#### {ngram}: paired per-fold transfer deltas (rho_widened - rho_narrow)",
        "",
        "| holdout | seed | ceiling | rho_narrow | rho_widened | delta | hw-gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["hw_gate_gated"] is False:
            gate = "UNSCOREABLE"
        elif r["hw_gate_passed"]:
            gate = "pass"
        else:
            gate = f"FAIL {r['hw_regressing_buckets']}"
        lines.append(
            f"| {r['holdout']} | {r['seed']} | {_u(r['ceiling'])} | {_f(r['rho_narrow'])} "
            f"| {_f(r['rho_widened'])} | {_f(r['delta'])} | {gate} |"
        )
    # summary
    deltas = [r["delta"] for r in rows]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses
    lines += [
        "",
        f"- cells: {len(deltas)}  wins(delta>0): {wins}  losses(delta<0): {losses}  "
        f"ties(|delta|<=1e-9): {ties}",
        f"- mean paired delta: {_f(sum(deltas) / len(deltas)) if deltas else 'n/a'}  "
        f"min: {_f(min(deltas)) if deltas else 'n/a'}  max: {_f(max(deltas)) if deltas else 'n/a'}",
    ]
    # sign-consistency per fold across seeds
    lines.append("- per-fold sign consistency across seeds:")
    by_fold: dict[str, list[float]] = {}
    for r in rows:
        by_fold.setdefault(r["holdout"], []).append(r["delta"])
    for holdout, ds in by_fold.items():
        signs = {("+" if d > 1e-9 else "-" if d < -1e-9 else "0") for d in ds}
        consistent = "consistent" if len(signs) == 1 else "MIXED"
        lines.append(f"    - {holdout}: {[round(d, 4) for d in ds]} -> {consistent}")
    return "\n".join(lines)


def high_wpm_table(res: dict, ngram: str) -> str:
    """Per-fold/seed per-bucket delta table (widened - narrow) at buckets >= floor."""
    widened = res[ngram]["transfer"]["widened"]
    lines = [
        f"#### {ngram}: high-wpm non-regression (widened vs narrow), floor >= {res['high_wpm_floor']} wpm",
        "",
        "| holdout | seed | gated | passed | regressing buckets | per-bucket deltas (all) |",
        "|---|---|---|---|---|---|",
    ]
    any_fail = False
    for holdout, fold in widened["folds"].items():
        for rec in fold["seeds"]:
            g = rec["high_wpm_gate_vs_narrow"]
            deltas = {int(k): round(v, 4) for k, v in g["deltas"].items()}
            passed = "UNSCOREABLE" if not g["gated"] else ("pass" if g["passed"] else "FAIL")
            if g["gated"] and not g["passed"]:
                any_fail = True
            lines.append(
                f"| {holdout} | {rec['seed']} | {g['gated']} | {passed} "
                f"| {g['regressing_high_buckets']} | {deltas} |"
            )
    lines += ["", f"- **verdict: {'FAIL — widened regresses a high-wpm bucket' if any_fail else 'PASS — no high-wpm regression in any cell'}**"]
    return "\n".join(lines)


def importance_table(res: dict, ngram: str) -> str:
    imp = res[ngram]["importance"]
    lines = [
        f"#### {ngram}: feature-importance (total gain) of the NEW direction columns, widened full-data model",
        "",
        "| new column | mean frac of total gain (over seeds) |",
        "|---|---|",
    ]
    for col in imp["new_columns"]:
        lines.append(f"| {col} | {imp['mean_col_frac'].get(col, 0.0):.6f} |")
    lines += [
        "",
        f"- **new columns' combined share of total gain: {imp['mean_new_columns_frac']:.6f} "
        f"({imp['mean_new_columns_frac'] * 100:.3f}%)**",
        "- per-seed used-columns (any gain > 0):",
    ]
    for s in imp["per_seed"]:
        used_new = [c for c in imp["new_columns"] if c in s["used_columns"]]
        lines.append(f"    - seed {s['seed']}: new columns used = {used_new or 'NONE'}")
    return "\n".join(lines)


def ranking_table(res: dict, ngram: str) -> str:
    r = res["ranking"].get(ngram)
    if r is None:
        return f"#### {ngram}: ranking — (skipped)\n"
    lines = [
        f"#### {ngram}: layout ordering over the named field, narrow vs widened (seed 0)",
        "",
        f"- kendall tau(narrow order, widened order): {r['kendall_tau']:+.4f}",
        f"- spearman rho(narrow fitness, widened fitness): {r['spearman_rho']:+.4f}",
        f"- positions moved: {r['n_positions_moved']} of {len(r['fit_narrow'])}  moved={r['moved']}",
        f"- argmin (best) narrow: {r['argmin_narrow']}   widened: {r['argmin_widened']}",
        "",
        "| rank | narrow order | widened order |",
        "|---|---|---|",
    ]
    on, ow = r["order_narrow"], r["order_widened"]
    for i in range(len(on)):
        marker = "" if on[i] == ow[i] else "  <-- moved"
        lines.append(f"| {i} | {on[i]} | {ow[i]}{marker} |")
    return "\n".join(lines)


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        res = json.load(f)
    ngrams = [n for n in ("bigram", "trigram") if n in res]
    out = ["# RETRAIN-DIRECTION — generated result tables", ""]
    out.append(f"seeds: {res['seeds']}   high-wpm floor: {res['high_wpm_floor']}")
    out.append("")
    for ngram in ngrams:
        out.append(f"## {ngram}")
        out.append("")
        out.append(high_wpm_table(res, ngram))
        out.append("")
        out.append(transfer_table(res, ngram))
        out.append("")
        out.append(importance_table(res, ngram))
        out.append("")
        out.append(ranking_table(res, ngram))
        out.append("")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
