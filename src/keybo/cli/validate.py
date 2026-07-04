"""`keybo validate` — leave-one-layout-out cross-layout trust check (OQ-5)."""

from __future__ import annotations

import argparse
import json

import numpy as np

from keybo.cli._paths import ensure_writable_output
from keybo.data.strokes import load_strokes


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strokes", required=True, help="Path to the bistroke TSV (new schema)")
    parser.add_argument(
        "--holdout",
        nargs="*",
        default=None,
        help="Layout(s) to hold out (default: every layout in the file, one fold each)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Training seeds; conclusions must hold across all of them",
    )
    parser.add_argument("--wpm-lo", type=int, default=40, help="Lowest WPM bucket edge")
    parser.add_argument("--wpm-hi", type=int, default=140, help="Highest WPM bucket edge")
    parser.add_argument("--bucket-width", type=int, default=20, help="WPM bucket width")
    parser.add_argument(
        "--min-cell-samples",
        type=int,
        default=10,
        help="Refuse (layout, ngram, wpm-bucket) cells with fewer samples than this",
    )
    parser.add_argument(
        "--n-boot", type=int, default=50, help="Bootstrap splits for the noise ceiling"
    )
    parser.add_argument("--hyperparams", help="JSON file of XGBoost params for the fold models")
    parser.add_argument("--out", help="Write the full report JSON to this path")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/a"
    return f"{x:+.3f}"


def run(args: argparse.Namespace) -> int:
    from keybo.training.validate import validate

    if args.out:
        ensure_writable_output(args.out, "--out")
    train_params = {}
    if args.hyperparams:
        with open(args.hyperparams, encoding="utf-8") as f:
            train_params = json.load(f)

    rows = load_strokes(args.strokes, ngram_len=2, wpm_threshold=0, min_samples=1)
    if not rows:
        raise SystemExit(f"no bigram stroke rows loaded from {args.strokes}")

    report = validate(
        rows,
        seeds=args.seeds,
        holdouts=args.holdout,
        wpm_lo=args.wpm_lo,
        wpm_hi=args.wpm_hi,
        bucket_width=args.bucket_width,
        min_cell_samples=args.min_cell_samples,
        n_boot=args.n_boot,
        train_params=train_params,
        progress=not args.no_progress,
    )

    print(
        f"leave-one-layout-out over {len(report['folds'])} fold(s), "
        f"seeds {report['config']['seeds']}, wpm [{args.wpm_lo}, {args.wpm_hi}) "
        f"x{args.bucket_width}, cell floor {args.min_cell_samples}"
    )
    header = (
        f"{'holdout':<10} {'cells':>5} {'ceiling':>8} | per-seed rho (frac of ceiling) | "
        "tau_all | beats-baseline"
    )
    print(header)
    print("-" * len(header))
    for layout, fold in report["folds"].items():
        ceiling = report["ceilings"][layout]
        rhos = " ".join(f"{_fmt(m['rho'])} ({_fmt(m['rho_frac_ceiling'])})" for m in fold["seeds"])
        taus = " ".join(_fmt(m["tau_all4"]) for m in fold["seeds"])
        beats = " ".join("yes" if m["beats_baseline"] else "NO" for m in fold["seeds"])
        print(f"{layout:<10} {fold['n_cells']:>5} {_fmt(ceiling):>8} | {rhos} | {taus} | {beats}")
    pooled = " ".join(f"seed {p['seed']}: {_fmt(p['tau_heldout'])}" for p in report["pooled"])
    print(f"pooled held-out layout ranking tau (fully out-of-sample): {pooled}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"report -> {args.out}")
    return 0
