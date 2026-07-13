"""`keybo shap-report` — SHAP feature-importance report for a trained model."""

from __future__ import annotations

import argparse
import json

import numpy as np

from keybo.analysis.shap_report import compute_shap, render_report
from keybo.cli._paths import ensure_writable_output
from keybo.data.strokes import load_strokes
from keybo.features import bigram_features_from_positions, trigram_features_from_positions
from keybo.geometry import ROW_STAGGERED_30
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.training.train import build_training_matrix


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Trained model artifact (.json)")
    parser.add_argument(
        "--on",
        choices=["grid", "data"],
        default="grid",
        help="Explain the serve grid (all position n-grams at --target-wpm; what drives "
        "the optimizer's table) or the training data distribution (needs --strokes)",
    )
    parser.add_argument(
        "--target-wpm", type=float, default=90.0, help="Scoring WPM for the grid matrix"
    )
    parser.add_argument("--strokes", help="Stroke TSV for --on data (bistrokes/tristrokes)")
    parser.add_argument(
        "--max-rows", type=int, default=50000, help="Row cap for the explained matrix"
    )
    parser.add_argument("--top-k", type=int, default=12, help="Features in detail panels")
    parser.add_argument(
        "--out-prefix",
        default="runs/shap",
        help="Path prefix for outputs: <prefix>_{ranking,beeswarm,dependence,interactions}.png "
        "+ <prefix>.json",
    )
    parser.add_argument(
        "--out-dir",
        help="Artifact directory: writes ranking.png, beeswarm.png, dependence.png, "
        "interactions.png, and report.json inside it (overrides --out-prefix)",
    )


def _grid_matrix(model: XGBoostTypingModel, target_wpm: float, max_rows: int) -> np.ndarray:
    geom = ROW_STAGGERED_30
    positions = [*geom.slots, geom.space_position]
    if model.metadata.ngram == "bigram":
        return np.vstack(
            [
                bigram_features_from_positions(geom, (a, b), wpm=target_wpm)
                for a in positions
                for b in positions
            ]
        )
    rows = [
        trigram_features_from_positions(geom, (a, b, c), wpm=target_wpm)
        for a in positions
        for b in positions
        for c in positions
    ]
    X = np.vstack(rows)
    if X.shape[0] > max_rows:
        idx = np.random.default_rng(0).choice(X.shape[0], max_rows, replace=False)
        X = X[idx]
    return X


def _data_matrix(model: XGBoostTypingModel, strokes_path: str, max_rows: int) -> np.ndarray:
    ngram = model.metadata.ngram
    rows = load_strokes(
        strokes_path, ngram_len=2 if ngram == "bigram" else 3, wpm_threshold=0, min_samples=1
    )
    X, _y = build_training_matrix(rows, ngram=ngram, target_wpm=0.0, progress=True)
    if X.shape[0] > max_rows:
        idx = np.random.default_rng(0).choice(X.shape[0], max_rows, replace=False)
        X = X[idx]
    return X


def run(args: argparse.Namespace) -> int:
    import os

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        prefix = os.path.join(args.out_dir, "report")
        json_path = os.path.join(args.out_dir, "report.json")
    else:
        prefix = args.out_prefix
        json_path = f"{args.out_prefix}.json"
    ensure_writable_output(json_path, "--out-dir" if args.out_dir else "--out-prefix")
    if args.on == "data" and not args.strokes:
        print("--on data requires --strokes <bistrokes/tristrokes tsv>")
        return 1

    model = XGBoostTypingModel.load(args.model)
    if args.on == "grid":
        X = _grid_matrix(model, args.target_wpm, args.max_rows)
    else:
        X = _data_matrix(model, args.strokes, args.max_rows)
    print(f"explaining {X.shape[0]} rows x {X.shape[1]} features ({args.on} matrix)")

    report = compute_shap(model, X)
    paths = render_report(report, prefix, top_k=args.top_k)
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    paths.append(json_path)

    share = report.importance_share()
    print(f"base value: {report.base_value:.1f} ms")
    print("top features (mean |SHAP| ms, share of importance, signed mean):")
    for name, mean_abs, mean_signed in report.ranking()[: args.top_k]:
        print(f"  {name:24s} {mean_abs:8.2f}  {share[name]:5.1f}%  {mean_signed:+8.2f}")
    if report.interaction_pairs:
        print("top interactions (mean |interaction| ms):")
        for a, b, v in report.interaction_pairs[:5]:
            print(f"  {a} x {b}: {v:.2f}")
    for p in paths:
        print(f"wrote {p}")
    return 0
