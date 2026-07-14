"""`keybo effect-curves` — pattern-class price vs WPM (contrast + SHAP attribution)."""

from __future__ import annotations

import argparse
import json

from keybo.analysis.effect_curves import compute_effect_curves, render_effect_curves
from keybo.cli._paths import ensure_writable_output
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        help="Trained bigram model artifact(s); several are ensemble-averaged "
        "(the production 3-seed convention)",
    )
    parser.add_argument(
        "--wpms",
        type=float,
        nargs="+",
        default=[50, 60, 70, 80, 90, 100, 110, 120, 130],
        help="WPM axis for the curves",
    )
    parser.add_argument(
        "--layout",
        help="Weight position pairs by this layout's corpus mass (a named layout like "
        "'qwerty' or a 30-char string); omit for uniform geometry weighting",
    )
    parser.add_argument(
        "--bigrams", help="Corpus bigram table (required with --layout)", default=None
    )
    parser.add_argument("--out-prefix", default="runs/effect_curves")
    parser.add_argument(
        "--out-dir",
        help="Artifact directory: writes curves.json + PNGs inside (overrides --out-prefix)",
    )


def _load_freqs(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def run(args: argparse.Namespace) -> int:
    import os

    models = [XGBoostTypingModel.load(p) for p in args.model]

    layout = None
    freqs = None
    if args.layout:
        if not args.bigrams:
            raise SystemExit("--layout needs --bigrams (the corpus table to weight by)")
        s = NAMED_LAYOUTS.get(args.layout, args.layout)
        layout = Layout(s, ROW_STAGGERED_30)
        freqs = _load_freqs(args.bigrams)

    curves = compute_effect_curves(models, wpms=list(args.wpms), layout=layout, bigram_freqs=freqs)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        prefix = os.path.join(args.out_dir, "curves")
        json_path = os.path.join(args.out_dir, "curves.json")
    else:
        prefix = args.out_prefix
        json_path = f"{args.out_prefix}.json"
    ensure_writable_output(json_path)
    with open(json_path, "w") as f:
        json.dump(curves.to_dict(), f, indent=1)
    written = render_effect_curves(curves, prefix)
    written.append(json_path)

    print(f"wpms: {curves.wpms}")
    print(f"pair weighting: {curves.weighted_by}")
    header = f"{'class':<16}" + "".join(f"{int(w):>8}" for w in curves.wpms)
    print(f"\ncontrast vs alternate (ms):\n{header}")
    for cls, ys in curves.contrast_ms.items():
        if cls == "alternate":
            continue
        print(f"{cls:<16}" + "".join(f"{y:>8.1f}" for y in ys))
    pct = curves.contrast_pct()
    print(f"\ncontrast vs alternate (% of alternate mean):\n{header}")
    for cls, ys in pct.items():
        if cls == "alternate":
            continue
        print(f"{cls:<16}" + "".join(f"{y:>7.1f}%" for y in ys))
    for note in curves.notes:
        print(f"note: {note}")
    print()
    for p in written:
        print(f"wrote {p}")
    return 0
