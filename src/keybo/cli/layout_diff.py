"""`keybo layout-diff` — frequency-weighted n-gram impact diff between two layouts."""

from __future__ import annotations

import argparse
import json

from keybo.analysis.layout_diff import diff_layouts, render_diff
from keybo.cli._paths import ensure_writable_output
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("layout_a", help="Baseline layout (named like 'qwerty' or 30 chars)")
    parser.add_argument("layout_b", help="Comparison layout (named or 30 chars)")
    parser.add_argument(
        "--bigram-model", required=True, nargs="+",
        help="Bigram model artifact(s), ensemble-averaged",
    )
    parser.add_argument(
        "--trigram-model", nargs="+", default=None,
        help="Conditioned-trigram model artifact(s); required for --ngrams trigram",
    )
    parser.add_argument(
        "--ngrams", choices=["bigram", "trigram"], default="trigram",
        help="Which n-gram level to diff",
    )
    parser.add_argument("--bigrams", required=True, help="Corpus bigram table (freq join + T2)")
    parser.add_argument("--trigrams", help="Corpus trigram table (for --ngrams trigram)")
    parser.add_argument("--target-wpm", type=float, default=90.0)
    parser.add_argument("--top", type=int, default=20, help="How many impacts to report")
    parser.add_argument("--out-prefix", default="runs/layout_diff")


def _load_freqs(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def _resolve(s: str) -> Layout:
    return Layout(NAMED_LAYOUTS.get(s, s), ROW_STAGGERED_30)


def run(args: argparse.Namespace) -> int:
    layout_a = _resolve(args.layout_a)
    layout_b = _resolve(args.layout_b)
    bi_models = [XGBoostTypingModel.load(p) for p in args.bigram_model]
    bigram_freqs = _load_freqs(args.bigrams)

    if args.ngrams == "trigram":
        if not args.trigram_model or not args.trigrams:
            raise SystemExit("--ngrams trigram needs --trigram-model and --trigrams")
        tri_models = [XGBoostTypingModel.load(p) for p in args.trigram_model]
        freqs = _load_freqs(args.trigrams)
        diff = diff_layouts(
            layout_a, layout_b, bi_models, freqs,
            trigram_models=tri_models, bigram_freqs=bigram_freqs,
            target_wpm=args.target_wpm,
        )
    else:
        diff = diff_layouts(
            layout_a, layout_b, bi_models, bigram_freqs,
            bigram_freqs=bigram_freqs, target_wpm=args.target_wpm,
        )

    ensure_writable_output(f"{args.out_prefix}.json")
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(diff.to_dict(k=args.top), f, indent=1)
    png = render_diff(diff, f"{args.out_prefix}.png", k=args.top)

    pct = 100.0 * diff.total_delta / diff.total_a if diff.total_a else 0.0
    print(f"A: {args.layout_a}\nB: {args.layout_b}")
    print(f"total {args.ngrams} objective: A {diff.total_a:.4g}  B {diff.total_b:.4g}  "
          f"delta {diff.total_delta:+.4g} ({pct:+.3f}%; negative = B faster)")
    print(f"\ntop {args.top} impacts (impact = freq × Δms):")
    print(f"{'ngram':<8}{'freq':>12}{'t_A ms':>9}{'t_B ms':>9}{'Δms':>8}{'impact':>14}  moved")
    for i in diff.top(args.top):
        print(f"{i.ngram.replace(' ', '␣'):<8}{i.freq:>12,}{i.t_a_ms:>9.1f}{i.t_b_ms:>9.1f}"
              f"{i.t_b_ms - i.t_a_ms:>8.1f}{i.impact:>14.3g}  {i.moved_chars}")
    print(f"\nwrote {args.out_prefix}.json + {png}")
    return 0
