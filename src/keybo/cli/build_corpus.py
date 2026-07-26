"""`keybo build-corpus` — regenerate the multi-source n-gram blend from local sources.

The reproducible replacement for the opaque single-source import: it names every source,
records its bytes and SHA-256, applies the declared per-register blend weights, and writes
the tables in the exact production format alongside a `manifest.json`.

    keybo build-corpus --out data/corpus/blend-v1
    keybo build-corpus --out /tmp/prose-only --weights prose=1.0 --no-anchor

The one component that is NOT reproducible is the iWeb `anchor` (licensed source, no
extraction script ever committed); it is consumed as committed derived counts, its hash
recorded, and `--no-anchor` drops it for a fully reproducible blend. See
`data/corpus/blend-v1/PROVENANCE.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from keybo.cli._paths import ensure_writable_output
from keybo.data.build_corpus import (
    DECLARED_TOTAL,
    DEFAULT_WEIGHTS,
    build_blend,
    default_sources,
    write_build,
)
from keybo.data.corpus import IWEB, resolve_corpus_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        default="data/corpus/blend-v1",
        help="Output directory for the tables + manifest.json (default data/corpus/blend-v1)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        metavar="REGISTER=SHARE",
        help=(
            "Override blend weights per register, e.g. --weights anchor=0.4 prose=0.3 "
            f"code=0.2 reference=0.1 (default: {', '.join(f'{k}={v}' for k, v in DEFAULT_WEIGHTS.items())})"
        ),
    )
    parser.add_argument(
        "--no-anchor",
        action="store_true",
        help="Exclude the non-reproducible iWeb anchor (yields a fully reproducible blend)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=DECLARED_TOTAL,
        help=f"Declared integer total each table sums to (default {DECLARED_TOTAL:,})",
    )
    parser.add_argument("--repo", default=None, help="Repo root to read prose sources from")
    parser.add_argument("--stdlib", default=None, help="Python stdlib root for the code register")
    parser.add_argument("--man-root", default=None, help="man page root for the reference register")


def _parse_weights(pairs: list[str] | None) -> dict[str, float]:
    if not pairs:
        return dict(DEFAULT_WEIGHTS)
    weights: dict[str, float] = {}
    for pair in pairs:
        register, _, value = pair.partition("=")
        if not register or not value:
            raise SystemExit(f"--weights entries must be REGISTER=SHARE, got {pair!r}")
        try:
            weights[register] = float(value)
        except ValueError:
            raise SystemExit(f"--weights share must be a number, got {value!r}") from None
    if any(w < 0 for w in weights.values()):
        raise SystemExit("--weights shares must be non-negative")
    if sum(weights.values()) <= 0:
        raise SystemExit("--weights shares must not all be zero")
    return weights


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    ensure_writable_output(str(out_dir / "manifest.json"), flag="--out")
    repo = Path(args.repo) if args.repo else _repo_root()
    weights = _parse_weights(args.weights)

    sources = default_sources(
        repo,
        stdlib=Path(args.stdlib) if args.stdlib else None,
        man_root=Path(args.man_root) if args.man_root else None,
    )
    # ⚠ The anchor is the iWeb source, ALWAYS — deliberately NOT production_corpus_dir().
    # build-corpus CONSUMES the anchor to PRODUCE a blend; pointing it at whatever the
    # production default happens to be would make the builder read its own output once
    # the default became a blend (CORPUS-SWAP-1 made it blend-v1), compounding the blend
    # weights on every rebuild. This is the same self-referential class of bug as the
    # rglob over the repo's own files that already cost blend-v1 byte-reproducibility.
    anchor_dir = None if args.no_anchor else resolve_corpus_dir(IWEB)

    print(f"building blend into {out_dir} (declared total {args.total:,} per table)")
    result = build_blend(sources, weights=weights, anchor_dir=anchor_dir, total=args.total)

    print("\n== sources ==")
    for entry in result.manifest["sources"]:
        flag = "" if entry["reproducible"] else "  [NOT REPRODUCIBLE — trust anchor]"
        size = entry.get("raw_bytes")
        size_text = f"{size:,} B" if size is not None else "committed counts"
        units = entry.get("units")
        unit_text = f", {units:,} files" if units is not None else ""
        print(
            f"  {entry['name']:24s} register={entry['register']:10s} {size_text}{unit_text}{flag}"
        )
        print(
            f"      sha256 {entry.get('sha256', next(iter(entry.get('files', {}).values()), {}).get('sha256', '-'))}"
        )

    print("\n== effective weights ==")
    for register, share in sorted(
        result.manifest["weights_effective"].items(), key=lambda kv: -kv[1]
    ):
        print(f"  {register:12s} {share:.4f}")

    written = write_build(result, out_dir)
    print("\n== outputs ==")
    for kind, info in result.manifest["outputs"].items():
        print(
            f"  {kind:10s} {info['types']:>8,} types  total {info['total']:,}  -> {', '.join(info['files'])}"
        )
    print(f"\nwrote {len(written)} files to {out_dir}")
    return 0
