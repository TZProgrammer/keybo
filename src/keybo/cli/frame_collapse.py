"""`keybo frame-collapse` — what a feature frame CANNOT distinguish, with no model and no training.

The cheapest frame audit in the repo: it answers "will this frame be able to price these cells at
all?" in ~2 seconds, before any model exists. See :mod:`keybo.analysis.frame_collapse` for the
estimators, the tolerance rule, and the nine things it CANNOT tell you.

Built-in frames are the campaign's own: ``served``, ``interp``, ``interp-wpm``, ``direction`` and
``kitchensink`` at order 2, and ``trigram``/``trigram-direction``/``trigram-kitchensink`` at order 3.
``--frame`` also accepts ``module:callable`` for a frame that does not exist yet — which is the whole
point of a BEFORE-training diagnostic, and the reason this command does not hard-code a frame list.
"""

from __future__ import annotations

import argparse
import importlib
import json as _json

import numpy as np

from keybo.analysis.frame_collapse import (
    FrameCollapse,
    cell_positions,
    format_report,
    frame_collapse,
    sweep_verdict,
    tolerance_sweep,
)
from keybo.data.corpus import CORPUS_ENV_VAR, PRODUCTION_DEFAULT, known_corpora
from keybo.features import (
    bigram_features_from_positions,
    interp_features_from_positions,
    interp_wpm_features_from_positions,
    trigram_features_from_positions,
)

#: name -> (order, builder). Each builder takes the scoring WPM and returns a featurizer, so ``wpm``
#: reaches the frames that carry it as a column WITHOUT the caller having to know which those are.
#: ``direction``/``kitchensink`` are FLAGS on the two ``*_features_from_positions`` entry points, not
#: separate functions, so the wider frames are spelled here as those flags rather than as imports —
#: which also means this registry cannot drift from the column lists those flags select.
_FRAMES: dict[str, tuple[int, object]] = {
    "served": (2, lambda wpm: lambda g, c: bigram_features_from_positions(g, c, wpm=wpm)),
    "interp": (2, lambda wpm: lambda g, c: interp_features_from_positions(g, c, wpm=wpm)),
    "interp-wpm": (2, lambda wpm: lambda g, c: interp_wpm_features_from_positions(g, c, wpm=wpm)),
    "direction": (
        2,
        lambda wpm: lambda g, c: bigram_features_from_positions(g, c, wpm=wpm, direction=True),
    ),
    "kitchensink": (
        2,
        lambda wpm: lambda g, c: bigram_features_from_positions(g, c, wpm=wpm, kitchensink=True),
    ),
    "trigram": (3, lambda wpm: lambda g, c: trigram_features_from_positions(g, c, wpm=wpm)),
    "trigram-direction": (
        3,
        lambda wpm: lambda g, c: trigram_features_from_positions(g, c, wpm=wpm, direction=True),
    ),
    "trigram-kitchensink": (
        3,
        lambda wpm: lambda g, c: trigram_features_from_positions(g, c, wpm=wpm, kitchensink=True),
    ),
}

#: The frames reported when ``--frame`` is not given: the served frame and the two INTERPFRAME-1
#: compared it against. Chosen so the default invocation reproduces the ledger's own table.
_DEFAULT_FRAMES = ("served", "interp", "interp-wpm")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frame",
        action="append",
        default=None,
        help=(
            f"Frame to diagnose, repeatable. Built-ins: {' | '.join(_FRAMES)} "
            f"(default: {' '.join(_DEFAULT_FRAMES)}). Also accepts 'module:callable' for a frame "
            "not in the registry — the callable takes (geometry, cell_tuple) and returns the "
            "feature row; pair it with --order for a non-bigram frame"
        ),
    )
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help=(
            "N-gram order of the cell space (2 = bigram, P**2 cells; 3 = trigram, P**3). "
            "Defaults to the built-in frame's own order; REQUIRED for a 'module:callable' frame "
            "that is not order 2"
        ),
    )
    parser.add_argument(
        "--geometry",
        choices=["k30", "k31"],
        default="k30",
        help=(
            "Board geometry (default k30 — the geometry the production time surface is built on, "
            "so its cell space matches the target table cell-for-cell)"
        ),
    )
    parser.add_argument(
        "--no-space",
        action="store_true",
        help=(
            "Exclude the space/thumb position from the cell space. ⚠ k30 WITH space and k31 WITHOUT "
            "it both give 31 positions / 961 cells but are DIFFERENT cell spaces with different "
            "answers (765 vs 775 distinct served rows) — this flag is that distinction"
        ),
    )
    parser.add_argument(
        "--target-wpm",
        type=float,
        default=90.0,
        help="WPM passed to the featurizer and used for the time surface (default 90)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help=(
            "Feature-equality tolerance: 0 (default) = EXACT bitwise; >0 quantizes each column to "
            "round(x/tol). The count can never exceed the EXACT count, but it is NOT monotone "
            "between two nonzero tolerances (bin boundaries move with tol), so any number produced "
            "at tol>0 must be reported WITH its tol"
        ),
    )
    parser.add_argument(
        "--tolerance-sweep",
        action="store_true",
        help=(
            "Report the distinct-row count at each of several tolerances instead of one table — "
            "the check for whether a frame's headline number depends on a float tolerance at all"
        ),
    )
    parser.add_argument(
        "--tols",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Tolerances for --tolerance-sweep (default 0 1e-15 1e-12 1e-9 1e-6 1e-3). Pass a wider "
            "list to find where a frame's count starts moving at all"
        ),
    )
    parser.add_argument(
        "--floor",
        action="store_true",
        help=(
            "Also compute the WITHIN-GROUP FLOOR against the production time surface: the best "
            "error any model on the frame could achieve. Loads 6 models, so it is the slow path "
            "(~10 s at order 2). Without it only the collapse structure is reported"
        ),
    )
    parser.add_argument(
        "--weight-layout",
        default="flagship-c3",
        help=(
            "Layout whose corpus bigram mass weights the cells for the mass share and the weighted "
            "floor (name or 30-char string; default flagship-c3). Only used with --floor"
        ),
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            f"Corpus for the weights and the surface: {' | '.join(known_corpora())} or a directory "
            f"(default {PRODUCTION_DEFAULT}; env: {CORPUS_ENV_VAR}). Only used with --floor"
        ),
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="With --floor, weight every cell equally instead of by corpus mass",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON object instead of text")


def _load_frame(spec: str, wpm: float) -> tuple[int, object]:
    """``(order, featurizer)`` for a built-in name or a ``module:callable`` path."""
    if spec in _FRAMES:
        order, builder = _FRAMES[spec]
        return order, builder(wpm)
    if ":" not in spec:
        raise SystemExit(
            f"unknown frame {spec!r}: not a built-in ({', '.join(_FRAMES)}) and not 'module:callable'"
        )
    mod_name, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise SystemExit(f"cannot import {mod_name!r} for frame {spec!r}: {e}") from e
    if not hasattr(mod, attr):
        raise SystemExit(f"{mod_name!r} has no attribute {attr!r} (frame {spec!r})")
    return 2, getattr(mod, attr)


def _surface_target(order: int, wpm: float, corpus: str | None):
    """``(geometry, target, weights_or_None, description)`` from the production time surface.

    The surface is what makes the floor MEANINGFUL: its per-cell millisecond table is the true value
    a model on any frame has to hit. At order 2 that is ``_T2``; at order 3 it is
    ``triple_ms_table()`` = ``T2[a,b] + Tcond[a,b,c]``, i.e. the same quantity ``card`` accumulates,
    so the order-3 floor is against the SHIPPED trigram surface rather than a re-derived one.
    """
    from keybo.analysis.timecard import default_surface

    surface = default_surface(wpm, corpus)
    if order == 2:
        return surface, surface._T2.ravel(), f"production T2 surface @ {wpm:g} WPM"
    if order == 3:
        return (
            surface,
            surface.triple_ms_table().ravel(),
            (f"production T2+Tcond surface @ {wpm:g} WPM"),
        )
    raise SystemExit(f"--floor has no surface target defined for order {order}")


def _cell_weights(surface, layout_spec: str, order: int, n_positions: int) -> np.ndarray:
    """Corpus mass per cell, in the odometer order :func:`feature_matrix` produces.

    Accumulated with ``np.add.at`` over the layout's character permutation, exactly as
    ``agent-artifacts/interpframe/resolution.py`` did — one weight path, so the mass share cannot
    drift from the weight the floor is computed under.
    """
    from keybo.analysis.shap_diff import _char_weight_tables
    from keybo.cli.analyze import _resolve

    _name, chars = _resolve(layout_spec)
    slot = surface._slot_of(chars)
    w3, w2, _covered = _char_weight_tables(surface, chars)
    perm = np.array([slot[c] for c in chars] + [slot[" "]], dtype=np.intp)
    grid = np.zeros((n_positions,) * order)
    if order == 2:
        np.add.at(grid, (perm[:, None], perm[None, :]), w2)
    elif order == 3:
        np.add.at(grid, (perm[:, None, None], perm[None, :, None], perm[None, None, :]), w3)
    else:
        raise SystemExit(f"no corpus weight table defined for order {order}")
    return grid.ravel()


def run(args: argparse.Namespace) -> int:
    from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31

    specs = list(args.frame) if args.frame else list(_DEFAULT_FRAMES)
    frames = {spec: _load_frame(spec, args.target_wpm) for spec in specs}
    orders = {args.order if args.order is not None else o for o, _f in frames.values()}
    if len(orders) > 1:
        raise SystemExit(
            f"frames of mixed order cannot share one cell space: {sorted(orders)}. "
            "Run one order per invocation, or pass --order to pin it"
        )
    order = orders.pop()

    geometry = ROW_STAGGERED_30 if args.geometry == "k30" else ROW_STAGGERED_31
    include_space = not args.no_space
    positions = cell_positions(geometry, include_space=include_space)

    target = weights = None
    target_name = ""
    if args.floor:
        surface, target, target_name = _surface_target(order, args.target_wpm, args.corpus)
        if surface.geometry is not geometry:
            raise SystemExit(
                f"--floor needs the cell space the surface is built on: the surface uses "
                f"{len(surface.geometry.slots)} slots but --geometry {args.geometry} has "
                f"{len(geometry.slots)}. The target table would not line up with the cells"
            )
        if args.no_space:
            raise SystemExit(
                "--floor is incompatible with --no-space: the surface's target table is built over "
                "slots PLUS space, so a space-less cell space has no matching target"
            )
        if not args.unweighted:
            weights = _cell_weights(surface, args.weight_layout, order, len(positions))
            target_name += f", corpus-weighted by {args.weight_layout}"
        else:
            target_name += ", unweighted"

    common = dict(
        geometry=geometry,
        order=order,
        positions=positions,
        target=target,
        weights=weights,
    )

    if args.tolerance_sweep:
        sweep_kwargs = dict(common)
        if args.tols is not None:
            sweep_kwargs["tols"] = args.tols
        sweeps = {
            spec: tolerance_sweep(feat, **sweep_kwargs) for spec, (_o, feat) in frames.items()
        }
        if args.json:
            print(
                _json.dumps(
                    {
                        "target": target_name or None,
                        "frames": {s: [r.as_dict() for r in rs] for s, rs in sweeps.items()},
                    },
                    indent=1,
                )
            )
            return 0
        print(_format_sweep(sweeps))
        return 0

    results: dict[str, FrameCollapse] = {
        spec: frame_collapse(feat, tol=args.tol, **common) for spec, (_o, feat) in frames.items()
    }
    if args.json:
        print(
            _json.dumps(
                {
                    "target": target_name or None,
                    "frames": {s: r.as_dict() for s, r in results.items()},
                },
                indent=1,
            )
        )
        return 0
    print(format_report(results, target_name=target_name))
    return 0


def _format_sweep(sweeps: dict[str, list[FrameCollapse]]) -> str:
    """The tolerance table: distinct rows per frame per tolerance, with the flatness verdict."""
    tols = [r.tol for r in next(iter(sweeps.values()))]
    width = max(len(k) for k in sweeps) + 2
    head = f"{'frame':<{width}}" + "".join(f"{('exact' if t == 0 else f'{t:g}'):>12}" for t in tols)
    lines = [
        f"FRAME COLLAPSE — distinct feature rows vs equality tolerance "
        f"({next(iter(sweeps.values()))[0].n_cells} cells)",
        head,
        "-" * len(head),
    ]
    verdicts = {name: sweep_verdict(rs) for name, rs in sweeps.items()}
    for name, rs in sweeps.items():
        lines.append(f"{name:<{width}}" + "".join(f"{r.distinct_feature_rows:>12}" for r in rs))
    flat = [n for n, v in verdicts.items() if v["flat"]]
    lines += [
        "",
        "Every count is bounded above by the EXACT count (quantization coarsens the exact partition).",
        "A flat row means the frame's headline number does not depend on the tolerance at all.",
        f"FLAT across this sweep: {', '.join(flat) if flat else '(none)'}",
    ]
    moved = [n for n in sweeps if n not in flat]
    if moved:
        lines.append(f"TOLERANCE-SENSITIVE (report the tol with the number): {', '.join(moved)}")
    risen = {n: v["rises"] for n, v in verdicts.items() if v["rises"]}
    if risen:
        lines += [
            "",
            "NOTE — a coarser tolerance produced MORE rows than a finer one on: "
            + ", ".join(
                f"{n} ({', '.join(f'{lo:g}->{hi:g}' for lo, hi in rs)})" for n, rs in risen.items()
            ),
            "This is REAL, not a bug: round(x/tol) bin boundaries move with tol, so a coarser grid",
            "can split a pair a finer grid merged (round(0.3/0.5)==round(0.4/0.5) but",
            "round(0.3/0.75)!=round(0.4/0.75)). It is why tol=0 (exact) is the default.",
        ]
    broken = [n for n, v in verdicts.items() if v["exceeds_exact"]]
    if broken:
        lines.append(
            f"!! BUG: exceeded the EXACT count on {', '.join(broken)} — that is impossible; "
            "the grouping rule is wrong."
        )
    return "\n".join(lines) + "\n"
