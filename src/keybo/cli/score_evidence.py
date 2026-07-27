"""`keybo score-evidence` — score layouts with FITTED weights, and validate them.

Two modes, because a scorer nobody has tested against the incumbents is not worth
shipping:

* default — derive a price per gauge from a fitted surface over a layout pool, print the
  weight table (per gauge AND per correlation cluster), and score the requested layouts;
* ``--validate`` — the out-of-sample test that decides whether the weights are worth
  using: leave-one-source-out against genkey / oxeylyzer-1 / oxeylyzer-2, with a noise
  placebo and the paired resolution floor.

Both modes print the modelled-only caveat and the frame every number lives on, in the
tool's own output rather than only in a write-up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from keybo.analysis import evidence_scorer as E
from keybo.analysis import evidence_validation as V
from keybo.analysis import surfaces as S
from keybo.cli._paths import ensure_writable_output
from keybo.data.corpus import CORPUS_ENV_VAR, IWEB, PRODUCTION_DEFAULT, known_corpora
from keybo.layouts import NAMED_LAYOUTS

#: Campaign layouts worth having on tap, mirroring `keybo analyze`'s registry so the two
#: commands name the same boards.
_EXTRA_NAMED = {
    "keybo-c30m": "fyu,.vgdnlhieaocstrmkj'q-bwpxz",
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "qwerty30m": V.QWERTY30M,
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "p16-balance": "frlwg'uyoksntdc.ieahvxmpb,-jqz",
}

#: The pool the weights are fitted over, when no explicit pool file is given: random C30M
#: permutations. Deliberately the WIDE pool for the default, because a curve fitted only on
#: optimized layouts has a narrow domain and would flag every ordinary board as
#: extrapolation.
POOL_KINDS = ("random", "archive", "file")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "layouts",
        nargs="*",
        help="Layouts to score: registry names and/or 30-char C30M strings",
    )
    parser.add_argument(
        "--surface-dir",
        required=True,
        help=(
            "Directory holding <NAME>.<frame>.npy fitted surfaces. REQUIRED: the repo "
            "vendors only the .standardized arrays, and those share the AALTO bigram "
            "tensor across sources, which weakens every cross-source claim"
        ),
    )
    parser.add_argument(
        "--surface-frame",
        default="native",
        choices=E.SURFACE_FRAMES,
        help=(
            "native (default) keeps each source's OWN bigram tensor — the honest frame for "
            "a cross-source claim; standardized substitutes the production AALTO tensor "
            "into all sources and is a labelled sensitivity only"
        ),
    )
    parser.add_argument(
        "--fit-source",
        default="COMMUNITY_BASE",
        help=(
            "Surface the weights are derived from (default COMMUNITY_BASE — chosen because "
            "AALTO is NOT independent of data/models/k31: the time card's served surface is "
            "bit-identical to AALTO_BASE)"
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Surfaces to use in --validate (default: every surface present in --surface-dir)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            f"Corpus for the gauge frame and the objective weighting: "
            f"{' | '.join(known_corpora())}, or a directory (default {PRODUCTION_DEFAULT}; "
            f"env: {CORPUS_ENV_VAR}). '--corpus {IWEB}' reproduces the frozen boards"
        ),
    )
    parser.add_argument(
        "--pool",
        default="random",
        choices=POOL_KINDS,
        help=(
            "Layout pool the weights are fitted over: random C30M permutations (default, "
            "the WIDE domain), archive (needs --pool-file with a Pareto archive), or file"
        ),
    )
    parser.add_argument(
        "--pool-file",
        default=None,
        help="JSON for --pool archive/file: a list of layout strings, or {'archive': [{'layout': ...}]}",
    )
    parser.add_argument(
        "--pool-size", type=int, default=300, help="Layouts sampled into the fitting pool"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for pool draw and fits")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the out-of-sample validation (leave-one-source-out vs the community scorers)",
    )
    parser.add_argument(
        "--lolo-folds",
        type=int,
        default=10,
        help="Grouped leave-one-layout-out folds in --validate (0 = true leave-one-out)",
    )
    parser.add_argument(
        "--placebo-repeats", type=int, default=20, help="Shuffled-label repeats for the placebo"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="Bootstrap replicates for paired advantage"
    )
    parser.add_argument("--out", default=None, help="Write the full result as JSON here")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")


def _resolve(spec: str) -> tuple[str, str]:
    key = spec.lower()
    if key in NAMED_LAYOUTS:
        return key, NAMED_LAYOUTS[key]
    if key in _EXTRA_NAMED:
        return key, _EXTRA_NAMED[key]
    if len(spec) == 30:
        return spec, spec
    raise SystemExit(
        f"unknown layout {spec!r}: not a registry name "
        f"({', '.join(sorted({**NAMED_LAYOUTS, **_EXTRA_NAMED}))}) and not a 30-char string"
    )


def _load_pool(args: argparse.Namespace) -> tuple[list[str], str]:
    """The fitting pool, plus a label naming how it was drawn."""
    rng = np.random.default_rng(args.seed)
    if args.pool == "random":
        pool = ["".join(rng.permutation(list(S.C30M))) for _ in range(args.pool_size)]
        return pool, f"random-c30m-{len(pool)}"
    if not args.pool_file:
        raise SystemExit(f"--pool {args.pool} needs --pool-file")
    data = json.loads(Path(args.pool_file).read_text())
    if isinstance(data, dict):
        entries = data.get("archive") or data.get("layouts") or []
        candidates = [e["layout"] if isinstance(e, dict) else e for e in entries]
    else:
        candidates = [e["layout"] if isinstance(e, dict) else e for e in data]
    candidates = [c for c in candidates if S.is_c30m(c)]
    if not candidates:
        raise SystemExit(f"--pool-file {args.pool_file!r} yielded no C30M layouts")
    # Deduplicate but keep order deterministic, then sample.
    unique = list(dict.fromkeys(candidates))
    if len(unique) > args.pool_size:
        idx = rng.choice(len(unique), args.pool_size, replace=False)
        unique = [unique[i] for i in sorted(idx)]
    return unique, f"{args.pool}-{len(unique)}"


def _available(surface_dir: str, frame: str) -> list[str]:
    """Surface names present in ``surface_dir`` for ``frame``, in report order."""
    directory = Path(surface_dir)
    names = []
    for family in S.FAMILIES:
        for name in S.surface_names(family):
            if (directory / f"{name}.{frame}.npy").is_file():
                names.append(name)
    return names


def _print_weights(weights: E.EvidenceWeights) -> None:
    print(f"\nfitted on {weights.source} ({weights.frame} frame), corpus {weights.corpus}")
    print(f"pool: {weights.pool_label}, n={weights.n_layouts}; frame: {E.FRAME_NOTE}")
    print(
        f"surrogate R^2 in-sample {weights.surrogate_r2_in_sample:.4f}"
        + (
            f", held-out {weights.surrogate_r2_holdout:.4f}"
            if weights.surrogate_r2_holdout is not None
            else ""
        )
        + f"; effective dof over {len(E.LIVE_GAUGES)} gauges = {weights.effective_dof:.2f}"
    )
    print(
        f"\n{'gauge':<12}{'share%':>7}{'weight':>11}{'ci95':>22}{'form':>10}{'R2':>7}"
        f"{'domain':>22}{'sign':>6}"
    )
    for row in weights.weight_table():
        low, high = row["weight_ci95"]
        domain = f"[{row['valid_domain'][0]:.3g},{row['valid_domain'][1]:.3g}]"
        # "??" marks a sign that contradicts the mechanism — see EvidenceWeights.sign_audit.
        sign = {True: "ok", False: "??", None: "-"}[row["sign_plausible"]]
        print(
            f"{row['metric']:<12}{row['shap_share_pct']:>7.1f}{row['weight_ms_per_unit']:>11.4f}"
            f"{f'[{low:+.4f},{high:+.4f}]':>22}{row['form']:>10}{row['r2']:>7.3f}{domain:>22}"
            f"{sign:>6}"
        )
    print(f"\nper correlation cluster (effective dof {weights.effective_dof:.2f}):")
    for key, members in sorted(
        weights.clusters.items(), key=lambda kv: -weights.cluster_shap_share_pct[kv[0]]
    ):
        print(
            f"  {weights.cluster_shap_share_pct[key]:5.1f}%  [{len(members)}] {key}"
            f"   sum weight {weights.cluster_weight[key]:+.4f}"
        )
    print(
        "\n  NOTE: sum the CLUSTER column, not the gauge column — the gauges restate each "
        "other (lsb|lsb-dist rho 1.00, sr-roll a subset of roll, oxey-style R^2 0.9937 on "
        "six others), so summing 14 gauge prices over-counts."
    )
    audit = weights.sign_audit()
    if audit["n_implausible"]:
        listed = ", ".join(f"{r['metric']}({r['weight']:+.3f})" for r in audit["implausible"])
        print(
            f"\n  ⚠ SIGN AUDIT: {audit['n_implausible']} of {audit['n_checked']} fitted signs "
            f"CONTRADICT the mechanism: {listed}."
            f"\n    {audit['interpretation']}"
        )
    warning = weights.transfer_warning()
    if warning:
        print(f"\n  ⚠ {warning}")


def _print_validation(report: V.ValidationReport) -> None:
    headline = report.headline()
    print("\n=== OUT-OF-SAMPLE VALIDATION (independent cells only) ===")
    print(
        f"pool {report.pool_label}, n={report.n_layouts}, corpus {report.corpus}, "
        f"{report.surface_frame} frame"
    )
    if not headline.get("cells"):
        print("no independent cell available — cannot make a cross-source claim")
        return
    print(
        f"\n{'fit -> test':<34}{'evidence':>10}{'best rival':>13}{'rival rho':>11}"
        f"{'delta':>9}{'ci95':>20}{'p>0':>7}{'placebo':>9}"
    )
    for row in headline["cells"]:
        low, high = row["delta_vs_best_ci95"]
        cell = f"{row['fit_source']} -> {row['test_source']}"
        placebo = row["placebo_spearman"]
        print(
            f"{cell:<34}{row['evidence_spearman']:>10.4f}{row['best_competitor']:>13}"
            f"{row['best_competitor_spearman']:>11.4f}{row['delta_vs_best']:>9.4f}"
            f"{f'[{low:+.3f},{high:+.3f}]':>20}{row['p_gt_0']:>7.3f}"
            f"{(f'{placebo:+.4f}' if placebo is not None else 'n/a'):>9}"
        )
    print(f"\nVERDICT: {headline['verdict']}")
    print(
        f"  mean delta rho vs best rival {headline['mean_delta_spearman_vs_best_competitor']:+.4f}"
        f"  (min {headline['min_delta']:+.4f}, max {headline['max_delta']:+.4f})"
    )
    ceiling = report.source_agreement
    if ceiling and np.isfinite(ceiling.get("mean", float("nan"))):
        print(
            f"\nCEILING — how well the INDEPENDENT sources agree with EACH OTHER on this pool: "
            f"mean rho {ceiling['mean']:+.4f} (min {ceiling['min']:+.4f}, max {ceiling['max']:+.4f})."
            f"\n  No scorer fitted on source A can be expected to rank source B better than A "
            f"ranks B. A LOW ceiling means a poor showing is a property of the POOL, not the "
            f"scorer:\n  the same pipeline wins 12/12 cells at ceiling +0.835 (random pool) and "
            f"loses 12/12 at +0.265 (near-optimal archive pool)."
        )
    if headline.get("evidence_rho_inside_placebo_band"):
        print(
            "\n  ⚠ EVERY cell's evidence rho lies INSIDE the noise-placebo band, so these "
            "weights do not transfer distinguishably from noise on this pool. Do not read the "
            "small positive correlations as weak-but-real transfer."
        )
    placebo = report.placebo
    print(
        f"\nNOISE PLACEBO ({placebo['repeats']} shuffled-label refits): "
        f"mean rho {placebo['spearman_mean']:+.4f}, mean |rho| {placebo['spearman_abs_mean']:.4f}, "
        f"p95 |rho| {placebo['spearman_abs_p95']:.4f}, range "
        f"[{placebo['spearman_min']:+.4f}, {placebo['spearman_max']:+.4f}]"
    )
    if report.lolo:
        print("\nLEAVE-ONE-LAYOUT-OUT (held-out predictions only):")
        for result in report.lolo:
            rivals = ", ".join(
                f"{name} {value:+.4f}" for name, value in result.competitor_spearman.items()
            )
            flag = "" if result.independent else "  [NOT independent]"
            print(
                f"  {result.fit_source} -> {result.test_source}: evidence "
                f"{result.spearman_held_out:+.4f} ({result.n_folds} folds); {rivals}{flag}"
            )
    if report.resolution:
        resolution = report.resolution
        print(
            f"\nPAIRED RESOLUTION ({resolution['n_seeds']} seeds, {resolution['n_layouts']} layouts): "
            f"unpaired floor {resolution['unpaired_floor_ms_per_trigram']:.4f}, "
            f"PAIRED floor {resolution['paired_floor_ms_per_trigram']:.4f} ms/trigram "
            f"(ratio {resolution['paired_over_unpaired']:.3f}); "
            f"SS layout/seed/resid = "
            f"{resolution['ss_share_pct']['layout']:.1f}/{resolution['ss_share_pct']['seed']:.1f}/"
            f"{resolution['ss_share_pct']['residual']:.1f}%"
        )
    print("\nCOMPETITOR DIRECTION CHECK (derived from qwerty-is-worst, not from metadata):")
    for name, verdict in report.competitor_orientation.items():
        print(f"  {name:<12} {verdict}")
    proof = report.direction_proof
    print(
        f"\nDIRECTION INVARIANCE (recomputed over {proof['ordered_pairs_checked']} ordered pairs): "
        f"max non-landing feature diff = {proof['max_abs_nonlanding_feature_diff']:.3e} -> "
        f"{proof['verdict']}"
    )
    print("\nWHAT THIS SCORER CANNOT EXPRESS:")
    for limitation in report.limitations:
        print(f"  - {limitation.name}: {limitation.verdict}")
    print(f"\n{V.MODELLED_ONLY_NOTE}")


def run(args: argparse.Namespace) -> int:
    if not Path(args.surface_dir).is_dir():
        raise SystemExit(f"--surface-dir {args.surface_dir!r}: not a directory")
    present = _available(args.surface_dir, args.surface_frame)
    if not present:
        raise SystemExit(
            f"no <NAME>.{args.surface_frame}.npy surfaces found in {args.surface_dir!r}"
        )
    if args.fit_source not in present:
        raise SystemExit(
            f"--fit-source {args.fit_source!r} not available in {args.surface_dir!r} "
            f"({args.surface_frame} frame); present: {', '.join(present)}"
        )
    if args.out:
        ensure_writable_output(args.out, "--out")

    context = E.gauge_context(args.corpus)
    objective = S.trigram_objective(S.default_trigram_path(args.corpus))
    pool, pool_label = _load_pool(args)
    print(
        f"fitting pool: {pool_label} ({len(pool)} layouts); computing the gauge frame "
        f"({len(E.LIVE_GAUGES)} live axes, {', '.join(E.INVARIANT_GAUGES)} excluded as "
        f"permutation-invariant)",
        flush=True,
    )
    X = E.gauge_matrix(pool, context, progress_every=max(50, len(pool) // 6))

    surfaces = {
        name: E.load_target_surface(name, args.surface_dir, args.surface_frame)
        for name in (args.sources or present)
    }
    targets = {
        name: E.surface_ms_per_trigram(pool, surface, objective)
        for name, surface in surfaces.items()
    }
    fit_surface = surfaces[args.fit_source]
    weights = E.fit_evidence_weights(
        pool,
        fit_surface,
        context,
        objective,
        pool_label=pool_label,
        X=X,
        y=targets[args.fit_source],
        seed=args.seed,
    )
    _print_weights(weights)

    payload: dict = {
        "weights": weights.to_dict(),
        "pool": pool_label,
        "surface_frame": args.surface_frame,
        "surfaces_present": present,
    }

    if args.layouts:
        specs = [_resolve(s) for s in args.layouts]
        print(f"\n{'layout':<16}{'score':>12}{'extrapolating':>15}  out-of-domain gauges")
        scored = []
        for name, lay in specs:
            if not S.is_c30m(lay):
                print(f"{name:<16}{'N/A':>12}{'':>15}  not a C30M permutation")
                continue
            result = weights.score_layout(lay, context)
            scored.append({"name": name, "layout": lay, **result})
            print(
                f"{name:<16}{result['score']:>12.4f}{str(result['extrapolating']):>15}  "
                f"{', '.join(result['out_of_domain']) or '-'}"
            )
        payload["scored"] = scored

    if args.validate:
        print("\nrunning cross-source validation ...", flush=True)
        competitors = V.competitor_scores(pool)
        reference = {name: values[0] for name, values in V.competitor_scores([V.QWERTY30M]).items()}
        cells = V.cross_source_validation(
            pool,
            surfaces,
            context,
            objective,
            X=X,
            targets=targets,
            competitors=competitors,
            bootstrap=args.bootstrap,
            seed=args.seed,
            progress=True,
        )
        lolo = []
        for cell in cells:
            if not cell.independent:
                continue
            lolo.append(
                V.leave_one_layout_out(
                    pool,
                    surfaces[cell.fit_source],
                    surfaces[cell.test_source],
                    context,
                    objective,
                    X=X,
                    fit_target=targets[cell.fit_source],
                    test_target=targets[cell.test_source],
                    competitors=competitors,
                    folds=args.lolo_folds,
                    seed=args.seed,
                )
            )
            print(f"  lolo: {cell.fit_source} -> {cell.test_source}", flush=True)
        independent = [c for c in cells if c.independent]
        placebo_test = independent[0].test_source if independent else args.fit_source
        placebo = V.noise_placebo(
            pool,
            fit_surface,
            targets[placebo_test],
            context,
            objective,
            X=X,
            fit_target=targets[args.fit_source],
            repeats=args.placebo_repeats,
            seed=args.seed,
        )
        resolution = _paired_resolution(args, pool, objective)
        report = V.ValidationReport(
            corpus=context.corpus_name,
            corpus_sha256=dict(context.identity.get("sha256", {})),
            surface_frame=args.surface_frame,
            n_layouts=len(pool),
            pool_label=pool_label,
            cells=cells,
            lolo=lolo,
            placebo=placebo,
            resolution=resolution,
            direction_proof=V.direction_invariance_proof(),
            limitations=V.structural_limitations(
                resolution["paired_floor_ms_per_trigram"] if resolution else None
            ),
            competitor_orientation=V.orient_scores(competitors, reference),
            weights={args.fit_source: weights},
            source_agreement=V.cross_source_agreement(targets),
        )
        _print_validation(report)
        payload["validation"] = report.to_dict()
    else:
        print(f"\n{V.MODELLED_ONLY_NOTE}")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


def _paired_resolution(args: argparse.Namespace, pool: list[str], objective) -> dict | None:
    """The paired floor from per-seed surfaces, when the directory carries them.

    Only ``COMMUNITY_BASE`` ships per-seed parts, so this is measured on that instrument and
    labelled as such — a floor from one instrument is not transferable to another.
    """
    directory = Path(args.surface_dir)
    per_seed = []
    for seed in (0, 1, 2):
        bigram = directory / f"COMMUNITY_BASE.bigram.seed{seed}.npy"
        conditional = directory / f"COMMUNITY_BASE.conditional.seed{seed}.npy"
        if not (bigram.is_file() and conditional.is_file()):
            return None
        per_seed.append(np.load(bigram)[:, :, None] + np.load(conditional))
    return {
        **V.paired_resolution(pool, per_seed, objective, seed=args.seed),
        "instrument": "COMMUNITY_BASE per-seed (native frame — the per-seed parts "
        "reconstruct .native exactly, max diff 0.0, NOT .standardized)",
    }
