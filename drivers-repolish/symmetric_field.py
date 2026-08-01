"""The campaign field AS-PUBLISHED vs after SYMMETRIC polish, on the reported gauge.

This is the measurement DEADCODE-1 and SEARCHPARAMS-1 each named and neither ran. Both defects
are properties of the COMPARISON rather than of any layout:

* the RULER — `optimize`'s default objective is bigram-only and ranks layouts INVERTED to the
  ms/char gauge every published number is quoted in (spearman 0.672), so a board selected by
  the default was selected on a different ruler than it was reported on;
* the POLISH — a searched layout got SA + 2-opt while the incumbents it was reported against
  were scored AS-IS, so the printed gap contained polish the incumbent never received.

Fixing either alone leaves a comparison that is still wrong on the other axis. This driver
fixes both at once: ONE ruler (the reported gauge), and the SAME polish applied to EVERY board
including the campaign's own winners.

What it reports, and the distinction that matters most:

1. as-published ms/char for all 13 boards, and the ordering;
2. symmetrically polished ms/char, and the ordering;
3. every PAIR whose order FLIPS between the two, with its margin;
4. for each flip, whether the margin clears the MODEL-SEED floor (0.135 ms/char).

(4) is what separates "the ordering changed" from "the ordering is RESOLVABLE". A flip whose
margin sits inside the estimator's own noise is not a new ranking — it is evidence that the two
boards are INDISTINGUISHABLE on this gauge, which is a different and more useful claim. The
seed floor is the right floor here because every board is a FIXED INPUT: the only noise between
two of these numbers is the gauge estimator's spread over its three model seeds. The campaign's
other floor (0.883) is the spread of stochastic SEARCH OUTCOMES and would be ~6.5x too loose.

Because the floor is an estimator-spread argument, the driver does not stop at comparing a
margin to 0.135: for every flipped pair it re-evaluates BOTH boards on each of the three
per-seed gauge tables separately and reports whether the flip holds on 3/3. A flip that holds
on every seed is structural even if its mean margin is small; one that does not is noise even
if the margin happens to exceed the floor.

Usage:  symmetric_field.py [--corpus NAME] [--no-three-opt] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boards import (  # noqa: E402
    CAMPAIGN_FIELD,
    SEARCH_SPREAD_FLOOR,
    SEED_FLOOR,
    assert_own_keybo,
    gauge,
)

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.optimize.local_search import three_opt, two_opt  # noqa: E402


def polish(lay30: str, scorer, three: bool = True) -> tuple[str, int]:
    """THE polish, in one place: 2-opt then (optionally) 3-opt. Returns (layout, n_evals-ish).

    2-opt first and 3-opt after — not instead — because both reach the same 3-opt optimum from
    these boards while the cheap moves first is measurably fewer evaluations: every improving
    swap 2-opt takes is one the 3-opt scan would otherwise re-derive over C(n,3) triples.

    ``two_opt``/``three_opt`` MUTATE the layout they are handed, so this builds a fresh
    ``Layout`` per call. Sharing one would let each board's polish start from the previous
    board's result — which would silently make every number after the first one wrong.
    """
    layout = Layout(lay30, ROW_STAGGERED_30)
    layout = two_opt(layout, scorer)
    if three:
        layout = three_opt(layout, scorer)
    return "".join(layout.chars), 0


def ms(lay30: str, scorer) -> float:
    return scorer.ms_per_char(Layout(lay30, ROW_STAGGERED_30))


def order(scores: dict[str, float]) -> list[str]:
    """Board names fastest-first. Ties broken by name so the ordering is deterministic."""
    return [n for n, _ in sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))]


def collapses(layouts: dict[str, str]) -> list[dict]:
    """Groups of boards that polish to the SAME layout — distinctions that vanished entirely.

    This is a separate output from `flips` because it is a different KIND of change and the
    pair-flip test cannot express it. When two boards polish to bit-identical layouts their
    polished margin is exactly 0.0, so `flips` (which compares strict `<` both ways) correctly
    reports no REORDERING — but "these two boards are now the same board" is a much stronger
    statement than any reordering, and reporting nothing would lose it.

    It is also what makes the printed polished ordering honest. `order` breaks ties by NAME, so a
    tie group renders as a plausible-looking alphabetical run (`archive-1843 < archive-1846 <
    lsb-sib`) that a reader will read as a ranking. Naming the groups is what stops that.
    """
    groups: dict[str, list[str]] = {}
    for name, lay in layouts.items():
        groups.setdefault(lay, []).append(name)
    return [
        {"polished_layout": lay, "boards": sorted(names)}
        for lay, names in groups.items()
        if len(names) > 1
    ]


def render_order(scores: dict[str, float], layouts: dict[str, str]) -> str:
    """The ordering as a string, with boards that polished to the SAME layout joined by `=`.

    `<` means "measured faster on this gauge"; `=` means "IS THE SAME BOARD after polish", not
    merely "scored equal". Rendering a tie group as a `<` chain would manufacture a ranking out
    of an alphabetical tiebreak.
    """
    same: dict[str, list[str]] = {}
    for name in order(scores):
        same.setdefault(layouts[name], []).append(name)
    seen: set[str] = set()
    parts = []
    for name in order(scores):
        lay = layouts[name]
        if lay in seen:
            continue
        seen.add(lay)
        parts.append(" = ".join(same[lay]))
    return " < ".join(parts)


def flips(before: dict[str, float], after: dict[str, float]) -> list[dict]:
    """Every unordered pair whose relative order differs between the two score maps.

    Compares PAIRS rather than rank positions: a single board moving several places shifts the
    rank index of every board it passes, which would report as many "changes" for one event.
    A pair flip is the thing that is actually a changed claim ("X beats Y").
    """
    names = sorted(before)
    out = []
    for a_i, a in enumerate(names):
        for b in names[a_i + 1 :]:
            before_a_first = before[a] < before[b]
            after_a_first = after[a] < after[b]
            if before_a_first != after_a_first:
                winner, loser = (a, b) if after_a_first else (b, a)
                out.append(
                    {
                        "pair": [a, b],
                        "published_faster": a if before_a_first else b,
                        "polished_faster": winner,
                        "published_margin": abs(before[a] - before[b]),
                        "polished_margin": abs(after[a] - after[b]),
                        "winner": winner,
                        "loser": loser,
                    }
                )
    return sorted(out, key=lambda f: -f["polished_margin"])


def seed_check(pairs: list[dict], polished: dict[str, str], corpus: str | None) -> None:
    """Annotate each flip with whether it holds on 3/3 per-seed gauge tables.

    The floor is an estimator-spread claim, so the direct test of "is this flip inside the
    noise" is to re-run the comparison on each seed's own ruler rather than only on their mean.
    Mutates the dicts in place (adds `seeds_agreeing` / `per_seed_margins` / `structural`).
    """
    if not pairs:
        return
    from keybo.analysis.timecard import TimeSurface, gauge_scorer_from_surface
    from keybo.data.corpus import load_frequencies, production_corpus_dir

    tri = load_frequencies(str(production_corpus_dir(corpus) / "trigrams.txt"))
    surface = TimeSurface(tri, target_wpm=90.0, keep_seed_tables=True)
    per_seed = [gauge_scorer_from_surface(surface, SF.C30M, table=t) for t in surface.seed_tables()]

    needed = {n for f in pairs for n in f["pair"]}
    by_seed = [{n: ms(polished[n], sc) for n in needed} for sc in per_seed]
    for f in pairs:
        a, b = f["pair"]
        margins = [s[f["winner"]] - s[f["loser"]] for s in by_seed]
        f["per_seed_margins"] = margins
        f["seeds_agreeing"] = sum(1 for m in margins if m < 0)
        f["structural"] = f["seeds_agreeing"] == len(margins)


def run(corpus: str | None, three: bool) -> dict:
    scorer, surface = gauge(corpus)

    published = {n: ms(lay, scorer) for n, lay in CAMPAIGN_FIELD.items()}

    polished_layout: dict[str, str] = {}
    polished: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for name, lay in CAMPAIGN_FIELD.items():
        t0 = time.time()
        new, _ = polish(lay, scorer, three=three)
        polished_layout[name] = new
        polished[name] = ms(new, scorer)
        deltas[name] = polished[name] - published[name]
        print(
            f"  polished {name:13s} {published[name]:11.6f} -> {polished[name]:11.6f} "
            f"({deltas[name]:+.6f}, {100.0 * deltas[name] / published[name]:+.4f}%) "
            f"moved={new != lay}  {time.time() - t0:.1f}s",
            flush=True,
        )

    pairs = flips(published, polished)
    seed_check(pairs, polished_layout, corpus)

    return {
        "corpus": corpus,
        "three_opt": three,
        "charset": SF.C30M,
        "coverage_pct": 100.0 * scorer._covered / max(surface.total_mass, 1),
        "seed_floor": SEED_FLOOR,
        "search_spread_floor_not_used": SEARCH_SPREAD_FLOOR,
        "boards": {
            n: {
                "published_layout": CAMPAIGN_FIELD[n],
                "polished_layout": polished_layout[n],
                "moved": polished_layout[n] != CAMPAIGN_FIELD[n],
                "published_ms_per_char": published[n],
                "polished_ms_per_char": polished[n],
                "delta_ms_per_char": deltas[n],
                "delta_pct": 100.0 * deltas[n] / published[n],
            }
            for n in CAMPAIGN_FIELD
        },
        "order_published": order(published),
        "order_polished": order(polished),
        "order_published_rendered": render_order(published, CAMPAIGN_FIELD),
        "order_polished_rendered": render_order(polished, polished_layout),
        "flips": pairs,
        "n_flips": len(pairs),
        "collapses": collapses(polished_layout),
        "n_distinct_polished_layouts": len(set(polished_layout.values())),
        "n_boards": len(CAMPAIGN_FIELD),
        "flips_structural": [f for f in pairs if f.get("structural")],
        "flips_inside_floor": [f for f in pairs if f["polished_margin"] <= SEED_FLOOR],
        "flips_clearing_floor_and_structural": [
            f for f in pairs if f["polished_margin"] > SEED_FLOOR and f.get("structural")
        ],
    }


def report(r: dict) -> None:
    print(f"\ncorpus={r['corpus']!r} three_opt={r['three_opt']} coverage={r['coverage_pct']:.4f}%")
    print(f"\n{'board':13s} {'published':>12s} {'polished':>12s} {'delta':>11s} {'delta%':>9s} mv")
    for n in r["order_published"]:
        b = r["boards"][n]
        print(
            f"{n:13s} {b['published_ms_per_char']:12.6f} {b['polished_ms_per_char']:12.6f} "
            f"{b['delta_ms_per_char']:+11.6f} {b['delta_pct']:+9.4f} "
            f"{'yes' if b['moved'] else ' no'}"
        )
    print(f"\nordering AS-PUBLISHED (fastest first):\n  {r['order_published_rendered']}")
    print("\nordering POLISHED (fastest first; '=' means the SAME board after polish):")
    print(f"  {r['order_polished_rendered']}")

    if r["collapses"]:
        print(
            f"\n{r['n_distinct_polished_layouts']} DISTINCT layouts remain from "
            f"{r['n_boards']} boards — these groups polished to the SAME board:"
        )
        for c in r["collapses"]:
            print(f"  {' = '.join(c['boards'])}  ->  {c['polished_layout']!r}")

    print(f"\n{r['n_flips']} pair flip(s); seed floor = {r['seed_floor']} ms/char")
    for f in r["flips"]:
        clears = f["polished_margin"] > r["seed_floor"]
        seeds = f.get("seeds_agreeing")
        n_seeds = len(f.get("per_seed_margins") or [])
        print(
            f"  {f['pair'][0]:13s} vs {f['pair'][1]:13s} "
            f"published: {f['published_faster']:13s} (margin {f['published_margin']:.6f}) -> "
            f"polished: {f['polished_faster']:13s} (margin {f['polished_margin']:.6f})  "
            f"{'CLEARS' if clears else 'INSIDE'} floor  seeds {seeds}/{n_seeds}"
        )

    n_real = len(r["flips_clearing_floor_and_structural"])
    print(
        f"\nof {r['n_flips']} flips: {n_real} clear the seed floor AND hold on 3/3 seed tables; "
        f"{len(r['flips_inside_floor'])} sit INSIDE the floor"
    )
    print(
        "  a flip inside the floor is NOT a new ranking — it says the two boards are "
        "INDISTINGUISHABLE on this gauge"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None)
    ap.add_argument(
        "--no-three-opt",
        action="store_true",
        help="polish with 2-opt only (the campaign's own polish depth), for the A/B",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    assert_own_keybo()
    r = run(args.corpus, three=not args.no_three_opt)
    report(r)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
