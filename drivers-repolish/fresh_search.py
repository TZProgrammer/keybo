"""A fresh search ON the reported gauge with the SAME polish the field gets — does it reach it?

The campaign's searches optimized the DEFAULT bigram objective and were reported on the ms/char
gauge, two rulers that rank layouts INVERTED (spearman 0.672). So "no search beat the incumbent"
was never a statement about the search's ability — it was a statement about a search graded on a
ruler it never optimized. This driver removes that confound: it searches the gauge itself, and
polishes its own output exactly as `symmetric_field.py` polishes every board, so the comparison
between "what a search finds" and "what the field contains" is on one ruler with one polish.

THE HONEST PRIOR IS THAT IT DOES NOT WIN, and this driver is written to make losing cheap to
report rather than to strain for a win: 0 of 256 attempts beat BALL-1 under the wrong ruler, and
0 of 32 beat arm B under the right one. arm B is additionally a STRICT 3-opt local optimum
(0/20300 improving reorderings), so a search that lands anywhere in its basin converges to
exactly it. Reaching-but-not-beating is therefore the EXPECTED result and is reported as such:
"the search rediscovers the field's best board" is a real finding about the landscape, and is
not dressed up as a win.

Each restart is independent (fresh Layout, own seed) so the restart count is the only budget
knob, and every restart's polished result is recorded — not just the best — because the
distribution of restart outcomes is what says whether the budget was adequate.

Usage:  fresh_search.py [--restarts N] [--seed S] [--alpha A] [--corpus NAME] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boards import CAMPAIGN_FIELD, SEED_FLOOR, assert_own_keybo, gauge  # noqa: E402
from symmetric_field import ms, polish  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.optimize.annealing import SimulatedAnnealing  # noqa: E402


def one_restart(scorer, seed: int, alpha: float, three: bool) -> dict:
    """One independent SA run from a fresh board, then THE polish. Returns its record.

    A fresh ``Layout`` per restart: SA mutates the layout it searches, so reusing one would
    start later restarts from an earlier restart's result and break both independence and the
    determinism the seed is supposed to buy.
    """
    t0 = time.time()
    layout = Layout(SF.C30M, ROW_STAGGERED_30)
    sa = SimulatedAnnealing(seed=seed, alpha=alpha, progress=False)
    best = sa.optimize(layout, scorer)
    raw = "".join(best.chars)
    raw_ms = ms(raw, scorer)
    pol, _ = polish(raw, scorer, three=three)
    return {
        "seed": seed,
        "raw_layout": raw,
        "raw_ms_per_char": raw_ms,
        "polished_layout": pol,
        "polished_ms_per_char": ms(pol, scorer),
        "polish_moved": pol != raw,
        "seconds": time.time() - t0,
    }


def run(restarts: int, seed0: int, alpha: float, corpus: str | None, three: bool) -> dict:
    scorer, surface = gauge(corpus)

    # The field's own polished scores, recomputed here rather than read from the other driver's
    # JSON: the comparison must be against the field as measured by THIS process's objective, so
    # a stale or differently-built JSON cannot silently become the bar.
    field_layouts = {n: polish(lay, scorer, three=three)[0] for n, lay in CAMPAIGN_FIELD.items()}
    field_polished = {n: ms(lay, scorer) for n, lay in field_layouts.items()}
    best_field = min(field_polished, key=lambda n: field_polished[n])
    bar = field_polished[best_field]
    # The bar LAYOUT is the polished one, not the published string it came from. Comparing the
    # search's polished output against an UNPOLISHED incumbent would be the same asymmetry this
    # whole arm exists to remove — and it would silently answer "did the search rediscover the
    # field's best board?" with a guaranteed no whenever that board moved under polish.
    bar_layout = field_layouts[best_field]
    print(
        f"bar: polished field best = {best_field} at {bar:.6f} ms/char "
        f"(polished layout {bar_layout!r})",
        flush=True,
    )

    runs = []
    for i in range(restarts):
        r = one_restart(scorer, seed0 + i, alpha, three)
        r["delta_vs_bar"] = r["polished_ms_per_char"] - bar
        r["beats_bar"] = r["delta_vs_bar"] < 0
        r["matches_bar_layout"] = r["polished_layout"] == bar_layout
        # An SA that returns its own start improved NOTHING: at the shipped alpha=0.999 the
        # annealer never gets cold, so the descent is entirely the polish. Recorded per restart
        # because "the search found nothing and the polish did all the work" and "the search
        # explored and lost" are different findings that produce the same losing number.
        r["sa_improved_nothing"] = r["raw_layout"] == SF.C30M
        runs.append(r)
        print(
            f"  restart {i + 1}/{restarts} seed={r['seed']} raw={r['raw_ms_per_char']:11.6f} "
            f"polished={r['polished_ms_per_char']:11.6f} delta_vs_bar={r['delta_vs_bar']:+.6f} "
            f"{'== bar layout' if r['matches_bar_layout'] else ''} {r['seconds']:.1f}s",
            flush=True,
        )

    champion = min(runs, key=lambda r: r["polished_ms_per_char"])
    n_beat = sum(1 for r in runs if r["beats_bar"])
    n_match = sum(1 for r in runs if r["matches_bar_layout"])
    return {
        "corpus": corpus,
        "three_opt": three,
        "restarts": restarts,
        "seed0": seed0,
        "alpha": alpha,
        "objective": "REPORTED GAUGE (analyze ms/char, K31 T2+Tcond, seed-averaged)",
        "seed_floor": SEED_FLOOR,
        "field_polished": field_polished,
        "bar_board": best_field,
        "bar_layout": bar_layout,
        "bar_ms_per_char": bar,
        "n_sa_improved_nothing": sum(1 for r in runs if r["sa_improved_nothing"]),
        "runs": runs,
        "champion": champion,
        "n_beating_bar": n_beat,
        "n_matching_bar_layout": n_match,
        "champion_delta_vs_bar": champion["polished_ms_per_char"] - bar,
        "champion_beats_bar": champion["polished_ms_per_char"] < bar,
        "champion_within_seed_floor": abs(champion["polished_ms_per_char"] - bar) <= SEED_FLOOR,
    }


def report(r: dict) -> None:
    print(f"\nobjective: {r['objective']}")
    print(f"restarts: {r['restarts']}  alpha={r['alpha']}  three_opt={r['three_opt']}")
    print(f"bar: {r['bar_board']} polished = {r['bar_ms_per_char']:.6f} ms/char")
    c = r["champion"]
    print(
        f"champion: seed={c['seed']} {c['polished_layout']!r} "
        f"{c['polished_ms_per_char']:.6f} ms/char (delta vs bar {r['champion_delta_vs_bar']:+.6f})"
    )
    print(f"restarts beating the bar: {r['n_beating_bar']}/{r['restarts']}")
    print(
        f"restarts landing EXACTLY on the bar layout: {r['n_matching_bar_layout']}/{r['restarts']}"
    )
    print(
        f"restarts whose SA returned its own START (improved nothing; the polish did all the "
        f"descent): {r['n_sa_improved_nothing']}/{r['restarts']}"
    )
    verdict = (
        "BEATS the polished field"
        if r["champion_beats_bar"]
        else (
            "REACHES the polished field (tie inside the seed floor)"
            if r["champion_within_seed_floor"]
            else "does NOT reach the polished field"
        )
    )
    print(f"verdict: the gauge-objective search {verdict}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restarts", type=int, default=8, help="independent SA restarts (>=8 asked)")
    ap.add_argument("--seed", type=int, default=20260801, help="seed of the first restart")
    ap.add_argument("--alpha", type=float, default=0.999, help="cooling rate (CLI default)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--no-three-opt", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    assert_own_keybo()
    r = run(args.restarts, args.seed, args.alpha, args.corpus, three=not args.no_three_opt)
    report(r)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
