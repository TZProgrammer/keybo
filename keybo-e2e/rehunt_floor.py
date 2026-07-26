"""REHUNT — does a dominator's margin clear the instrument's RESOLUTION FLOOR?

⚠ THE FLOOR BOUNDS ms/char ONLY. GEOMEAN-1 measured **0.7186 ms/char** (max per-seed spread over
the six incumbents, iWeb, 90 WPM). It does NOT bound the 12 ratio gauges — those are in different
units, and quoting it against them is a units error CORPUS-SWAP-1 corrected. So this driver does
exactly one thing: for each confirmed dominator, it measures predicted ms/char for the dominator
and for its incumbent, and asks whether |Δ ms/char| clears the floor. Every gauge margin is
reported in its OWN units with NO floor comparison.

Positive control: the six-layout GEOMEAN-1 table is re-measured here and must reproduce, else the
floor being compared against is not the measured one.

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. Predicted time is a
MODEL output, not observed typing speed — clearing the floor would mean "resolvable by this
instrument", never "faster in practice".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402

#: GEOMEAN-1's measured table (iWeb, 90 WPM) — the positive control, re-measured not trusted.
GEOMEAN1_TABLE = {
    "keybo-lsb": (253.2104, 0.5061),
    "keybo-lsb+lm": (253.2657, 0.5643),
    "lsb-sib": (253.2896, 0.7186),
    "archive-1843": (253.4523, 0.6281),
    "archive-1846": (253.4586, 0.6230),
    "qwerty": (262.4294, 0.9600),
}
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True, help="rehunt-verification.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wpm", type=float, default=90.0)
    args = ap.parse_args()

    from keybo.analysis.timecard import TimeSurface

    # ONE SURFACE PER CORPUS. The trigram weights are what a corpus swap moves, so a cell hunted
    # on blend-v1-no-anchor must be TIMED on blend-v1-no-anchor: timing every cell on iWeb would
    # difference a no-anchor dominator against an iWeb clock and compare it to an iWeb-measured
    # floor. The floor is a per-seed SPREAD, so it is corpus-specific too and is re-measured on
    # each corpus rather than borrowing GEOMEAN-1's iWeb number (which stays the positive control
    # for the iWeb surface only). Corpora are named explicitly, never taken from a bare
    # `data/corpus` (CORPUS-SWAP-1).
    surfaces: dict[str, object] = {}

    def surface_for(corpus: str):
        if corpus not in surfaces:
            tri = CE.load_freq(CE.CORPUS_DIRS[corpus] / "trigrams.txt")
            surfaces[corpus] = TimeSurface(tri, target_wpm=args.wpm, keep_seed_tables=True)
        return surfaces[corpus]

    def measure(lay30: str, corpus: str) -> dict:
        surface = surface_for(corpus)
        totals = surface.seed_totals(lay30)
        card = surface.card(lay30)
        chars = card.total_ms / card.ms_per_char
        per_seed = [t / chars for t in totals]
        return {
            "corpus": corpus,
            "ms_per_char": card.ms_per_char,
            "per_seed_ms_per_char": per_seed,
            "per_seed_spread": float(max(per_seed) - min(per_seed)),
            "per_seed_sd": float(np.std(per_seed, ddof=1)),
        }

    def floor_for(corpus: str) -> float:
        """The resolution floor ON THIS CORPUS: max per-seed spread over the 5 incumbents."""
        return max(measure(lay, corpus)["per_seed_spread"] for lay in CE.INCUMBENTS.values())

    # ---- POSITIVE CONTROL: reproduce GEOMEAN-1's measured table ---------------------------
    control = {}
    for name, (ms_ref, spread_ref) in GEOMEAN1_TABLE.items():
        lay = QWERTY30M if name == "qwerty" else CE.INCUMBENTS[name]
        got = measure(lay, "iweb")
        control[name] = {
            "geomean1_ms_per_char": ms_ref,
            "measured_ms_per_char": got["ms_per_char"],
            "ms_per_char_delta": got["ms_per_char"] - ms_ref,
            "geomean1_per_seed_spread": spread_ref,
            "measured_per_seed_spread": got["per_seed_spread"],
            "reproduces": bool(
                abs(got["ms_per_char"] - ms_ref) < 5e-4
                and abs(got["per_seed_spread"] - spread_ref) < 5e-4
            ),
        }
    n_ok = sum(v["reproduces"] for v in control.values())
    floor_measured = max(v["measured_per_seed_spread"] for v in control.values() if v != "qwerty")
    incumbent_floor = max(
        v["measured_per_seed_spread"] for k, v in control.items() if k != "qwerty"
    )
    print(f"== POSITIVE CONTROL vs GEOMEAN-1's measured table (iWeb, {args.wpm} WPM) ==")
    for name, v in control.items():
        print(
            f"  {name:14s} ms/char {v['measured_ms_per_char']:9.4f} "
            f"(geomean-1 {v['geomean1_ms_per_char']:9.4f}, Δ {v['ms_per_char_delta']:+.2e})  "
            f"spread {v['measured_per_seed_spread']:.4f} vs {v['geomean1_per_seed_spread']:.4f}  "
            f"{'✅' if v['reproduces'] else '❌'}"
        )
    print(f"  reproduces: {n_ok}/{len(control)}")
    print(
        f"  MEASURED resolution floor (max per-seed spread over the 5 incumbents) = "
        f"{incumbent_floor:.4f} ms/char  [GEOMEAN-1: 0.7186]\n"
    )

    # ---- the dominators -------------------------------------------------------------------
    verify = json.loads(Path(args.verify).read_text())
    doms = [r for r in verify["rows"] if r["slow_dominates"]]
    seen: dict[tuple[str, str], dict] = {}
    floors: dict[str, float] = {}
    rows = []
    for r in doms:
        lay = r["layout"]
        corpus = r["corpus"]  # time the cell on the corpus it was HUNTED on
        if (lay, corpus) not in seen:
            seen[(lay, corpus)] = measure(lay, corpus)
        cand = seen[(lay, corpus)]
        if r["target"].startswith("IDEAL"):
            continue  # no single incumbent to difference against
        inc_lay = CE.INCUMBENTS[r["target"]]
        if (inc_lay, corpus) not in seen:
            seen[(inc_lay, corpus)] = measure(inc_lay, corpus)
        inc = seen[(inc_lay, corpus)]
        if corpus not in floors:
            floors[corpus] = floor_for(corpus)
        floor = floors[corpus]
        delta = cand["ms_per_char"] - inc["ms_per_char"]
        rows.append(
            dict(
                corpus=corpus,
                time_model_corpus=corpus,
                arm=r["arm"],
                frame=r["frame"],
                target=r["target"],
                layout=lay,
                cand_ms_per_char=cand["ms_per_char"],
                incumbent_ms_per_char=inc["ms_per_char"],
                delta_ms_per_char=delta,
                faster_than_incumbent=bool(delta < 0),
                abs_delta=abs(delta),
                floor_ms_per_char=floor,
                clears_floor=bool(abs(delta) > floor),
                cand_per_seed_spread=cand["per_seed_spread"],
                incumbent_per_seed_spread=inc["per_seed_spread"],
                # gauge margins stay in their OWN units, with NO floor comparison
                gauge_margins={
                    a: float(r["axes_slow"][a] - r["target_axes_slow"][a])
                    for a in r["per_axis_verdict"]
                },
                axes_won=r["axes_won"],
                wscissor_pct_change=r["wscissor_pct_change"],
            )
        )

    print("\n== per-corpus MEASURED resolution floor (max per-seed spread over the 5 incumbents) ==")
    for corpus, floor in sorted(floors.items()):
        print(f"  {corpus:9s} {CE.CORPUS_LABELS[corpus]:20s} floor = {floor:.4f} ms/char")
    print(
        f"\n== {len(rows)} confirmed dominator/incumbent pairs, predicted ms/char — EACH TIMED ON "
        f"THE CORPUS IT WAS HUNTED ON ==\n   (Δ > 0 means the DOMINATOR is SLOWER than the "
        f"incumbent it dominates)"
    )
    print(
        f"{'corpus':9s} {'arm':4s} {'frame':9s} {'target':14s} {'cand':>9s} {'incumb':>9s} "
        f"{'Δ':>9s} {'floor':>8s} {'|Δ|>floor':>10s} {'faster?':>8s}"
    )
    for row in sorted(rows, key=lambda x: -x["abs_delta"]):
        print(
            f"{row['corpus']:9s} {row['arm']:4s} {row['frame']:9s} {row['target']:14s} "
            f"{row['cand_ms_per_char']:9.4f} {row['incumbent_ms_per_char']:9.4f} "
            f"{row['delta_ms_per_char']:+9.4f} {row['floor_ms_per_char']:8.4f} "
            f"{str(row['clears_floor']):>10s} {str(row['faster_than_incumbent']):>8s}"
        )
    n_clear = sum(r["clears_floor"] for r in rows)
    n_faster = sum(r["faster_than_incumbent"] for r in rows)
    n_slower_resolvably = sum(
        r["clears_floor"] and not r["faster_than_incumbent"] for r in rows
    )
    print(
        f"\n  clears its corpus's floor: {n_clear}/{len(rows)}"
        f"   (largest |Δ| = {max((r['abs_delta'] for r in rows), default=0.0):.4f})"
    )
    print(f"  FASTER than the incumbent they dominate: {n_faster}/{len(rows)}")
    print(
        f"  RESOLVABLY SLOWER (clears the floor in the wrong direction): "
        f"{n_slower_resolvably}/{len(rows)}"
    )

    out = dict(
        wpm=args.wpm,
        time_model_tabling=(
            "ONE surface per corpus: every pair is timed on the corpus its cell was HUNTED on, "
            "because the trigram weights are exactly what a corpus swap moves. Timing a "
            "no-anchor dominator on an iWeb clock (and against an iWeb-measured floor) would be "
            "a cross-corpus difference masquerading as a speed margin."
        ),
        positive_control=control,
        positive_control_reproduces=f"{n_ok}/{len(control)}",
        iweb_measured_floor_ms_per_char=incumbent_floor,
        geomean1_floor_ms_per_char=0.7186,
        per_corpus_measured_floor_ms_per_char=floors,
        floor_scope=(
            "The floor bounds ms/char ONLY. It does NOT bound the 12 ratio gauges — different "
            "units. Gauge margins are reported in their own units with no floor comparison."
        ),
        n_pairs=len(rows),
        n_clearing_floor=n_clear,
        n_faster_than_incumbent=n_faster,
        n_resolvably_slower=n_slower_resolvably,
        sign_note=(
            "Delta > 0 means the DOMINATOR is SLOWER than the incumbent it dominates on predicted "
            "time. A dominator that clears the floor with a positive delta is resolvably WORSE on "
            "predicted time while clearing the whole gauge bar."
        ),
        rows=rows,
        note=(
            "MODELED/gauge only; tau saturated at 1.0, Phase-D cancelled. Predicted time is a "
            "MODEL output, not observed typing speed. Nothing promoted, no adoption claim."
        ),
    )
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
