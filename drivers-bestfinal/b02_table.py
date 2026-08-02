"""B02 — the master evidence table, EMITTED not transcribed (CLOSING-1 rule 2).

For every board in the admissible field (the 10 DISTINCT boards per REPOLISH-1) plus the
recovered PRICEBAND-1 frontier boards:
  * ms/char on the reported gauge (shipped TimeSurface.card, C30M, target 90 WPM)
  * the 15 shipped gauges, using `cli/analyze.py`'s EXACT conventions (11 kmstats +
    scissor / imbalance / oxey-style / comfort, incl. comfort's full-corpus denominator)
  * sg_dist -- computed here from `geometry.distance`, labelled MINE, because the shipped
    gauge is on the UNMERGED branch `sgdist-ship` (NORMOPT-1 correction 3)
  * per-finger time share (%) -- the user has asked for this repeatedly
  * fit on all three shipped fitted surfaces (AALTO / COMMUNITY / POOL)
  * distance from its OWN 2-opt optimum on the gauge, plus the Hamming distance the
    polish MOVES (the caveat-#3 column: a shrinking gap is DISTANCE-from-own-optimum,
    not layout equivalence)

Everything through the SHIPPED API so the numbers are commensurable with the ledger's.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "drivers-priceband"))

import numpy as np  # noqa: E402

import keybo  # noqa: E402

WT = os.path.abspath(os.path.join(HERE, ".."))
assert keybo.__file__.startswith(WT), f"WRONG KEYBO: {keybo.__file__} not under {WT}"

from keybo.analysis.kmstats import STAT_NAMES, KmStats  # noqa: E402
from keybo.analysis.surfaces import (  # noqa: E402
    default_trigram_path,
    load_surface,
    score_fit,
    trigram_objective,
)
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import PRODUCTION_SKIPGRAMS, load_frequencies, production_corpus_dir  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.scoring.comfort import ComfortBigramScorer  # noqa: E402
from keybo.scoring.oxey import OxeyStyleScorer  # noqa: E402

from boards import FIELD  # noqa: E402
from fasteval import CHARS, FastSurface  # noqa: E402
from fastsfb import FastGauges  # noqa: E402
from search import IJ, Objective, swap_perms  # noqa: E402

FRONTIER = "/local/home/zegertho/agent/state/bestfinal/artifacts/b01_frontier_boards.json"
OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b02_master_table.json"

SEED_FLOOR = 0.135    # MODEL-seed floor: right for FIXED boards
SEARCH_FLOOR = 0.883  # SEARCH-seed spread: right for search-derived quantities

# REPOLISH-1: 13 published boards -> 10 DISTINCT under symmetric polish.
COLLAPSE = {
    "archive-1843": "archive-trio", "archive-1846": "archive-trio", "lsb-sib": "archive-trio",
    "BALL-1": "armB-ball", "arm-B": "armB-ball",
}
# Prereg §2b: class COMMUNITY boards are judged AS PUBLISHED (polish destroys the asset).
CLASS_COMMUNITY = {"semimak", "graphite", "qwerty30m"}


def sg_dist_mine(lay30, tri):
    """Skip-gram span = corpus-weighted geometry.distance(a, c) over trigrams a?c.

    Denominator = the layout-restricted trigram mass with space allowed in any position
    (the ms/char denominator's convention). Space-involving a/c pairs contribute 0 span
    but DO count in the denominator, matching how `skipgram_span.py` was described.
    """
    g = ROW_STAGGERED_30
    slot = {ch: i for i, ch in enumerate(lay30)}
    num = den = 0.0
    for ng, f in tri.items():
        if len(ng) != 3 or not all(ch == " " or ch in slot for ch in ng):
            continue
        den += f
        a, c = ng[0], ng[2]
        if a != " " and c != " ":
            num += f * g.distance(g.slots[slot[a]], g.slots[slot[c]])
    return num / den if den else float("nan")


def two_opt_profile(obj, fs, lay30):
    """Distance from the board's OWN 2-opt optimum on the gauge + what polish MOVES."""
    p = fs.perm(lay30)
    base = obj.ms(p)
    mss = np.array([obj.ms(q) for q in swap_perms(p)])
    d = mss - base
    cur, pc = base, p
    for _ in range(200):
        Q = swap_perms(pc)
        m = np.array([obj.ms(q) for q in Q])
        k = int(np.argmin(m))
        if m[k] >= cur - 1e-12:
            break
        pc, cur = Q[k], float(m[k])
    slot_to_char = {int(s): CHARS[i] for i, s in enumerate(pc[:30])}
    polished = "".join(slot_to_char[i] for i in range(30))
    return {
        "base_ms": base,
        "n_improving_2opt": int((d < -1e-12).sum()),
        "best_2opt_delta": float(d.min()),
        "at_own_2opt_optimum": bool((d < -1e-12).sum() == 0),
        "polished_ms": cur,
        "polish_gain": base - cur,
        "polish_gain_seed_floors": (base - cur) / SEED_FLOOR,
        "polish_gain_clears_seed_floor": bool(base - cur > SEED_FLOOR),
        "polished_layout": polished,
        "hamming_moved": sum(1 for a, b in zip(lay30, polished) if a != b),
    }


def main():
    fs = FastSurface()
    fg = FastGauges()
    obj = Objective(fs, fg)
    surf = default_surface()

    cdir = production_corpus_dir(None)
    bigrams = load_frequencies(str(cdir / "bigrams.txt"))
    skipgrams = load_frequencies(str(cdir / PRODUCTION_SKIPGRAMS))
    trigrams = load_frequencies(str(cdir / "trigrams.txt"))
    kms = KmStats(bigrams, skipgrams, trigrams)
    oxey = OxeyStyleScorer(bigrams, skipgrams, trigrams)
    comfort = ComfortBigramScorer(bigrams, skipgram_freqs=skipgrams)
    bigram_mass = sum(bigrams.values())

    print("== RECONCILIATION GATE (before any new number) ==")
    for name, pub in (("arm-B", 253.900579), ("BALL-1", 253.966426)):
        m = fs.ms_per_char(FIELD[name])
        c = surf.card(FIELD[name]).ms_per_char
        print(f"  {name:8s} fast {m:.6f} | shipped card {c:.6f} | published {pub:.6f} "
              f"| fast-vs-pub {abs(m - pub):.2e} | fast-vs-card {abs(m - c):.2e}")
        assert abs(m - pub) < 1e-5, name
        assert abs(m - c) < 1e-4, f"{name}: fast evaluator diverges from shipped card"

    boards = dict(FIELD)
    fr = json.load(open(FRONTIER))["frontier"]
    for r in fr.values():
        if not r["matches_field_board"]:
            boards[f"FRONTIER@sfb<={r['cap']:g}"] = r["layout"]

    surfaces = {k: load_surface(f"{k}_TRI_PS_FREQ_PRIOR")
                for k in ("AALTO", "COMMUNITY", "POOL")}
    objtri = trigram_objective(default_trigram_path())

    rows = {}
    for name, lay in sorted(boards.items()):
        layout = Layout(lay, ROW_STAGGERED_30)
        g = dict(kms.stats(lay))
        shares = oxey.pattern_shares(layout)
        g["scissor"] = shares["scissor"]
        g["imbalance"] = shares["imbalance"]
        g["oxey-style"] = oxey.fitness(layout)          # LOWER is better (oxey.py:92)
        g["comfort"] = comfort.fitness(layout) / bigram_mass
        card = surf.card(lay)
        tot = sum(card.per_finger_ms.values())
        rows[name] = {
            "layout": lay,
            "ms_per_char": card.ms_per_char,
            "ms_per_char_fast": fs.ms_per_char(lay),
            "coverage_pct": card.coverage_pct,
            "gauges": {k: float(v) for k, v in g.items()},
            "sg_dist_mine": sg_dist_mine(lay, surf.tri),
            "per_finger_pct": {f: 100.0 * v / tot for f, v in card.per_finger_ms.items()},
            "surfaces": {k: score_fit(lay, s, objtri) for k, s in surfaces.items()},
            "collapse_group": COLLAPSE.get(name),
            "class": "COMMUNITY" if name in CLASS_COMMUNITY else "OWN",
            "twoopt": two_opt_profile(obj, fs, lay),
        }
        print(f"  scored {name:24s} ms={card.ms_per_char:9.4f} sfb={g['sfb']:7.4f} "
              f"2opt_impr={rows[name]['twoopt']['n_improving_2opt']:4d} "
              f"hamming={rows[name]['twoopt']['hamming_moved']:3d}")

    json.dump({"rows": rows, "n_swaps": len(IJ), "gauge_names": sorted(g.keys()),
               "floors": {"model_seed": SEED_FLOOR, "search_seed": SEARCH_FLOOR}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  ({len(rows)} boards, {len(g)} gauges)")


if __name__ == "__main__":
    main()
