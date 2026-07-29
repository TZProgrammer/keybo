"""DIRECTION-CONTROLS-1: the checks that license the other three probes' numbers.

Four controls, each aimed at a specific way this investigation could be wrong:

C1  **Is my trigram classifier the shipped ``redir`` gauge?** DIRECTION-FIELD-1 reports a
    redirect share over a top-N slice using a locally-assembled ``tri_class``. Control:
    run that same classifier over the FULL corpus with the shipped denominator and compare
    to ``KmStats.stats()['redir']`` for every layout. Any disagreement invalidates the
    field sweep.

C2  **Is the served BIGRAM frame still direction-blind?** (THEORY-1 confirmation on current
    code.) Enumerate all ordered slot pairs and compare each bigram's feature vector to its
    reverse, excluding the features that describe the LANDING key alone. THEORY-1 registered
    a max difference of exactly 0.0; re-derive it rather than cite it.

C3  **Does the TRIGRAM frame carry a channel the bigram frame lacks?** The complement of
    C2: show a swap of two of three keys moves the trigram feature vector, and name which
    columns move. Without this, "the trigram model sees direction" is an assertion.

C4  **Could the 2026-07-05 roll-additivity probe's strata contain the user's cell?** That
    probe read ONLY qwerty rows (``parts[0] != "qwerty": continue``) and bucketed same-hand
    runs by ``|column|`` monotonicity. So its ``run-redirect`` stratum can only contain
    finger paths qwerty's geometry actually produces. Enumerate them and ask whether a
    three-DISTINCT-finger same-row reversal -- the user's cell -- is among them. This is a
    scope check on a registered null, done from the probe's own source.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

from keybo.analysis.kmstats import _KEYS, KmStats, _is_redirect  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED, _shared_corpora  # noqa: E402
from keybo.data.corpus import production_corpus_dir  # noqa: E402
from keybo.features.ngram import (  # noqa: E402
    bigram_features_from_positions,
    trigram_features_from_positions,
)
from keybo.features.schema import BIGRAM_FEATURE_NAMES, TRIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

CANDIDATES = {
    "armB": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "MID": "flmpg.yuo,sntcdireahkxbwv'-jzq",
    "armH-hdln": "flmpg-,uoysntcdireahkxvwb.'jzq",
}

#: Bigram features that describe the LANDING key alone, so they are EXPECTED to differ
#: under a swap (the second key changes). THEORY-1's claim is about everything else.
_LANDING_ONLY = {"bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral"}


def c1_classifier_is_the_shipped_gauge() -> dict:
    corpus_dir = production_corpus_dir(None)
    bi, sk, tri = _shared_corpora(corpus_dir)
    km = KmStats(bi, sk, tri)
    registry = {**NAMED_LAYOUTS, **_EXTRA_NAMED, **CANDIDATES}
    worst = 0.0
    rows = {}
    for name, lay in sorted(registry.items()):
        slot_of = {ch: i for i, ch in enumerate(lay)}
        hit = total = 0
        for ngram, freq in km.tri.items():
            keys = [slot_of.get(ch) for ch in ngram]
            if any(k is None for k in keys):
                continue
            total += freq
            a, b, c = (_KEYS[k] for k in keys)
            if _is_redirect(a, b, c):
                hit += freq
        mine = 100.0 * hit / total if total else 0.0
        shipped = km.stats(lay)["redir"]
        rows[name] = {"mine": mine, "shipped": shipped, "abs_diff": abs(mine - shipped)}
        worst = max(worst, abs(mine - shipped))
    print(
        f"C1  local redir vs shipped KmStats redir, {len(rows)} layouts, max |diff| = {worst:.3e}"
    )
    if worst > 1e-9:
        bad = {k: v for k, v in rows.items() if v["abs_diff"] > 1e-9}
        raise SystemExit(f"C1 FAILED — classifier is not the shipped gauge: {bad}")
    print("    PASSED: the field sweep's predicate IS kmstats' redir predicate")
    return {"max_abs_diff": worst, "per_layout": rows}


def c2_bigram_frame_is_direction_blind() -> dict:
    slots = list(G.slots)
    idx = [i for i, n in enumerate(BIGRAM_FEATURE_NAMES) if n not in _LANDING_ONLY]
    names = [BIGRAM_FEATURE_NAMES[i] for i in idx]
    worst, worst_at, n = 0.0, None, 0
    per_feature = dict.fromkeys(names, 0.0)
    for a, b in itertools.permutations(slots, 2):
        fwd = bigram_features_from_positions(G, (a, b), wpm=90.0)[idx]
        rev = bigram_features_from_positions(G, (b, a), wpm=90.0)[idx]
        d = np.abs(fwd - rev)
        n += 1
        for name, value in zip(names, d, strict=True):
            per_feature[name] = max(per_feature[name], float(value))
        if d.max() > worst:
            worst, worst_at = float(d.max()), (a, b)
    moved = {k: v for k, v in per_feature.items() if v > 0}
    print(
        f"C2  {n} ordered slot pairs; max non-landing feature |difference| under "
        f"reversal = {worst:.6g}"
    )
    print(f"    features checked ({len(names)}): {', '.join(names)}")
    print(f"    features that MOVE under reversal: {moved or 'NONE'}")
    if worst == 0.0:
        print("    CONFIRMED: the served bigram frame has no direction-of-travel channel")
    else:
        print(f"    ⚠ THEORY-1 DOES NOT REPRODUCE: worst at {worst_at}")
    return {
        "n_pairs": n,
        "max_abs_diff": worst,
        "per_feature_max": per_feature,
        "features_checked": names,
    }


def c3_trigram_frame_has_a_direction_channel() -> dict:
    """Does swapping two of three keys move the trigram vector, and which columns?"""
    # The user's exact cell: the three same-row keys ring/middle/index on the left hand.
    ring, middle, index_ = (-4, 3), (-3, 3), (-2, 3)
    roll = (ring, middle, index_)  # monotone inward: the word `you` on flagship-c3
    redirect = (ring, index_, middle)  # reversal:        the word `you` on keybo-lsb
    a = trigram_features_from_positions(G, roll, wpm=90.0)
    b = trigram_features_from_positions(G, redirect, wpm=90.0)
    moved = {
        name: (float(x), float(y))
        for name, x, y in zip(TRIGRAM_FEATURE_NAMES, a, b, strict=True)
        if x != y
    }
    print(
        f"C3  ring->middle->index (roll) vs ring->index->middle (redirect): "
        f"{len(moved)} of {len(TRIGRAM_FEATURE_NAMES)} trigram features differ"
    )
    for name, (x, y) in moved.items():
        print(f"      {name:24s} {x:>9.4f} -> {y:>9.4f}")
    return {"n_features": len(TRIGRAM_FEATURE_NAMES), "moved": moved}


def c4_roll_additivity_strata_coverage() -> dict:
    """Which finger paths qwerty's geometry can put in the probe's ``run-redirect`` bucket.

    Reimplements the 2026-07-05 probe's OWN bucketing rule (roll_error_probe.py) on the
    qwerty layout it was restricted to. This says what the stratum COULD contain; it does
    not claim what the data actually sampled (that table is not in this tree).
    """
    qwerty = NAMED_LAYOUTS["qwerty"]
    paths: dict[str, int] = {}
    same_row_distinct: dict[str, int] = {}
    for triple in itertools.product(range(30), repeat=3):
        a, b, c = (G.slots[s] for s in triple)
        if not (G.hand(a[0]) == G.hand(b[0]) == G.hand(c[0]) != 0):
            continue
        going1, going2 = abs(b[0]) - abs(a[0]), abs(c[0]) - abs(b[0])
        if going1 == 0 or going2 == 0:
            continue  # the probe's "run-flat" bucket
        if (going1 > 0) == (going2 > 0):
            continue  # "run-continue"
        fingers = tuple(G.finger(p[0]).name for p in (a, b, c))
        key = "->".join(f[1] for f in fingers)  # P/R/M/I, hand-agnostic
        paths[key] = paths.get(key, 0) + 1
        if a[1] == b[1] == c[1] and len(set(fingers)) == 3:
            same_row_distinct[key] = same_row_distinct.get(key, 0) + 1
    # the user's cell, hand-agnostic: three distinct fingers, same row, direction reverses
    users_cell = sorted(same_row_distinct)
    print(
        f"C4  qwerty run-redirect stratum admits {len(paths)} distinct finger paths "
        f"over {sum(paths.values())} slot triples"
    )
    print(
        f"    of those, SAME-ROW with three DISTINCT fingers (the user's cell): "
        f"{sum(same_row_distinct.values())} triples, paths {users_cell}"
    )
    # is the user's specific R->I->M / equivalent present?
    print(f"    is 'R->I->M' present? {'YES' if 'R->I->M' in same_row_distinct else 'NO'}")
    print(f"    all admitted paths: {', '.join(sorted(paths))}")
    return {
        "n_paths": len(paths),
        "paths": paths,
        "same_row_three_distinct_fingers": same_row_distinct,
        "qwerty": qwerty,
    }


def main() -> int:
    assert_module_under("keybo", REPO)
    out = {
        "C1_classifier_is_shipped_gauge": c1_classifier_is_the_shipped_gauge(),
        "C2_bigram_direction_blind": c2_bigram_frame_is_direction_blind(),
        "C3_trigram_direction_channel": c3_trigram_frame_has_a_direction_channel(),
        "C4_roll_additivity_strata": c4_roll_additivity_strata_coverage(),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_controls.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
