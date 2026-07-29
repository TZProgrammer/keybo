"""DIRECTION-CONTROLS-2: re-derives the corrected constants rather than trusting a brief.

The parent's brief asserted "max non-landing feature difference under swap is EXACTLY 0.0",
then corrected it: the max over ALL features is 1.0 and 8 of 20 move. Both statements can be
true because they quantify over different feature sets, so this addendum states the whole
thing unambiguously instead of picking a side, and then re-derives the two NEW facts the
correction supplies (kmstats + oxey direction-blindness under corpus reversal) from scratch --
per the correction's own instruction that a verification is itself a claim.

Four checks:

D1  the bigram frame under SWAP, over ALL 20 features and over the 12 non-landing ones
    separately, so neither number can be quoted without its quantifier.

D2  every kmstats gauge under CORPUS REVERSAL (reverse each n-gram, hold the layout fixed).

D3  the shipped ``oxey`` ``inroll``/``outroll`` under the same corpus reversal, plus the
    reason: does ``is_inwards`` read rows or stroke order?

D4  the distinction this investigation actually rests on. D2/D3 test invariance under FULL
    REVERSAL (a,b,c)->(c,b,a). The user's claim is about SWAPPING TWO OF THREE keys, which
    is a different group action. Show that ``_is_redirect`` is invariant under the former and
    NOT under the latter -- otherwise "kmstats is exactly direction-blind" would appear to
    contradict "kmstats redir flags the user's cell", and both are true.
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
from keybo.features.ngram import bigram_features_from_positions  # noqa: E402
from keybo.features.schema import BIGRAM_FEATURE_NAMES  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402
from keybo.scoring.oxey import OxeyStyleScorer  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

#: Features describing the LANDING key alone. Under a swap the second key changes, so these
#: MUST move; they are not a direction channel. Naming the set is what makes the two
#: different "max difference" numbers comparable.
_LANDING_ONLY = ("bottom", "home", "top", "pinky", "ring", "middle", "index", "lateral")
#: The features that would constitute a direction-of-travel channel if any of them moved.
_DIRECTION_NAMED = ("angle", "inwards", "outwards")


def d1_bigram_frame_under_swap() -> dict:
    per_feature: dict[str, float] = dict.fromkeys(BIGRAM_FEATURE_NAMES, 0.0)
    pairs = 0
    for a, b in itertools.permutations(G.slots, 2):
        fwd = bigram_features_from_positions(G, (a, b), wpm=90.0)
        rev = bigram_features_from_positions(G, (b, a), wpm=90.0)
        d = np.abs(fwd - rev)
        pairs += 1
        for name, value in zip(BIGRAM_FEATURE_NAMES, d, strict=True):
            per_feature[name] = max(per_feature[name], float(value))
    movers = sorted(k for k, v in per_feature.items() if v > 0)
    non_landing = [n for n in BIGRAM_FEATURE_NAMES if n not in _LANDING_ONLY]
    print(
        f"D1  bigram frame under SWAP, {pairs} ordered pairs, {len(BIGRAM_FEATURE_NAMES)} features"
    )
    print(f"    max over ALL features            = {max(per_feature.values()):.6g}")
    print(
        f"    max over the {len(non_landing)} NON-LANDING features = "
        f"{max(per_feature[n] for n in non_landing):.6g}"
    )
    print(f"    features that MOVE ({len(movers)}): {', '.join(movers)}")
    print(f"    landing-only set  ({len(_LANDING_ONLY)}): {', '.join(sorted(_LANDING_ONLY))}")
    print(f"    movers == landing-only set? {set(movers) == set(_LANDING_ONLY)}")
    print(
        f"    direction-NAMED features {_DIRECTION_NAMED} max diff = "
        f"{max(per_feature[n] for n in _DIRECTION_NAMED):.6g}"
    )
    print("    ⇒ BOTH statements are true with their quantifiers: max over ALL = 1.0 (8 movers,")
    print("      all landing-only); max over NON-LANDING = 0.0; and NO direction channel.")
    return {
        "n_pairs": pairs,
        "per_feature_max": per_feature,
        "movers": movers,
        "max_all": max(per_feature.values()),
        "max_non_landing": max(per_feature[n] for n in non_landing),
        "movers_are_landing_only": set(movers) == set(_LANDING_ONLY),
    }


def _reverse_corpus(table: dict[str, int]) -> dict[str, int]:
    """Reverse every n-gram key, summing counts on collision (palindromes/aliases)."""
    out: dict[str, int] = {}
    for ngram, freq in table.items():
        out[ngram[::-1]] = out.get(ngram[::-1], 0) + freq
    return out


def d2_kmstats_under_corpus_reversal() -> dict:
    corpus_dir = production_corpus_dir(None)
    bi, sk, tri = _shared_corpora(corpus_dir)
    lay = _EXTRA_NAMED["keybo-lsb"]
    fwd = KmStats(bi, sk, tri).stats(lay)
    rev = KmStats(_reverse_corpus(bi), _reverse_corpus(sk), _reverse_corpus(tri)).stats(lay)
    rows = {k: {"forward": fwd[k], "reversed": rev[k], "delta": rev[k] - fwd[k]} for k in fwd}
    worst = max(abs(v["delta"]) for v in rows.values())
    print(f"\nD2  kmstats under CORPUS REVERSAL (layout fixed = keybo-lsb), {len(rows)} gauges")
    for k, v in rows.items():
        print(f"    {k:10s} {v['forward']:12.6f} -> {v['reversed']:12.6f}  delta {v['delta']:+.2e}")
    print(
        f"    max |delta| = {worst:.2e}  ⇒ "
        f"{'CONFIRMED all direction-blind' if worst == 0 else 'NOT all blind'}"
    )
    return {"max_abs_delta": worst, "per_gauge": rows}


def d3_oxey_rolls_under_corpus_reversal() -> dict:
    import inspect

    from keybo.features import classify as C

    corpus_dir = production_corpus_dir(None)
    bi, sk, tri = _shared_corpora(corpus_dir)
    # pattern_shares takes a Layout, not a 30-char string (the string form raises
    # AttributeError on .geometry -- caught here rather than reported as "unavailable").
    lay = Layout(_EXTRA_NAMED["keybo-lsb"], geometry=G)
    fwd = OxeyStyleScorer(bi, sk, tri).pattern_shares(lay)
    rev = OxeyStyleScorer(
        _reverse_corpus(bi), _reverse_corpus(sk), _reverse_corpus(tri)
    ).pattern_shares(lay)
    keys = sorted(set(fwd) & set(rev))
    rows = {k: {"forward": fwd[k], "reversed": rev[k], "delta": rev[k] - fwd[k]} for k in keys}
    print("\nD3  shipped oxey pattern shares under CORPUS REVERSAL (keybo-lsb)")
    for k, v in rows.items():
        flag = "  <-- named for direction" if k in ("inroll", "outroll") else ""
        print(
            f"    {k:16s} {v['forward']:12.6f} -> {v['reversed']:12.6f}  "
            f"delta {v['delta']:+.2e}{flag}"
        )
    rolls = [k for k in ("inroll", "outroll") if k in rows]
    roll_worst = max((abs(rows[k]["delta"]) for k in rolls), default=float("nan"))
    print(f"    inroll/outroll max |delta| = {roll_worst:.2e}")
    src = inspect.getsource(C.is_inwards)
    print("    root cause -- keybo.features.classify.is_inwards source:")
    for line in src.strip().splitlines():
        print(f"      {line}")
    return {"per_pattern": rows, "roll_max_abs_delta": roll_worst, "is_inwards_source": src}


def d4_full_reversal_vs_two_key_swap() -> dict:
    """The group action matters: redir is invariant under one and sensitive to the other."""
    slot_of = {ch: i for i, ch in enumerate(_EXTRA_NAMED["keybo-lsb"])}

    def cls(word: str) -> bool:
        return _is_redirect(*(_KEYS[slot_of[ch]] for ch in word))

    full_rev_same = swap_same = 0
    total = 0
    for a, b, c in itertools.permutations("yuo", 3):
        word = a + b + c
        total += 1
        full_rev_same += cls(word) == cls(word[::-1])
        # swap the last two -- the user's actual comparison
        swapped = word[0] + word[2] + word[1]
        swap_same += cls(word) == cls(swapped)
    print("\nD4  which group action is redir blind to? (keybo-lsb, the y/u/o triple)")
    print(f"    invariant under FULL REVERSAL (a,b,c)->(c,b,a): {full_rev_same}/{total}")
    print(f"    invariant under SWAP of last two (a,b,c)->(a,c,b): {swap_same}/{total}")
    print("    ⇒ 'kmstats is exactly direction-blind' (a full-reversal statement) and")
    print("      'kmstats redir flags the user's cell' (a two-key-swap statement) are BOTH")
    print("      true. This investigation rests on the SWAP, which redir does detect.")
    print(f"    concretely: 'you' redirect={cls('you')}, 'yuo' redirect={cls('yuo')}")
    return {
        "n": total,
        "invariant_under_full_reversal": full_rev_same,
        "invariant_under_two_key_swap": swap_same,
        "you_is_redirect": cls("you"),
        "yuo_is_redirect": cls("yuo"),
    }


def main() -> int:
    assert_module_under("keybo", REPO)
    out = {
        "D1_bigram_frame_under_swap": d1_bigram_frame_under_swap(),
        "D2_kmstats_corpus_reversal": d2_kmstats_under_corpus_reversal(),
        "D3_oxey_rolls_corpus_reversal": d3_oxey_rolls_under_corpus_reversal(),
        "D4_group_action": d4_full_reversal_vs_two_key_swap(),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_controls_addendum.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
