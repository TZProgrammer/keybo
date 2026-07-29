"""DIRECTION-TRIGRAM-1: can the shipped instruments price the ORDER of three keys?

The user's observation: layouts that place y,u,o in physical left-to-right order ``yuo``
make the high-frequency English word ``you`` a same-hand DIRECTION REVERSAL (index->ring->
middle or ring->index->middle), where the physical order ``you`` would make it a monotone
roll. THEORY-1 proved the served BIGRAM feature vector has no direction-of-travel channel.
This probe asks the question one level up, where the feature schema *does* carry a
direction channel (``redirect`` / ``bad_redirect`` in ``features.ngram``).

Method — fix the three PHYSICAL SLOTS a layout devotes to {y,u,o} and enumerate all six
orders in which those slots can be visited. Price each order on every ordered instrument:

  * the shipped measured-keystroke surface (``analysis.timecard.TimeSurface``): the
    campaign objective's per-trigram cost ``T2[a,b] + Tc[a,b,c]``;
  * the three fitted surfaces (``analysis.surfaces``): ``S[a,b,c]``;
  * the community direction predicates (``kmstats._is_redirect`` / ``_is_roll`` /
    ``sr-roll``, and oxeylyzer-1's ``_v1_pattern`` four-class partition).

Everything here reads the SHIPPED tables through the shipped modules. No timing model is
reimplemented. The one thing computed locally is the per-trigram *indexing* into those
tables, which is positive-controlled in ``_positive_control_trigram_indexing`` against
``TimeSurface.card`` on a single-trigram corpus.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.analysis.community import _v1_pattern  # noqa: E402
from keybo.analysis.kmstats import _KEYS, _direction, _is_redirect, _is_roll  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.cli.analyze import _EXTRA_NAMED  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layouts import NAMED_LAYOUTS  # noqa: E402
from keybo.testkit import assert_module_under  # noqa: E402

REGISTRY = {**NAMED_LAYOUTS, **_EXTRA_NAMED}


def _tri_cost(surface: TimeSurface, slots: tuple[int, int, int]) -> float:
    """One ordered trigram's cost on the measured-keystroke surface, in predicted ms.

    Mirrors ``TimeSurface.card``'s per-ngram term exactly: ``T2[a,b] + Tc[a,b,c]``.
    Positive-controlled below.
    """
    a, b, c = slots
    return float(surface._T2[a, b] + surface._Tc[a, b, c])


def _positive_control_trigram_indexing(surface: TimeSurface) -> None:
    """Prove ``_tri_cost`` equals what the SHIPPED card charges for that trigram.

    Build a one-trigram corpus with count 1 and check ``card().total_ms`` equals
    ``_tri_cost``. This is the control that licenses every per-permutation number below;
    it runs BEFORE any of them is used.
    """
    lay = REGISTRY["keybo-lsb"]
    probes = ["you", "yuo", "uoy", "the", "ing"]
    for tri in probes:
        one = TimeSurface({tri: 1}, target_wpm=90.0)
        one._T2, one._Tc = surface._T2, surface._Tc  # same tables, one-trigram corpus
        slot_of = {ch: i for i, ch in enumerate(lay)}
        slots = tuple(slot_of[ch] for ch in tri)
        want = one.card(lay).total_ms
        got = _tri_cost(surface, slots)
        if abs(want - got) > 1e-9:
            raise SystemExit(
                f"POSITIVE CONTROL FAILED for {tri!r}: shipped card={want!r} local={got!r} — "
                f"the local indexing is not the shipped per-trigram charge"
            )
    print(f"  positive control PASSED: _tri_cost == TimeSurface.card total for {probes}")


def _km_class(slots: tuple[int, int, int]) -> str:
    """The kmstats trigram class of an ordered slot triple (shipped predicates)."""
    a, b, c = (_KEYS[s] for s in slots)
    if _is_redirect(a, b, c):
        return "redir"
    if _is_roll(a, b, c):
        return "sr-roll" if a.row == b.row == c.row else "roll"
    if a.hand != b.hand and a.hand == c.hand:
        return "alt"
    return "(none)"


def _oxey_class(slots: tuple[int, int, int]) -> str:
    """oxeylyzer-1's four-class redirect partition, or the label it returns otherwise."""
    return _v1_pattern(*(_KEYS[s].finger for s in slots))


def _dirs(slots: tuple[int, int, int]) -> str:
    """kmstats direction of each constituent bigram: + inward, - outward, 0 none."""
    a, b, c = (_KEYS[s] for s in slots)
    sym = {1: "+", -1: "-", 0: "0"}
    return sym[_direction(a, b)] + sym[_direction(b, c)]


def _finger_path(slots: tuple[int, int, int]) -> str:
    return "->".join(G.finger(G.slots[s][0]).name for s in slots)


def probe_layout(name: str, lay30: str, surface: TimeSurface, surf_arrays: dict) -> dict:
    """Price all six visit-orders of the three slots this layout gives {y,u,o}."""
    slot_of = {ch: i for i, ch in enumerate(lay30)}
    if not all(ch in slot_of for ch in "yuo"):
        return {"name": name, "skipped": "layout lacks one of y/u/o"}
    triple = tuple(slot_of[ch] for ch in "yuo")  # (slot of y, slot of u, slot of o)
    rows = []
    for order in itertools.permutations("yuo"):
        slots = tuple(slot_of[ch] for ch in order)
        row = {
            "word": "".join(order),
            "slots": list(slots),
            "positions": [list(G.slots[s]) for s in slots],
            "fingers": _finger_path(slots),
            "dirs": _dirs(slots),
            "km_class": _km_class(slots),
            "oxey_class": _oxey_class(slots),
            "timecard_ms": _tri_cost(surface, slots),
        }
        if S.is_c30m(lay30):
            # The surfaces' axes are indexed by SLOT, not by character: ``score_fit`` does
            # ``surface[perm[i], ...]`` where ``perm[i]`` is the slot the layout puts C30M's
            # i-th character on -- i.e. exactly ``lay30.index(char)``, which is what our
            # ``slots`` already are. Verified equal for y/u/o on keybo-lsb before relying
            # on it, so no permutation is needed here.
            for sname, arr in surf_arrays.items():
                row[f"surf_{sname}"] = float(arr[slots[0], slots[1], slots[2]])
        rows.append(row)
    return {
        "name": name,
        "layout": lay30,
        "yuo_slots": list(triple),
        "yuo_positions": {ch: list(G.slots[slot_of[ch]]) for ch in "yuo"},
        "yuo_fingers": {ch: G.finger(G.slots[slot_of[ch]][0]).name for ch in "yuo"},
        "same_hand": len({G.hand(G.slots[slot_of[ch]][0]) for ch in "yuo"}) == 1,
        "same_row": len({G.slots[slot_of[ch]][1] for ch in "yuo"}) == 1,
        "orders": rows,
    }


def main() -> int:
    assert_module_under("keybo", REPO)
    print(f"keybo tree: {REPO}")
    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    print(f"corpus: {production_corpus_dir(None)}  ({len(tri)} trigrams)")
    freqs = {"".join(p): tri.get("".join(p), 0) for p in itertools.permutations("yuo")}
    print("corpus frequency of the six orderings:")
    for word, f in sorted(freqs.items(), key=lambda kv: -kv[1]):
        print(f"  {word}  {f:>10,}")

    print("\nbuilding the shipped TimeSurface (loads 6 models)...")
    surface = TimeSurface(tri, target_wpm=90.0)
    _positive_control_trigram_indexing(surface)

    surf_arrays = {}
    for sname in S.surface_names(S.DEFAULT_FAMILY):
        if sname in S.available_surfaces(None):
            surf_arrays[sname] = S.load_surface(sname, None)
    print(f"  fitted surfaces loaded: {list(surf_arrays)}")

    out = {
        "corpus": str(production_corpus_dir(None)),
        "order_frequencies": freqs,
        "layouts": [probe_layout(n, s, surface, surf_arrays) for n, s in sorted(REGISTRY.items())],
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "direction_trigram_probe.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
