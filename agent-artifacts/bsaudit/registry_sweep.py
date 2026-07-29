"""Registry sweep: the cell-keying's information content, and the keybo-lsb vs +lm gap.

Q3 (trap 38): ``bad_scissor_cell`` builds a STRING key from a lossy display form. Two ways it
could lose information: (a) two distinct (finger-pair, dy) classes mapping to one string
[collision], (b) the key silently AGGREGATING something a reader would expect it to keep
[hand]. Both are measured, and (b) is measured against what the code can actually recover.

Q4 (the user's catch): keybo-lsb scores BETTER than keybo-lsb+lm on this gauge. The user said
"I believe our bad scissors, or something, is wrong" and was right — it ranked those two on a
support-boundary artifact. So decompose the gap exactly: which fingers, which cells, which
bigrams, and is the driver at the predicate's boundary or in its interior.

Everything through the SHIPPED accessors (share / by_finger / by_cell) on the shipped corpus.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from keybo.analysis.bad_scissor import (
    FINGER_ORDER,
    BadScissor,
    bad_scissor,
    bad_scissor_cell,
    bad_scissor_finger,
)
from keybo.cli.analyze import _EXTRA_NAMED, _shared_corpora, production_corpus_dir
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.testkit import assert_discriminating, assert_module_under

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)
assert_module_under("keybo.analysis.bad_scissor", ROOT)

REGISTRY = {k: v for k, v in {**NAMED_LAYOUTS, **_EXTRA_NAMED}.items() if len(v) == 30}


def cell_key_information() -> dict:
    """What does the cell key keep, and what does it drop, over all 108 qualifying pairs."""
    full: dict[tuple, set[str]] = {}
    for a, b in itertools.product(sorted(G.slots), repeat=2):
        if not bad_scissor(G, a, b):
            continue
        key = bad_scissor_cell(G, a, b)
        hand = "L" if a[0] < 0 else "R"
        finger = bad_scissor_finger(G, a, b)
        adjacency = "adj" if C.is_adjacent(G, a, b) else "non"
        rows = tuple(sorted((a[1], b[1])))
        # The full identity a reader might think a "cell" carries.
        full.setdefault(key, set()).add(f"{hand}|{finger}|{adjacency}|rows{rows}")
    return {
        "n_cell_keys": len(full),
        "keys_that_merge_more_than_one_full_class": {
            k: sorted(v) for k, v in sorted(full.items()) if len(v) > 1
        },
        "merged_class_counts": {k: len(v) for k, v in sorted(full.items())},
        "hand_is_recoverable_from_cell_key": all(
            len({s.split("|")[0] for s in v}) == 1 for v in full.values()
        ),
        "adjacency_is_recoverable_from_cell_key": all(
            len({s.split("|")[2] for s in v}) == 1 for v in full.values()
        ),
        "rowpair_is_recoverable_from_cell_key": all(
            len({s.split("|")[3] for s in v}) == 1 for v in full.values()
        ),
    }


def main() -> int:
    bigrams, _sk, _tri = _shared_corpora(production_corpus_dir("iweb"))
    scorer = BadScissor(bigrams)

    print("=== Q3: WHAT THE CELL KEY KEEPS AND DROPS (all 108 qualifying pairs) ===")
    info = cell_key_information()
    print(f"  distinct cell keys                      = {info['n_cell_keys']}")
    print(f"  hand recoverable from the key           = {info['hand_is_recoverable_from_cell_key']}")
    print(f"  adjacency recoverable from the key      = {info['adjacency_is_recoverable_from_cell_key']}")
    print(f"  row-pair recoverable from the key       = {info['rowpair_is_recoverable_from_cell_key']}")
    print("  per-key merged (hand|finger|adjacency|rows) classes:")
    for k, n in info["merged_class_counts"].items():
        print(f"    {k:22s} merges {n} full classes")

    # SWEEP all 15, then the two-layout question.
    rows = {}
    for label, lay in sorted(REGISTRY.items()):
        L = Layout(lay, G)
        cells = scorer.by_cell(L)
        rows[label] = {
            "share": scorer.share(L),
            "by_finger": scorer.by_finger(L),
            "by_cell": cells,
            "dy1": sum(v for k, v in cells.items() if k.endswith("dy1")),
            "dy2": sum(v for k, v in cells.items() if k.endswith("dy2")),
        }
        rows[label]["dy1_pct_of_total"] = (
            100.0 * rows[label]["dy1"] / rows[label]["share"] if rows[label]["share"] else 0.0
        )
    assert_discriminating([r["share"] for r in rows.values()], "registry shares")

    print(f"\n=== REGISTRY SWEEP ({len(rows)} layouts), share DESC ===")
    print(f"  {'layout':16s} {'share':>9s} {'dy1':>8s} {'dy2':>8s} {'dy1%':>7s}")
    for label, r in sorted(rows.items(), key=lambda kv: -kv[1]["share"]):
        print(f"  {label:16s} {r['share']:9.5f} {r['dy1']:8.5f} {r['dy2']:8.5f} "
              f"{r['dy1_pct_of_total']:6.2f}%")

    # --- Q4: the user's pair -------------------------------------------------------------
    A, B = "keybo-lsb", "keybo-lsb+lm"
    la, lb = Layout(REGISTRY[A], G), Layout(REGISTRY[B], G)
    sa, sb = rows[A]["share"], rows[B]["share"]
    print(f"\n=== Q4: THE USER'S CATCH — {A} vs {B} ===")
    print(f"  {A:16s} share = {sa:.5f}")
    print(f"  {B:16s} share = {sb:.5f}")
    print(f"  gap (+lm worse by)     = {sb - sa:+.5f}   RATIO {B}/{A} = {sb / sa:.5f}")
    print(f"  qwerty anchor          = {rows['qwerty']['share']:.5f}  "
          f"(ratio qwerty/{A} = {rows['qwerty']['share'] / sa:.4f}, "
          f"qwerty/{B} = {rows['qwerty']['share'] / sb:.4f})")
    print(f"  the two layouts differ in these positions: ", end="")
    diffpos = [i for i, (x, y) in enumerate(zip(REGISTRY[A], REGISTRY[B])) if x != y]
    print(f"{diffpos} chars {[(REGISTRY[A][i], REGISTRY[B][i]) for i in diffpos]}")

    print("\n  per-finger gap (+lm minus lsb):")
    fg = {}
    for f in FINGER_ORDER:
        d = rows[B]["by_finger"][f] - rows[A]["by_finger"][f]
        fg[f] = d
        if abs(d) > 1e-12:
            print(f"    {f:10s} {rows[A]['by_finger'][f]:8.5f} -> "
                  f"{rows[B]['by_finger'][f]:8.5f}  ({d:+.5f})")
    nonzero = [f for f, d in fg.items() if abs(d) > 1e-12]
    print(f"    => fingers that move: {nonzero}")

    print("\n  per-cell gap (+lm minus lsb):")
    cg = {}
    for c in sorted(set(rows[A]["by_cell"]) | set(rows[B]["by_cell"])):
        d = rows[B]["by_cell"].get(c, 0.0) - rows[A]["by_cell"].get(c, 0.0)
        cg[c] = d
        if abs(d) > 1e-12:
            print(f"    {c:22s} {rows[A]['by_cell'].get(c, 0.0):8.5f} -> "
                  f"{rows[B]['by_cell'].get(c, 0.0):8.5f}  ({d:+.5f})")

    # Which BIGRAMS drive it, and are they at the predicate's boundary?
    denom = sum(f for bg, f in bigrams.items()
                if len(bg) == 2 and " " not in bg and all(la.has_key(ch) for ch in bg))
    contrib = {}
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg:
            continue
        if not (all(la.has_key(c) for c in bg) and all(lb.has_key(c) for c in bg)):
            continue
        fa = bad_scissor(G, la.pos(bg[0]), la.pos(bg[1]))
        fb = bad_scissor(G, lb.pos(bg[0]), lb.pos(bg[1]))
        if fa != fb:
            contrib[bg] = {
                "freq": freq, "pct": 100.0 * freq / denom,
                "flagged_in": B if fb else A,
                "cell": bad_scissor_cell(G, lb.pos(bg[0]), lb.pos(bg[1])) if fb
                else bad_scissor_cell(G, la.pos(bg[0]), la.pos(bg[1])),
                "finger": bad_scissor_finger(G, lb.pos(bg[0]), lb.pos(bg[1])) if fb
                else bad_scissor_finger(G, la.pos(bg[0]), la.pos(bg[1])),
                "dy": abs((lb if fb else la).pos(bg[0])[1] - (lb if fb else la).pos(bg[1])[1]),
            }
    top = sorted(contrib.items(), key=lambda kv: -kv[1]["pct"])
    print(f"\n  BIGRAMS whose FLAG STATUS DIFFERS between the two layouts: {len(top)}")
    print(f"  {'bg':4s} {'pct':>9s}  flagged-in       cell                  finger    dy")
    cum = {A: 0.0, B: 0.0}
    for bg, d in top[:25]:
        cum[d["flagged_in"]] += d["pct"]
        print(f"  {bg!r:4s} {d['pct']:9.5f}  {d['flagged_in']:16s} {d['cell']:21s} "
              f"{d['finger']:9s} {d['dy']}")
    tot = {k: sum(d["pct"] for _bg, d in top if d["flagged_in"] == k) for k in (A, B)}
    print(f"  mass flagged only in {A:16s} = {tot[A]:.5f}")
    print(f"  mass flagged only in {B:16s} = {tot[B]:.5f}")
    print(f"  net (should equal the share gap {sb - sa:+.5f}) = {tot[B] - tot[A]:+.5f}")

    dy_of_gap: dict[int, float] = {}
    for _bg, d in top:
        s = +d["pct"] if d["flagged_in"] == B else -d["pct"]
        dy_of_gap[d["dy"]] = dy_of_gap.get(d["dy"], 0.0) + s
    print(f"  the gap by dy: {dy_of_gap}")

    out = {
        "cell_key_information": info,
        "registry": rows,
        "user_pair": {
            "a": A, "b": B, "share_a": sa, "share_b": sb,
            "gap_b_minus_a": sb - sa, "ratio_b_over_a": sb / sa,
            "qwerty_share": rows["qwerty"]["share"],
            "differing_positions": diffpos,
            "per_finger_gap": fg, "per_cell_gap": cg,
            "n_bigrams_with_differing_flag": len(top),
            "mass_only_in_a": tot[A], "mass_only_in_b": tot[B],
            "gap_by_dy": dy_of_gap,
            "differing_bigrams": {bg: d for bg, d in top},
        },
    }
    p = ROOT / "agent-artifacts/bsaudit/registry_sweep.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
