"""lmscissor: decompose the keybo-lsb -> keybo-lsb+lm bad_scissor delta.

Re-derives the parent's numbers from source (trap 20), then decomposes the +0.3628
share delta by dy, by cell, and by individual bigram.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import (  # noqa: E402
    BadScissor,
    bad_scissor,
    bad_scissor_cell,
    bad_scissor_finger,
)
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.layout import Layout  # noqa: E402

LSB = "pyuo,vgdnlhiea.cstrmkj-z'fwbxq"
LSB_LM = "pyuo,vgdnmhiea.cstrlkj-z'fwbxq"

corpus_dir = production_corpus_dir(None)  # blend-v1 = PRODUCTION_DEFAULT
print(f"corpus_dir = {corpus_dir}")
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"bigram table entries = {len(bigrams)}, total mass = {sum(bigrams.values())}")

# --- 0. confirm the two layouts are a pure l<->m swap in the same two slots ------------
assert len(LSB) == len(LSB_LM) == 30
diff = [i for i in range(30) if LSB[i] != LSB_LM[i]]
print(f"\n=== differing slots: {diff} ===")
for i in diff:
    print(
        f"  slot {i}: lsb={LSB[i]!r} -> +lm={LSB_LM[i]!r}   "
        f"pos={ROW_STAGGERED_30.slots[i]}  finger={ROW_STAGGERED_30.finger(ROW_STAGGERED_30.slots[i][0]).value}"
    )
assert sorted(LSB) == sorted(LSB_LM), "not a permutation of each other"

lay_a = Layout(LSB, ROW_STAGGERED_30)
lay_b = Layout(LSB_LM, ROW_STAGGERED_30)
bs = BadScissor(bigrams)

# --- 1. re-derive the headline share --------------------------------------------------
sa, sb = bs.share(lay_a), bs.share(lay_b)
print(f"\n=== share (space-EXCLUDED denominator, the production convention) ===")
print(f"  keybo-lsb    {sa:.4f}")
print(f"  keybo-lsb+lm {sb:.4f}")
print(f"  delta        {sb - sa:+.4f}")

sa_inc, sb_inc = bs.share(lay_a, exclude_space=False), bs.share(lay_b, exclude_space=False)
print(f"\n=== share (space-INCLUDED / oxey denominator — WRONG for this gauge, trap 9) ===")
print(f"  keybo-lsb    {sa_inc:.4f}   ratio {sa/sa_inc:.4f}")
print(f"  keybo-lsb+lm {sb_inc:.4f}   ratio {sb/sb_inc:.4f}")
print(f"  delta        {sb_inc - sa_inc:+.4f}")

# --- 2. by finger --------------------------------------------------------------------
fa, fb = bs.by_finger(lay_a), bs.by_finger(lay_b)
print("\n=== by_finger (share pp) ===")
for k in fa:
    print(f"  {k:9s}  {fa[k]:8.4f} -> {fb[k]:8.4f}   {fb[k]-fa[k]:+8.4f}")

# --- 3. by cell (finger-pair x dy) ---------------------------------------------------
ca, cb = bs.by_cell(lay_a), bs.by_cell(lay_b)
keys = sorted(set(ca) | set(cb))
print("\n=== by_cell (share pp) ===")
for k in keys:
    va, vb = ca.get(k, 0.0), cb.get(k, 0.0)
    flag = "  <<<" if abs(vb - va) > 1e-9 else ""
    print(f"  {k:22s}  {va:8.4f} -> {vb:8.4f}   {vb-va:+8.4f}{flag}")

# --- 4. by dy subtotal ----------------------------------------------------------------
print("\n=== by dy subtotal ===")
for dy in ("dy1", "dy2"):
    ta = sum(v for k, v in ca.items() if k.endswith(dy))
    tb = sum(v for k, v in cb.items() if k.endswith(dy))
    print(f"  {dy}:  {ta:8.4f} -> {tb:8.4f}   {tb-ta:+8.4f}")

# --- 5. bigram-level attribution ------------------------------------------------------
geom = ROW_STAGGERED_30


def flagged(layout: Layout) -> dict[str, int]:
    out = {}
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg:
            continue
        if not all(layout.has_key(c) for c in bg):
            continue
        a, b = layout.pos(bg[0]), layout.pos(bg[1])
        if bad_scissor(geom, a, b):
            out[bg] = freq
    return out


def denom(layout: Layout) -> int:
    return sum(
        f
        for bg, f in bigrams.items()
        if len(bg) == 2 and " " not in bg and all(layout.has_key(c) for c in bg)
    )


fa_bg, fb_bg = flagged(lay_a), flagged(lay_b)
da, db = denom(lay_a), denom(lay_b)
print(f"\n=== denominators: lsb {da}  +lm {db}  (identical? {da == db}) ===")
print(f"  numerators:   lsb {sum(fa_bg.values())}  +lm {sum(fb_bg.values())}")

added = {bg: f for bg, f in fb_bg.items() if bg not in fa_bg}
removed = {bg: f for bg, f in fa_bg.items() if bg not in fb_bg}
print(f"\n=== bigrams NEWLY flagged in +lm ({len(added)}), by mass ===")
tot_add = 0
for bg, f in sorted(added.items(), key=lambda kv: -kv[1]):
    a, b = lay_b.pos(bg[0]), lay_b.pos(bg[1])
    pp = 100.0 * f / db
    tot_add += pp
    print(
        f"  {bg!r:8s} freq={f:12d}  {pp:7.4f}pp  cell={bad_scissor_cell(geom,a,b):22s} "
        f"finger={bad_scissor_finger(geom,a,b):8s} posA={a} posB={b}"
    )
print(f"  --> total added {tot_add:.4f}pp")

print(f"\n=== bigrams NO LONGER flagged in +lm ({len(removed)}), by mass ===")
tot_rm = 0
for bg, f in sorted(removed.items(), key=lambda kv: -kv[1]):
    a, b = lay_a.pos(bg[0]), lay_a.pos(bg[1])
    pp = 100.0 * f / da
    tot_rm += pp
    print(
        f"  {bg!r:8s} freq={f:12d}  {pp:7.4f}pp  cell={bad_scissor_cell(geom,a,b):22s} "
        f"finger={bad_scissor_finger(geom,a,b):8s} posA={a} posB={b}"
    )
print(f"  --> total removed {tot_rm:.4f}pp")
print(f"\n  net = +{tot_add:.4f} - {tot_rm:.4f} = {tot_add - tot_rm:+.4f}pp")

# --- 6. bottom-key census (BADSCISSOR-1's c/x tail) ------------------------------------
print("\n=== bottom-key census of the flagged mass ===")
for label, lay, fl, dn in (("keybo-lsb", lay_a, fa_bg, da), ("keybo-lsb+lm", lay_b, fb_bg, db)):
    by_bottom: dict[str, int] = {}
    for bg, f in fl.items():
        pa, pb = lay.pos(bg[0]), lay.pos(bg[1])
        bottom = bg[0] if pa[1] < pb[1] else bg[1]
        by_bottom[bottom] = by_bottom.get(bottom, 0) + f
    tot = sum(by_bottom.values())
    print(f"  {label}:  total numerator {tot}")
    for ch, f in sorted(by_bottom.items(), key=lambda kv: -kv[1]):
        print(f"     bottom={ch!r}  {100.0*f/tot:6.2f}% of flagged   {100.0*f/dn:7.4f}pp")

# --- 7. added-mass bottom-key share ---------------------------------------------------
add_bottom: dict[str, int] = {}
for bg, f in added.items():
    pa, pb = lay_b.pos(bg[0]), lay_b.pos(bg[1])
    bottom = bg[0] if pa[1] < pb[1] else bg[1]
    add_bottom[bottom] = add_bottom.get(bottom, 0) + f
print("\n=== bottom-key census of the ADDED mass ===")
tot = sum(add_bottom.values()) or 1
for ch, f in sorted(add_bottom.items(), key=lambda kv: -kv[1]):
    print(f"  bottom={ch!r}  {100.0*f/tot:6.2f}% of added   {100.0*f/db:7.4f}pp")

# --- 8. incumbent narrow is_scissor for comparison ------------------------------------
print("\n=== incumbent narrow is_scissor, same denominator ===")
na = bs.share_of(lay_a, C.is_scissor)
nb = bs.share_of(lay_b, C.is_scissor)
print(f"  keybo-lsb {na:.4f} -> keybo-lsb+lm {nb:.4f}  delta {nb-na:+.4f}")


def narrow_flagged(layout: Layout) -> dict[str, int]:
    out = {}
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg:
            continue
        if not all(layout.has_key(c) for c in bg):
            continue
        if C.is_scissor(geom, layout.pos(bg[0]), layout.pos(bg[1])):
            out[bg] = freq
    return out


nfa, nfb = narrow_flagged(lay_a), narrow_flagged(lay_b)
n_add = {bg: f for bg, f in nfb.items() if bg not in nfa}
n_rm = {bg: f for bg, f in nfa.items() if bg not in nfb}
print(f"  narrow added:   {sorted(n_add.items(), key=lambda kv: -kv[1])}")
print(f"  narrow removed: {sorted(n_rm.items(), key=lambda kv: -kv[1])}")

# --- 9. l vs m corpus frequency -------------------------------------------------------
print("\n=== unigram-ish mass of l vs m from the bigram table ===")
for ch in "lm":
    mass = sum(f for bg, f in bigrams.items() if len(bg) == 2 and ch in bg)
    print(f"  {ch!r}: bigram mass touching it = {mass}")

json.dump(
    {
        "share": {"keybo-lsb": sa, "keybo-lsb+lm": sb, "delta": sb - sa},
        "by_finger": {"keybo-lsb": fa, "keybo-lsb+lm": fb},
        "by_cell": {"keybo-lsb": ca, "keybo-lsb+lm": cb},
        "added_bigrams": added,
        "removed_bigrams": removed,
        "denominator": {"keybo-lsb": da, "keybo-lsb+lm": db},
        "narrow_is_scissor": {"keybo-lsb": na, "keybo-lsb+lm": nb},
    },
    open("/tmp/lmscissor_probe1.json", "w"),
    indent=2,
)
print("\nwrote /tmp/lmscissor_probe1.json")
