#!/usr/bin/env python3
"""K9 audit — the iWeb 1-skip.txt "charset-TRUNCATED, not a different pass" kill.

FINDING (killed 3/3, claimed verdict WRONG): iWeb `1-skip.txt` is charset-TRUNCATED
(59 of 64 chars), NOT "a different, unreproducible pass" as every docstring says.

The kill's DECISIVE ground (votes 1 and 2, independently derived): two arithmetic
impossibilities. A charset restriction of a marginalization is POINTWISE <= the full
marginalization and can never invent a key. If 1-skip.txt exceeds it on 90% of keys, and
contains a key the marginalization does not, then 1-skip.txt is NOT a restricted view — so
the shipped docstring is right and the finding's verdict WRONG is itself wrong.

I re-derive this from the committed iWeb tables at f4c917a, with no reference to the votes'
scripts (which are in a scratch dir I did not author).
"""
import sys, gzip
from pathlib import Path
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()

ROOT = Path("/tmp/refaudit")
def find(*names):
    for base in (ROOT / "data/corpus", ROOT / "data/corpus/iweb", ROOT / "data"):
        for n in names:
            for p in (base / n, base / (n + ".gz")):
                if p.exists(): return p
    hits = [p for n in names for p in ROOT.rglob(n)] + \
           [p for n in names for p in ROOT.rglob(n + ".gz")]
    return hits[0] if hits else None

def load(p):
    op = gzip.open if str(p).endswith(".gz") else open
    d = {}
    with op(p, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d

p_tri = find("trigrams.txt"); p_1s = find("1-skip.txt"); p_1s31 = find("1-skip31.txt")
print("resolved paths (must be iWeb, NOT blend-v1):")
for label, p in (("trigrams", p_tri), ("1-skip", p_1s), ("1-skip31", p_1s31)):
    print(f"  {label:10s} {p}")
# The iWeb tables are the top-level data/corpus ones; blend-v1 has its own copies.
assert p_1s and "blend-v1" not in str(p_1s), f"resolved the WRONG tree: {p_1s}"

tri, s1, s31 = load(p_tri), load(p_1s), load(p_1s31)
print(f"\ntrigrams {len(tri)} keys / total {sum(tri.values()):,}")
print(f"1-skip   {len(s1)} keys / total {sum(s1.values()):,}")
print(f"1-skip31 {len(s31)} keys / total {sum(s31.values()):,}")

# full marginalization from the trigram table
marg = {}
for k, v in tri.items():
    if len(k) == 3:
        marg[k[0] + k[2]] = marg.get(k[0] + k[2], 0) + v
print(f"marginalize(tri) -> {len(marg)} keys / total {sum(marg.values()):,}")

# charset of each
cs = lambda d: {c for k in d for c in k}
c1, c31 = cs(s1), cs(s31)
print(f"\n1-skip charset {len(c1)} · 1-skip31 charset {len(c31)} · "
      f"absent from 1-skip: {sorted(c31 - c1)}")

print()
print("=" * 78)
print("IMPOSSIBILITY 1 — a restriction is POINTWISE <=. Does 1-skip EXCEED marg?")
print("=" * 78)
common = set(s1) & set(marg)
exceed = [k for k in common if s1[k] > marg[k]]
below = [k for k in common if s1[k] < marg[k]]
exact = [k for k in common if s1[k] == marg[k]]
print(f"  common keys                    : {len(common)}")
print(f"  1-skip EXCEEDS marg on         : {len(exceed)}  ({100*len(exceed)/len(common):.1f}%)"
      f"   (vote 1 said 3129/3474 = 90.1%)")
print(f"  1-skip BELOW marg on           : {len(below)}")
print(f"  EXACT agreement                : {len(exact)}  ({100*len(exact)/len(common):.2f}%)"
      f"   (vote 1 said 345/3473 = 9.93%)")
print(f"  => a charset restriction can NEVER exceed. Exceedances found: {len(exceed)} "
      f"=> NOT a restricted view: {len(exceed) > 0}")

print()
print("=" * 78)
print("IMPOSSIBILITY 2 — a restriction cannot INVENT a key")
print("=" * 78)
invented = [k for k in s1 if k not in marg]
print(f"  keys in 1-skip absent from marginalize(tri): {len(invented)}")
if invented:
    show = sorted(invented)[:8]
    for k in show:
        instr = all(c in c1 for c in k)
        print(f"     {k!r} count={s1[k]}  (both chars inside 1-skip's own 59-charset: {instr})")
print(f"  vote 1's specific exhibit 'Z<': in 1-skip? {'Z<' in s1}  "
      f"marg['Z<']={marg.get('Z<', 0)}")
print(f"  => a restriction cannot invent a key. Invented keys: {len(invented)} "
      f"=> NOT a restricted view: {len(invented) > 0}")

print()
print("=" * 78)
print("CONTROL — does the SAME test call 1-skip31 a restriction/identity of marg?")
print("  (1-skip31 IS documented as the marginalization, so it MUST come out clean.")
print("   If my test cannot tell the two files apart, it is measuring nothing.)")
print("=" * 78)
common31 = set(s31) & set(marg)
ex31 = [k for k in common31 if s31[k] > marg[k]]
inv31 = [k for k in s31 if k not in marg]
exact31 = [k for k in common31 if s31[k] == marg[k]]
print(f"  1-skip31 vs marg: common={len(common31)} exceed={len(ex31)} "
      f"invented={len(inv31)} EXACT={len(exact31)} ({100*len(exact31)/len(common31):.2f}%)")
sep = (len(ex31) == 0 and len(inv31) == 0) and (len(exceed) > 0 or len(invented) > 0)
print(f"  => the test SEPARATES the two files: {sep} "
      f"{'✅ control passes' if sep else '❌ TEST CANNOT DISCRIMINATE'}")
if not sep:
    raise SystemExit("control failed — the discriminator does not discriminate")

print()
print("=" * 78)
print("VERDICT ON THE K9 KILL")
print("=" * 78)
print(f"  The finding claimed 1-skip.txt is the SAME marginalization, charset-truncated.")
print(f"  Re-derived independently: it exceeds the full marginalization on "
      f"{len(exceed)} of {len(common)} common keys and invents {len(invented)} keys.")
print(f"  Both are impossible for a restricted view. The SAME test scores 1-skip31 clean")
print(f"  ({len(exact31)}/{len(common31)} exact, 0 exceed, 0 invented).")
print(f"  => the shipped docstring ('a different, unreproducible pass') is ACCURATE.")
print(f"  => the KILL VERIFIES. The finder's verdict WRONG was itself wrong.")
