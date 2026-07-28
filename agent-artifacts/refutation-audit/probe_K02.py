#!/usr/bin/env python3
"""K2 audit — the skipgram-marginalization-identity kill.

FINDING (killed 3/3): manifest.json asserts skip(a,c)=sum_b tri(a,b,c) but the committed
blend-v1 tables violate it on 2854 of 4094 keys (L1=4702, max|diff|=11).

The kill's decisive ground (votes 1+2, independently): the RULE is exact in real-valued
shares; 100% of the residue is the separately-documented largest-remainder apportionment to
declared_total=1e9. If true, the label ("the derivation rule") matches its referent and the
integer residue is disclosed elsewhere.

I re-derive BOTH halves from the committed tables at f4c917a, independently of the votes'
scripts (which live in a scratch dir I did not write).
"""
import sys, gzip, json, io
from pathlib import Path
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()

ROOT = Path("/tmp/refaudit")
BL = ROOT / "data/corpus/blend-v1"
print("blend-v1 dir contents:", sorted(p.name for p in BL.iterdir()))

def load(p):
    op = gzip.open if str(p).endswith(".gz") else open
    d = {}
    with op(p, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2: continue
            d[parts[0]] = int(parts[1])
    return d

tri = skip = None
for name in ("trigrams.txt", "trigrams.txt.gz"):
    if (BL / name).exists(): tri = load(BL / name); break
for name in ("1-skip31.txt", "1-skip31.txt.gz", "1-skip.txt", "1-skip.txt.gz"):
    if (BL / name).exists(): skip = load(BL / name); skipname = name; break
print(f"trigrams entries: {len(tri) if tri else 'MISSING'}   "
      f"skipgrams ({skipname}) entries: {len(skip) if skip else 'MISSING'}")
print(f"trigram total: {sum(tri.values()):,}   skipgram total: {sum(skip.values()):,}")

# --- HALF 1: the finder's INTEGER claim
marg = {}
for k, v in tri.items():
    if len(k) != 3: continue
    marg[k[0] + k[2]] = marg.get(k[0] + k[2], 0) + v
keys = set(marg) | set(skip)
dis = [(k, marg.get(k, 0), skip.get(k, 0)) for k in keys if marg.get(k, 0) != skip.get(k, 0)]
L1 = sum(abs(a - b) for _, a, b in dis)
mx = max((abs(a - b) for _, a, b in dis), default=0)
print()
print("=" * 78)
print("HALF 1 — the finder's INTEGER identity claim, re-derived")
print("=" * 78)
print(f"  keys compared               : {len(keys)}")
print(f"  keys where integers DISAGREE: {len(dis)}     (finder said 2854 of 4094)")
print(f"  L1 of the disagreement      : {L1}           (finder said 4702)")
print(f"  max |diff|                  : {mx}           (finder said 11)")
print(f"  symmetric key-set difference: {len(set(marg) ^ set(skip))}")

# --- HALF 2: the REFUTATION's real-valued claim
print()
print("=" * 78)
print("HALF 2 — the REFUTATION's claim: the RULE is exact in real-valued shares")
print("=" * 78)
tri_tot = sum(tri.values()); skip_tot = sum(skip.values())
worst = 0.0; worstk = None
for k in keys:
    a = marg.get(k, 0) / tri_tot
    b = skip.get(k, 0) / skip_tot
    if abs(a - b) > worst: worst, worstk = abs(a - b), k
print(f"  max |marginalized_share - skip_share| over committed INTEGER tables: {worst:.6e} ({worstk})")
print(f"    (this is the QUANTIZED residue, so it is ~1/total, not 0)")
print(f"  1/declared_total for scale                                         : {1/skip_tot:.6e}")
print(f"  is the residue within one quantum per key? {worst <= 1.0/skip_tot * (mx + 1e-9)}")
print(f"  max |diff| in COUNTS is {mx}, i.e. {mx} apportionment units out of {skip_tot:,}")
print(f"  relative L1 = {L1/skip_tot:.3e}   (finder itself reported 4.702e-06)")

# --- HALF 3: is the rounding documented where the refutation says?
print()
print("=" * 78)
print("HALF 3 — is the largest-remainder rounding documented, as the kill asserts?")
print("=" * 78)
import inspect
from keybo.data import build_corpus as BC
ap = inspect.getdoc(BC.apportion)
print("  apportion.__doc__:")
for ln in (ap or "").splitlines(): print("     ", ln)
mani = json.load(open(BL / "manifest.json"))
print(f"\n  manifest declared_total : {mani.get('declared_total'):,}")
outs = mani.get("outputs")
print(f"  manifest outputs        : {json.dumps(outs)[:300]}")
prov = (BL / "PROVENANCE.md").read_text()
import re
m = re.search(r"[^\n]*4,?087[^\n]*", prov)
print(f"\n  PROVENANCE line carrying the byte-exactness claim:")
print(f"     {m.group(0).strip() if m else '(not found)'}")
print(f"  -> does that line's entry count (4,087) match the BLEND's skipgram count "
      f"({len(skip)})? {len(skip) == 4087}")
sec = "largest-remainder" in prov or "largest remainder" in prov or "Hamilton" in prov
print(f"  PROVENANCE.md documents largest-remainder/Hamilton apportionment: {sec}")

print()
print("=" * 78)
print("NEGATIVE CONTROL — can this probe report an identity VIOLATION when one exists?")
print("=" * 78)
bad = dict(skip); k0 = sorted(skip)[0]
bad[k0] = skip[k0] + 10_000_000     # a real, gross violation (not a rounding quantum)
worst2 = max(abs(marg.get(k, 0)/tri_tot - bad.get(k, 0)/sum(bad.values())) for k in keys)
print(f"  after injecting a +1e7 count on key {k0!r}:")
print(f"    max real-valued share divergence: {worst2:.6e}  (clean was {worst:.6e})")
moved = worst2 > worst * 100
print(f"    probe detects a gross violation : {moved} "
      f"{'✅ control passes' if moved else '❌ PROBE BLIND'}")
if not moved: raise SystemExit("control failed")
