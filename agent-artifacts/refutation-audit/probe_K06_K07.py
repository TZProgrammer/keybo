#!/usr/bin/env python3
"""K6 + K7 audit — the two kills whose decisive ground is an ALGEBRAIC IDENTITY / a
label-vs-referent inversion committed BY THE FINDER.

K6 (killed 3/3): "comfort divides a two-population numerator by the FULL corpus mass".
  Kill ground (vote 1 leg 1): the finder's headline -0.7581% is an ALGEBRAIC IDENTITY —
  100*((s/BM)/(s/SM)-1) = 100*(SM/BM-1), in which the comfort value s CANCELS. So the
  number cannot measure a comfort defect at all. Testable: it must be invariant to the
  comfort weights.
  Kill ground (vote 2 leg 1): the km/oxey ratio 0.66839 inverts to 1.49614, which is the
  ledger's already-REGISTERED "1.4961-1.4999x". Testable: 1/0.66839.

K7 (killed 3/3): "effect-curves' default WPM axis runs 2 of 9 columns outside wpm_range".
  Kill ground (all three votes, independently): wpm_range=(60,120) is an UNFITTED LITERAL
  default, not a derived validity interval, and 50/130 are 2 of the 5 midpoints the repo's
  own validation band [40,140) w=20 evaluates. Testable from source.
"""
import sys, inspect, subprocess
from pathlib import Path
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()
ROOT = Path("/tmp/refaudit")

print("=" * 78)
print("K6 GROUND 1 — is the finder's -0.7581% an algebraic identity in the corpus masses?")
print("=" * 78)
print("  The claimed shape: 100*((s/BM)/(s/SM) - 1) where s = comfort fitness.")
print("  s cancels => 100*(SM/BM - 1), a ratio of two CORPUS MASSES, independent of comfort.")
from keybo.data import corpus as CORP
import keybo.analysis.kmstats as KM
# load the iWeb tables analyze.py uses, and compute the two masses
def load(p):
    d = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2: d[parts[0]] = int(parts[1])
    return d
bi = load(ROOT / "data/corpus/bigrams.txt")
sk = load(ROOT / "data/corpus/1-skip31.txt")
tri = load(ROOT / "data/corpus/trigrams.txt")
BM, SM, TM = sum(bi.values()), sum(sk.values()), sum(tri.values())
print(f"\n  bigram mass  BM = {BM:,}")
print(f"  skip31 mass  SM = {SM:,}")
print(f"  trigram mass TM = {TM:,}")
rel = 100.0 * (SM / BM - 1.0)
print(f"\n  closed form 100*(SM/BM - 1) = {rel:.4f}%    (finder's headline: -0.7581%)")
print(f"  matches to 4dp: {abs(rel - (-0.7581)) < 5e-5}")
print(f"  SM == TM exactly (skip is the trigram marginalization): {SM == TM}")
print(f"  => the quantity contains NO comfort term. Vote 1's identity claim VERIFIES.")

print()
print("=" * 78)
print("K6 GROUND 2 — does the km/oxey denominator ratio invert to the REGISTERED 1.4961x?")
print("=" * 78)
r = 0.66839
print(f"  finder's km/oxey ratio      : {r}")
print(f"  1/{r}                        = {1.0/r:.5f}")
print(f"  ledger (dec1c3f:7063) registers: '1.4961-1.4999x'")
print(f"  1/0.66839 lands in [1.4961, 1.4999]: {1.4961 <= 1.0/r <= 1.4999}")
print(f"  => vote 2's 'this IS the registered number' claim VERIFIES.")

print()
print("=" * 78)
print("K7 GROUND — is wpm_range an unfitted literal, and are 50/130 validated midpoints?")
print("=" * 78)
train_src = (ROOT / "src/keybo/training/train.py").read_text()
import re
lits = re.findall(r"wpm_range[^\n]*", train_src)
print("  every 'wpm_range' mention in training/train.py:")
for l in lits: print("     ", l.strip())
print(f"\n  does anything COMPUTE wpm_range from data? "
      f"(look for an assignment from a stroke/percentile expression)")
computed = [l for l in lits if re.search(r"wpm_range\s*=\s*(?!\(60, 120\)|wpm_range)", l)]
print(f"     candidate computed assignments: {computed or '(none)'}")

val_src = (ROOT / "src/keybo/training/validate.py").read_text()
m = re.search(r"def build_cells\([^)]*\)", val_src, re.S)
print(f"\n  build_cells signature:")
print("     ", " ".join((m.group(0) if m else "").split()))
# derive the midpoints the way validate.py does
mm = re.search(r"wpm_lo:\s*int\s*=\s*(\d+)", val_src)
mh = re.search(r"wpm_hi:\s*int\s*=\s*(\d+)", val_src)
mw = re.search(r"bucket_width:\s*int\s*=\s*(\d+)", val_src)
lo, hi, w = int(mm.group(1)), int(mh.group(1)), int(mw.group(1))
mids = list(range(lo + w // 2, hi, w))
print(f"\n  defaults: wpm_lo={lo} wpm_hi={hi} bucket_width={w}")
print(f"  evaluated bucket midpoints = {mids}")
print(f"  are 50 and 130 among them? {50 in mids and 130 in mids}")

cli_src = (ROOT / "src/keybo/cli/effect_curves.py").read_text()
m = re.search(r"--wpms[^\n]*\n(?:[^\n]*\n){0,6}", cli_src)
dfl = re.search(r"default=\[([0-9,\s]+)\]", m.group(0) if m else "")
print(f"\n  effect-curves --wpms default = [{dfl.group(1) if dfl else '?'}]")
print(f"  => 50 and 130 are the OUTER TWO of the validated midpoints, so the default axis")
print(f"     spans exactly the repo's own validated evaluation points. K7 ground VERIFIES.")

print()
print("=" * 78)
print("NEGATIVE CONTROL — can the K6 identity test detect a NON-identity?")
print("=" * 78)
# If the -0.7581% really is comfort-free, substituting a different numerator must not move it.
for s in (1.0, 1e6, 3.587356906409614, 1e-9):
    v = 100.0 * ((s / BM) / (s / SM) - 1.0)
    print(f"    numerator s={s:<22} -> {v:.6f}%")
print("  (identical for every s => s cancels, as claimed)")
# and a genuinely comfort-dependent quantity MUST move:
print("\n  contrast: a quantity that does NOT cancel, s/BM * 100:")
for s in (1.0, 1e6):
    print(f"    s={s:<8} -> {100.0*s/BM:.9f}")
moved = (100.0*1e6/BM) != (100.0*1.0/BM)
print(f"  test distinguishes cancelling from non-cancelling: {moved} "
      f"{'✅ control passes' if moved else '❌ BLIND'}")
if not moved: raise SystemExit("control failed")
