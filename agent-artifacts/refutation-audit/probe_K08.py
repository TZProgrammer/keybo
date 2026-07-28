#!/usr/bin/env python3
"""K8 audit — the KNOWN_LAYOUTS['mtgap'] kill (3/3).

FINDING (claimed verdict WRONG): mtgap is not a layout core (25 letters, 2 shift-state
chars); 'mtgap' is unreachable and 25% of mtgap-family boards get the wrong label.

The kill's shared DECISIVE ground, reached independently by all three votes: the finder's
"COUNTING PROOF" assumes a well-formed core is 26 letters + 4 punctuation, hence max
agreement 29 < 30 and 'mtgap' unreachable by pigeonhole. All three votes say the REAL
mtgap-family capture is 24 letters / 6 punctuation, so the premise (and thus the pigeonhole)
is unsound.

That is checkable against the shipped capture zip. I extract the real boards myself.
"""
import sys, zipfile, json, io, csv, re
from pathlib import Path
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import preflight  # noqa
print()
ROOT = Path("/tmp/refaudit")

from keybo.data.community import (KNOWN_LAYOUTS, main30_from_monkeytype, identify_layout,
                                  _VARIANT_THRESHOLD)

mt = KNOWN_LAYOUTS["mtgap"]
print("=" * 78)
print("THE FINDING'S OBSERVATION (uncontested by any vote) — re-verified")
print("=" * 78)
print(f"  KNOWN_LAYOUTS['mtgap'] = {mt!r}")
L = sum(1 for c in mt if c.isalpha())
print(f"  length={len(mt)} letters={L} non-letters={sorted(set(c for c in mt if not c.isalpha()))}")
print(f"  contains 'z': {'z' in mt}   contains '\"': {'\"' in mt}   contains ':': {':' in mt}")
print(f"  unique among the 9 entries in NOT being 26 letters + 4 punct: "
      f"{[n for n,v in KNOWN_LAYOUTS.items() if (sum(1 for c in v if c.isalpha()), len(v)-sum(1 for c in v if c.isalpha())) != (26,4)]}")

print()
print("=" * 78)
print("THE KILL'S PREMISE TEST — are REAL captures 26 letters + 4 punct?")
print("=" * 78)
z = ROOT / "data/community/raw/kiakl_form_responses_20260712.zip"
print(f"  capture artifact: {z}  exists={z.exists()}  size={z.stat().st_size:,} B")
# ⚠ FIRST ATTEMPT WAS WRONG AND ITS CONTROL CAUGHT IT: a regex over raw bytes picked up the
# SHIFTED-repeat half of each monkeytype string, yielding uppercase garbage and 0 registry
# matches. The repo's own path (load_sessions, community.py:192) reads the JSON "layout"
# field. Use that -- the authoritative location, per the campaign's own trap #1.
boards = []
with zipfile.ZipFile(z) as zf:
    for n in zf.namelist():
        if not n.lower().endswith(".json"):
            continue
        try:
            doc = json.loads(zf.read(n).decode("utf-8", "replace"))
        except Exception:
            continue
        sessions = doc if isinstance(doc, list) else [doc]
        for sess in sessions:
            if not isinstance(sess, dict):
                continue
            lay = sess.get("layout", "")
            if not isinstance(lay, str):
                continue
            b = main30_from_monkeytype(lay)
            if b and len(b) == 30 and len(set(b)) == 30:
                boards.append((n, b))
print(f"\n  sessions with a usable 30-key board: {len(boards)}")
uniqb = {}
for n, b in boards:
    uniqb.setdefault(b, identify_layout(b))
print(f"  distinct boards: {len(uniqb)}")
from collections import Counter
print(f"  label census: {dict(Counter(uniqb.values()))}")
print()
for b, lab in sorted(uniqb.items(), key=lambda kv: kv[1]):
    lt = sum(1 for c in b if c.isalpha())
    print(f"    {lab:22s} letters={lt} punct={30-lt}  {b!r}")

fam = {b: lab for b, lab in uniqb.items() if "mtgap" in lab}
print(f"\n  boards labelled mtgap / mtgap-variant: {len(fam)}")
for b, lab in fam.items():
    lt = sum(1 for c in b if c.isalpha())
    agree = sum(1 for x, c in zip(b, mt) if x == c)
    print(f"    {lab:16s} letters={lt} punct={30-lt} "
          f"agreement_with_reference={agree}/30 (_VARIANT_THRESHOLD={_VARIANT_THRESHOLD})")
    print(f"      board = {b!r}")
prem = {sum(1 for c in b if c.isalpha()) for b in fam} if fam else set()
print(f"\n  letter-counts of the real mtgap-family board(s): {sorted(prem) or '(none found)'}")
print(f"  => finder's premise 'a well-formed core = 26 letters + 4 punct' true of it? "
      f"{prem == {26} if prem else 'N/A'}")
print(f"     (all three votes say the real board is 24 letters / 6 punct)")

print()
print("=" * 78)
print("CONTROL — does my extractor recover a KNOWN-GOOD board correctly?")
print("=" * 78)
labels = sorted(set(uniqb.values()))
print(f"  distinct labels my extraction produced: {labels}")
ok = any(l in KNOWN_LAYOUTS for l in labels)
print(f"  at least one EXACT 30/30 registry match recovered: {ok} "
      f"{'✅ extractor works' if ok else '⚠ extractor may be mis-slicing — treat counts as UNCHECKED'}")
