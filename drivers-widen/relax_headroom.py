"""ARM-1 relaxation headroom (counts only, no accuracy output): how many NON-QWERTY pids does
each single participant-filter relaxation add, vs the shipped filter?

Shipped filter (keystrokes.py:304-329): FINGERS=="9-10" AND AVG_WPM_15>=40 AND
KEYBOARD_TYPE in {full,laptop} AND LAYOUT in 4-supported.
We vary ONE axis at a time (and the defensible combo) and count distinct non-qwerty pids that
would newly qualify. This tells us which relaxation is worth a rebuild and lets us justify it.
"""
from __future__ import annotations
import csv, sys
from collections import Counter, defaultdict

META = "/local/home/zegertho/keybo-e2e/dataset/Keystrokes/files/metadata_participants.txt"
csv.field_size_limit(sys.maxsize)
SUPPORTED = {"qwerty","azerty","dvorak","qwertz"}
NONQW = {"azerty","dvorak","qwertz"}

def rows():
    with open(META, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            yield r

def val(r,k): return (r.get(k) or "").strip()
def wpm(r):
    try: return float(val(r,"AVG_WPM_15") or "0")
    except ValueError: return 0.0

# predicates
def p_layout(r): return val(r,"LAYOUT").lower() in SUPPORTED
def p_fingers_910(r): return val(r,"FINGERS")=="9-10"
def p_fingers_710(r): return val(r,"FINGERS") in {"9-10","7-8"}
def p_kb_std(r): return val(r,"KEYBOARD_TYPE").lower() in {"full","laptop"}
def p_wpm40(r): return wpm(r)>=40
def p_wpm0(r): return True  # no wpm floor

FILTERS = {
  "shipped (F9-10, wpm>=40, KB full/laptop)": lambda r: p_fingers_910(r) and p_wpm40(r) and p_kb_std(r),
  "relax wpm floor -> 0 (session-wpm gates)": lambda r: p_fingers_910(r) and p_wpm0(r) and p_kb_std(r),
  "relax wpm floor -> 25":                    lambda r: p_fingers_910(r) and wpm(r)>=25 and p_kb_std(r),
  "relax FINGERS -> 7-10":                    lambda r: p_fingers_710(r) and p_wpm40(r) and p_kb_std(r),
  "relax BOTH (F7-10, wpm>=0)":               lambda r: p_fingers_710(r) and p_wpm0(r) and p_kb_std(r),
  "relax ALL3 (+KB any non-mobile)":          lambda r: p_fingers_710(r) and p_wpm0(r) and val(r,"KEYBOARD_TYPE").lower() in {"full","laptop","small"},
}

# count distinct pids per layout under each filter (layout must be supported for all)
counts = {name: defaultdict(set) for name in FILTERS}
for r in rows():
    if not p_layout(r): continue
    lay = val(r,"LAYOUT").lower()
    pid = val(r,"PARTICIPANT_ID")
    for name, fn in FILTERS.items():
        if fn(r):
            counts[name][lay].add(pid)

print(f"{'filter':44s}{'qwertz':>8s}{'azerty':>8s}{'dvorak':>8s}{'NONQW':>8s}{'qwerty':>9s}")
base = None
for name, fn in FILTERS.items():
    c = counts[name]
    nq = sum(len(c[l]) for l in NONQW)
    row = f"{name:44s}{len(c['qwertz']):8d}{len(c['azerty']):8d}{len(c['dvorak']):8d}{nq:8d}{len(c['qwerty']):9d}"
    print(row)
    if base is None: base = {l:set(c[l]) for l in SUPPORTED}

# For each relaxation show NET-NEW non-qwerty pids vs shipped
ship = counts["shipped (F9-10, wpm>=40, KB full/laptop)"]
print("\n=== NET-NEW non-qwerty pids vs shipped (by relaxation) ===")
for name in FILTERS:
    if name.startswith("shipped"): continue
    c = counts[name]
    for lay in ("dvorak","azerty","qwertz"):
        new = len(c[lay] - ship[lay])
        print(f"  {name:42s} {lay:8s} +{new}")
