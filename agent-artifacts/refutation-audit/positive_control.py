#!/usr/bin/env python3
"""POSITIVE CONTROL for the reconstruction.

A reconstruction that reports "23 survived / 14 killed" is only trustworthy if
it can ALSO report a wrong number when fed wrong data. Three controls:

  C1  Independence: does my 23-survivor set match the report's independently
      written CONFIRMED list (rank2=6, rank3=5, rank4=10 = 21 + 2 method?) and
      the digest's 19 triages + 23 triage AGENTS?
  C2  Mutation: flip one vote in the raw transcript-derived data and confirm the
      tally MOVES (a harness that cannot see a change cannot see a defect).
  C3  Parser negative control: feed the finding-parser a prompt with the fields
      renamed and confirm it returns None rather than silently yielding a
      finding with all-None fields that would inflate the count.
"""
import json, os, re, sys, copy
from collections import Counter

M = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-map.json"
D = ("/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
     "ultracode-audit-1/child-artifacts/journal-digest.json")
mp = json.load(open(M)); dg = json.load(open(D))

def tally(recs):
    surv = kill = 0
    for r in recs:
        good = [v for v in r["votes"] if v["refuted"] is not None]
        nref = sum(1 for v in good if v["refuted"] is True)
        if len(good) > 0 and nref < 2: surv += 1
        else: kill += 1
    return surv, kill

allr = mp["killed"] + mp["survived"]
s, k = tally(allr)
print("=== C1 · INDEPENDENCE ===")
print(f"my tally from raw transcripts        : {s} survived / {k} killed  (total {s+k})")
print(f"digest triages (one per survivor)    : {len(dg['triages'])}")
print(f"report's headline (independent prose): 23 CONFIRMED / 14 refuted")
print(f"profiles-index headline              : 23 CONFIRMED / 14 refuted")
# triage agents ran once per survivor; count them
print(f"triage AGENTS in transcripts         : 23  (from census.py role-by-prompt)")
print(f"  -> triage agents (23) vs digest triages (19): DISCREPANCY of "
      f"{23-len(dg['triages'])} — digest lost {23-len(dg['triages'])} triage records")

def _flip(rs, want_nref, to, n):
    done = 0
    for r in rs:
        good = [v for v in r["votes"] if v["refuted"] is not None]
        nref = sum(1 for v in good if v["refuted"] is True)
        if nref != want_nref: continue
        for v in r["votes"]:
            if v["refuted"] is (not to):
                v["refuted"] = to; done += 1
                break
        if done >= n: return f"{done} vote(s) on {r['title'][:50]!r}"
    return f"{done} vote(s) (target not found)"

def _kill_panel(rs):
    """Null every vote on a SURVIVOR. The workflow rule is
    `survives = nGoodVotes > 0 && nRefuted < 2`, so a dead panel must flip a
    survivor to killed. Aiming this at an already-killed finding is a NO-OP and
    tells you nothing — that was the first version of this control and it
    reported a false 'HARNESS BLIND'."""
    for r in rs:
        good = [v for v in r["votes"] if v["refuted"] is not None]
        if len(good) > 0 and sum(1 for v in good if v["refuted"] is True) < 2:
            for v in r["votes"]: v["refuted"] = None
            return f"all 3 votes on SURVIVOR {r['title'][:50]!r}"
    return "no survivor found"

print("\n=== C2 · MUTATION (can the harness report a problem?) ===")
for name, mut in [
    ("flip 1 refuting vote -> not-refuted on a 2/3 kill",
     lambda rs: _flip(rs, want_nref=2, to=False, n=1)),
    ("flip 1 non-refuting vote -> refuted on a 1/3 survivor",
     lambda rs: _flip(rs, want_nref=1, to=True, n=1)),
    ("null out ALL votes on a SURVIVOR (dead panel)",
     lambda rs: _kill_panel(rs)),
]:
    rs = copy.deepcopy(allr)
    changed = mut(rs)
    s2, k2 = tally(rs)
    moved = (s2, k2) != (s, k)
    print(f"  {name}")
    print(f"     mutated: {changed}   tally -> {s2}/{k2}   MOVED: {moved} "
          f"{'✅ control passes' if moved else '❌ HARNESS BLIND'}")

print("\n=== C3 · PARSER NEGATIVE CONTROL ===")
sys.path.insert(0, "/tmp/refaudit/agent-artifacts/refutation-audit")
import reconstruct as R
real = ("## THE FINDING UNDER TEST\n\n- **title:** X\n- **file:** a.py\n"
        "- **symbol:** s\n- **verdict claimed:** WRONG\n")
print("  real verify prompt      ->", "parsed" if R.parse_verify_prompt(real) else "None")
print("  header removed          ->", "parsed" if R.parse_verify_prompt(real.replace(
    "## THE FINDING UNDER TEST", "## SOMETHING ELSE")) else "None (✅ refuses)")
print("  empty string            ->", "parsed" if R.parse_verify_prompt("") else "None (✅ refuses)")
bad = R.parse_verify_prompt("## THE FINDING UNDER TEST\n\n(no fields at all)\n")
print("  header but NO fields    ->", "parsed with title=%r (⚠ would inflate count if unguarded)" % (bad or {}).get("title"))


print("\n=== C4 · title-key integrity (the C3 inflation risk, checked on real data) ===")
titles = [r["title"] for r in allr]
print("  findings with title=None :", sum(1 for t in titles if t is None),
      "(any >0 would be a parser artifact inflating the count)")
print("  duplicate titles         :", sum(n-1 for n in Counter(titles).values() if n > 1))
print("  distinct titles          :", len(set(titles)), "of", len(titles))
bad = [r for r in allr if r["file"] is None or r["symbol"] is None]
print("  records missing file/symbol:", len(bad))
