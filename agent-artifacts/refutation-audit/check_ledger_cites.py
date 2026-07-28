#!/usr/bin/env python3
"""Check every ledger line number cited by a REFUTING vote against the tree at f4c917a.

The audit's own 23 confirmed findings ALL had stale citations (drift +2..+44) because
PREREGISTRATIONS.md grew 1021 lines mid-audit (8209 -> 9230). A refutation whose ground is
"already registered at line N" is FALSE if line N does not carry that registration.

Method: for each (kill, vote, line) we print the line AT that number in the audit-era tree
(dec1c3f, the base every finder claimed) AND in my base f4c917a, plus a content search for
the claim the vote attached to it. Bare line numbers cannot be trusted either way, so the
verdict comes from the CONTENT SEARCH, not the line.
"""
import json, subprocess, sys, os

ROOT = "/tmp/refaudit"
CLAIMS = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-claims.json"
rows = json.load(open(CLAIMS))

def git_show(sha, path):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:{path}"],
                       capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else None

BASES = {}
for sha in ("dec1c3f", "f4c917a"):
    L = git_show(sha, "PREREGISTRATIONS.md")
    BASES[sha] = L
    print(f"PREREGISTRATIONS.md @ {sha}: {len(L) if L else 'MISSING'} lines")

print()
for x in rows:
    if not x["ledger_lines"]: continue
    print("=" * 100)
    print(f"K{x['kill']} vote{x['vote']} ({x['agent']}) — {x['title'][:80]}")
    for ln in x["ledger_lines"]:
        n = int(ln)
        print(f"  --- cited line {n} ---")
        for sha, L in BASES.items():
            if L is None or n > len(L):
                print(f"      @{sha}: OUT OF RANGE (file is {len(L) if L else 0} lines)")
            else:
                print(f"      @{sha}: {L[n-1][:150]!r}")
