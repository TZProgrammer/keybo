#!/usr/bin/env python3
"""Extract the CHECKABLE claims out of every refuting vote.

Four classes, per the brief's triage:
  LEDGER   — "already registered at PREREGISTRATIONS.md:NNNN" (a bare line number, and the
             ledger grew 1021 lines during the audit, so every one is suspect)
  TRAPS    — "already TOOLING-TRAPS #NN"
  ARTIFACT — a cited file path (must be `ls`-ed; two audit findings cited a test that doesn't exist)
  NUMBER   — a numeric constant load-bearing for the refutation (the known failure mode)
"""
import json, re, os
from collections import defaultdict

M = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-map.json"
mp = json.load(open(M))

LEDGER = re.compile(r"(?:PREREGISTRATIONS(?:\.md)?|ledger|preregistrations)[^\n]{0,40}?"
                    r"\b(?:l\.?|line|:)\s?(\d{3,5})", re.I)
LEDGER2 = re.compile(r"\bl\.(\d{3,5})")
TRAPS = re.compile(r"(?:TOOLING-)?TRAPS?\s*#?\s*(\d{1,2})|trap\s+#?(\d{1,2})", re.I)
PATH = re.compile(r"(?:/(?:tmp|local)/[\w./\-]+|(?:src|tests|data|docs|agent-artifacts|state|legacy)/[\w./\-]+\.(?:py|json|md|npz|npy|txt|gz))")
NUM = re.compile(r"(?<![\w.])[-+]?\d+\.\d{3,}(?:[eE][-+]?\d+)?|(?<![\w.])\d+\.\d+e[-+]?\d+", re.I)

rows = []
for i, r in enumerate(mp["killed"], 1):
    for j, v in enumerate(r["votes"], 1):
        if v["refuted"] is not True: continue
        blob = "\n".join(str(v.get(k) or "") for k in
                         ("reasoning", "evidence_cmd", "evidence_output", "verdict_correction"))
        led = sorted({m for m in LEDGER.findall(blob)} | {m for m in LEDGER2.findall(blob)}, key=int)
        tr = sorted({(a or b) for a, b in TRAPS.findall(blob)}, key=int)
        paths = sorted(set(PATH.findall(blob)))
        nums = sorted(set(NUM.findall(blob)))
        rows.append({"kill": i, "vote": j, "agent": v["agent"], "title": r["title"],
                     "ledger_lines": led, "traps": tr, "paths": paths, "numbers": nums})

print(f"{'K':>3} {'V':>2}  {'ledger lines':<34} {'traps':<16} paths  nums")
for x in rows:
    print(f"{x['kill']:>3} {x['vote']:>2}  {','.join(x['ledger_lines'])[:33]:<34} "
          f"{','.join('#'+t for t in x['traps'])[:15]:<16} {len(x['paths']):>5}  {len(x['numbers']):>4}")

tot = lambda k: sum(len(x[k]) for x in rows)
print(f"\nrefuting votes: {len(rows)}")
print(f"ledger line citations: {tot('ledger_lines')}  · trap citations: {tot('traps')}"
      f"  · path citations: {tot('paths')}  · numeric constants: {tot('numbers')}")

dest = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-claims.json"
json.dump(rows, open(dest, "w"), indent=1)
print("wrote", dest)

print("\n=== all distinct ledger line numbers cited by a REFUTING vote ===")
alll = sorted({int(l) for x in rows for l in x["ledger_lines"]})
print(alll)
print("\n=== all distinct traps cited ===")
print(sorted({int(t) for x in rows for t in x["traps"]}))
print("\n=== all distinct paths cited (to be ls-ed) ===")
for p in sorted({p for x in rows for p in x["paths"]}): print("  ", p)
