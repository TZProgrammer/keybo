#!/usr/bin/env python3
"""Which verdict did journal-digest.json drop? (111 reconstructed vs 110 in digest)"""
import json, os, glob
D = ("/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
     "ultracode-audit-1/child-artifacts/journal-digest.json")
M = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-map.json"
dg = json.load(open(D)); mp = json.load(open(M))

def sig(v):
    # reasoning is the most distinctive free-text field
    return (v.get("reasoning") or "")[:200]

dig = {}
for v in dg["verdicts"]:
    dig.setdefault(sig(v), []).append(v)

rec = []
for r in mp["killed"] + mp["survived"]:
    for v in r["votes"]:
        rec.append((r["title"], v))

print("reconstructed verdicts:", len(rec), " digest verdicts:", len(dg["verdicts"]))
missing = []
seen = {k: len(v) for k, v in dig.items()}
for title, v in rec:
    s = sig(v)
    if seen.get(s, 0) > 0:
        seen[s] -= 1
    else:
        missing.append((title, v))
print("\nreconstructed verdicts NOT found in digest:", len(missing))
for title, v in missing:
    print("  finding:", title[:100])
    print("  refuted =", v["refuted"], " lens_applicable =", v["lens_applicable"], " agent", v["agent"])
    print("  reasoning[:400]:", (v["reasoning"] or "")[:400].replace("\n", " "))
leftover = {k: n for k, n in seen.items() if n > 0}
print("\ndigest verdicts NOT matched by reconstruction:", sum(leftover.values()))
for k, n in leftover.items():
    print("  x%d %s" % (n, k[:180].replace("\n", " ")))

# refuted tallies
import collections
print("\nrefuted tally reconstructed:", collections.Counter(v["refuted"] for _, v in rec))
print("refuted tally digest       :", collections.Counter(v["refuted"] for v in dg["verdicts"]))
