#!/usr/bin/env python3
"""Logical-agent census: map each workflow cache key to its successful agent's role.
Distinguishes logical agents (159 cache keys) from files on disk (168, incl. retries).
"""
import json, os, glob, re
from collections import Counter, defaultdict

W = os.path.expanduser(
    "~/.claude/projects/-local-home-zegertho-agent-workspaces-ultracode-audit/"
    "43318f15-a9ab-42db-94c7-199bf3619621/subagents/workflows/wf_32ff2687-938")
J = os.path.join(W, "journal.jsonl")

recs = [json.loads(l) for l in open(J)]
bykey = defaultdict(lambda: {"started": [], "result": []})
for r in recs:
    bykey[r["key"]][r["type"]].append(r["agentId"])

def msg_text(m):
    c = m.get("content")
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(b.get("text","") for b in c
                         if isinstance(b, dict) and b.get("type")=="text")
    return ""

def first_prompt(aid):
    p = os.path.join(W, f"agent-{aid}.jsonl")
    if not os.path.exists(p): return ""
    for l in open(p):
        r = json.loads(l)
        if r.get("type")=="user" and isinstance(r.get("message"), dict):
            t = msg_text(r["message"])
            if t: return t
    return ""

def role(t):
    if "## THE FINDING UNDER TEST" in t: return "verify"
    if "## YOUR JOB: TRIAGE" in t: return "triage"
    if "THE NEXT ROUND'S WORK-LIST" in t: return "critic"
    if "MANDATORY READING" in t or "surface_examined" in t: return "finder"
    return "unknown"

def result_shape(res):
    if not isinstance(res, dict): return "non-dict:"+type(res).__name__
    k = set(res)
    if "refuted" in k: return "verdict"
    if "findings" in k: return "finder"
    if "final_verdict" in k: return "triage"
    if "next_worklist" in k: return "critic"
    return "other:"+",".join(sorted(k))[:60]

results = {r["agentId"]: r["result"] for r in recs if r["type"]=="result"}

logical = []
for k, v in bykey.items():
    succ = v["result"][-1] if v["result"] else None
    logical.append({"key": k, "started": v["started"], "success": succ,
                    "n_attempts": len(v["started"]),
                    "role_by_prompt": role(first_prompt(succ or v["started"][0])),
                    "role_by_result": result_shape(results.get(succ)) if succ else "DEAD"})

print("=== LOGICAL AGENTS (one per workflow cache key) ===")
print("distinct cache keys        :", len(logical))
print("with a result              :", sum(1 for a in logical if a["success"]))
print("DEAD (no result, retried)  :", sum(1 for a in logical if not a["success"]))
print("keys needing >1 attempt    :", sum(1 for a in logical if a["n_attempts"]>1))
print("total attempts (= files)   :", sum(a["n_attempts"] for a in logical))
print()
print("role by PROMPT :", Counter(a["role_by_prompt"] for a in logical))
print("role by RESULT :", Counter(a["role_by_result"] for a in logical))
print()
print("cross-tab (prompt, result):")
for kk, n in Counter((a["role_by_prompt"], a["role_by_result"]) for a in logical).most_common():
    print("   ", kk, n)
print()
print("=== the DEAD agent ===")
for a in logical:
    if not a["success"]:
        print("  key", a["key"][:22], "attempts", a["n_attempts"],
              "role_by_prompt", a["role_by_prompt"])
        t = first_prompt(a["started"][0])
        m = re.search(r"^- \*\*title:\*\* (.*)$", t, re.M)
        print("  finding under test:", (m.group(1)[:110] if m else "(none — not a verify agent)"))
        m2 = re.search(r"Your scratch dir[^\n]*\n\s*(\S+)", t)
        print("  scratch:", m2.group(1) if m2 else "?")
        print("  prompt len:", len(t))
