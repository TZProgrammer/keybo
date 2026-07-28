#!/usr/bin/env python3
"""ls every path a REFUTING vote cited. Two findings in the original audit cited a
regression test that does not exist — a refutation citing a nonexistent artifact is
worse, because nothing downstream ever re-reads it."""
import json, os, subprocess
rows = json.load(open("/local/home/zegertho/agent/state/refaudit/artifacts/refutation-claims.json"))
ROOT = "/tmp/refaudit"

def repo_exists(rel, sha="dec1c3f"):
    r = subprocess.run(["git","-C",ROOT,"cat-file","-e",f"{sha}:{rel}"],capture_output=True)
    return r.returncode == 0

allp = {}
for x in rows:
    for p in x["paths"]:
        allp.setdefault(p, []).append(f"K{x['kill']}v{x['vote']}")

print(f"{'status':<26} {'cited by':<18} path")
tally = {}
for p, who in sorted(allp.items()):
    w = ",".join(who)
    if p.startswith("/"):
        if os.path.exists(p): st = "EXISTS (abs)"
        else: st = "MISSING (abs)"
    else:
        e_now = os.path.exists(os.path.join(ROOT, p))
        e_old = repo_exists(p)
        st = ("EXISTS repo" if e_now else "absent-worktree") + ("/dec1c3f-yes" if e_old else "/dec1c3f-NO")
    tally[st] = tally.get(st, 0) + 1
    print(f"{st:<26} {w:<18} {p}")
print()
for k, v in sorted(tally.items()): print(f"  {k}: {v}")
