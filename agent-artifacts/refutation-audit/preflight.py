#!/usr/bin/env python3
"""Pre-flight: prove I am testing /tmp/refaudit, not the shared clone.

The trap (documented in TOOLING-TRAPS and in keybo/testkit.py on qap-audit): the
shared clone's .venv carries an editable .pth into repos/keybo/src, so a probe
silently tests the WRONG TREE while every printed path looks right.
"""
import importlib, os, sys
from pathlib import Path

ROOT = Path("/tmp/refaudit").resolve()
FORBIDDEN = Path("/local/home/zegertho/repos/keybo").resolve()

def check(module_name):
    m = importlib.import_module(module_name)
    loc = getattr(m, "__file__", None)
    if loc is None:
        raise SystemExit(f"FATAL {module_name!r} has no __file__")
    p = Path(loc).resolve()
    under = p.is_relative_to(ROOT)
    bad = p.is_relative_to(FORBIDDEN)
    print(f"  {module_name:32s} -> {p}")
    print(f"      under /tmp/refaudit: {under}   under SHARED clone: {bad}")
    if not under or bad:
        raise SystemExit(f"FATAL: {module_name} is NOT the worktree copy — harness untrustworthy")
    return p

print("sys.executable:", sys.executable)
print("PYTHONPATH    :", os.environ.get("PYTHONPATH"))
print("sys.path[0:4] :", sys.path[:4])
print("\n--- module resolution ---")
for name in ["keybo", "keybo.analysis.surfaces", "keybo.scoring.oxey",
             "keybo.data.community", "keybo.features.classify",
             "keybo.analysis.effect_curves", "keybo.training.tune",
             "keybo.data.corpus", "keybo.analysis.select"]:
    check(name)

# NEGATIVE CONTROL: the checker must be able to FAIL. Point it at a module that
# provably lives outside ROOT and confirm it raises.
print("\n--- negative control (the checker must reject a wrong-tree module) ---")
import json as _json
p = Path(_json.__file__).resolve()
print(f"  json -> {p}  (stdlib, provably outside {ROOT})")
try:
    check("json")
except SystemExit as e:
    print(f"  ✅ checker REJECTED it: {e}")
else:
    raise SystemExit("❌ CONTROL FAILED: checker accepted a module outside the worktree")

# git identity of the tree under test
import subprocess
sha = subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
st  = subprocess.run(["git","-C",str(ROOT),"status","--porcelain"],capture_output=True,text=True).stdout
print(f"\ntree HEAD: {sha}")
print(f"dirty files: {len([l for l in st.splitlines() if l.strip()])}")
print("\n✅ PRE-FLIGHT PASSED — probes below test /tmp/refaudit")
