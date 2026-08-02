"""pick2 step 4: the axes a learner actually FEELS, plus adoption friction.

Speed cannot separate the leading group (step 2), so the decision has to come from axes that
are (i) not the fitted model and (ii) things a human notices. Via the shipped `keybo analyze`
so every gauge keeps its own registered denominator -- I do not re-implement any of them.

Two guards applied, both from the repo's own instruments:
* `sfr`, `alt`, `imbalance` are EXACTLY invariant under within-hand permutation
  (`analysis/discrimination.py`), so a tie on them is FORCED, not agreement. They are EXCLUDED
  from any win count here.
* charset-dependent cells render N/A rather than a number; N/A is never counted as a win.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL  # noqa: E402

HERE = Path(__file__).resolve().parent
PY_EXE = sys.executable
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "blend-v1"
WPM = sys.argv[2] if len(sys.argv) > 2 else "90"


def main() -> int:
    t0 = time.time()
    cmd = [PY_EXE, "-m", "keybo", "analyze", *ALL.values(),
           "--ref", ALL["qwerty"], "--corpus", CORPUS, "--target-wpm", WPM,
           "--attribution", "--scissor-pairs", "--json"]
    print(f"running: keybo analyze <{len(ALL)} boards> --corpus {CORPUS} --target-wpm {WPM}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[2]))
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"analyze failed rc={r.returncode}")
    data = json.loads(r.stdout)
    # re-key rows from layout string back to my names
    by_lay = {v: k for k, v in ALL.items()}
    data["rows"] = {by_lay.get(k, k): v for k, v in data["rows"].items()}
    if len(data["rows"]) != len(ALL):
        raise SystemExit(f"row count mismatch: {len(data['rows'])} != {len(ALL)}")
    out = HERE / f"gauges_{CORPUS}_wpm{WPM}.json"
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f}s, {len(data['rows'])} rows)")
    print("discrimination:", json.dumps(data.get("discrimination", {}).get("forced_ties", {}))[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
