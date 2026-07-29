"""EXTERNAL ANCHOR: every number I report must come out of `keybo analyze --json`.

My other drivers call ``BadScissor`` directly, which shares a component with its target — the
same class I am auditing. This driver instead shells the SHIPPED CLI (``keybo analyze --json``)
in a subprocess and re-derives the headline facts from ITS output, so the report rests on the
production path a user actually runs, not on my in-process import.

It also verifies the CLI's own emitted metadata (``denominator``, ``attribution_rule``) says
what the module docstring claims, since that string is what a JSON consumer reads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/tmp/bsaudit")
VENV_PY = "/local/home/zegertho/repos/keybo/.venv/bin/python"
LAYOUTS = ["qwerty", "keybo-lsb", "keybo-lsb+lm", "flagship-c3", "lsb-sib", "dvorak",
           "graphite", "semimak", "archive-1843", "archive-1846", "colemak", "qwerty30m",
           "keybo-c30m", "p13stab-win", "p16-balance"]


def run_cli(args: list[str]) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    cmd = [VENV_PY, "-m", "keybo.cli", *args]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"CLI failed rc={proc.returncode}\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def main() -> int:
    # POSITIVE CONTROL FIRST: prove the subprocess is running THIS tree, not the shared
    # clone (the editable .pth trap). Ask the interpreter where keybo came from.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    where = subprocess.run(
        [VENV_PY, "-c",
         "from keybo.testkit import assert_module_under as a;"
         f"print(a('keybo', {str(ROOT)!r}));"
         f"print(a('keybo.analysis.bad_scissor', {str(ROOT)!r}))"],
        cwd=ROOT, capture_output=True, text=True, env=env)
    print("=== POSITIVE CONTROL: which tree does the SUBPROCESS import? ===")
    print(where.stdout.strip() or where.stderr.strip())
    if where.returncode != 0:
        raise SystemExit("the subprocess is NOT on /tmp/bsaudit — every number below "
                         "would describe the wrong tree")
    # And the negative control: without PYTHONPATH it must resolve elsewhere, proving the
    # guard is load-bearing rather than vacuous.
    neg = subprocess.run([VENV_PY, "-c", "import keybo; print(keybo.__file__)"],
                         cwd=ROOT, capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    print(f"  NEGATIVE control (no PYTHONPATH): {neg.stdout.strip()}")
    print(f"  guard is load-bearing: {str(ROOT) not in neg.stdout}\n")

    # ⚠ The CLI's DEFAULT corpus is blend-v1; the spec's pinned values (and my other
    # drivers) are on iWeb. Pass --corpus explicitly so the comparison is like-for-like
    # instead of silently comparing two corpora — the exact ALLGAUGE-1 failure mode.
    data = run_cli(["analyze", *LAYOUTS, "--json", "--no-time", "--corpus", "iweb"])
    print(f"=== `keybo analyze --json --corpus iweb` ===")
    print(f"  corpus reported by the CLI: {data.get('corpus')!r}  "
          f"skipgram_table={data.get('skipgram_table')!r}")
    assert data.get("corpus") == "iweb", f"asked for iweb, CLI used {data.get('corpus')!r}"
    rows = data["rows"]
    print(f"  returned {len(rows)} layouts")

    bs = {name: row["bad_scissor"] for name, row in rows.items() if "bad_scissor" in row}
    print(f"  layouts carrying a bad_scissor block: {len(bs)}")
    first = next(iter(bs.values()))
    print(f"  emitted keys: {sorted(first)}")
    print(f"  denominator      = {first.get('denominator')!r}")
    print(f"  attribution_rule = {first.get('attribution_rule')!r}")

    print("\n=== SHARES FROM THE SHIPPED PATH (share DESC) ===")
    for name, b in sorted(bs.items(), key=lambda kv: -kv[1]["share"]):
        cells = b["by_cell"]
        dy1 = sum(v for k, v in cells.items() if k.endswith("dy1"))
        dy2 = sum(v for k, v in cells.items() if k.endswith("dy2"))
        print(f"  {name:16s} share={b['share']:9.5f} dy1={dy1:8.5f} dy2={dy2:8.5f} "
              f"dy1%={100 * dy1 / b['share']:6.2f} dy2%={100 * dy2 / b['share']:6.3f}")

    # The user's pair, straight off the shipped path.
    a, b = bs["keybo-lsb"], bs["keybo-lsb+lm"]
    print("\n=== Q4 VIA THE SHIPPED PATH ===")
    print(f"  keybo-lsb    share = {a['share']:.5f}")
    print(f"  keybo-lsb+lm share = {b['share']:.5f}")
    print(f"  gap = {b['share'] - a['share']:+.5f}  ratio = {b['share'] / a['share']:.5f}")
    print(f"  qwerty/keybo-lsb ratio = {bs['qwerty']['share'] / a['share']:.4f}")
    dy2a = sum(v for k, v in a["by_cell"].items() if k.endswith("dy2"))
    dy2b = sum(v for k, v in b["by_cell"].items() if k.endswith("dy2"))
    print(f"  dy2 mass: keybo-lsb {dy2a:.5f} vs +lm {dy2b:.5f}  "
          f"delta {dy2b - dy2a:+.9f}  (EXACTLY EQUAL: {dy2b == dy2a})")
    dy1a = sum(v for k, v in a["by_cell"].items() if k.endswith("dy1"))
    dy1b = sum(v for k, v in b["by_cell"].items() if k.endswith("dy1"))
    print(f"  dy1 mass: keybo-lsb {dy1a:.5f} vs +lm {dy1b:.5f}  delta {dy1b - dy1a:+.9f}")
    print(f"  => 100% of the gap is dy1: "
          f"{abs((dy1b - dy1a) - (b['share'] - a['share'])) < 1e-9}")
    fdiff = {f: b["by_finger"][f] - a["by_finger"][f] for f in a["by_finger"]}
    print(f"  fingers that move: {[f for f, d in fdiff.items() if abs(d) > 1e-12]}")

    # Cross-check my in-process numbers against the CLI's.
    mine = json.loads((ROOT / "agent-artifacts/bsaudit/registry_sweep.json").read_text())
    print("\n=== IN-PROCESS vs SHIPPED-CLI (max abs diff per layout) ===")
    worst = 0.0
    for name, b2 in bs.items():
        if name not in mine["registry"]:
            continue
        d = abs(b2["share"] - mine["registry"][name]["share"])
        worst = max(worst, d)
    print(f"  max |share_cli - share_inprocess| over {len(bs)} layouts = {worst:.3e}")
    print(f"  => my in-process figures ARE the shipped figures: {worst < 1e-9}")

    out = {
        "denominator": first.get("denominator"),
        "attribution_rule": first.get("attribution_rule"),
        "emitted_keys": sorted(first),
        "shares": {k: v["share"] for k, v in bs.items()},
        "user_pair": {
            "keybo_lsb": a["share"], "keybo_lsb_lm": b["share"],
            "gap": b["share"] - a["share"], "ratio": b["share"] / a["share"],
            "dy2_exactly_equal": dy2b == dy2a,
            "dy1_carries_the_whole_gap":
                abs((dy1b - dy1a) - (b["share"] - a["share"])) < 1e-9,
            "fingers_that_move": [f for f, d in fdiff.items() if abs(d) > 1e-12],
        },
        "max_abs_diff_vs_inprocess": worst,
    }
    p = ROOT / "agent-artifacts/bsaudit/shipped_path_anchor.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
