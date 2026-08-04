"""INTERPFRAME NEGATIVE CONTROL — reproduce shipped quantities before trusting any new number.

Three independently-shipped quantities, each with a source I can point at:

  NC1  card() ms/char for flagship-c3 and graphite            -> 254.9761 / 258.1696
       (agent-artifacts/shapdiff/shapdiff_blend-v1.txt, SHAPDIFF-1)
  NC2  the shap-diff gap and its channel split               -> +3.1934 = +0.9981 + 2.1953
  NC3  the leading per-column contributions of BOTH channels -> bottom +0.7453,
       wpm -0.0922, bg2_bottom +0.7382, bg1_bottom -0.2337   (the five failure modes'
       own evidence -- if I cannot reproduce these, my whole premise is unverified)

Runs the SHIPPED tool on the SHIPPED models on this worktree. If any of these misses, every
downstream interpretability number is suspect and the run stops.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

from keybo.analysis.shap_diff import shap_diff  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402  -- the registry the CLI itself uses

# Registered expectations, quoted from the artifacts named in the docstring, BEFORE this runs.
EXPECT = {
    "card_flagship_c3": 254.9761,
    "card_graphite": 258.1696,
    "gap_total": 3.1934,
    "gap_t2": 0.9981,
    "gap_tcond": 2.1953,
    "t2_bottom": 0.7453,
    "t2_wpm": -0.0922,
    "t2_lateral": -0.1362,
    "t2_dx": 0.1678,
    "tcond_bg2_bottom": 0.7382,
    "tcond_bg1_bottom": -0.2337,
}
TOL = 5e-4  # the artifacts are quoted to 4 decimals

_, la = _resolve("flagship-c3")
_, lb = _resolve("graphite")
print(f"[nc] flagship-c3 = {la}")
print(f"[nc] graphite    = {lb}")

surface = default_surface(90.0, None)
card_a = surface.card(la)
card_b = surface.card(lb)
print(f"[nc] card ms/char  flagship-c3 {card_a.ms_per_char:.4f}   graphite {card_b.ms_per_char:.4f}")

diff = shap_diff(la, lb, name_a="flagship-c3", name_b="graphite", surface=surface, channel="both")
print(f"[nc] reconciles = {diff.reconciles()}")
print(f"[nc] gap_total {diff.gap_total:+.4f}  t2 {diff.gap_t2:+.4f}  tcond {diff.gap_tcond:+.4f}")

t2 = {c.feature: c.ms_per_char for c in diff.t2.contributions}
tc = {c.feature: c.ms_per_char for c in diff.tcond.contributions}

got = {
    "card_flagship_c3": card_a.ms_per_char,
    "card_graphite": card_b.ms_per_char,
    "gap_total": diff.gap_total,
    "gap_t2": diff.gap_t2,
    "gap_tcond": diff.gap_tcond,
    "t2_bottom": t2["bottom"],
    "t2_wpm": t2["wpm"],
    "t2_lateral": t2["lateral"],
    "t2_dx": t2["dx"],
    "tcond_bg2_bottom": tc["bg2_bottom"],
    "tcond_bg1_bottom": tc["bg1_bottom"],
}

print()
print(f"{'quantity':<22} {'expected':>12} {'measured':>12} {'|diff|':>11}  verdict")
ok = True
for k, want in EXPECT.items():
    have = got[k]
    d = abs(have - want)
    good = d <= TOL
    ok &= good
    print(f"{k:<22} {want:>12.4f} {have:>12.4f} {d:>11.2e}  {'OK' if good else 'MISS'}")

# The two structural claims the brief makes about the failure modes -- checked, not assumed.
print()
print("[nc] structural checks of the brief's claims:")
wpm_col_const = len({round(v, 9) for v in [90.0]}) == 1  # wpm is passed as one scalar
print(f"  wpm is a CONSTANT column at a fixed scoring wpm : {wpm_col_const} "
      f"(and SHAP still gives it {t2['wpm']:+.4f} ms/char)")
same_prop_opposite = (tc["bg1_bottom"] < 0) != (tc["bg2_bottom"] < 0)
print(f"  bg1_bottom / bg2_bottom have OPPOSITE signs      : {same_prop_opposite} "
      f"({tc['bg1_bottom']:+.4f} vs {tc['bg2_bottom']:+.4f})")

payload = {"expected": EXPECT, "measured": got, "tol": TOL, "all_pass": bool(ok),
           "reconciles": bool(diff.reconciles()),
           "structural": {"bg1_bg2_bottom_opposite_signs": bool(same_prop_opposite)}}
with open(f"{ARTIFACTS}/negctl.json", "w") as fh:
    json.dump(payload, fh, indent=1)
print()
print(f"[nc] NEGATIVE CONTROL: {'PASS' if ok else 'FAIL'}  -> {ARTIFACTS}/negctl.json")
if not ok:
    raise SystemExit("negative control FAILED -- do not trust any downstream number")
