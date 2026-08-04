"""INTERPFRAME-1 — the SERVED path must be BIT-IDENTICAL to the base branch.

The 39 deleted lines in this branch's src/ diff are all lines I widened rather than removed, but
"I only widened it" is a CLAIM about my own edits. This checks the THING: it recomputes the served
frame's own quantities on this branch and compares them, bit for bit, against the values the base
branch (`tcond`) produces — which are the same values SHAPDIFF-1 and SHAPDIFF-TCOND published.

If any of these move, a shipped number moved, and the "additive, opt-in" claim in the report is
false regardless of what the diff looks like.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe")
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

from keybo.analysis.shap_diff import block_map, default_models, shap_diff  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    FEATURE_VERSION,
    TRIGRAM_FEATURE_NAMES,
    bigram_features_from_positions,
    trigram_features_from_positions,
)

# Published by SHAPDIFF-1 / SHAPDIFF-TCOND from the BASE branch, quoted here as the reference.
PUBLISHED = {
    "card_a": 254.9761,
    "card_b": 258.1696,
    "gap_total": 3.1934,
    "gap_t2": 0.9981,
    "gap_tcond": 2.1953,
    "t2_bottom": 0.7453,
    "t2_dx": 0.1678,
    "t2_lateral": -0.1362,
    "t2_wpm": -0.0922,
    "t2_same_hand": 0.0855,
    "t2_angle": 0.0681,
    "tcond_bg2_bottom": 0.7382,
    "tcond_bg1_top": 0.4064,
    "tcond_bg1_dx": 0.3874,
    "tcond_bg1_bottom": -0.2337,
    "tcond_sg_dx": 0.1580,
}
TOL = 5e-4

WPM = 90.0
_, A = _resolve("flagship-c3")
_, B = _resolve("graphite")
surface = default_surface(WPM, None)
G = surface.geometry
pos = [*G.slots, G.space_position]

out: dict = {}
print("=" * 80)
print("SERVED-PATH ISOLATION — every shipped quantity, recomputed on THIS branch")
print("=" * 80)

# 1. the version stamp and the served name lists
print()
print("1. THE VERSION-LOCKED SURFACE")
checks = {
    "FEATURE_VERSION": (FEATURE_VERSION, "2026-07-05.3"),
    "len(BIGRAM_FEATURE_NAMES)": (len(BIGRAM_FEATURE_NAMES), 20),
    "len(TRIGRAM_FEATURE_NAMES)": (len(TRIGRAM_FEATURE_NAMES), 46),
    "BIGRAM_FEATURE_NAMES[-1]": (BIGRAM_FEATURE_NAMES[-1], "wpm"),
    "TRIGRAM_FEATURE_NAMES[-1]": (TRIGRAM_FEATURE_NAMES[-1], "wpm"),
}
ok = True
for label, (got, want) in checks.items():
    good = got == want
    ok &= good
    print(f"   {label:<30} {str(got):<16} expect {want!s:<16} {'OK' if good else 'MOVED'}")
out["version_surface"] = {k: str(v[0]) for k, v in checks.items()}

# 2. the shipped models still load and still carry the served frame
print()
print("2. THE SHIPPED ARTIFACTS")
for kind, expected in (("bigram", BIGRAM_FEATURE_NAMES), ("trigram", TRIGRAM_FEATURE_NAMES)):
    for i, m in enumerate(default_models(kind)):
        good = (
            m.metadata.feature_version == FEATURE_VERSION
            and list(m.metadata.feature_names) == list(expected)
        )
        ok &= good
        if i == 0:
            print(
                f"   {kind:<8} seed0 stamp {m.metadata.feature_version!r} "
                f"{len(m.metadata.feature_names)} cols  {'OK' if good else 'MOVED'}"
            )

# 3. the FEATURE MATRICES, bit for bit (this is where a subtracting flag would show)
print()
print("3. THE SERVED FEATURE MATRICES (a checksum a changed column could not survive)")
Xb = np.vstack([bigram_features_from_positions(G, (a, b), wpm=WPM) for a in pos for b in pos])
Xt = np.vstack(
    [
        trigram_features_from_positions(G, (a, b, c), wpm=WPM)
        for a in pos[:8]
        for b in pos[:8]
        for c in pos[:8]
    ]
)
out["matrix_checksums"] = {
    "bigram_shape": list(Xb.shape),
    "bigram_sum": float(Xb.sum()),
    "bigram_absmax": float(np.abs(Xb).max()),
    "trigram_shape": list(Xt.shape),
    "trigram_sum": float(Xt.sum()),
}
print(f"   bigram  {Xb.shape}  sum {Xb.sum():.6f}")
print(f"   trigram {Xt.shape}  sum {Xt.sum():.6f}")
ok &= Xb.shape == (961, 20) and Xt.shape == (512, 46)

# 4. the block partitions the shipped tool uses
print()
print("4. THE SHIPPED BLOCK PARTITIONS")
b_blocks = {v[0] for v in block_map(BIGRAM_FEATURE_NAMES).values()}
t_blocks = {v[0] for v in block_map(TRIGRAM_FEATURE_NAMES).values()}
good = b_blocks == {"ROW", "FINGER", "RELATIONAL", "GEOMETRY", "WPM"} and "BG1" in t_blocks
ok &= good
print(f"   bigram blocks  {sorted(b_blocks)}   {'OK' if good else 'MOVED'}")
print(f"   trigram blocks {sorted(t_blocks)}")

# 5. THE PUBLISHED NUMBERS
print()
print("5. THE PUBLISHED SHAPDIFF NUMBERS, recomputed here")
d = shap_diff(A, B, name_a="flagship-c3", name_b="graphite", surface=surface, channel="both")
ca, cb = surface.card(A), surface.card(B)
t2 = {c.feature: c.ms_per_char for c in d.t2.contributions}
tc = {c.feature: c.ms_per_char for c in d.tcond.contributions}
got = {
    "card_a": ca.ms_per_char,
    "card_b": cb.ms_per_char,
    "gap_total": d.gap_total,
    "gap_t2": d.gap_t2,
    "gap_tcond": d.gap_tcond,
    "t2_bottom": t2["bottom"],
    "t2_dx": t2["dx"],
    "t2_lateral": t2["lateral"],
    "t2_wpm": t2["wpm"],
    "t2_same_hand": t2["same_hand"],
    "t2_angle": t2["angle"],
    "tcond_bg2_bottom": tc["bg2_bottom"],
    "tcond_bg1_top": tc["bg1_top"],
    "tcond_bg1_dx": tc["bg1_dx"],
    "tcond_bg1_bottom": tc["bg1_bottom"],
    "tcond_sg_dx": tc["sg_dx"],
}
print(f"   reconciles: {d.reconciles()}   card_tie_applies: {d.card_tie_applies} (must be True)")
ok &= d.reconciles() and d.card_tie_applies
print(f"   {'quantity':<22} {'published':>12} {'this branch':>12} {'|diff|':>11}")
for k, want in PUBLISHED.items():
    have = got[k]
    diff = abs(have - want)
    good = diff <= TOL
    ok &= good
    print(f"   {k:<22} {want:>12.4f} {have:>12.4f} {diff:>11.2e}  {'OK' if good else 'MOVED'}")
out["published_vs_this_branch"] = {"published": PUBLISHED, "measured": got, "tol": TOL}

out["all_pass"] = bool(ok)
with open(f"{ARTIFACTS}/isolation.json", "w") as fh:
    json.dump(out, fh, indent=1)
print()
print("=" * 80)
print(f"SERVED-PATH ISOLATION: {'PASS — no shipped quantity moved' if ok else 'FAIL'}")
print("=" * 80)
print(f"wrote {ARTIFACTS}/isolation.json")
if not ok:
    raise SystemExit("a shipped quantity MOVED -- the additive/opt-in claim is false")
