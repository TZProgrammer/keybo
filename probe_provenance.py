"""lmscissor (2): verify the registered "~55%" / "-56% middle-pinky leaf" claim against the
gauge it was actually measured on (tb_objective_v2's dy==2 scissor family).

If it reproduces, the comment is CORRECT-but-gauge-specific, not wrong — and the apparent
contradiction with bad-scissor is a support difference, not an error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V2 = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/v2")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.append(str(V2))  # append, NOT insert(0) — trap 35

from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}

# check the v2 module does not reach back into the shared clone (trap 35)
src = (V2 / "tb_objective_v2.py").read_text()
if "repos/keybo" in src:
    hits = [ln for ln in src.splitlines() if "repos/keybo" in ln]
    print(f"⚠ TRAP 35: tb_objective_v2.py references repos/keybo:\n  " + "\n  ".join(hits))
else:
    print("trap-35 check: tb_objective_v2.py has no hardcoded repos/keybo path ✓")

import tb_objective_v2 as V  # noqa: E402

print(f"v2 SCISSOR_LEAVES: {V.SCISSOR_LEAVES}")

corpus_dir = production_corpus_dir(None)
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"corpus = {corpus_dir.name}")

# Reproduce the leaf shares directly from the v2 event predicate, over corpus bigrams.
ev = V._scissor_event


def leaf_masses(spec):
    """pair_name -> mass, plus bin_name -> mass, over dy==2 events only (v2's own gate)."""
    slot_of = {ch: i for i, ch in enumerate(spec)}
    by_pair: dict[str, float] = {}
    by_bin: dict[str, float] = {}
    total = 0.0
    den = 0
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg:
            continue
        if not all(c in slot_of for c in bg):
            continue
        den += freq
        a = G.slots[slot_of[bg[0]]]
        b = G.slots[slot_of[bg[1]]]
        e = ev(a, b, G)
        if e is None:
            continue
        by_pair[e.pair_name] = by_pair.get(e.pair_name, 0.0) + freq
        by_bin[e.bin_name] = by_bin.get(e.bin_name, 0.0) + freq
        total += freq
    return by_pair, by_bin, total, den


res = {}
for label, spec in LAYOUTS.items():
    res[label] = leaf_masses(spec)

pa, ba, ta, da = res["keybo-lsb"]
pb, bb, tb, db = res["keybo-lsb+lm"]

print(f"\n{'='*96}")
print("THE v2 GAUGE (dy==2 ONLY) — bigram MASS share per scissor leaf, blend-v1")
print(f"{'='*96}")
print(f"{'leaf':<18}{'keybo-lsb':>14}{'keybo-lsb+lm':>15}{'delta':>12}{'rel %':>10}")
for leaf in V.SCISSOR_LEAVES:
    va = 100.0 * pa.get(leaf, 0.0) / da
    vb = 100.0 * pb.get(leaf, 0.0) / db
    rel = (100.0 * (vb - va) / va) if va else float("nan")
    star = "  <<< the registered claim" if leaf == "middle_pinky" else ""
    print(f"{leaf:<18}{va:>14.4f}{vb:>15.4f}{vb-va:>+12.4f}{rel:>+9.1f}%{star}")
va, vb = 100.0 * ta / da, 100.0 * tb / db
print(f"{'TOTAL (dy==2)':<18}{va:>14.4f}{vb:>15.4f}{vb-va:>+12.4f}{100.0*(vb-va)/va:>+9.1f}%")

print(f"\n{'='*96}")
print("THE SUB-BIN THAT VETOED IT (v2's no-compensation gate)")
print(f"{'='*96}")
keys = sorted(set(ba) | set(bb))
print(f"{'bin':<58}{'keybo-lsb':>12}{'+lm':>12}{'rel %':>10}")
for k in keys:
    x = 100.0 * ba.get(k, 0.0) / da
    y = 100.0 * bb.get(k, 0.0) / db
    if abs(y - x) < 1e-9:
        continue
    rel = (100.0 * (y - x) / x) if x else float("inf")
    mark = "  <<< the +537% bin" if "middle_pinky" in k and "adverse" in k and "top_to_bottom" in k else ""
    print(f"{k:<58}{x:>12.4f}{y:>12.4f}{rel:>+9.1f}%{mark}")

json.dump(
    {
        "leaf_share_pct": {
            "keybo-lsb": {k: 100.0 * v / da for k, v in pa.items()},
            "keybo-lsb+lm": {k: 100.0 * v / db for k, v in pb.items()},
        },
        "bin_share_pct": {
            "keybo-lsb": {k: 100.0 * v / da for k, v in ba.items()},
            "keybo-lsb+lm": {k: 100.0 * v / db for k, v in bb.items()},
        },
    },
    open("/tmp/lmscissor_provenance.json", "w"),
    indent=2,
)
print("\nwrote /tmp/lmscissor_provenance.json")
