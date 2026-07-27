"""lmscissor (d) per-source: what do the three fitted surfaces say about the two layouts?

Uses the repo's own documented API (`surfaces.model_scores`) so the numbers are on the campaign's
frame. POOL is NOT independent of AALTO (it pools the sources) — reported and labelled as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis import surfaces as S  # noqa: E402
from keybo.data.corpus import production_corpus_dir  # noqa: E402

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}
corpus_dir = production_corpus_dir(None)
tri_path = str(corpus_dir / "trigrams.txt")
print(f"corpus = {corpus_dir.name}; trigram table = {tri_path}")
print(f"FRAME_NOTE: {S.FRAME_NOTE}")
print(f"WPM note: {S.wpm_note(S.BAKED_WPM)}")

out = {}
for label, spec in LAYOUTS.items():
    res = S.model_scores(spec, trigram_path=tri_path)
    out[label] = res
    print(f"\n=== {label} ===")
    print(f"  available: {res.get('available')}  reason: {res.get('reason')}")
    for name, cell in (res.get("surfaces") or {}).items():
        print(f"  {name:<32} fit {cell.get('fit')}")

print(f"\n{'='*96}\nPER-SOURCE COMPARISON (fit, lower = faster)\n{'='*96}")
a_s = (out["keybo-lsb"].get("surfaces") or {})
b_s = (out["keybo-lsb+lm"].get("surfaces") or {})
print(f"{'surface':<34}{'keybo-lsb':>14}{'keybo-lsb+lm':>16}{'delta':>12}  winner")
for name in sorted(set(a_s) | set(b_s)):
    fa = a_s.get(name, {}).get("fit")
    fb = b_s.get(name, {}).get("fit")
    if fa is None or fb is None:
        print(f"{name:<34}{'n/a':>14}{'n/a':>16}")
        continue
    win = "keybo-lsb" if fa < fb else ("keybo-lsb+lm" if fb < fa else "tie")
    note = "  (NOT independent — pools AALTO)" if name.startswith("POOL") else ""
    print(f"{name:<34}{fa:>14.6f}{fb:>16.6f}{fb-fa:>+12.6f}  {win}{note}")

json.dump(out, open("/tmp/lmscissor_persource.json", "w"), indent=2, default=str)
print("\nwrote /tmp/lmscissor_persource.json")
