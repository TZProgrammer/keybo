"""lmscissor REFLECT audit item 4: is any OTHER repair contaminated the way R7 is?

R7's flaw: it prices a bigram using a cell whose SAMPLE SUPPORT is 99.96% a different sub-class
(lower=index) than the bigram being priced (lower=middle). So the test is: for every repair that
uses a MEASURED number, does any bigram carrying material corpus mass get priced by a cell whose
support is dominated by a sub-class that bigram is not in?

R1..R5 are PREDICATES (weights 0/1/4 by geometry), so they cannot be "contaminated by a measured
cell" at all — they carry no measurement. The audit for them is different and is done here too:
does the predicate's own SUPPORT contain the deciding mass? (i.e. is the repair even able to see
the difference — R2's known failure.)

R6 and R7 are the measured ones. For each, list every cell it actually USES to price mass that
differs between the layouts, and report that cell's own sub-class purity.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import _DEX  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

H = json.load(open("/tmp/lmscissor_harvest.json"))
CELLS = H["cells"]
LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}
bigrams = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))


def kind(x):
    return G.finger(x).value.split("-")[1]


def feats(lay, bg):
    a, b = lay.pos(bg[0]), lay.pos(bg[1])
    if not C.same_hand(G, a, b) or C.same_finger(G, a, b):
        return None
    dy = abs(a[1] - b[1])
    if dy == 0:
        return None
    ka, kb = kind(a[0]), kind(b[0])
    lk = ka if a[1] < b[1] else kb
    uk = kb if a[1] < b[1] else ka
    return dy, lk, uk, _DEX[lk] < _DEX[uk], C.is_adjacent(G, a, b)


# ---------- 1. which bigrams actually DIFFER between the two layouts? -------------------
per_layout = {}
for label, spec in LAYOUTS.items():
    lay = Layout(spec, G)
    d = {}
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg or not all(lay.has_key(c) for c in bg):
            continue
        f = feats(lay, bg)
        if f is not None:
            d[bg] = f
    per_layout[label] = d

allbg = set(per_layout["keybo-lsb"]) | set(per_layout["keybo-lsb+lm"])
differing = {
    bg for bg in allbg
    if per_layout["keybo-lsb"].get(bg) != per_layout["keybo-lsb+lm"].get(bg)
}
mass = sum(bigrams[bg] for bg in differing)
print(f"bigrams whose CLASS differs between the layouts: {len(differing)}, "
      f"corpus mass {mass} ({100*mass/sum(bigrams.values()):.4f}% of all bigram mass)")
top = sorted(differing, key=lambda b: -bigrams[b])[:14]
print(f"  top by mass: {[(b, bigrams[b]) for b in top]}")


# ---------- 2. sub-class purity of every cell in the harvest ---------------------------
def purity(key: str):
    """For an aggregate cell, how concentrated is its sample support in one lower-finger?"""
    c = CELLS.get(key)
    if c is None:
        return None
    # recompute lower-finger split is impossible from the aggregate alone for C:/A:/X: keys,
    # so use the E: (explicit lower|upper|dy) cells that COMPOSE it.
    return sum(n for bg in c["bigrams"].values() for (_s, n) in bg.values())


def explicit_support(lk, uk, dy):
    key = f"E:{lk}|{uk}|dy{dy}"
    c = CELLS.get(key)
    if c is None:
        return 0, 0
    n = sum(x for bg in c["bigrams"].values() for (_s, x) in bg.values())
    return n, len(c["bigrams"])


def coarse_split(orient, dy, adj):
    """Sample census of a coarse cell BY lower-finger, via its composing explicit cells."""
    tot = defaultdict(int)
    for key, c in CELLS.items():
        if not key.startswith("E:"):
            continue
        body = key[2:]
        lk, uk, dyt = body.split("|")
        if int(dyt[2:]) != dy:
            continue
        wl = _DEX[lk] < _DEX[uk]
        if ("weakLOWER" if wl else "weakTOP") != orient:
            continue
        # adjacency is a POSITION property, not derivable from (lk,uk) alone -> approximate by
        # checking whether the pair can be adjacent at all; report both and flag the caveat.
        n = sum(x for bg in c["bigrams"].values() for (_s, x) in bg.values())
        tot[lk] += n
    return dict(tot)


print("\n" + "=" * 100)
print("AUDIT 4 — per-repair contamination check")
print("=" * 100)

print("\nR1 / R2 / R3 / R4 / R5 are PREDICATES (0/1/4 weights from geometry only).")
print("  They use NO measured cell, so R7-style contamination is structurally impossible.")
print("  The relevant failure for them is instead: does the repair's SUPPORT contain the")
print("  deciding mass? Checked below.")

REPAIRS_PRED = {
    "shipped": lambda f: 1.0 if f[3] else 0.0,
    "R1": lambda f: 1.0,
    "R2": lambda f: (4.0 if f[0] == 2 else 1.0) if f[3] else 0.0,
    "R3": lambda f: 1.0 if f[0] == 2 else 0.0,
    "R4": lambda f: 1.0 if (f[3] or (f[0] == 2 and not f[4])) else 0.0,
    "R5": lambda f: 1.0 if f[3] else (0.0 if f[1] == "index" else 1.0),
}
print(f"\n  {'repair':<10}{'differing mass IN support':>28}{'as % of differing mass':>26}")
for name, w in REPAIRS_PRED.items():
    seen = 0
    for bg in differing:
        fa = per_layout["keybo-lsb"].get(bg)
        fb = per_layout["keybo-lsb+lm"].get(bg)
        wa = w(fa) if fa else 0.0
        wb = w(fb) if fb else 0.0
        if wa or wb:
            seen += bigrams[bg]
    print(f"  {name:<10}{seen:>28}{100.0*seen/mass:>25.2f}%")

print("\nR6 / R7 use MEASURED cells. For every cell each uses to price DIFFERING mass,")
print("report that cell's own support and (for coarse cells) its lower-finger census.")

for rname, use_coarse_only in (("R6 (explicit, coarse fallback)", False), ("R7 (coarse only)", True)):
    print(f"\n  --- {rname} ---")
    used = defaultdict(int)
    for bg in differing:
        for label in LAYOUTS:
            f = per_layout[label].get(bg)
            if f is None:
                continue
            dy, lk, uk, wl, adj = f
            ek = f"E:{lk}|{uk}|dy{dy}"
            ck = f"C:{'weakLOWER' if wl else 'weakTOP'}|dy{dy}|{'adj' if adj else 'nonadj'}"
            if use_coarse_only:
                used[ck] += bigrams[bg]
            else:
                n, nbg = explicit_support(lk, uk, dy)
                used[ek if (n >= 200 and nbg >= 3) else ck] += bigrams[bg]
    for key, m in sorted(used.items(), key=lambda kv: -kv[1]):
        c = CELLS.get(key)
        n = sum(x for bgv in c["bigrams"].values() for (_s, x) in bgv.values()) if c else 0
        nbg = len(c["bigrams"]) if c else 0
        note = ""
        if key.startswith("C:"):
            orient, dyt, adj = key[2:].split("|")
            census = coarse_split(orient, int(dyt[2:]), adj)
            tot = sum(census.values()) or 1
            dom = max(census.items(), key=lambda kv: kv[1])
            note = (f"  <-- COARSE; lower-finger census {census}; "
                    f"dominant lower={dom[0]} at {100*dom[1]/tot:.2f}%")
        print(f"    {key:<44} mass {m:>12}  cell n {n:>9}  bigrams {nbg:>3}{note}")

json.dump({"differing_bigrams": sorted(differing), "differing_mass": mass},
          open("/tmp/lmscissor_audit4.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_audit4.json")
