"""lmscissor: the empirical arbiter (fast, from the harvested aggregates).

Question (c)+(d): score both layouts by the MEASURED relative excess of the class each bigram
falls into, and bootstrap the DIFFERENCE (bigram-identity clustered) so the verdict carries
uncertainty rather than a bare point estimate.

Three pricing policies, reported separately because they differ in what they do with cells the
Aalto sample barely covers:
  P1  explicit cell only; unsupported -> unpriced (0 contribution, reported as unpriced%)
  P2  explicit cell, else the COARSE orientation x dy x adjacency cell   (= repair R6)
  P3  explicit cell, else the LOWER-FINGER-matched cell (lower_kind x dy)
P2's fallback is the one to distrust for pinky-upper pairs: its weakTOP|dy2|nonadj cell is
99.96% lower-key-is-index by sample count.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import _DEX  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

MIN_RAW, MIN_PIDS, MIN_BIGRAMS = 200, 20, 3
H = json.load(open("/tmp/lmscissor_harvest.json"))
BASE = {int(k): v[0] / v[1] for k, v in H["baseline"].items() if v[1] >= MIN_RAW}
CELLS = H["cells"]
print(f"baseline ms: { {k: round(v,2) for k,v in sorted(BASE.items())} }")

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}


def rel_of(key: str, ids=None):
    """Relative excess of a cell, pooling the given bigram identities (with replacement OK)."""
    cell = CELLS.get(key)
    if cell is None:
        return None
    per_bg = cell["bigrams"]
    ids = list(per_bg) if ids is None else ids
    pooled: dict[int, list] = defaultdict(lambda: [0.0, 0])
    for bg in ids:
        for bk, (s, n) in per_bg[bg].items():
            slot = pooled[int(bk)]
            slot[0] += s
            slot[1] += n
    rels = [
        s / n / BASE[bk] - 1.0
        for bk, (s, n) in pooled.items()
        if bk in BASE and n >= MIN_RAW
    ]
    return sum(rels) / len(rels) if rels else None


def is_supported(key: str) -> bool:
    cell = CELLS.get(key)
    if cell is None:
        return False
    n_raw = sum(n for bg in cell["bigrams"].values() for (_s, n) in bg.values())
    return (
        n_raw >= MIN_RAW
        and cell["n_pids"] >= MIN_PIDS
        and len(cell["bigrams"]) >= MIN_BIGRAMS
        and rel_of(key) is not None
    )


SUPPORTED = {k for k in CELLS if is_supported(k)}
print(f"supported cells: {len(SUPPORTED)} of {len(CELLS)}")

# ---- the per-layout bigram -> cell mapping ------------------------------------------
corpus_dir = production_corpus_dir(None)
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"corpus = {corpus_dir.name} (bigrams.txt)")


def kind(x):
    return G.finger(x).value.split("-")[1]


def mapped(spec):
    lay = Layout(spec, G)
    rows, den = [], 0
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg:
            continue
        if not all(lay.has_key(c) for c in bg):
            continue
        den += freq
        a, b = lay.pos(bg[0]), lay.pos(bg[1])
        if not C.same_hand(G, a, b) or C.same_finger(G, a, b):
            continue
        dy = abs(a[1] - b[1])
        if dy == 0:
            continue
        ka, kb = kind(a[0]), kind(b[0])
        lk = ka if a[1] < b[1] else kb
        uk = kb if a[1] < b[1] else ka
        orient = "weakLOWER" if _DEX[lk] < _DEX[uk] else "weakTOP"
        adj = "adj" if C.is_adjacent(G, a, b) else "nonadj"
        rows.append(
            (
                freq,
                f"E:{lk}|{uk}|dy{dy}",
                f"C:{orient}|dy{dy}|{adj}",
                f"L:lower={lk}|dy{dy}",
                bg,
            )
        )
    return rows, den


MAP = {label: mapped(spec) for label, spec in LAYOUTS.items()}
POLICIES = {
    "P1 explicit only (else unpriced)": None,
    "P2 explicit -> COARSE fallback (= R6)": "coarse",
    "P3 explicit -> LOWER-FINGER fallback": "lowmatch",
}


def score(label, policy, cache):
    rows, den = MAP[label]
    num = unpriced = 0.0
    for freq, ek, ck, lk, _bg in rows:
        r = cache.get(ek)
        if r is None and policy is not None:
            r = cache.get(ck if policy == "coarse" else lk)
        if r is None:
            unpriced += freq
            continue
        num += freq * r
    return 100.0 * num / den, 100.0 * unpriced / den


def build_cache(rng=None):
    cache = {}
    for key in SUPPORTED:
        ids = list(CELLS[key]["bigrams"])
        if rng is not None:
            ids = [rng.choice(ids) for _ in ids]
        r = rel_of(key, ids)
        if r is not None:
            cache[key] = r
    return cache


point = build_cache()

print(f"\n{'='*104}")
print("ROW-TRAVEL COST INDEX from the MEASURED Aalto surface (blend-v1 bigram weights)")
print("pp of layout-restricted space-excluded bigram mass x relative excess. LOWER IS BETTER.")
print(f"{'='*104}")
print(f"{'policy':<42}{'keybo-lsb':>12}{'keybo-lsb+lm':>14}{'delta':>10}{'unpriced%':>11}  winner")
results = {}
for pname, policy in POLICIES.items():
    a, ua = score("keybo-lsb", policy, point)
    b, ub = score("keybo-lsb+lm", policy, point)
    win = "keybo-lsb" if a < b else ("keybo-lsb+lm" if b < a else "tie")
    print(f"{pname:<42}{a:>12.4f}{b:>14.4f}{b-a:>+10.4f}{max(ua,ub):>10.3f}%  {win}")
    results[pname] = {"keybo-lsb": a, "keybo-lsb+lm": b, "delta": b - a, "winner": win}

print(f"\n{'='*104}\nBOOTSTRAP of the DIFFERENCE (bigram-identity clustered, 2000 draws)\n{'='*104}")
rng = random.Random(20260727)
boot = {p: [] for p in POLICIES}
for _ in range(2000):
    cache = build_cache(rng)
    for pname, policy in POLICIES.items():
        a, _ = score("keybo-lsb", policy, cache)
        b, _ = score("keybo-lsb+lm", policy, cache)
        boot[pname].append(b - a)

summary = {}
for pname in POLICIES:
    d = sorted(boot[pname])
    lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d))]
    p_lm = sum(1 for x in d if x < 0) / len(d)
    res = lo > 0 or hi < 0
    print(
        f"  {pname}\n      delta {results[pname]['delta']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
        f"   P(+lm better) {p_lm:.3f}   => {'RESOLVED' if res else 'NOT RESOLVED'}"
    )
    summary[pname] = {
        "delta": results[pname]["delta"],
        "ci": [lo, hi],
        "p_lm_better": p_lm,
        "resolved": bool(res),
    }

# ---- cell-level decomposition of P2 --------------------------------------------------
print(f"\n{'='*104}\nCELL DECOMPOSITION of the P2 difference (which classes move it)\n{'='*104}")
contrib = defaultdict(float)
for label, sign in (("keybo-lsb", -1.0), ("keybo-lsb+lm", +1.0)):
    rows, den = MAP[label]
    for freq, ek, ck, _lk, _bg in rows:
        r = point.get(ek, point.get(ck))
        if r is None:
            continue
        contrib[ek] += sign * 100.0 * freq * r / den
print(f"{'explicit cell':<38}{'rel used':>10}{'delta contrib':>15}")
for ek, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:12]:
    r = point.get(ek)
    print(f"{ek:<38}{(f'{r:+.4f}' if r is not None else '  fb'):>10}{v:>+15.4f}")
print(f"{'TOTAL':<38}{'':>10}{sum(contrib.values()):>+15.4f}")

# ---- the decisive split, printed with support --------------------------------------
print(f"\n{'='*104}\nTHE DECISIVE SPLIT: weak-on-TOP dy2 by whether the LOWER key is the index\n{'='*104}")
for key in sorted(k for k in CELLS if k.startswith("X:")):
    c = CELLS[key]
    n_raw = sum(n for bg in c["bigrams"].values() for (_s, n) in bg.values())
    r = rel_of(key)
    print(
        f"  {key:<44} rel {(f'{r:+.4f}' if r is not None else '  n/a  '):>9}  n_raw {n_raw:>9}  "
        f"pids {c['n_pids']:>6}  bigrams {len(c['bigrams']):>3}  "
        f"{'SUPPORTED' if key in SUPPORTED else 'unsupported'}"
    )

json.dump({"results": results, "bootstrap": summary}, open("/tmp/lmscissor_arbiter.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_arbiter.json")
