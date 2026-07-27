"""lmscissor: the empirical arbiter, with uncertainty propagated.

R6 (explicit measured cells) flips the order; R7 (coarse cells) does not. But the coarse cell
that R7 uses to price `bl` (weakTOP|dy2|nonadj = -0.0500) is **99.96% lower-key-is-index** by
sample count, so R7 prices a middle-lower pair with an index-lower number — the exact
generalization this investigation identified as the defect. R6 is the better-targeted estimator
but rests partly on thin cells.

So: build a row-travel cost index directly from the measured surface, and BOOTSTRAP it
(resampling bigram identities within each cell, the clustering BADSCISSOR-1 used) to ask whether
the keybo-lsb vs keybo-lsb+lm difference is resolvable at all.

  index(layout) = sum_bg freq(bg) * rel(cell(bg)) / sum_bg freq(bg)
                  over same-hand distinct-finger bigrams (rel is defined relative to the
                  same-row same-hand two-finger baseline, so same-row pairs contribute 0).

Three fallback policies for cells with insufficient support, reported separately.
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

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}
TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
_ABS = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}
MIN_RAW, MIN_PIDS, MIN_BIGRAMS = 200, 20, 3


def bucket_of(w):
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


# ---- 1. harvest the surface, keeping per-bigram-identity detail so we can bootstrap -----
# explicit cell -> bigram identity -> bucket -> [durations]
explicit: dict[str, dict[str, dict[int, list]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)
# coarse cell (orientation x dy x adjacency) -> same
coarse: dict[str, dict[str, dict[int, list]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)
# lower-finger-matched fallback: (lower_kind, dy) -> same
lowmatch: dict[str, dict[str, dict[int, list]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)
baseline: dict[int, list] = defaultdict(list)
pids_of: dict[str, set] = defaultdict(set)

with open(TSV, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        ngram = parts[2]
        if PUNCT & set(ngram) or " " in ngram:
            continue
        nums = parts[1].replace("(", " ").replace(")", " ").replace(",", " ").split()
        if len(nums) != 4:
            continue
        try:
            ax, ay, bx, by = (int(n) for n in nums)
        except ValueError:
            continue
        if ax == 0 or bx == 0 or (ax > 0) != (bx > 0):
            continue
        fa, fb = _ABS[abs(ax)], _ABS[abs(bx)]
        if fa == fb:
            continue
        samples = []
        for tok in parts[4:]:
            tok = tok.strip()
            if not tok.startswith("("):
                continue
            p = tok.strip("()").split(",")
            if len(p) < 3:
                continue
            try:
                samples.append((int(p[0]), float(p[1]), int(p[2])))
            except ValueError:
                continue
        if not samples:
            continue
        if ay == by:
            for w, d, _p in samples:
                bk = bucket_of(w)
                if bk:
                    baseline[bk].append(d)
            continue
        dy = abs(ay - by)
        lower_kind = fa if ay < by else fb
        upper_kind = fb if ay < by else fa
        weak_lower = _DEX[lower_kind] < _DEX[upper_kind]
        adjacent = abs(abs(ax) - abs(bx)) == 1 or {abs(ax), abs(bx)} == {6, 4}
        ek = f"{lower_kind}|{upper_kind}|dy{dy}"
        ck = f"{'weakLOWER' if weak_lower else 'weakTOP'}|dy{dy}|{'adj' if adjacent else 'nonadj'}"
        lk = f"lower={lower_kind}|dy{dy}"
        for table, key in ((explicit, ek), (coarse, ck), (lowmatch, lk)):
            for w, d, p in samples:
                bk = bucket_of(w)
                if bk:
                    table[key][ngram][bk].append(d)
                    pids_of[key].add(p)

base_ms = {bk: sum(v) / len(v) for bk, v in baseline.items() if len(v) >= MIN_RAW}
print(f"baseline ms: { {k: round(v,2) for k,v in sorted(base_ms.items())} }")


def estimate(table, key, id_subset=None):
    """Relative excess of a cell, pooling the given bigram identities. None if unsupported."""
    per_id = table[key]
    ids = list(per_id) if id_subset is None else id_subset
    agg = defaultdict(list)
    for bid in ids:
        for bk, ds in per_id[bid].items():
            agg[bk].extend(ds)
    rels, n_raw = [], 0
    for bk, ds in agg.items():
        n_raw += len(ds)
        if bk in base_ms and len(ds) >= MIN_RAW:
            rels.append(sum(ds) / len(ds) / base_ms[bk] - 1.0)
    if not rels:
        return None, n_raw, len(set(ids))
    return sum(rels) / len(rels), n_raw, len(set(ids))


def supported(table, key) -> bool:
    if key not in table:
        return False
    rel, n_raw, nbg = estimate(table, key)
    return rel is not None and n_raw >= MIN_RAW and nbg >= MIN_BIGRAMS and len(pids_of[key]) >= MIN_PIDS


# ---- 2. the per-layout bigram -> cell mapping -----------------------------------------
corpus_dir = production_corpus_dir(None)
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"corpus = {corpus_dir.name}")


def kind(x):
    return G.finger(x).value.split("-")[1]


def cells_for(spec):
    """[(freq, explicit_key, coarse_key, lowmatch_key)] for same-hand distinct-finger row travel."""
    lay = Layout(spec, G)
    out = []
    den = 0
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
        lower_kind = ka if a[1] < b[1] else kb
        upper_kind = kb if a[1] < b[1] else ka
        weak_lower = _DEX[lower_kind] < _DEX[upper_kind]
        adjacent = C.is_adjacent(G, a, b)
        out.append(
            (
                freq,
                f"{lower_kind}|{upper_kind}|dy{dy}",
                f"{'weakLOWER' if weak_lower else 'weakTOP'}|dy{dy}|{'adj' if adjacent else 'nonadj'}",
                f"lower={lower_kind}|dy{dy}",
                bg,
            )
        )
    return out, den


mapped = {label: cells_for(spec) for label, spec in LAYOUTS.items()}

POLICIES = {
    "P1 explicit only (unsupported cells -> 0)": "drop",
    "P2 explicit, fall back to COARSE (= R6)": "coarse",
    "P3 explicit, fall back to LOWER-FINGER-matched": "lowmatch",
}


def score(spec_rows, den, policy, rel_cache):
    num = 0.0
    unpriced = 0.0
    for freq, ek, ck, lk, _bg in spec_rows:
        r = rel_cache.get(("e", ek))
        if r is None:
            if policy == "drop":
                unpriced += freq
                continue
            key = ("c", ck) if policy == "coarse" else ("l", lk)
            r = rel_cache.get(key)
            if r is None:
                unpriced += freq
                continue
        num += freq * r
    return 100.0 * num / den, 100.0 * unpriced / den


def build_cache(resample: random.Random | None = None):
    cache = {}
    for tag, table in (("e", explicit), ("c", coarse), ("l", lowmatch)):
        for key in table:
            if not supported(table, key):
                continue
            ids = list(table[key])
            if resample is not None:
                ids = [resample.choice(ids) for _ in ids]
            rel, _n, _b = estimate(table, key, ids)
            if rel is not None:
                cache[(tag, key)] = rel
    return cache


point_cache = build_cache()
print(f"\nsupported cells: explicit {sum(1 for t,_ in point_cache if t=='e')}, "
      f"coarse {sum(1 for t,_ in point_cache if t=='c')}, "
      f"lowmatch {sum(1 for t,_ in point_cache if t=='l')}")

print(f"\n{'='*104}")
print("ROW-TRAVEL COST INDEX from the MEASURED Aalto surface (blend-v1 bigram weights)")
print("units: pp of layout-restricted space-excluded bigram mass, x relative excess. LOWER BETTER.")
print(f"{'='*104}")
print(f"{'policy':<46}{'keybo-lsb':>12}{'keybo-lsb+lm':>14}{'delta':>10}  {'unpriced%':>10}  winner")
results = {}
for pname, policy in POLICIES.items():
    a, ua = score(*mapped["keybo-lsb"], policy, point_cache)
    b, ub = score(*mapped["keybo-lsb+lm"], policy, point_cache)
    win = "keybo-lsb" if a < b else ("keybo-lsb+lm" if b < a else "tie")
    print(f"{pname:<46}{a:>12.4f}{b:>14.4f}{b-a:>+10.4f}  {max(ua,ub):>9.3f}%  {win}")
    results[pname] = {"keybo-lsb": a, "keybo-lsb+lm": b, "delta": b - a, "winner": win}

# ---- 3. bootstrap the DIFFERENCE ------------------------------------------------------
print(f"\n{'='*104}\nBOOTSTRAP of the layout DIFFERENCE (bigram-identity clustered, 1000 draws)\n{'='*104}")
rng = random.Random(20260727)
boot = {p: [] for p in POLICIES}
for _ in range(1000):
    cache = build_cache(rng)
    for pname, policy in POLICIES.items():
        a, _ = score(*mapped["keybo-lsb"], policy, cache)
        b, _ = score(*mapped["keybo-lsb+lm"], policy, cache)
        boot[pname].append(b - a)

boot_summary = {}
for pname in POLICIES:
    d = sorted(boot[pname])
    lo, hi = d[25], d[974]
    frac_neg = sum(1 for x in d if x < 0) / len(d)
    print(
        f"  {pname}\n      delta point {results[pname]['delta']:+.4f}  "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"P(+lm better) = {frac_neg:.3f}  "
        f"=> {'RESOLVED' if (lo>0 or hi<0) else 'NOT RESOLVED'}"
    )
    boot_summary[pname] = {
        "delta": results[pname]["delta"],
        "ci": [lo, hi],
        "p_lm_better": frac_neg,
        "resolved": bool(lo > 0 or hi < 0),
    }

# ---- 4. what drives it: top contributing cells ---------------------------------------
print(f"\n{'='*104}\nCELL-LEVEL DECOMPOSITION of the P2 difference (top movers)\n{'='*104}")
contrib = defaultdict(float)
for label, sign in (("keybo-lsb", -1.0), ("keybo-lsb+lm", +1.0)):
    rows, den = mapped[label]
    for freq, ek, ck, lk, _bg in rows:
        r = point_cache.get(("e", ek), point_cache.get(("c", ck)))
        if r is None:
            continue
        contrib[ek] += sign * 100.0 * freq * r / den
movers = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:14]
print(f"{'explicit cell (lower|upper|dy)':<40}{'rel':>9}{'delta contrib':>15}")
for ek, v in movers:
    r = point_cache.get(("e", ek))
    rs = f"{r:+.4f}" if r is not None else "  fb  "
    print(f"{ek:<40}{rs:>9}{v:>+15.4f}")
print(f"{'TOTAL':<40}{'':>9}{sum(contrib.values()):>+15.4f}")

json.dump({"results": results, "bootstrap": boot_summary}, open("/tmp/lmscissor_arbiter.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_arbiter.json")
