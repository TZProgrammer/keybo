"""lmscissor (c)+(d): score keybo-lsb vs keybo-lsb+lm under

  1. the shipped flat bad-scissor gauge  (the incumbent verdict)
  2. candidate REPAIRS (each a single, named change to the support/weighting)
  3. the MEASURED Aalto surface itself — expected relative excess per bigram, which needs no
     predicate at all and is the closest thing to an empirical arbiter.

Every repair is reported whether or not it flips the order (the brief's requirement).
Bootstrap CIs (bigram-clustered) on the cells the verdict rests on.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from keybo.analysis.bad_scissor import _DEX, bad_scissor  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30 as G  # noqa: E402
from keybo.layout import Layout  # noqa: E402

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}

SURFACE = json.load(open("/tmp/lmscissor_surface.json"))["cells"]
corpus_dir = production_corpus_dir(None)
bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
print(f"corpus = {corpus_dir.name} (bigrams.txt); surface = /tmp/lmscissor_surface.json")


def kind(x: int) -> str:
    return G.finger(x).value.split("-")[1]


def features(lay: Layout, bg: str):
    """(dy, pair, adjacent, weak_is_lower, lower_kind, upper_kind) or None."""
    a, b = lay.pos(bg[0]), lay.pos(bg[1])
    if not C.same_hand(G, a, b) or C.same_finger(G, a, b):
        return None
    dy = abs(a[1] - b[1])
    if dy == 0:
        return None
    ka, kb = kind(a[0]), kind(b[0])
    pair = "-".join(sorted((ka, kb), key=lambda k: -_DEX[k]))
    lower_kind = ka if a[1] < b[1] else kb
    upper_kind = kb if a[1] < b[1] else ka
    weak_is_lower = _DEX[lower_kind] < _DEX[upper_kind]
    return dy, pair, C.is_adjacent(G, a, b), weak_is_lower, lower_kind, upper_kind


# ---- the repairs ---------------------------------------------------------------------
# Each maps features -> weight (0 = not in support).
def w_shipped(f):
    return 1.0 if f[3] else 0.0


def w_r1_drop_dexterity(f):
    """R1: drop the lower-key-dexterity condition entirely — any same-hand row travel counts."""
    return 1.0


def w_r2_dy2_over_dy1(f):
    """R2: keep the shipped support but weight dy=2 above dy=1 (4:1, from the measured ratio
    +0.7044 / +0.1415 ~ 5x; 4 chosen as a round, conservative value)."""
    if not f[3]:
        return 0.0
    return 4.0 if f[0] == 2 else 1.0


def w_r3_all_2row(f):
    """R3: include ALL two-row same-hand reaches regardless of which finger is lower
    (i.e. the 'wide' served support), and nothing else."""
    return 1.0 if f[0] == 2 else 0.0


def w_r4_shipped_plus_nonadj_2row(f):
    """R4: shipped support PLUS the non-adjacent 2-row reaches both incumbents miss,
    regardless of orientation (the user's proposed addition)."""
    if f[3]:
        return 1.0
    return 1.0 if (f[0] == 2 and not f[2]) else 0.0


def w_r5_exclusion_scoped_to_index(f):
    """R5: THE MINIMAL, EVIDENCE-SCOPED REPAIR. Keep the shipped predicate, but restrict the
    weak-on-TOP EXCLUSION to the class it was actually measured on: lower key = INDEX finger.
    A weak-on-top pair whose lower key is middle/ring re-enters the support."""
    dy, _pair, _adj, weak_lower, lower_kind, _upper = f
    if weak_lower:
        return 1.0
    return 0.0 if lower_kind == "index" else 1.0


def w_r6_measured(f):
    """R6: no predicate — weight by the MEASURED relative excess of the fully-explicit
    (lower finger x upper finger x dy) cell. Unmeasured cells fall back to their
    orientation x dy x adjacency aggregate."""
    dy, _pair, adj, weak_lower, lower_kind, upper_kind = f
    key = f"D|lower={lower_kind}|upper={upper_kind}|dy{dy}"
    cell = SURFACE.get(key)
    if cell and cell["status"] == "MEASURED" and cell["rel"] is not None:
        return cell["rel"]
    fb = SURFACE.get(
        f"B|{'weakLOWER' if weak_lower else 'weakTOP'}|dy{dy}|{'adj' if adj else 'nonadj'}"
    )
    return fb["rel"] if fb and fb["rel"] is not None else 0.0


def w_r7_measured_coarse(f):
    """R7: same idea but using only the WELL-MEASURED coarse cells (orientation x dy x
    adjacency) — avoids resting on the thin explicit cells. All 8 cells clear the floors."""
    dy, _pair, adj, weak_lower, _lk, _uk = f
    cell = SURFACE.get(f"B|{'weakLOWER' if weak_lower else 'weakTOP'}|dy{dy}|{'adj' if adj else 'nonadj'}")
    return cell["rel"] if cell and cell["rel"] is not None else 0.0


REPAIRS = [
    ("shipped bad-scissor (flat, weak-lower only)", w_shipped),
    ("R1 drop lower-key-dexterity condition", w_r1_drop_dexterity),
    ("R2 shipped support, dy2 weighted 4x dy1", w_r2_dy2_over_dy1),
    ("R3 all 2-row reaches only (= wide support)", w_r3_all_2row),
    ("R4 shipped + nonadjacent 2-row (any orient)", w_r4_shipped_plus_nonadj_2row),
    ("R5 exclusion scoped to lower-key=index", w_r5_exclusion_scoped_to_index),
    ("R6 MEASURED surface, explicit cells", w_r6_measured),
    ("R7 MEASURED surface, coarse cells only", w_r7_measured_coarse),
]

print(f"\n{'='*104}")
print("SCORES — weighted mass as % of space-excluded layout-restricted bigram mass (blend-v1)")
print("LOWER IS BETTER for every row. 'winner' = the layout with the smaller value.")
print(f"{'='*104}")
print(f"{'gauge / repair':<46}{'keybo-lsb':>12}{'keybo-lsb+lm':>14}{'delta':>10}{'  winner':<16}")

verdicts = {}
for name, wfun in REPAIRS:
    vals = {}
    for label, spec in LAYOUTS.items():
        lay = Layout(spec, G)
        num = 0.0
        den = 0
        for bg, freq in bigrams.items():
            if len(bg) != 2 or " " in bg:
                continue
            if not all(lay.has_key(c) for c in bg):
                continue
            den += freq
            f = features(lay, bg)
            if f is None:
                continue
            w = wfun(f)
            if w:
                num += w * freq
        vals[label] = 100.0 * num / den
    a, b = vals["keybo-lsb"], vals["keybo-lsb+lm"]
    winner = "keybo-lsb" if a < b else ("keybo-lsb+lm" if b < a else "tie")
    flip = "  <-- FLIPS" if winner == "keybo-lsb+lm" else ""
    print(f"{name:<46}{a:>12.4f}{b:>14.4f}{b-a:>+10.4f}  {winner:<14}{flip}")
    verdicts[name] = {"keybo-lsb": a, "keybo-lsb+lm": b, "delta": b - a, "winner": winner}

# ---- bootstrap CI on the decisive measured cells -------------------------------------
print(f"\n{'='*104}\nBOOTSTRAP CIs on the cells the verdict rests on (bigram-clustered, 2000 draws)\n{'='*104}")

TSV = Path("/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv")
PUNCT = frozenset(".,'-;/[]\\=")
BUCKETS = (40, 60, 80, 100, 120)
_ABS = {6: "pinky", 5: "pinky", 4: "ring", 3: "middle", 2: "index", 1: "index"}


def bucket_of(w):
    for b in BUCKETS:
        if b - 10 <= w < b + 10:
            return b
    return None


TARGETS = {
    "lower=pinky|upper=middle|dy1  (what +lm ADDS: `ld`)": ("pinky", "middle", 1),
    "lower=middle|upper=pinky|dy2  (what +lm RELIEVES: `bl`)": ("middle", "pinky", 2),
    "lower=pinky|upper=middle|dy2  (the motivating class)": ("pinky", "middle", 2),
    "lower=middle|upper=index|dy2  (the strongest class)": ("middle", "index", 2),
}

per_bigram = {k: defaultdict(lambda: defaultdict(list)) for k in TARGETS}
baseline = defaultdict(list)
quote_slot_hits = {k: 0 for k in TARGETS}

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
        for tname, (lk, uk, tdy) in TARGETS.items():
            if lower_kind == lk and upper_kind == uk and dy == tdy:
                if 6 in (abs(ax), abs(bx)):
                    quote_slot_hits[tname] += 1
                for w, d, _p in samples:
                    bk = bucket_of(w)
                    if bk:
                        per_bigram[tname][ngram][bk].append(d)

base_ms = {bk: sum(v) / len(v) for bk, v in baseline.items() if len(v) >= 200}
rng = random.Random(20260727)

boot_out = {}
for tname in TARGETS:
    table = per_bigram[tname]
    ids = sorted(table)
    if not ids:
        print(f"  {tname}: NO DATA")
        continue

    def rel_from(id_list):
        agg = defaultdict(list)
        for bid in id_list:
            for bk, ds in table[bid].items():
                agg[bk].extend(ds)
        rels = []
        for bk, ds in agg.items():
            if bk in base_ms and len(ds) >= 200:
                rels.append(sum(ds) / len(ds) / base_ms[bk] - 1.0)
        return sum(rels) / len(rels) if rels else None

    point = rel_from(ids)
    draws = []
    for _ in range(2000):
        samp = [rng.choice(ids) for _ in ids]
        r = rel_from(samp)
        if r is not None:
            draws.append(r)
    draws.sort()
    lo = draws[int(0.025 * len(draws))] if draws else float("nan")
    hi = draws[int(0.975 * len(draws))] if draws else float("nan")
    n_raw = sum(len(d) for bg in table.values() for d in bg.values())
    print(
        f"  {tname}\n      point {point if point is None else f'{point:+.4f}'}   "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]   n_raw {n_raw}   distinct bigrams {len(ids)}   "
        f"|x|=6 rows {quote_slot_hits[tname]}"
    )
    print(f"      bigrams: {ids[:20]}")
    boot_out[tname] = {
        "point": point,
        "ci": [lo, hi],
        "n_raw": n_raw,
        "n_bigrams": len(ids),
        "quote_slot_rows": quote_slot_hits[tname],
    }

json.dump({"verdicts": verdicts, "bootstrap": boot_out}, open("/tmp/lmscissor_repairs.json", "w"), indent=2)
print("\nwrote /tmp/lmscissor_repairs.json")
