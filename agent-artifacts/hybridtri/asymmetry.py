"""TRIAXIS-1 §7 — THE ASYMMETRY: which of a trigram's three keys can the frame describe?

Registered at HYBRIDTRI-preregistration.md §7 before measuring, and pursued regardless of the axis
answer because it is cheap, model-free, and it bears on whether a "fix" would destroy a real
distinction.

TCOND-1 verified the structure: `bg1_* == bigram_features(a,b)` and `bg2_* == bigram_features(b,c)`,
and because a bigram's placement one-hots describe the SECOND key of its pair, `bg1_{bottom,home,
top}` is the trigram's MIDDLE key and `bg2_{...}` its THIRD. It then stated: key `a`'s absolute
placement appears in NO column at all.

⚠ MY REGISTERED PREDICTION IS A WEAKER AND MORE PRECISE CLAIM THAN THE PARENT'S:
`a`'s ABSOLUTE placement (row / finger one-hots) is absent, but `a`'s POSITION is partly
RECOVERABLE from the relational columns (`bg1_dx/dy/distance`, `sg_*`) GIVEN b and c. So the frame
is asymmetric in ABSOLUTE DESCRIPTION while retaining RELATIONAL information.

THE REFUTING OBSERVATION, registered: two cells that differ ONLY in `a` and share a feature row.
If such a pair exists, `a` is genuinely unrecoverable there (the strong claim holds at that cell).
If NONE exists, the strong claim "key a is invisible" is FALSE as stated and mine is the right one.

Three measurements, all model-free:
  A1 -- ABSOLUTE-DESCRIPTION CENSUS. For each of the three keys, which columns are a function of
        that key's absolute position ALONE? Measured by PERTURBATION, not by reading names:
        vary one key over all positions with the other two FIXED, and record which columns move.
  A2 -- RECOVERABILITY. Over the 29791 cells: how many distinct (b,c) pairs have two different `a`
        values sharing an identical feature row? That is the exact set where `a` is unrecoverable.
        Compared against the same question for `b` and for `c`, so "asymmetric" is a comparison.
  A3 -- DOES IT MATTER? Weight the A2 collapse by the corpus, and by the TRUTH's spread within
        each such group (the searchable quantity, EXPLOIT-1's own null-space measure).
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, assert_tree  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import surfaces as SF  # noqa: E402
from keybo.analysis.timecard import default_surface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.features import TRIGRAM_FEATURE_NAMES, trigram_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

WPM = 90.0
CHARS, GEO = SF.C30M, ROW_STAGGERED_30
POS = [*GEO.slots, GEO.space_position]
NP_ = len(POS)
NAMES = list(TRIGRAM_FEATURE_NAMES)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


out: dict = {
    "prereg": "agent-artifacts/hybridtri/HYBRIDTRI-preregistration.md @ 5a5d3c3 §7",
    "n_positions": NP_,
    "n_cells": NP_**3,
    "n_columns": len(NAMES),
}

# =========================================================================================
# A1 -- ABSOLUTE-DESCRIPTION CENSUS, BY PERTURBATION rather than by reading column names.
#
# A column is "a function of key K's absolute position ALONE" iff, holding the OTHER two keys
# fixed, its value is determined by K -- AND that determination does not depend on which values
# the other two are held at. Measured over several distinct held-fixed contexts, because a single
# context could accidentally make a relational column look absolute.
# =========================================================================================
log("A1: absolute-description census by PERTURBATION over 6 held-fixed contexts")
rng = np.random.default_rng(20260804)
CONTEXTS = [tuple(int(i) for i in rng.integers(0, NP_, size=3)) for _ in range(6)]
log(f"  contexts (index triples): {CONTEXTS}")

# For each key slot k and each column j: does column j vary when ONLY key k varies?
varies = {k: np.zeros(len(NAMES), dtype=bool) for k in (0, 1, 2)}
# And: is column j's value the SAME function of key k across contexts? (i.e. absolute in k)
same_fn = {k: np.ones(len(NAMES), dtype=bool) for k in (0, 1, 2)}
for k in (0, 1, 2):
    curves = []
    for ctx in CONTEXTS:
        rows = []
        for p in range(NP_):
            cell = list(ctx)
            cell[k] = p
            rows.append(trigram_features_from_positions(GEO, tuple(POS[i] for i in cell), wpm=WPM))
        curves.append(np.vstack(rows))  # (NP_, ncols)
    stack = np.stack(curves)  # (nctx, NP_, ncols)
    varies[k] = (stack.max(axis=1) - stack.min(axis=1) > 1e-12).any(axis=0)
    # absolute in k == the curve over p is IDENTICAL in every context
    same_fn[k] = np.abs(stack - stack[0:1]).max(axis=(0, 1)) <= 1e-12

census = {}
for k, label in ((0, "a (first)"), (1, "b (middle)"), (2, "c (third)")):
    absolute = [n for j, n in enumerate(NAMES) if varies[k][j] and same_fn[k][j]]
    relational = [n for j, n in enumerate(NAMES) if varies[k][j] and not same_fn[k][j]]
    census[label] = {
        "n_columns_that_vary_with_this_key": int(varies[k].sum()),
        "n_absolute_columns": len(absolute),
        "absolute_columns": absolute,
        "n_relational_columns": len(relational),
        "relational_columns": relational,
    }
    log(
        f"  key {label:<12} varies {int(varies[k].sum()):>3} cols  |  "
        f"ABSOLUTE {len(absolute):>2}  RELATIONAL {len(relational):>3}"
    )
    log(f"      absolute: {absolute}")
out["A1_census"] = census

a_abs = census["a (first)"]["n_absolute_columns"]
out["A1_verdict"] = {
    "a_has_zero_absolute_columns": bool(a_abs == 0),
    "n_absolute_a": a_abs,
    "n_absolute_b": census["b (middle)"]["n_absolute_columns"],
    "n_absolute_c": census["c (third)"]["n_absolute_columns"],
    "n_columns_varying_with_a": census["a (first)"]["n_columns_that_vary_with_this_key"],
}
log(
    f"  => key a has {a_abs} absolute columns; b has "
    f"{census['b (middle)']['n_absolute_columns']}, c has "
    f"{census['c (third)']['n_absolute_columns']}. "
    f"But {census['a (first)']['n_columns_that_vary_with_this_key']} columns DO vary with a."
)

# =========================================================================================
# A2 -- RECOVERABILITY: is `a` actually unrecoverable anywhere?
#
# Registered refuting observation: two cells differing ONLY in `a` that share a feature row.
# Counted for all three key slots, so "asymmetric" is a MEASURED comparison rather than a name
# argument. 29791 x 46 is ~1.4M floats -- fine in memory.
# =========================================================================================
log("")
log("A2: recoverability -- cells differing in ONE key only that share a feature row")
X = np.vstack(
    [
        trigram_features_from_positions(GEO, (pa, pb, pc), wpm=WPM)
        for pa in POS
        for pb in POS
        for pc in POS
    ]
)
log(f"  built the full {X.shape} trigram matrix")
X3 = X.reshape(NP_, NP_, NP_, len(NAMES))

recov = {}
for k, label in ((0, "a (first)"), (1, "b (middle)"), (2, "c (third)")):
    # move axis k to the front, flatten the other two into "context", then count contexts where
    # two distinct k-values share a row
    Y = np.moveaxis(X3, k, 0).reshape(NP_, NP_ * NP_, len(NAMES))
    n_ctx_collapsed = 0
    n_pairs_collapsed = 0
    ctx_flags = np.zeros(NP_ * NP_, dtype=bool)
    for ctx in range(NP_ * NP_):
        u, cnt = np.unique(Y[:, ctx, :], axis=0, return_counts=True)
        if len(u) < NP_:
            n_ctx_collapsed += 1
            ctx_flags[ctx] = True
            n_pairs_collapsed += int(sum(c * (c - 1) // 2 for c in cnt if c > 1))
    recov[label] = {
        "n_contexts": NP_ * NP_,
        "n_contexts_where_this_key_is_UNRECOVERABLE": n_ctx_collapsed,
        "share_of_contexts": n_ctx_collapsed / (NP_ * NP_),
        "n_collapsed_pairs": n_pairs_collapsed,
    }
    log(
        f"  key {label:<12} unrecoverable in {n_ctx_collapsed:>4}/{NP_ * NP_} contexts "
        f"({n_ctx_collapsed / (NP_ * NP_):>6.1%})   collapsed pairs {n_pairs_collapsed}"
    )
out["A2_recoverability"] = recov
out["A2_verdict"] = {
    "a_unrecoverable_share": recov["a (first)"]["share_of_contexts"],
    "b_unrecoverable_share": recov["b (middle)"]["share_of_contexts"],
    "c_unrecoverable_share": recov["c (third)"]["share_of_contexts"],
    # The registered refuting observation: does a pair differing ONLY in `a` and sharing a row
    # actually exist? If NOT, "key a is invisible" is false as stated.
    "refuting_pair_exists_for_a": bool(recov["a (first)"]["n_collapsed_pairs"] > 0),
    "a_is_STRICTLY_worse_than_b_and_c": bool(
        recov["a (first)"]["share_of_contexts"] > recov["b (middle)"]["share_of_contexts"]
        and recov["a (first)"]["share_of_contexts"] > recov["c (third)"]["share_of_contexts"]
    ),
}

# --- A2b: WHAT the collapse actually IS, since A2 came back symmetric and tiny --------------
#
# A2 says all three keys collapse in exactly 1 of 961 contexts. That is a suspiciously equal
# answer, and an equal answer to a question about ASYMMETRY needs its mechanism named rather than
# reported as a share. Two candidates, distinguished here:
#   (i) a genuine three-way symmetry of the frame, or
#  (ii) ONE degenerate context that has nothing to do with the a/b/c question.
# Identified exactly: the collapsing context in all three cases is the SPACE-SPACE context, and
# the collapsed members are LEFT-RIGHT MIRROR pairs (x, y) / (-x, y). Named, not inferred.
log("")
log("A2b: what the one collapsing context IS, and whether the collapse is a MIRROR")
a2b = {}
for k, label in ((0, "a (first)"), (1, "b (middle)"), (2, "c (third)")):
    Y = np.moveaxis(X3, k, 0).reshape(NP_, NP_ * NP_, len(NAMES))
    ctxs = []
    for ctx in range(NP_ * NP_):
        u, cinv, cnt = np.unique(Y[:, ctx, :], axis=0, return_inverse=True, return_counts=True)
        if len(u) == NP_:
            continue
        i1, i2 = divmod(ctx, NP_)
        groups = []
        all_mirror = True
        for g in np.flatnonzero(cnt > 1):
            members = [POS[p] for p in np.flatnonzero(cinv.ravel() == g)]
            # a MIRROR pair is exactly two members with equal row and opposite column
            is_mirror = (
                len(members) == 2
                and members[0][1] == members[1][1]
                and members[0][0] == -members[1][0]
            )
            all_mirror = all_mirror and is_mirror
            groups.append({"members": [list(m) for m in members], "is_mirror_pair": is_mirror})
        ctxs.append(
            {
                "fixed_positions": [list(POS[i1]), list(POS[i2])],
                "both_fixed_are_space": bool(POS[i1] == GEO.space_position)
                and bool(POS[i2] == GEO.space_position),
                "n_groups": len(groups),
                "all_groups_are_mirror_pairs": all_mirror,
                "groups": groups,
            }
        )
    a2b[label] = ctxs
    for c in ctxs:
        log(
            f"  key {label:<12} fixed={c['fixed_positions']}  both_space={c['both_fixed_are_space']}"
            f"  groups={c['n_groups']}  all mirror pairs={c['all_groups_are_mirror_pairs']}"
        )
out["A2b_collapsing_contexts"] = a2b
out["A2b_verdict"] = {
    "every_collapse_is_in_the_space_space_context": bool(
        all(c["both_fixed_are_space"] for cs in a2b.values() for c in cs)
    ),
    "every_collapse_is_a_LEFT_RIGHT_MIRROR_pair": bool(
        all(c["all_groups_are_mirror_pairs"] for cs in a2b.values() for c in cs)
    ),
}
log(
    f"  => ALL collapses sit in the space-space context: "
    f"{out['A2b_verdict']['every_collapse_is_in_the_space_space_context']}"
)
log(
    f"  => ALL collapses are left-right MIRROR pairs: "
    f"{out['A2b_verdict']['every_collapse_is_a_LEFT_RIGHT_MIRROR_pair']}"
)

# --- A2c: the corpus-relevant question -- LETTER keys only ---------------------------------
#
# The A2 sweep ranges every key over all 31 positions INCLUDING the thumb/space slot, and the one
# collapse it finds is a space-space artifact. But no trigram the corpus types has space in two of
# three slots at meaningful mass, and the interp frame's own `_is_letter_key` gate exists precisely
# because space is not a describable key. So the honest form of "can the frame tell key K apart?"
# restricts BOTH the varying key and the fixed context to LETTER keys.
log("")
log("A2c: the same question restricted to LETTER keys (space excluded) -- the corpus-relevant form")
LETTER = [i for i, p in enumerate(POS) if p[0] != 0]
log(f"  {len(LETTER)} letter positions of {NP_} (space at index {NP_ - 1} excluded)")
a2c = {}
for k, label in ((0, "a (first)"), (1, "b (middle)"), (2, "c (third)")):
    Y = np.moveaxis(X3, k, 0)  # (NP_, NP_, NP_, ncols) with the varying key first
    n_ctx = 0
    n_coll = 0
    n_pairs = 0
    for i1 in LETTER:
        for i2 in LETTER:
            n_ctx += 1
            block = Y[LETTER][:, i1, i2, :]
            u, cnt = np.unique(block, axis=0, return_counts=True)
            if len(u) < len(LETTER):
                n_coll += 1
                n_pairs += int(sum(c * (c - 1) // 2 for c in cnt if c > 1))
    a2c[label] = {
        "n_letter_contexts": n_ctx,
        "n_contexts_where_this_key_is_UNRECOVERABLE": n_coll,
        "share_of_contexts": n_coll / n_ctx,
        "n_collapsed_pairs": n_pairs,
    }
    log(
        f"  key {label:<12} unrecoverable in {n_coll:>4}/{n_ctx} LETTER contexts "
        f"({n_coll / n_ctx:>6.1%})   collapsed pairs {n_pairs}"
    )
out["A2c_letter_only"] = a2c
out["A2c_verdict"] = {
    "a_unrecoverable_share_letters": a2c["a (first)"]["share_of_contexts"],
    "b_unrecoverable_share_letters": a2c["b (middle)"]["share_of_contexts"],
    "c_unrecoverable_share_letters": a2c["c (third)"]["share_of_contexts"],
    "any_key_unrecoverable_on_letters": bool(any(v["n_collapsed_pairs"] > 0 for v in a2c.values())),
}
log(
    f"  => ANY key unrecoverable on letter-only contexts: "
    f"{out['A2c_verdict']['any_key_unrecoverable_on_letters']}"
)

# =========================================================================================
# A3 -- DOES IT MATTER? Weight the whole-frame collapse by corpus mass and by the TRUTH's spread.
# =========================================================================================
log("")
log("A3: does the collapse matter? corpus mass + within-group spread of the TRUTH")
surface = default_surface(WPM, None)
T3 = surface.triple_ms_table() if hasattr(surface, "triple_ms_table") else None
if T3 is None:
    raise SystemExit("ABORT: TimeSurface has no triple_ms_table() on this tree")
truth = np.asarray(T3, dtype=np.float64).ravel()
if truth.size != NP_**3:
    raise SystemExit(f"ABORT: triple table has {truth.size} entries, expected {NP_**3}")

tri = {
    k: v
    for k, v in load_frequencies(str(production_corpus_dir(None) / "trigrams.txt")).items()
    if len(k) == 3
}
IDX = {c: i for i, c in enumerate(CHARS)}
IDX[" "] = NP_ - 1
F3 = np.zeros((NP_, NP_, NP_))
for ng, f in tri.items():
    try:
        F3[IDX[ng[0]], IDX[ng[1]], IDX[ng[2]]] += f
    except KeyError:
        continue
w = F3.ravel()

_, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
coll = cnt[inv] > 1
sd = np.zeros(len(cnt))
for g in np.flatnonzero(cnt > 1):
    sd[g] = truth[inv == g].std()
out["A3_impact"] = {
    "distinct_feature_rows": int(len(cnt)),
    "collapsed_cells": int(coll.sum()),
    "collapsed_mass_share": float(w[coll].sum() / w.sum()),
    "searchable_nullspace_ms": float((w * sd[inv]).sum() / w.sum()),
    "max_within_group_truth_spread_ms": float(sd.max()),
    # ⚠ INVARIANT 3 FLAG: `triple_ms_table` = T2 + Tcond, and T2 is produced by the SERVED BIGRAM
    # frame while Tcond is produced by the SERVED TRIGRAM frame -- so this target is SELF-GENERATED
    # w.r.t. the frame being measured. FRAMEDIAG-1 §e3 measured exactly this and found the order-3
    # floor tautological (0/1785 groups with spread). Reported so the number is not read as a
    # two-frame contrast; the COLLAPSE COUNTS above are target-free and unaffected.
    "target_is_self_generated": True,
    "target": "TimeSurface.triple_ms_table() = T2 + Tcond, self-generated wrt BOTH served frames",
}
i = out["A3_impact"]
log(
    f"  rows {i['distinct_feature_rows']}/{NP_**3}  collapsed cells {i['collapsed_cells']}  "
    f"collapsed mass {i['collapsed_mass_share']:.2%}"
)
log(
    f"  searchable null space {i['searchable_nullspace_ms']:.6f} ms   "
    f"max within-group truth spread {i['max_within_group_truth_spread_ms']:.6f} ms"
)
log("  ⚠ target_is_self_generated=True -- the spread numbers are a TAUTOLOGY (FRAMEDIAG-1 §e3)")

log("")
log("=" * 96)
log("THE ASYMMETRY VERDICT")
log("=" * 96)
log(
    f"  A1: key a has {a_abs} ABSOLUTE columns (b: {out['A1_verdict']['n_absolute_b']}, "
    f"c: {out['A1_verdict']['n_absolute_c']}) -- so 'a's absolute placement is absent': "
    f"{out['A1_verdict']['a_has_zero_absolute_columns']}"
)
log(
    f"  A1: but {out['A1_verdict']['n_columns_varying_with_a']} columns DO vary with a "
    f"(relationally)"
)
log(
    f"  A2: a unrecoverable in {out['A2_verdict']['a_unrecoverable_share']:.1%} of (b,c) contexts; "
    f"b in {out['A2_verdict']['b_unrecoverable_share']:.1%}; "
    f"c in {out['A2_verdict']['c_unrecoverable_share']:.1%}"
)
log(
    f"  A2: 'key a is INVISIBLE' as a strong claim -- refuting pair exists: "
    f"{out['A2_verdict']['refuting_pair_exists_for_a']}"
)
log(
    f"  A2: a strictly worse than BOTH b and c: {out['A2_verdict']['a_is_STRICTLY_worse_than_b_and_c']}"
)

with open(f"{ARTIFACTS}/asymmetry.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/asymmetry.json")
