"""TASK-1, part 2: injectivity is not learnability. HOW is key a's absolute position carried?

Part 1 found the served 46-col row is INJECTIVE enough that a's row/finger never collides.
But a lookup-table injection is not the same as a feature a depth-3 tree can use. Three
sharper questions:

  Q1. WHICH columns carry it? Fit a depth-3 decision tree (the served depth) to predict a's
      row (3-class) and a's finger (4-class) from the 46 served columns. Report accuracy.
      Then repeat at depth 6 and unlimited, to separate "information present" from
      "information reachable at the served capacity".
  Q2. Is it carried by the CONTINUOUS columns acting as a fingerprint, or by an honest
      structural signal? Re-run Q1 with the continuous geometry columns (dx/dy/distance/
      angle, sg_*) DROPPED, leaving only the discrete predicates + one-hots.
  Q3. The parent's reflection-ambiguity claim: is a's row pinned by (bg1_dy, b's row) alone,
      or does it need more? Test the specific 2-column subset the parent named, and then
      the subset including the SIGNED bg1_angle.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from keybo.features import classify as C
from keybo.features.ngram import _trigram_row_from_positions
from keybo.features.schema import TRIGRAM_FEATURE_NAMES
from keybo.geometry import ROW_STAGGERED_31

G = ROW_STAGGERED_31
SLOTS = [*G.slots, G.space_position]
NAMES = TRIGRAM_FEATURE_NAMES

CONTINUOUS = [
    "sg_dx", "sg_dy", "sg_distance",
    "bg1_dx", "bg1_dy", "bg1_distance", "bg1_angle",
    "bg2_dx", "bg2_dy", "bg2_distance", "bg2_angle",
]


def row_class(p):
    return {1: 0, 2: 1, 3: 2}.get(p[1], 3)  # 3 = space row (y=0)


def finger_class(p):
    ax = abs(p[0])
    if ax in (5, 6):
        return 0
    if ax == 4:
        return 1
    if ax == 3:
        return 2
    if ax in (1, 2):
        return 3
    return 4  # thumb / space


def build():
    X, ya_row, ya_fing = [], [], []
    for a, b, c in itertools.product(SLOTS, repeat=3):
        if a == b or b == c:
            continue
        row = _trigram_row_from_positions(G, a, b, c, 0.0)
        X.append([row[n] for n in NAMES])
        ya_row.append(row_class(a))
        ya_fing.append(finger_class(a))
    return np.array(X), np.array(ya_row), np.array(ya_fing)


def fit_acc(X, y, depth, seed=0):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    clf.fit(X, y)
    return float(clf.score(X, y)), clf


def top_feats(clf, names, k=6):
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:k]
    return [{"col": names[i], "importance": round(float(imp[i]), 4)} for i in order if imp[i] > 0]


def main():
    X, y_row, y_fing = build()
    out = {"n": int(X.shape[0]), "n_cols": int(X.shape[1]), "served_depth": 3}

    # Q1 -- full frame, three capacities
    out["Q1_full_frame"] = {}
    for depth in (3, 6, None):
        acc_r, clf_r = fit_acc(X, y_row, depth)
        acc_f, clf_f = fit_acc(X, y_fing, depth)
        out["Q1_full_frame"][f"depth_{depth}"] = {
            "a_row_train_acc": round(acc_r, 4),
            "a_finger_train_acc": round(acc_f, 4),
            "a_row_top_cols": top_feats(clf_r, NAMES),
            "a_finger_top_cols": top_feats(clf_f, NAMES),
        }
    # majority-class baselines
    out["baselines"] = {
        "a_row_majority": round(float(np.bincount(y_row).max() / len(y_row)), 4),
        "a_finger_majority": round(float(np.bincount(y_fing).max() / len(y_fing)), 4),
    }

    # Q2 -- discrete only (drop the continuous geometry fingerprint)
    keep = [i for i, n in enumerate(NAMES) if n not in CONTINUOUS]
    Xd = X[:, keep]
    names_d = [NAMES[i] for i in keep]
    out["Q2_discrete_only"] = {"n_cols": len(keep), "dropped": CONTINUOUS}
    for depth in (3, 6, None):
        acc_r, clf_r = fit_acc(Xd, y_row, depth)
        acc_f, clf_f = fit_acc(Xd, y_fing, depth)
        out["Q2_discrete_only"][f"depth_{depth}"] = {
            "a_row_train_acc": round(acc_r, 4),
            "a_finger_train_acc": round(acc_f, 4),
            "a_row_top_cols": top_feats(clf_r, names_d),
        }

    # Q3 -- the parent's specific reflection claim
    def sub(cols):
        idx = [NAMES.index(c) for c in cols]
        return X[:, idx], cols

    for label, cols in (
        ("bg1_dy_plus_b_row", ["bg1_dy", "bg1_bottom", "bg1_home", "bg1_top"]),
        ("bg1_dy_plus_b_row_plus_angle", ["bg1_dy", "bg1_bottom", "bg1_home", "bg1_top", "bg1_angle"]),
        ("bg1_dy_b_row_angle_sgdy_c_row", [
            "bg1_dy", "bg1_bottom", "bg1_home", "bg1_top", "bg1_angle",
            "sg_dy", "bg2_bottom", "bg2_home", "bg2_top",
        ]),
    ):
        Xs, _ = sub(cols)
        accs = {}
        for depth in (3, None):
            a, _ = fit_acc(Xs, y_row, depth)
            accs[f"depth_{depth}"] = round(a, 4)
        # exact-collision count on this subset alone
        from collections import defaultdict
        bk = defaultdict(set)
        for i in range(Xs.shape[0]):
            bk[tuple(np.round(Xs[i], 12))].add(int(y_row[i]))
        amb = sum(1 for v in bk.values() if len(v) > 1)
        out.setdefault("Q3_reflection", {})[label] = {
            "cols": cols,
            "a_row_acc": accs,
            "distinct_subset_rows": len(bk),
            "subset_rows_ambiguous_in_a_row": amb,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
