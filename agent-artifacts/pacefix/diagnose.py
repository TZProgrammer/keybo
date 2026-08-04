"""PACEFIX-1 §M-A + §M-B — WHY rho was EXACTLY 1.000000, then what breaks it.

TWO measurements, deliberately on DIFFERENT code paths so neither is an algebraic function of the
other (prereg invariant 5):

  M-A. BOOSTER STRUCTURE. Read the trained booster's own trees (its JSON dump -- NOT the
       prediction path; trees_to_dataframe needs pandas, absent from this venv) and answer the four candidate explanations
       directly: does wpm SPLIT at all (H-SPLIT/(a))? where (H-ROOT/(b))? and -- the actual
       definition of an interaction -- in how many trees does wpm appear BELOW a geometric column
       on the SAME root-to-leaf path?

  M-B. RANK IDENTITY. Re-run GATEFOLDS-1's PREDICTION-invariance instrument (train_bigram_model +
       models/base.to_ms, all in-data position pairs x the 5 bucket midpoints) per ARM. Records
       raw LOGRAT spread, within-bucket rank-identity /5, and rho(b40,b120).

M-A is the diagnosis, M-B is the consequence. The prereg BARS "rank identity BREAKS" as
rho(b40,b120) < 1.000000 AND rank-identical buckets < 5/5.

⚠ POSITIVE CONTROL ON MY OWN INSTRUMENT: the two UNCHANGED arms must reproduce GATEFOLDS-1's
published numbers (served rho 0.793006 @ 1/5; interp-wpm 1.000000 @ 5/5, raw spread 7.777e-02). If
they do not, my instrument is broken and nothing else in this file may be read.

ARMS -- ONE VARIABLE EACH from the interp-wpm baseline (prereg invariant 2):
  served              reference (20c, UNCONSTRAINED -- see prereg C2)
  interp-wpm          the DEAD-1 baseline: 11c, ALL 11 constrained incl. wpm at -1
  interp-wpm-nomono   ONE variable: monotone OFF          (tests H-MONO-BLOCK, my primary)
  interp-wpm-depth6   ONE variable: max_depth 3 -> 6      (tests H-DEPTH, predicted to LOSE)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from _boot import ARTIFACTS, SCRATCH, assert_tree, load_rows_cached, require  # noqa: E402

assert_tree()

import keybo.features.ngram as NG  # noqa: E402
import keybo.features.schema as SCHEMA  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

# Brief-decay defence: assert every symbol I lean on exists on THIS tree (prereg invariant 6).
replacement_frame = require(NG, "replacement_frame")
BIGRAM_FEATURE_NAMES = list(require(SCHEMA, "BIGRAM_FEATURE_NAMES"))
BIGRAM_INTERP_WPM_MONOTONE = require(SCHEMA, "BIGRAM_INTERP_WPM_MONOTONE")

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


GEO = ROW_STAGGERED_31
BUCKETS = [40, 60, 80, 100, 120]
MIDPOINTS = [b + 10.0 for b in BUCKETS]

# ---- prereg C1/C2: record the CONSTRAINT FACTS from the code, before any model exists --------
_, IW_NAMES, IW_MONO, IW_STAMP, _ = replacement_frame("wpm")
IW_NAMES = list(IW_NAMES)
CONSTRAINT_FACTS = {
    "interp_wpm_n_columns": len(IW_NAMES),
    "interp_wpm_names": IW_NAMES,
    "interp_wpm_monotone_tuple": list(IW_MONO),
    "interp_wpm_all_columns_constrained": all(c != 0 for c in IW_MONO),
    "interp_wpm_wpm_index": IW_NAMES.index("wpm"),
    "interp_wpm_wpm_constraint": int(IW_MONO[IW_NAMES.index("wpm")]),
    "schema_tuple_matches_registry": list(BIGRAM_INTERP_WPM_MONOTONE) == list(IW_MONO),
    # prereg C2: the served frame has NO monotone tuple; monotone_constraints is set in exactly
    # one place, inside `if interp:` (train.py:436). Asserted here rather than argued.
    "served_n_columns": len(BIGRAM_FEATURE_NAMES),
    "served_has_monotone_tuple_in_schema": any(
        n.startswith("BIGRAM_MONOTONE") for n in dir(SCHEMA)
    ),
}
assert CONSTRAINT_FACTS["schema_tuple_matches_registry"], "registry/schema disagree on constraints"

print()
print("=" * 100)
print("PREREG C1/C2 — THE CONSTRAINT FACTS, read from code before any model is trained")
print("=" * 100)
print(f"  interp-wpm columns: {len(IW_NAMES)}   monotone tuple: {tuple(IW_MONO)}")
print(f"  ALL 11 columns constrained? {CONSTRAINT_FACTS['interp_wpm_all_columns_constrained']}"
      f"   wpm at index {CONSTRAINT_FACTS['interp_wpm_wpm_index']} -> constraint "
      f"{CONSTRAINT_FACTS['interp_wpm_wpm_constraint']}")
print(f"  served frame has a BIGRAM_MONOTONE tuple in schema.py? "
      f"{CONSTRAINT_FACTS['served_has_monotone_tuple_in_schema']}  "
      f"(=> the served frame trains UNCONSTRAINED; GATEFOLDS-1's 'its OWN monotone constraints' "
      f"is wrong)")

# =============================================================================================
# ARM TABLE — one variable each
# =============================================================================================
ARMS = [
    # label,               interp flag, monotone, extra params,        what varies vs interp-wpm
    ("served", False, True, {}, "reference: 20c served frame (unconstrained)"),
    ("interp-wpm", "wpm", True, {}, "BASELINE (DEAD-1): 11c, all constrained"),
    ("interp-wpm-nomono", "wpm", False, {}, "ONE VAR: monotone OFF"),
    ("interp-wpm-depth6", "wpm", True, {"max_depth": 6}, "ONE VAR: max_depth 3->6"),
]

out: dict = {
    "prereg": "agent-artifacts/pacefix/PACEFIX-preregistration.md",
    "purpose": "M-A booster structure (WHY rho=1.000000) + M-B rank identity per arm",
    "geometry": "ROW_STAGGERED_31",
    "buckets": BUCKETS,
    "bucket_midpoints": MIDPOINTS,
    "constraint_facts": CONSTRAINT_FACTS,
    "reference_values_gatefolds": {
        "served": {"rho_b40_b120": 0.793006, "n_rank_identical": 1},
        "interp-wpm": {"rho_b40_b120": 1.000000, "n_rank_identical": 5,
                       "raw_spread": 7.777e-02},
        "source": "agent-artifacts/gatefolds/invariance.json (base gatefolds 986f3a6)",
    },
    "arms": {},
}

log("loading rows (cached)")
rows = load_rows_cached()
log(f"{len(rows)} rows; layouts {sorted({r.layout for r in rows})}")
EVAL_PAIRS = sorted({tuple(r.positions) for r in rows})
log(f"{len(EVAL_PAIRS)} distinct position pairs present in the data")


def builder_for(flag):
    """(builder, names) for a frame flag; `False` means the served frame."""
    if flag is False:
        return (
            lambda g, p, wpm: bigram_features_from_positions(g, p, wpm=wpm),
            list(BIGRAM_FEATURE_NAMES),
        )
    b, names, _mono, _stamp, _tag = replacement_frame(flag)
    return (lambda g, p, wpm: b(g, p, wpm=wpm)), list(names)


# --- M-A helpers -----------------------------------------------------------------------------
# GEOMETRIC = every column that is not `wpm`. An "interaction path" is the literal thing DEAD-1
# needed: on ONE root-to-leaf path, a wpm split BELOW a geometric split (or vice versa), so the
# leaf value depends on wpm AND geometry jointly rather than as a sum of two main effects.
def booster_structure(model, names):
    """Split/gain attribution + wpm's position in the tree, from the booster's JSON DUMP.

    ⚠ Uses ``get_dump(dump_format="json")`` rather than ``trees_to_dataframe()``: the latter needs
    pandas, which is NOT installed in this venv (measured -- ModuleNotFoundError). The JSON dump is
    also the more direct read: it carries each node's ``split`` NAME and ``depth`` and nests its
    children, so the path walk needs no ID-suffix parsing.
    """
    booster = model._regressor.get_booster()
    # The dump labels splits by NAME only if the booster carries feature names; assert rather than
    # hope, because an unnamed dump would report every split as "f10" and silently find no `wpm`.
    booster.feature_names = list(names)
    trees = [json.loads(t) for t in booster.get_dump(dump_format="json", with_stats=True)]
    n_trees = len(trees)

    gain_sum: dict[str, float] = {}
    split_count: dict[str, int] = {}
    depth_of_wpm_splits: list[int] = []
    trees_with_wpm = 0
    trees_with_wpm_below_geom = 0
    trees_with_geom_below_wpm = 0
    n_interaction_paths = 0
    n_leaves_total = 0

    for tree in trees:
        saw_wpm = False
        saw_wpm_below_geom = False
        saw_geom_below_wpm = False

        # DFS carrying the set of features split on ABOVE the current node on THIS path.
        stack = [(tree, 0, frozenset())]
        while stack:
            nd, depth, above = stack.pop()
            if "leaf" in nd:
                n_leaves_total += 1
                # THE DEFINITION: this leaf's value depends on wpm AND on >=1 geometric column,
                # i.e. a genuine wpm-x-geometry term rather than a sum of two main effects.
                if "wpm" in above and (above - {"wpm"}):
                    n_interaction_paths += 1
                continue
            feat = str(nd["split"])
            gain_sum[feat] = gain_sum.get(feat, 0.0) + float(nd.get("gain", 0.0))
            split_count[feat] = split_count.get(feat, 0) + 1
            if feat == "wpm":
                saw_wpm = True
                depth_of_wpm_splits.append(depth)
                if above - {"wpm"}:  # a geometric split sits above this wpm split
                    saw_wpm_below_geom = True
            elif "wpm" in above:  # a geometric split sits below a wpm split
                saw_geom_below_wpm = True
            nxt = above | {feat}
            for child in nd.get("children", []):
                stack.append((child, depth + 1, nxt))
        trees_with_wpm += int(saw_wpm)
        trees_with_wpm_below_geom += int(saw_wpm_below_geom)
        trees_with_geom_below_wpm += int(saw_geom_below_wpm)

    total_gain = float(sum(gain_sum.values()))
    gain_share = {
        f: {
            "gain": gain_sum[f],
            "gain_share": (gain_sum[f] / total_gain) if total_gain else 0.0,
            "n_splits": split_count[f],
        }
        for f in sorted(gain_sum, key=lambda k: -gain_sum[k])
    }
    has_wpm = "wpm" in names
    wpm_info = {
        "column_present": has_wpm,
        "n_splits": gain_share.get("wpm", {}).get("n_splits", 0),
        "gain_share": gain_share.get("wpm", {}).get("gain_share", 0.0),
    }

    wpm_info.update(
        {
            "n_trees": n_trees,
            "n_trees_with_a_wpm_split": trees_with_wpm,
            "n_trees_wpm_BELOW_a_geometric_split": trees_with_wpm_below_geom,
            "n_trees_geometric_BELOW_wpm": trees_with_geom_below_wpm,
            "n_trees_with_ANY_wpm_geometry_path": trees_with_wpm_below_geom
            + trees_with_geom_below_wpm,
            "n_leaves_total": n_leaves_total,
            "n_leaves_depending_on_wpm_AND_geometry": n_interaction_paths,
            "frac_leaves_interaction": (n_interaction_paths / n_leaves_total)
            if n_leaves_total
            else 0.0,
            "depth_of_wpm_splits_histogram": {
                str(d): int(depth_of_wpm_splits.count(d)) for d in sorted(set(depth_of_wpm_splits))
            },
            "min_depth_of_a_wpm_split": min(depth_of_wpm_splits) if depth_of_wpm_splits else None,
            "max_depth_of_a_wpm_split": max(depth_of_wpm_splits) if depth_of_wpm_splits else None,
        }
    )
    return {"n_trees": n_trees, "total_gain": total_gain, "per_feature": gain_share,
            "wpm": wpm_info}


print()
print("=" * 100)
print("M-A + M-B PER ARM")
print("=" * 100)

for label, flag, mono, extra, varies in ARMS:
    log(f"ARM {label}: train_bigram_model  ({varies})")
    kw = {"interp": flag, "monotone": mono} if flag is not False else {}
    model = train_bigram_model(
        rows, target_wpm=90.0, geometry=GEO, random_state=0, n_jobs=8, **kw, **extra
    )
    names = list(model.metadata.feature_names)
    has_wpm = "wpm" in names
    build, _ = builder_for(flag)

    # ARM IDENTITY from the MODEL, never from my label (present != effective).
    # ⚠ THE KEY IS `interp_frame`, NOT `frame` (train.py:535/558 -- `frame_tag` is STORED under
    # "interp_frame"; "frame" is a key INSIDE it). Reading the wrong key returned {} for every arm,
    # i.e. "no constraints" for a fully-constrained model -- the exact "rc=0 with all-None output is
    # a key-not-present bug, not a measurement" hazard. It was caught only because the assertion
    # below demands the constraints be PRESENT for a mono arm; without it, this arm would have
    # reported "constraints absent" and I would have mis-attributed the rank identity. So: assert
    # the key exists, rather than `.get()`-ing into silence.
    _training = model.metadata.extra.get("training") or {}
    if flag is not False and "interp_frame" not in _training:
        raise SystemExit(
            f"{label}: metadata.extra['training'] has no 'interp_frame' key "
            f"(keys: {sorted(_training)}) -- the frame record moved; fix the driver, do not .get()"
        )
    resolved = _training.get("interp_frame") or {}
    identity = {
        "label": label,
        "one_variable_vs_interp_wpm": varies,
        "n_columns": len(names),
        "feature_version": model.metadata.feature_version,
        "has_wpm_column": has_wpm,
        "monotone_flag_passed": mono,
        "resolved_monotone_constraints": list(resolved.get("monotone_constraints") or ()),
        "resolved_frame_tag": resolved.get("frame"),
        "max_depth_effective": int(model._regressor.get_params()["max_depth"]),
        "extra_params": dict(extra),
    }
    # The whole point of the nomono arm is that the tuple is ABSENT; assert that, don't assume it.
    if flag is not False:
        if mono:
            assert identity["resolved_monotone_constraints"], f"{label}: constraints missing"
        else:
            assert not identity["resolved_monotone_constraints"], f"{label}: constraints PRESENT"

    struct = booster_structure(model, names)

    # ---- M-B: rank identity across buckets --------------------------------------------------
    raw_by_bucket, ms_by_bucket = {}, {}
    for _bucket, mid in zip(BUCKETS, MIDPOINTS, strict=True):
        X = np.vstack([np.asarray(build(GEO, p, mid), dtype=np.float64) for p in EVAL_PAIRS])
        assert X.shape[1] == len(names), f"{label}: {X.shape[1]} cols vs model's {len(names)}"
        raw = model.predict(X)
        ms = model.to_ms(raw, X, None if has_wpm else np.full(len(EVAL_PAIRS), mid))
        raw_by_bucket[_bucket] = np.asarray(raw, dtype=np.float64)
        ms_by_bucket[_bucket] = np.asarray(ms, dtype=np.float64)

    R = np.vstack([raw_by_bucket[b] for b in BUCKETS])
    raw_spread = float((R.max(axis=0) - R.min(axis=0)).max())
    ranks = {b: np.argsort(np.argsort(ms_by_bucket[b])) for b in BUCKETS}
    ref = ranks[BUCKETS[0]]
    identical = {str(b): bool(np.array_equal(ranks[b], ref)) for b in BUCKETS}
    n_ident = sum(identical.values())

    from scipy.stats import spearmanr

    rho_lo_hi = float(spearmanr(ms_by_bucket[BUCKETS[0]], ms_by_bucket[BUCKETS[-1]]).statistic)
    # ALL pairwise bucket rhos -- the prereg reports "within-bucket rho per bucket-pair".
    pairwise = {}
    for i, a in enumerate(BUCKETS):
        for b in BUCKETS[i + 1:]:
            pairwise[f"b{a}_vs_b{b}"] = float(
                spearmanr(ms_by_bucket[a], ms_by_bucket[b]).statistic
            )
    min_pairwise = min(pairwise.values())

    # THE REGISTERED BAR, evaluated here so the verdict cannot drift from the numbers.
    breaks = bool(rho_lo_hi < 1.0 and n_ident < len(BUCKETS))

    out["arms"][label] = {
        "identity": identity,
        "M_A_booster_structure": struct,
        "M_B_rank_identity": {
            "n_eval_pairs": len(EVAL_PAIRS),
            "max_raw_lograt_spread_over_buckets": raw_spread,
            "within_bucket_rank_identical_to_b40": identical,
            "n_buckets_rank_identical": n_ident,
            "spearman_b40_vs_b120_ms": rho_lo_hi,
            "pairwise_bucket_rho": pairwise,
            "min_pairwise_bucket_rho": min_pairwise,
            "RANK_IDENTITY_BREAKS": breaks,
        },
        "ms_mean_per_bucket": {str(b): float(ms_by_bucket[b].mean()) for b in BUCKETS},
    }

    w = struct["wpm"]
    print(f"\n  {label:<19} {len(names):2}c  {varies}")
    print(f"    identity: version={identity['feature_version']}  depth="
          f"{identity['max_depth_effective']}  constraints="
          f"{tuple(identity['resolved_monotone_constraints']) or 'NONE'}")
    print(f"    M-A wpm: present={w['column_present']}  splits={w['n_splits']}  "
          f"gain_share={w['gain_share']:.6f}  trees_with_wpm={w['n_trees_with_a_wpm_split']}"
          f"/{w['n_trees']}")
    print(f"         wpm-below-geometry trees={w['n_trees_wpm_BELOW_a_geometric_split']}  "
          f"geometry-below-wpm trees={w['n_trees_geometric_BELOW_wpm']}  "
          f"interaction leaves={w['n_leaves_depending_on_wpm_AND_geometry']}"
          f"/{w['n_leaves_total']} ({w['frac_leaves_interaction']:.4f})")
    print(f"         wpm split depth histogram: {w['depth_of_wpm_splits_histogram'] or '{}'}")
    print(f"    M-B raw LOGRAT spread={raw_spread:.4e}  rank-identical {n_ident}/5  "
          f"rho(b40,b120)={rho_lo_hi:.6f}  min pairwise rho={min_pairwise:.6f}")
    print(f"         => RANK IDENTITY BREAKS: {breaks}")

# =============================================================================================
# POSITIVE CONTROL: the UNCHANGED arms must reproduce GATEFOLDS-1's published numbers
# =============================================================================================
ref = out["reference_values_gatefolds"]
ctl = {}
for label in ("served", "interp-wpm"):
    got = out["arms"][label]["M_B_rank_identity"]
    exp = ref[label]
    ctl[label] = {
        "rho_expected": exp["rho_b40_b120"],
        "rho_measured": got["spearman_b40_vs_b120_ms"],
        "abs_diff_rho": abs(got["spearman_b40_vs_b120_ms"] - exp["rho_b40_b120"]),
        "n_identical_expected": exp["n_rank_identical"],
        "n_identical_measured": got["n_buckets_rank_identical"],
        "reproduces": bool(
            abs(got["spearman_b40_vs_b120_ms"] - exp["rho_b40_b120"]) < 5e-6
            and got["n_buckets_rank_identical"] == exp["n_rank_identical"]
        ),
    }
out["positive_control_vs_gatefolds"] = ctl
print()
print("=" * 100)
print("POSITIVE CONTROL — do my UNCHANGED arms reproduce GATEFOLDS-1's published numbers?")
print("=" * 100)
for label, c in ctl.items():
    print(f"  {label:<12} rho {c['rho_measured']:.6f} vs published {c['rho_expected']:.6f} "
          f"(|diff| {c['abs_diff_rho']:.2e})  rank-identical {c['n_identical_measured']}"
          f"/{c['n_identical_expected']}  REPRODUCES={c['reproduces']}")
if not all(c["reproduces"] for c in ctl.values()):
    print("  ⚠⚠ INSTRUMENT DISAGREES WITH THE PUBLISHED NUMBERS — nothing above may be read.")

with open(f"{ARTIFACTS}/diagnose.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/diagnose.json")

# =============================================================================================
# WHAT THIS DECIDES — the four candidate explanations, scored from M-A/M-B
# =============================================================================================
base = out["arms"]["interp-wpm"]
bw = base["M_A_booster_structure"]["wpm"]
print()
print("=" * 100)
print("WHAT THIS DECIDES about the FOUR candidate explanations of rho = 1.000000")
print("=" * 100)
print(f"  (a) H-SPLIT 'trees never split on wpm': wpm n_splits={bw['n_splits']}, "
      f"gain_share={bw['gain_share']:.6f} => "
      f"{'SUPPORTED' if bw['n_splits'] == 0 else 'REFUTED (it DOES split)'}")
print(f"  (b) H-ROOT  'wpm splits are a per-bucket SHIFT, not a pair-dependent term': "
      f"interaction leaves={bw['n_leaves_depending_on_wpm_AND_geometry']}/{bw['n_leaves_total']}"
      f" => {'SUPPORTED (no interaction paths)' if bw['n_leaves_depending_on_wpm_AND_geometry'] == 0 else 'paths EXIST -- see (c)'}")
nm = out["arms"]["interp-wpm-nomono"]["M_B_rank_identity"]
d6 = out["arms"]["interp-wpm-depth6"]["M_B_rank_identity"]
print(f"  (c) H-MONO-BLOCK 'the constraint forbids the sign pattern': dropping monotone gives "
      f"rho={nm['spearman_b40_vs_b120_ms']:.6f} @ {nm['n_buckets_rank_identical']}/5 => "
      f"{'SUPPORTED (identity BREAKS)' if nm['RANK_IDENTITY_BREAKS'] else 'REFUTED (identity HOLDS)'}")
print(f"  H-DEPTH (registered prediction: LOSES): depth 6 gives "
      f"rho={d6['spearman_b40_vs_b120_ms']:.6f} @ {d6['n_buckets_rank_identical']}/5 => "
      f"{'identity BREAKS' if d6['RANK_IDENTITY_BREAKS'] else 'identity HOLDS (prediction WON)'}")
print(f"  (d) H-TARGET 'LOGRAT already absorbed the pace structure' => the STRUCTURAL answer: "
      f"{'SURVIVES (no intervention broke it)' if not (nm['RANK_IDENTITY_BREAKS'] or d6['RANK_IDENTITY_BREAKS']) else 'REFUTED as a COMPLETE explanation (something broke it)'}")

os.makedirs(SCRATCH, exist_ok=True)
with open(f"{SCRATCH}/diagnose.sentinel", "w") as fh:
    fh.write("ok\n")
log(f"SENTINEL {SCRATCH}/diagnose.sentinel")
