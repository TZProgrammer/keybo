"""TRIAXIS-1 §6 — WHICH AXIS IS THE TRIGRAM CHANNEL ACTUALLY WORST ON?

Registered at HYBRIDTRI-preregistration.md §6 before any normalization number existed. The parent
has written both of these, and both are its own:

  (i) INTERPFRAME-1 §a: "the trigram frame is where M3 is worst -- 51 split pairs, 3.0465 ms/char"
 (ii) FRAMEDIAG-1 §e1: "the TRIGRAM frame is BETTER RESOLVED than the served bigram frame
      (0.9401 vs 0.7960; largest group 2 vs 4), so FM5's 'trigram is worst' does NOT hold on the
      RESOLUTION axis"

These are two different axes and the word "worst" was doing double duty. This driver settles it
WITH NUMBERS before any frame or tool is built, because the answer decides WHAT to build.

⚠ THE NORMALIZATION IS REGISTERED, ALL THREE WAYS, BEFORE MEASURING (§6.3), because a raw count of
51-vs-7 is partly a COLUMN COUNT: the trigram frame has 46 columns and a bg1_/bg2_ mirror pair for
every placement feature, so it has far more same-property PAIR OPPORTUNITIES than a 20-column
frame. Picking the normalization after seeing the numbers is how a null becomes a finding.

  N1 raw count                    -- what the parent quoted
  N2 per same-property PAIR OPPORTUNITY  -- conflicts / opportunities. THE DECISIVE ONE.
  N3 conflict mass / the channel's own |attribution| mass

VERDICT RULE (§6.4): "worst on the split-pairs axis" iff worse on N2 **AND** N3, not merely N1.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/local/home/zegertho/repos/keybo-wt-hybridtri/agent-artifacts/hybridtri")
from _boot import ARTIFACTS, assert_tree, load_by_path, require  # noqa: E402

assert_tree()

import numpy as np  # noqa: E402

from keybo.analysis import frame_collapse as FC  # noqa: E402
from keybo.analysis import shap_diff as SD  # noqa: E402
from keybo.cli.analyze import _resolve  # noqa: E402
from keybo.features import (  # noqa: E402
    BIGRAM_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    bigram_features_from_positions,
    trigram_features_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402

for _n in ("shap_diff", "block_map"):
    require(SD, _n)

M = load_by_path(
    "interpframe_metrics_axis",
    "/local/home/zegertho/repos/keybo-wt-interpframe/agent-artifacts/interpframe/metrics.py",
)
for _n in ("m3_splitpairs", "same_property_groups", "trigram_same_property", "m1_maxcorr"):
    require(M, _n)

WPM = 90.0
GEO = ROW_STAGGERED_30
PAIR = ("flagship-c3", "graphite")  # INTERPFRAME-1's own pair, so the numbers are comparable
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


out: dict = {
    "prereg": "agent-artifacts/hybridtri/HYBRIDTRI-preregistration.md @ 5a5d3c3 §6",
    "pair": list(PAIR),
    "wpm": WPM,
}

# =========================================================================================
# STEP 1 -- THE RESOLUTION AXIS (model-free, so no target-provenance question arises for the
# COUNTS; see the flag below for the floors).
# =========================================================================================
log("STEP 1: the RESOLUTION axis (model-free structure)")
res = {}
for label, order, fn in (
    ("served bigram (20c)", 2, lambda g, c: bigram_features_from_positions(g, c, wpm=WPM)),
    ("served trigram (46c)", 3, lambda g, c: trigram_features_from_positions(g, c, wpm=WPM)),
):
    r = FC.frame_collapse(fn, GEO, order=order, include_space=True).as_dict()
    res[label] = r
    log(
        f"  {label:<22} cells {r['n_cells']:>6}  cols {r['n_columns']:>3}  "
        f"rows {r['distinct_feature_rows']:>6}  resolution {r['resolution']:.4f}  "
        f"collapsed {r['collapsed_share']:.1%}  largest group {r['largest_group']}"
    )
out["resolution_axis"] = res
b_res, t_res = res["served bigram (20c)"], res["served trigram (46c)"]
out["resolution_axis_verdict"] = {
    "trigram_resolution": t_res["resolution"],
    "bigram_resolution": b_res["resolution"],
    "trigram_is_worse_on_resolution": bool(t_res["resolution"] < b_res["resolution"]),
    "trigram_largest_group": t_res["largest_group"],
    "bigram_largest_group": b_res["largest_group"],
}
log(
    f"  => trigram WORSE on resolution? "
    f"{out['resolution_axis_verdict']['trigram_is_worse_on_resolution']}"
)

# =========================================================================================
# STEP 2 -- THE SPLIT-PAIRS AXIS. Needs attributions, so it runs the SHIPPED shap_diff on the
# SAME pair INTERPFRAME-1 used, both channels, with the SHIPPED grouping.
# =========================================================================================
log("")
log(f"STEP 2: the SPLIT-PAIRS axis -- shap_diff on {PAIR[0]} -> {PAIR[1]}, both channels")
_, lay_a = _resolve(PAIR[0])
_, lay_b = _resolve(PAIR[1])
result = SD.shap_diff(lay_a, lay_b, channel="both", target_wpm=WPM)

CHANNELS = {
    "T2 (served bigram, 20c)": (result.t2, list(BIGRAM_FEATURE_NAMES)),
    "Tcond (served trigram, 46c)": (result.tcond, list(TRIGRAM_FEATURE_NAMES)),
}
log(f"  gap_total {result.gap_total:+.6f}  (decomposed {result.decomposed_share_pct:.1f}%)")

sp = {}
for label, (chan, names) in CHANNELS.items():
    if chan is None:
        raise SystemExit(f"ABORT: channel {label} is None -- shap_diff did not decompose it")
    got = list(chan.feature_names)
    if got != names:
        raise SystemExit(f"ABORT: {label} carries {len(got)} columns, expected {len(names)}")
    attrib = np.array([c.ms_per_char for c in chan.contributions], dtype=np.float64)
    order = [c.feature for c in chan.contributions]
    if order != names:
        # reorder to the schema's order so the grouping lookup is by name, never by position
        idx = {n: i for i, n in enumerate(order)}
        attrib = np.array([attrib[idx[n]] for n in names])
    r = M.m3_splitpairs(names, attrib)
    groups = M.same_property_groups(names)
    # N2's DENOMINATOR: the same-property PAIR OPPORTUNITIES the registered grouping contains.
    # Counted over LIVE columns on the same >= min_abs threshold m3 uses, so numerator and
    # denominator are drawn from the same population -- counting all C(n,2) pairs including dead
    # columns would inflate the trigram denominator with columns that cannot conflict.
    thr = r["min_abs_threshold"]
    by_name = dict(zip(names, attrib, strict=True))
    opportunities = 0
    for g in groups:
        live = [n for n in sorted(g) if abs(by_name.get(n, 0.0)) >= thr]
        opportunities += len(live) * (len(live) - 1) // 2
    total_abs = float(np.abs(attrib).sum())
    sp[label] = {
        "n_columns": len(names),
        "gap_ms_per_char": float(chan.gap),
        "N1_splitpairs_raw": r["splitpairs"],
        "N1_conflict_mass_ms": r["conflict_mass_ms_per_char"],
        "N2_pair_opportunities": opportunities,
        "N2_conflicts_per_opportunity": (r["splitpairs"] / opportunities)
        if opportunities
        else None,
        "N3_total_abs_attrib_ms": total_abs,
        "N3_conflict_mass_share_of_abs": (r["conflict_mass_ms_per_char"] / total_abs)
        if total_abs
        else None,
        "N3_conflict_mass_share_of_gap": (r["conflict_mass_ms_per_char"] / abs(float(chan.gap)))
        if chan.gap
        else None,
        "n_groups": len(groups),
        "top_pairs": r["pairs"][:6],
    }
    s = sp[label]
    log(
        f"  {label:<30} N1 {s['N1_splitpairs_raw']:>3} pairs / {s['N1_conflict_mass_ms']:.4f} ms  "
        f"| N2 {s['N1_splitpairs_raw']}/{s['N2_pair_opportunities']} = "
        f"{s['N2_conflicts_per_opportunity']:.4f}  "
        f"| N3 {s['N3_conflict_mass_share_of_abs']:.4f} of |attrib|, "
        f"{s['N3_conflict_mass_share_of_gap']:.4f} of gap"
    )
out["splitpairs_axis"] = sp

B = sp["T2 (served bigram, 20c)"]
T = sp["Tcond (served trigram, 46c)"]
out["splitpairs_axis_verdict"] = {
    "N1_trigram_worse": bool(T["N1_splitpairs_raw"] > B["N1_splitpairs_raw"]),
    "N1_ratio": T["N1_splitpairs_raw"] / B["N1_splitpairs_raw"] if B["N1_splitpairs_raw"] else None,
    "N2_trigram_worse": bool(T["N2_conflicts_per_opportunity"] > B["N2_conflicts_per_opportunity"]),
    "N2_ratio": (T["N2_conflicts_per_opportunity"] / B["N2_conflicts_per_opportunity"])
    if B["N2_conflicts_per_opportunity"]
    else None,
    "N3_trigram_worse_vs_abs": bool(
        T["N3_conflict_mass_share_of_abs"] > B["N3_conflict_mass_share_of_abs"]
    ),
    "N3_ratio_vs_abs": (T["N3_conflict_mass_share_of_abs"] / B["N3_conflict_mass_share_of_abs"])
    if B["N3_conflict_mass_share_of_abs"]
    else None,
    "N3_trigram_worse_vs_gap": bool(
        T["N3_conflict_mass_share_of_gap"] > B["N3_conflict_mass_share_of_gap"]
    ),
}
v = out["splitpairs_axis_verdict"]
# THE REGISTERED VERDICT RULE (§6.4): N2 AND N3, not merely N1.
v["REGISTERED_trigram_worst_on_splitpairs"] = bool(
    v["N2_trigram_worse"] and v["N3_trigram_worse_vs_abs"]
)
log("")
log("=" * 96)
log("THE AXIS VERDICT (registered rule: worse on N2 AND N3, not merely N1)")
log("=" * 96)
log(
    f"  N1 raw count:            trigram {T['N1_splitpairs_raw']} vs bigram {B['N1_splitpairs_raw']}"
    f"  => trigram worse: {v['N1_trigram_worse']}  ({v['N1_ratio']:.2f}x)"
)
log(
    f"  N2 per opportunity:      trigram {T['N2_conflicts_per_opportunity']:.4f} vs bigram "
    f"{B['N2_conflicts_per_opportunity']:.4f}  => trigram worse: {v['N2_trigram_worse']}  "
    f"({v['N2_ratio']:.2f}x)"
)
log(
    f"  N3 mass / |attrib|:      trigram {T['N3_conflict_mass_share_of_abs']:.4f} vs bigram "
    f"{B['N3_conflict_mass_share_of_abs']:.4f}  => trigram worse: {v['N3_trigram_worse_vs_abs']}  "
    f"({v['N3_ratio_vs_abs']:.2f}x)"
)
log(
    f"  N3 mass / own gap:       trigram {T['N3_conflict_mass_share_of_gap']:.4f} vs bigram "
    f"{B['N3_conflict_mass_share_of_gap']:.4f}  => trigram worse: {v['N3_trigram_worse_vs_gap']}"
)
log("")
log(
    f"  REGISTERED VERDICT -- trigram WORST on split-pairs: "
    f"{v['REGISTERED_trigram_worst_on_splitpairs']}"
)
log(
    f"  REGISTERED VERDICT -- trigram WORST on resolution:  "
    f"{out['resolution_axis_verdict']['trigram_is_worse_on_resolution']}"
)

# H_D was: split-pairs YES, resolution NO.
out["H_D"] = {
    "registered": "split-pairs YES, resolution NO",
    "splitpairs_yes": v["REGISTERED_trigram_worst_on_splitpairs"],
    "resolution_no": not out["resolution_axis_verdict"]["trigram_is_worse_on_resolution"],
    "holds": bool(
        v["REGISTERED_trigram_worst_on_splitpairs"]
        and not out["resolution_axis_verdict"]["trigram_is_worse_on_resolution"]
    ),
}
log(f"  H_D ('split-pairs yes, resolution no') HOLDS: {out['H_D']['holds']}")

# --- §7's MAPPING, evaluated rather than chosen -------------------------------------------
if out["H_D"]["splitpairs_yes"] and out["H_D"]["resolution_no"]:
    build = "TOOL: grouped/Owen-Shapley attribution over blocks"
elif out["resolution_axis_verdict"]["trigram_is_worse_on_resolution"]:
    build = "FRAME: a trigram interp frame would be justified"
else:
    build = "NOTHING: neither claim survives normalization"
out["what_to_build"] = build
log(f"  => §7 mapping says BUILD: {build}")

with open(f"{ARTIFACTS}/axis.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
log(f"wrote {ARTIFACTS}/axis.json")
