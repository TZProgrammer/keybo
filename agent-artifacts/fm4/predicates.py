"""FM4 step 2: EXHAUSTIVE predicate comparison, frame column vs same-named gauge.

For each of the four colliding names (``scissor``, ``lsb``, ``redirect``, ``bad_redirect``) plus
the two the parent flagged that are NOT exact-name collisions (``lateral`` vs ``lat-span``,
``inwards``/``outwards``), compare the FRAME's per-ngram predicate against the GAUGE's
per-ngram predicate over the FULL enumeration of ROW_STAGGERED_30 / ROW_STAGGERED_31 -- every
ordered position pair and (for trigram names) every ordered triple. Not a sample: the whole
space, the same bar TCOND-1 met.

The gauge side is taken from the gauge's OWN code path, never re-derived here:
  * ``scissor``  gauge -> keybo.scoring.oxey.pattern_shares' own ``C.is_scissor`` call
  * ``lsb``      gauge -> keybo.analysis.kmstats._is_lsb  (keymeow's, on _Key)
  * ``lsb-narrow`` gauge -> keybo.cli.analyze._narrow_lsb_share's own predicate
  * ``redirect``/``bad_redirect`` gauge -> keybo.analysis.community._v1_pattern (the
    parity-gated oxeylyzer-1 port that oxey._trigram_class delegates to)
  * ``lat-span`` gauge -> keybo.features.classify.lateral_span (LateralSpan's per-cell value)
"""

from __future__ import annotations

import itertools
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import keybo  # noqa: E402
from keybo.analysis import kmstats as KM  # noqa: E402
from keybo.features import classify as C  # noqa: E402
from keybo.features.ngram import (  # noqa: E402
    _placement_row_from_positions,
    _trigram_level_from_positions,
)
from keybo.geometry import ROW_STAGGERED_30, ROW_STAGGERED_31  # noqa: E402
from keybo.scoring.oxey import _trigram_class  # noqa: E402

print("keybo.__file__ =", keybo.__file__)

G30, G31 = ROW_STAGGERED_30, ROW_STAGGERED_31
SLOTS30, SLOTS31 = list(G30.slots), list(G31.slots)
print(f"K30 slots={len(SLOTS30)}  K31 slots={len(SLOTS31)}")

# --- the GAUGE side, each pulled from its own module ---------------------------------------
# kmstats works on its own _Key objects indexed by SLOT (row-major 30). Build the slot->_Key
# map exactly as kmstats._KEYS does, and a position->slot map to line the two universes up.
POS_TO_SLOT30 = {pos: i for i, pos in enumerate(SLOTS30)}


def km_key(pos):
    """kmstats' own _Key for a K30 position (None for a position kmstats has no key for)."""
    slot = POS_TO_SLOT30.get(pos)
    return None if slot is None else KM._KEYS[slot]


def gauge_lsb_keymeow(g, a, b):
    ka, kb = km_key(a), km_key(b)
    if ka is None or kb is None:
        return None  # outside kmstats' 30-slot universe (K31 quote slot)
    return bool(KM._is_lsb(ka, kb))


def gauge_scissor_oxey(g, a, b):
    """oxey.pattern_shares' scissor: literally C.is_scissor (read at oxey.py:242)."""
    return bool(C.is_scissor(g, a, b))


def gauge_lat_span(g, a, b):
    """LateralSpan's per-cell quantity: classify.lateral_span (graded, float)."""
    return C.lateral_span(g, a, b)


def gauge_trigram_class(g, a, b, c):
    """The GAUGE's own trigram class: oxey._trigram_class, which delegates to the
    parity-gated _v1_pattern and rolls its four redirect labels onto redirect/bad_redirect.
    Called EXACTLY as oxey.pattern_shares calls it, so no re-derivation."""
    return _trigram_class(g, a, b, c)


# --- the FRAME side, always through the real row builder ----------------------------------
def frame_col(g, a, b, name):
    return _placement_row_from_positions(g, a, b)[name]


def frame_tri(g, a, b, c, name):
    return _trigram_level_from_positions(g, a, b, c)[name]


results = {}


def report(key, n, disagree, extra=None):
    frac = 0.0 if not n else 100.0 * disagree / n
    verdict = "EQUAL" if disagree == 0 else "DIFFERENT"
    results[key] = {
        "n": n,
        "disagreements": disagree,
        "disagree_pct": frac,
        "verdict": verdict,
        **(extra or {}),
    }
    print(f"  {key:<52} n={n:<6} disagree={disagree:<6} ({frac:6.2f}%)  -> {verdict}")


# ===== BIGRAM-LEVEL NAMES =================================================================
print("\n=== BIGRAM: over ALL ordered position pairs ===")
for gname, g, slots in (("K30", G30, SLOTS30), ("K31", G31, SLOTS31)):
    pairs = [(a, b) for a, b in itertools.product(slots, repeat=2)]
    ordered_distinct = [(a, b) for a, b in pairs if a != b]

    # --- scissor: frame column vs oxey gauge -----------------------------------------------
    d = sum(
        1
        for a, b in pairs
        if bool(frame_col(g, a, b, "scissor")) != gauge_scissor_oxey(g, a, b)
    )
    report(f"{gname} scissor: FRAME vs oxey/comfort gauge", len(pairs), d)

    # --- lsb: frame column vs the keymeow `lsb` gauge (STAT_NAMES, frozen) -----------------
    checked = [(a, b) for a, b in pairs if gauge_lsb_keymeow(g, a, b) is not None]
    d = sum(
        1 for a, b in checked if bool(frame_col(g, a, b, "lsb")) != gauge_lsb_keymeow(g, a, b)
    )
    report(
        f"{gname} lsb: FRAME(is_lsb) vs keymeow gauge(_is_lsb)",
        len(checked),
        d,
        {"skipped_outside_kmstats_universe": len(pairs) - len(checked)},
    )
    # both directions of the subset question
    only_frame = sum(
        1 for a, b in checked if frame_col(g, a, b, "lsb") and not gauge_lsb_keymeow(g, a, b)
    )
    only_gauge = sum(
        1 for a, b in checked if not frame_col(g, a, b, "lsb") and gauge_lsb_keymeow(g, a, b)
    )
    results[f"{gname} lsb: FRAME(is_lsb) vs keymeow gauge(_is_lsb)"].update(
        {"frame_only": only_frame, "gauge_only": only_gauge}
    )
    print(f"      (frame-only firings={only_frame}, gauge-only firings={only_gauge})")

    # --- lsb: frame column vs the `lsb-narrow` gauge (analyze's own) -----------------------
    d = sum(
        1 for a, b in pairs if bool(frame_col(g, a, b, "lsb")) != bool(C.is_lsb(g, a, b))
    )
    report(f"{gname} lsb: FRAME vs lsb-narrow gauge (is_lsb)", len(pairs), d)

    # --- lateral (the column) vs lat-span (the gauge) --------------------------------------
    # Different TYPES (one-hot vs graded), so compare AGREEMENT AS PREDICATES: does
    # `lateral != 0` agree with `lat_span != 0`?
    d = sum(
        1
        for a, b in pairs
        if bool(frame_col(g, a, b, "lateral")) != (gauge_lat_span(g, a, b) > 0.0)
    )
    report(f"{gname} lateral(col) vs lat-span gauge (nonzero agreement)", len(pairs), d)
    # and the graded-vs-onehot value comparison
    dv = sum(1 for a, b in pairs if frame_col(g, a, b, "lateral") != gauge_lat_span(g, a, b))
    results[f"{gname} lateral(col) vs lat-span gauge (nonzero agreement)"][
        "value_disagreements"
    ] = dv
    print(f"      (VALUE disagreements, one-hot vs graded: {dv} of {len(pairs)})")

    # --- where the lat-span geometry ACTUALLY surfaces: dx --------------------------------
    # The parent's claim: lat-span surfaces as `dx`, not `lateral`. Test it: is lat_span a
    # deterministic function of dx alone? (it is dx - neutral, floored, so: check that
    # lat_span > 0 implies dx > 0 and that lat_span is recoverable from dx + finger pair)
    nz_span_dx0 = sum(
        1 for a, b in pairs if gauge_lat_span(g, a, b) > 0.0 and frame_col(g, a, b, "dx") == 0.0
    )
    results[f"{gname}_latspan_vs_dx"] = {
        "latspan_nonzero_but_dx_zero": nz_span_dx0,
        "note": "lat_span = max(0, stagger_adjusted_dx - neutral); dx IS stagger_adjusted_dx",
    }
    print(f"      lat-span>0 while dx==0: {nz_span_dx0}  (dx is the same raw quantity)")

    # --- inwards/outwards swap-invariance (the FM4 'names lie' instance) -------------------
    sw_in = sum(
        1 for a, b in ordered_distinct
        if frame_col(g, a, b, "inwards") != frame_col(g, b, a, "inwards")
    )
    sw_out = sum(
        1 for a, b in ordered_distinct
        if frame_col(g, a, b, "outwards") != frame_col(g, b, a, "outwards")
    )
    results[f"{gname}_inwards_swap"] = {
        "n_ordered_distinct": len(ordered_distinct),
        "inwards_changes_under_swap": sw_in,
        "outwards_changes_under_swap": sw_out,
    }
    print(f"      inwards changes under swap: {sw_in}/{len(ordered_distinct)}; outwards: {sw_out}")

    # is the frame's inwards == the oxey `inroll` gauge? (oxey delegates to C.is_inwards)
    d = sum(1 for a, b in pairs if bool(frame_col(g, a, b, "inwards")) != bool(C.is_inwards(g, a, b)))
    report(f"{gname} inwards: FRAME vs oxey inroll gauge", len(pairs), d)

# ===== TRIGRAM-LEVEL NAMES ================================================================
print("\n=== TRIGRAM: over ALL ordered triples (K30) ===")
# TCOND-1's frame: all 30^3 triples. Report both that and the all-distinct subset.
all_triples = list(itertools.product(SLOTS30, repeat=3))
distinct_triples = [(a, b, c) for a, b, c in all_triples if a != b and b != c and a != c]
print(f"  all triples={len(all_triples)}  all-distinct={len(distinct_triples)}")

for label, triples in (("all 30^3", all_triples), ("all-distinct", distinct_triples)):
    for col in ("redirect", "bad_redirect"):
        d = frame_only = gauge_only = 0
        for a, b, c in triples:
            fv = bool(frame_tri(G30, a, b, c, col))
            cls = gauge_trigram_class(G30, a, b, c)
            # The gauge dispatches EXACTLY ONE class per triple (_TRIGRAM_CLASS is a partition),
            # so `bad_redirect` is its own class and `redirect` is the non-bad one. The frame's
            # `bad_redirect` is NESTED inside its `redirect` (ngram.py: bad = redirect and ...),
            # so the honest gauge-side comparison for `redirect` is the UNION of the two classes
            # -- otherwise the nesting difference, not the predicate, drives every disagreement.
            gv = cls == "bad_redirect" if col == "bad_redirect" else cls in ("redirect", "bad_redirect")
            if fv != gv:
                d += 1
                if fv:
                    frame_only += 1
                else:
                    gauge_only += 1
        report(
            f"{col}: FRAME vs oxey gauge (_trigram_class) [{label}]",
            len(triples),
            d,
            {"frame_only": frame_only, "gauge_only": gauge_only},
        )

# The gauge's redirect classes EXCLUDE any triple with a same-finger constituent (v1 returns
# None for Sfb/Sft). The frame's `redirect` does NOT gate on same-finger -- that is REDIRGATE-1's
# whole finding. Quantify that as the mechanism behind whatever disagreement count appears above.
sf_redirect = sum(
    1
    for a, b, c in all_triples
    if frame_tri(G30, a, b, c, "redirect")
    and (C.same_finger(G30, a, b) or C.same_finger(G30, b, c))
)
results["mechanism_redirect_samefinger"] = {
    "frame_redirect_firings_with_samefinger_constituent": sf_redirect,
    "note": "the gauge (_v1_pattern) returns None for these; the frame counts them",
}
print(f"\n  frame `redirect` firings with a same-finger constituent: {sf_redirect}")

out_path = os.path.join(os.path.dirname(__file__), "predicates.json")
with open(out_path, "w") as fh:
    json.dump({"keybo_file": keybo.__file__, "results": results}, fh, indent=2)
print(f"\nwrote {out_path}")
