"""One honest per-model-set T2 attribution, with its OWN anchor — shared by every driver.

Why this exists rather than calling ``shap_diff()`` everywhere. ``shap_diff``'s external bars
compare against the SHIPPED 3-seed-mean table (``TimeSurface._T2``), which is exactly right for a
production run and exactly wrong for two things this study needs:

* a SINGLE-seed attribution (M6 seed stability) — one seed's table differs from the 3-seed mean by
  a MODEL difference, ~ms, so the shipped bar refuses it. That refusal is CORRECT behaviour and
  must not be relaxed inside ``shap_diff``; the right move is to give the single-seed run its own
  anchor, which is what this does.
* a NON-production model set (the interp POC), for the same reason.

So this reuses ``shap_diff``'s own LMDI core and ``_shap_tables`` verbatim — the delicate parts are
not reimplemented — and supplies the anchor from the SAME models being attributed. The internal
identities are asserted here at the same bars ``ChannelAttribution.reconciles`` uses, plus the
EXTERNAL tie to those models' own ``predict``-side table (an independent xgboost code path), so a
run through this helper is bar-for-bar as checked as a production one, minus only the ``card()``
tie that is a claim about the shipped surface.
"""

from __future__ import annotations

import numpy as np

from keybo.analysis.shap_diff import _lmdi_channel, _shap_tables
from keybo.features import bigram_features_from_positions, interp_features_from_positions

REL_TOL = 1e-9
ADD_TOL = 1e-5
GAUGE_TOL = 1e-3


def t2_attribution(
    models,
    geometry,
    surface,
    layout_a: str,
    layout_b: str,
    w2_char: np.ndarray,
    covered: int,
    target_wpm: float = 90.0,
    frame: str = "served",
    shuffle_seed: int | None = None,
) -> dict:
    """The T2-channel LMDI attribution for ``models``, anchored on their OWN table.

    ``w2_char`` / ``covered`` are the CHARACTER-space bigram weight (the trigram table's
    first-two-character marginal) and its total mass — the gauge's own T2 weighting, passed in so
    every arm provably shares one weight table rather than each deriving its own.

    Returns a dict with ``names``, ``ms`` (the per-column ms/char attribution, summing to ``gap``),
    ``gap``, and every residual. Raises if an INTERNAL identity fails: an attribution that does not
    sum to the gap it claims to decompose is worse than no attribution.
    """
    slot_a = surface._slot_of(layout_a)
    slot_b = surface._slot_of(layout_b)
    chars = layout_a
    perm_a = np.array([slot_a[c] for c in chars] + [slot_a[" "]], dtype=np.intp)
    perm_b = np.array([slot_b[c] for c in chars] + [slot_b[" "]], dtype=np.intp)

    tables = _shap_tables(models, geometry, target_wpm, 2, frame)
    names = list(tables[5])
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    weighted, ms_a, ms_b, log_a, log_b, resid_log, attrib, d_ms_cells = _lmdi_channel(
        tables, perm_a, perm_b, w2_char, covered, 2, rng
    )
    contributions = weighted.sum(axis=(0, 1))
    w_norm = w2_char / max(covered, 1)
    level_a = float((w_norm * ms_a).sum())
    level_b = float((w_norm * ms_b).sum())
    gap = level_b - level_a

    # --- the INTERNAL bars (exact algebra; they catch an attribution bug) ------------------
    resid_cell = float(
        np.abs(attrib.sum(axis=-1) - d_ms_cells).max() / max(np.abs(d_ms_cells).max(), 1e-300)
    )
    resid_sum = float(abs(float(contributions.sum()) - gap) / max(abs(gap), 1e-300))

    # --- the EXTERNAL bar: THESE models' own predict()-side gap ----------------------------
    # A different xgboost code path (predict, not pred_contribs) over the same cells and weights.
    positions = [*geometry.slots, geometry.space_position]
    n_pos = len(positions)
    featurize = interp_features_from_positions if frame == "interp" else bigram_features_from_positions
    vecs = np.vstack([featurize(geometry, (a, b), wpm=target_wpm) for a in positions for b in positions])
    kw = {"wpm": target_wpm} if frame == "interp" else {}
    own_table = np.mean(
        [m.predict_ms(vecs, **kw).reshape(n_pos, n_pos) for m in models], axis=0
    )
    i_a, i_b = np.ix_(perm_a, perm_a), np.ix_(perm_b, perm_b)
    own_gap = float(((w2_char * own_table[i_b]).sum() - (w2_char * own_table[i_a]).sum()) / max(covered, 1))
    resid_gap_vs_own = float(abs(gap - own_gap))
    resid_table_vs_own = float(np.abs(np.mean(tables[3], axis=0) - own_table).max())

    out = {
        "frame": frame,
        "names": names,
        "ms": contributions,
        "log_a": log_a,
        "log_b": log_b,
        "gap": gap,
        "level_a": level_a,
        "level_b": level_b,
        "own_gap": own_gap,
        "n_models": len(models),
        "shuffled": shuffle_seed is not None,
        "resid_cell_lmdi": resid_cell,
        "resid_feature_sum": resid_sum,
        "resid_gap_vs_own": resid_gap_vs_own,
        "resid_table_vs_own": resid_table_vs_own,
        "resid_additivity": float(tables[4]),
        "resid_log_vs_predict": float(resid_log),
    }
    out["reconciles"] = bool(
        resid_cell <= REL_TOL
        and resid_sum <= REL_TOL
        and resid_gap_vs_own <= GAUGE_TOL
        and float(tables[4]) <= ADD_TOL
        and float(resid_log) <= ADD_TOL
    )
    if not out["reconciles"] and shuffle_seed is None:
        raise AssertionError(
            f"attribution does not reconcile (frame={frame}, n_models={len(models)}): "
            f"cell {resid_cell:.3e} sum {resid_sum:.3e} gap_vs_own {resid_gap_vs_own:.3e} "
            f"additivity {tables[4]:.3e} log {resid_log:.3e}"
        )
    return out


def char_bigram_weight(surface, layout_a: str):
    """``(w2_char, covered)``: the gauge's own T2 weight — the trigram table's first-two-char
    marginal on the layout's charset, and the covered mass that is the ms/char denominator.

    Derived from the TRIGRAM table, never from ``bigrams.txt``: ``triple_ms_table``'s docstring
    records that using the standalone bigram table here is ~1.5e-2 wrong, and shap_diff carries
    that substitution as a FAILING negative control.
    """
    from keybo.analysis.shap_diff import _char_weight_tables

    _w3, w2, covered = _char_weight_tables(surface, layout_a)
    return w2, covered
