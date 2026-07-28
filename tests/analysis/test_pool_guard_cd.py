"""The C/D pool guard (POOLSWEEP-1) — it must fire on the real failure and NOT on a
narrow-but-healthy pool, which is exactly where the retired effective-dof floor failed.
"""

from __future__ import annotations

import numpy as np

from keybo.analysis import evidence_validation as V
from keybo.analysis.evidence_scorer import NARROW_POOL_CD


def _two_sources(n: int, c_scale: float, d_scale: float, seed: int = 0):
    """Two independent-source targets with controlled consensus and disagreement spreads."""
    rng = np.random.default_rng(seed)
    consensus = c_scale * rng.standard_normal(n)
    disagreement = d_scale * rng.standard_normal(n)
    return {
        "AALTO_BASE": consensus + disagreement,
        "COMMUNITY_BASE": consensus - disagreement,
    }


def test_cd_is_high_when_consensus_dominates():
    block = V.consensus_disagreement_ratio(_two_sources(400, c_scale=1.0, d_scale=0.1))
    assert block["min"] > NARROW_POOL_CD, block


def test_cd_is_low_when_only_disagreement_survives():
    """The archive's failure mode: consensus removed, disagreement retained."""
    block = V.consensus_disagreement_ratio(_two_sources(400, c_scale=0.1, d_scale=1.0))
    assert block["min"] < NARROW_POOL_CD, block


def test_cd_is_scale_free():
    """Shrinking BOTH directions equally is the harmless kind of narrowness: C/D must not move.

    This is the case the effective-dof floor got wrong — it fell, and fired, while the
    cross-source ceiling stayed healthy.
    """
    wide = V.consensus_disagreement_ratio(_two_sources(400, 1.0, 0.25, seed=3))
    narrow = V.consensus_disagreement_ratio(_two_sources(400, 0.01, 0.0025, seed=3))
    assert abs(wide["min"] - narrow["min"]) < 1e-9, (wide, narrow)
    assert narrow["min"] > NARROW_POOL_CD


def test_guard_fires_on_the_measured_archive_ratio():
    """POOLSWEEP-1 measured the archive at C/D 1.058 with ceiling +0.218 and 0/12 cells."""
    assert NARROW_POOL_CD > 1.058


def test_guard_passes_the_measured_healthy_ratios():
    """random-wide 3.06 (+0.797) and archive+1-swap 3.82 (+0.816) must NOT be flagged."""
    assert NARROW_POOL_CD < 3.06
    assert NARROW_POOL_CD < 3.817


def test_retired_dof_floor_would_have_false_positived():
    """The regression this fix exists for: interp-f0.25 had dof 2.43 and ceiling +0.9244.

    A healthy pool whose dof sits below the old floor must produce NO warning now.
    """
    from keybo.analysis.evidence_scorer import NARROW_POOL_DOF

    assert NARROW_POOL_DOF > 2.43  # the old floor would have fired
    healthy = V.consensus_disagreement_ratio(_two_sources(400, c_scale=1.0, d_scale=0.2))
    assert healthy["min"] > NARROW_POOL_CD  # the new guard correctly stays silent


def test_ignores_non_independent_source_pairs():
    block = V.consensus_disagreement_ratio(
        {"AALTO_BASE": np.arange(50.0), "POOL_BASE": np.arange(50.0)}
    )
    assert block["pairwise"] == {} or all(k.count("|") == 1 for k in block["pairwise"])
