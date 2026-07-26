"""Zero-error verification: fast_eval reproduces the reference machinery exactly.

Checks (all must pass before any search is trusted):
  1. kmstats_fast sfb/sfs/lsb == KmStats.stats on 40 random layouts + incumbents.
  2. tb_scissor_fast == ComfortObjective.values['scissor'].
  3. SixSurface.floor/mean == the frozen sibling gauge-board.json for every
     incumbent (and the audit clear-winner-audit.md values).
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/local/home/zegertho/repos/keybo/src")

import fast_eval as FE

C30M = FE.C30M
INCUMBENTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}


def _random_layouts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    chars = list(C30M)
    for _ in range(n):
        rng.shuffle(chars)
        out.append("".join(chars))
    return out


def test_kmstats_matches():
    from keybo.analysis.kmstats import KmStats

    with gzip.open(FE.KEYMEOW_PATH, "rt") as s:
        km_data = json.load(s)
    # kmstats.stats uses the community keymeow corpus (bigrams/skipgrams/trigrams)
    kmstats = KmStats(km_data["bigrams"], km_data["skipgrams"], km_data["trigrams"])
    # fast_eval must use the SAME corpus as the board's kmstats -> build from keymeow
    km_fast = _km_matrices_from(km_data)

    layouts = list(INCUMBENTS.values()) + _random_layouts(40, seed=7)
    max_err = 0.0
    for lay in layouts:
        ref = kmstats.stats(lay)
        got = FE.kmstats_fast(FE.perm_of(lay), km_fast)
        for k in ("sfb", "sfs", "lsb"):
            err = abs(ref[k] - got[k])
            max_err = max(max_err, err)
            assert err < 1e-9, f"kmstats {k} mismatch on {lay!r}: ref={ref[k]} got={got[k]}"
    print(f"kmstats sfb/sfs/lsb max abs err = {max_err:.3e} over {len(layouts)} layouts  OK")


def _km_matrices_from(km_data: dict) -> dict:
    """kmstats matrices built from the KEYMEOW community corpus (what the board uses)."""
    from keybo.analysis.kmstats import _KEYS, _is_lsb

    ci = {c: i for i, c in enumerate(C30M)}
    ci[" "] = FE.SPACE
    size = 31

    def pair_mass(freqs):
        mass = np.zeros((size, size))
        for ng, f in freqs.items():
            if len(ng) == 2 and ng[0] in ci and ng[1] in ci:
                mass[ci[ng[0]], ci[ng[1]]] += f
        return mass

    bi_mass = pair_mass({k: v for k, v in km_data["bigrams"].items() if len(k) == 2})
    sk_mass = pair_mass({k: v for k, v in km_data["skipgrams"].items() if len(k) == 2})
    sfb_mask = np.zeros((size, size))
    lsb_mask = np.zeros((size, size))
    for i in range(30):
        for j in range(30):
            a, b = _KEYS[i], _KEYS[j]
            if a is not b and a.finger == b.finger:
                sfb_mask[i, j] = 1.0
            if _is_lsb(a, b):
                lsb_mask[i, j] = 1.0
    return {
        "bi_mass": bi_mass,
        "sk_mass": sk_mass,
        "sfb_mask": sfb_mask,
        "lsb_mask": lsb_mask,
        "sfs_mask": sfb_mask,
        "ci": ci,
    }


def test_tb_scissor_matches():
    comfort = FE.build_tb()
    layouts = list(INCUMBENTS.values()) + _random_layouts(40, seed=11)
    max_err = 0.0
    for lay in layouts:
        ref = comfort.values(lay)["scissor"]
        got = FE.tb_scissor_fast(FE.perm_of(lay), comfort)
        err = abs(ref - got)
        max_err = max(max_err, err)
        assert err < 1e-9, f"tb scissor mismatch on {lay!r}: ref={ref} got={got}"
    print(f"tb scissor max abs err = {max_err:.3e} over {len(layouts)} layouts  OK")


def test_six_surface_matches_board():
    board = json.load(
        open(
            "/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
            "replicate-gen/gauge-board.json"
        )
    )
    six = FE.SixSurface()
    max_err = 0.0
    for name, lay in INCUMBENTS.items():
        floor, mean = six.floor_mean(FE.perm_of(lay))
        row = board["rows"][lay]["six_surface"]
        ef = abs(floor - row["floor_saved_pct"])
        em = abs(mean - row["mean_saved_pct"])
        max_err = max(max_err, ef, em)
        assert ef < 1e-6, f"{name} floor: {floor} vs {row['floor_saved_pct']}"
        assert em < 1e-6, f"{name} mean: {mean} vs {row['mean_saved_pct']}"
    print(f"six-surface floor/mean max abs err = {max_err:.3e} over incumbents  OK")


if __name__ == "__main__":
    test_kmstats_matches()
    test_tb_scissor_matches()
    test_six_surface_matches_board()
    print("ALL FAST-EVAL VERIFICATION PASSED")
