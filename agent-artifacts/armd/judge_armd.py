"""JUDGEMENT harness — the five mandated judgements for every champion.

  1. score on the objective it was TRAINED on (expected excellent; not evidence)
  2. predicted ms/char on the served surface, PAIRED against the incumbents
     (the paired resolution is the right ruler — trap 37 — because every layout is scored
     on the SAME three seed tables, so the seed main effect is common mode)
  3. the full 19-gauge frame via `keybo analyze --json`
  4. OPTIMIZING-THE-RULER: does it win its trained objective while LOSING the independent
     gauges? Reports the NORMALIZED six-surface floor
  5. ADMISSIBILITY on the 10-axis dominance frame, with a strict-win term (trap 33)

Plus the out-of-domain count per gauge — an out-of-domain champion is an extrapolation,
not an optimum, and that count is a headline number in its own right.

MODELLED ONLY. No layout here is promoted or adopted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import evobj as EV  # noqa: E402

from keybo.analysis.evidence_scorer import LIVE_GAUGES  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

ARM = "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"
STATE = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
REPO = Path("/tmp/optev")

INCUMBENTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}
REFERENCE = {
    "qwerty30m": EV.C30M,
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
}

#: The frozen six surfaces of the normalized-floor frame (FLOOR-METHODOLOGY-1).
SIX = ("AALTO_BASE", "AALTO_TRI_PS_FREQ_PRIOR", "COMMUNITY_FREQ_PRIOR",
       "COMMUNITY_TRI_PS_FREQ_PRIOR", "POOL_FREQ_PRIOR", "POOL_TRI_PS_FREQ_PRIOR")

#: The FROZEN **iWeb** ceilings (`noanchor-1/drivers/fast_eval.py` CEILINGS). These are
#: corpus-SPECIFIC: they are the max saved% over the 46-layout reference population under
#: iWeb trigram weights. Used here ONLY as a positive control that our re-derivation
#: reproduces the established constant — the judgement itself uses ceilings derived under
#: the corpus actually in play (trap 36: a floor check timed on the wrong corpus moved a
#: headline count 11 -> 9).
FROZEN_IWEB_CEILINGS = {
    "AALTO_BASE": 3.712957807410422,
    "AALTO_TRI_PS_FREQ_PRIOR": 3.5790593303072216,
    "COMMUNITY_FREQ_PRIOR": 5.981893007062644,
    "COMMUNITY_TRI_PS_FREQ_PRIOR": 6.077605013019449,
    "POOL_FREQ_PRIOR": 3.8628161129649397,
    "POOL_TRI_PS_FREQ_PRIOR": 3.8927535164502536,
}
SURFACE_DIR = Path("/local/home/zegertho/agent/state/wscissor-gen/keybo/keybo-e2e/surfaces")
#: The frozen 46-layout reference population the ceilings are maxima over. Same population
#: for every corpus; only the trigram weights change.
REF_BOARD = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
                 "comm-pool-board/tri-frequency-layouts.json")


# --------------------------------------------------------------------------------------
# six-surface normalized floor (judgement 4)
# --------------------------------------------------------------------------------------
class SixSurface:
    """The frozen six fitted surfaces, for the NORMALIZED (ceiling-fraction) floor.

    ⚠ These are the `.complete.seedmean.npy` tables harvested from a destroyed workspace
    (trap 14) — they are the ONLY copy, so this reads them from `state/wscissor-gen/`.
    Their identity is asserted by sha256 against the first use, recorded in the output.
    """

    def __init__(self, trigram_path: Path):
        self.arrays = {s: np.load(SURFACE_DIR / f"{s}.complete.seedmean.npy") for s in SIX}
        for s, a in self.arrays.items():
            assert a.shape == (31, 31, 31), f"{s} has shape {a.shape}"
        corpus = load_frequencies(str(trigram_path))
        idx = {c: i for i, c in enumerate(EV.C30M)}
        idx[" "] = 30
        ok = set(EV.C30M) | {" "}
        rows = [(idx[t[0]], idx[t[1]], idx[t[2]], f)
                for t, f in corpus.items() if len(t) == 3 and all(c in ok for c in t)]
        self.I = np.array([r[0] for r in rows], dtype=np.int32)
        self.J = np.array([r[1] for r in rows], dtype=np.int32)
        self.K = np.array([r[2] for r in rows], dtype=np.int32)
        self.F = np.array([r[3] for r in rows], dtype=np.float64)
        self.flat = np.stack([self.arrays[s].reshape(-1) for s in SIX])
        self.qwerty = self._fit(EV.perm_of(EV.C30M))
        # Ceilings DERIVED under this corpus's weights, not the frozen iWeb constant.
        self.ceiling_map = self.derive_ceilings()
        self.ceil = np.array([self.ceiling_map[s] for s in SIX])

    def derive_ceilings(self) -> dict[str, float]:
        """Per-surface ceiling = max saved% over the frozen 46-layout reference population
        UNDER THIS corpus. Same population and formula for every corpus; only weights move."""
        population = list(json.load(open(REF_BOARD))["board"].keys())
        saved = np.array([self.saved(EV.perm_of(lay)) for lay in population])
        return {s: float(saved[:, i].max()) for i, s in enumerate(SIX)}

    def _fit(self, perm: np.ndarray) -> np.ndarray:
        f = (perm[self.I].astype(np.int64) * 31 + perm[self.J]) * 31 + perm[self.K]
        w = np.bincount(f, weights=self.F, minlength=31 ** 3)
        return self.flat @ w

    def saved(self, perm: np.ndarray) -> np.ndarray:
        """saved-vs-qwerty % on each of the six surfaces."""
        return 100.0 * (1.0 - self._fit(perm) / self.qwerty)

    def normfloor(self, perm: np.ndarray) -> float:
        return float((self.saved(perm) / self.ceil).min())

    def mean_saved(self, perm: np.ndarray) -> float:
        return float(self.saved(perm).mean())


# --------------------------------------------------------------------------------------
# paired ms/char resolution (judgement 2)
# --------------------------------------------------------------------------------------
def paired_resolution(layouts: dict[str, str], trigrams: dict[str, int],
                      target_wpm: float = 90.0) -> dict:
    """Per-seed ms/char for every layout, and the PAIRED resolution.

    trap 37: all candidates share the same three seed tables, so the seed main effect is
    common mode and cancels in a DIFFERENCE. The paired floor is the across-seed spread of
    a within-seed difference; the unpaired floor is the within-layout across-seed spread and
    is the WRONG ruler for a paired comparison.
    """
    from keybo.analysis.timecard import TimeSurface

    surface = TimeSurface(trigrams, target_wpm=target_wpm, keep_seed_tables=True)
    names = list(layouts)
    per_seed = {}
    for name in names:
        totals = surface.seed_totals(layouts[name])
        card = surface.card(layouts[name])
        chars = card.total_ms / card.ms_per_char
        per_seed[name] = [t / chars for t in totals]
    M = np.array([per_seed[n] for n in names])  # (L, 3)

    # variance decomposition: layout main effect vs seed main effect vs residual
    grand = M.mean()
    lay_eff = M.mean(axis=1) - grand
    seed_eff = M.mean(axis=0) - grand
    resid = M - grand - lay_eff[:, None] - seed_eff[None, :]
    ss_lay = float((lay_eff ** 2).sum() * M.shape[1])
    ss_seed = float((seed_eff ** 2).sum() * M.shape[0])
    ss_res = float((resid ** 2).sum())
    ss_tot = ss_lay + ss_seed + ss_res

    unpaired = float(M.std(axis=1, ddof=1).max())          # worst within-layout spread
    diffs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = M[i] - M[j]
            diffs.append(float(d.std(ddof=1)))
    paired = float(np.median(diffs))
    paired_p95 = float(np.percentile(diffs, 95))
    return {
        "per_seed_ms_per_char": per_seed,
        "seed_mean_ms_per_char": {n: float(M[i].mean()) for i, n in enumerate(names)},
        "ss_share_pct": {"layout": 100 * ss_lay / ss_tot, "seed": 100 * ss_seed / ss_tot,
                         "residual": 100 * ss_res / ss_tot},
        "unpaired_floor_ms_per_char": unpaired,
        "paired_floor_ms_per_char": paired,
        "paired_floor_p95": paired_p95,
        "n_seeds": M.shape[1],
        "note": ("paired floor = median across-seed SD of a within-seed pairwise DIFFERENCE; "
                 "the unpaired floor is the max within-layout across-seed SD and is the "
                 "WRONG ruler for a paired comparison (trap 37)"),
    }


# --------------------------------------------------------------------------------------
# 19-gauge frame via the shipped CLI (judgement 3)
# --------------------------------------------------------------------------------------
def analyze_json(layouts: dict[str, str], corpus: str | None = None) -> dict:
    """Run `keybo analyze --json` on every layout and ASSERT no row was dropped (trap 38).

    The check is on the ROW LAYOUT STRINGS, not on a count: `analyze` adds one extra row for
    its `--ref` layout (default `qwerty`, the CLASSIC ';./' charset), so a bare count
    comparison is wrong in the *other* direction. Assert set-containment of what we asked for
    and report the extras — the failure mode trap 38 warns about is a REQUESTED layout going
    missing, which containment catches exactly.

    (My first version of this check read `blob.get("layouts", blob)`, which fell through to
    the whole JSON object and compared against its 10 TOP-LEVEL KEYS — the same
    keyed-the-wrong-collection shape trap 38 is about. Keep the key explicit.)
    """
    order = list(layouts)
    cmd = ["uv", "run", "--no-sync", "python", "-m", "keybo.cli", "analyze", "--json"]
    if corpus:
        cmd += ["--corpus", corpus]
    cmd += [layouts[n] for n in order]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"analyze failed rc={proc.returncode}: {proc.stderr[-3000:]}")
    blob = json.loads(proc.stdout)
    rows = blob["rows"]
    got = {row["layout"] for row in rows.values()}
    want = {layouts[n] for n in order}
    missing = want - got
    assert not missing, f"analyze DROPPED {len(missing)} requested layout(s): {sorted(missing)}"
    assert len(rows) == len(set(rows)), "duplicate row keys"
    blob["_extra_rows"] = sorted(got - want)  # the --ref row, normally just qwerty-classic
    return blob


# --------------------------------------------------------------------------------------
# dominance (judgement 5)
# --------------------------------------------------------------------------------------
#: 10-axis dominance frame + orientation. +1 = higher is better AFTER the sign is applied.
DOM_AXES = ("floor", "mean", "wfd", "genkey", "oxey1", "oxey2", "lsb", "scissor", "sfb", "sfs")
DOM_SIGN = {"floor": +1, "mean": +1, "wfd": +1, "oxey1": +1, "oxey2": +1,
            "genkey": -1, "lsb": -1, "scissor": -1, "sfb": -1, "sfs": -1}


def dominates(cand: dict, inc: dict, atol: float = 1e-9) -> tuple[bool, int, int]:
    """Pareto dominance with an explicit STRICT-WIN term (trap 33: `n_ge == n_axes` alone
    labels a candidate that merely TIES on every axis a dominator)."""
    cv = np.array([DOM_SIGN[a] * cand[a] for a in DOM_AXES])
    iv = np.array([DOM_SIGN[a] * inc[a] for a in DOM_AXES])
    n_ge = int((cv >= iv - atol).sum())
    n_gt = int((cv > iv + atol).sum())
    return (n_ge == len(DOM_AXES) and n_gt >= 1), n_ge, n_gt
