"""Fast vectorized evaluator for closure #3 (mixed-operator in-loop NSGA-II).

Reproduces the EXACT board axes used by the clear-winner audit, but fast enough
for an EA inner loop:

  * six-surface modeled speed (AALTO_BASE + 5 candidate surfaces): trigram-freq
    weighted sum over the pre-built 31^3 complete surfaces. FLOOR + MEAN of
    saved-vs-qwerty%.
  * kmstats SFB / SFS / LSB: keymeow-class percentages. kmstats.stats is 21 ms/
    layout (Python triple loop over the corpus). We reduce each to a bilinear
    form over the char->slot permutation: for a fixed 30-char charset the
    per-ngram-kind denominator is a CONSTANT (every layout covers the same
    n-grams), and each metric's per-pair value depends only on the two slots'
    (finger, hand, kind, x, row). So score = 100 * sum_ab F[a,b] * M[slot[a],
    slot[b]] / total, with M a 30x30 slot-pair cost precomputed once.
  * tb_objective scissor / lsb / sfb: bilinear via the ComfortObjective cost
    matrices (already 30x30). We reuse tb's own matrices verbatim.

Everything is checked to zero error against KmStats.stats and
ComfortObjective.values, and against the frozen sibling board, before any search.

MODELED/gauge only. Held-layout tau saturated; Phase-D cancelled.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/local/home/zegertho/repos/keybo")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
SPACE = 30
SURFACES = [
    "AALTO_BASE",
    "AALTO_TRI_PS_FREQ_PRIOR",
    "COMMUNITY_FREQ_PRIOR",
    "COMMUNITY_TRI_PS_FREQ_PRIOR",
    "POOL_FREQ_PRIOR",
    "POOL_TRI_PS_FREQ_PRIOR",
]
HERE = Path(__file__).resolve().parent
SURFACE_DIR = HERE / "surfaces"

# FLOOR-METHODOLOGY-1: ceiling-fraction NORMALIZED floor. The raw min() six-surface
# floor is scale-broken (Community saved% ranges ~1.6x wider, so Community binds the
# raw floor 0/46 times across the frozen board -> raw min() silently discards it). The
# normalized floor divides each surface's saved% by that surface's ceiling (max saved%
# over the frozen 46-layout reference population) so all three data sources participate.
# These ceilings are FROZEN, copied verbatim from the floor3 norm_stats_ref46 (the
# established reference population). test_normfloor.py re-derives them from the frozen
# board and asserts they match to <1e-9, so the constant is auditable, not a bare copy.
CEILINGS = {
    "AALTO_BASE": 3.712957807410422,
    "AALTO_TRI_PS_FREQ_PRIOR": 3.5790593303072216,
    "COMMUNITY_FREQ_PRIOR": 5.981893007062644,
    "COMMUNITY_TRI_PS_FREQ_PRIOR": 6.077605013019449,
    "POOL_FREQ_PRIOR": 3.8628161129649397,
    "POOL_TRI_PS_FREQ_PRIOR": 3.8927535164502536,
}

BIGRAM_PATH = REPO / "data/corpus/bigrams.txt"
TRIGRAM_PATH = REPO / "data/corpus/trigrams.txt"
SKIP_PATH = REPO / "data/corpus/1-skip.txt"
KEYMEOW_PATH = REPO / "data/community/vendored/keymeow-keybo.json.gz"


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------
def load_freq(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as source:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def perm_of(layout: str) -> np.ndarray:
    """char (C30M order) -> slot index, with space fixed at slot 30."""
    if len(layout) != 30 or set(layout) != set(C30M):
        raise ValueError(f"not a C30M permutation: {layout!r}")
    return np.array([layout.index(c) for c in C30M] + [SPACE], dtype=np.int64)


# ---------------------------------------------------------------------------
# trigram objective (six-surface speed)
# ---------------------------------------------------------------------------
def build_objective() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corpus = load_freq(TRIGRAM_PATH)
    ci = {c: i for i, c in enumerate(C30M)}
    ci[" "] = SPACE
    cs = set(C30M) | {" "}
    entries = [
        (ci[t[0]], ci[t[1]], ci[t[2]], f)
        for t, f in corpus.items()
        if len(t) == 3 and all(c in cs for c in t)
    ]
    return (
        np.array([e[0] for e in entries]),
        np.array([e[1] for e in entries]),
        np.array([e[2] for e in entries]),
        np.array([e[3] for e in entries], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# kmstats SFB / SFS / LSB — vectorized slot-pair cost matrices (31x31, space=30)
# ---------------------------------------------------------------------------
def _km_keys():
    from keybo.analysis.kmstats import _KEYS

    return _KEYS


def build_kmstats_matrices() -> dict[str, object]:
    """Precompute the char-pair frequency mass (31x31, C30M+space) and the
    slot-pair 0/1 masks for kmstats sfb/sfs/lsb. Space has no key, so any pair
    touching space contributes 0 (masked out).

    IMPORTANT: the frozen board's kmstats uses the KEYMEOW community corpus
    (keymeow-keybo.json.gz bigrams/skipgrams), NOT the general-English
    bigrams.txt. Verified: fast path matches KmStats.stats to 0.0 only when
    built from keymeow. (test_fast_eval.py::test_kmstats_matches.)"""
    from keybo.analysis.kmstats import _KEYS, _is_lsb

    with gzip.open(KEYMEOW_PATH, "rt") as source:
        km_data = json.load(source)
    bi = {k: v for k, v in km_data["bigrams"].items() if len(k) == 2}
    sk = {k: v for k, v in km_data["skipgrams"].items() if len(k) == 2}

    ci = {c: i for i, c in enumerate(C30M)}
    ci[" "] = SPACE
    size = 31

    def pair_mass(freqs: dict[str, int]) -> np.ndarray:
        """(31x31 char-pair mass over C30M+space). kmstats denominators count
        only n-grams fully on the 30 keys; a char-pair involving space is not on
        a key, so it is excluded from both the metric sums and the total (space
        has no _KEYS entry) — enforced by the on_key mask in kmstats_fast."""
        mass = np.zeros((size, size), dtype=np.float64)
        for ng, f in freqs.items():
            if len(ng) != 2:
                continue
            if ng[0] not in ci or ng[1] not in ci:
                continue
            mass[ci[ng[0]], ci[ng[1]]] += f
        return mass

    bi_mass = pair_mass(bi)
    sk_mass = pair_mass(sk)

    # slot-pair masks over 31 slots (index 30 = space, has no key -> all 0)
    sfb_mask = np.zeros((size, size), dtype=np.float64)  # a is not b and same finger
    lsb_mask = np.zeros((size, size), dtype=np.float64)
    for i in range(30):
        for j in range(30):
            a, b = _KEYS[i], _KEYS[j]
            if a is not b and a.finger == b.finger:
                sfb_mask[i, j] = 1.0
            if _is_lsb(a, b):
                lsb_mask[i, j] = 1.0
    # sfs uses the same same-finger mask as sfb (a is not b and same finger)
    sfs_mask = sfb_mask
    return {
        "bi_mass": bi_mass,  # (31,31) char-pair mass, C30M+space order
        "sk_mass": sk_mass,
        "sfb_mask": sfb_mask,  # (31,31) slot-pair 0/1
        "lsb_mask": lsb_mask,
        "sfs_mask": sfs_mask,
        "ci": ci,
    }


def kmstats_fast(perm: np.ndarray, km: dict) -> dict[str, float]:
    """kmstats sfb/sfs/lsb as percentages, matching KmStats.stats to <1e-9.

    perm: char(C30M)->slot (len 31, space at index 30 -> slot 30).
    For a char-pair (a,b) placed at slots (perm[a], perm[b]), the metric fires
    iff mask[perm[a], perm[b]] == 1. Denominator = mass over char-pairs whose
    BOTH chars are on a key (slot < 30 after placement — all C30M chars are, and
    space maps to slot 30 which is masked)."""
    bi_mass = km["bi_mass"]
    sk_mass = km["sk_mass"]
    # placed masks: value at char-pair (a,b) = mask[perm[a], perm[b]]
    p = perm
    sfb_placed = km["sfb_mask"][p[:, None], p[None, :]]  # (31,31) over char pairs
    lsb_placed = km["lsb_mask"][p[:, None], p[None, :]]
    # on-key char pairs: both chars land on slots 0..29 (space -> slot 30)
    on_key = np.ones((31, 31), dtype=np.float64)
    on_key[SPACE, :] = 0.0
    on_key[:, SPACE] = 0.0
    bi_total = float((bi_mass * on_key).sum())
    sk_total = float((sk_mass * on_key).sum())
    sfb = 100.0 * float((bi_mass * sfb_placed).sum()) / bi_total
    lsb = 100.0 * float((bi_mass * lsb_placed).sum()) / bi_total
    sfs = 100.0 * float((sk_mass * sfb_placed).sum()) / sk_total
    return {"sfb": sfb, "sfs": sfs, "lsb": lsb}


# ---------------------------------------------------------------------------
# tb_objective scissor (and lsb/sfb) — reuse ComfortObjective cost matrices
# ---------------------------------------------------------------------------
def build_tb():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "closure3_tb", "/local/home/zegertho/keybo-e2e/tb_objective.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["closure3_tb"] = mod
    spec.loader.exec_module(mod)
    comfort = mod.ComfortObjective()
    # cost matrices are 31x31 (slots + space); probability is 31x31 char-pair
    return comfort


def tb_scissor_fast(perm: np.ndarray, comfort) -> float:
    prob = comfort._bigram_probability  # (31,31) char-pair prob, C30M+space
    cost = comfort.submetrics["scissor"]._cost  # (31,31) slot-pair
    placed = cost[perm[:, None], perm[None, :]]
    return 100.0 * float((prob * placed).sum())


# ---------------------------------------------------------------------------
# six-surface speed
# ---------------------------------------------------------------------------
class SixSurface:
    def __init__(self, seed_level: str = "mean"):
        self.means = {
            s: np.load(SURFACE_DIR / f"{s}.complete.seedmean.npy") for s in SURFACES
        }
        self.seeds = {
            s: {
                k: np.load(SURFACE_DIR / f"{s}.complete.seed{k}.npy") for k in (0, 1, 2)
            }
            for s in SURFACES
        }
        self.I, self.J, self.K, self.F = build_objective()
        # int32 index arrays for fast batch flat-index construction
        self.I32 = self.I.astype(np.int32)
        self.J32 = self.J.astype(np.int32)
        self.K32 = self.K.astype(np.int32)
        # stacked flat for batch scoring
        self.mean_flat = np.stack([self.means[s].reshape(-1) for s in SURFACES])
        # FLOOR-METHODOLOGY-1: per-surface ceilings aligned to SURFACES order, for the
        # ceiling-fraction normalized floor (norm_s = saved_s / ceiling_s; floor = min).
        self.ceilings = np.array([CEILINGS[s] for s in SURFACES])
        self.qwerty = np.array([self._score_one(perm_of(C30M), self.means[s]) for s in SURFACES])
        self.qwerty_seed = {
            s: np.array([self._score_one(perm_of(C30M), self.seeds[s][k]) for k in (0, 1, 2)])
            for s in SURFACES
        }

    def _score_one(self, perm: np.ndarray, surface: np.ndarray) -> float:
        return float((self.F * surface[perm[self.I], perm[self.J], perm[self.K]]).sum())

    def saved(self, perm: np.ndarray) -> np.ndarray:
        """saved-vs-qwerty% on each of the six surfaces (mean model)."""
        pi = perm[self.I]
        pj = perm[self.J]
        pk = perm[self.K]
        flat = (pi * 31 + pj) * 31 + pk
        gathered = self.mean_flat[:, flat]  # (6, T)
        fit = (gathered * self.F).sum(axis=1)  # (6,)
        return 100.0 * (1.0 - fit / self.qwerty)

    def saved_batch(self, perms: np.ndarray) -> np.ndarray:
        """perms: (B,31) -> (B,6) saved-vs-qwerty%.

        Reduces each layout's 22k trigrams to a 29791-bin frequency histogram
        (per-flat-index mass) then a single (B,29791)@(29791,6) matmul against
        the six flattened surfaces. ~3-4x faster than the (6,B,T) gather and far
        less memory. Verified identical to the gather to <1e-11."""
        B = perms.shape[0]
        p32 = perms.astype(np.int32)
        flat = (p32[:, self.I32] * 31 + p32[:, self.J32]) * 31 + p32[:, self.K32]  # (B,T) int32
        W = np.empty((B, 29791), dtype=np.float64)
        F = self.F
        for b in range(B):
            W[b] = np.bincount(flat[b], weights=F, minlength=29791)
        fit = W @ self.mean_flat.T  # (B,6)
        return 100.0 * (1.0 - fit / self.qwerty)

    def floor_mean(self, perm: np.ndarray) -> tuple[float, float]:
        s = self.saved(perm)
        return float(s.min()), float(s.mean())

    def normfloor(self, perm: np.ndarray) -> float:
        """FLOOR-METHODOLOGY-1 ceiling-fraction NORMALIZED six-surface floor:
        min over the 6 surfaces of saved_s / ceiling_s. This is the HEADLINE floor
        axis for the dominance verdict (the raw min() floor silently discards the
        wider-scale Community surface; normalization lets all three sources vote)."""
        return float((self.saved(perm) / self.ceilings).min())

    def normfloor_batch(self, perms: np.ndarray) -> np.ndarray:
        """perms: (B,31) -> (B,) ceiling-fraction normalized floor. Vectorized via
        saved_batch; identical to normfloor() per-layout."""
        return (self.saved_batch(perms) / self.ceilings).min(axis=1)

    def seed_sd(self, perm: np.ndarray) -> dict[str, float]:
        out = {}
        for s in SURFACES:
            by = np.array(
                [
                    100.0 * (1.0 - self._score_one(perm, self.seeds[s][k]) / self.qwerty_seed[s][k])
                    for k in (0, 1, 2)
                ]
            )
            out[s] = float(by.std())
        return out
