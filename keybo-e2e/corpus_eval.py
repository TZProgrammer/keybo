"""GEN-ON-BLEND — corpus-parameterized board evaluator.

Re-points the wider-dominance in-loop evaluator (island NSGA-II, normalized floor,
mixed operators) at an arbitrary corpus directory, so the SAME search can be run with
the gauge corpus = `data/corpus` (iWeb) or `data/corpus/blend-v1`.

WHAT MOVES UNDER A CORPUS SWAP (audited in source + verified by probes/plumb_probe.py;
see PREREG-gen-on-blend.md §3 — this is the load-bearing correction to the brief):

  * floor / mean  <- trigram weights (`<corpus>/trigrams.txt`)              MOVES
  * scissor       <- tb ComfortObjective._bigram_probability                 MOVES
                     (`<corpus>/bigrams.txt`)
  * lsb / sfb     <- KmStats bigrams. The frozen board tables these from the KEYMEOW
    / sfs           vendored corpus, NOT data/corpus. ARM-A keeps keymeow (control);
                    ARM-B re-tables them to the corpus (primary — the multi-source
                    user's board, and the only arm where the sfb blocker can move).
  * wfd / genkey / oxey1 / oxey2 — each community tool scores on its OWN vendored
                    corpus and takes no corpus argument. IMMOVABLE under any arm.

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. No realized or
observed speed/ranking claim. The model is held FIXED; only the corpus varies.
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per process: the EA parallelizes across islands with a process
# pool, so per-worker BLAS threading only oversubscribes. Must precede numpy import.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CLONE = HERE.parent
if str(CLONE / "src") not in sys.path:
    sys.path.insert(0, str(CLONE / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
SPACE = 30
NSLOT = 31
SURFACES = [
    "AALTO_BASE",
    "AALTO_TRI_PS_FREQ_PRIOR",
    "COMMUNITY_FREQ_PRIOR",
    "COMMUNITY_TRI_PS_FREQ_PRIOR",
    "POOL_FREQ_PRIOR",
    "POOL_TRI_PS_FREQ_PRIOR",
]
SURFACE_DIR = HERE / "surfaces"
KEYMEOW_PATH = CLONE / "data/community/vendored/keymeow-keybo.json.gz"

#: corpus directories. "iweb" = the production single-source tables; "blend" = blend-v1;
#: "noanchor" = blend-v1-no-anchor, the fully reproducible variant (NO-ANCHOR-1).
#: no-anchor drops the 50%-weight non-redistributable iWeb anchor and renormalizes the
#: remaining registers to prose 0.50 / code 0.30 / reference 0.20 (2x their blend-v1
#: shares) — done by build_corpus.blend_tables' `weight / weight_mass`, not by hand.
#: REHUNT: `noanchor` is not committed on this branch (it lives on `blend-no-anchor`), so it is
#: staged under keybo-e2e/corpora/ with its md5s recorded in the run manifest. Named explicitly
#: on every run — `data/corpus` is blend-v1 by default on some branches (CORPUS-SWAP-1).
CORPUS_DIRS = {
    "iweb": CLONE / "data/corpus",
    "blend": CLONE / "data/corpus/blend-v1",
    "noanchor": HERE / "corpora/blend-v1-no-anchor",
}
#: display labels for the three corpora (report/table headers)
CORPUS_LABELS = {
    "iweb": "iWeb",
    "blend": "blend-v1",
    "noanchor": "blend-v1-no-anchor",
}
#: which skipgram file each corpus dir uses for kmstats' sfs.
#: The frozen board's kmstats uses keymeow skipgrams; for a corpus-tabled sfs we use
#: 1-skip31.txt, the marginalized convention blend-v1 reproduces byte-exactly (see
#: blend-v1/PROVENANCE.md §4) and the one the other harnesses read.
SKIP_NAME = "1-skip31.txt"

#: The 46-layout reference population for the normalized-floor ceilings. Frozen board;
#: the SAME population is reused for every corpus so only the corpus weights change.
REF_BOARD = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
    "comm-pool-board/tri-frequency-layouts.json"
)
#: Frozen iWeb ceilings (wider-dominance fast_eval.CEILINGS) — used only as a positive
#: control that our re-derivation reproduces the established constant.
FROZEN_IWEB_CEILINGS = {
    "AALTO_BASE": 3.712957807410422,
    "AALTO_TRI_PS_FREQ_PRIOR": 3.5790593303072216,
    "COMMUNITY_FREQ_PRIOR": 5.981893007062644,
    "COMMUNITY_TRI_PS_FREQ_PRIOR": 6.077605013019449,
    "POOL_FREQ_PRIOR": 3.8628161129649397,
    "POOL_TRI_PS_FREQ_PRIOR": 3.8927535164502536,
}

INCUMBENTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
}

#: The 10 dominance axes and their orientation (+1 = higher better after sign).
AXES = ["floor", "mean", "wfd", "genkey", "oxey1", "oxey2", "lsb", "scissor", "sfb", "sfs"]
SIGN = {
    "floor": +1,
    "mean": +1,
    "wfd": +1,  # stored negative (cost as neg): higher = better
    "oxey1": +1,
    "oxey2": +1,
    "genkey": -1,
    "lsb": -1,
    "scissor": -1,
    "sfb": -1,
    "sfs": -1,
}
#: axes that no corpus change can move (each tool scores on its own vendored corpus)
INVARIANT_AXES = ("wfd", "genkey", "oxey1", "oxey2")


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------
def load_freq(path: Path) -> dict[str, int]:
    """`ngram<TAB>count` table. Mirrors keybo.data.corpus.load_frequencies' int() parse."""
    out: dict[str, int] = {}
    with open(path) as source:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return out


def corpus_tables(corpus: str) -> tuple[dict, dict, dict]:
    """(bigrams, skipgrams, trigrams) for a named corpus."""
    directory = CORPUS_DIRS[corpus]
    return (
        load_freq(directory / "bigrams.txt"),
        load_freq(directory / SKIP_NAME),
        load_freq(directory / "trigrams.txt"),
    )


def perm_of(layout: str) -> np.ndarray:
    """char (C30M order) -> slot index, space pinned at slot 30. Rejects non-perms."""
    if len(layout) != 30 or set(layout) != set(C30M):
        raise ValueError(f"not a C30M permutation: {layout!r}")
    return np.array([layout.index(c) for c in C30M] + [SPACE], dtype=np.int64)


def movable_to_layout(movable) -> str:
    """movable[c] = slot of C30M char c -> the 30-char layout string."""
    slots = [""] * 30
    for c, slot in enumerate(movable):
        slots[slot] = C30M[c]
    return "".join(slots)


# ---------------------------------------------------------------------------
# kmstats sfb / sfs / lsb — bilinear slot-pair form over ANY bigram/skip tabling
# ---------------------------------------------------------------------------
def build_kmstats_matrices(bigrams: dict, skipgrams: dict) -> dict:
    """Precompute the (31,31) char-pair mass and slot-pair 0/1 masks for kmstats
    sfb/sfs/lsb under an arbitrary tabling.

    For a fixed 30-char charset the per-kind denominator is a CONSTANT (every layout
    covers the same n-grams), and each metric's per-pair value depends only on the two
    slots — so score = 100 * sum_ab F[a,b] * M[slot[a], slot[b]] / total. Space has no
    key, so any pair touching space contributes to neither the metric nor the total.

    Verified to 0.0 against KmStats.stats for BOTH the keymeow and the corpus tabling
    (test_corpus_eval.py::test_kmstats_fast_matches_slow_all_tablings)."""
    from keybo.analysis.kmstats import _KEYS, _is_lsb

    ci = {c: i for i, c in enumerate(C30M)}
    ci[" "] = SPACE

    def pair_mass(freqs: dict) -> np.ndarray:
        mass = np.zeros((NSLOT, NSLOT), dtype=np.float64)
        for ngram, freq in freqs.items():
            if len(ngram) != 2:
                continue
            if ngram[0] not in ci or ngram[1] not in ci:
                continue
            mass[ci[ngram[0]], ci[ngram[1]]] += freq
        return mass

    sfb_mask = np.zeros((NSLOT, NSLOT), dtype=np.float64)
    lsb_mask = np.zeros((NSLOT, NSLOT), dtype=np.float64)
    for i in range(30):
        for j in range(30):
            a, b = _KEYS[i], _KEYS[j]
            if a is not b and a.finger == b.finger:
                sfb_mask[i, j] = 1.0
            if _is_lsb(a, b):
                lsb_mask[i, j] = 1.0

    on_key = np.ones((NSLOT, NSLOT), dtype=np.float64)
    on_key[SPACE, :] = 0.0
    on_key[:, SPACE] = 0.0
    bi_mass = pair_mass(bigrams)
    sk_mass = pair_mass(skipgrams)
    return {
        "bi_mass": bi_mass,
        "sk_mass": sk_mass,
        "sfb_mask": sfb_mask,
        "lsb_mask": lsb_mask,  # sfs reuses sfb_mask (same same-finger predicate)
        "bi_total": float((bi_mass * on_key).sum()),
        "sk_total": float((sk_mass * on_key).sum()),
    }


def keymeow_tables() -> tuple[dict, dict, dict]:
    """The vendored keymeow bigrams/skipgrams/trigrams the FROZEN board tables from."""
    with gzip.open(KEYMEOW_PATH, "rt") as source:
        data = json.load(source)
    return data["bigrams"], data["skipgrams"], data["trigrams"]


def kmstats_fast(perm: np.ndarray, km: dict) -> dict[str, float]:
    """kmstats sfb/sfs/lsb percentages; matches KmStats.stats to <1e-9."""
    placed_sfb = km["sfb_mask"][perm[:, None], perm[None, :]]
    placed_lsb = km["lsb_mask"][perm[:, None], perm[None, :]]
    return {
        "sfb": 100.0 * float((km["bi_mass"] * placed_sfb).sum()) / km["bi_total"],
        "lsb": 100.0 * float((km["bi_mass"] * placed_lsb).sum()) / km["bi_total"],
        "sfs": 100.0 * float((km["sk_mass"] * placed_sfb).sum()) / km["sk_total"],
    }


# ---------------------------------------------------------------------------
# tb scissor over an arbitrary bigram tabling
# ---------------------------------------------------------------------------
def _import_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_comfort(bigrams: dict, skipgrams: dict):
    """ComfortObjective over an explicit tabling (default paths would load iWeb)."""
    tb = sys.modules.get("gob_tb") or _import_path("gob_tb", str(HERE / "tb_objective_ref.py"))
    return tb.ComfortObjective(bigrams=bigrams, skipgrams=skipgrams)


def tb_scissor_fast(perm: np.ndarray, comfort) -> float:
    prob = comfort._bigram_probability  # (31,31) char-pair prob
    cost = comfort.submetrics["scissor"]._cost  # (31,31) slot-pair
    return 100.0 * float((prob * cost[perm[:, None], perm[None, :]]).sum())


# ---------------------------------------------------------------------------
# six-surface speed over an arbitrary trigram tabling
# ---------------------------------------------------------------------------
class SixSurface:
    """Six-surface modeled speed with the trigram weights from ONE corpus.

    The surfaces themselves are the frozen 31^3 model tables (the model is held FIXED —
    RESELECT-90-110 closed the model axis). Only the trigram frequency weights change.
    `ceilings` are the per-surface maxima of saved% over the 46-layout reference
    population *under this corpus*, so the normalized floor is not a two-corpus hybrid.
    """

    def __init__(self, corpus: str = "iweb", ceilings: dict[str, float] | None = None):
        self.corpus = corpus
        self.means = {s: np.load(SURFACE_DIR / f"{s}.complete.seedmean.npy") for s in SURFACES}
        trigrams = load_freq(CORPUS_DIRS[corpus] / "trigrams.txt")
        ci = {c: i for i, c in enumerate(C30M)}
        ci[" "] = SPACE
        charset = set(C30M) | {" "}
        entries = [
            (ci[t[0]], ci[t[1]], ci[t[2]], f)
            for t, f in trigrams.items()
            if len(t) == 3 and all(c in charset for c in t)
        ]
        if not entries:
            raise ValueError(f"corpus {corpus!r} has no C30M-covered trigrams")
        self.I = np.array([e[0] for e in entries])
        self.J = np.array([e[1] for e in entries])
        self.K = np.array([e[2] for e in entries])
        self.F = np.array([e[3] for e in entries], dtype=np.float64)
        self.I32, self.J32, self.K32 = (
            self.I.astype(np.int32),
            self.J.astype(np.int32),
            self.K.astype(np.int32),
        )
        self.mean_flat = np.stack([self.means[s].reshape(-1) for s in SURFACES])
        self.qwerty = np.array([self._score_one(perm_of(C30M), self.means[s]) for s in SURFACES])
        if ceilings is None:
            ceilings = derive_ceilings(self)
        self.ceiling_map = dict(ceilings)
        self.ceilings = np.array([ceilings[s] for s in SURFACES])

    def _score_one(self, perm: np.ndarray, surface: np.ndarray) -> float:
        return float((self.F * surface[perm[self.I], perm[self.J], perm[self.K]]).sum())

    def saved(self, perm: np.ndarray) -> np.ndarray:
        """saved-vs-qwerty% on each of the six surfaces."""
        flat = (perm[self.I] * NSLOT + perm[self.J]) * NSLOT + perm[self.K]
        fit = (self.mean_flat[:, flat] * self.F).sum(axis=1)
        return 100.0 * (1.0 - fit / self.qwerty)

    def saved_batch(self, perms: np.ndarray) -> np.ndarray:
        """perms: (B,31) -> (B,6). Reduces each layout's trigrams to a 31^3 histogram,
        then one (B,29791)@(29791,6) matmul. Identical to saved() to <1e-11."""
        batch = perms.shape[0]
        p32 = perms.astype(np.int32)
        flat = (p32[:, self.I32] * NSLOT + p32[:, self.J32]) * NSLOT + p32[:, self.K32]
        weights = np.empty((batch, NSLOT**3), dtype=np.float64)
        for b in range(batch):
            weights[b] = np.bincount(flat[b], weights=self.F, minlength=NSLOT**3)
        return 100.0 * (1.0 - (weights @ self.mean_flat.T) / self.qwerty)

    def floor_mean(self, perm: np.ndarray) -> tuple[float, float]:
        """RAW min()/mean over the six surfaces (kept for continuity with closure-3)."""
        saved = self.saved(perm)
        return float(saved.min()), float(saved.mean())

    def normfloor(self, perm: np.ndarray) -> float:
        """FLOOR-METHODOLOGY-1 ceiling-fraction NORMALIZED floor: min_s saved_s/ceiling_s.
        The HEADLINE floor axis — a raw min() is scale-broken (Community's saved% range is
        ~1.6x wider, so it binds the raw floor 0/46 times and is silently discarded)."""
        return float((self.saved(perm) / self.ceilings).min())

    def normfloor_batch(self, perms: np.ndarray) -> np.ndarray:
        return (self.saved_batch(perms) / self.ceilings).min(axis=1)


def reference_population() -> list[str]:
    """The frozen 46-layout reference population for the ceilings."""
    with open(REF_BOARD) as fh:
        return list(json.load(fh)["board"].keys())


def derive_ceilings(six: SixSurface) -> dict[str, float]:
    """Per-surface ceiling = max saved% over the reference population UNDER six's corpus.
    Same population and same formula for every corpus; only the weights change."""
    population = reference_population()
    saved = np.array([six.saved(perm_of(lay)) for lay in population])
    return {s: float(saved[:, i].max()) for i, s in enumerate(SURFACES)}


# ---------------------------------------------------------------------------
# corpus-invariant community gauges (wfd / genkey / oxey1 / oxey2)
# ---------------------------------------------------------------------------
#: The two wfd paths. **`corrected` is the only correct one** and the default here.
#:
#: ``legacy`` reproduces the frozen artifacts' number, which was taken on a board that is not a
#: permutation of the 31 keys: the campaign's ``oxey_ports.perm_arrays`` hand-rolls the index
#: arrays, never assigns ``;`` a slot, so ``;`` keeps its ``np.zeros`` default and lands on dof 0
#: — evicting the slot-0 character and duplicating another. It exists ONLY so a preflight can
#: positive-control that we reproduce the campaign's frame before correcting it. It must never
#: rank or gate a layout. See ``keybo.analysis.community.Oxeylyzer2.wfd_legacy_board`` and
#: ``wfd_fix.CorrectedWfd``.
WFD_MODES = ("corrected", "legacy")


def build_invariant_gauges(wfd_mode: str = "corrected") -> dict:
    """genkey / oxey1 / oxey2 / wfd. These take NO corpus argument — each scores on its
    own vendored corpus, so they are IMMOVABLE under any corpus swap (prereg §3).

    ``wfd_mode='corrected'`` routes wfd through the VALIDATED ``community._dof_arrays``
    (permutation-checked). ``'legacy'`` reproduces the frozen corrupt-board number for the
    positive control only. The mode is an explicit constructor argument, never a global, so a
    worker process cannot silently inherit the wrong one.
    """
    from keybo.analysis.community import community_suite

    if wfd_mode not in WFD_MODES:
        raise ValueError(f"wfd_mode must be one of {WFD_MODES}, got {wfd_mode!r}")
    genkey, oxey1, oxey2 = community_suite(";")
    sys.path.insert(0, str(HERE))
    from wfd_fix import CorrectedWfd

    chars31 = list(C30M) + [";"]
    corrected = CorrectedWfd()
    if list(corrected.o2.chars) != chars31:  # pragma: no cover - contract check
        raise ValueError("wfd character universe disagrees with the board's C30M charset")
    return {
        "genkey": genkey,
        "oxey1": oxey1,
        "oxey2": oxey2,
        "o2": corrected,
        "chars31": chars31,
        "wfd_mode": wfd_mode,
        "wfd": corrected.wfd if wfd_mode == "corrected" else corrected.wfd_legacy,
    }


# ---------------------------------------------------------------------------
# The board, under one ARM
# ---------------------------------------------------------------------------
class ArmBoard:
    """The 10-axis board under one (corpus, arm) combination.

    arm='A': lsb/sfb/sfs tabled from KEYMEOW (the frozen board's own definition) —
             the control, directly comparable to the frozen wider-dominance verdict.
    arm='B': lsb/sfb/sfs tabled from the CORPUS — the multi-source user's board, and the
             only arm on which the sfb blocker can move (prereg §3).
    """

    def __init__(
        self, corpus: str = "iweb", arm: str = "A", ceilings=None, wfd_mode: str = "corrected"
    ):
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        self.corpus = corpus
        self.arm = arm
        self.wfd_mode = wfd_mode
        bigrams, skipgrams, _ = corpus_tables(corpus)
        self.six = SixSurface(corpus, ceilings=ceilings)
        self.comfort = build_comfort(bigrams, skipgrams)  # scissor: always the corpus
        if arm == "A":
            km_bi, km_sk, _ = keymeow_tables()
            self.km = build_kmstats_matrices(km_bi, km_sk)
        else:
            self.km = build_kmstats_matrices(bigrams, skipgrams)
        self.gauges = build_invariant_gauges(wfd_mode)

    # -- fast path (search inner loop) --------------------------------------
    def axes(self, layout: str, floor_kind: str = "norm") -> dict[str, float]:
        """The 10 board axes. floor_kind='norm' (headline) or 'raw' (continuity)."""
        perm = perm_of(layout)
        if floor_kind == "norm":
            floor = self.six.normfloor(perm)
            mean = float(self.six.saved(perm).mean())
        else:
            floor, mean = self.six.floor_mean(perm)
        km = kmstats_fast(perm, self.km)
        # wfd takes the LAYOUT STRING, not a hand-rolled index array: the construction goes
        # through community._dof_arrays, which permutation-checks it. That check's absence is
        # the entire campaign wfd bug (trap 28).
        return {
            "floor": floor,
            "mean": mean,
            "wfd": float(self.gauges["wfd"](layout)),
            "genkey": float(self.gauges["genkey"].score_primed(layout)),
            "oxey1": float(self.gauges["oxey1"].score_primed(layout)),
            "oxey2": float(self.gauges["oxey2"].score_primed(layout)),
            "lsb": km["lsb"],
            "scissor": tb_scissor_fast(perm, self.comfort),
            "sfb": km["sfb"],
            "sfs": km["sfs"],
        }

    # -- slow ground-truth path (verification only) ------------------------
    def axes_slow(self, layout: str, floor_kind: str = "norm") -> dict[str, float]:
        """Same 10 axes via the SLOW reference machinery — KmStats.stats (Python triple
        loop) and ComfortObjective.values — with ZERO fast-path reuse. Used to verify
        every REPORTED layout at zero error (prereg §6.4)."""
        from keybo.analysis.kmstats import KmStats

        perm = perm_of(layout)
        if floor_kind == "norm":
            floor = self.six.normfloor(perm)
            mean = float(self.six.saved(perm).mean())
        else:
            floor, mean = self.six.floor_mean(perm)
        if self.arm == "A":
            bi, sk, tri = keymeow_tables()
        else:
            bi, sk, tri = corpus_tables(self.corpus)
        stats = KmStats(bi, sk, tri).stats(layout)
        # ZERO-REUSE wfd: a fresh scorer + an explicit Python loop, sharing no cached array
        # with the fast path. `legacy` has no slow twin by design — it is not a measurement
        # of a layout, so there is nothing to verify it against.
        if self.wfd_mode == "legacy":
            wfd_value = float(self.gauges["wfd"](layout))
        else:
            wfd_value = float(self.gauges["o2"].wfd_slow_reference(layout))
        return {
            "floor": floor,
            "mean": mean,
            "wfd": wfd_value,
            "genkey": float(self.gauges["genkey"].score_primed(layout)),
            "oxey1": float(self.gauges["oxey1"].score_primed(layout)),
            "oxey2": float(self.gauges["oxey2"].score_primed(layout)),
            "lsb": stats["lsb"],
            "scissor": float(self.comfort.values(layout)["scissor"]),
            "sfb": stats["sfb"],
            "sfs": stats["sfs"],
        }

    def incumbent_axes(self, floor_kind: str = "norm") -> dict[str, dict]:
        """The five incumbents scored on THIS arm's board. A dominance test is only
        meaningful when candidate and incumbent are scored the same way, so the
        incumbents are always re-tabled to match the arm."""
        return {
            name: {"layout": lay, **self.axes(lay, floor_kind)} for name, lay in INCUMBENTS.items()
        }

    # -- batch objectives for the EA inner loop -----------------------------
    def evaluate_batch(self, movables: np.ndarray) -> np.ndarray:
        """movables: (B,30) char->slot -> (B,6) MINIMIZATION objectives:
        [-normfloor, -mean, scissor, lsb, sfb, sfs]."""
        batch = movables.shape[0]
        perms = np.empty((batch, NSLOT), dtype=np.int64)
        perms[:, :30] = movables
        perms[:, 30] = SPACE
        saved = self.six.saved_batch(perms)
        out = np.empty((batch, 6), dtype=np.float64)
        out[:, 0] = -(saved / self.six.ceilings).min(axis=1)  # maximize NORMALIZED floor
        out[:, 1] = -saved.mean(axis=1)  # maximize mean (a mean discards no source)
        gather_i = perms[:, :, None]
        gather_j = perms[:, None, :]
        sfb_placed = self.km["sfb_mask"][gather_i, gather_j]
        lsb_placed = self.km["lsb_mask"][gather_i, gather_j]
        scissor_placed = self.comfort.submetrics["scissor"]._cost[gather_i, gather_j]
        out[:, 2] = 100.0 * np.einsum("ab,iab->i", self.comfort._bigram_probability, scissor_placed)
        out[:, 3] = (
            100.0 * np.einsum("ab,iab->i", self.km["bi_mass"], lsb_placed) / self.km["bi_total"]
        )
        out[:, 4] = (
            100.0 * np.einsum("ab,iab->i", self.km["bi_mass"], sfb_placed) / self.km["bi_total"]
        )
        out[:, 5] = (
            100.0 * np.einsum("ab,iab->i", self.km["sk_mass"], sfb_placed) / self.km["sk_total"]
        )
        return out


# ---------------------------------------------------------------------------
# dominance
# ---------------------------------------------------------------------------
def oriented(axes: dict) -> np.ndarray:
    """Board axes as an all-'higher-is-better' vector."""
    return np.array([SIGN[a] * axes[a] for a in AXES])


def dominates(cand: dict, inc: dict, atol: float = 1e-9) -> tuple[bool, int, int]:
    """Does cand dominate inc on all 10 axes (>= everywhere, > somewhere)?
    Returns (is_dominator, n_ge, n_strictly_gt)."""
    cv = oriented(cand)
    iv = oriented(inc)
    n_ge = int(np.sum(cv >= iv - atol))
    n_gt = int(np.sum(cv > iv + atol))
    return (n_ge == len(AXES) and n_gt >= 1), n_ge, n_gt
