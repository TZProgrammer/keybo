"""MODELNORM-1 — the three fitted model surfaces on the **native** frame, and the
user's per-model 0-1 anchored normalization.

WHAT THIS IMPLEMENTS (the user's design, verbatim in intent)
-----------------------------------------------------------
  1. Score ~100 RANDOMLY GENERATED layouts on each model. That distribution defines the
     model's **"0"**.
  2. Run a search that maximizes EACH model ALONE. That per-model optimum defines its
     **"1"**.
  3. Normalize every model's score into [0,1] against ITS OWN range.
  4. Optimize the normalized blend with an explicit per-model preference weight ON TOP.

    norm_m(L) = (zero_m - fit_m(L)) / (zero_m - one_m)

``fit`` is predicted TIME (lower = faster), so the numerator is *inverted* on purpose:
``fit == zero_m`` -> 0, ``fit == one_m`` (the per-model optimum, the FASTEST) -> 1. Higher
normalized = better. :func:`assert_direction` pins this against a known ordering, because a
sign error here would invert every preference weight (the campaign has already shipped one
sign error of exactly this shape).

WHY `.native` AND NOT `.standardized` (trap 5, re-derived here)
--------------------------------------------------------------
The shipped ``keybo.analysis.surfaces`` resolves ``<NAME>.standardized.npy{,.gz}`` and
nothing else. In the standardized frame every source carries **AALTO's** bigram tensor:
verified in this module's own guard, ``standardized - native`` is exactly independent of the
third slot (max variation over c = 1.14e-13) and is **EXACTLY 0.0 for AALTO**, i.e.
standardization substitutes AALTO's ``T2`` into COMMUNITY and POOL. Three "models" sharing a
bigram tensor would destroy this whole exercise, so :class:`NativeSurfaces` loads the
``.native`` arrays and :meth:`NativeSurfaces.assert_native_frame` refuses to run otherwise.

FRAME AND UNITS (both stated because both have cost this campaign a retraction)
------------------------------------------------------------------------------
* geometry-only ``g``; the layout-independent ``b(ngram)`` term is excluded (same frame as
  ``keybo.analysis.surfaces``).
* the arrays are **BAKED at 90 WPM** and cannot be re-evaluated at another WPM (7 of 8
  per-seed models are gone), so a 90-110 WPM objective is NOT honourable on these columns.
* the corpus is named on every number; default is the production ``blend-v1``.

MODELLED ONLY: tau saturated at 1.0, Phase-D cancelled. Nothing here is a claim about
realized typing speed, and no layout here is promoted or adopted.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

#: C30M character order — the slot order the surfaces were built in; space is slot 30.
C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
SPACE = 30
#: The three model pools, in report order. These are the "three models" of the design.
MODELS = ("AALTO", "COMMUNITY", "POOL")
#: The campaign's peak family.
FAMILY = "TRI_PS_FREQ_PRIOR"
#: The WPM the arrays were materialized at. NOT a knob.
BAKED_WPM = 90.0
FRAME_NOTE = "geometry-only (g); the layout-independent b(ngram) term is excluded"

#: The surviving native surfaces. ``keybo.analysis.surfaces`` cannot reach these (it only
#: resolves ``.standardized``), so the path is explicit and its digest is recorded.
NATIVE_DIR = Path(
    "/local/home/zegertho/agent/state/keybo-selmethod/artifacts/"
    "old-new-layout-comparison/tri_frequency_old_new_surfaces"
)

#: Candidate layouts this arm reports on. Strings are the shipped registry's
#: (``keybo.cli.analyze._EXTRA_NAMED`` / ``keybo.layouts.NAMED_LAYOUTS``) verbatim, plus the
#: two campaign arms. A name here is scored by the same code as every other row.
CANDIDATES: dict[str, str] = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "arm-B": "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "arm-A": "udy.,fgpmliheaocsntr-k'qjwzbvx",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}


def perm_of(layout: str) -> np.ndarray:
    """char (C30M order) -> slot index, with space pinned at slot 30.

    Validates that the input IS a permutation of C30M (trap 13: a hand-rolled index array
    next to a validating helper is the bug's habitat, so there is exactly one constructor).
    """
    if len(layout) != 30 or set(layout) != set(C30M):
        raise ValueError(f"not a C30M permutation: {layout!r}")
    return np.array([layout.index(c) for c in C30M] + [SPACE], dtype=np.int64)


def layout_of(perm: np.ndarray) -> str:
    """Inverse of :func:`perm_of` — the 30-char row-major layout string."""
    out = [""] * 30
    for char_index, char in enumerate(C30M):
        out[int(perm[char_index])] = char
    return "".join(out)


def layout_key(layout: str) -> int:
    """Stable 64-bit digest of a layout string (trap 8: never ``hash()``, it is salted)."""
    return int.from_bytes(hashlib.blake2b(layout.encode(), digest_size=8).digest(), "little")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# corpus objective
# ---------------------------------------------------------------------------
def build_trigram_objective(trigram_path: Path) -> tuple[np.ndarray, ...]:
    """``(i, j, k, freq)`` over the C30M+space trigrams of one corpus file.

    Routed through the shipped loader so the corpus convention is the repo's, not a
    hand-rolled second parse. ``keybo`` resolves from whatever tree the interpreter was
    started in — this module deliberately does NOT insert a repo path (trap 35: a driver that
    hardcodes ``~/repos/keybo`` at ``sys.path[0]`` silently un-isolates the worktree it is run
    from). Run it with the worktree's own ``uv run --no-sync``.
    """
    from keybo.data.corpus import load_frequencies

    corpus = load_frequencies(str(trigram_path))
    index = {c: i for i, c in enumerate(C30M)}
    index[" "] = SPACE
    charset = set(C30M) | {" "}
    entries = [
        (index[t[0]], index[t[1]], index[t[2]], f)
        for t, f in corpus.items()
        if len(t) == 3 and all(c in charset for c in t)
    ]
    return (
        np.array([e[0] for e in entries], dtype=np.int32),
        np.array([e[1] for e in entries], dtype=np.int32),
        np.array([e[2] for e in entries], dtype=np.int32),
        np.array([e[3] for e in entries], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# the three native surfaces
# ---------------------------------------------------------------------------
class NativeSurfaces:
    """The three ``<MODEL>_TRI_PS_FREQ_PRIOR.native`` 31^3 surfaces + the corpus objective.

    ``fit`` is the campaign QAP objective ``sum_t F[t] * S[p(t0), p(t1), p(t2)]`` in predicted
    ms on the geometry frame — the SAME formula as ``keybo.analysis.surfaces.score_fit``,
    which :func:`assert_matches_shipped` pins against the shipped code on the standardized
    arrays (a positive control on the *arithmetic*, run on the frame where the shipped code
    can reach the same arrays).
    """

    def __init__(self, corpus: str | None = None, native_dir: Path = NATIVE_DIR) -> None:
        from keybo.data.corpus import corpus_name_for, production_corpus_dir

        self.native_dir = Path(native_dir)
        self.corpus_dir = production_corpus_dir(corpus)
        self.corpus_name = corpus_name_for(self.corpus_dir)
        self.trigram_path = self.corpus_dir / "trigrams.txt"
        self.surfaces = {
            m: np.load(self.native_dir / f"{m}_{FAMILY}.native.npy") for m in MODELS
        }
        self.digests = {
            m: sha256_of(self.native_dir / f"{m}_{FAMILY}.native.npy") for m in MODELS
        }
        for m, arr in self.surfaces.items():
            if arr.shape != (31, 31, 31):
                raise ValueError(f"surface {m} has shape {arr.shape}, expected (31,31,31)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"surface {m} holds non-finite values")
        self.I, self.J, self.K, self.F = build_trigram_objective(self.trigram_path)
        self.total_freq = float(self.F.sum())
        #: (3, 29791) flattened surfaces, in MODELS order — the batch matmul's right side.
        self.flat = np.stack([self.surfaces[m].reshape(-1) for m in MODELS])
        self.assert_native_frame()

    # -- guards ------------------------------------------------------------
    def assert_native_frame(self) -> dict:
        """TRAP 5 as an executable assertion, not a comment.

        Loads the standardized twins and proves (a) they differ from the native arrays for
        COMMUNITY and POOL, (b) they are IDENTICAL for AALTO, and (c) the difference is
        exactly the ``c``-independent (bigram) part — i.e. standardization substitutes
        AALTO's ``T2`` everywhere. Then asserts THIS object holds the native arrays.
        """
        report = {}
        for m in MODELS:
            std_path = self.native_dir / f"{m}_{FAMILY}.standardized.npy"
            std = np.load(std_path)
            delta = std - self.surfaces[m]
            over_c = float(np.abs(delta - delta[:, :, :1]).max())
            report[m] = {
                "max_abs_std_minus_native": float(np.abs(delta).max()),
                "max_variation_of_delta_over_third_slot": over_c,
            }
            # the delta is a pure bigram-tensor substitution: no dependence on slot c
            assert over_c < 1e-9, f"{m}: std-native varies over c by {over_c}"
        assert report["AALTO"]["max_abs_std_minus_native"] == 0.0, (
            "AALTO native != AALTO standardized; the frame identity assumed here is wrong"
        )
        for m in ("COMMUNITY", "POOL"):
            assert report[m]["max_abs_std_minus_native"] > 1.0, (
                f"{m} standardized == native, so the .native frame is not distinguishable "
                f"and the three models would share AALTO's bigram tensor"
            )
        # and the arrays we actually hold must BE the native ones
        for m in MODELS:
            held = self.surfaces[m]
            nat = np.load(self.native_dir / f"{m}_{FAMILY}.native.npy")
            assert np.array_equal(held, nat), f"{m}: held array is not the .native file"
        self.frame_report = report
        return report

    def assert_matches_shipped(self) -> float:
        """Positive control: our ``fit`` == the shipped ``surfaces.score_fit``.

        Run on the STANDARDIZED arrays, because that is the frame the shipped resolver can
        reach — this controls the *arithmetic* (index order, corpus restriction, weighting),
        which is frame-independent. Returns the max abs error over the candidate set.
        """
        from keybo.analysis import surfaces as S

        objective = S.trigram_objective(str(self.trigram_path))
        worst = 0.0
        for layout in CANDIDATES.values():
            for m in MODELS:
                std = np.load(self.native_dir / f"{m}_{FAMILY}.standardized.npy")
                theirs = S.score_fit(layout, std, objective)
                perm = perm_of(layout)
                mine = float(
                    (self.F * std[perm[self.I], perm[self.J], perm[self.K]]).sum()
                )
                worst = max(worst, abs(theirs - mine))
        assert worst < 1e-6, f"fit disagrees with shipped surfaces.score_fit by {worst}"
        return worst

    # -- scoring -----------------------------------------------------------
    def fit_one(self, perm: np.ndarray) -> np.ndarray:
        """(3,) predicted-ms fit per model, in MODELS order. Lower = faster."""
        gathered = self.flat[
            :, (perm[self.I] * 31 + perm[self.J]) * 31 + perm[self.K]
        ]
        return (gathered * self.F).sum(axis=1)

    #: Rows per histogram tile in :meth:`fit_batch`. The (B, 29791) weight matrix is the hot
    #: allocation: a 435-row tile is 104 MB and falls out of cache, so the batch is walked in
    #: small tiles that reuse ONE buffer. Measured 1.5k -> 5.9k evals/s/process at B=435.
    TILE = 16

    def fit_batch(self, perms: np.ndarray, tile: int | None = None) -> np.ndarray:
        """(B,31) perms -> (B,3) predicted-ms fits.

        Histograms each layout's trigrams into the 29791 flat surface bins, then one
        ``(TILE, 29791) @ (29791, 3)`` matmul per tile.

        ⚠ **Every matmul is done at EXACTLY the same shape**, zero-padding the final partial
        tile. That is not cosmetic: BLAS picks its kernel and blocking from the operand shape,
        so a `(9, 29791)` product and a `(16, 29791)` product return results differing by
        ~1e-15 relative on the same rows. Without the padding a layout's score would depend on
        *how many other layouts happened to be in its batch* — the objective would not be a
        function of the layout, and neither the search nor its checkpoint-resume would be
        reproducible. With it, :meth:`fit_batch` is bit-exact across every batch length and
        tile size (pinned in ``test_fit_batch_is_batch_length_invariant``).
        """
        perms = np.asarray(perms, dtype=np.int32)
        batch = perms.shape[0]
        step = int(tile or self.TILE)
        out = np.empty((batch, 3), dtype=np.float64)
        buffer = np.zeros((step, 29791), dtype=np.float64)
        for lo in range(0, batch, step):
            hi = min(batch, lo + step)
            rows = perms[lo:hi]
            flat_index = (rows[:, self.I] * 31 + rows[:, self.J]) * 31 + rows[:, self.K]
            span = hi - lo
            for b in range(span):
                buffer[b] = np.bincount(flat_index[b], weights=self.F, minlength=29791)
            if span < step:  # zero the padding rows so the shape is constant but they add 0
                buffer[span:] = 0.0
            out[lo:hi] = (buffer @ self.flat.T)[:span]
        return out

    def fit_of_layout(self, layout: str) -> np.ndarray:
        return self.fit_one(perm_of(layout))

    def identity(self) -> dict:
        return {
            "models": list(MODELS),
            "family": FAMILY,
            "frame": "native",
            "frame_note": FRAME_NOTE,
            "baked_wpm": BAKED_WPM,
            "wpm_caveat": (
                "the arrays are BAKED at 90 WPM and cannot be re-evaluated (7 of 8 per-seed "
                "models are gone), so a 90-110 WPM objective is NOT honourable on these columns"
            ),
            "corpus": self.corpus_name,
            "corpus_trigrams": str(self.trigram_path),
            "corpus_trigrams_sha256": sha256_of(self.trigram_path),
            "surface_dir": str(self.native_dir),
            "surface_sha256": self.digests,
            "native_frame_guard": self.frame_report,
            "n_trigrams": int(self.I.size),
            "total_trigram_freq": self.total_freq,
            # the BLAS operand shape every published fit was computed at: changing it moves a
            # fit by ~1e-15 relative, so a number is only reconcilable if the shape is named.
            "tile": int(self.TILE),
            "numpy": np.__version__,
        }


# ---------------------------------------------------------------------------
# the normalization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Anchors:
    """Per-model ``(zero, one)`` anchor pair in predicted-ms units, plus provenance.

    ``zero`` comes from the random-layout distribution (step 1), ``one`` from the per-model
    optimized search (step 2). Both are recorded with the statistic and the budget/seed that
    produced them, because the "1" anchor is an optimizer output and therefore
    budget-dependent — the normalization is not reproducible without them.
    """

    zero: dict[str, float]
    one: dict[str, float]
    zero_statistic: str
    zero_n: int
    zero_seed: int
    zero_sd: dict[str, float]
    one_provenance: dict


class BlendNormalizer:
    """The user's normalization + the weighted blend.

    ``norm_m(L) = (zero_m - fit_m(L)) / (zero_m - one_m)``   (1 = best, 0 = random-mean)
    ``blend(L)  = sum_m w_m * norm_m(L) / sum_m w_m``        (weights = PREFERENCE)

    The weights are normalized to sum 1 so a blend value stays on the same 0-1 scale for any
    weighting, which is what makes (1,1,1) and (2,1,1) comparable numbers rather than
    differently-scaled ones.
    """

    def __init__(self, anchors: Anchors, weights: dict[str, float] | None = None) -> None:
        self.anchors = anchors
        self.set_weights(weights or dict.fromkeys(MODELS, 1.0))
        self.span = np.array(
            [anchors.zero[m] - anchors.one[m] for m in MODELS], dtype=np.float64
        )
        if not np.all(self.span > 0):
            raise ValueError(
                f"anchor span must be positive (zero slower than one) for every model; got "
                f"{dict(zip(MODELS, self.span.tolist(), strict=True))}"
            )
        self.zero_vec = np.array([anchors.zero[m] for m in MODELS], dtype=np.float64)

    def set_weights(self, weights: dict[str, float]) -> None:
        vec = np.array([float(weights.get(m, 0.0)) for m in MODELS], dtype=np.float64)
        if np.any(vec < 0):
            raise ValueError(f"preference weights must be non-negative; got {weights}")
        if vec.sum() <= 0:
            raise ValueError(f"preference weights must not be all-zero; got {weights}")
        self.raw_weights = {m: float(v) for m, v in zip(MODELS, vec, strict=True)}
        self.weights = vec / vec.sum()

    # -- normalization ------------------------------------------------------
    def normalize(self, fits: np.ndarray) -> np.ndarray:
        """(..,3) predicted-ms fits -> (..,3) normalized scores. 1 = BEST (fastest)."""
        return (self.zero_vec - np.asarray(fits, dtype=np.float64)) / self.span

    def blend(self, fits: np.ndarray) -> np.ndarray:
        """(..,3) fits -> (..,) weighted normalized blend. HIGHER = better."""
        return self.normalize(fits) @ self.weights

    def objective(self, fits: np.ndarray) -> np.ndarray:
        """The MINIMIZED form of :meth:`blend` (the search minimizes)."""
        return -self.blend(fits)


def assert_direction(surf: NativeSurfaces, norm: BlendNormalizer, one_layouts: dict) -> dict:
    """TRAP 3 as an executable assertion: 1 = BEST, and the sign cannot be inverted silently.

    Three independent checks:
      * each model's own optimum normalizes to ~1.0 on that model;
      * ``qwerty30m`` normalizes well below every optimum (it is the slowest candidate);
      * a FASTER layout always gets a HIGHER normalized score (monotone-decreasing in ms).
    """
    out = {}
    for m_index, m in enumerate(MODELS):
        fits = surf.fit_of_layout(one_layouts[m])
        value = float(norm.normalize(fits)[m_index])
        assert abs(value - 1.0) < 1e-9, f"{m}: its own optimum normalizes to {value}, not 1.0"
        out[f"{m}_optimum_normalizes_to"] = value
    qwerty = norm.normalize(surf.fit_of_layout(CANDIDATES["qwerty30m"]))
    out["qwerty30m_normalized"] = qwerty.tolist()
    for m_index, m in enumerate(MODELS):
        assert qwerty[m_index] < 0.9, (
            f"qwerty30m normalizes to {qwerty[m_index]} on {m}; a naive (x-lo)/(hi-lo) would "
            f"put the WORST layout near 1 — the sign is inverted"
        )
    # monotonicity: faster (lower ms) must map to higher normalized
    arm_b = surf.fit_of_layout(CANDIDATES["arm-B"])
    q = surf.fit_of_layout(CANDIDATES["qwerty30m"])
    faster = arm_b < q
    higher = norm.normalize(arm_b) > norm.normalize(q)
    assert np.array_equal(faster, higher), (
        f"direction broken: faster={faster.tolist()} but higher-normalized={higher.tolist()}"
    )
    out["monotone_faster_is_higher"] = True
    return out


# ---------------------------------------------------------------------------
# anchors: step 1 (random "0") and step 2 loading (optimized "1")
# ---------------------------------------------------------------------------
def random_layouts(n: int, seed: int) -> np.ndarray:
    """(n,31) random C30M permutations with space pinned at slot 30.

    A fresh ``default_rng(seed)`` — the pool is reproducible from ``(n, seed)`` alone, which
    the "0" anchor's provenance records.
    """
    rng = np.random.default_rng(seed)
    perms = np.empty((n, 31), dtype=np.int32)
    for i in range(n):
        perms[i, :30] = rng.permutation(30)
        perms[i, 30] = SPACE
    return perms


def zero_anchor(
    surf: NativeSurfaces, n: int, seed: int, statistic: str = "mean"
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    """Step 1: the "0" anchor from ``n`` random layouts. Returns ``(anchor, sd, fits)``.

    ``statistic`` is reported, not assumed: ``mean`` is the default because it is the
    distribution's centre of mass and its standard error falls as ``sd/sqrt(n)`` (so the n=100
    vs n=1000 comparison the design asks for is a clean sqrt-n check), while ``median`` is
    reported alongside so a reader can see the distribution is not skewed enough to matter.
    """
    perms = random_layouts(n, seed)
    fits = surf.fit_batch(perms)  # (n,3)
    if statistic == "mean":
        anchor = fits.mean(axis=0)
    elif statistic == "median":
        anchor = np.median(fits, axis=0)
    else:
        raise ValueError(f"unknown zero statistic {statistic!r}")
    return (
        {m: float(v) for m, v in zip(MODELS, anchor, strict=True)},
        {m: float(v) for m, v in zip(MODELS, fits.std(axis=0, ddof=1), strict=True)},
        fits,
    )


def ceiling_fraction_anchors(surf: NativeSurfaces, pool: dict[str, str]) -> dict[str, float]:
    """The PRIOR anchoring for comparison: "1" = the best layout already in a fixed set.

    This is the ceiling-fraction normalization the campaign used before (FLOOR-METHODOLOGY-1).
    It is here so the report can QUANTIFY the difference between the two anchorings on the
    same layouts, which is the specific improvement the user's design claims.
    """
    best = {m: float("inf") for m in MODELS}
    for layout in pool.values():
        fits = surf.fit_of_layout(layout)
        for m_index, m in enumerate(MODELS):
            best[m] = min(best[m], float(fits[m_index]))
    return best


def load_anchors(path: Path) -> Anchors:
    with open(path) as handle:
        blob = json.load(handle)
    return Anchors(
        zero=blob["zero"],
        one=blob["one"],
        zero_statistic=blob["zero_statistic"],
        zero_n=blob["zero_n"],
        zero_seed=blob["zero_seed"],
        zero_sd=blob["zero_sd"],
        one_provenance=blob["one_provenance"],
    )
