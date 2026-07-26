"""Fitted model surfaces (aalto / community / pool) as a reportable gauge (ALLGAUGE-1).

Each surface is a 31x31x31 float64 array ``S[slot_a, slot_b, slot_c]`` of predicted
milliseconds for a trigram typed on those three slots (slot 30 is space). A layout's
**fit** is the corpus-weighted sum

    fit(layout) = sum over trigrams (c1,c2,c3) of F[c1,c2,c3] * S[p(c1), p(c2), p(c3)]

where ``p`` maps a character to the slot the layout puts it on. Lower is faster. This is
the campaign's QAP objective, reproduced exactly (positive-controlled bit-for-bit against
``all-gauge-table.json``'s ``speed.surfaces[*].fit`` for 9 layouts x 8 surfaces).

Three things about these numbers that the column label has to carry, because getting any
of them wrong has already cost this campaign a retraction:

**1. The frame is g, not g + b.** The trained decomposition is
``time = g(geometry, wpm) + b(ngram)``. Only ``g`` is served: ``b`` depends on the
character n-gram, not on where it sits, so it is identical for every layout and cannot
change a ranking (``train.py``: "scoring deliberately ignores it"). These surfaces are
the ``g`` (geometry) frame. A number on the ``g + b`` frame is a different quantity and
must be labelled as such.

**2. The surfaces are BAKED at 90 WPM.** The generator
(``run_tri_frequency_layouts_old_new.py``) hardcodes ``target_wpm=90.0`` /
``wpm=90.0`` when it materializes each array, and the per-seed models behind seven of the
eight surfaces no longer exist, so a surface *cannot* be re-evaluated at another WPM.
``--target-wpm`` therefore moves the measured-keystroke time card but **not** these
columns; :func:`wpm_note` states that instead of letting the number pass for something it
is not. (The user's stated objective band is 90-110 WPM; 90 is the bottom of it and is
what these arrays hold.)

**3. Which surface.** ``BASE`` / ``FREQ_PRIOR`` / ``TRI_PS_FREQ_PRIOR`` are different
fitted models, not variants of one number. Every cell names its own surface;
``TRI_PS_FREQ_PRIOR`` is the campaign's peak model and the default.

Surfaces are looked up in this order, first hit wins per surface name:

1. the ``--surface-dir`` override (or ``KEYBO_SURFACE_DIR``), holding either
   ``<NAME>.standardized.npy`` or ``<NAME>.standardized.npy.gz``;
2. the repo's vendored ``data/surfaces/`` (gzipped; the three ``TRI_PS_FREQ_PRIOR``
   surfaces, ~650 KB total).

A surface that is not found is reported as unavailable with a message naming the
directories searched — never a traceback, so a user without the (4.7 MB, out-of-repo)
full set still gets every other gauge.
"""

from __future__ import annotations

import gzip
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

#: C30M character order — the slot order the surfaces were built in, plus space at 30.
C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
#: The WPM the vendored arrays were materialized at. NOT a knob: it is baked in.
BAKED_WPM = 90.0
#: Surface families, in report order.
FAMILIES = ("BASE", "FREQ_PRIOR", "TRI_PS_FREQ_PRIOR")
#: Model pools, in report order.
POOLS = ("AALTO", "COMMUNITY", "POOL")
#: The campaign's peak model — the default family.
DEFAULT_FAMILY = "TRI_PS_FREQ_PRIOR"

#: The frame these fits live on, spelled out for every report that prints one.
FRAME_NOTE = "geometry-only (g); the layout-independent b(ngram) term is excluded"

_VENDORED = Path(__file__).resolve().parents[3] / "data" / "surfaces"


def surface_names(family: str = DEFAULT_FAMILY) -> tuple[str, ...]:
    """The three ``<POOL>_<FAMILY>`` surface names for one family."""
    if family not in FAMILIES:
        raise ValueError(f"unknown surface family {family!r}; expected one of {FAMILIES}")
    return tuple(f"{pool}_{family}" for pool in POOLS)


def _search_dirs(override: str | None = None) -> list[Path]:
    """Directories to look for surfaces in, highest priority first."""
    dirs: list[Path] = []
    explicit = override or os.environ.get("KEYBO_SURFACE_DIR")
    if explicit:
        dirs.append(Path(explicit))
    dirs.append(_VENDORED)
    return dirs


def _resolve(name: str, override: str | None = None) -> Path | None:
    for directory in _search_dirs(override):
        for suffix in (".standardized.npy", ".standardized.npy.gz"):
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


@lru_cache(maxsize=8)
def available_surfaces(override: str | None = None) -> tuple[str, ...]:
    """Surface names resolvable right now, across every family (sorted)."""
    found = []
    for family in FAMILIES:
        for name in surface_names(family):
            if _resolve(name, override) is not None:
                found.append(name)
    return tuple(sorted(found))


def searched_dirs_note(override: str | None = None) -> str:
    return ", ".join(str(d) for d in _search_dirs(override))


@lru_cache(maxsize=12)
def load_surface(name: str, override: str | None = None) -> np.ndarray:
    """Load one 31x31x31 surface (transparently gunzipping a ``.gz``).

    Raises ``FileNotFoundError`` if the surface is not resolvable — callers that want
    graceful degradation should gate on :func:`available_surfaces` first.
    """
    path = _resolve(name, override)
    if path is None:
        raise FileNotFoundError(
            f"model surface {name!r} not found; searched {searched_dirs_note(override)}"
        )
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            values = np.load(io.BytesIO(handle.read()))
    else:
        values = np.load(path)
    if values.shape != (31, 31, 31):
        raise ValueError(f"model surface {name!r} has shape {values.shape}, expected (31, 31, 31)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"model surface {name!r} holds non-finite values")
    return values


def is_c30m(lay30: str) -> bool:
    """Whether a layout is an exact C30M permutation (the only charset the surfaces index)."""
    return len(lay30) == len(C30M) and set(lay30) == set(C30M)


def layout_permutation(lay30: str) -> np.ndarray:
    """Slot index per C30M character, with space appended at index 30."""
    if not is_c30m(lay30):
        raise ValueError("layout must be an exact C30M permutation")
    return np.array([lay30.index(character) for character in C30M] + [len(C30M)])


@lru_cache(maxsize=2)
def trigram_objective(trigram_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(i, j, k, freq)`` arrays over the C30M+space trigrams of one corpus file.

    Trigrams touching a character outside C30M+space are dropped — the surfaces have no
    slot for them. This is the campaign objective's own restriction, reproduced.
    """
    from keybo.data.corpus import load_frequencies

    corpus = load_frequencies(trigram_path)
    index = {character: position for position, character in enumerate(C30M)}
    index[" "] = len(C30M)
    charset = set(C30M) | {" "}
    entries = [
        (index[t[0]], index[t[1]], index[t[2]], frequency)
        for t, frequency in corpus.items()
        if len(t) == 3 and all(character in charset for character in t)
    ]
    return (
        np.array([e[0] for e in entries]),
        np.array([e[1] for e in entries]),
        np.array([e[2] for e in entries]),
        np.array([e[3] for e in entries], dtype=np.float64),
    )


def default_trigram_path(corpus: str | None = None) -> str:
    """The production corpus's trigram table (``corpus``: a name or path, else the default)."""
    from keybo.data.corpus import production_corpus_dir

    return str(production_corpus_dir(corpus) / "trigrams.txt")


def score_fit(lay30: str, surface: np.ndarray, objective) -> float:
    """Corpus-weighted trigram fit in predicted ms (lower = faster).

    ``sum F[c1,c2,c3] * S[p(c1), p(c2), p(c3)]`` — the QAP objective, on the served
    geometry frame (see the module docstring: ``b(ngram)`` is excluded).
    """
    permutation = layout_permutation(lay30)
    first, second, third, frequency = objective
    return float(
        (frequency * surface[permutation[first], permutation[second], permutation[third]]).sum()
    )


def wpm_note(target_wpm: float) -> str:
    """A note that states the baked WPM whenever the request differs from it."""
    if float(target_wpm) == BAKED_WPM:
        return f"surfaces are baked at {BAKED_WPM:g} WPM, matching --target-wpm"
    return (
        f"surfaces are BAKED at {BAKED_WPM:g} WPM and cannot be re-evaluated "
        f"(the per-seed models behind them are gone), so these columns are NOT at the "
        f"requested {float(target_wpm):g} WPM; the measured-keystroke time card is"
    )


def model_scores(
    lay30: str,
    *,
    family: str = DEFAULT_FAMILY,
    target_wpm: float = BAKED_WPM,
    ref_lay30: str | None = None,
    surface_dir: str | None = None,
    trigram_path: str | None = None,
) -> dict:
    """Every available surface's fit for one layout, with the frame/WPM caveats attached.

    Returns a dict that is always shaped the same, so a report can render it without
    branching on availability:

    * ``available`` — False when the layout's charset is wrong for the surfaces, or when
      no surface of the requested family could be found. ``reason`` says which.
    * ``surfaces`` — per surface: ``fit`` (ms, lower faster), ``saved_vs_ref_pct``
      (against ``ref_lay30`` when it is scorable), and ``surface`` (its own name).
    * ``frame`` / ``baked_wpm`` / ``wpm_matches_request`` / ``wpm_note`` — the labels.
    """
    names = surface_names(family)
    resolvable = [name for name in names if name in available_surfaces(surface_dir)]
    base = {
        "family": family,
        "frame": FRAME_NOTE,
        "baked_wpm": BAKED_WPM,
        "wpm_matches_request": float(target_wpm) == BAKED_WPM,
        "wpm_note": wpm_note(target_wpm),
        "surfaces": {},
    }
    if not resolvable:
        return {
            **base,
            "available": False,
            "reason": (
                f"model surfaces not available for family {family!r}: none of "
                f"{', '.join(names)} found in {searched_dirs_note(surface_dir)}"
            ),
        }
    if not is_c30m(lay30):
        return {
            **base,
            "available": False,
            "reason": (
                "layout is not a C30M permutation, so the modeled surfaces (locked to the "
                "C30M 31-slot table) cannot be indexed: charset mismatch"
            ),
            "surfaces": dict.fromkeys(resolvable),
        }
    objective = trigram_objective(trigram_path or default_trigram_path())
    ref_ok = ref_lay30 is not None and is_c30m(ref_lay30)
    cells = {}
    for name in resolvable:
        surface = load_surface(name, surface_dir)
        fit = score_fit(lay30, surface, objective)
        cell = {"surface": name, "fit": fit, "saved_vs_ref_pct": None}
        if ref_ok:
            ref_fit = score_fit(ref_lay30, surface, objective)
            if ref_fit > 0:
                cell["saved_vs_ref_pct"] = 100.0 * (ref_fit - fit) / ref_fit
        cells[name] = cell
    return {**base, "available": True, "reason": None, "surfaces": cells}
