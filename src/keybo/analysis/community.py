"""Exact ports of the community analyzers, over vendored data (KAN-1, rule b330ab4).

Ports of genkey, oxeylyzer-1 and oxeylyzer-2 scoring, adapted from the parity-gated
campaign ports (P14 rule 01546c8 / P16 rule 6dc4727) and pinned by golden-fixture
tests (tests/analysis/test_community_parity.py) against the real binaries' outputs:

* genkey  — generate.go ``Score`` @ f1f4173, on genkey's own parsed corpus
  (vendored): ``3*fspeed + 1*LSB% + 0.3*|index balance|``; uniform-column
  fingering; trigram term off (stock config).
* oxeylyzer-2 — ``score_cache`` = weighted (same-finger) bigrams + stretch
  bigrams on the 31-key ANSI dof; weights sfb -7 / sfs -1 / stretch -3,
  distances x100 as integers (the binary's exact arithmetic).
* oxeylyzer-1 — ``score_with_precision(usize::MAX)`` = full-trigram pattern
  score + fspeed + stretch + pinky-ring, with the 1/7-decayed skipgram mix.

All three run on the tools' OWN corpora (vendored under
``data/community/vendored/``), because their scores are only meaningful in
their native corpus convention; the analyzer's shared-corpus stats live in
:mod:`keybo.analysis.kmstats`.

Each scorer also exposes an exact term decomposition (``components``) and a
"primed" score (``score_primed``, KAN-PRIME-1): the tool restricted to its
mechanical-strain terms, with the tool's native weights — dropping (T) the
hand-tuned same-finger/time-proxy terms that the measured speed surfaces
supersede, and (S) the trigram flow-preference table (rolls/alternation/
redirect taste weights). ``score()`` is defined as the sum of components, so
the golden parity tests gate the decomposition's exactness.

Layouts are our canonical 30-char row-major strings. The oxeylyzer boards are
31-key (they see a pinned character on the home-row quote slot): C30M-charset
layouts pin ``;`` there and classic-charset layouts pin ``'`` — chosen
automatically from the layout string, matching how semimak/graphite's own .dof
files encode the same convention. The 30 characters are given, so the single
left-over character has exactly one place to go; there is no second convention.

.. warning::
   The campaign's frozen dominance artifacts carry a *different* wfd, produced by
   ``oxey_ports.perm_arrays`` and long documented as "the apostrophe-pinned
   convention". It is a **bug**, not a convention — the board it scores is not a
   permutation. :meth:`Oxeylyzer2.wfd_legacy_board` reproduces it for artifact
   reconciliation only; :meth:`Oxeylyzer2.wfd` is the correct quantity.
"""

from __future__ import annotations

import gzip
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

#: repo-relative vendored data root (resolved against this file, not cwd)
_VENDOR = Path(__file__).resolve().parents[3] / "data" / "community" / "vendored"

# ---- 31-key ANSI geometry shared by both oxeylyzers (libdof conventions) --------------------
# fingers as libdof enums: LP=0 LR=1 LM=2 LI=3 RI=6 RM=7 RR=8 RP=9
_ROW_FINGERS = [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]
FINGERS = _ROW_FINGERS + _ROW_FINGERS + [9] + _ROW_FINGERS
_POS = (
    [(1.5 + i, 1.0) for i in range(10)]
    + [(1.75 + i, 2.0) for i in range(11)]
    + [(2.25 + i, 3.0) for i in range(10)]
)  # key LEFT edges (ANSI row starts 1.5/1.75/2.25, anchor (1,1)); centers = +0.5
_FLEN = {0: -0.15, 1: 0.35, 2: 0.25, 3: -0.30, 6: -0.30, 7: 0.25, 8: 0.35, 9: -0.15}
_XFO = {(0, 1): 0.8, (1, 2): 0.4, (2, 3): 0.1, (6, 7): 0.1, (7, 8): 0.4, (8, 9): 0.8}
N31 = 31
_HAND = [0 if f <= 3 else 1 for f in FINGERS]

#: our slot order (30, row-major) -> dof position index; the quote slot is dof 20
SLOT2DOF = list(range(10)) + list(range(10, 20)) + list(range(21, 31))
APOS_DOF = 20

#: the two 30-key character universes; the 31st (quote-slot) character is whichever of
#: ``;`` / ``'`` the universe leaves over — see :func:`pinned_char`.
C30M_CHARS = "qwertyuiopasdfghjkl'zxcvbnm,.-"
CLASSIC_CHARS = "qwertyuiopasdfghjkl;zxcvbnm,./"


def _load_vendored(name: str) -> dict:
    with gzip.open(_VENDOR / name, "rt") as fh:
        return json.load(fh)


def _dx_dy(i: int, j: int, use_flen: bool) -> tuple[float, float]:
    """o2/v1 dx_dy: collapsed 1u key centers, flen y-shift, signed-dx crossing rule."""
    cx1, cy1 = _POS[i][0] + 0.5, _POS[i][1] + 0.5
    cx2, cy2 = _POS[j][0] + 0.5, _POS[j][1] + 0.5
    f1, f2 = FINGERS[i], FINGERS[j]
    if use_flen:
        cy1 += _FLEN[f1]
        cy2 += _FLEN[f2]
    dx = abs(cx1 - cx2)
    dy = abs(cy1 - cy2)
    xo = _XFO.get((min(f1, f2), max(f1, f2)), 0.0)
    if f1 > f2 and cx1 < cx2 + xo or f1 < f2 and cx1 + xo > cx2:
        dx = -dx
    return dx, dy


@lru_cache(maxsize=1)
def _stretch_pairs() -> tuple[tuple[int, int, int], ...]:
    """(i, j, int(stretch*100)) per the StretchCache: same hand, diff finger, >0.001."""
    out = []
    for i in range(N31):
        for j in range(i + 1, N31):
            f1, f2 = FINGERS[i], FINGERS[j]
            if f1 == f2 or _HAND[i] != _HAND[j]:
                continue
            dx, dy = _dx_dy(i, j, use_flen=True)
            xo = _XFO.get((min(f1, f2), max(f1, f2)), 0.0)
            x_overlap = max(0.0, xo - dx * 1.3 + 0.3333 * dy)
            stretch = math.hypot(dx, dy) + x_overlap - 1.35 * abs(f1 - f2)
            if stretch > 0.001:
                out.append((i, j, int(stretch * 100.0)))
    return tuple(out)


@lru_cache(maxsize=1)
def _samefinger_pairs() -> tuple[tuple[int, int, float], ...]:
    """(i, j, plain center distance) for same-finger pairs (flen cancels)."""
    out = []
    for i in range(N31):
        for j in range(i + 1, N31):
            if FINGERS[i] != FINGERS[j]:
                continue
            dx, dy = _dx_dy(i, j, use_flen=False)
            out.append((i, j, math.hypot(dx, dy)))
    return tuple(out)


def _load_freq_matrix(dic: dict, chars: list[str], scale: float) -> np.ndarray:
    idx = {c: k for k, c in enumerate(chars)}
    m = np.zeros((len(chars), len(chars)), dtype=np.int64)
    for key, f in dic.items():
        if len(key) == 2 and key[0] in idx and key[1] in idx:
            m[idx[key[0]], idx[key[1]]] = int(f * scale)
    return m


def pinned_char(lay30: str) -> str:
    """The character the oxeylyzer boards pin on the quote slot for this layout.

    C30M layouts (which carry ``'`` in the 30 block) pin ``;`` — the convention
    semimak/graphite's own .dof files use; classic layouts (``;`` in the block)
    pin ``'`` — the community default for qwerty-punctuation boards.
    """
    return ";" if "'" in lay30 else "'"


def legacy_board_of(lay30: str, chars31: list[str] | None = None) -> str:
    """The 31-character board the campaign's ``perm_arrays`` actually produced, as a string.

    Diagnostic for :meth:`Oxeylyzer2.wfd_legacy_board`: it makes the corruption visible
    rather than leaving it as a number that differs. Indexed by dof, so position 0 is the
    top-left key and position :data:`APOS_DOF` is the quote slot. For a C30M layout the
    result is *not* a permutation — ``;`` appears on dof 0, ``q`` appears twice, and the
    character that belongs on slot 0 is missing.

        >>> legacy_board_of("pyuo,vgdnlhiea.cstrmkj-z'fwbxq")
        ";yuo,vgdnlhiea.cstrm'kj-zqfwbxq"
    """
    chars31 = list(chars31) if chars31 is not None else list(C30M_CHARS) + [";"]
    index = {character: position for position, character in enumerate(chars31)}
    dof_of_char = np.zeros(N31, dtype=np.int64)
    for slot, character in enumerate(lay30):
        dof_of_char[index[character]] = SLOT2DOF[slot]
    dof_of_char[index["'"]] = APOS_DOF
    char_at_dof = np.zeros(N31, dtype=np.int64)
    char_at_dof[dof_of_char] = np.arange(N31)
    return "".join(chars31[i] for i in char_at_dof)


def check_dof_permutation(dof_of_char: np.ndarray) -> np.ndarray:
    """Raise unless ``dof_of_char`` assigns each of the 31 characters a distinct key.

    Use this on **any** hand-rolled character→key mapping. It is the check whose absence
    is the entire `wfd_legacy_board` bug: a mapping built by assigning into a zero-filled
    array silently leaves unassigned characters on dof 0, and the scatter that inverts it
    (``char_at_dof[dof_of_char] = arange(31)``) then hides the collision by dropping one
    character and duplicating another. Nothing downstream can detect that — the arithmetic
    stays valid, only the board is impossible — so it has to be caught here.

    Returns ``dof_of_char`` unchanged, so it can wrap a construction inline.
    """
    if sorted(dof_of_char.tolist()) != list(range(N31)):
        counts = np.bincount(dof_of_char, minlength=N31)
        raise ValueError(
            "character->key mapping is not a permutation of the 31 keys: "
            f"keys with no character {[int(d) for d in np.flatnonzero(counts == 0)]}, "
            f"keys with more than one {[int(d) for d in np.flatnonzero(counts > 1)]}"
        )
    return dof_of_char


def _dof_arrays(lay30: str, chars31: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(char_at_dof, dof_of_char) index arrays for a 30-char layout + pinned char."""
    if (
        len(lay30) != len(SLOT2DOF)
        or len(chars31) != N31
        or len(set(lay30)) != len(SLOT2DOF)
        or len(set(chars31)) != N31
        or set(lay30) != set(chars31[:-1])
    ):
        raise ValueError("layout and pinned character must form a 31-character permutation")
    idx = {c: k for k, c in enumerate(chars31)}
    dof_of_char = np.empty(N31, dtype=np.int64)
    for slot, ch in enumerate(lay30):
        dof_of_char[idx[ch]] = SLOT2DOF[slot]
    dof_of_char[idx[chars31[30]]] = APOS_DOF
    # belt and braces: the input checks above make this unreachable, but the assertion is
    # what actually protects the scatter below (see check_dof_permutation).
    check_dof_permutation(dof_of_char)
    char_at_dof = np.empty(N31, dtype=np.int64)
    char_at_dof[dof_of_char] = np.arange(N31)
    return char_at_dof, dof_of_char


class Oxeylyzer2:
    """oxeylyzer-2 ``score_cache`` (higher = better; large negative integers)."""

    W_SFB, W_SFS, W_STR = -7, -1, -3
    FW = {0: 77, 1: 32, 2: 24, 3: 21, 6: 21, 7: 24, 8: 32, 9: 77}

    def __init__(self, chars31: list[str]):
        d = _load_vendored("oxeylyzer2-english.json.gz")
        self.chars = list(chars31)
        B = _load_freq_matrix(d["bigrams"], self.chars, d["bigram_total"])
        S = _load_freq_matrix(d["skipgrams"], self.chars, d["skipgram_total"])
        self.SFW = self.W_SFB * B + self.W_SFS * S
        self.STW = (B + (S * 7.0).astype(np.int64)) * self.W_STR
        sf = _samefinger_pairs()
        self.SF_I = np.array([i for i, _, _ in sf])
        self.SF_J = np.array([j for _, j, _ in sf])
        self.SF_D = np.array(
            [int(dist * 100.0) * self.FW[FINGERS[i]] for i, _, dist in sf], dtype=np.int64
        )
        st = _stretch_pairs()
        self.ST_I = np.array([i for i, _, _ in st])
        self.ST_J = np.array([j for _, j, _ in st])
        self.ST_D = np.array([dint for _, _, dint in st], dtype=np.int64)

    def components(self, lay30: str) -> dict[str, int]:
        """Exact term split: ``score == wfd + stretch``. wfd is the same-finger
        time proxy (T); stretch is the mechanical-strain term (C)."""
        cad, _ = _dof_arrays(lay30, self.chars)
        a, b = cad[self.SF_I], cad[self.SF_J]
        wb = int(((self.SFW[a, b] + self.SFW[b, a]) * self.SF_D).sum())
        a, b = cad[self.ST_I], cad[self.ST_J]
        return {"wfd": wb, "stretch": int(((self.STW[a, b] + self.STW[b, a]) * self.ST_D).sum())}

    def score(self, lay30: str) -> int:
        c = self.components(lay30)
        return c["wfd"] + c["stretch"]

    def wfd(self, lay30: str) -> int:
        """The weighted-(same-)finger-distance component alone.

        The quote slot carries the character :func:`pinned_char` selects for this layout
        (``;`` for a C30M board, ``'`` for a classic one) — the layout's 30 characters are
        given, so the one left-over character has exactly one place to go. This is the
        only correct wfd. See :meth:`wfd_legacy_board` for the campaign-era number, which
        is a bug rather than a second convention.
        """
        return self.components(lay30)["wfd"]

    def wfd_legacy_board(self, lay30: str) -> int:
        """wfd on the campaign's CORRUPT board — reproducible, but not a valid layout.

        This is the number in every frozen dominance artifact (``wscissor-allgauge``,
        ``flagship-compare``, ``board-blend-reselect``'s ``primes``, every hunt's
        ``best_axes.wfd``), produced by the campaign's ``oxey_ports.perm_arrays``. It was
        documented as "the apostrophe-pinned convention" — pin ``'`` on the quote slot
        instead of the layout's own quote character. **It is not a convention.** The
        mapping it evaluates is not a permutation of the 31 keys:

        * ``'`` is moved to the quote slot, but ``;`` is never assigned a position, so it
          keeps its zero-initialised default and lands on **dof 0** (top-left, left pinky);
        * the character that genuinely sits on slot 0 is therefore **evicted** from the
          board entirely;
        * the dof that ``'`` vacated is refilled by index 0 — so ``q`` is typed on **two**
          keys.

        Only ``qwerty30m`` escapes (its slot-0 character *is* ``q``, so the board
        degenerates to a valid-but-wrong permutation and moves by 0.08% instead of 1-7%).
        Because qwerty was the reference layout for the campaign's direction derivation,
        the bug hid behind the one layout it barely touches.

        Kept so the frozen artifacts stay bit-reproducible and so a re-analysis can quote
        the number it must reconcile against. **Do not use it to rank or gate layouts** —
        14 of 42 frozen per-incumbent dominance verdicts do not survive the correction.
        Use :meth:`wfd` for anything new.
        """
        if "'" not in self.chars:
            raise ValueError("this board has no apostrophe to pin on the quote slot")
        if "'" not in lay30:
            # The hand-rolled mapping below is only sound when the pinned character IS ',
            # i.e. a classic-charset board -- which this guard rejects. So every input it
            # accepts is one it scores wrongly; there is no correct input.
            raise ValueError(
                "wfd_legacy_board needs ' among the layout's 30 characters "
                "(the legacy board moves it to the quote slot)"
            )
        # NOTE: no permutation assert here, deliberately -- this method's whole purpose is to
        # evaluate the non-permutation, so asserting would make every call raise and the frozen
        # artifacts unreconcilable. The guard belongs on the CORRECT path, where it already is
        # (`_dof_arrays`), and on any NEW hand-rolled mapping. `legacy_board_of` exposes the
        # broken board so a caller can see the damage instead of asserting against it.
        index = {character: position for position, character in enumerate(self.chars)}
        dof_of_char = np.zeros(N31, dtype=np.int64)
        for slot, character in enumerate(lay30):
            dof_of_char[index[character]] = SLOT2DOF[slot]
        dof_of_char[index["'"]] = APOS_DOF
        char_at_dof = np.zeros(N31, dtype=np.int64)
        char_at_dof[dof_of_char] = np.arange(N31)
        a, b = char_at_dof[self.SF_I], char_at_dof[self.SF_J]
        return int(((self.SFW[a, b] + self.SFW[b, a]) * self.SF_D).sum())

    def wfd_apostrophe_pinned(self, lay30: str) -> int:
        """Deprecated campaign-era name for :meth:`wfd_legacy_board`.

        The old name asserts a convention that does not exist. Kept working because
        campaign drivers and artifact-reconciliation scripts call it.
        """
        warnings.warn(
            "wfd_apostrophe_pinned names a bug, not a convention: the board it scores is "
            "not a permutation (';' lands on dof 0, the slot-0 character is evicted, and "
            "'q' is duplicated). Use wfd() for anything new, or wfd_legacy_board() to "
            "reconcile a frozen artifact.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wfd_legacy_board(lay30)

    def score_primed(self, lay30: str) -> int:
        """oxey2' (KAN-PRIME-1): strain residual — the stretch term only, native
        weights; the wfd time proxy is superseded by the measured surfaces."""
        return self.components(lay30)["stretch"]


_BAD = {0, 1, 2, 7, 8, 9}  # v1: non-index, non-thumb fingers


def _v1_pattern(f1: int, f2: int, f3: int) -> str | None:
    """oxeylyzer-1 trigram_patterns.rs classification, by finger enums."""
    h1, h2, h3 = f1 > 3, f2 > 3, f3 > 3
    if (h1, h2, h3) in ((False, True, False), (True, False, True)):
        return "alternates_sfs" if f1 == f3 else "alternates"
    if h1 == h2 == h3:
        if f1 == f2 == f3 or f1 == f2 or f2 == f3:
            return None  # Sft / BadSfb: no trigram weight
        if (f1 < f2) == (f2 > f3):
            bad = f1 in _BAD and f2 in _BAD and f3 in _BAD
            sfs = f1 == f3
            if bad:
                return "bad_redirects_sfs" if sfs else "bad_redirects"
            return "redirects_sfs" if sfs else "redirects"
        return "onehands"
    if f1 == f2 or f2 == f3:
        return None  # Sfb
    if (h1, h2, h3) == (False, False, True):
        inroll = f1 < f2
    elif (h1, h2, h3) == (True, False, False):
        inroll = f2 < f3
    elif (h1, h2, h3) == (True, True, False):
        inroll = f1 > f2
    elif (h1, h2, h3) == (False, True, True):
        inroll = f2 > f3
    else:  # pragma: no cover — all hand patterns enumerated above
        return None
    return "inrolls" if inroll else "outrolls"


class Oxeylyzer1:
    """oxeylyzer-1 displayed Score = ``score_with_precision(usize::MAX)`` (higher = better)."""

    W = dict(sfbs=-7.0, sfs=-1.0, stretches=-0.3)
    WT = dict(
        pinky_ring_bigrams=-20,
        inrolls=250,
        outrolls=240,
        onehands=90,
        alternates=40,
        alternates_sfs=10,
        redirects=-340,
        redirects_sfs=-420,
        bad_redirects=-490,
        bad_redirects_sfs=-550,
    )  # scale(x) = int(x*100)
    FW = {0: 1.4, 1: 3.6, 2: 4.8, 3: 5.5, 6: 5.5, 7: 4.8, 8: 3.6, 9: 1.4}
    MAXFW = 5.5

    def __init__(self, chars31: list[str]):
        d = _load_vendored("oxeylyzer1-english.json.gz")
        self.chars = list(chars31)
        idx = {c: k for k, c in enumerate(self.chars)}
        cf = 0.01  # convert_f = f/100
        B = _load_freq_matrix(d["bigrams"], self.chars, cf * d["bigram_total"])
        S1 = _load_freq_matrix(d["skipgrams"], self.chars, cf * d["skipgram_total"])
        S2 = _load_freq_matrix(d["skipgrams2"], self.chars, cf * d["skipgram2_total"])
        S3 = _load_freq_matrix(d["skipgrams3"], self.chars, cf * d["skipgram3_total"])
        self.B = B
        r = self.W["sfs"] / self.W["sfbs"]  # 1/7 dsfb decay
        mix = B + S1 * r + S2 * r**2 + S3 * r**3
        sfwb = (mix * self.W["sfbs"]).astype(np.int64)
        swb = (mix * self.W["stretches"]).astype(np.int64)
        self.SFW = sfwb + sfwb.T  # symmetrized data-side (analyzer_data.rs)
        self.STW = swb + swb.T
        sf = _samefinger_pairs()
        self.SF_I = np.array([i for i, _, _ in sf])
        self.SF_J = np.array([j for _, j, _ in sf])
        self.SF_D = np.array(
            [int(dist * 100.0 * (self.MAXFW / self.FW[FINGERS[i]])) for i, _, dist in sf],
            dtype=np.int64,
        )
        st = _stretch_pairs()
        self.ST_I = np.array([i for i, _, _ in st])
        self.ST_J = np.array([j for _, j, _ in st])
        self.ST_D = np.array([dint for _, _, dint in st], dtype=np.int64)
        pr = [
            (i, j)
            for i in range(N31)
            for j in range(N31)
            if i != j and _HAND[i] == _HAND[j] and {FINGERS[i], FINGERS[j]} in ({0, 1}, {8, 9})
        ]
        self.PR_I = np.array([i for i, _ in pr])
        self.PR_J = np.array([j for _, j in pr])
        keep = [(t, f) for t, f in d["trigrams"].items() if all(c in idx for c in t)]
        self.T_C = np.array([[idx[t[0]], idx[t[1]], idx[t[2]]] for t, _ in keep])
        self.T_F = np.array([int(f * cf * d["trigram_total"]) for _, f in keep], dtype=np.int64)
        PW = np.zeros((N31, N31, N31), dtype=np.int64)
        PW_RED = np.zeros((N31, N31, N31), dtype=np.int64)
        _red = {"redirects", "redirects_sfs", "bad_redirects", "bad_redirects_sfs"}
        for i in range(N31):
            for j in range(N31):
                for k in range(N31):
                    pat = _v1_pattern(FINGERS[i], FINGERS[j], FINGERS[k])
                    if pat:
                        PW[i, j, k] = self.WT[pat]
                        if pat in _red:
                            PW_RED[i, j, k] = self.WT[pat]
        self.PW = PW
        self.PW_RED = PW_RED

    def components(self, lay30: str) -> dict[str, int]:
        """Exact term split: ``score == fspeed + stretch + pinky_ring + trigrams``.
        fspeed is the same-finger time proxy (T); trigrams is the flow-preference
        table (S; its redirect-only part reported for the +R sensitivity);
        stretch and pinky_ring are the mechanical-strain terms (C)."""
        cad, dof = _dof_arrays(lay30, self.chars)
        a, b = cad[self.SF_I], cad[self.SF_J]
        fspeed = int((self.SFW[a, b] * self.SF_D).sum())
        a, b = cad[self.ST_I], cad[self.ST_J]
        stretch = int((self.STW[a, b] * self.ST_D).sum())
        a, b = cad[self.PR_I], cad[self.PR_J]
        pinky_ring = int(self.B[a, b].sum()) * self.WT["pinky_ring_bigrams"]
        t0, t1, t2 = dof[self.T_C[:, 0]], dof[self.T_C[:, 1]], dof[self.T_C[:, 2]]
        return {
            "fspeed": fspeed,
            "stretch": stretch,
            "pinky_ring": pinky_ring,
            "trigrams": int((self.T_F * self.PW[t0, t1, t2]).sum()),
            "trigrams_redirect_part": int((self.T_F * self.PW_RED[t0, t1, t2]).sum()),
        }

    def score(self, lay30: str) -> int:
        c = self.components(lay30)
        return c["fspeed"] + c["stretch"] + c["pinky_ring"] + c["trigrams"]

    def score_primed(self, lay30: str, keep_redirects: bool = False) -> int:
        """oxey1' (KAN-PRIME-1): strain residual — stretch + pinky_ring, native
        weights. ``keep_redirects`` is the registered +R sensitivity variant
        (redirect penalties read as discomfort rather than taste)."""
        c = self.components(lay30)
        return (
            c["stretch"] + c["pinky_ring"] + (c["trigrams_redirect_part"] if keep_redirects else 0)
        )


class Genkey:
    """genkey ``Score`` (lower = better), on genkey's own parsed corpus."""

    KPS = [1.5, 3.6, 4.8, 5.5, 5.5, 4.8, 3.6, 1.5]
    FSPEED_W, LSB_W, IDX_W = 3.0, 1.0, 0.3
    SFB_W, DSFB_W, KEYTRAVEL, LATERAL = 1.0, 0.5, 0.01, 1.4
    COL_FINGER = [0, 1, 2, 3, 3, 4, 4, 5, 6, 7]

    def __init__(self) -> None:
        d = _load_vendored("genkey-keybo.json.gz")
        self.B = {k: float(v) for k, v in d["bigrams"].items() if len(k) == 2}
        self.S = {k: float(v) for k, v in d["skipgrams"].items() if len(k) == 2}
        self.L = {k: float(v) for k, v in d["letters"].items()}

    def components(self, lay30: str) -> dict[str, float]:
        """Exact term split: ``score == 3.0*fspeed + 1.0*lsb_pct + 0.3*index_imbalance_pct``.
        fspeed is the same-finger time proxy (T); lsb_pct and index_imbalance_pct
        are the mechanical-strain/balance terms (C). Stock config has no trigram
        (flow) term."""
        g = [list(lay30[0:10]), list(lay30[10:20]), list(lay30[20:30])]
        total = sum(self.L.get(g[r][c], 0.0) for r in range(3) for c in range(10))
        if total <= 0:
            inf = float("inf")
            return {"fspeed": inf, "lsb_pct": inf, "index_imbalance_pct": inf}
        fmap: dict[int, list[tuple[int, int]]] = {f: [] for f in range(8)}
        for c in range(10):
            for r in range(3):
                fmap[self.COL_FINGER[c]].append((r, c))
        fs_total = 0.0
        for f, posits in fmap.items():
            s = 0.0
            for i in range(len(posits)):
                for j in range(i, len(posits)):
                    r1, c1 = posits[i]
                    r2, c2 = posits[j]
                    k1, k2 = g[r1][c1], g[r2][c2]
                    sfb = self.B.get(k1 + k2, 0.0)
                    dsfb = self.S.get(k1 + k2, 0.0)
                    if i != j:
                        sfb += self.B.get(k2 + k1, 0.0)
                        dsfb += self.S.get(k2 + k1, 0.0)
                    dx = float(c1 - c2)
                    dy = float(r1 - r2)
                    dist = self.LATERAL * dx * dx + dy * dy + 2 * self.KEYTRAVEL
                    s += (self.SFB_W * sfb + self.DSFB_W * dsfb) * dist
            fs_total += (800.0 * s / total) / self.KPS[f]
        lsb = 0.0
        for fi, fm in ((3, 2), (4, 5)):
            for r1, c1 in fmap[fi]:
                for r2, c2 in fmap[fm]:
                    if abs(c1 - c2) >= 2:
                        k1, k2 = g[r1][c1], g[r2][c2]
                        lsb += self.B.get(k1 + k2, 0.0) + self.B.get(k2 + k1, 0.0)
        left = sum(self.L.get(g[r][c], 0.0) for r, c in fmap[3])
        right = sum(self.L.get(g[r][c], 0.0) for r, c in fmap[4])
        return {
            "fspeed": fs_total,
            "lsb_pct": 100.0 * lsb / total,
            "index_imbalance_pct": abs(100.0 * (right - left) / total),
        }

    def score(self, lay30: str) -> float:
        c = self.components(lay30)
        return (
            self.FSPEED_W * c["fspeed"]
            + self.LSB_W * c["lsb_pct"]
            + self.IDX_W * c["index_imbalance_pct"]
        )

    def score_primed(self, lay30: str) -> float:
        """genkey' (KAN-PRIME-1): strain/balance residual — LSB% + index-balance,
        native weights; the 3x fspeed time proxy is superseded by the measured
        surfaces."""
        c = self.components(lay30)
        return self.LSB_W * c["lsb_pct"] + self.IDX_W * c["index_imbalance_pct"]


@lru_cache(maxsize=4)
def community_suite(pinned: str) -> tuple[Genkey, Oxeylyzer1, Oxeylyzer2]:
    """The three scorers for a given pinned quote-slot character (cached: data loads once).

    ``pinned == ";"`` selects the C30M character universe (26 letters + ``' , . -``
    on the board, ``;`` pinned); ``pinned == "'"`` the classic universe
    (``; , . /`` on the board, ``'`` pinned).
    """
    chars31 = list(C30M_CHARS if pinned == ";" else CLASSIC_CHARS) + [pinned]
    return Genkey(), Oxeylyzer1(chars31), Oxeylyzer2(chars31)
