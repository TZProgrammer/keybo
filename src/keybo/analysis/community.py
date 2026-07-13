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

Layouts are our canonical 30-char row-major strings. The oxeylyzer boards are
31-key (they see a pinned character on the home-row quote slot): C30M-charset
layouts pin ``;`` there and classic-charset layouts pin ``'`` — chosen
automatically from the layout string, matching how semimak/graphite's own .dof
files encode the same convention.
"""

from __future__ import annotations

import gzip
import json
import math
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


def _dof_arrays(lay30: str, chars31: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(char_at_dof, dof_of_char) index arrays for a 30-char layout + pinned char."""
    idx = {c: k for k, c in enumerate(chars31)}
    dof_of_char = np.empty(N31, dtype=np.int64)
    for slot, ch in enumerate(lay30):
        dof_of_char[idx[ch]] = SLOT2DOF[slot]
    dof_of_char[idx[chars31[30]]] = APOS_DOF
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

    def score(self, lay30: str) -> int:
        cad, _ = _dof_arrays(lay30, self.chars)
        a, b = cad[self.SF_I], cad[self.SF_J]
        wb = int(((self.SFW[a, b] + self.SFW[b, a]) * self.SF_D).sum())
        a, b = cad[self.ST_I], cad[self.ST_J]
        return wb + int(((self.STW[a, b] + self.STW[b, a]) * self.ST_D).sum())

    def wfd(self, lay30: str) -> int:
        """The weighted-(same-)finger-distance component alone."""
        cad, _ = _dof_arrays(lay30, self.chars)
        a, b = cad[self.SF_I], cad[self.SF_J]
        return int(((self.SFW[a, b] + self.SFW[b, a]) * self.SF_D).sum())


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
        for i in range(N31):
            for j in range(N31):
                for k in range(N31):
                    pat = _v1_pattern(FINGERS[i], FINGERS[j], FINGERS[k])
                    if pat:
                        PW[i, j, k] = self.WT[pat]
        self.PW = PW

    def score(self, lay30: str) -> int:
        cad, dof = _dof_arrays(lay30, self.chars)
        a, b = cad[self.SF_I], cad[self.SF_J]
        fspeed = (self.SFW[a, b] * self.SF_D).sum()
        a, b = cad[self.ST_I], cad[self.ST_J]
        stretch = (self.STW[a, b] * self.ST_D).sum()
        a, b = cad[self.PR_I], cad[self.PR_J]
        pinky_ring = int(self.B[a, b].sum()) * self.WT["pinky_ring_bigrams"]
        tri = (
            self.T_F * self.PW[dof[self.T_C[:, 0]], dof[self.T_C[:, 1]], dof[self.T_C[:, 2]]]
        ).sum()
        return int(fspeed + stretch + pinky_ring + tri)


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

    def score(self, lay30: str) -> float:
        g = [list(lay30[0:10]), list(lay30[10:20]), list(lay30[20:30])]
        total = sum(self.L.get(g[r][c], 0.0) for r in range(3) for c in range(10))
        if total <= 0:
            return float("inf")
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
        return (
            self.FSPEED_W * fs_total
            + self.LSB_W * 100.0 * lsb / total
            + self.IDX_W * abs(100.0 * (right - left) / total)
        )


@lru_cache(maxsize=4)
def community_suite(pinned: str) -> tuple[Genkey, Oxeylyzer1, Oxeylyzer2]:
    """The three scorers for a given pinned quote-slot character (cached: data loads once).

    ``pinned == ";"`` selects the C30M character universe (26 letters + ``' , . -``
    on the board, ``;`` pinned); ``pinned == "'"`` the classic universe
    (``; , . /`` on the board, ``'`` pinned).
    """
    c30m = "qwertyuiopasdfghjkl'zxcvbnm,.-"
    classic = "qwertyuiopasdfghjkl;zxcvbnm,./"
    chars31 = list(c30m if pinned == ";" else classic) + [pinned]
    return Genkey(), Oxeylyzer1(chars31), Oxeylyzer2(chars31)
