"""Exact Python ports of oxeylyzer-2 and oxeylyzer-1 scoring (P14 rule 01546c8).

Both tools score a 31-key ANSI/traditional layout (our 30 slots + pinned apostrophe).
Ported from source:
  o2: score_cache = weighted_bigrams.total + stretch_bigrams.total
      (core/src/{analyze,cached_layout,analyzer_data}.rs, analyzer-config.toml weights)
  v1: score_with_precision = trigram_score(top-1000) + stretch + fspeed + pinky_ring
      (oxeylyzer-core/src/{generate,fast_layout,analyzer_data,trigram_patterns}.rs,
       ~/.config/oxeylyzer/config.toml weights; usage term inert at penalty=0)
Key geometry facts (verified in source): 1u keys + KEY_EDGE_OFFSET=0.5 collapse each key
box to its center; flen finger-length y-offsets; signed-dx crossing rule feeds x_overlap.
PARITY GATE (registered): rank corr 1.0 + <=5% ratio spread vs the real binaries on
>=8 layouts, else the gauge is excluded from the in-loop objective.
"""

import json
import math
import os
import subprocess

import numpy as np

O2_ROOT = "/home/zegertho/gk-parity/oxeylyzer-2"
V1_ROOT = "/home/zegertho/gk-parity/oxeylyzer"
V1_LAYOUT_DIR = os.path.expanduser("~/.local/share/oxeylyzer/static/layouts/english")

# ---- shared 31-key geometry (dof order: top 10, home 11 incl ', bottom 10) ---------------
# fingers as libdof enums: LP=0 LR=1 LM=2 LI=3 RI=6 RM=7 RR=8 RP=9
_ROW_FINGERS = [0, 1, 2, 3, 3, 6, 6, 7, 8, 9]
FINGERS = _ROW_FINGERS + _ROW_FINGERS + [9] + _ROW_FINGERS
_X0 = [1.5, 1.75, 2.25]  # ANSI row starts (top/home/bottom), anchor (1,1)
POS = (
    [(1.5 + i, 1.0) for i in range(10)]
    + [(1.75 + i, 2.0) for i in range(11)]
    + [(2.25 + i, 3.0) for i in range(10)]
)  # key LEFT edges; centers = +0.5
FLEN = {0: -0.15, 1: 0.35, 2: 0.25, 3: -0.30, 6: -0.30, 7: 0.25, 8: 0.35, 9: -0.15}
XFO = {(0, 1): 0.8, (1, 2): 0.4, (2, 3): 0.1, (6, 7): 0.1, (7, 8): 0.4, (8, 9): 0.8}
N31 = 31
HAND = [0 if f <= 3 else 1 for f in FINGERS]

# our slot order (30) -> dof position index; apostrophe = dof 20, char index 30
SLOT2DOF = list(range(10)) + list(range(10, 20)) + list(range(21, 31))
APOS_DOF = 20


def _centers():
    cx = np.array([x + 0.5 for x, _ in POS])
    cy = np.array([y + 0.5 for _, y in POS])
    return cx, cy


def _dx_dy(i, j, use_flen):
    """o2/v1 dx_dy for collapsed 1u keys: center deltas, flen y-shift, signed-dx crossing."""
    cx, cy = _centers()
    f1, f2 = FINGERS[i], FINGERS[j]
    y1 = cy[i] + (FLEN[f1] if use_flen else 0.0)
    y2 = cy[j] + (FLEN[f2] if use_flen else 0.0)
    dx = abs(cx[i] - cx[j])
    dy = abs(y1 - y2)
    xo = XFO.get((min(f1, f2), max(f1, f2)), 0.0)
    if f1 > f2 and cx[i] < cx[j] + xo:
        dx = -dx
    elif f1 < f2 and cx[i] + xo > cx[j]:
        dx = -dx
    return dx, dy


def _stretch_pairs():
    """(i, j, stretch_int) per o2/v1 StretchCache: same hand, diff finger, stretch>0.001."""
    out = []
    for i in range(N31):
        for j in range(i + 1, N31):
            f1, f2 = FINGERS[i], FINGERS[j]
            if f1 == f2 or HAND[i] != HAND[j]:
                continue
            dx, dy = _dx_dy(i, j, use_flen=True)
            xo = XFO.get((min(f1, f2), max(f1, f2)), 0.0)
            x_overlap = max(0.0, xo - dx * 1.3 + 0.3333 * dy)
            stretch = math.hypot(dx, dy) + x_overlap - 1.35 * abs(f1 - f2)
            if stretch > 0.001:
                out.append((i, j, int(stretch * 100.0)))
    return out


def _samefinger_pairs():
    """(i, j, plain center dist) for pairs on the same finger (flen cancels)."""
    out = []
    for i in range(N31):
        for j in range(i + 1, N31):
            if FINGERS[i] != FINGERS[j]:
                continue
            dx, dy = _dx_dy(i, j, use_flen=False)
            out.append((i, j, math.hypot(dx, dy)))
    return out


def _load_freq_matrix(dic, chars, scale):
    """char-pair dict -> 31x31 int matrix over our char universe."""
    idx = {c: k for k, c in enumerate(chars)}
    m = np.zeros((len(chars), len(chars)), dtype=np.int64)
    for key, f in dic.items():
        if len(key) == 2 and key[0] in idx and key[1] in idx:
            m[idx[key[0]], idx[key[1]]] = int(f * scale)
    return m


class O2Port:
    """oxeylyzer-2 score_cache for our-shape layouts. Higher = better (negatives)."""

    W_SFB, W_SFS, W_STR = -7, -1, -3
    FW = {0: 77, 1: 32, 2: 24, 3: 21, 6: 21, 7: 24, 8: 32, 9: 77}

    def __init__(self, chars31):
        d = json.load(open(f"{O2_ROOT}/data/english.json"))
        self.chars = chars31
        B = _load_freq_matrix(d["bigrams"], chars31, d["bigram_total"])
        S = _load_freq_matrix(d["skipgrams"], chars31, d["skipgram_total"])
        self.SFW = self.W_SFB * B + self.W_SFS * S
        self.STW = (B + (S * 7.0).astype(np.int64)) * self.W_STR
        sf = _samefinger_pairs()
        self.SF_I = np.array([i for i, _, _ in sf])
        self.SF_J = np.array([j for _, j, _ in sf])
        self.SF_D = np.array([int(dist * 100.0) * self.FW[FINGERS[i]] for i, _, dist in sf],
                             dtype=np.int64)
        st = _stretch_pairs()
        self.ST_I = np.array([i for i, _, _ in st])
        self.ST_J = np.array([j for _, j, _ in st])
        self.ST_D = np.array([dint for _, _, dint in st], dtype=np.int64)

    def score(self, char_at_dof):
        """char_at_dof: int array of 31 char indices by dof position."""
        a, b = char_at_dof[self.SF_I], char_at_dof[self.SF_J]
        wb = ((self.SFW[a, b] + self.SFW[b, a]) * self.SF_D).sum()
        a, b = char_at_dof[self.ST_I], char_at_dof[self.ST_J]
        stw = ((self.STW[a, b] + self.STW[b, a]) * self.ST_D).sum()
        return int(wb + stw)

    def wfd(self, char_at_dof):
        """weighted_bigrams total alone (o2 'weighted finger distance')."""
        a, b = char_at_dof[self.SF_I], char_at_dof[self.SF_J]
        return int(((self.SFW[a, b] + self.SFW[b, a]) * self.SF_D).sum())


# ---- v1 trigram pattern classification (trigram_patterns.rs) ------------------------------
_BAD = {0, 1, 2, 7, 8, 9}  # non-index, non-thumb


def _v1_pattern(f1, f2, f3):
    h1, h2, h3 = f1 > 3, f2 > 3, f3 > 3  # False=left (enums 0-3), True=right (6-9)
    if (h1, h2, h3) in ((False, True, False), (True, False, True)):
        return "alternates_sfs" if f1 == f3 else "alternates"
    if h1 == h2 == h3:
        if f1 == f2 == f3:
            return None  # Sft: no trigram weight
        if f1 == f2 or f2 == f3:
            return None  # BadSfb
        redir = (f1 < f2) == (f2 > f3)
        if redir:
            bad = f1 in _BAD and f2 in _BAD and f3 in _BAD
            sfs = f1 == f3
            return ("bad_redirects_sfs" if sfs else "bad_redirects") if bad else (
                "redirects_sfs" if sfs else "redirects")
        return "onehands"
    if f1 == f2 or f2 == f3:
        return None  # Sfb
    if (h1, h2, h3) in ((False, False, True), (True, False, False),
                        (True, True, False), (False, True, True)):
        if (h1, h2, h3) == (False, False, True):
            inroll = f1 < f2
        elif (h1, h2, h3) == (True, False, False):
            inroll = f2 < f3
        elif (h1, h2, h3) == (True, True, False):
            inroll = f1 > f2
        else:
            inroll = f2 > f3
        return "inrolls" if inroll else "outrolls"
    return None  # Other


class V1Port:
    """oxeylyzer-1 score_with_precision(1000) for our-shape layouts. Higher = better."""

    W = dict(sfbs=-7.0, sfs=-1.0, stretches=-0.3)
    WT = dict(pinky_ring_bigrams=-20, inrolls=250, outrolls=240, onehands=90,
              alternates=40, alternates_sfs=10, redirects=-340, redirects_sfs=-420,
              bad_redirects=-490, bad_redirects_sfs=-550)  # scale(x)=int(x*100)
    FW = {0: 1.4, 1: 3.6, 2: 4.8, 3: 5.5, 6: 5.5, 7: 4.8, 8: 3.6, 9: 1.4}
    MAXFW = 5.5

    def __init__(self, chars31, trigram_precision=None):
        # None = all trigrams: the repl's displayed Score (our quoted gauge) is
        # score_with_precision(usize::MAX); the top-1000 cut is generation-only.
        d = json.load(open(os.path.expanduser(
            "~/.local/share/oxeylyzer/static/language_data/english.json")))
        self.chars = chars31
        idx = {c: k for k, c in enumerate(chars31)}
        cf = 0.01  # convert_f = f/100
        B = _load_freq_matrix(d["bigrams"], chars31, cf * d["bigram_total"])
        S1 = _load_freq_matrix(d["skipgrams"], chars31, cf * d["skipgram_total"])
        S2 = _load_freq_matrix(d["skipgrams2"], chars31, cf * d["skipgram2_total"])
        S3 = _load_freq_matrix(d["skipgrams3"], chars31, cf * d["skipgram3_total"])
        self.B = B
        r = self.W["sfs"] / self.W["sfbs"]  # 1/7
        mix = B + S1 * r + S2 * r**2 + S3 * r**3
        sfwb = (mix * self.W["sfbs"]).astype(np.int64)
        swb = (mix * self.W["stretches"]).astype(np.int64)
        self.SFW = sfwb + sfwb.T  # symmetrized data-side (analyzer_data.rs:172-179)
        self.STW = swb + swb.T
        sf = _samefinger_pairs()
        self.SF_I = np.array([i for i, _, _ in sf])
        self.SF_J = np.array([j for _, j, _ in sf])
        self.SF_D = np.array(
            [int(dist * 100.0 * (self.MAXFW / self.FW[FINGERS[i]])) for i, _, dist in sf],
            dtype=np.int64)
        st = _stretch_pairs()
        self.ST_I = np.array([i for i, _, _ in st])
        self.ST_J = np.array([j for _, j, _ in st])
        self.ST_D = np.array([dint for _, _, dint in st], dtype=np.int64)
        pr = [(i, j) for i in range(N31) for j in range(N31) if i != j
              and HAND[i] == HAND[j] and {FINGERS[i], FINGERS[j]} in ({0, 1}, {8, 9})]
        self.PR_I = np.array([i for i, _ in pr])
        self.PR_J = np.array([j for _, j in pr])
        # top-N trigrams in corpus (json) order, then on-layout filter — mirrors take(N)
        tris = list(d["trigrams"].items())
        if trigram_precision is not None:
            tris = tris[:trigram_precision]
        keep = [(t, f) for t, f in tris if all(c in idx for c in t)]
        self.T_C = np.array([[idx[t[0]], idx[t[1]], idx[t[2]]] for t, _ in keep])
        self.T_F = np.array([int(f * cf * d["trigram_total"]) for _, f in keep],
                            dtype=np.int64)
        PW = np.zeros((N31, N31, N31), dtype=np.int64)
        for i in range(N31):
            for j in range(N31):
                for k in range(N31):
                    pat = _v1_pattern(FINGERS[i], FINGERS[j], FINGERS[k])
                    if pat:
                        PW[i, j, k] = self.WT[pat]
        self.PW = PW

    def score(self, char_at_dof, dof_of_char):
        a, b = char_at_dof[self.SF_I], char_at_dof[self.SF_J]
        fspeed = (self.SFW[a, b] * self.SF_D).sum()
        a, b = char_at_dof[self.ST_I], char_at_dof[self.ST_J]
        stretch = (self.STW[a, b] * self.ST_D).sum()
        a, b = char_at_dof[self.PR_I], char_at_dof[self.PR_J]
        pinky_ring = int(self.B[a, b].sum()) * self.WT["pinky_ring_bigrams"]
        tri = (self.T_F * self.PW[dof_of_char[self.T_C[:, 0]],
                                  dof_of_char[self.T_C[:, 1]],
                                  dof_of_char[self.T_C[:, 2]]]).sum()
        return int(fspeed + stretch + pinky_ring + tri)


# ---- layout plumbing -----------------------------------------------------------------------
def perm_arrays(lay30, chars31):
    """our 30-char layout string -> (char_at_dof[31], dof_of_char[31])."""
    idx = {c: k for k, c in enumerate(chars31)}
    dof_of_char = np.zeros(N31, dtype=np.int64)
    for slot, c in enumerate(lay30):
        dof_of_char[idx[c]] = SLOT2DOF[slot]
    dof_of_char[idx["'"]] = APOS_DOF
    char_at_dof = np.zeros(N31, dtype=np.int64)
    char_at_dof[dof_of_char] = np.arange(N31)
    return char_at_dof, dof_of_char


def write_dof(lay30, name, path):
    rows = [" ".join(lay30[0:10]), " ".join(list(lay30[10:20]) + ["'"]),
            " ".join(lay30[20:30])]
    dof = {"name": name, "authors": ["keybo"], "board": "ansi", "year": 2026,
           "layers": {"main": rows}, "fingering": "traditional"}
    json.dump(dof, open(path, "w"), indent=1)


def _zip_scores(out, names, cast):
    vals = []
    for line in out.splitlines():
        t = line.strip().lstrip("> ").strip().lower()
        if t.startswith("score:"):
            vals.append(cast(t.split()[-1].replace(",", "")))
    if len(vals) != len(names):
        raise RuntimeError(f"expected {len(names)} score lines, got {len(vals)}")
    return dict(zip(names, vals))


def repl_scores_o2(names):
    cmds = "".join(f"analyze {n}\n" for n in names) + "q\n"
    out = subprocess.run(["./target/release/repl"], input=cmds, cwd=O2_ROOT,
                         capture_output=True, text=True, timeout=600).stdout
    return _zip_scores(out, names, lambda s: int(float(s)))


def repl_scores_v1(names):
    cmds = "".join(f"analyze {n}\n" for n in names) + "q\n"
    out = subprocess.run(["./target/release/oxeylyzer"], input=cmds, cwd=V1_ROOT,
                         capture_output=True, text=True, timeout=600).stdout
    return _zip_scores(out, names, float)
