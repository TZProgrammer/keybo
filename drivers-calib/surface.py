"""The surface layer: per-seed (T2, Tc) tables, the same-finger MASK, and board scoring.

One place that (a) loads the 3 shipped seeds from data/models/k31 (READ-ONLY) and the 22 rescued
tables, (b) builds the same-finger cell mask from the GEOMETRY (not from a hardcoded list), and
(c) scores a board on a (T2, Tc) pair. Every driver reads this, so the corrected and uncorrected
paths cannot drift into two different objectives.
"""
import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
from _guard import MIN_N, SEEDS, SEEDTABLES, SHIPPED, WPM  # noqa: F401


def geometry():
    from keybo.geometry import ROW_STAGGERED_30 as G
    return G


def positions():
    G = geometry()
    return [*G.slots, G.space_position]


def load_shipped_model(stem):
    """Inflate a vendored .gz model pair into a temp dir and load it. NEVER writes SHIPPED."""
    from keybo.models.xgboost_model import XGBoostTypingModel
    with tempfile.TemporaryDirectory() as td:
        for suf in (".json", ".meta.json"):
            with gzip.open(f"{SHIPPED}/{stem}{suf}.gz", "rb") as s, \
                 open(Path(td) / f"{stem}{suf}", "wb") as d:
                shutil.copyfileobj(s, d)
        return XGBoostTypingModel.load(str(Path(td) / f"{stem}.json"))


_VECS = None


def _trigram_vecs():
    global _VECS
    if _VECS is None:
        from keybo.features import trigram_features_from_positions
        G, POS = geometry(), positions()
        _VECS = np.vstack([trigram_features_from_positions(G, (a, b, c), wpm=WPM)
                           for a in POS for b in POS for c in POS])
    return _VECS


def build_tables_from_models(bi, tri):
    """(T2, Tc) for one seed pair, exactly as timecard.TimeSurface / tournament's build() do."""
    from keybo.scoring.table_scorer import TableBigramScorer
    G, POS = geometry(), positions()
    n = len(POS)
    ph = "qwertyuiopasdfghjkl;zxcvbnm,./'"[: len(G.slots)]
    T2 = np.asarray(TableBigramScorer(bi, {}, target_wpm=WPM, chars=ph, geometry=G)._T, float)
    Tc = np.asarray(tri.predict_ms(_trigram_vecs()).reshape(n, n, n), float)
    return T2, Tc


def load_all_seed_tables(seeds=SEEDS, verbose=True):
    """Per-seed (T2, Tc): seeds 0-2 rebuilt from the SHIPPED models, 3-24 from the rescued npz."""
    T2s, Tcs = [], []
    for s in seeds:
        if s <= 2:
            T2, Tc = build_tables_from_models(load_shipped_model(f"bigram_reg31_seed{s}"),
                                             load_shipped_model(f"trigram_cond31_seed{s}"))
            src = "SHIPPED models"
        else:
            z = np.load(f"{SEEDTABLES}/tables_seed{s}.npz")
            T2, Tc = np.asarray(z["T2"], float), np.asarray(z["Tc"], float)
            src = "rescued npz"
        if verbose:
            print(f"  seed{s:<3d} from {src:<14s} T2{T2.shape} Tc{Tc.shape}")
        T2s.append(T2)
        Tcs.append(Tc)
    return T2s, Tcs


def same_finger_mask():
    """(n,n) bool: TRUE exactly where the model's own `same_finger` feature fires, a != b.

    Derived from Geometry.same_finger via features.classify (the SAME predicate the feature row
    uses), so the corrected cells are by construction the cells the feature marks -- not a
    hand-listed set that could drift from it. Space (index n-1) is excluded because
    Geometry.same_finger returns False for the thumb, and pick2 excluded space-touching pairs
    as a different motor act.
    """
    from keybo.features import classify as C
    G, POS = geometry(), positions()
    n = len(POS)
    M = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and C.same_finger(G, POS[i], POS[j]):
                M[i, j] = True
    return M


def corrected_T2(T2, delta, mask=None, mode="add"):
    """T2 with a same-finger surcharge. `add`: +delta ms. `mul`: x delta (the robustness arm)."""
    M = same_finger_mask() if mask is None else mask
    out = T2.copy()
    if mode == "add":
        out[M] = out[M] + delta
    elif mode == "mul":
        out[M] = out[M] * delta
    else:
        raise ValueError(f"unknown mode {mode!r} (known: add, mul)")
    return out


def corpus(name=None):
    from keybo.data.corpus import load_frequencies, production_corpus_dir
    d = production_corpus_dir(name)
    tri = {k: v for k, v in load_frequencies(str(d / "trigrams.txt")).items() if len(k) == 3}
    return d, tri


def board_arrays(lay30, tri_freq, ngrams=None):
    """(i, j, k, f) index+frequency arrays for one board over the corpus rows it can type."""
    POS = positions()
    n = len(POS)
    slot = {ch: i for i, ch in enumerate(lay30)}
    slot[" "] = n - 1
    A, B, C_, F = [], [], [], []
    for ng in (tri_freq if ngrams is None else ngrams):
        try:
            a, b, c = slot[ng[0]], slot[ng[1]], slot[ng[2]]
        except KeyError:
            continue
        A.append(a); B.append(b); C_.append(c); F.append(tri_freq[ng])
    return (np.array(A, np.intp), np.array(B, np.intp), np.array(C_, np.intp),
            np.array(F, float))


def mspc(arr, T2, Tc):
    """ms/char: the corpus total over the mass this board covers -- TimeCard.ms_per_char."""
    a, b, c, f = arr
    return float(((T2[a, b] + Tc[a, b, c]) * f).sum() / f.sum())


def sf_share(arr, mask=None):
    """The board's SAME-FINGER share of the trigram-weighted BIGRAM term.

    The fraction of corpus mass whose FIRST TRANSITION (a->b, the pair T2 prices) is
    same-finger. This is the analytic multiplier in gate A3: a +delta surcharge on those cells
    raises ms/char by exactly delta * this share. It is NOT the `sfb` percentage -- that is a
    bigram-corpus count with a letters-only denominator (kmstats), a different quantity.
    """
    M = same_finger_mask() if mask is None else mask
    a, b, _c, f = arr
    return float(f[M[a, b]].sum() / f.sum())
