"""Fast exact evaluator for the OPTEVIDENCE arms.

One pass over a layout produces, EXACTLY (to float round-off):

  * the 14 live evidence gauges (the ones `EvidenceWeights` prices),
  * the evidence score (arm A objective; lower = faster, it is a predicted-time loss),
  * predicted ms/char on the served K31 surface at 90 WPM (arm B objective).

Everything is a bilinear (bigram/skipgram) or trilinear (trigram) form over the
char->slot permutation, because for a FIXED charset every layout covers the same set of
n-grams — so each gauge's denominator is a CONSTANT and only the numerator moves.

Conventions are NOT re-derived. The kernels are built by CALLING the validated predicates
(`keybo.analysis.kmstats._KEYS`/`_is_lsb`/`_trigram_value`, `keybo.features.classify`,
`DEFAULT_COMFORT`, `DEFAULT_OXEY_WEIGHTS`) on every slot pair/triple, so a convention
change upstream changes this evaluator too (trap 28: route through the validated path).

⚠ TWO GEOMETRIES, deliberately. `kmstats` carries its own board (`_KEYS`: x = stagger+col,
rows 0/1/2, `_COL_FINGER`) while oxey/comfort use `ROW_STAGGERED_30` (signed x, rows 3/2/1).
The SLOT ORDER is identical (row-major, top row first), so slot index means the same
physical key in both — but the per-slot attributes do not, and mixing them silently changes
`sfb-dist`, `lsb-dist` and `scissor`. Each family gets kernels from its own geometry.

⚠ THREE DENOMINATORS, deliberately (trap 9). kmstats excludes space-touching n-grams from
BOTH numerator and denominator; oxey's `pattern_shares` denominator INCLUDES them (because
`Layout.has_key(' ')` is True) while its numerators cannot fire on space; `comfort` divides
by the FULL bigram mass (1e9), most of which is never scored. All three are asserted against
the real scorers by `positive_control()`.

MODELLED ONLY: the served surface is a fitted model, not a measurement of realized typing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from keybo.analysis import kmstats as KM
from keybo.analysis.evidence_scorer import LIVE_GAUGES
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.features import classify as C
from keybo.geometry import ROW_STAGGERED_30
from keybo.scoring.comfort import DEFAULT_COMFORT
from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
NS = 31  # 30 letter slots + space at index 30
SPACE = 30

#: kmstats gauges, in the order their kernels are stacked.
KM_BI = ("sfb", "sfb-dist", "lsb", "lsb-dist")
KM_SK = ("sfs", "sfs-dist")
KM_TRI = ("alt", "roll", "sr-roll", "redir")
#: oxey pattern classes that are bigram / skipgram / trigram level.
OX_BI = ("sfb", "alternate", "lsb", "scissor", "inroll", "outroll")
OX_SK = ("dsfb",)
OX_TRI = ("onehand", "redirect", "bad_redirect")


# --------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------
def _km_bigram_kernels() -> np.ndarray:
    """(4, 31, 31) kmstats bigram kernels; zero wherever a slot is space.

    Zeroing space makes the numerator over the space-INCLUSIVE mass identical to the
    numerator over kmstats' own space-exclusive mass, so one histogram serves both
    families (the denominators still differ, and are separate constants)."""
    out = np.zeros((len(KM_BI), NS, NS))
    for s in range(SPACE):
        for t in range(SPACE):
            a, b = KM._KEYS[s], KM._KEYS[t]
            distinct = s != t
            same_finger = distinct and a.finger == b.finger
            lsb = KM._is_lsb(a, b)
            out[0, s, t] = 1.0 if same_finger else 0.0
            out[1, s, t] = KM._distance(a, b) if same_finger else 0.0
            out[2, s, t] = 1.0 if lsb else 0.0
            out[3, s, t] = abs(a.x - b.x) if lsb else 0.0
    return out


def _km_skip_kernels() -> np.ndarray:
    out = np.zeros((len(KM_SK), NS, NS))
    for s in range(SPACE):
        for t in range(SPACE):
            a, b = KM._KEYS[s], KM._KEYS[t]
            same_finger = s != t and a.finger == b.finger
            out[0, s, t] = 1.0 if same_finger else 0.0
            out[1, s, t] = KM._distance(a, b) if same_finger else 0.0
    return out


def _km_trigram_kernels() -> np.ndarray:
    out = np.zeros((len(KM_TRI), NS, NS, NS))
    keys = KM._KEYS
    for s in range(SPACE):
        for t in range(SPACE):
            for u in range(SPACE):
                a, b, c = keys[s], keys[t], keys[u]
                for i, short in enumerate(KM_TRI):
                    out[i, s, t, u] = KM._trigram_value(short, a, b, c)
    return out


def _positions31():
    return (*ROW_STAGGERED_30.slots, ROW_STAGGERED_30.space_position)


def _oxey_bigram_kernels() -> np.ndarray:
    """(6, 31, 31) oxey bigram-class indicators. Space IS allowed to participate — and
    `alternate` genuinely fires on space pairs (hand(0) == 0 -> not same_hand -> ALTERNATE),
    which is the published convention, so it must not be zeroed."""
    g = ROW_STAGGERED_30
    pos = _positions31()
    out = np.zeros((len(OX_BI), NS, NS))
    for s in range(NS):
        for t in range(NS):
            a, b = pos[s], pos[t]
            cls = C.classify_positions(g, a, b)
            out[0, s, t] = 1.0 if (cls is C.BigramClass.SAME_FINGER and a != b) else 0.0
            out[1, s, t] = 1.0 if cls is C.BigramClass.ALTERNATE else 0.0
            out[2, s, t] = 1.0 if C.is_lsb(g, a, b) else 0.0
            out[3, s, t] = 1.0 if C.is_scissor(g, a, b) else 0.0
            out[4, s, t] = 1.0 if C.is_inwards(g, a, b) else 0.0
            out[5, s, t] = 1.0 if C.is_outwards(g, a, b) else 0.0
    return out


def _oxey_skip_kernels() -> np.ndarray:
    g = ROW_STAGGERED_30
    pos = _positions31()
    out = np.zeros((len(OX_SK), NS, NS))
    for s in range(NS):
        for t in range(NS):
            a, b = pos[s], pos[t]
            out[0, s, t] = 1.0 if (g.same_finger(a[0], b[0]) and a != b) else 0.0
    return out


def _oxey_trigram_kernels() -> np.ndarray:
    g = ROW_STAGGERED_30
    pos = _positions31()
    out = np.zeros((len(OX_TRI), NS, NS, NS))
    for s in range(NS):
        for t in range(NS):
            for u in range(NS):
                a, b, c3 = pos[s], pos[t], pos[u]
                ha, hb, hc = g.hand(a[0]), g.hand(b[0]), g.hand(c3[0])
                if not (ha == hb == hc and ha != 0):
                    continue
                d1 = abs(b[0]) - abs(a[0])
                d2 = abs(c3[0]) - abs(b[0])
                if d1 and d2 and (d1 > 0) == (d2 > 0):
                    out[0, s, t, u] = 1.0
                elif d1 and d2:
                    out[1, s, t, u] = 1.0
                    if not any(abs(p[0]) in (1, 2) for p in (a, b, c3)):
                        out[2, s, t, u] = 1.0
    return out


def _comfort_kernels() -> tuple[np.ndarray, np.ndarray]:
    """((31,31) bigram penalty, (31,31) skipgram lag2 penalty), in ms-equivalents."""
    g = ROW_STAGGERED_30
    pos = _positions31()
    w = {name: value for name, (value, _why) in DEFAULT_COMFORT.items()}
    bi = np.zeros((NS, NS))
    sk = np.zeros((NS, NS))
    for s in range(NS):
        for t in range(NS):
            a, b = pos[s], pos[t]
            pen = 0.0
            for p in (a, b):
                if p[1] != 2 and p[1] != 0:
                    pen += w["off_home"] / 2
                if p[1] == 1:
                    pen += w["bottom_row"] / 2
            cls = C.classify_positions(g, a, b)
            if cls is C.BigramClass.SAME_FINGER and a != b:
                pen += w["sfb"]
            if C.is_scissor(g, a, b):
                pen += w["scissor"]
            if C.is_lsb(g, a, b):
                pen += w["lsb"]
            bi[s, t] = pen
            if a != b and g.same_finger(a[0], b[0]):
                sk[s, t] = w["lag2_reuse"]
    return bi, sk


# --------------------------------------------------------------------------------------
# corpus reduction
# --------------------------------------------------------------------------------------
def _char_index() -> dict[str, int]:
    idx = {ch: i for i, ch in enumerate(C30M)}
    idx[" "] = SPACE
    return idx


def _pair_mass(freqs: dict[str, int], allow_space: bool) -> tuple[np.ndarray, float]:
    """(31,31) char-pair mass and its total, over pairs fully typable on a C30M board."""
    idx = _char_index()
    ok = set(C30M) | ({" "} if allow_space else set())
    mass = np.zeros((NS, NS))
    total = 0.0
    for ng, f in freqs.items():
        if len(ng) != 2 or ng[0] not in ok or ng[1] not in ok:
            continue
        mass[idx[ng[0]], idx[ng[1]]] += f
        total += f
    return mass, total


def _triple_mass(freqs: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Sparse trigram mass over C30M+space, plus BOTH totals (with / without space)."""
    idx = _char_index()
    ok = set(C30M) | {" "}
    ii, jj, kk, vv = [], [], [], []
    total_sp = 0.0
    total_nosp = 0.0
    for ng, f in freqs.items():
        if len(ng) != 3 or any(c not in ok for c in ng):
            continue
        ii.append(idx[ng[0]])
        jj.append(idx[ng[1]])
        kk.append(idx[ng[2]])
        vv.append(f)
        total_sp += f
        if " " not in ng:
            total_nosp += f
    return (
        np.array(ii, dtype=np.int32),
        np.array(jj, dtype=np.int32),
        np.array(kk, dtype=np.int32),
        np.array(vv, dtype=np.float64),
        total_sp,
        total_nosp,
    )


# --------------------------------------------------------------------------------------
# the evaluator
# --------------------------------------------------------------------------------------
@dataclass
class Curve:
    metric: str
    form: str
    coeffs: np.ndarray
    knot: float | None
    domain: tuple[float, float]

    def price(self, x: np.ndarray) -> np.ndarray:
        c = self.coeffs
        if self.form == "linear":
            return c[0] + c[1] * x
        if self.form == "quadratic":
            return c[0] + c[1] * x + c[2] * x * x
        return c[0] + c[1] * x + c[2] * np.maximum(x - self.knot, 0.0)


class FastEval:
    """Batch evaluator: layouts (as 31-slot permutations) -> gauges, evidence score, ms/char.

    `perms` is `(B, 31)` int: `perms[b, i]` is the SLOT that char `i` (C30M order, space
    last) occupies in layout `b`. `perms[:, 30] == 30` always (space is fixed).
    """

    def __init__(self, corpus: str | None = None, weights_json: str | Path | None = None,
                 target_wpm: float = 90.0, with_surface: bool = True):
        directory = production_corpus_dir(corpus)
        self.corpus_dir = directory
        bigrams = load_frequencies(str(directory / "bigrams.txt"))
        skipgrams = load_frequencies(str(directory / "1-skip31.txt"))
        trigrams = load_frequencies(str(directory / "trigrams.txt"))
        self.bigram_mass_full = float(sum(bigrams.values()))

        self.F_bi, self.bg_total_sp = _pair_mass(bigrams, allow_space=True)
        _, self.bi_total_nosp = _pair_mass(bigrams, allow_space=False)
        self.F_sk, self.sg_total_sp = _pair_mass(skipgrams, allow_space=True)
        _, self.sk_total_nosp = _pair_mass(skipgrams, allow_space=False)
        self.I, self.J, self.K, self.Fv, self.tg_total_sp, self.tri_total_nosp = _triple_mass(trigrams)

        # ---- stacked bigram/skipgram kernels, flattened to (n, 961) ----
        kb = np.concatenate([_km_bigram_kernels(), _oxey_bigram_kernels()], axis=0)
        ks = np.concatenate([_km_skip_kernels(), _oxey_skip_kernels()], axis=0)
        cbi, csk = _comfort_kernels()
        kb = np.concatenate([kb, cbi[None]], axis=0)
        ks = np.concatenate([ks, csk[None]], axis=0)
        self.KB = kb.reshape(kb.shape[0], -1)  # (11, 961)
        self.KS = ks.reshape(ks.shape[0], -1)  # (4, 961)
        self.nb_km, self.nb_ox = len(KM_BI), len(OX_BI)
        self.ns_km, self.ns_ox = len(KM_SK), len(OX_SK)

        # ---- stacked trigram kernels, flattened to (n, 29791) ----
        kt = np.concatenate([_km_trigram_kernels(), _oxey_trigram_kernels()], axis=0)
        self.nt_km, self.nt_ox = len(KM_TRI), len(OX_TRI)
        kt = kt.reshape(kt.shape[0], -1)

        # ---- served surface as one more trigram "kernel" row ----
        self.with_surface = with_surface
        if with_surface:
            from keybo.analysis.timecard import TimeSurface

            surface = TimeSurface(trigrams, target_wpm=target_wpm)
            S = surface._T2[:, :, None] + surface._Tc  # (31,31,31) ms per trigram occurrence
            kt = np.concatenate([kt, S.reshape(1, -1)], axis=0)
            self.surface_row = kt.shape[0] - 1
            self.covered_mass = self.tg_total_sp  # timecard's `chars` for a C30M layout
        else:
            self.surface_row = None
        self.KT = np.ascontiguousarray(kt.T)  # (29791, n_t) for W @ KT

        # ---- imbalance: linear-then-abs, so it gets its own vector ----
        # m[c] = sum over qualifying oxey bigrams of f/2 * (occurrences of c)
        idx = _char_index()
        m = np.zeros(NS)
        for ng, f in bigrams.items():
            if len(ng) != 2 or any(c not in set(C30M) | {" "} for c in ng):
                continue
            for ch in ng:
                m[idx[ch]] += f / 2.0
        m[SPACE] = 0.0  # space has hand 0 and never enters hand_load
        self.imb_m = m
        self.imb_total = float(m.sum())
        hand = np.array([ROW_STAGGERED_30.hand(p[0]) for p in _positions31()], dtype=np.float64)
        self.slot_hand = hand

        # ---- oxey weights ----
        ow = {name: value for name, (value, _why) in DEFAULT_OXEY_WEIGHTS.items()}
        self.oxw_bi = np.array([ow[n] for n in OX_BI])
        self.oxw_sk = np.array([ow[n] for n in OX_SK])
        self.oxw_tri = np.array([ow[n] for n in OX_TRI])
        self.oxw_imb = ow["imbalance"]

        # ---- evidence curves ----
        self.curves: list[Curve] | None = None
        self.weights_meta: dict | None = None
        if weights_json is not None:
            self.load_weights(weights_json)

    # -- weights ----------------------------------------------------------------------
    def load_weights(self, path: str | Path) -> None:
        blob = json.load(open(path))
        w = blob["weights"] if "weights" in blob and "weights" in blob["weights"] else blob
        table = w["weights"]
        by_metric = {g["metric"]: g for g in table}
        assert set(by_metric) == set(LIVE_GAUGES), (
            f"weights cover {sorted(by_metric)} but LIVE_GAUGES is {sorted(LIVE_GAUGES)}"
        )
        self.curves = [
            Curve(
                metric=name,
                form=by_metric[name]["form"],
                coeffs=np.array(by_metric[name]["coeffs"], dtype=np.float64),
                knot=by_metric[name]["knot"],
                domain=tuple(by_metric[name]["valid_domain"]),
            )
            for name in LIVE_GAUGES
        ]
        self.weights_meta = {
            k: w[k] for k in ("source", "surface_frame", "corpus", "pool", "n_layouts",
                              "effective_dof", "base_value_ms_per_trigram")
            if k in w
        }

    # -- gauges -----------------------------------------------------------------------
    def gauges(self, perms: np.ndarray) -> dict[str, np.ndarray]:
        """`(B, 31)` permutations -> dict of gauge -> `(B,)` values, in analyzer units."""
        perms = np.ascontiguousarray(perms, dtype=np.int32)
        B = perms.shape[0]
        inv = np.empty_like(perms)
        np.put_along_axis(inv, perms, np.broadcast_to(np.arange(NS, dtype=np.int32), perms.shape), axis=1)

        # bigram / skipgram: G2[s,t] = F[inv_s, inv_t]
        G2 = self.F_bi[inv[:, :, None], inv[:, None, :]].reshape(B, -1)
        G2s = self.F_sk[inv[:, :, None], inv[:, None, :]].reshape(B, -1)
        VB = G2 @ self.KB.T   # (B, 11)
        VS = G2s @ self.KS.T  # (B, 4)

        # trigram: histogram over the 29791 slot-triples
        flat = (perms[:, self.I].astype(np.int64) * NS + perms[:, self.J]) * NS + perms[:, self.K]
        W = np.zeros((B, NS ** 3))
        for b in range(B):
            W[b] = np.bincount(flat[b], weights=self.Fv, minlength=NS ** 3)
        VT = W @ self.KT  # (B, n_t)

        out: dict[str, np.ndarray] = {}
        for i, name in enumerate(KM_BI):
            out[name] = 100.0 * VB[:, i] / self.bi_total_nosp
        for i, name in enumerate(KM_SK):
            out[name] = 100.0 * VS[:, i] / self.sk_total_nosp
        for i, name in enumerate(KM_TRI):
            out[name] = 100.0 * VT[:, i] / self.tri_total_nosp

        ox_bi = 100.0 * VB[:, self.nb_km:self.nb_km + self.nb_ox] / self.bg_total_sp
        ox_sk = 100.0 * VS[:, self.ns_km:self.ns_km + self.ns_ox] / self.sg_total_sp
        ox_tri = 100.0 * VT[:, self.nt_km:self.nt_km + self.nt_ox] / self.tg_total_sp
        signed = (self.imb_m[inv] * self.slot_hand[np.arange(NS)][None, :]).sum(axis=1)
        imbalance = 100.0 * np.abs(signed) / self.imb_total

        out["scissor"] = ox_bi[:, OX_BI.index("scissor")]
        out["imbalance"] = imbalance
        out["oxey-style"] = (
            ox_bi @ self.oxw_bi + ox_sk @ self.oxw_sk + ox_tri @ self.oxw_tri
            + self.oxw_imb * imbalance
        )
        comfort_abs = VB[:, -1] + VS[:, -1]
        out["comfort"] = comfort_abs / self.bigram_mass_full
        if self.surface_row is not None:
            out["_total_ms"] = VT[:, self.surface_row]
            out["_ms_per_char"] = VT[:, self.surface_row] / self.covered_mass
        return out

    # -- objectives -------------------------------------------------------------------
    def evidence_score(self, gauges: dict[str, np.ndarray]) -> np.ndarray:
        assert self.curves is not None, "load_weights() first"
        total = np.zeros_like(gauges[LIVE_GAUGES[0]])
        for curve in self.curves:
            total = total + curve.price(gauges[curve.metric])
        return total

    def out_of_domain(self, gauges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        assert self.curves is not None
        return {
            c.metric: (gauges[c.metric] < c.domain[0]) | (gauges[c.metric] > c.domain[1])
            for c in self.curves
        }

    def evaluate(self, perms: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        g = self.gauges(perms)
        return self.evidence_score(g), g.get("_ms_per_char"), g


# --------------------------------------------------------------------------------------
# layout <-> permutation
# --------------------------------------------------------------------------------------
def perm_of(lay30: str) -> np.ndarray:
    """char (C30M order) -> slot; space pinned at slot 30."""
    if len(lay30) != 30 or set(lay30) != set(C30M):
        raise ValueError(f"not a C30M permutation: {lay30!r}")
    p = np.array([lay30.index(ch) for ch in C30M] + [SPACE], dtype=np.int32)
    assert sorted(p.tolist()) == list(range(NS)), "perm_of produced a non-permutation"
    return p


def layout_of(perm: np.ndarray) -> str:
    """Inverse of :func:`perm_of` — the 30-char row-major layout string."""
    slots = [""] * SPACE
    for i, ch in enumerate(C30M):
        slots[int(perm[i])] = ch
    assert all(slots), "perm did not cover every slot"
    return "".join(slots)
