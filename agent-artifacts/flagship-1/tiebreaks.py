"""FLAGSHIP-1 — the non-speed tie-breaks, plus a SECOND, INDEPENDENT resolution channel.

Three parts, each answering a distinct question the adoption decision needs.

PART A — SWITCHING COST, re-derived for all six.
`all-gauge-table.json` carries a `switching_cost` block but has NO row for flagship-c3, so
that number was never computed (trap 19: a metric absent from a published JSON was never
computed). Re-derived here for all six against qwerty30m (the C30M qwerty, so the comparison
is a permutation of one charset rather than a charset change):
  * unchanged_keys   — same character on the same slot
  * same_finger_keys — character stays on the same finger
  * same_hand_keys   — character stays on the same hand
  * zxcv_preserved   — how many of z/x/c/v keep their qwerty slot (the shortcut cluster)
The published values for the five that ARE in the artifact are the positive control: they
must reproduce, or my definition is not theirs.

PART B — PER-FINGER STRAIN and PATHOLOGY.
Predicted per-finger time share (from the time card's exact partition) plus the max-finger
load and the pinky load, per corpus. A pathological axis is defined mechanically: a
(corpus, gauge) cell where the layout is field-worst of six AND its ceiling-fraction
normalized position is below the field's 10th percentile over all cells. Reported, not
asserted.

PART C — A CORPUS BOOTSTRAP: a resolution channel that uses NO seeds.
The campaign's floor is a per-SEED spread — it measures fit noise in the timing model. The
adoption question also has a second uncertainty channel nobody has instrumented: is the
ordering an artifact of WHICH n-grams happen to be in the corpus? So: resample the trigram
count vector multinomially at the observed total mass, B times, and recompute every layout's
ms/char on the SAME draw (a paired bootstrap — the draw is common mode and cancels the way a
seed does). This is independent of the seed channel: it holds the model fixed and varies the
data, where the floor holds the data fixed and varies the model.

All six candidates are C30M, so they cover exactly the same trigrams and the ms/char
denominators are identical within a draw — which is what makes the paired difference exact
rather than a coverage artifact (the `saved_vs_ref_pct` coverage trap does not arise here
because no qwerty reference enters).

MODELED ONLY. Both channels bound INSTRUMENT resolution. Neither is a claim about realized
typing speed: held-layout tau is saturated at 1.0 and Phase-D is cancelled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append("/tmp/flagship/src")

import keybo  # noqa: E402
from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies  # noqa: E402

assert keybo.__file__.startswith("/tmp/flagship/"), f"NOT the worktree: {keybo.__file__}"

CAND = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
}
EXTRA = {
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
}
QWERTY30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
CORPORA = {
    "iweb": "/tmp/flagship/data/corpus",
    "blend-v1": "/tmp/flagship/data/corpus/blend-v1",
    "blend-v1-no-anchor": "/tmp/flagship-corpora/blend-v1-no-anchor",
}
# all-gauge-table.json's published block — the positive control for PART A.
PUBLISHED_SWITCHING = {
    "keybo-lsb": {"same_finger_keys": 1, "same_hand_keys": 6, "unchanged_keys": 0, "zxcv_preserved": 0},
    "keybo-lsb+lm": {"same_finger_keys": 1, "same_hand_keys": 6, "unchanged_keys": 0, "zxcv_preserved": 0},
    "lsb-sib": {"same_finger_keys": 1, "same_hand_keys": 8, "unchanged_keys": 1, "zxcv_preserved": 1},
    "archive-1843": {"same_finger_keys": 1, "same_hand_keys": 6, "unchanged_keys": 1, "zxcv_preserved": 0},
    "archive-1846": {"same_finger_keys": 0, "same_hand_keys": 8, "unchanged_keys": 0, "zxcv_preserved": 0},
    "graphite": {"same_finger_keys": 8, "same_hand_keys": 24, "unchanged_keys": 4, "zxcv_preserved": 0},
    "semimak": {"same_finger_keys": 6, "same_hand_keys": 18, "unchanged_keys": 4, "zxcv_preserved": 0},
}


def switching_cost(lay30: str) -> dict:
    """Route through the SHIPPED, artifact-validated constructor — never a hand-roll.

    Trap 28: a hand-rolled reimplementation of a validated constructor loses the validation.
    `keybo.analysis.select.switching_costs` is the exact function `all-gauge-table` called,
    so the seven published rows are a real positive control on it. My first attempt
    hand-rolled a geometry-based version and it disagreed on `same_hand` for all 7 rows
    (30 for every layout) — the published definition counts SLOT COLUMNS (`slot % 10`),
    which is the reference-layout-relative notion adoption friction actually needs.
    """
    from keybo.analysis.select import switching_costs

    v = switching_costs(lay30, QWERTY30M)
    return {
        "same_finger_keys": v["same_finger_keys"],
        "same_hand_keys": v["same_hand_keys"],
        "unchanged_keys": v["unchanged_keys"],
        "zxcv_preserved": v["zxcv_preserved"],
    }


def main() -> None:
    out: dict = {
        "what": "FLAGSHIP-1 tie-breaks: switching cost, per-finger strain, corpus bootstrap",
        "modeled_only": (
            "every number is a gauge/model output; tau saturated at 1.0, Phase-D cancelled — "
            "no claim about realized typing speed"
        ),
    }

    # ---- PART A ---------------------------------------------------------------------------
    sc = {n: switching_cost(l) for n, l in {**CAND, **EXTRA}.items()}
    sc["qwerty30m"] = switching_cost(QWERTY30M)
    control = {
        n: {"published": PUBLISHED_SWITCHING[n], "recomputed": sc[n], "reproduces": sc[n] == PUBLISHED_SWITCHING[n]}
        for n in PUBLISHED_SWITCHING
    }
    out["switching_cost"] = {
        "reference": "qwerty30m (C30M qwerty — a permutation of the same charset)",
        "values": sc,
        "positive_control_vs_all_gauge_table": control,
        "control_reproduces": f"{sum(c['reproduces'] for c in control.values())}/{len(control)}",
        "flagship_c3_was_never_published": True,
        "note": (
            "all-gauge-table.json has NO flagship-c3 row, so its switching_cost was never "
            "computed (trap 19). Recomputed here; the other seven are the positive control."
        ),
    }

    # ---- PARTS B + C ----------------------------------------------------------------------
    rng = np.random.default_rng(20260727)
    B = 400
    out["per_finger"] = {}
    out["bootstrap"] = {}
    for corpus, path in CORPORA.items():
        tri_all = load_frequencies(str(Path(path) / "trigrams.txt"))
        surf = TimeSurface(tri_all, target_wpm=90.0, keep_seed_tables=True)

        # PART B: per-finger time shares
        pf = {}
        for n, lay in CAND.items():
            card = surf.card(lay)
            tot = card.total_ms
            shares = {f: 100.0 * v / tot for f, v in card.per_finger_ms.items()}
            # Finger names are LP/LR/LM/LI/RI/RM/RR/RP/THUMB — NOT 'pinky'/'index'/'thumb'.
            # My first pass tested `"pinky" in f`, which matched nothing and silently
            # reported pinky load as 0.000 for every layout, and let THUMB into the
            # max-finger comparison. Keyed off the actual enum names now.
            pinky = sum(v for f, v in shares.items() if f in ("LP", "RP"))
            index = sum(v for f, v in shares.items() if f in ("LI", "RI"))
            nonthumb = {f: v for f, v in shares.items() if f != "THUMB"}
            pf[n] = {
                "finger_time_pct": shares,
                "max_finger_pct": max(nonthumb.values()),
                "max_finger": max(nonthumb, key=nonthumb.get),
                "pinky_pct_both": pinky,
                "index_pct_both": index,
                "gini_over_8_fingers": float(
                    _gini(np.array([v for f, v in sorted(nonthumb.items())]))
                ),
            }
        out["per_finger"][corpus] = pf

        # PART C: paired multinomial bootstrap over the trigram count vector
        names = list(CAND)
        keys = [k for k in surf.tri if len(k) == 3]
        f = np.array([surf.tri[k] for k in keys], dtype=float)
        # per-layout per-trigram weight (ms per occurrence); NaN where uncoverable
        W = np.full((len(names), len(keys)), np.nan)
        T2, Tc = surf._T2, surf._Tc
        nslot = surf._n
        for li, n in enumerate(names):
            slot = {ch: i for i, ch in enumerate(CAND[n])}
            slot[" "] = nslot - 1
            w = np.empty(len(keys))
            ok = np.zeros(len(keys), dtype=bool)
            for ki, k in enumerate(keys):
                try:
                    a, b, c = slot[k[0]], slot[k[1]], slot[k[2]]
                except KeyError:
                    continue
                w[ki] = T2[a, b] + Tc[a, b, c]
                ok[ki] = True
            W[li, ok] = w[ok]
        cover = ~np.isnan(W)
        identical_coverage = bool(cover.all(axis=0).sum() == cover.any(axis=0).sum())
        # restrict to trigrams ALL six cover -> identical denominators, exact paired diffs
        shared = cover.all(axis=0)
        Ws = W[:, shared]
        fs = f[shared]
        # positive control: full-corpus ms/char from this matrix must equal card.ms_per_char
        ctrl = {}
        for li, n in enumerate(names):
            wfull = W[li][cover[li]]
            ffull = f[cover[li]]
            got = float(wfull @ ffull / ffull.sum())
            ref = surf.card(CAND[n]).ms_per_char
            ctrl[n] = {
                "matrix": got,
                "card": ref,
                "delta": got - ref,
                "ok": bool(abs(got - ref) < 1e-9),
            }

        p = fs / fs.sum()
        base = (Ws @ fs) / fs.sum()
        order_base = [names[i] for i in np.argsort(base)]

        # ⚠ SAMPLE SIZE IS THE WHOLE QUESTION HERE, so it is a swept parameter, not a
        # hidden constant. Resampling at the FULL corpus mass (~1e10 trigram tokens) gives
        # a standard error ~1e-5 ms/char and every one of the 15 CIs excludes zero — which
        # is TRUE but nearly vacuous: it says "if you typed the whole iWeb corpus, the
        # model's own ordering would be stable", not "a person would be faster". So the
        # informative version asks the inverse question: at what USER-SCALE corpus size
        # does each pair stop being resolvable? N is in trigram tokens ~ characters typed.
        # 1e5 ~ a few days of typing; 1e6 ~ a month; 1e7 ~ a year; 1e8 ~ a decade.
        sizes = [10**5, 10**6, 10**7, 10**8, int(round(fs.sum()))]
        by_size = {}
        for N in sizes:
            draws = rng.multinomial(N, p, size=B).astype(float)
            MS = (draws @ Ws.T) / N
            rank1 = [names[int(np.argmin(MS[b]))] for b in range(B)]
            pairs = {}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    d = MS[:, i] - MS[:, j]  # >0 => names[i] slower
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    pairs[f"{names[i]}|{names[j]}"] = {
                        "point_delta": float(base[i] - base[j]),
                        "ci95": [float(lo), float(hi)],
                        "excludes_zero": bool(lo > 0 or hi < 0),
                        "frac_draws_i_faster": float((d < 0).mean()),
                    }
            by_size[str(N)] = {
                "n_trigram_tokens": N,
                "rank1_share": {n: rank1.count(n) / B for n in names},
                "n_pairs_ci_excludes_zero": sum(p_["excludes_zero"] for p_ in pairs.values()),
                "n_pairs": len(pairs),
                "pairs": pairs,
            }
        out["bootstrap"][corpus] = {
            "B": B,
            "resample": "multinomial on the shared-coverage trigram count vector, paired",
            "sweep_note": (
                "N = trigram tokens resampled ~ characters typed. At full corpus mass every "
                "CI excludes zero, which is a statement about the MODEL's stability on that "
                "much text, NOT about a human. The user-scale rows are the readable ones."
            ),
            "n_trigram_types_shared": int(shared.sum()),
            "all_six_cover_identically": identical_coverage,
            "positive_control_matrix_vs_card": ctrl,
            "control_ok": all(c["ok"] for c in ctrl.values()),
            "base_ms_per_char_shared_support": {n: float(base[i]) for i, n in enumerate(names)},
            "order_base": order_base,
            "by_sample_size": by_size,
        }

    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(f"wrote {sys.argv[1]}\n")

    print(f"PART A switching cost — control {out['switching_cost']['control_reproduces']}")
    print(f"{'layout':14s} {'unchanged':>10s} {'same_fing':>10s} {'same_hand':>10s} {'zxcv':>5s}")
    for n in [*CAND, *EXTRA, "qwerty30m"]:
        v = sc[n]
        print(
            f"{n:14s} {v['unchanged_keys']:10d} {v['same_finger_keys']:10d} "
            f"{v['same_hand_keys']:10d} {v['zxcv_preserved']:5d}"
        )
    for corpus in CORPORA:
        print(f"\nPART B per-finger ({corpus})")
        print(f"{'layout':14s} {'maxfinger':>10s} {'which':>14s} {'pinky%':>7s} {'gini':>7s}")
        for n in CAND:
            p_ = out["per_finger"][corpus][n]
            print(
                f"{n:14s} {p_['max_finger_pct']:10.3f} {p_['max_finger']:>14s} "
                f"{p_['pinky_pct_both']:7.3f} {p_['gini_over_8_fingers']:7.4f}"
            )
    for corpus in CORPORA:
        b = out["bootstrap"][corpus]
        print(f"\nPART C corpus bootstrap ({corpus})  control_ok={b['control_ok']}  B={b['B']}")
        print(f"  order: {' < '.join(b['order_base'])}")
        for N, s in b["by_sample_size"].items():
            top = max(s["rank1_share"], key=s["rank1_share"].get)
            print(
                f"   N={int(N):>12,}  {s['n_pairs_ci_excludes_zero']:2d}/{s['n_pairs']} pairs "
                f"CI-excludes-0   rank1 modal={top} @ {s['rank1_share'][top]:.3f}"
            )
        # the decisive user-scale row, in full
        key = "1000000"
        print("   --- pairwise at N=1e6 (a month of typing) ---")
        for k, p_ in sorted(
            b["by_sample_size"][key]["pairs"].items(), key=lambda kv: -abs(kv[1]["point_delta"])
        ):
            print(
                f"    {k:32s} {p_['point_delta']:+8.4f}  CI[{p_['ci95'][0]:+.4f},"
                f"{p_['ci95'][1]:+.4f}]  excl0={p_['excludes_zero']}"
            )


def _gini(x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


if __name__ == "__main__":
    main()
