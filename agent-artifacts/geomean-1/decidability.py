"""GEOMEAN-1 step 5 — is the surviving rank-1 DECIDABLE, or is its margin noise?

A1/A4/A5 survive tests (i), (ii'), (iii). Test (iv) already fails on the SPEED instrument.
This asks the same question of the AGGREGATE's own margin, which tests (i)-(iii) never do:
they only ask whether the ORDER is stable, never whether the GAP is bigger than the
aggregate's own uncertainty. A rule whose rank-1 never flips but whose margin is inside its
own noise is not a selection rule — it is a coin that happens to have landed the same way in
every cell I looked at.

Two independent noise models, because a margin can be small relative to different things:

  N1 GAUGE-PERTURBATION BOOTSTRAP. The 15 corpus-sensitive gauges are corpus statistics, so
     they carry corpus sampling error. Resample the CORPUS (multinomial bootstrap over
     bigram/skipgram/trigram counts) B times, rescore the 6-layout field, and report how
     often each layout is rank-1. This is the honest "is the winner reproducible" question.
     Cost is why it is done on the 6-layout field only.

  N2 SEED-PERTURBATION OF THE SPEED-AUGMENTED AGGREGATE. An aggregate that includes a speed
     column inherits the 3-seed estimator spread measured in validation.json. Rescore with
     each seed's own ms/char in place of the seed-mean and report rank-1 per seed.

Plus a PLACEBO (trap 17: an attribution needs a same-size control). If the rank-1 of a
19-gauge aggregate is stable, is that because the frame is informative, or because ANY
19-column aggregate over these 6 layouts is stable? Control: 19 columns of pure noise, and
19 columns that are RANDOM ROTATIONS of the real gauge columns (same marginals, destroyed
cross-layout structure). If the placebo is also "stable", stability is not evidence.

MODELED/gauge only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("/local/home/zegertho/agent/state/geomean/artifacts/geomean-1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregates import ANCHOR, CANDIDATES, FIELD6, gauge_matrix, grouping_from_correlation  # noqa: E402
from aggregates import load_corr, load_pool, score_field  # noqa: E402
from validate import derive_sign  # noqa: E402

B_BOOT = 200
RNG = np.random.default_rng(20260726)


def bootstrap_corpus(rng, counts: dict[str, int]) -> dict[str, int]:
    """One multinomial resample of an n-gram count table at the SAME total mass."""
    keys = list(counts)
    n = np.array([counts[k] for k in keys], dtype=np.int64)
    total = int(n.sum())
    p = n / total
    draw = rng.multinomial(total, p)
    return {k: int(v) for k, v in zip(keys, draw, strict=True) if v > 0}


def score_one_corpus(bi, sk, tri, lays: list[str]) -> np.ndarray:
    """(n_layouts x 15) corpus-sensitive gauge matrix under one (possibly resampled) corpus."""
    from keybo.analysis.kmstats import KmStats
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout
    from keybo.scoring.comfort import ComfortBigramScorer
    from keybo.scoring.oxey import OxeyStyleScorer

    km = KmStats(bi, sk, tri)
    ox = OxeyStyleScorer(bi, sk, tri)
    cf = ComfortBigramScorer(bi, skipgram_freqs=sk)
    mass = sum(bi.values())
    S = (
        "sfr", "sfb", "sfs", "sfb-dist", "sfs-dist", "lsb", "lsb-dist",
        "alt", "roll", "sr-roll", "redir", "scissor", "imbalance", "oxey-style", "comfort",
    )
    rows = []
    for lay in lays:
        L = Layout(lay, ROW_STAGGERED_30)
        g = dict(km.stats(lay))
        sh = ox.pattern_shares(L)
        g["scissor"] = sh["scissor"]
        g["imbalance"] = sh["imbalance"]
        g["oxey-style"] = ox.fitness(L)
        g["comfort"] = cf.fitness(L) / mass
        rows.append([g[k] for k in S])
    return np.asarray(rows, float)


def main() -> int:
    from keybo.data.corpus import load_frequencies

    d = load_pool()
    corr = load_corr()
    named = d["named"]
    gauges = list(d["frame"]["sensitive"]) + list(d["frame"]["invariant"])
    groups = grouping_from_correlation(corr, gauges, "thr_0.8")
    report: dict = {
        "purpose": "GEOMEAN-1: is the surviving rank-1 decidable? bootstrap + placebo",
        "B_bootstrap": B_BOOT,
        "seed": 20260726,
        "corpus_bootstrap": {},
        "placebo": {},
        "seed_perturbation": {},
    }

    cdir = Path("/local/home/zegertho/repos/keybo/data/corpus")
    bi0 = load_frequencies(str(cdir / "bigrams.txt"))
    sk0 = load_frequencies(str(cdir / "1-skip31.txt"))
    tri0 = load_frequencies(str(cdir / "trigrams.txt"))
    sign, gnames, _ = derive_sign(d, "iweb")
    inv_of = {n: [d["invariant"][named[n]][g] for g in d["frame"]["invariant"]] for n in (*FIELD6, ANCHOR)}

    # ------------------------------------------------------------------ N1 corpus bootstrap
    for field_name, field in (("with_qwerty", [*FIELD6, ANCHOR]), ("without_qwerty", list(FIELD6))):
        lays = [named[n] for n in field]
        INV = np.asarray([inv_of[n] for n in field], float)
        wins = {c: dict.fromkeys(field, 0) for c in CANDIDATES}
        margins = {c: [] for c in CANDIDATES}
        for b in range(B_BOOT):
            bi = bootstrap_corpus(RNG, bi0)
            sk = bootstrap_corpus(RNG, sk0)
            tri = bootstrap_corpus(RNG, tri0)
            Xs = score_one_corpus(bi, sk, tri, lays)
            X = np.hstack([Xs, INV])
            sc = score_field(X, gauges, sign, groups)
            for c in CANDIDATES:
                order = np.argsort(-sc[c], kind="mergesort")
                wins[c][field[order[0]]] += 1
                top, second = sc[c][order[0]], sc[c][order[1]]
                margins[c].append(float((top - second) / top) if top else 0.0)
            if b % 50 == 0:
                print(f"  bootstrap {field_name} {b}/{B_BOOT}", flush=True)
        report["corpus_bootstrap"][field_name] = {
            c: {
                "rank1_frequency": {k: v / B_BOOT for k, v in wins[c].items() if v},
                "modal_winner": max(wins[c], key=lambda k: wins[c][k]),
                "modal_share": max(wins[c].values()) / B_BOOT,
                "relative_margin_median": round(float(np.median(margins[c])), 6),
                "relative_margin_p05": round(float(np.percentile(margins[c], 5)), 6),
            }
            for c in CANDIDATES
        }

    # ------------------------------------------------------------------------- placebo
    # Same SIZE (19 columns), same 6-7 rows, but the cross-layout structure is destroyed.
    for field_name, field in (("with_qwerty", [*FIELD6, ANCHOR]), ("without_qwerty", list(FIELD6))):
        lays = [named[n] for n in field]
        Xreal, _ = gauge_matrix(d, "iweb", lays)
        out = {}
        for pname in ("pure_noise", "column_shuffled"):
            wins = {c: dict.fromkeys(field, 0) for c in CANDIDATES}
            for _ in range(B_BOOT):
                if pname == "pure_noise":
                    Xp = RNG.standard_normal(Xreal.shape)
                    sgn = dict.fromkeys(gauges, 1)
                else:
                    # permute each real column independently: identical marginals per gauge,
                    # no cross-gauge / cross-layout structure.
                    Xp = np.column_stack([RNG.permutation(Xreal[:, j]) for j in range(Xreal.shape[1])])
                    sgn = sign
                sc = score_field(Xp, gauges, sgn, groups)
                for c in CANDIDATES:
                    wins[c][field[int(np.argmax(sc[c]))]] += 1
            out[pname] = {
                c: {
                    "modal_share": max(wins[c].values()) / B_BOOT,
                    "rank1_frequency": {k: v / B_BOOT for k, v in wins[c].items() if v},
                }
                for c in CANDIDATES
            }
        report["placebo"][field_name] = out

    # -------------------------------------------------- N2 seed perturbation (speed-augmented)
    val = json.loads((OUT / "validation.json").read_text())["resolution_floor"]["layouts"]
    for field_name, field in (("with_qwerty", [*FIELD6, ANCHOR]), ("without_qwerty", list(FIELD6))):
        lays = [named[n] for n in field]
        Xreal, _ = gauge_matrix(d, "iweb", lays)
        g_aug = [*gauges, "ms_per_char"]
        sign_aug = {**sign, "ms_per_char": -1}
        groups_aug = [*groups, ["ms_per_char"]]
        per_seed = {}
        for s in range(3):
            col = np.array([[val[n]["per_seed_ms_per_char"][s]] for n in field])
            sc = score_field(np.hstack([Xreal, col]), g_aug, sign_aug, groups_aug)
            per_seed[f"seed{s}"] = {
                c: [field[i] for i in np.argsort(-sc[c], kind="mergesort")][0] for c in CANDIDATES
            }
        colm = np.array([[val[n]["ms_per_char_seedmean"]] for n in field])
        scm = score_field(np.hstack([Xreal, colm]), g_aug, sign_aug, groups_aug)
        per_seed["seedmean"] = {
            c: [field[i] for i in np.argsort(-scm[c], kind="mergesort")][0] for c in CANDIDATES
        }
        per_seed["stable_across_seeds"] = {
            c: len({per_seed[f"seed{s}"][c] for s in range(3)}) == 1 for c in CANDIDATES
        }
        report["seed_perturbation"][field_name] = per_seed

    (OUT / "decidability.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {OUT / 'decidability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
