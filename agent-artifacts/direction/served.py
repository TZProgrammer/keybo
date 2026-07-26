"""THE SERVED FRAME: does the direction channel change what we optimize for?

Answers the two questions the brief makes decisive, and gates the verdict on the NGRAM-FE
trap (a model that fits BETTER on the full frame while the SERVED geometry COLLAPSES).

    1. served rho/ceiling                 -> is the served surface still trustworthy?
    2. optimizer-tensor Spearman(new,old) -> does the optimizer see the same landscape?
    3. incumbent reordering                -> does it change the ANSWER?

FRAME (stated because getting it wrong has cost this campaign a retraction):
  * time = g(geometry, wpm) + b(ngram); ONLY g is served. b is keyed by n-gram identity, is
    layout-independent, and cancels exactly in any layout comparison (train.py: "scoring
    deliberately ignores it"). Every number here is on the g frame.
  * WPM 90, ROW_STAGGERED_30, 3 seeds, seed-MEAN tensors.
  * Corpus for the weighted sums: data/corpus/{bigrams,trigrams}.txt = iWeb, SINGLE-SOURCE
    (ledger GAP-CORPUS-1). Corpus md5 is recorded in the output.
  * Space IS included in the optimizer tensor (slot 30) because the training pipeline emits
    space bigrams and the scorer includes them — train/serve parity.

RESOLUTION FLOOR: per-seed layout spreads are 0.70-0.99 ms/char (theory-1 E1), so a
reordering driven by a gap below ~1 ms/char is NOT a real change. Every reported flip is
printed with its gap AND the per-seed spread, and labelled resolved / BELOW RESOLUTION.

Outputs artifacts/direction_served.json. Publishes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

WT = "/local/home/zegertho/agent/state/direction/wt-direction"
sys.path.insert(0, f"{WT}/src")

from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.features import bigram_features_from_positions  # noqa: E402
from keybo.geometry import ROW_STAGGERED_30  # noqa: E402
from keybo.training.train import train_bigram_model  # noqa: E402

sys.path.insert(0, "/local/home/zegertho/agent/state/direction/scratch")
from refit import COMM_LABELS, REG_LOLO, load_surface  # noqa: E402

REPO = Path(WT)
OUT = Path("/local/home/zegertho/agent/state/direction/artifacts")
CORPUS = Path("/local/home/zegertho/repos/keybo/data/corpus")

#: C30M slot order — the 30 letter slots the campaign's surfaces are built in.
C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"

#: The 9 incumbents the brief names. keybo-lsb+lm from tests/analysis/test_community_wfd_frames.py;
#: the rest from cli/analyze.py _EXTRA_NAMED and layouts.NAMED_LAYOUTS.
INCUMBENTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "qwerty": "qwerty" + "uiop" + "asdfghjkl'" + "zxcvbnm,.-",
}

SEEDS = [0, 1, 2]
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def load_bigram_freqs() -> dict[str, int]:
    """iWeb bigram counts. Only bigrams whose BOTH chars are on the 30-key board count."""
    freqs: dict[str, int] = {}
    for line in (CORPUS / "bigrams.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        bg, n = parts[0], int(parts[1])
        if len(bg) == 2:
            freqs[bg] = freqs.get(bg, 0) + n
    return freqs


def serve_grid(model, geometry, direction: bool = False, placebo: bool = False) -> np.ndarray:
    """The 31x31 optimizer tensor T2[slot_a, slot_b] in MILLISECONDS at wpm 90.

    Slot 30 is space. This IS the object the optimizer queries: it is layout-independent
    (indexed by SLOT, not character), so it is the honest place to compare two fitted
    surfaces — unlike raw training data, which is ~98.7% qwerty (OQ-1: correlation is not
    price).
    """
    positions = [*geometry.slots, geometry.space_position]
    X = np.vstack(
        [
            bigram_features_from_positions(
                geometry, (a, b), wpm=90.0, direction=direction, placebo=placebo
            )
            for a in positions
            for b in positions
        ]
    )
    return model.to_ms(model.predict(X), X).reshape(len(positions), len(positions))


def perm_of(layout: str, geometry) -> np.ndarray:
    """char-slot permutation: perm[i] = slot index of C30M[i]'s character in this layout.

    Built so a fitness sum can index the tensor by CORPUS character. Space maps to slot 30.
    """
    pos_of = {ch: i for i, ch in enumerate(layout)}
    return np.array([pos_of[ch] for ch in C30M], dtype=np.int64)


def fitness(T2: np.ndarray, layout: str, freqs: dict[str, int], geometry) -> tuple[float, float]:
    """(total ms, ms per char) for a layout under a served bigram tensor.

    Only bigrams typable on the board are summed (the scorer's own rule). Space is slot 30.
    """
    slot_of = {ch: i for i, ch in enumerate(layout)}
    slot_of[" "] = 30
    total = 0.0
    weight = 0.0
    for bg, n in freqs.items():
        a, b = bg[0], bg[1]
        if a not in slot_of or b not in slot_of:
            continue
        total += T2[slot_of[a], slot_of[b]] * n
        weight += n
    return total, (total / weight if weight else float("nan"))


def main() -> None:
    surface = os.environ.get("KEYBO_SURFACE", "AALTO")
    out_path = OUT / (os.environ.get("KEYBO_OUT") or f"direction_served_{surface}.json")
    n_jobs = int(os.environ.get("KEYBO_NJOBS", "16"))
    geom = ROW_STAGGERED_30

    freqs = load_bigram_freqs()
    corpus_md5 = hashlib.md5((CORPUS / "bigrams.txt").read_bytes()).hexdigest()
    log(f"corpus bigrams.txt md5={corpus_md5} ({len(freqs)} bigrams)")

    rows = load_surface(surface)
    log(f"{surface}: {len(rows)} rows, {sum(len(r.samples) for r in rows)} samples")

    out: dict = {
        "meta": {
            "surface": surface,
            "frame": "g(geometry, wpm) ONLY — the additive per-ngram term b is excluded "
            "(layout-independent, ranking-irrelevant; train.py says scoring ignores it)",
            "wpm": 90.0,
            "geometry": "ROW_STAGGERED_30 (+ space at slot 30)",
            "seeds": SEEDS,
            "corpus": f"data/corpus/bigrams.txt (iWeb, SINGLE-SOURCE, ledger GAP-CORPUS-1)",
            "corpus_md5": corpus_md5,
            "recipe": "REG_LOLO production bigram params, LOGRAT target space",
            "resolution_floor_ms_per_char": "~1.0 (theory-1 E1: per-seed layout spreads 0.70-0.99)",
        },
        "arms": {},
    }

    tensors: dict[str, list[np.ndarray]] = {}
    for arm, kw in [("v1", {}), ("placebo", {"placebo": True}), ("v2", {"direction": True})]:
        per_seed = []
        for seed in SEEDS:
            m = train_bigram_model(
                rows,
                target_wpm=90.0,
                geometry=geom,
                random_state=seed,
                n_jobs=n_jobs,
                **kw,
                **REG_LOLO,
            )
            per_seed.append(
                serve_grid(
                    m,
                    geom,
                    direction=bool(kw.get("direction")),
                    placebo=bool(kw.get("placebo")),
                )
            )
            log(f"  {arm} seed {seed}: served tensor built")
        tensors[arm] = per_seed
        out["arms"][arm] = {"n_seeds": len(per_seed)}

    # --- 1. optimizer-tensor Spearman, new vs old --------------------------------------
    # THE NGRAM-FE GATE. A fit gain with a collapsed served surface is a REJECT.
    off_diag = ~np.eye(31, dtype=bool)

    def rho_between(a: np.ndarray, b: np.ndarray) -> float:
        return float(spearmanr(a[off_diag], b[off_diag]).statistic)

    mean = {k: np.mean(v, axis=0) for k, v in tensors.items()}
    seed_spread = {
        k: float(np.mean(np.std(np.stack(v), axis=0))) for k, v in tensors.items()
    }
    out["optimizer_tensor"] = {
        "note": (
            "Spearman over the 930 OFF-DIAGONAL cells of the 31x31 seed-mean served tensor. "
            "This is the layout-neutral serve grid, not training data (OQ-1: the corpus is "
            "~98.7% qwerty, so correlation there is not price)."
        ),
        "rho_v2_vs_v1": rho_between(mean["v2"], mean["v1"]),
        "rho_placebo_vs_v1": rho_between(mean["placebo"], mean["v1"]),
        "rho_v2_vs_placebo": rho_between(mean["v2"], mean["placebo"]),
        "mean_abs_diff_ms_v2_v1": float(np.mean(np.abs(mean["v2"] - mean["v1"])[off_diag])),
        "max_abs_diff_ms_v2_v1": float(np.max(np.abs(mean["v2"] - mean["v1"])[off_diag])),
        "mean_abs_diff_ms_placebo_v1": float(
            np.mean(np.abs(mean["placebo"] - mean["v1"])[off_diag])
        ),
        "mean_seed_spread_ms": seed_spread,
        # Per-seed rho too: a seed-mean rho can hide per-seed disagreement.
        "rho_v2_vs_v1_per_seed": [
            rho_between(tensors["v2"][i], tensors["v1"][i]) for i in range(len(SEEDS))
        ],
    }
    log(f"optimizer-tensor rho v2-vs-v1 = {out['optimizer_tensor']['rho_v2_vs_v1']:.4f} "
        f"(placebo-vs-v1 {out['optimizer_tensor']['rho_placebo_vs_v1']:.4f})")

    # --- 2. THE DIRECTION ASYMMETRY the tensor can now express -------------------------
    # T2[a,b] vs T2[b,a] on the served surface. Under v1 this is a pure landing-key
    # difference; under v2 it can carry a genuine direction effect. Reported in ms.
    def asym_stats(T: np.ndarray) -> dict:
        d = T - T.T
        v = d[off_diag]
        return {
            "mean_abs_asymmetry_ms": float(np.mean(np.abs(v))),
            "p90_abs_asymmetry_ms": float(np.percentile(np.abs(v), 90)),
            "max_abs_asymmetry_ms": float(np.max(np.abs(v))),
        }

    out["direction_asymmetry"] = {
        "note": (
            "T2[a,b] - T2[b,a] on the served tensor. Under v1 any asymmetry is the "
            "LANDING-KEY price wearing a direction-shaped name (THEORY-1 D2); under v2 the "
            "surface can additionally express travel direction. The DIFFERENCE between the "
            "two arms' asymmetry is the direction channel's served magnitude."
        ),
        **{arm: asym_stats(mean[arm]) for arm in ("v1", "placebo", "v2")},
    }

    # --- 3. incumbent reordering -------------------------------------------------------
    board: dict[str, dict] = {}
    for name, layout in INCUMBENTS.items():
        if sorted(layout) != sorted(C30M):
            log(f"  WARNING: {name} charset != C30M; skipping")
            continue
        entry = {}
        for arm in ("v1", "placebo", "v2"):
            per_seed = [fitness(T, layout, freqs, geom)[1] for T in tensors[arm]]
            entry[arm] = {
                "ms_per_char": float(np.mean(per_seed)),
                "per_seed": [float(x) for x in per_seed],
                "seed_spread": float(np.std(per_seed)),
            }
        board[name] = entry
    out["board"] = board

    def ranking(arm: str) -> list[str]:
        return sorted(board, key=lambda n: board[n][arm]["ms_per_char"])

    rank_v1, rank_pl, rank_v2 = ranking("v1"), ranking("placebo"), ranking("v2")
    out["rankings"] = {"v1": rank_v1, "placebo": rank_pl, "v2": rank_v2}
    out["ranking_identical_v1_v2"] = rank_v1 == rank_v2
    out["ranking_identical_placebo_v2"] = rank_pl == rank_v2
    scores_v1 = [board[n]["v1"]["ms_per_char"] for n in rank_v1]
    scores_v2 = [board[n]["v2"]["ms_per_char"] for n in rank_v1]
    out["board_spearman_v2_vs_v1"] = float(spearmanr(scores_v1, scores_v2).statistic)

    # Adjacent-pair gaps vs the resolution floor: a flip inside the floor is NOT a change.
    floor = 1.0
    flips = []
    for i, a in enumerate(rank_v1):
        for b in rank_v1[i + 1 :]:
            if rank_v2.index(a) > rank_v2.index(b):
                gap_v1 = board[b]["v1"]["ms_per_char"] - board[a]["v1"]["ms_per_char"]
                gap_v2 = board[a]["v2"]["ms_per_char"] - board[b]["v2"]["ms_per_char"]
                spread = max(
                    board[a]["v1"]["seed_spread"],
                    board[b]["v1"]["seed_spread"],
                    board[a]["v2"]["seed_spread"],
                    board[b]["v2"]["seed_spread"],
                )
                flips.append(
                    {
                        "pair": [a, b],
                        "gap_v1_ms_per_char": gap_v1,
                        "gap_v2_ms_per_char": gap_v2,
                        "max_seed_spread": spread,
                        "resolved": bool(min(gap_v1, gap_v2) > floor),
                        "verdict": (
                            "RESOLVED FLIP"
                            if min(gap_v1, gap_v2) > floor
                            else "BELOW RESOLUTION (do not report as a reordering)"
                        ),
                    }
                )
    out["flips_v1_to_v2"] = flips

    out_path.write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {out_path}")

    # --- report -----------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"SERVED FRAME — {surface} — g(geometry,wpm) only, wpm 90, iWeb, 3-seed mean")
    print("=" * 92)
    ot = out["optimizer_tensor"]
    print(f"optimizer-tensor Spearman (930 off-diagonal cells of the 31x31 serve grid):")
    print(f"   v2      vs v1: {ot['rho_v2_vs_v1']:.6f}   per-seed {['%.4f' % r for r in ot['rho_v2_vs_v1_per_seed']]}")
    print(f"   placebo vs v1: {ot['rho_placebo_vs_v1']:.6f}   <- width-only control")
    print(f"   v2      vs placebo: {ot['rho_v2_vs_placebo']:.6f}  <- ATTRIBUTABLE to direction")
    print(f"   mean|diff| v2-v1 {ot['mean_abs_diff_ms_v2_v1']:.3f} ms  "
          f"(placebo-v1 {ot['mean_abs_diff_ms_placebo_v1']:.3f} ms), max {ot['max_abs_diff_ms_v2_v1']:.3f} ms")
    print(f"   mean per-seed spread: " +
          ", ".join(f"{k} {v:.3f} ms" for k, v in ot["mean_seed_spread_ms"].items()))

    print("\ndirection asymmetry on the served tensor  T2[a,b] - T2[b,a]:")
    for arm in ("v1", "placebo", "v2"):
        a = out["direction_asymmetry"][arm]
        print(f"   {arm:8s} mean|asym| {a['mean_abs_asymmetry_ms']:7.3f} ms   "
              f"p90 {a['p90_abs_asymmetry_ms']:7.3f}   max {a['max_abs_asymmetry_ms']:8.3f}")

    print("\nincumbent board (ms/char, lower = faster; +/- is the 3-seed spread):")
    print(f"{'layout':16s} {'v1':>18s} {'placebo':>18s} {'v2':>18s} {'v2-v1':>9s}")
    for name in rank_v1:
        e = board[name]
        print(
            f"{name:16s} "
            f"{e['v1']['ms_per_char']:11.4f}+-{e['v1']['seed_spread']:.3f}  "
            f"{e['placebo']['ms_per_char']:11.4f}+-{e['placebo']['seed_spread']:.3f}  "
            f"{e['v2']['ms_per_char']:11.4f}+-{e['v2']['seed_spread']:.3f}  "
            f"{e['v2']['ms_per_char'] - e['v1']['ms_per_char']:+9.4f}"
        )
    print(f"\nranking v1 == v2 ? {out['ranking_identical_v1_v2']}   "
          f"board Spearman {out['board_spearman_v2_vs_v1']:.6f}")
    print(f"v1 order: {' < '.join(rank_v1)}")
    print(f"v2 order: {' < '.join(rank_v2)}")
    if flips:
        print("\nflips, each against the ~1 ms/char resolution floor:")
        for f in flips:
            print(f"   {f['pair'][0]} <-> {f['pair'][1]}: gap_v1 {f['gap_v1_ms_per_char']:+.4f} "
                  f"gap_v2 {f['gap_v2_ms_per_char']:+.4f} spread {f['max_seed_spread']:.4f} "
                  f"=> {f['verdict']}")
    else:
        print("\nno pairwise flips between v1 and v2 on this surface.")
    print("ALL-DONE")


if __name__ == "__main__":
    main()
