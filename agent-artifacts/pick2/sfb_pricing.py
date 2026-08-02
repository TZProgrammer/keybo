"""pick2 step 10: does the fitted model PRICE same-finger bigrams the way the raw data does?

This is the test that resolves the central contradiction. The model carries a `same_finger`
feature, so its ms/char ALREADY charges for same-finger bigrams. Two possibilities:

  (a) it prices sfb correctly -> the campaign boards genuinely buy something worth 2.4x the sfb,
      and the community analyzers (which charge sfb by hand-set rule) are the ones overcharging;
  (b) it UNDERPRICES sfb vs the raw timings -> the campaign boards' lead is partly an artifact of
      a cheap same-finger penalty, and the community tools are right.

Test: take the K31 stroke rows, split by whether the position bigram is same-finger, and compare
the RAW observed median interval against the MODEL's prediction for the same position pair, in the
serve WPM bucket where support is thickest. Both are ms for the same physical transitions, so the
comparison is apples-to-apples; the practice term `b` is layout-independent and cancels in the
same-finger-vs-other CONTRAST, which is what is read here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.strokes import load_strokes  # noqa: E402
from keybo.geometry import ROW_STAGGERED_31  # noqa: E402
from keybo.training.validate import build_cells  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

HERE = Path(__file__).resolve().parent
BI = Path.home() / "keybo-e2e" / "bistrokes31_v1.tsv"
SERVE = 80          # the production serve bucket
MIN_N = 30          # thicker than the 10-sample production floor: this is a per-CLASS estimate


def main() -> int:
    import keybo
    print("keybo.__file__ =", keybo.__file__)
    G = ROW_STAGGERED_31
    positions = [*G.slots, G.space_position]
    pos_index = {p: i for i, p in enumerate(positions)}

    t0 = time.time()
    print(f"loading {BI.name} ...")
    rows = load_strokes(str(BI), ngram_len=2, wpm_threshold=0, min_samples=1)
    cells = build_cells(rows, 40, 140, 20, 1)
    print(f"  {len(rows)} rows -> {len(cells)} cells ({time.time() - t0:.0f}s)")

    # the model's own bigram table at the serve bucket's centre wpm
    surf = TimeSurface({}, target_wpm=float(SERVE + 10), keep_seed_tables=True)
    T2 = surf._T2

    # aggregate raw samples per position pair, serve bucket only
    agg: dict[tuple[int, int], list[float]] = {}
    for c in cells:
        if c.bucket != SERVE:
            continue
        try:
            a = pos_index[tuple(int(v) for v in c.positions[0])]
            b = pos_index[tuple(int(v) for v in c.positions[1])]
        except KeyError:
            continue
        agg.setdefault((a, b), []).extend(float(s[1]) for s in c.samples)

    same, other = [], []
    for (a, b), vals in agg.items():
        if len(vals) < MIN_N or a == b:
            continue
        if a >= 30 or b >= 30:      # skip space-touching (different motor act)
            continue
        raw = float(np.median(vals))
        pred = float(T2[a, b])
        fa, fb = G.finger(positions[a][0]), G.finger(positions[b][0])
        (same if fa == fb else other).append((raw, pred, len(vals)))

    for tag, arr in (("SAME-FINGER", same), ("OTHER", other)):
        require_finite([x for r, p, _ in arr for x in (r, p)], f"{tag} raw/pred")
        print(f"  {tag:12s} {len(arr):4d} position pairs, "
              f"{sum(n for _, _, n in arr):8d} raw samples")

    rs, ps = np.array([r for r, _, _ in same]), np.array([p for _, p, _ in same])
    ro, po = np.array([r for r, _, _ in other]), np.array([p for _, p, _ in other])
    print(f"\n{'':16s} {'RAW median ms':>14s} {'MODEL ms':>10s}")
    print(f"{'same-finger':16s} {np.median(rs):14.2f} {np.median(ps):10.2f}")
    print(f"{'other':16s} {np.median(ro):14.2f} {np.median(po):10.2f}")
    raw_pen = float(np.median(rs) - np.median(ro))
    mod_pen = float(np.median(ps) - np.median(po))
    print(f"\nSAME-FINGER PENALTY (same minus other):")
    print(f"  RAW DATA says   {raw_pen:+8.2f} ms")
    print(f"  MODEL says      {mod_pen:+8.2f} ms")
    print(f"  ratio model/raw {mod_pen / raw_pen:8.3f}"
          if raw_pen else "  (raw penalty is zero)")

    # bootstrap CI on the raw penalty (position pairs are the resampling unit)
    rng = np.random.default_rng(0)
    boots = [float(np.median(rng.choice(rs, len(rs))) - np.median(rng.choice(ro, len(ro))))
             for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  raw penalty bootstrap CI95 [{lo:+.2f}, {hi:+.2f}] ms  (4000 resamples over pairs)")
    verdict = ("MODEL UNDERPRICES sfb" if mod_pen < lo else
               "MODEL OVERPRICES sfb" if mod_pen > hi else
               "MODEL's sfb penalty is INSIDE the raw CI -- correctly priced")
    print(f"  => {verdict}")

    out = {"serve_bucket": SERVE, "min_samples_per_pair": MIN_N,
           "n_same_finger_pairs": len(same), "n_other_pairs": len(other),
           "raw_penalty_ms": raw_pen, "model_penalty_ms": mod_pen,
           "raw_penalty_ci95": [float(lo), float(hi)], "verdict": verdict,
           "raw_median_same": float(np.median(rs)), "raw_median_other": float(np.median(ro)),
           "model_median_same": float(np.median(ps)), "model_median_other": float(np.median(po))}
    (HERE / "sfb_pricing.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE / 'sfb_pricing.json'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
