"""pick2 step 2: how big a speed difference can this instrument actually RESOLVE?

Ranking before this question is answered is the campaign's own most-repeated error
("measure your ruler, never borrow it"). So: measure the ruler, then rank.

Three noise channels, all computed from MY OWN runs, none borrowed:

* SEED (paired):  for a board PAIR, the difference is taken on the SAME seed table and the
  spread of that difference over the 3 seeds is the pair's own uncertainty. This is the right
  channel for a same-surface comparison (ledger:10535 makes exactly this point).
* SEED (unpaired): the within-board spread over seeds, which is what the ledger's ~0.135
  "resolution floor" measures. Reported for comparability, but it is the WRONG channel here
  (it contains the seed common-mode that cancels in a paired difference).
* FRAME: does the sign survive changing the corpus (iweb <-> blend-v1) and the target WPM
  (90/110/120)? A ranking that flips under a defensible frame change is not a result.

A pair is called RESOLVED only if it clears the paired bar AND is sign-stable across all 6
frames. n=3 seeds is a small n and a t-interval on 2 df is wide -- that is a property of the
shipped artifact (3 seeds is all there is), and it is why the sign-stability leg is required
rather than optional.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRAMES = [(w, c) for w in (90, 110, 120) for c in ("blend-v1", "iweb")]
#: Student t, 2 df, two-sided 95%
T95_DF2 = 4.302653


def load(wpm: int, corpus: str) -> dict:
    return json.loads((HERE / f"speed_wpm{wpm}_{corpus}.json").read_text())


def main() -> int:
    data = {f: load(*f) for f in FRAMES}
    base = data[(90, "blend-v1")]
    names = list(base["rows"])
    ref = "qwerty"

    # ---- channel 1+2: the ruler, on the production frame
    per_seed = {n: np.array(base["rows"][n]["per_seed_ms_per_char"]) for n in names}
    within = {n: float(np.std(per_seed[n], ddof=1)) for n in names}
    print("RULER, channel A -- WITHIN-board seed spread (the ledger's 'resolution floor' channel)")
    w = np.array(list(within.values()))
    print(f"  sd over 3 seeds, per board: median {np.median(w):.4f}  min {w.min():.4f}  "
          f"max {w.max():.4f}   (ledger quotes ~0.135-0.72 for this channel)")

    pairs = list(itertools.combinations(names, 2))
    paired_sd = {}
    for a, b in pairs:
        d = per_seed[a] - per_seed[b]
        paired_sd[(a, b)] = float(np.std(d, ddof=1))
    ps = np.array(list(paired_sd.values()))
    print("\nRULER, channel B -- PAIRED per-seed difference spread (the correct channel)")
    print(f"  sd of the 3 paired diffs, over {len(pairs)} board pairs: median {np.median(ps):.4f}  "
          f"min {ps.min():.4f}  max {ps.max():.4f}  p90 {np.percentile(ps, 90):.4f}")
    print(f"  => pairing removes the seed common mode: median paired sd is "
          f"{np.median(ps) / np.median(w):.2f}x the within-board sd")

    # ---- the verdict per pair
    out = {}
    for a, b in pairs:
        d = per_seed[a] - per_seed[b]
        m, sd = float(d.mean()), paired_sd[(a, b)]
        half = T95_DF2 * sd / np.sqrt(3)
        signs = set()
        for f in FRAMES:
            ra, rb = data[f]["rows"][a], data[f]["rows"][b]
            df = np.array(ra["per_seed_ms_per_char"]) - np.array(rb["per_seed_ms_per_char"])
            signs.add(int(np.sign(df.mean())))
        ci_excludes_zero = abs(m) > half
        out[f"{a}|{b}"] = {
            "mean_diff": m, "paired_sd": sd, "ci95_half": float(half),
            "ci95_excludes_zero": bool(ci_excludes_zero),
            "sign_stable_6_frames": len(signs) == 1,
            "resolved": bool(ci_excludes_zero and len(signs) == 1),
            "faster": (a if m < 0 else b),
        }

    res = [k for k, v in out.items() if v["resolved"]]
    print(f"\nRESOLVED pairs: {len(res)} of {len(pairs)} "
          f"({100 * len(res) / len(pairs):.0f}%) clear the paired 95% CI AND are sign-stable")

    # ---- how each candidate stands vs qwerty (the decision-relevant contrast)
    print(f"\nvs {ref} (paired over 3 seeds, production frame wpm90/blend-v1):")
    print(f"  {'board':14s} {'ms/char':>9s} {'delta':>8s} {'pairedsd':>9s} {'95%half':>8s} "
          f"{'saved%':>7s}  verdict")
    rows = []
    for n in names:
        if n == ref:
            continue
        k = f"{ref}|{n}" if f"{ref}|{n}" in out else f"{n}|{ref}"
        v = out[k]
        d = per_seed[n] - per_seed[ref]
        m = float(d.mean())
        saved = -100.0 * m / float(per_seed[ref].mean())
        rows.append((m, n, v, saved))
    for m, n, v, saved in sorted(rows):
        verdict = "RESOLVED" if v["resolved"] else (
            "sign-unstable" if v["ci95_excludes_zero"] else "inside noise")
        print(f"  {n:14s} {base['rows'][n]['ms_per_char']:9.4f} {m:+8.4f} "
              f"{v['paired_sd']:9.4f} {v['ci95_half']:8.4f} {saved:+7.2f}%  {verdict}")

    (HERE / "margins.json").write_text(json.dumps(
        {"frames": [list(f) for f in FRAMES], "within_board_sd": within,
         "paired": out,
         "ruler": {"within_median": float(np.median(w)),
                   "paired_median": float(np.median(ps)),
                   "paired_p90": float(np.percentile(ps, 90))}}, indent=1))
    print(f"\nwrote {HERE / 'margins.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
