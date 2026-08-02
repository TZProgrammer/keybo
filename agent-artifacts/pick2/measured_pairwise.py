"""pick2 step 8b: the measured-vs-extrapolated test, done PAIRWISE -- and the guard that
should have been there the first time.

My first attempt (`measured_only.py`) intersected the observed position-trigram sets of all 27
boards. That intersection is EMPTY (0 trigrams), so every operand was `nan`, and the script
printed `rank CO` numbers from sorting nans and the verdict "speed-equivalent" for all 27 boards
-- with `fastest = qwerty` because `min()` over nans returns the first element. A comparison whose
operands were never computed returned the answer that means "no difference". That is verbatim the
failure mode `keybo.verdicts` was written for, and I did not call it. Fixed here: `require_finite`
guards every operand before it becomes a claim, and the empty-intersection case now RAISES.

The intersection is empty for a structural reason worth recording: measurement coverage is
LAYOUT-SPECIFIC (each board maps the corpus onto its own position triples), so "measured for every
board at once" is a much stronger demand than "measured for this pair". A 27-way intersection over
sets covering ~37% each was never going to be non-empty. The right frame for a PAIRWISE verdict is
pairwise co-observation, which is what a comparison actually needs.
"""

from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE  # noqa: E402

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402
from keybo.verdicts import EmptyComparison, require_finite  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "observed_sets.pkl"
T95_DF2 = 4.302653

#: the cohort the verdict is read from: every speed-gate board, plus the boards that LEAD any
#: independent axis, plus the qwerty control. Chosen before the numbers are read.
COHORT = ["arm-B", "BALL-1", "MID", "armH-hdln", "p11-w05", "p10-w05",
          "keybo-lsb", "keybo-c30m", "p13stab-win", "flagship-c3",
          "semimak", "recurva", "graphite", "canary", "colemak", "dvorak", "qwerty"]


def main() -> int:
    import keybo
    print("keybo.__file__ =", keybo.__file__)
    obs = pickle.loads(CACHE.read_bytes())
    positions, tri_serve = obs["positions"], obs["tri_serve"]

    cdir = production_corpus_dir("blend-v1")
    surf = TimeSurface(load_frequencies(str(cdir / "trigrams.txt")), target_wpm=90.0,
                       keep_seed_tables=True)
    common_chars = set.intersection(*(set(l) for l in ALL.values())) | {" "}
    ng = [k for k, v in surf.tri.items() if len(k) == 3 and set(k) <= common_chars]
    f_all = np.array([surf.tri[k] for k in ng], dtype=np.float64)

    idx, mask = {}, {}
    for n in COHORT:
        slot = surf._slot_of(ALL[n])
        a = np.array([slot[g[0]] for g in ng], np.int32)
        b = np.array([slot[g[1]] for g in ng], np.int32)
        c = np.array([slot[g[2]] for g in ng], np.int32)
        idx[n] = (a, b, c)
        mask[n] = np.array([(positions[x], positions[y], positions[z]) in tri_serve
                            for x, y, z in zip(a, b, c, strict=True)], bool)

    print(f"\ncohort of {len(COHORT)}; charset-common trigrams {len(ng)}")
    print(f"27-way intersection was EMPTY -- confirming: "
          f"{int(np.logical_and.reduce([mask[n] for n in COHORT]).sum())} co-observed across this cohort too\n")

    def score(n, sel):
        a, b, c = idx[n]
        a, b, c, f = a[sel], b[sel], c[sel], f_all[sel]
        m = float(f.sum())
        require_finite([m], f"{n} co-observed mass")
        if m <= 0.0:
            raise EmptyComparison(f"{n}: co-observed subset carries zero mass")
        ps = [float(((T2[a, b] + Tc[a, b, c]) * f).sum()) / m
              for T2, Tc in zip(surf._T2s, surf._Tcs, strict=True)]
        return require_finite(ps, f"{n} co-observed per-seed ms/char"), m

    # ---- pairwise: does the sign of the speed difference SURVIVE restricting to co-observed?
    allsel = np.ones(len(ng), bool)
    full = {n: score(n, allsel)[0] for n in COHORT}
    rows, flips, kept, skipped = {}, [], [], []
    for a, b in itertools.combinations(COHORT, 2):
        co = mask[a] & mask[b]
        m_co = float(f_all[co].sum())
        pct = 100.0 * m_co / float(f_all.sum())
        if co.sum() == 0 or m_co <= 0:
            skipped.append(f"{a}|{b}")
            continue
        pa, ma = score(a, co)
        pb, _ = score(b, co)
        d_co = float(np.mean(np.array(pa) - np.array(pb)))
        sd_co = float(np.std(np.array(pa) - np.array(pb), ddof=1))
        d_full = float(np.mean(np.array(full[a]) - np.array(full[b])))
        rec = {"co_mass_pct": pct, "n_co": int(co.sum()),
               "delta_full": d_full, "delta_co": d_co, "sd_co": sd_co,
               "ci95_half_co": T95_DF2 * sd_co / np.sqrt(3),
               "sign_flip": bool(np.sign(d_full) != np.sign(d_co))}
        rows[f"{a}|{b}"] = rec
        (flips if rec["sign_flip"] else kept).append(f"{a}|{b}")

    print(f"PAIRWISE CO-OBSERVED TEST over {len(rows)} pairs ({len(skipped)} skipped: empty overlap)")
    print(f"  sign PRESERVED: {len(kept)}   sign FLIPPED: {len(flips)}  "
          f"({100 * len(flips) / max(len(rows), 1):.0f}%)")
    pcts = [r["co_mass_pct"] for r in rows.values()]
    print(f"  co-observed mass per pair: median {np.median(pcts):.2f}%  "
          f"min {min(pcts):.2f}%  max {max(pcts):.2f}%")
    if flips:
        print("\n  FLIPPED pairs (the fitted ranking is NOT anchored in measurement for these):")
        for k in sorted(flips, key=lambda k: -abs(rows[k]["delta_full"]))[:22]:
            r = rows[k]
            print(f"    {k:28s} full {r['delta_full']:+7.3f} -> co-obs {r['delta_co']:+7.3f} "
                  f"(co mass {r['co_mass_pct']:5.2f}%)")

    # the decision-relevant slice: every gate board vs every independent-axis leader
    print("\n  DECISION SLICE -- speed-gate boards vs the boards that lead the independent axes:")
    gates = ["arm-B", "BALL-1", "MID", "armH-hdln", "p11-w05", "p10-w05"]
    others = ["keybo-lsb", "keybo-c30m", "semimak", "recurva", "graphite", "canary", "colemak", "dvorak"]
    print(f"    {'pair':30s} {'full':>8s} {'co-obs':>8s} {'sd':>7s} {'95%half':>8s}  verdict")
    for a in gates:
        for b in others:
            k = f"{a}|{b}" if f"{a}|{b}" in rows else f"{b}|{a}"
            if k not in rows:
                continue
            r = rows[k]
            sgn = 1.0 if k.startswith(a) else -1.0
            df, dc = sgn * r["delta_full"], sgn * r["delta_co"]
            v = ("FLIPPED: co-obs says the other board is faster" if np.sign(df) != np.sign(dc)
                 else ("holds, resolvable" if abs(dc) > r["ci95_half_co"] else "holds, inside noise"))
            print(f"    {a + ' vs ' + b:30s} {df:+8.3f} {dc:+8.3f} {r['sd_co']:7.3f} "
                  f"{r['ci95_half_co']:8.3f}  {v}")

    (HERE / "measured_pairwise.json").write_text(json.dumps(
        {"cohort": COHORT, "n_pairs": len(rows), "sign_flips": flips, "sign_kept": kept,
         "skipped_empty": skipped, "pairs": rows}, indent=1))
    print(f"\nwrote {HERE / 'measured_pairwise.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
