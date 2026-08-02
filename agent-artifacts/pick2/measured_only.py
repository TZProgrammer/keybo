"""pick2 step 8: THE DECISIVE TEST -- does the speed advantage survive on MEASURED territory?

Step 3 found every non-qwerty board rides ~60-70% model extrapolation on the trigram term. That
raises the question a support percentage cannot answer: is the predicted advantage LOCATED in the
measured region, or is it manufactured in the region the model invents?

The test: score the boards on the corpus trigrams whose POSITION-triple the K31 study actually
observed, and compare that ranking to the all-corpus ranking. Two frames, both reported:

  (a) OWN-observed  -- each board on its own measured subset. Informative, NOT comparable
                       (different boards -> different subsets, the same defect `common_ngrams`
                       fixes for charsets).
  (b) CO-OBSERVED   -- the corpus trigrams measured for EVERY compared board. Strictly
                       comparable, and it is the frame the verdict is read from.

If the advantage holds in (b), the fitted ranking is anchored in measurement. If it collapses or
inverts, the ~+3.7% headline is a statement about position n-grams nobody in the study typed.

Caches the observed position sets to disk (the 8-minute TSV pass is the cost).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE  # noqa: E402

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "observed_sets.pkl"
BI = Path.home() / "keybo-e2e" / "bistrokes31_v1.tsv"
TRI = Path.home() / "keybo-e2e" / "tristrokes31_cond_v1.tsv"
T95_DF2 = 4.302653


def observed_sets():
    if CACHE.is_file():
        print(f"  loading cached observed sets from {CACHE}")
        return pickle.loads(CACHE.read_bytes())
    from keybo.analysis.select import RawSupport
    t0 = time.time()
    print("  building RawSupport from the K31 stroke tables (~8 min)...")
    rs = RawSupport.from_tsvs(BI, TRI)
    d = {"tri_serve": rs.tri_serve, "tri_any": rs.tri_any,
         "bi_serve": rs.bi_serve, "bi_any": rs.bi_any, "positions": rs.positions}
    CACHE.write_bytes(pickle.dumps(d))
    print(f"  built + cached in {time.time() - t0:.0f}s")
    return d


def main() -> int:
    import keybo
    print("keybo.__file__ =", keybo.__file__)
    obs = observed_sets()
    positions = obs["positions"]
    tri_serve = obs["tri_serve"]
    print(f"  observed position-trigrams (serve): {len(tri_serve)} of {31**3}")

    corpus = "blend-v1"
    cdir = production_corpus_dir(corpus)
    tri_all = load_frequencies(str(cdir / "trigrams.txt"))
    surf = TimeSurface(tri_all, target_wpm=90.0, keep_seed_tables=True)

    # charset-common corpus first (so the charset artifact is already removed)
    common_chars = set.intersection(*(set(l) for l in ALL.values())) | {" "}
    tri = {k: v for k, v in surf.tri.items() if len(k) == 3 and set(k) <= common_chars}
    ng = list(tri)
    f_all = np.array([tri[k] for k in ng], dtype=np.float64)
    print(f"  charset-common trigrams: {len(ng)}")

    # per board: slot indices and the observed mask
    idx, mask = {}, {}
    for name, lay in ALL.items():
        slot = surf._slot_of(lay)
        a = np.array([slot[g[0]] for g in ng], np.int32)
        b = np.array([slot[g[1]] for g in ng], np.int32)
        c = np.array([slot[g[2]] for g in ng], np.int32)
        idx[name] = (a, b, c)
        mask[name] = np.array(
            [(positions[x], positions[y], positions[z]) in tri_serve
             for x, y, z in zip(a, b, c, strict=True)], dtype=bool)

    co = np.logical_and.reduce([mask[n] for n in ALL])
    print(f"  CO-OBSERVED trigrams (measured for ALL {len(ALL)} boards): {int(co.sum())} "
          f"= {100 * f_all[co].sum() / f_all.sum():.2f}% of the charset-common mass\n")

    def score(name, sel):
        a, b, c = idx[name]
        a, b, c, f = a[sel], b[sel], c[sel], f_all[sel]
        m = f.sum()
        per_seed = [float(((T2[a, b] + Tc[a, b, c]) * f).sum()) / m
                    for T2, Tc in zip(surf._T2s, surf._Tcs, strict=True)]
        mean = float(((surf._T2[a, b] + surf._Tc[a, b, c]) * f).sum()) / m
        return mean, per_seed, float(m)

    rows = {}
    allsel = np.ones(len(ng), bool)
    for name in ALL:
        m_all, ps_all, _ = score(name, allsel)
        m_own, ps_own, mass_own = score(name, mask[name])
        m_co, ps_co, mass_co = score(name, co)
        rows[name] = {
            "provenance": PROVENANCE[name],
            "all": {"ms_per_char": m_all, "per_seed": ps_all},
            "own_observed": {"ms_per_char": m_own, "per_seed": ps_own,
                             "mass_pct": 100 * mass_own / f_all.sum()},
            "co_observed": {"ms_per_char": m_co, "per_seed": ps_co},
        }

    order = sorted(ALL, key=lambda n: rows[n]["all"]["ms_per_char"])
    rk_all = {n: i + 1 for i, n in enumerate(order)}
    rk_co = {n: i + 1 for i, n in enumerate(sorted(ALL, key=lambda n: rows[n]["co_observed"]["ms_per_char"]))}
    print(f"{'board':14s} {'prov':9s} {'ALL-corpus':>11s} {'CO-OBSERVED':>12s} {'rank ALL':>9s} "
          f"{'rank CO':>8s} {'move':>6s}")
    for n in order:
        r = rows[n]
        print(f"{n:14s} {r['provenance']:9s} {r['all']['ms_per_char']:11.3f} "
              f"{r['co_observed']['ms_per_char']:12.3f} {rk_all[n]:9d} {rk_co[n]:8d} "
              f"{rk_all[n] - rk_co[n]:+6d}")

    from scipy.stats import spearmanr
    x = [rows[n]["all"]["ms_per_char"] for n in ALL]
    y = [rows[n]["co_observed"]["ms_per_char"] for n in ALL]
    r, p = spearmanr(x, y)
    print(f"\nspearman(ALL-corpus, CO-OBSERVED) over {len(ALL)} boards: rho {r:+.4f} p={p:.3g}")

    # the paired verdict on the CO-OBSERVED frame
    fastest_co = min(ALL, key=lambda n: rows[n]["co_observed"]["ms_per_char"])
    print(f"\nCO-OBSERVED frame: fastest = {fastest_co}")
    print(f"  {'board':14s} {'delta':>8s} {'sd':>7s} {'95%half':>8s}  verdict")
    for n in sorted(ALL, key=lambda n: rows[n]["co_observed"]["ms_per_char"]):
        d = (np.array(rows[n]["co_observed"]["per_seed"])
             - np.array(rows[fastest_co]["co_observed"]["per_seed"]))
        m, sd = float(d.mean()), float(np.std(d, ddof=1))
        half = T95_DF2 * sd / np.sqrt(3)
        print(f"  {n:14s} {m:+8.3f} {sd:7.4f} {half:8.4f}  "
              f"{'RESOLVABLY SLOWER' if m > half else 'speed-equivalent'}")

    (HERE / "measured_only.json").write_text(json.dumps(
        {"corpus": corpus, "n_charset_common": len(ng), "n_co_observed": int(co.sum()),
         "co_observed_mass_pct": float(100 * f_all[co].sum() / f_all.sum()),
         "spearman_all_vs_co": [float(r), float(p)], "rows": rows}, indent=1))
    print(f"\nwrote {HERE / 'measured_only.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
