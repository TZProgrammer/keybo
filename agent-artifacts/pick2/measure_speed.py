"""pick2 step 1: the primary speed axis, PER SEED, with the resolvable margin measured first.

Design decisions, stated before any number is read:

* The gauge is the campaign's own primary one: predicted ms/char on the K31 measured-keystroke
  surface (``T2[a,b] + Tcond[a,b,c]`` summed over the trigram corpus / covered mass). It is the
  only axis in this repo fitted to REAL keystroke timings rather than asserted by a hand rule.
* ``ms_per_char`` (per CHARACTER TYPED), never raw totals: the cohort spans 3 charsets and a raw
  total charges a wider charset for covering more corpus mass (``TimeCard`` docstring).
* PER SEED, and the pairwise comparison is PAIRED: each board pair is differenced on the SAME
  seed table, so the seed's common-mode error cancels. The ledger's ~0.135 ms/char "resolution
  floor" is an UNPAIRED estimator floor (:10405, and :10535 states the paired quantity is the
  right one for a same-surface comparison). I compute BOTH and report the stricter verdict.
* Verification, not trust: the vectorized sum is parity-gated against the shipped
  ``TimeSurface.card`` / ``seed_totals`` before any result is used.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE, validate  # noqa: E402

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import corpus_identity, load_frequencies, production_corpus_dir  # noqa: E402
from keybo.verdicts import require_finite  # noqa: E402

TARGET_WPM = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
CORPUS = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path(__file__).resolve().parent / (
    f"speed_wpm{int(TARGET_WPM)}_{(CORPUS or 'default').replace('/', '_')}.json"
)


def env_check() -> dict:
    """The silent-hazard check: WHICH checkout is `keybo` resolving to, on which branch."""
    import subprocess

    import keybo

    root = Path(keybo.__file__).resolve().parents[2]
    g = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *a], capture_output=True, text=True
    ).stdout.strip()
    info = {
        "keybo_file": keybo.__file__,
        "checkout": str(root),
        "branch": g("branch", "--show-current"),
        "head": g("rev-parse", "HEAD"),
        "dirty": g("status", "--short"),
        "python": sys.executable,
    }
    print(f"keybo.__file__ = {info['keybo_file']}")
    print(f"checkout       = {info['checkout']}  branch={info['branch']}  head={info['head'][:8]}")
    print(f"dirty          = {info['dirty'] or '(clean)'}")
    return info


class Vec:
    """Vectorized per-seed corpus sum, parity-gated against the shipped loop."""

    def __init__(self, surf: TimeSurface):
        self.s = surf
        self.tri = list(surf.tri.items())
        self.f = np.array([v for _, v in self.tri], dtype=np.float64)
        self.ng = [k for k, _ in self.tri]
        self.n = surf._n

    def idx(self, lay30: str):
        slot = self.s._slot_of(lay30)  # REFUSES a malformed layout (30 distinct chars)
        a = np.full(len(self.ng), -1, np.int32)
        b = a.copy()
        c = a.copy()
        for i, g in enumerate(self.ng):
            try:
                a[i], b[i], c[i] = slot[g[0]], slot[g[1]], slot[g[2]]
            except KeyError:
                continue
        ok = a >= 0
        return a[ok], b[ok], c[ok], self.f[ok]

    def per_seed(self, lay30: str):
        a, b, c, f = self.idx(lay30)
        covered = float(f.sum())
        totals = [
            float(((T2[a, b] + Tc[a, b, c]) * f).sum())
            for T2, Tc in zip(self.s._T2s, self.s._Tcs, strict=True)
        ]
        mean_total = float(((self.s._T2[a, b] + self.s._Tc[a, b, c]) * f).sum())
        return totals, covered, mean_total


def main() -> int:
    n = validate()
    env = env_check()
    corpus_dir = production_corpus_dir(CORPUS)
    ident = corpus_identity(corpus_dir)
    print(f"\ncorpus = {ident.get('corpus')} @ {corpus_dir}")
    print(f"target_wpm = {TARGET_WPM}, candidates = {n}\n")

    t0 = time.time()
    tri = load_frequencies(str(corpus_dir / "trigrams.txt"))
    surf = TimeSurface(tri, target_wpm=TARGET_WPM, keep_seed_tables=True)
    print(f"surface built ({time.time() - t0:.1f}s, 6 models, 3 seeds kept)")

    V = Vec(surf)

    # ---- PARITY GATE: my vectorized sum vs the shipped loop, on 3 structurally different boards
    print("\nparity gate (vectorized vs shipped TimeSurface):")
    for probe in ("qwerty", "graphite", "BALL-1"):
        lay = ALL[probe]
        mine, cov, mine_mean = V.per_seed(lay)
        theirs = surf.seed_totals(lay)
        card = surf.card(lay)
        d_seed = max(abs(x - y) / y for x, y in zip(mine, theirs, strict=True))
        d_tot = abs(mine_mean - card.total_ms) / card.total_ms
        d_mpc = abs(mine_mean / cov - card.ms_per_char) / card.ms_per_char
        print(f"  {probe:12s} seed_totals rel {d_seed:.3e}  total_ms rel {d_tot:.3e}  "
              f"ms/char rel {d_mpc:.3e}")
        if max(d_seed, d_tot, d_mpc) > 1e-12:
            raise SystemExit(f"PARITY FAILED on {probe}: my sum is not the shipped gauge")
    print("  => parity OK to <1e-12 on all three probes")

    # ---- measure every candidate
    rows = {}
    for name, lay in ALL.items():
        totals, covered, mean_total = V.per_seed(lay)
        require_finite([*totals, mean_total, covered], f"{name} speed operands")
        rows[name] = {
            "layout": lay,
            "provenance": PROVENANCE[name],
            "covered_mass": covered,
            "coverage_pct": 100.0 * covered / surf.total_mass,
            "per_seed_ms_per_char": [t / covered for t in totals],
            "ms_per_char": mean_total / covered,   # seed-MEAN table (the shipped convention)
            "mean_of_seed_ms_per_char": float(np.mean([t / covered for t in totals])),
        }
        print(f"  {name:14s} ms/char {rows[name]['ms_per_char']:10.4f}  "
              f"cov {rows[name]['coverage_pct']:5.2f}%  "
              f"per-seed {[round(x, 3) for x in rows[name]['per_seed_ms_per_char']]}")

    out = {
        "env": env,
        "corpus": ident,
        "target_wpm": TARGET_WPM,
        "gauge": "K31 measured-keystroke surface: ms/char = sum_tri f*(T2[a,b]+Tcond[a,b,c]) / covered",
        "seeds": [0, 1, 2],
        "rows": rows,
        "elapsed_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
