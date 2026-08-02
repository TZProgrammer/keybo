"""pick2 step 3: is each board's predicted gain backed by MEASUREMENT or by EXTRAPOLATION?

The gauge that ranks the boards is a fitted model. A fitted model returns a number for every
position n-gram, including position n-grams NOBODY IN THE STUDY EVER TYPED. So "board X is
faster" can mean two very different things:

  (a) the corpus mass that makes X fast lands on position n-grams the K31 study OBSERVED, or
  (b) it lands where the model is inventing, i.e. the gain is a tree extrapolation.

This is decision-critical and it is the axis a speed column cannot show. `select.RawSupport`
exists for exactly this (SELECT-1) -- I use the shipped implementation rather than rolling my
own, at its registered production convention (serve bucket 80, min_cell 10, wpm [40,140) x 20).

Reads the K31 stroke tables read-only from ~/keybo-e2e/ (the tables the shipped k31 models were
trained on). ~1.2 GB of TSV, so this is the slow step.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE  # noqa: E402

from keybo.analysis.select import RawSupport  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

BI = Path.home() / "keybo-e2e" / "bistrokes31_v1.tsv"
TRI = Path.home() / "keybo-e2e" / "tristrokes31_cond_v1.tsv"
HERE = Path(__file__).resolve().parent
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "blend-v1"


def main() -> int:
    import keybo
    print("keybo.__file__ =", keybo.__file__)
    for p in (BI, TRI):
        if not p.is_file():
            raise SystemExit(f"missing stroke table {p}")
        print(f"  {p}  {p.stat().st_size / 1e6:.0f} MB")

    t0 = time.time()
    print("\nbuilding RawSupport from the K31 stroke tables (production convention)...")
    rs = RawSupport.from_tsvs(BI, TRI)
    print(f"  built in {time.time() - t0:.0f}s")
    print(f"  observed position-bigram sets:  serve {len(rs.bi_serve):6d}   any {len(rs.bi_any):6d}")
    print(f"  observed position-trigram sets: serve {len(rs.tri_serve):6d}   any {len(rs.tri_any):6d}")

    cdir = production_corpus_dir(CORPUS)
    bf = load_frequencies(str(cdir / "bigrams.txt"))
    tf = load_frequencies(str(cdir / "trigrams.txt"))
    print(f"  corpus = {CORPUS} @ {cdir}\n")

    rows = {}
    print(f"  {'board':14s} {'bi_serve%':>10s} {'bi_any%':>9s} {'tri_serve%':>11s} {'tri_any%':>9s}")
    for name, lay in ALL.items():
        s = rs.support(lay, bf, tf)
        rows[name] = {"layout": lay, "provenance": PROVENANCE[name], **s}
        print(f"  {name:14s} {s['bi_serve_pct']:10.3f} {s['bi_any_pct']:9.3f} "
              f"{s['tri_serve_pct']:11.3f} {s['tri_any_pct']:9.3f}")

    out = {
        "convention": "serve bucket 80, min_cell 10, wpm [40,140) width 20 (RawSupport default)",
        "stroke_tables": {"bigram": str(BI), "trigram": str(TRI)},
        "observed_sets": {
            "bi_serve": len(rs.bi_serve), "bi_any": len(rs.bi_any),
            "tri_serve": len(rs.tri_serve), "tri_any": len(rs.tri_any),
        },
        "corpus": CORPUS,
        "rows": rows,
        "elapsed_s": time.time() - t0,
    }
    (HERE / f"support_{CORPUS}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE / f'support_{CORPUS}.json'}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
