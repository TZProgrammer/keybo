"""pick2 step 7: speed on the COMMON typable subset -- the strictly comparable frame.

Step 1 scored every board on its OWN typable subset (ms/char over covered mass). That is the
shipped `TimeCard` convention and it is right for same-charset cohorts, but this cohort spans
THREE charsets, and then the boards are being scored on DIFFERENT corpus subsets: a classic
board (`;` `/`) never pays for the apostrophe it cannot type, and a C30M board (`'` `-`) never
pays for the semicolon. `keybo score` already fixes this with `common_ngrams` -- the intersection
of what every compared board can type -- and I apply the same fix to the primary time gauge.

Reported BOTH ways, because they answer different questions and the difference is the finding:
  own-subset  : "how fast is this board at the text it can type" (flatters a narrow charset)
  common      : "how fast is this board at the text EVERY board can type" (strictly comparable)
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE  # noqa: E402

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import corpus_identity, load_frequencies, production_corpus_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
T95_DF2 = 4.302653


def run(wpm: float, corpus: str) -> dict:
    cdir = production_corpus_dir(corpus)
    tri_all = load_frequencies(str(cdir / "trigrams.txt"))
    surf = TimeSurface(tri_all, target_wpm=wpm, keep_seed_tables=True)

    common_chars = set.intersection(*(set(l) for l in ALL.values())) | {" "}
    tri = {k: v for k, v in surf.tri.items() if len(k) == 3 and set(k) <= common_chars}
    total_all = sum(v for k, v in surf.tri.items() if len(k) == 3)
    mass = sum(tri.values())
    ng = list(tri)
    f = np.array([tri[k] for k in ng], dtype=np.float64)

    rows = {}
    for name, lay in ALL.items():
        slot = surf._slot_of(lay)  # refuses malformed layouts
        a = np.array([slot[g[0]] for g in ng], np.int32)
        b = np.array([slot[g[1]] for g in ng], np.int32)
        c = np.array([slot[g[2]] for g in ng], np.int32)
        per_seed = [float(((T2[a, b] + Tc[a, b, c]) * f).sum()) / mass
                    for T2, Tc in zip(surf._T2s, surf._Tcs, strict=True)]
        mean = float(((surf._T2[a, b] + surf._Tc[a, b, c]) * f).sum()) / mass
        rows[name] = {"layout": lay, "provenance": PROVENANCE[name],
                      "per_seed_ms_per_char": per_seed, "ms_per_char": mean}
    return {"corpus": corpus, "corpus_identity": corpus_identity(cdir), "target_wpm": wpm,
            "common_chars": "".join(sorted(common_chars)),
            "n_common_trigrams": len(tri),
            "common_mass_pct_of_corpus": 100.0 * mass / total_all,
            "rows": rows}


def main() -> int:
    import keybo
    print("keybo.__file__ =", keybo.__file__)
    t0 = time.time()
    frames = {}
    for wpm in (90, 110, 120):
        for corpus in ("blend-v1", "iweb"):
            frames[f"{wpm}|{corpus}"] = run(float(wpm), corpus)
            print(f"  frame wpm{wpm} {corpus}: {frames[f'{wpm}|{corpus}']['n_common_trigrams']} "
                  f"common trigrams = {frames[f'{wpm}|{corpus}']['common_mass_pct_of_corpus']:.2f}% of corpus")
    base = frames["90|blend-v1"]
    print(f"\ncommon charset = {base['common_chars']!r}  "
          f"({len(base['common_chars'])} chars incl. space)")
    print(f"common subset  = {base['n_common_trigrams']} trigrams = "
          f"{base['common_mass_pct_of_corpus']:.2f}% of the trigram corpus\n")

    own = json.loads((HERE / "speed_wpm90_blend-v1.json").read_text())["rows"]
    rows = base["rows"]
    names = sorted(rows, key=lambda n: rows[n]["ms_per_char"])
    print(f"{'board':14s} {'prov':9s} {'COMMON':>9s} {'own-subset':>11s} {'shift':>7s} {'rank move':>10s}")
    own_rank = {n: i + 1 for i, n in enumerate(sorted(own, key=lambda n: own[n]["ms_per_char"]))}
    for i, n in enumerate(names):
        d = rows[n]["ms_per_char"] - own[n]["ms_per_char"]
        mv = own_rank[n] - (i + 1)
        print(f"{n:14s} {rows[n]['provenance']:9s} {rows[n]['ms_per_char']:9.4f} "
              f"{own[n]['ms_per_char']:11.4f} {d:+7.3f} {mv:+10d}")

    # gate on the common frame
    fastest = names[0]
    gate = []
    for n in names:
        d = np.array(rows[n]["per_seed_ms_per_char"]) - np.array(rows[fastest]["per_seed_ms_per_char"])
        m, sd = float(d.mean()), float(np.std(d, ddof=1))
        half = T95_DF2 * sd / np.sqrt(3)
        signs = {int(np.sign((np.array(frames[k]["rows"][n]["per_seed_ms_per_char"])
                            - np.array(frames[k]["rows"][fastest]["per_seed_ms_per_char"])).mean()))
                 for k in frames}
        if not ((m > half) and len(signs) == 1):
            gate.append(n)
    print(f"\nCOMMON-frame speed gate (fastest = {fastest}): {len(gate)} pass -> {gate}")

    out = {"frames": frames, "gate_fastest": fastest, "gate_passers": gate,
           "elapsed_s": time.time() - t0}
    (HERE / "common.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {HERE / 'common.json'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
