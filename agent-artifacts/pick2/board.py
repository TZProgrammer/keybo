"""pick2 step 5: the decision board -- speed + the axes a learner feels + adoption friction.

Selection of axes is stated BEFORE the numbers are read, and each is included for a reason a
human learner would recognise, not because it exists:

  speed      ms/char on the K31 fitted surface (primary; but step 2 showed it cannot separate
             the leading group, so it acts as a GATE not a ranker)
  sfb        same-finger bigram % -- one finger typing two different keys in a row. The single
             most-felt defect; every community tool ranks it.
  sfs-dist   same-finger SKIP distance -- the same defect one key apart, weighted by travel.
  scissor    awkward row-crossing pairs on adjacent fingers (the gauge the user themself
             flagged as suspect -> `bad_scissor` reported alongside, both denominators named)
  lsb-dist   lateral-stretch distance (index/middle reaching outward)
  lat-span   the coverage-invariant lateral span (LSBWIDEN-1 says this is the rankable one;
             `lsb-narrow` is layout-dependent and NOT rankable -- excluded from win counts)
  redir      redirects (direction reversal within one hand) + bad_redirects (no index finger)
  roll/alt   rolls and alternation -- the axes the community optimizes FOR
  home_pct   corpus mass on the home row (effort proxy; also the Goodhart sanity gate)
  pinky_pct  corpus mass on the pinkies (the finger people complain about)
  imbalance  hand-load imbalance
  switching  adoption friction vs qwerty: keys unchanged, zxcv shortcut block preserved

EXCLUDED from every win count, with cause:
  sfr, alt, imbalance -- exactly invariant under within-hand permutation (discrimination.py),
                         so a tie is FORCED. `alt`/`imbalance` are still PRINTED (they do vary
                         across hand partitions) but never counted as a "win" between two
                         boards that share a partition.
  lsb-narrow          -- not rankable across layouts (LSBWIDEN-1).
  oxey-style/comfort  -- printed; these are scalarizations of the same axes, so counting them
                         alongside their components would double-count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE  # noqa: E402

from keybo.analysis.select import QWERTY30M, hand_balance_pct, switching_costs, usage_stats  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = "blend-v1"

#: axis -> (path into the analyze row, lower_is_better)
AXES = {
    "sfb":       (("gauges", "sfb"), True),
    "sfs-dist":  (("gauges", "sfs-dist"), True),
    "scissor":   (("gauges", "scissor"), True),
    "badsciss":  (("bad_scissor", "share"), True),
    "lsb-dist":  (("gauges", "lsb-dist"), True),
    "lat-span":  (("gauges", "lat-span"), True),
    "redir":     (("gauges", "redir"), True),
    "badredir":  (("redirects", "bad_redirects_total"), True),
    "sg_dist":   (("gauges", "sg_dist"), True),
    "roll":      (("gauges", "roll"), False),
}
#: printed but NOT counted (invariance / double-count / not-rankable)
UNCOUNTED = {
    "sfr":        (("gauges", "sfr"), True, "within-hand invariant -> tie is FORCED"),
    "alt":        (("gauges", "alt"), False, "within-hand invariant -> tie is FORCED"),
    "imbalance":  (("gauges", "imbalance"), True, "within-hand invariant -> tie is FORCED"),
    "lsb-narrow": (("gauges", "lsb-narrow"), True, "not rankable across layouts (LSBWIDEN-1)"),
    "oxey-style": (("gauges", "oxey-style"), True, "scalarization of counted axes"),
    "comfort":    (("gauges", "comfort"), True, "scalarization of counted axes"),
}


def dig(row, path):
    v = row
    for k in path:
        v = v[k]
    return v


def main() -> int:
    g = json.loads((HERE / f"gauges_{CORPUS}_wpm90.json").read_text())["rows"]
    speed = json.loads((HERE / f"speed_wpm90_{CORPUS}.json").read_text())["rows"]
    marg = json.loads((HERE / "margins.json").read_text())
    sup_path = HERE / f"support_{CORPUS}.json"
    sup = json.loads(sup_path.read_text())["rows"] if sup_path.is_file() else {}

    cdir = production_corpus_dir(CORPUS)
    bf = load_frequencies(str(cdir / "bigrams.txt"))
    # single-letter mass from the bigram table's first character (the corpus's own marginal)
    letter = {}
    for ng, f in bf.items():
        for c in ng[:1]:
            letter[c] = letter.get(c, 0) + f

    rows = {}
    for n, lay in ALL.items():
        u = usage_stats(lay, letter)
        # switching_costs needs a reference of the SAME charset: it looks every character of the
        # candidate up in the reference. qwerty30m (C30M, has ' and -) cannot host a board
        # carrying ';' or '/', so classic-charset boards get the classic qwerty reference.
        # A board on neither charset (dvorak/sturdy carry both ' and ;) has no matching
        # qwerty and its switching cost is not defined -- recorded as None, never a wrong number.
        ref = None
        if set(lay) == set(QWERTY30M):
            ref = QWERTY30M
        elif set(lay) == set(ALL["qwerty"]):
            ref = ALL["qwerty"]
        sw = switching_costs(lay, ref) if ref else None
        rows[n] = {
            "layout": lay,
            "provenance": PROVENANCE[n],
            "ms_per_char": speed[n]["ms_per_char"],
            "per_seed_ms_per_char": speed[n]["per_seed_ms_per_char"],
            "coverage_pct": speed[n]["coverage_pct"],
            "axes": {k: dig(g[n], p) for k, (p, _) in AXES.items()},
            "uncounted": {k: dig(g[n], p) for k, (p, _, _) in UNCOUNTED.items()},
            "home_row_pct": u["home_row_pct"],
            "pinky_pct": u["pinky_pct"],
            "left_hand_pct": hand_balance_pct(lay, letter),
            "fingers_pct": u["fingers"],
            "switching": sw,
            "switching_ref": ref,
            "support": sup.get(n, {}),
        }
    out = {"corpus": CORPUS, "axes_counted": {k: v[1] for k, v in AXES.items()},
           "axes_uncounted": {k: v[2] for k, v in UNCOUNTED.items()},
           "ruler": marg["ruler"], "rows": rows}
    (HERE / "board.json").write_text(json.dumps(out, indent=1))

    # ---- print
    ax = list(AXES)
    print(f"{'board':14s} {'prov':9s} {'ms/char':>9s} {'home%':>6s} {'pinky%':>7s} "
          + " ".join(f"{a[:8]:>8s}" for a in ax))
    order = sorted(rows, key=lambda n: rows[n]["ms_per_char"])
    for n in order:
        r = rows[n]
        print(f"{n:14s} {r['provenance']:9s} {r['ms_per_char']:9.3f} {r['home_row_pct']:6.1f} "
              f"{r['pinky_pct']:7.2f} " + " ".join(f"{r['axes'][a]:8.3f}" for a in ax))
    print(f"\nwrote {HERE / 'board.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
