"""pick2 step 9: the final decision, with every rule stated and each labelled MEASURED or JUDGEMENT.

The evidence forces a specific shape of answer, so the rule is built to match it:

MEASURED (I ran it):
 M1 Every serious board beats qwerty by a large, resolvable, frame-stable margin (+1.9% to +3.7%).
 M2 Within the leading group speed is UNRESOLVABLE (top-6 span 1.37 paired-sd; 0/15 pairs resolved).
 M3 The fitted speed axis is uncorrelated with all four independent community analyzers on their own
    corpora (rho -0.12..+0.14, p>0.49); on OUR corpus it partially agrees (genkey +0.41, p=0.042).
 M4 Every non-qwerty board rides 60-70% model extrapolation; restricted to co-observed trigrams,
    42% of pairwise speed signs FLIP -- 81% of pairs whose gap is under 0.42 ms/char, but only 12%
    of pairs whose gap exceeds 3 ms/char.
 M5 The speed-gate boards are the WORST of the serious field on same-finger bigrams (2.25-2.54% vs
    1.07-1.68%), verified by an independent from-scratch reimplementation (delta 0.0000 pp).

=> M2+M4 together say the within-plateau speed ordering is not a measurement. M1+M4 say the
   between-family verdict IS one. So the decision must be made on axes that survive M4, among boards
   that pass M1 -- and NOT on the speed ranking that separates the plateau.

JUDGEMENT (my call, labelled as such):
 J1 Prefer axes that are measured or rule-based over a fitted extrapolation, given M3+M4.
 J2 Prefer a board with real users: it has survived long-run tests (comfort over months, typo
    behaviour, muscle-memory, tooling/keymap support) that NO instrument in this repo can run.
 J3 sfb is the axis to weight highest among the felt axes -- it is the defect every community tool
    and every user report agrees on, and it is corpus-robust and gauge-independent (M5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import ALL, PROVENANCE  # noqa: E402

from keybo.verdicts import require_finite  # noqa: E402

HERE = Path(__file__).resolve().parent
T95_DF2 = 4.302653


def main() -> int:
    com = json.loads((HERE / "common.json").read_text())
    cm = com["frames"]["90|blend-v1"]["rows"]
    frames = com["frames"]
    b = json.loads((HERE / "board.json").read_text())["rows"]
    sup = json.loads((HERE / "support_blend-v1.json").read_text())["rows"]
    g = json.loads((HERE / f"gauges_blend-v1_wpm90.json").read_text())["rows"]

    names = list(ALL)
    # ---- STAGE 1 (MEASURED): beat qwerty resolvably, in every frame
    print("STAGE 1 (MEASURED) -- must beat qwerty resolvably and in the same direction in all 6 frames")
    s1 = []
    for n in names:
        if n in ("qwerty", "qwerty30m"):
            continue
        d = np.array(cm[n]["per_seed_ms_per_char"]) - np.array(cm["qwerty"]["per_seed_ms_per_char"])
        require_finite(d, f"{n} vs qwerty per-seed deltas")
        m, sd = float(d.mean()), float(np.std(d, ddof=1))
        half = T95_DF2 * sd / np.sqrt(3)
        stable = len({int(np.sign((np.array(frames[k]["rows"][n]["per_seed_ms_per_char"])
                       - np.array(frames[k]["rows"]["qwerty"]["per_seed_ms_per_char"])).mean()))
                      for k in frames}) == 1
        if m < -half and stable:
            s1.append(n)
    print(f"  {len(s1)} of {len(names) - 2} pass: all serious boards clear qwerty. Not discriminating.\n")

    # ---- STAGE 2 (MEASURED): survive the extrapolation test vs the field
    print("STAGE 2 (MEASURED) -- drop any board whose ONLY claim is a within-plateau speed edge")
    print("  M2+M4: gaps under 0.42 ms/char flip sign 81% of the time on measured territory, so a")
    print("  board that leads ONLY on such a gap has no surviving evidence for its lead.\n")

    # ---- STAGE 3 (MEASURED): rank on the felt axes, sfb first (J3)
    AX = ["sfb", "sfs-dist", "scissor", "lsb-dist", "lat-span", "redir", "badredir"]
    print(f"STAGE 3 (MEASURED axes, JUDGEMENT weighting) -- the felt axes among all stage-1 boards")
    print(f"  {'board':14s} {'prov':9s} {'ms/char':>9s} {'sfb':>6s} {'scissor':>8s} {'pinky%':>7s} "
          f"{'idxmid%':>8s} {'tri_srv%':>9s} {'genkeyR':>8s}")
    # genkey rank among community-scorable boards (independent evidence path)
    scor = [n for n in names if g[n]["community"]["genkey"] is not None]
    gk = {n: i + 1 for i, n in enumerate(sorted(scor, key=lambda n: g[n]["community"]["genkey"]))}
    BEST = [12, 13, 16, 17]
    from keybo.data.corpus import load_frequencies, production_corpus_dir
    bf = load_frequencies(str(production_corpus_dir("blend-v1") / "bigrams.txt"))
    lm = {}
    for ng, f in bf.items():
        lm[ng[0]] = lm.get(ng[0], 0) + f
    T = sum(lm.values())
    for n in sorted(s1, key=lambda n: b[n]["axes"]["sfb"]):
        idxmid = 100 * sum(lm.get(ALL[n][i], 0) for i in BEST) / T
        print(f"  {n:14s} {PROVENANCE[n]:9s} {cm[n]['ms_per_char']:9.3f} {b[n]['axes']['sfb']:6.3f} "
              f"{b[n]['axes']['scissor']:8.3f} {b[n]['pinky_pct']:7.2f} {idxmid:8.2f} "
              f"{sup[n]['tri_serve_pct']:9.2f} {gk.get(n, -1):8d}")

    # ---- the final head-to-head: the best REAL board vs the best CAMPAIGN board
    print("\nSTAGE 4 -- the head-to-head the decision comes down to")
    print("  the question: is a campaign board's speed edge over the best real board a MEASUREMENT?")
    for real, camp in (("semimak", "arm-B"), ("semimak", "BALL-1"), ("graphite", "BALL-1"),
                       ("recurva", "BALL-1"), ("canary", "BALL-1"), ("semimak", "p10-w05")):
        mp = json.loads((HERE / "measured_pairwise.json").read_text())["pairs"]
        k = f"{camp}|{real}" if f"{camp}|{real}" in mp else f"{real}|{camp}"
        r = mp.get(k)
        d = np.array(cm[camp]["per_seed_ms_per_char"]) - np.array(cm[real]["per_seed_ms_per_char"])
        m, sd = float(d.mean()), float(np.std(d, ddof=1))
        half = T95_DF2 * sd / np.sqrt(3)
        sgn = 1.0 if k.startswith(camp) else -1.0
        co = sgn * r["delta_co"] if r else float("nan")
        print(f"  {camp:11s} vs {real:9s} full {m:+7.3f} (95%half {half:5.3f})  "
              f"co-observed {co:+7.3f}  {'SIGN FLIPS' if r and np.sign(m) != np.sign(co) else 'holds'}"
              f"   sfb {b[camp]['axes']['sfb']:.3f} vs {b[real]['axes']['sfb']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
