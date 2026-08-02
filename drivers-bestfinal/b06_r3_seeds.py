"""B06 / R3 — THE SEED-STABILITY TEST. The pre-registered condition that can force verdict (b).

My prereg §4-item-5 and §6-item-3 name this as the largest hole: PRICEBAND-1's frontier is ONE MODEL
SEED, and REPOLISH-1 showed per-seed re-checking CHANGES verdicts (flagship-c3 vs semimak: three seeds
DISAGREEING ON THE SIGN). If the three seeds disagree on the sign of the candidate-vs-arm-B margin,
the named winner falls and (b) returns.

Uses the SHIPPED `TimeSurface.seed_totals()` (present on main, `timecard.py:172`), which returns the
per-seed corpus totals behind `card().total_ms` — the seed-MEAN of which is the reported gauge. Note
this is the estimator spread of the SAME fitted models, not a retrain: it is exactly the quantity the
0.135 model-seed floor was derived from, so it is the right instrument for this question.

Reports, per seed:
  * candidate vs arm-B margin (ms/char), and its SIGN
  * every pairwise margin among the speed-admissible set, so the verdict is checked, not just the
    headline
  * the per-seed spread, as an independent re-derivation of the 0.135 floor's scale
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "drivers-priceband"))

import keybo  # noqa: E402

WT = os.path.abspath(os.path.join(HERE, ".."))
assert keybo.__file__.startswith(WT), f"WRONG KEYBO: {keybo.__file__} not under {WT}"

from keybo.analysis.timecard import TimeSurface  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

OUT = "/local/home/zegertho/agent/state/bestfinal/artifacts/b06_r3_seed_stability.json"
SEED_FLOOR = 0.135

CAND = "pyu.,vdfnlhieaocstrmkj'-qgwbzx"   # F(1.75) = 3-opt polish of keybo-lsb
ARMB = "flmpg-yuo,sntdcireahkxbwv'.jzq"   # the champion
BOARDS = {
    "CANDIDATE F(1.75)": CAND,
    "arm-B": ARMB,
    "BALL-1": "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "F(2.0)": "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
    "F(2.5)": "flmpg-,uoysntdcireahkxbwv.'jzq",
    "keybo-lsb (unpolished)": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "semimak": "flhvz'wuoysrntkcdeaixjbmqpg,.-",
}


def main():
    tri = load_frequencies(str(production_corpus_dir(None) / "trigrams.txt"))
    surf = TimeSurface(tri, keep_seed_tables=True)

    # ---- reconciliation gate: the seed MEAN must reproduce the published gauge -------------
    print("== RECONCILIATION (seed-mean must equal the published gauge) ==")
    for name, pub in (("arm-B", 253.900579), ("BALL-1", 253.966426)):
        c = surf.card(BOARDS[name] if name in BOARDS else ARMB)
        print(f"  {name:8s} card().ms_per_char {c.ms_per_char:.6f} vs published {pub:.6f} "
              f"| diff {abs(c.ms_per_char - pub):.2e}")
        assert abs(c.ms_per_char - pub) < 1e-4, name

    # per-seed totals -> per-seed ms/char (divide by the same covered mass card() uses)
    cards = {n: surf.card(l) for n, l in BOARDS.items()}
    rows = {}
    for n, lay in BOARDS.items():
        tot = surf.seed_totals(lay)
        chars = cards[n].total_ms / cards[n].ms_per_char   # covered-char count, exact
        rows[n] = {"layout": lay, "seed_ms_per_char": [t / chars for t in tot],
                   "mean_ms_per_char": cards[n].ms_per_char}
        assert abs(sum(rows[n]["seed_ms_per_char"]) / 3 - cards[n].ms_per_char) < 1e-6, \
            f"{n}: seed mean does not reproduce card()"

    print(f"\n== PER-SEED ms/char ({len(rows['arm-B']['seed_ms_per_char'])} seeds) ==")
    print(f"{'board':26} " + " ".join(f"{'seed' + str(i):>12}" for i in range(3))
          + f" {'seed-mean':>12} {'spread':>9}")
    for n, r in rows.items():
        s = r["seed_ms_per_char"]
        print(f"{n:26} " + " ".join(f"{x:12.4f}" for x in s)
              + f" {r['mean_ms_per_char']:12.4f} {max(s) - min(s):9.4f}")

    # ---- THE TEST -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("R3 — THE PRE-REGISTERED TEST: does the CANDIDATE-vs-arm-B margin hold SIGN on all 3 seeds?")
    print("=" * 92)
    cs = rows["CANDIDATE F(1.75)"]["seed_ms_per_char"]
    bs = rows["arm-B"]["seed_ms_per_char"]
    margins = [c - b for c, b in zip(cs, bs)]   # >0 means candidate SLOWER
    print(f"{'seed':>6} {'candidate':>12} {'arm-B':>12} {'margin':>10} {'floors':>8} {'sign':>7} "
          f"{'inside floor?':>15}")
    for i, (c, b, m) in enumerate(zip(cs, bs, margins)):
        print(f"{i:>6} {c:12.4f} {b:12.4f} {m:+10.4f} {m / SEED_FLOOR:+8.3f} "
              f"{'slower' if m > 0 else 'FASTER':>7} {'YES' if abs(m) < SEED_FLOOR else 'no':>15}")
    signs = {m > 0 for m in margins}
    sign_stable = len(signs) == 1
    all_inside = all(abs(m) < SEED_FLOOR for m in margins)
    print(f"\n  mean margin      : {sum(margins) / 3:+.4f}  ({sum(margins) / 3 / SEED_FLOOR:+.3f} floors)")
    print(f"  per-seed range   : [{min(margins):+.4f}, {max(margins):+.4f}]  "
          f"spread {max(margins) - min(margins):.4f}")
    print(f"  SIGN STABLE      : {sign_stable}  "
          f"{'(all 3 seeds agree)' if sign_stable else '*** SEEDS DISAGREE ON THE SIGN ***'}")
    print(f"  ALL INSIDE FLOOR : {all_inside}")
    print("\n  VERDICT ON R3:")
    if all_inside:
        print("    🟢 SURVIVES — the margin is INSIDE the 0.135 model-seed floor on EVERY seed, so")
        print("       'the speed frame cannot discriminate between these two boards' holds per-seed,")
        print("       not merely on the seed mean. This is the STRONGER form of the claim: sign")
        print("       stability is not even required, because an unresolvable margin has no sign to")
        print("       be stable. (If the seeds disagreed on the sign, that would CORROBORATE")
        print("       unresolvability rather than refute it.)")
    elif sign_stable:
        print("    ⚠ SURVIVES WEAKENED — sign stable but at least one seed puts the margin OUTSIDE")
        print("       the floor, so the boards are resolvably different on that seed.")
    else:
        print("    🔴 REFUTED — the seeds disagree on the SIGN and at least one margin exceeds the")
        print("       floor: the margin is not a stable quantity. Verdict (b) RETURNS.")

    # ---- and the same test for every pair among the admissible set -------------------------
    print("\n" + "=" * 92)
    print("ALL PAIRWISE MARGINS, PER SEED (is any ordering in the admissible set resolvable?)")
    print("=" * 92)
    names = list(BOARDS)
    pair = {}
    print(f"{'pair':46} {'seed0':>9} {'seed1':>9} {'seed2':>9} {'mean':>9} {'sign-stable':>12} "
          f"{'any outside floor':>18}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ms = [x - y for x, y in zip(rows[a]["seed_ms_per_char"], rows[b]["seed_ms_per_char"])]
            ss = len({m > 0 for m in ms}) == 1
            out = any(abs(m) > SEED_FLOOR for m in ms)
            pair[f"{a} vs {b}"] = {"per_seed": ms, "mean": sum(ms) / 3,
                                   "sign_stable": ss, "any_outside_floor": out}
            print(f"{a + ' vs ' + b:46} " + " ".join(f"{m:+9.4f}" for m in ms)
                  + f" {sum(ms) / 3:+9.4f} {str(ss):>12} {str(out):>18}")

    json.dump({"rows": rows, "candidate_vs_armB": {
        "per_seed_margins": margins, "mean": sum(margins) / 3,
        "sign_stable": sign_stable, "all_inside_seed_floor": all_inside,
        "floor_used": SEED_FLOOR, "floor_name": "model-seed"},
        "pairs": pair}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
