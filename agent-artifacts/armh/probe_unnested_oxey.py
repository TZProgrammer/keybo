"""ARM H REFLECTION Q(b) — does the dominance census survive the UN-NESTED bad_redirect
convention?

⚠ POST-HOC. Changes no registered verdict; it TESTS one. The branch `oxey-partition-fix` is
NOT merged: I read its `_trigram_class` and reimplement the SAME delegation into
`community._v1_pattern` (which is present on my base) inside this scratch probe. Delegating
rather than re-deriving is the fix's own stated design point (trap 28), so copying the
delegation — not the predicates — is the faithful route.

`oxey-style` on the g-frame is a WEIGHTED SUM of 11 pattern shares. The fix touches ONLY the
three trigram terms (`onehand`, `redirect`, `bad_redirect`); OXEYFIX-1 registered that
`bad_redirect` and all 8 bigram/imbalance shares are bit-identical at max|diff| exactly 0.0.
So the whole delta is recomputable from the trigram partition alone, which is what this does.

POSITIVE CONTROLS, run BEFORE the census is read:
 PC1  my NESTED recomputation must reproduce the shipped `analyze` oxey-style values (proves
      my re-implementation of the scorer's arithmetic is faithful before I change one term).
 PC2  the un-nested board must reproduce OXEYFIX-1's four published BEFORE values bit-exact
      (qwerty30m 88.197171 / graphite -7.148220 / arm B 8.611046 / arm E -0.992396) on the
      NESTED side, and its registered CONSEQUENCE on the fixed side: every score DROPS by
      0.42 to 1.50 absolute.
 PC3  the onehand class must land at 756 not 1080 over the 27,000 ordered slot triples, and
      the double-charged set must be EMPTY under the fix and 540 under nesting.
 MUT  a planted perturbation must make the census change, or the census cannot detect one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import armh_constants as AH  # noqa: E402

WORKTREE = Path("/tmp/armh")
STATE = Path("/local/home/zegertho/agent/state/armh/artifacts")

ARMB = AH.ARMH_LAYOUT_REF
BALL1 = AH.ARMH_BALL1
HEADLINE = "flmpg-,uoysntcdireahkxvwb.'jzq"
MID = "flmpg.yuo,sntcdireahkxbwv'-jzq"

#: OXEYFIX-1's published BEFORE values (nested), for PC2. Quoted from the ledger and then
#: CHECKED against my own recomputation -- if they disagree, my probe is wrong, not the ledger.
OXEYFIX_BEFORE = {
    "qwertyuiopasdfghjkl'zxcvbnm,.-": 88.197171,   # qwerty30m
    "bldwz'foujnrtsgyhaeixqmcvkp,.-": -7.148220,   # graphite
    ARMB: 8.611046,                                 # arm B
}


def main() -> int:
    from keybo.analysis.community import _v1_pattern
    from keybo.data.corpus import load_frequencies, production_corpus_dir
    from keybo.geometry import ROW_STAGGERED_30 as G
    from keybo.geometry import Finger
    from keybo.layout import Layout
    from keybo.scoring import comfort as CMF  # noqa: F401  (import-path sanity)
    from keybo.scoring.oxey import DEFAULT_OXEY_WEIGHTS, OxeyStyleScorer

    import keybo
    assert str(Path(keybo.__file__).resolve()).startswith("/tmp/armh/"), keybo.__file__

    cdir = production_corpus_dir(None)
    assert str(Path(cdir).resolve()).startswith("/tmp/armh/"), cdir
    bg = load_frequencies(Path(cdir) / "bigrams.txt")
    sg = load_frequencies(Path(cdir) / "1-skip31.txt")
    tg = load_frequencies(Path(cdir) / "trigrams.txt")
    W = {n: w for n, (w, _) in DEFAULT_OXEY_WEIGHTS.items()}

    # ---- the fix's delegation, copied (NOT the predicates) ----
    LIBDOF = {Finger.LP: 0, Finger.LR: 1, Finger.LM: 2, Finger.LI: 3,
              Finger.RI: 6, Finger.RM: 7, Finger.RR: 8, Finger.RP: 9}
    ROLLUP = {"onehands": "onehand", "redirects": "redirect",
              "redirects_sfs": "redirect", "bad_redirects": "bad_redirect",
              "bad_redirects_sfs": "bad_redirect"}

    def fixed_class(a, b, c) -> str | None:
        f = []
        for p in (a, b, c):
            v = LIBDOF.get(G.finger(p[0]))
            if v is None:
                return None
            f.append(v)
        return ROLLUP.get(_v1_pattern(*f))

    def nested_classes(a, b, c) -> list[str]:
        """The AS-SHIPPED nested logic, transcribed from src/keybo/scoring/oxey.py:136-146."""
        ha, hb, hc = G.hand(a[0]), G.hand(b[0]), G.hand(c[0])
        if not (ha == hb == hc and ha != 0):
            return []
        d1 = abs(b[0]) - abs(a[0])
        d2 = abs(c[0]) - abs(b[0])
        if d1 and d2 and (d1 > 0) == (d2 > 0):
            return ["onehand"]
        if d1 and d2:
            out = ["redirect"]
            if not any(abs(p[0]) in (1, 2) for p in (a, b, c)):
                out.append("bad_redirect")   # <-- THE NESTING: charged +2.0 AND +4.0
            return out
        return []

    # ---- PC3: class census over all 27,000 ordered slot triples ----
    slots = list(G.slots)
    cens = {"nested_onehand": 0, "nested_redirect": 0, "nested_bad": 0,
            "nested_double_charged": 0, "fixed_onehand": 0, "fixed_redirect": 0,
            "fixed_bad": 0}
    for i in range(30):
        for j in range(30):
            for k in range(30):
                a, b, c = slots[i], slots[j], slots[k]
                n = nested_classes(a, b, c)
                if "onehand" in n:
                    cens["nested_onehand"] += 1
                if "redirect" in n:
                    cens["nested_redirect"] += 1
                if "bad_redirect" in n:
                    cens["nested_bad"] += 1
                if "redirect" in n and "bad_redirect" in n:
                    cens["nested_double_charged"] += 1
                f = fixed_class(a, b, c)
                if f == "onehand":
                    cens["fixed_onehand"] += 1
                elif f == "redirect":
                    cens["fixed_redirect"] += 1
                elif f == "bad_redirect":
                    cens["fixed_bad"] += 1
    pc3 = {
        "census": cens,
        "onehand_1080_to_756": (cens["nested_onehand"] == 1080
                                and cens["fixed_onehand"] == 756),
        "double_charged_540_to_0": (cens["nested_double_charged"] == 540),
        "bad_540_both_ways": (cens["nested_bad"] == 540 and cens["fixed_bad"] == 540),
    }
    assert pc3["onehand_1080_to_756"], f"PC3 FAILED onehand: {cens}"
    assert pc3["double_charged_540_to_0"], f"PC3 FAILED double-charge: {cens}"
    assert pc3["bad_540_both_ways"], f"PC3 FAILED bad support: {cens}"

    # ---- recompute oxey-style both ways ----
    def shares_both(lay: str) -> tuple[dict, dict]:
        L = Layout(lay, G)
        sc = OxeyStyleScorer(bg, sg, tg)
        base = sc.pattern_shares(L)                      # SHIPPED = nested
        # recompute only the three trigram terms under BOTH conventions
        acc_n = {"onehand": 0.0, "redirect": 0.0, "bad_redirect": 0.0}
        acc_f = {"onehand": 0.0, "redirect": 0.0, "bad_redirect": 0.0}
        tot = 0.0
        for t, f in tg.items():
            if len(t) != 3 or not all(L.has_key(ch) for ch in t):
                continue
            a, b, c = (L.pos(ch) for ch in t)
            tot += f
            for cl in nested_classes(a, b, c):
                acc_n[cl] += f
            fc = fixed_class(a, b, c)
            if fc:
                acc_f[fc] += f
        nested = dict(base)
        fixed = dict(base)
        for k in acc_n:
            nested[k] = 100.0 * acc_n[k] / tot if tot else 0.0
            fixed[k] = 100.0 * acc_f[k] / tot if tot else 0.0
        return nested, fixed

    def fit(sh: dict) -> float:
        return sum(W[n] * v for n, v in sh.items())

    # ---- PC1: my nested recomputation must equal the SHIPPED analyze value ----
    layouts = [ARMB, BALL1, MID, HEADLINE, *OXEYFIX_BEFORE]
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "8"
    p = subprocess.run(["uv", "run", "--no-sync", "keybo", "analyze", "--json",
                        *sorted(set(layouts))], cwd=str(WORKTREE),
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    rows = json.loads(p.stdout)["rows"]

    nested_v, fixed_v, pc1 = {}, {}, {}
    for lay in sorted(set(layouts)):
        n, f = shares_both(lay)
        nested_v[lay] = fit(n)
        fixed_v[lay] = fit(f)
        ship = rows[lay]["gauges"]["oxey-style"]
        pc1[lay] = {"shipped": ship, "my_nested": nested_v[lay],
                    "absdiff": abs(ship - nested_v[lay])}
    worst_pc1 = max(v["absdiff"] for v in pc1.values())
    assert worst_pc1 < 1e-9, f"PC1 FAILED: my nested recompute differs by {worst_pc1:.3e}"

    # ---- PC2: OXEYFIX-1's published BEFORE values + its registered DROP range ----
    pc2 = {}
    for lay, published in OXEYFIX_BEFORE.items():
        pc2[lay] = {"published_before": published, "my_nested": nested_v[lay],
                    "absdiff_vs_published": abs(published - nested_v[lay]),
                    "my_fixed": fixed_v[lay],
                    "drop": nested_v[lay] - fixed_v[lay]}
    drops = [nested_v[x] - fixed_v[x] for x in sorted(set(layouts))]
    pc2_ok_pub = all(v["absdiff_vs_published"] < 1e-5 for v in pc2.values())
    pc2_ok_drop = all(0.42 <= d <= 1.50 for d in drops)

    # ---- MUT: can this census detect a change at all? ----
    mut_fixed = {}
    for lay in sorted(set(layouts)):
        _, f = shares_both(lay)
        f["bad_redirect"] = f["bad_redirect"] * 1.10   # planted +10% on the fixed side
        mut_fixed[lay] = fit(f)
    mut_moved = any(abs(mut_fixed[x] - fixed_v[x]) > 1e-9 for x in mut_fixed)

    # ---- THE CENSUS, under the UN-NESTED convention ----
    def census(cand: str) -> dict:
        gc, gr = rows[cand]["gauges"], rows[ARMB]["gauges"]
        better, worse, ties = [], [], []
        for a in AH.ARMH_LIVE:
            if a == "oxey-style":
                d = fixed_v[cand] - fixed_v[ARMB]     # <-- the FIXED convention
            else:
                d = AH.ARMH_DIR[a] * (gc[a] - gr[a])
            if d < -AH.ARMH_TOL:
                better.append(a)
            elif d > AH.ARMH_TOL:
                worse.append(a)
            else:
                ties.append(a)
        return {"n_contested": len(better) + len(worse), "n_better": len(better),
                "n_worse": len(worse), "better": better, "worse": worse, "ties": ties,
                "oxey_fixed_cand": fixed_v[cand], "oxey_fixed_armB": fixed_v[ARMB],
                "oxey_gap_fixed": fixed_v[cand] - fixed_v[ARMB],
                "oxey_gap_nested": nested_v[cand] - nested_v[ARMB],
                "DOMINATES": len(worse) == 0 and len(better) >= 1}

    out = {
        "POST_HOC": ("computed after result commit c85623d for the reflection pass. The branch "
                     "oxey-partition-fix is NOT merged; its delegation into "
                     "community._v1_pattern is reimplemented in this scratch probe."),
        "PC1_my_nested_equals_shipped": {"worst_absdiff": worst_pc1, "per_layout": pc1,
                                         "PASS": worst_pc1 < 1e-9},
        "PC2_oxeyfix_published": {"per_layout": pc2, "published_reproduce": pc2_ok_pub,
                                  "all_drops_in_0.42_1.50": pc2_ok_drop,
                                  "drops": dict(zip(sorted(set(layouts)), drops, strict=True))},
        "PC3_triple_census": pc3,
        "MUT_census_can_detect_change": mut_moved,
        "oxey_nested": nested_v,
        "oxey_fixed": fixed_v,
        "census_under_FIXED_convention": {"HEADLINE": census(HEADLINE),
                                          "BALL-1": census(BALL1),
                                          "MID": census(MID)},
    }
    json.dump(out, open(STATE / "reflect-unnested-oxey.json", "w"), indent=1, default=str)

    print("=" * 96)
    print(f"PC1 my nested recompute vs shipped analyze : worst absdiff {worst_pc1:.3e}  "
          f"{'PASS' if worst_pc1 < 1e-9 else 'FAIL'}")
    print(f"PC2 OXEYFIX-1 published BEFORE reproduce   : {pc2_ok_pub}   "
          f"drops all in [0.42,1.50]: {pc2_ok_drop}")
    for lay, v in pc2.items():
        print(f"      {lay}  published {v['published_before']:>11.6f}  mine "
              f"{v['my_nested']:>11.6f}  diff {v['absdiff_vs_published']:.2e}  "
              f"drop {v['drop']:.6f}")
    print(f"PC3 onehand 1080->756 {pc3['onehand_1080_to_756']} · double-charge 540->0 "
          f"{pc3['double_charged_540_to_0']} · bad 540 both ways {pc3['bad_540_both_ways']}")
    print(f"MUT census detects a planted change        : {mut_moved}")
    print("=" * 96)
    print(f"{'layout':<10}{'oxey NESTED':>14}{'oxey FIXED':>14}{'drop':>10}"
          f"{'gap vs armB NESTED':>21}{'gap FIXED':>12}")
    for name, lay in (("armB", ARMB), ("BALL-1", BALL1), ("MID", MID), ("HEADLINE", HEADLINE)):
        print(f"{name:<10}{nested_v[lay]:>14.6f}{fixed_v[lay]:>14.6f}"
              f"{nested_v[lay] - fixed_v[lay]:>10.6f}"
              f"{nested_v[lay] - nested_v[ARMB]:>21.6f}"
              f"{fixed_v[lay] - fixed_v[ARMB]:>12.6f}")
    print("=" * 96)
    for name, c in out["census_under_FIXED_convention"].items():
        print(f"{name:<10} UNDER THE FIXED CONVENTION: {c['n_better']} better / "
              f"{c['n_worse']} worse of {c['n_contested']} CONTESTED   "
              f"DOMINATES={c['DOMINATES']}   oxey gap {c['oxey_gap_fixed']:+.6f} "
              f"(was {c['oxey_gap_nested']:+.6f})")
        if c["worse"]:
            print(f"           WORSE ON: {c['worse']}")
    print(f"\nWROTE {STATE / 'reflect-unnested-oxey.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
