"""Assemble the five mandated judgements for every arm's champion into one JSON + table.

Run AFTER the three arms complete. Reads `runs/arm-<arm>.json`, judges each champion
against the five incumbents and three reference layouts, and writes
`judgement.json` + a printable table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import evobj as EV  # noqa: E402
import judge as J  # noqa: E402

from keybo.analysis.evidence_scorer import LIVE_GAUGES  # noqa: E402
from keybo.data.corpus import load_frequencies, production_corpus_dir  # noqa: E402

STATE = Path("/local/home/zegertho/agent/state/optevidence/artifacts")
ARMS = ("evidence", "baseline", "constrained")
WRONG_SIGNED = ("scissor", "sfb", "sfb-dist", "lsb-dist", "sfs")


def main() -> int:
    out: dict = {"arms": {}, "modelled_only": (
        "MODELLED ONLY: every number is attribution of a FITTED timing surface, not a "
        "measurement of realized typing speed. tau saturated at 1.0 and Phase-D was "
        "cancelled. No layout here is promoted or adopted.")}

    # ---- champions ----
    champs: dict[str, str] = {}
    for arm in ARMS:
        path = STATE / "runs" / f"arm-{arm}.json"
        if not path.exists():
            print(f"MISSING {path} — arm not finished", file=sys.stderr)
            continue
        blob = json.load(open(path))
        champs[f"champ-{arm}"] = blob["champion"]["layout"]
        out["arms"][arm] = {
            "champion": blob["champion"], "unique_evals": blob["unique_evals"],
            "islands": blob["islands"], "epochs_run": blob["epochs_run"],
            "seed": blob["seed"], "bounds": blob.get("bounds"),
            "objective_unit": blob["objective_unit"], "elapsed_s": blob["elapsed_s"],
            "top10": blob["top50"][:10],
        }
    assert champs, "no arm produced a champion"

    everyone = {**champs, **J.INCUMBENTS, **J.REFERENCE}
    names = list(everyone)
    print(f"judging {len(names)} layouts: {names}")

    corpus_dir = production_corpus_dir(None)
    out["corpus"] = str(corpus_dir)
    from keybo.data.corpus import corpus_identity
    out["corpus_identity"] = corpus_identity(corpus_dir)

    # ---- J1: trained-objective score + J2 ms/char + out-of-domain, via the fast path ----
    fe = EV.FastEval(corpus=None, weights_json=str(
        "/local/home/zegertho/agent/state/evidence-scorer/artifacts/arm-random400-native.json"),
        with_surface=True)
    perms = np.stack([EV.perm_of(everyone[n]) for n in names])
    g = fe.gauges(perms)
    ev = fe.evidence_score(g)
    ood = fe.out_of_domain(g)
    curve_domain = {c.metric: list(c.domain) for c in fe.curves}

    out["per_layout"] = {}
    for i, n in enumerate(names):
        ood_set = sorted([k for k in LIVE_GAUGES if bool(ood[k][i])])
        out["per_layout"][n] = {
            "layout": everyone[n],
            "evidence_score": float(ev[i]),
            "ms_per_char": float(g["_ms_per_char"][i]),
            "total_ms": float(g["_total_ms"][i]),
            "gauges14": {k: float(g[k][i]) for k in LIVE_GAUGES},
            "out_of_domain": ood_set,
            "n_out_of_domain": len(ood_set),
            "out_of_domain_detail": {
                k: {"level": float(g[k][i]), "valid_domain": curve_domain[k],
                    "distance_outside": float(max(curve_domain[k][0] - g[k][i],
                                                  g[k][i] - curve_domain[k][1], 0.0))}
                for k in ood_set},
        }
    out["valid_domains"] = curve_domain

    # ---- J2 paired: per-seed ms/char + the paired resolution ----
    trigrams = load_frequencies(str(corpus_dir / "trigrams.txt"))
    print("computing paired resolution (loads 6 models, keeps seed tables) ...", flush=True)
    out["paired"] = J.paired_resolution(everyone, trigrams)

    # ---- J4: normalized six-surface floor (corpus-matched ceilings) ----
    print("deriving corpus-matched ceilings + normalized floors ...", flush=True)
    six = J.SixSurface(corpus_dir / "trigrams.txt")
    six_iweb = J.SixSurface(Path("/tmp/optev/data/corpus/trigrams.txt"))
    pc = max(abs(six_iweb.ceiling_map[s] - J.FROZEN_IWEB_CEILINGS[s]) for s in J.SIX)
    out["ceilings"] = {"corpus_derived": six.ceiling_map, "frozen_iweb": J.FROZEN_IWEB_CEILINGS,
                       "iweb_rederivation_worst_abs_diff": pc,
                       "positive_control": "PASS" if pc < 1e-9 else "FAIL"}
    assert pc < 1e-9, f"ceiling re-derivation failed the iWeb positive control: {pc:.3e}"
    for n in names:
        p = EV.perm_of(everyone[n])
        out["per_layout"][n]["normfloor"] = six.normfloor(p)
        out["per_layout"][n]["mean_saved_pct"] = six.mean_saved(p)
        out["per_layout"][n]["saved_per_surface"] = six.saved(p).tolist()

    # ---- J3: the 19-gauge frame via the shipped CLI ----
    print("running keybo analyze --json on all layouts ...", flush=True)
    an = J.analyze_json(everyone)
    out["analyze_frame"] = {"gauge_frame": an["gauge_frame"], "corpus": an["corpus"],
                            "skipgram_table": an["skipgram_table"],
                            "target_wpm": an["target_wpm"]}
    out["analyze_extra_rows"] = an["_extra_rows"]  # the --ref row, not one of ours
    spec_to_name = {everyone[n]: n for n in names}
    rows19: dict[str, dict] = {}
    for row in an["rows"].values():
        name = spec_to_name.get(row["layout"])
        if name is not None:  # skip analyze's own --ref row
            rows19[name] = row
    assert set(rows19) == set(names), f"missing rows for {sorted(set(names) - set(rows19))}"
    for n in names:
        out["per_layout"][n]["gauges19"] = {
            **rows19[n]["gauges"],
            "genkey": rows19[n]["community"]["genkey"],
            "oxeylyzer1": rows19[n]["community"]["oxeylyzer1"],
            "oxeylyzer2": rows19[n]["community"]["oxeylyzer2"],
            "wfd": rows19[n]["community"]["wfd"],
        }
        out["per_layout"][n]["community_primed"] = rows19[n]["community_primed"]
        out["per_layout"][n]["analyze_ms_per_char"] = rows19[n]["time"]["ms_per_char"]

    # ---- J5: dominance on the 10-axis frame ----
    print("building dominance axes ...", flush=True)
    dom_axes: dict[str, dict] = {}
    for n in names:
        p = EV.perm_of(everyone[n])
        r = rows19[n]
        dom_axes[n] = {
            "floor": six.normfloor(p), "mean": six.mean_saved(p),
            "wfd": r["community"]["wfd"],
            "genkey": r["community_primed"]["genkey_primed"],
            "oxey1": r["community_primed"]["oxey1_primed"],
            "oxey2": r["community_primed"]["oxey2_primed"],
            "lsb": r["kmstats"]["lsb"], "sfb": r["kmstats"]["sfb"], "sfs": r["kmstats"]["sfs"],
            "scissor": r["gauges"]["scissor"],
        }
    out["dominance_axes"] = dom_axes
    out["dominance_frame"] = {"axes": list(J.DOM_AXES), "sign": J.DOM_SIGN,
                              "predicate": "n_ge == 10 AND n_strict >= 1 (trap 33)"}
    out["dominance"] = {}
    for cn in champs:
        res = {}
        for inn in J.INCUMBENTS:
            is_dom, n_ge, n_gt = J.dominates(dom_axes[cn], dom_axes[inn])
            res[inn] = {"dominates": is_dom, "n_ge": n_ge, "n_strict": n_gt}
        out["dominance"][cn] = {
            "vs_incumbents": res,
            "dominator_exists": any(v["dominates"] for v in res.values()),
            "best_n_ge": max(v["n_ge"] for v in res.values()),
        }

    # ---- J4 continued: optimizing-the-ruler, win counts on the independent gauges ----
    #: LOWER-better direction for each of the 19 gauges. sfr is a PERMUTATION INVARIANT
    #: (trap 23) so it is excluded from win counts — it is a tie by construction.
    LOWER_BETTER = {
        "sfb": True, "sfs": True, "sfb-dist": True, "sfs-dist": True, "lsb": True,
        "lsb-dist": True, "redir": True, "scissor": True, "imbalance": True,
        "comfort": True, "oxey-style": True,
        "alt": False, "roll": False, "sr-roll": False,
        "genkey": True, "oxeylyzer1": False, "oxeylyzer2": False, "wfd": False,
    }
    out["gauge_direction"] = LOWER_BETTER
    out["invariant_excluded"] = ["sfr"]
    out["ruler_check"] = {}
    for cn in champs:
        per_inc = {}
        for inn in J.INCUMBENTS:
            wins = []
            losses = []
            for gauge, lower in LOWER_BETTER.items():
                a = out["per_layout"][cn]["gauges19"][gauge]
                b = out["per_layout"][inn]["gauges19"][gauge]
                if a == b:
                    continue
                better = (a < b) if lower else (a > b)
                (wins if better else losses).append(gauge)
            per_inc[inn] = {"n_gauges_scored": len(LOWER_BETTER), "wins": sorted(wins),
                            "losses": sorted(losses), "n_win": len(wins), "n_loss": len(losses)}
        out["ruler_check"][cn] = {
            "per_incumbent": per_inc,
            "normfloor": out["per_layout"][cn]["normfloor"],
            "normfloor_negative": out["per_layout"][cn]["normfloor"] < 0,
            "note": ("effective dof over the 19 gauges is ~4-5, so a raw win count "
                     "over-counts independent evidence ~4x (trap 39); sfr excluded as a "
                     "permutation invariant (trap 23)"),
        }

    path = STATE / "judgement.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nWROTE {path}")

    # ---- printable table ----
    print(f"\n{'=' * 118}")
    hdr = (f"{'layout':<16}{'evidence':>11}{'ms/char':>10}{'d_ms vs best inc':>18}"
           f"{'normfloor':>11}{'mean_sav':>10}{'n_ood':>7}{'dom?':>7}")
    print(hdr)
    print("-" * 118)
    best_inc_ms = min(out["per_layout"][n]["ms_per_char"] for n in J.INCUMBENTS)
    best_inc = min(J.INCUMBENTS, key=lambda n: out["per_layout"][n]["ms_per_char"])
    order = list(champs) + list(J.INCUMBENTS) + list(J.REFERENCE)
    for n in order:
        r = out["per_layout"][n]
        d = r["ms_per_char"] - best_inc_ms
        dom = ""
        if n in out.get("dominance", {}):
            dom = "YES" if out["dominance"][n]["dominator_exists"] else "no"
        print(f"{n:<16}{r['evidence_score']:>11.4f}{r['ms_per_char']:>10.4f}{d:>+18.4f}"
              f"{r['normfloor']:>11.6f}{r['mean_saved_pct']:>10.4f}{r['n_out_of_domain']:>7}{dom:>7}")
    print("-" * 118)
    print(f"best incumbent on ms/char: {best_inc} at {best_inc_ms:.4f}")
    print(f"paired resolution: {out['paired']['paired_floor_ms_per_char']:.4f} ms/char "
          f"(p95 {out['paired']['paired_floor_p95']:.4f}); unpaired floor "
          f"{out['paired']['unpaired_floor_ms_per_char']:.4f}")
    print(f"SS shares: {out['paired']['ss_share_pct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
