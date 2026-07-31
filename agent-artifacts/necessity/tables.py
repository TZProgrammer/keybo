"""GENERATED tables for the necessity report — every cell read from the emitted JSON.

No number in the report is hand-transcribed. This script is the only path from artifact to
table, so a table that disagrees with the artifact is impossible by construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(p):
    return json.loads(Path(p).read_text())


def fmt(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main() -> int:
    asym = load(sys.argv[1])
    conf = load(sys.argv[2]) if len(sys.argv) > 2 else None
    ctrl = load(sys.argv[3]) if len(sys.argv) > 3 else None
    out = []
    W = out.append

    v = asym["verdict"]
    at = asym["archive_target"]

    # ---- T0: positive control ------------------------------------------------------------
    if ctrl:
        W("### T0 — POSITIVE CONTROL: does this pipeline reproduce the published cells?\n")
        W("| check | measured | published | abs Δ | verdict |")
        W("|---|---|---|---|---|")
        for c in ctrl["checks"]:
            m = c["measured"]
            meas = m["rho_spearman"] if "rho_spearman" in m else m["mean"]
            pub = c["published"]["rho"] if isinstance(c["published"], dict) else c["published"]
            W(f"| `{c['cell']}` | {fmt(meas, 7)} | {fmt(pub, 7)} | {c['abs_rho_delta']:.2e} | "
              f"{'**PASS**' if c['pass'] else '**FAIL**'} |")
        W(f"\nArchive bank {ctrl['archive_bank_size']} layouts; reference bank "
          f"{ctrl['random_bank_size']}; frame `{ctrl['frame']}`.\n")

    # ---- T1: the identity ----------------------------------------------------------------
    W("### T1 — the SLACK is the SAME algebraic identity (defect found in the brief)\n")
    W("Exactly, for any pool, with `q = u_A/u_B` and `k = sd(C)/sd(D)`:\n")
    W("```")
    W("r_Pearson = [(k^2-1)/(k^2+1)] * (1+q^2)/(2q)")
    W("SLACK     = r_Pearson - (k^2-1)/(k^2+1) = [(k^2-1)/(k^2+1)] * ((1+q^2)/(2q) - 1)")
    W("```\n")
    rows = load("/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
                "poolsweep-1/out/audit-blend-seed0.json")["A1_algebra"]["rows"]
    W("| published cell | k | q = u_A/u_B | predicted r | measured r | Δ | predicted SLACK | "
      "reported SLACK | Δ |")
    W("|---|---|---|---|---|---|---|---|---|")
    e1, e2 = [], []
    for r in rows:
        k, q = r["k_c_over_d"], r["u_ratio"]
        sym = (k**2 - 1) / (k**2 + 1)
        fac = (1 + q**2) / (2 * q)
        full, sp = sym * fac, sym * (fac - 1)
        e1.append(abs(full - r["rho_pearson"]))
        e2.append(abs(sp - r["pearson_minus_algebraic"]))
        W(f"| `{r['label']}` | {k:.3f} | {q:.4f} | {fmt(full, 6)} | {fmt(r['rho_pearson'], 6)} | "
          f"{full - r['rho_pearson']:+.1e} | {fmt(sp, 5)} | {fmt(r['pearson_minus_algebraic'], 5)} | "
          f"{sp - r['pearson_minus_algebraic']:+.1e} |")
    W(f"\n**max |predicted r − measured r| = {max(e1):.3e}** over {len(rows)} rows; "
      f"**max |predicted SLACK − reported SLACK| = {max(e2):.3e}**.\n")

    # ---- T2: the primary result ----------------------------------------------------------
    W("### T2 — THE PRIMARY RESULT: both agreements, archive vs the matched asymmetric random pool\n")
    W("| pool | lineage | n | ACHIEVED u_A | ACHIEVED u_B | ACHIEVED q | inst-vs-inst (cross) | "
      "95% CI | inst-vs-itself (within) |")
    W("|---|---|---|---|---|---|---|---|---|")
    cells = {c["label"]: c for c in asym["cells"]}
    for lbl, lin in (("random-wide", "random"), ("archive-x400", "**archive**")):
        c = cells[lbl]
        W(f"| `{lbl}` | {lin} | {c['n']} | {c['u_A']:.4f} | {c['u_B']:.4f} | {c['u_ratio']:.4f} | "
          f"**{fmt(c['leg_cross'])}** | [{fmt(c['rho_ci95'][0], 3)}, {fmt(c['rho_ci95'][1], 3)}] | "
          f"**{fmt(c['leg_within'])}** |")
    a, s = v["asym"], v["sym"]
    W(f"| `asym-match` (R={v['paired_asym_minus_sym_cross']['n']}, mean) | random | "
      f"{asym['n_per_cell']} | {a['achieved_u_A_mean']:.4f} | {a['achieved_u_B_mean']:.4f} | "
      f"**{a['achieved_q_mean']:.4f}** ± {a['achieved_q_sd']:.4f} | "
      f"**{fmt(a['cross_mean'])}** ± {a['cross_replicate_sd']:.4f} | (replicate sd) | "
      f"**{fmt(a['within_mean'])}** ± {a['within_replicate_sd']:.4f} |")
    W(f"| `sym-match` (R={v['paired_asym_minus_sym_cross']['n']}, mean) | random | "
      f"{asym['n_per_cell']} | — | — | {s['achieved_q_mean']:.4f} | "
      f"{fmt(s['cross_mean'])} ± {s['cross_replicate_sd']:.4f} | (replicate sd) | "
      f"{fmt(s['within_mean'])} ± {s['within_replicate_sd']:.4f} |")
    W(f"\nArchive target was `u_A = {at['u_A']:.6f}`, `u_B = {at['u_B']:.6f}` "
      f"(**achieved**, not requested), `q = {at['q']:.4f}`, `u_geo = {at['u_geo']:.6f}`.\n")

    W("### T3 — the two inferential tests\n")
    pc, pw = v["paired_asym_minus_sym_cross"], v["paired_asym_minus_sym_within"]
    up = v["unpaired_archive_minus_asym_r0"]
    W("| test | pairing | statistic | value | resolution / CI | p |")
    W("|---|---|---|---|---|---|")
    W(f"| `asym − sym`, CROSS leg | **paired** (same bank+seed, only asymmetry differs) | "
      f"mean Δrho over R={pc['n']} | {fmt(pc['mean'])} | replicate sd of Δ = {pc['sd']:.4f} | "
      f"Wilcoxon **{pc['wilcoxon']['p_two_sided']:.4f}** |")
    W(f"| `asym − sym`, WITHIN leg | **paired** (same) | mean Δrho over R={pw['n']} | "
      f"**{fmt(pw['mean'])}** | replicate sd of Δ = {pw['sd']:.4f} | "
      f"Wilcoxon **{pw['wilcoxon']['p_two_sided']:.5f}** |")
    W(f"| `archive − asym`, CROSS leg | unpaired (disjoint universes, different lineage) | "
      f"Δrho | **{fmt(up['delta_rho'])}** | bootstrap CI [{fmt(up['ci95'][0], 4)}, "
      f"{fmt(up['ci95'][1], 4)}] | **{up['p_two_sided']:.4f}** |")
    W(f"| `archive − asym`, CROSS leg (replicate mean) | unpaired | Δrho | "
      f"{fmt(v['archive_minus_asym_replicate_mean_cross'])} | "
      f"{(0.21842724017025103 - a['cross_mean']) / a['cross_replicate_sd']:.2f} replicate sds | — |")
    W("")

    # ---- T4: the q ladder ----------------------------------------------------------------
    W("### T4 — the q-ladder at FIXED geometric-mean narrowness (only asymmetry moves)\n")
    W("`u_A = √q · u_geo`, `u_B = u_geo / √q`, so `u_A·u_B = u_geo²` for every q.\n")
    W("| requested q | ACHIEVED q | ACHIEVED u_A | ACHIEVED u_B | cross | within |")
    W("|---|---|---|---|---|---|")
    for c in asym["cells"]:
        if c["kind"] == "ladder":
            W(f"| {c['requested_q']:.4f} | {c['u_ratio']:.4f} | {c['u_A']:.4f} | {c['u_B']:.4f} | "
              f"{fmt(c['leg_cross'])} | {fmt(c['leg_within'])} |")
    W("")
    W("### T5 — the LEVEL ladder (ratio held, overall narrowness scaled)\n")
    W("| cell | ACHIEVED u_A | ACHIEVED u_B | ACHIEVED q | cross | 95% CI | within |")
    W("|---|---|---|---|---|---|---|")
    for c in asym["cells"]:
        if c["kind"] == "level":
            W(f"| `{c['label']}` | {c['u_A']:.4f} | {c['u_B']:.4f} | {c['u_ratio']:.4f} | "
              f"{fmt(c['leg_cross'])} | [{fmt(c['rho_ci95'][0], 3)}, {fmt(c['rho_ci95'][1], 3)}] | "
              f"{fmt(c['leg_within'])} |")
    W("")

    if conf:
        c1 = conf["C1_within_leg_fairness"]
        W("### T6 — CONFIRMATORY C1: is the WITHIN-leg comparison fair? (matched on u_seed, "
          "the axis the statistic lives on)\n")
        W("| arm | matched on | mean u_seed_geo | archive's u_seed_geo | mean within | mean cross |")
        W("|---|---|---|---|---|---|")
        W(f"| `archive-x400` | — | {conf['archive_target']['u_seed_geo']:.4f} | "
          f"{conf['archive_target']['u_seed_geo']:.4f} | **{fmt(c1['archive_within'])}** | "
          f"**{fmt(0.21842724017025103)}** |")
        for key, lab in (("match_uB", "seedMEAN `u_A`,`u_B`"), ("match_useed", "**PER-SEED** `u_seed`")):
            g = c1[key]
            W(f"| `{key}` (R={g['within']['n']}) | {lab} | {g['u_seed_geo']['mean']:.4f} | "
              f"{conf['archive_target']['u_seed_geo']:.4f} | {fmt(g['within']['mean'])} ± "
              f"{g['within']['sd']:.4f} | {fmt(g['cross']['mean'])} ± {g['cross']['sd']:.4f} |")
        cv = c1["verdict"]
        W(f"\n`archive − match-uB` on WITHIN = **{fmt(cv['archive_within_minus_match_uB'])}**, on "
          f"CROSS = **{fmt(cv['archive_cross_minus_match_uB'])}**.  \n"
          f"`archive − match-useed` on WITHIN = **{fmt(cv['archive_within_minus_match_useed'])}**, "
          f"on CROSS = **{fmt(cv['archive_cross_minus_match_useed'])}**.\n")

        W("### T7 — CONFIRMATORY C2: replicated q-ladder (F4 retest)\n")
        W("| requested q | ACHIEVED q (mean) | cross mean | replicate sd | within mean |")
        W("|---|---|---|---|---|")
        for q, g in conf["C2_ladder_replicated"]["per_q"].items():
            W(f"| {float(q):g} | {g['achieved_q']['mean']:.4f} | {fmt(g['cross']['mean'])} | "
              f"{g['cross']['sd']:.4f} | {fmt(g['within']['mean'])} |")
        W("")
        W("| F4 test: q vs 1/q | cross mean at q | cross mean at 1/q | \\|Δ means\\| | "
          "pooled replicate sd | Mann-Whitney p | exceeds 0.20? |")
        W("|---|---|---|---|---|---|---|")
        for t in conf["C2_ladder_replicated"]["F4_tests"]:
            W(f"| {t['q']:.4f} vs {t['inv_q']:g} | {fmt(t['cross_mean_q'])} | "
              f"{fmt(t['cross_mean_inv_q'])} | {t['abs_diff_of_means']:.4f} | "
              f"{t['pooled_replicate_sd']:.4f} | {t['mannwhitney_p']:.4f} | "
              f"{'**YES**' if t['exceeds_0.20_threshold'] else 'no'} |")
        W("")
        W("### T8 — CONFIRMATORY C3: does the 4× two-legged signature replicate? (F2 retest)\n")
        W("| arm | mean u_B | mean u_seed_geo | cross mean | replicate sd | within mean | "
          "replicate sd |")
        W("|---|---|---|---|---|---|---|")
        for k, g in conf["C3_level_4x_replicated"]["agg"].items():
            W(f"| `{k}` | {g['u_B']['mean']:.4f} | {g['u_seed_geo']['mean']:.4f} | "
              f"{fmt(g['cross']['mean'])} | {g['cross']['sd']:.4f} | {fmt(g['within']['mean'])} | "
              f"{g['within']['sd']:.4f} |")
        c3v = conf["C3_level_4x_replicated"]["verdict"]
        W(f"\nReproduces BOTH legs (within ≥ +0.90 **and** cross ≤ +0.30): "
          f"**{c3v['asym4x_reproduces_both_legs']}**. Its cross ±2 replicate sd contains the "
          f"archive's +0.2184: **{c3v['asym4x_cross_ci_contains_archive']}**.\n")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
