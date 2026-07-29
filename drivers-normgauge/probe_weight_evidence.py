"""FIND-phase probe: which of (a) precision, (b) independence, (c) held-out, (d) equal
is actually IDENTIFIABLE from data reachable from this worktree?

(c) is the strongest if available, so it is checked FIRST and concretely: does a
per-source held-out fold exist that one source could predict for another?
"""
import json
from pathlib import Path

E2E = Path("/local/home/zegertho/keybo-e2e")
COMM = Path("/local/home/zegertho/repos/keybo/data/community/processed")
SS = Path("/local/home/zegertho/agent/state/scissorsupport/artifacts")

print("=== (c) HELD-OUT FEASIBILITY: what stroke tables exist, and whose rows are in them? ===")
for label, p in (("aalto_tri (cond v3)", E2E / "tristrokes_cond_v3.tsv"),
                 ("aalto_tri31", E2E / "tristrokes31_cond_v1.tsv"),
                 ("comm_tri_last", COMM / "tristrokes_last_community.tsv"),
                 ("comm_tri", COMM / "tristrokes_community.tsv")):
    print(f"  {label:22s} exists={p.exists()}  size={p.stat().st_size if p.exists() else 0:,}")

print()
print("=== the SURFACES are a fitted function of these tables. Can I refit? ===")
print("  A held-out weighting needs: refit surface_m on a TRAIN split of source m,")
print("  then score its predictions against a HELD-OUT split of source m' != m.")
print("  Requirement: (i) the training recipe, (ii) both stroke tables, (iii) xgboost.")
print("  What I have vs need is printed by probe_refit_feasibility.py.")

print()
print("=== (a) PRECISION: per-cell support counts — do they exist for the FULL surface? ===")
d = json.load(open(SS / "ss2_support_counts.json"))
print("  scissorsupport counted support for", len(d["cell_price_and_support"]),
      "NAMED CELL GROUPS (a scissor-neighbourhood partition), not all 29,791 surface cells.")
print("  totals (its own frame):", d["totals"])
f = json.load(open(SS / "ss2d_support_filtered.json"))
tot_a = sum(v["AALTO_samples"] for v in f["groups"].values())
tot_c = sum(v["COMMUNITY_samples"] for v in f["groups"].values())
print(f"  ss2d (COVERED-pair-filtered) sums: AALTO {tot_a:,} COMMUNITY {tot_c:,} ratio {tot_a/tot_c:.1f}x")
tot_a2 = sum(v["AALTO_samples"] for v in d["aggregate_support"].values())
tot_c2 = sum(v["COMMUNITY_samples"] for v in d["aggregate_support"].values())
print(f"  ss2  (UNfiltered)          sums: AALTO {tot_a2:,} COMMUNITY {tot_c2:,} ratio {tot_a2/tot_c2:.1f}x")
print("  => the 643x is the COVERED-PAIR-FILTERED scissor-neighbourhood ratio.")
print("     The whole-table ratio is", f"{d['totals']['AALTO_samples']/d['totals']['COMMUNITY_samples']:.1f}x",
      f"({d['totals']['AALTO_samples']:,} / {d['totals']['COMMUNITY_samples']:,}).")
