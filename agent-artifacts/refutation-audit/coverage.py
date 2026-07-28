#!/usr/bin/env python3
"""Honest coverage ledger: for each of the 14 kills, what did I check and HOW?

The brief asks for coverage stated explicitly ("I checked N of M kills, these grounds
verified, these could not be checked"). This is that table, generated from what the drivers
in this directory actually do -- not from memory.
"""
import json
M = "/local/home/zegertho/agent/state/refaudit/artifacts/refutation-map.json"
mp = json.load(open(M))

# depth codes:
#   RD = re-derived the decisive ground myself from the tree/data, with a control
#   LC = ledger/source citation checked at dec1c3f (and f4c917a where relevant)
#   AR = arithmetic re-derived by hand
#   MU = live mutation run in an isolated copy, liveness proven
#   NX = not independently checked (states why)
COVER = {
 1:  dict(verdict="KILL VERIFIES",  depth="MU+LC",
      ground="the repo gate DOES bite the finder's mutation (finder said the flag cannot report False)",
      how="applied _REPO_SKIP_DIRS->frozenset() in an isolated copy, proved live (path+marker+walk count), 43 passed clean -> 1 FAILED mutated"),
 2:  dict(verdict="KILL VERIFIES",  depth="RD+AR",
      ground="the residue is largest-remainder quantization (11 units in 1e9), and apportion()+PROVENANCE document it",
      how="reloaded the committed tables: 2854/4094 disagree, L1=4702, max|diff|=11 (finder exact); max real-share divergence 1.100e-08; control +1e7 -> 9.7e-03"),
 3:  dict(verdict="KILL VERIFIES",  depth="RD",
      ground="docstring says 'over the class's pairs' (met as written); and the exactness metric mixes polarities",
      how="class mask is a strict SUBSET of same_hand's firing set (0 rows outside); under one polarity `alternate` mismatches 870 vs 312; 5 other classes exact 0/0 as control"),
 4:  dict(verdict="KILL VERIFIES",  depth="LC",
      ground="the finder swept the CORRECTED column; the registered h2h used HISTORICAL semantics",
      how="dec1c3f:5098 registers 'tri-serve 13.9%' = the historical 13.8625; :5236 says h2h used historical; :5222 shows 13.86->38.51 correction; :4496 pins min_cell_samples=10 as the campaign frame. The 20/23-contiguous historical sweep itself I did NOT re-run (needs the ~14min rs.pkl build)"),
 5:  dict(verdict="KILL VERIFIES",  depth="LC+AR",
      ground="'no AALTO_FREQ_PRIOR' is already registered; and the finder's 82x divides incommensurable units",
      how="dec1c3f:7123 and :7724 carry the registration verbatim; FLOOR-METHODOLOGY-1 confirms 0.0017 is a TRI_PS dimensionless ceilfrac mean while 0.1388 is FREQ raw percentage points -- different panels AND units"),
 6:  dict(verdict="KILL VERIFIES",  depth="RD+AR",
      ground="the finder's headline -0.7581% is an algebraic identity with the comfort value cancelling",
      how="from committed tables BM=515,596,120 SM=511,687,503 -> 100*(SM/BM-1) = -0.7581% to 4dp; invariant across s in {1,1e6,3.587,1e-9}; and 1/0.66839=1.49613 lands in the registered 1.4961-1.4999x"),
 7:  dict(verdict="KILL VERIFIES",  depth="RD",
      ground="wpm_range is an unfitted literal, and 50/130 are 2 of the 5 VALIDATED midpoints",
      how="every wpm_range mention in training/train.py is a signature default (60,120); nothing computes it; build_cells defaults 40/140/20 -> midpoints [50,70,90,110,130]"),
 8:  dict(verdict="KILL VERIFIES",  depth="RD",
      ground="the counting proof's premise (a core is 26 letters + 4 punct) is false of the real capture",
      how="extracted boards via the repo's own loader path from the shipped zip: 9135 sessions, 9 distinct boards; the one real mtgap-family board is 24 letters/6 punct, agreement 25/30. Control: qwerty/colemak/colemak-dh recovered EXACTLY (my first regex attempt failed this control and was discarded)"),
 9:  dict(verdict="KILL VERIFIES",  depth="RD",
      ground="two arithmetic impossibilities: 1-skip EXCEEDS the marginalization and INVENTS a key",
      how="1-skip exceeds on 3128/3473 common keys, invents 'Z<' (marg=0); control: the SAME test scores 1-skip31 at 4087/4087 exact, 0 exceed, 0 invented. (Vote 1 said 3129/3474 -- off by one, conclusion unaffected)"),
 10: dict(verdict="KILL FAILS -> RESURRECTED (narrowed)", depth="RD+LC",
      ground="'credited elsewhere as the trigram gauge sr-roll' + 'already registered'",
      how="sr-roll occurs 0x in scoring/oxey.py, kmstats not imported, not a DEFAULT_OXEY_WEIGHTS key, is a _TRIGRAM_METRICS member of analysis/kmstats.py, and is a separate co-equal GAUGE_NAMES entry. All 'already registered' cites are feature-schema/D1-driver/effect_curves. Re-derived 324/108/216/0; control credits 108/108 under a finger-order predicate; A3 re-derivation avoids the accused predicates entirely"),
 11: dict(verdict="KILL VERIFIES",  depth="RD",
      ground="the docstring SELF-DOCUMENTS the bar as max-over-candidates; label == referent",
      how="read tune.py at dec1c3f: docstring says 'the maximum achieved by any candidate', code is `best_tau = max(r[2] for r in results)` -- verbatim match. The 200k-pool lexicographic equivalence test I did NOT re-run"),
 12: dict(verdict="KILL VERIFIES",  depth="AR+LC",
      ground="the 17/9/9 bar IS a normalized 8-finger capacity share; and the 'convention gap' IS a pure rescale",
      how="dec1c3f:5353 registers c_f=kappa*m_f/sum(m), m=(.6,.85,1,1,1,1,.85,.6), sum=6.9 -> 0.6/6.9=8.696%~9%, 2*0.6/6.9=17.391%~17%; 1/(1-0.165806)=1.198762 = the finder's '19.9% gap'; and dropping the guard ENTIRELY from sweep1_result.json still yields NONE qualifying"),
 13: dict(verdict="KILL VERIFIES",  depth="RD",
      ground="the 3.23% coverage denominator is wrong -- most trigram columns are bigram compositions",
      how="42 of 46 TRIGRAM_FEATURE_NAMES are bg1_/bg2_/sg_ (19+19+4); only 4 are genuinely trigram-level. Votes said 38 and 43; the exact count is 42 -- NEITHER vote was right, though the direction holds"),
 14: dict(verdict="KILL VERIFIES",  depth="RD",
      ground="G30 and G31 are the same feature-computing object, so feeding G31 adds zero detection power",
      how="the slots field is read nowhere in src/keybo/features/ (grep rc=1); G30/G31 differ in that ONE dataclass field; max|features(G30)-features(G31)| = 0.0 exactly over K30 pairs and a 12^3 trigram subgrid. Control: a ONE-ROW stagger change moves features by 1.527e+02 (a UNIFORM shift is inert -- my first control's bug)"),
}

kills = mp["killed"]
print(f"KILLS: {len(kills)}   CHECKED: {len(COVER)}\n")
verd = {}
for i, r in enumerate(kills, 1):
    c = COVER[i]
    verd[c["verdict"]] = verd.get(c["verdict"], 0) + 1
    print("=" * 100)
    print(f"K{i:<2} [{r['n_refuted']}/{r['n_votes_returned']}] {r['title'][:88]}")
    print(f"     file      : {r['file']}")
    print(f"     VERDICT   : {c['verdict']}   (depth {c['depth']})")
    print(f"     ground    : {c['ground']}")
    print(f"     how I checked: {c['how']}")
print()
print("=" * 100)
print("SUMMARY")
for k, v in sorted(verd.items(), key=lambda kv: -kv[1]):
    print(f"  {v:2d}  {k}")
print()
print("DEPTH CODES: RD=re-derived from tree/data with a control · LC=ledger/source citation")
print("             checked at dec1c3f · AR=arithmetic re-derived · MU=live mutation, isolated copy")
print()
print("NOT INDEPENDENTLY RE-RUN (named honestly, all inside kills whose OTHER grounds verified):")
print("  K4  the 23-threshold historical min_cell sweep (needs a ~14 min rs.pkl rebuild)")
print("  K11 the 200k-pool lexicographic-equivalence test")
print("  K10 the finder's corpus-MASS figures (32-63%) and the qwerty-asymmetry claim")
print("  3 scratch driver scripts cited under abbreviated /local/.../ paths are MISSING on disk")
print("     (K7v3 x2, K10v1 a1_angle_identity.py) -> those legs are unrecheckable FROM THE ARTIFACT;")
print("     for K7 I re-derived the ground from source instead, and for K10 that leg is A3 above.")
