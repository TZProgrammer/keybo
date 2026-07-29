# Memory — normgauge

## Current State
- **Status:** investigating
- **Task:** Three normalized per-model gauges (aalto-n/comm-n/pool-n) on the shipped `.standardized` frame + one combined weighted objective wired into the shipped optimizer, with pre-registered evidence-based weights, mutation-proven tests, and a >=3-seed run.
- **Next action:** read MODELNORM-1 prereg + the shipped surface/scorer/optimizer code paths.
- **Blocked on:** nothing

## Constraints from parent (do not rediscover)
- Anchors STABILIZE (2 seeds @10M -> identical champion, gap 0.0).
- n=100 random pool is SUFFICIENT for the zero anchor (validated at n=100/1000/10000).
- Scheme reorders NOTHING (0 discordant pairs) — buys an interpretable weight, not a re-ranking.
- qwerty30m normalizes to 0.42-0.56, NOT ~0 -> NOT a direction guard. Guard = each model's own optimum -> exactly 1.0.
- MODELNORM anchors NOT reusable: computed on `.native`; shipped resolver reads `.standardized`. AALTO native==std, COMMUNITY 287.6987 vs 285.0791, POOL 289.6164 vs 282.6804. RECOMPUTE on .standardized.
- ms/char is rank-identical to AALTO (spearman +1.0000 over 9 layouts); NOT a 4th model.
- POOL is fitted on the UNION of AALTO+COMMUNITY sources -> equal weights are NOT neutral.
- AALTO 7,669,316 in-frame samples vs COMMUNITY 11,930 (643x).
- oxey-style CHANGED 2026-07-28 (nested bad_redirect fix) — pre-2026-07-28 ledger values ~0.65-1.45 HIGHER. Use fresh.
- Do NOT edit PREREGISTRATIONS.md, do NOT push, do NOT raise a CR, do NOT adopt/recommend a layout.

## FIND-phase findings (all on /tmp/normgauge, blend-v1 corpus, .standardized frame)

### F1 🟢 The shipped `.standardized` arrays == the `state/keybo-selmethod` `.standardized` arrays, bit-exact
`np.array_equal(S.load_surface(f"{p}_TRI_PS_FREQ_PRIOR"), np.load(NAT/f"{p}_..standardized.npy"))` = True for all 3.
And AALTO `.standardized == .native` bit-exact (max|d|=0.0); COMMUNITY max|d|=121.5530; POOL max|d|=50.7435.
Array means (GENERATED, not retyped): AALTO std 280.9888 / nat 280.9888; COMMUNITY std 285.0791 / nat 287.6987;
POOL std 282.6804 / nat 289.6164. → parent's numbers CONFIRMED exactly.

### F2 🟢 POSITIVE CONTROL FOR MY WHOLE PIPELINE: MODELNORM's AALTO champion reproduces to -2.3e-15
Rescoring MODELNORM's three champions on MY frame/loader/corpus:
 AALTO      223236317224.4177 vs its 223236317224.4182 → reldiff -2.324e-15  ← same frame, so must match: IT DOES
 COMMUNITY  222904485955.3453 vs its 219828038256.7244 → reldiff +1.399e-02  ← different frame, so must differ
 POOL       227610570664.7610 vs its 235438602522.1906 → reldiff -3.325e-02  ← different frame, so must differ
This is a free end-to-end control on corpus + loader + fit arithmetic + charset, obtained BEFORE I compute anything.

### F3 🟢 `standardized - native` is EXACTLY independent of the 3rd slot → it is a BIGRAM-level substitution
max variation over c: AALTO 0.0, COMMUNITY 1.14e-13, POOL 1.14e-13. So standardization replaces each source's
own T2 (bigram tensor) with AALTO's, keeping its own conditional trigram increment.
⚠ CONSEQUENCE, and it is the single most important structural fact for this task: **on the SHIPPED frame all three
"models" already share AALTO's bigram tensor.** The three differ ONLY in their conditional trigram increment.
This WEAKENS the independence of comm/pool from aalto *relative to the native frame* — the opposite direction from
MODELNORM, which chose `.native` for exactly this reason. I must ship on `.standardized` (the user's gauge must be
the shipped one) but must REPORT this. 🟡

### F4 🟢 POOL is a near-exact convex blend of AALTO and COMMUNITY *at cell level* on the shipped frame
OLS over all 29,791 cells: POOL = +26.7935 + 0.454530*AALTO + 0.449591*COMMUNITY, R2 = 0.87400, resid sd 12.329 ms.
Coef sum 0.9041. Cell-level corr: A-C 0.7192, A-P 0.8417, C-P 0.8880.
→ (b) independence correction is IDENTIFIABLE and quantifiable, not just assertable. Nearly equal coefficients
(0.4545 / 0.4496) is the strongest evidence for the "POOL is the union" claim I can get without refitting.

### F5 🔴 THE 643x HAS A NARROWER SCOPE THAN "relative reliability of the two sources" — A WRONG-CONSTANT RISK
I traced it: 7,669,316 / 11,930 = 642.86x comes from `ss2d_support_filtered.json`, summing 6 scissor-neighbourhood
cell groups under a COVERED-PAIR filter. Same artifact family gives:
  - ss2 (unfiltered, same 6 groups):  7,669,316 / 151,365 =  50.7x
  - ss2 `totals` (whole stroke table): 18,535,823 / 401,543 = 46.2x
⚠ AND THE FILTER IS ASYMMETRIC: AALTO's count is IDENTICAL in both artifacts (7,669,316) while COMMUNITY drops
92.1% (151,365 → 11,930). So 643x is not "AALTO has 643x the data"; it is the ratio *within the scissor
neighbourhood after a per-pair coverage filter that only bit COMMUNITY*. The honest whole-table figure is ~46x.
→ The parent's CONCLUSION (AALTO is far better supported; COMMUNITY prices some cells off 5-242 samples) is TRUE
and I confirm it. The CONSTANT 643x must not be used as the reliability ratio for a weight. This is exactly the
"wrong constant attached to a true conclusion" the brief told me to sweep for — found in the brief itself.

### F6 evaluator: hist-then-matmul 149 us / 3-model eval (6,695/s); gather 391 us. reldiff 1.1e-14 between them.
⚠ Per MODELNORM-1 CORRECTION the hist-then-matmul idiom is the BLAS shape-dependence class → must pin the tile.

### F7 (c) held-out predictive weighting: stroke tables DO exist
aalto tristrokes_cond_v3.tsv 565MB + tristrokes31_cond_v1.tsv 571MB (labels: azerty/dvorak/qwerty/qwertz only),
community tristrokes_last_community.tsv 14.7MB. So a refit-and-cross-predict IS physically possible but is a
full model-training campaign (xgboost, 3 seeds, CAND4 recipe) — out of scope for this arm's budget. DECIDE IN PREREG.

## PRE-REGISTERED at commit 64c9ddf (BEFORE any anchor/weight result)
`drivers-normgauge/PREREGISTRATION.md`. Decision tree, falsifiers, ESS shrinkage form, 10 predictions.

## Code shipped (not a driver path)
- `src/keybo/scoring/model_norm.py` — SurfaceFits (tiled/shape-stable evaluator), Anchors
  (normalize + assert_direction + assert_matches_surfaces drift refusal + JSON schema),
  BlendSpec, ModelBlendScorer(IScorer).
- `tests/scoring/test_model_norm.py` — 25 tests incl. harness mutation control.

## RESULTS SO FAR
### R1 🟢 evaluator: padded/tiled path is BIT-stable across batch lengths; unpadded differs 5.6e-15
So the BLAS shape-dependence guard has a LIVE positive control (the mutation control asserts
the unpadded path really is batch-dependent, so the guard can't silently retire).
Parity vs shipped `score_fit`: rel 1.4e-16 (summation order), COMMUNITY bit-exact.

### R2 🟢 (a) precision, MY OWN measurement on the surface-cell frame
AALTO 26,368,247 samples / 5,219 cells covered / median 728 per cell / min 10
COMMUNITY   29,047 samples / 1,044 cells covered / median  18 per cell / min 10
whole-surface ratio 907.78x. Registered ESS = coverage x sqrt(median depth).

### R3 🟢 (b) independence MEASURED at fit level (400-layout random pool, seed 20260728)
POOL = 0.498757*AALTO + 0.508017*COMMUNITY + c, R2 = 0.93881, unique variance share = 0.0612.
Near-exactly symmetric → POOL is ~a sample mean of the other two, NOT an independent 3rd vote.
Cell level agrees: 0.454530/0.449591, R2 0.87400. Two frames agree → structure is trustworthy.

### R4 🟢 anchor searches (5M unique evals requested, 40 islands, 3 seeds, IDENTICAL budget)
At island 33/40: AALTO 2/3 seeds at 223236317224.4177 = MODELNORM's 10M champion EXACTLY.
COMMUNITY 3/3 at 222447818165.8890. POOL 2/3 at 227268377105.3342.
→ P1 (AALTO within +0.05% of target) will PASS; anchors reproduce across seeds.

### R5 🟢 ANCHORS OF RECORD (shipped `.standardized`, blend-v1, baked 90 WPM, g-frame)
`drivers-normgauge/anchors.json`, commit-pending. Budget: 5,000,000 unique evals REQUESTED per
model per seed, ACHIEVED 5,000,263-5,003,863; 40 islands; seeds 20260728/20260901/20261015.
zero (n=100, seed 20260728, statistic=mean):
  AALTO 243118526775.9713  COMMUNITY 249483317974.6619  POOL 247979864398.5926
one (CONSERVATIVE = the SLOWER of the per-seed bests):
  AALTO 223241709941.1167  span  8.1758% of zero  seed spread 0.0271% of span
  COMMUNITY 222447818165.8890  span 10.8366%  seed spread EXACTLY 0.0000
  POOL 227268377105.3342  span  8.3521%  seed spread EXACTLY 0.0000
Champions: AALTO lnfdg-,yehcrstmaoiupxzbwv.kq'j / COMMUNITY cstr,kdeaigflnmypo.uwzqbxvh-j' /
POOL cyea,krstpguoi-mlndfwj'.qhvxzb
GATE: AALTO anchor is +0.0024% vs MODELNORM's 10M-eval target (bar +0.05%) → **MET**.
2 of 3 AALTO seeds hit MODELNORM's champion fit 223236317224.4177 EXACTLY; COMMUNITY 3/3 and
POOL 3/3 identical layout+fit. So the "one" anchors REPRODUCE (confirms MODELNORM's stability
finding on a different frame and a different searcher).
🟢 EXTRA FREE CONTROL I did not plan: my AALTO **zero** anchor 243118526775.9713 matches
MODELNORM's 243118526775.97125 — the random pool (n=100, seed 20260728) reproduces bit-for-bit
across two independent implementations. Strong evidence the pool constructor + evaluator agree.
Stability: n=1000 moves the zero by -0.979 SE (AALTO) / -0.162 (COMMUNITY) / -0.602 (POOL) →
n=100 CONFIRMED sufficient, independently of MODELNORM.

### R6 🟢 TESTS: 27 pass, rc=0 from a sentinel, mutation-controlled
🔴 **THE MUTATION CONTROL FOUND A REAL DEFECT IN ITSELF** — worth a KB entry:
CPython validates a `.pyc` by (source mtime TRUNCATED TO THE SECOND, source size). My
sign-inverting mutant is SIZE-PRESERVING (`a-b` → `b-a`), so mutate-then-restore inside one
mtime second left a cache CPython considered VALID while holding the OTHER version's bytecode.
Verified directly: pyc recorded mtime 1785288965 / size 24429, exactly matching the restored
source. Caught ONLY by testkit's restore-to-green check — without it the harness would have
reported "mutant caught" for a mutant that never executed. Fix: `_write_module` unlinks the
`.pyc` and asserts the mutant stays size-preserving so the hazard stays covered.

### R7 🔴 AMENDMENT 1 (commit d517811): I KILLED TWO DEFECTS IN MY OWN PREREG, before any (c) result
(a) **My prereg said "n=7 community participants" — the TRAINING subset has FOUR** (200001,
200003, 200006, 200007). The 7 are in the whole file. Conclusion ("thin") holds and STRENGTHENS
→ a WRONG CONSTANT ATTACHED TO A TRUE CONCLUSION, in my own prereg. 5th instance this campaign.
(b) **My registered participant bootstrap was a NO-OP on the AALTO side.** It kept a cell if ANY
drawn pid appeared. Measured: held-out AALTO = 24,079 cells / 55,404 pids / median 139 pids per
cell → **0.999992 of cells survive every resample** (min 0.999917), so cell VALUES never move and
the CI collapses → it would have MANUFACTURED SIGNIFICANCE on the side with the MOST data.
COMMUNITY fails the OPPOSITE way (866 cells / 4 pids / median 1 pid per cell → 0.6827 survive), so
a one-sided check would have looked fine. Fixed: cluster bootstrap that RE-AGGREGATES each cell's
value from the drawn participants' samples. Conservative direction → makes my own falsifier EASIER
to fire, so the amendment can't rescue my preferred branch.
(c) THIRD fix, same pass: the falsifier compares rho/ceiling values, so the pooled SE must be on
the rho/ceiling SCALE, not the raw rho scale. Unit mismatch = borrowed-ruler error in miniature.

### R8 🟢 BLEND RUNS COMPLETE (15 cells: 5 weightings x 3 seeds, 5M unique evals each)
POSITIVE CONTROL: solo-COMMUNITY and solo-POOL reproduce their anchor layouts at EXACTLY
1.0000000000 (3/3 seeds each). solo-AALTO = 1.0002713069 — CORRECT, not a bug: it found
MODELNORM's champion, which is 5,392,716 ms FASTER than my CONSERVATIVE anchor (slower of two
seeds), and the excess 2.713e-04 == (anchor-fit)/span EXACTLY. The one-sided-bound property.
Champions: equal → cstrk,.eaygfdlmpnioubzvwxh-qj' (best 0.95990)
           drop-pool → clndf,geihrmstp.aouywzxbvk-qj' (best 0.94955)
Across-seed sd: equal 0.00058, drop-pool 0.00054, solo-AALTO 0.00016, solo-COMM/POOL 0.0.

### R9 🟠 THE (a) BRANCH IS NEARLY SOLO-AALTO — must be reported, it's the honest weakness
Registered ESS: AALTO 5219*sqrt(728) = 140,816 vs COMMUNITY 1044*sqrt(18) = 4,429 → 31.79x.
→ w = AALTO 0.9102 / COMMUNITY 0.0286 / POOL 0.0612.
⚠ Per the brief's point 2 (spearman(ms/char, AALTO) = +1.0000), w_AALTO=0.91 means branch (a) is
≈ optimizing ms/char, i.e. what the campaign ALREADY does. My shrinkage form kept COMMUNITY from
being deleted outright (raw counts would give 0.00103) but did NOT make it a real vote.
**So if (c) is refuted, the honest headline is: the evidence supports a weighting that is nearly
solo-AALTO, and the three-model gauge then buys interpretability, not a new direction.**

### R10 🟢 (c) HELD-OUT CROSS-PREDICTION FIRED — P6 REFUTED (I predicted it would fail)
No refit needed: the sources are DISJOINT so each is already out-of-sample for the other's surface.
 AALTO surface -> held-out COMMUNITY: rho=+0.2786 ceiling=0.5150 rho/ceil=+0.5410 (866 cells, 4 pids)
 COMMUNITY surface -> held-out AALTO: rho=+0.4115 ceiling=0.9757 rho/ceil=+0.4217 (23,714 cells, 55,404 pids)
gap 0.1193 > pooled SE 0.0822 and neither CI crosses 0 → (c) USABLE.
**DECIDED WEIGHTS: aalto-n 0.5276 / comm-n 0.4112 / pool-n 0.0612** (A2-consistent variant:
0.5411 / 0.3977 / 0.0612).
⚠ Note this is a MUCH more balanced weighting than branch (a) would have given (0.9102/0.0286),
so the choice of branch is load-bearing for the weights even though it is not for the ranking.

### R11 🔴 AMENDMENT 2 (commit 11755aa): A THIRD DEFECT, found AFTER the result — reported, not hidden
**THE TELL: COMMUNITY's point estimate 0.411458 lies OUTSIDE its own CI [0.364336, 0.372002].**
CAUSE (arithmetic, not guess): replicates aggregate with a PLAIN MEAN, point estimate uses the
shipped IQR-MEAN. Gap 0.032228 vs CI half-width 0.003833 → **bias is 8.41x the half-width**. CI
midpoint 0.368169 is near the plain-mean estimate (gap 0.011), not the IQR one (gap 0.043). So the
interval is honest FOR A DIFFERENT STATISTIC than the one it was placed around.
ALSO: A1's fix cured the mechanism but not the consequence — boot_median_surviving_cells = 23714 of
23714 = EXACTLY 1.000 on the AALTO side (55,404 pids at median 138/cell), so COMMUNITY's SE 0.00207
is an artifact of cell richness. That interval width is a LOWER BOUND on uncertainty, NOT precision.
BLAST RADIUS (measured): SAME BRANCH either way; weights move ≤0.0136; refuting (c) would need a
**41.8x** SE widening (0.002067 → 0.086462). Pooled SE is dominated by the AALTO side (0.0822),
which comes from the pid-POOR held-out COMMUNITY data where the bootstrap DOES move (650/866).
RESOLUTION: ship the internally-consistent (plain point + plain CI) variant; report both.
NOT CLAIMED: that the intervals are tight, that 0.00207 is meaningful, or that (c) is settled
beyond that 41.8x margin.

### R12 🟢 DELIVERABLE 2 DONE: wired into the SHIPPED optimizer, not a driver
`keybo optimize --model-weight aalto-n=0.5411 --model-weight comm-n=0.3977 --model-weight pool-n=0.0612
--model-anchors drivers-normgauge/anchors.json`. Same SA + 2-opt + best-of-N + Goodhart postflight;
best-of-N loop FACTORED into `_run_search()` so the two objectives can't drift.
🔴 TWO REAL DEFECTS from running it: (i) shipped `.0f` fitness format printed a 0-1 objective as "-1",
hiding the whole search → format by MAGNITUDE; (ii) **reading `args.model_weight` directly broke TWO
PRE-EXISTING tests** that hand-build a SimpleNamespace (test_comfort_weight_loads_the_adjacent_
skipgram_corpus, test_oxey_weight_loads_the_production_skipgram_convention) → `getattr(...,None)`.
My own 27+9 tests all passed while those 2 went red — the reason to run the FULL suite.
🔴 (iii) my first e2e test disabled `--no-local-search`; on THIS objective the 2-opt polish does
nearly all the work (0.523429 off at max_outer 60 AND 300, vs 0.941646 on) — that test would have
"proved" the objective wasn't being optimized. Fixed + documented in the test docstring.

### R13 🔴 AUDITED THE PARENT'S TWO MID-TURN BRIEF CORRECTIONS — found a wrong constant IN a correction
Parent's standing instruction: re-derive its figures, its verifications are claims too. I did.
- CORRECTION 1 (bigram direction channel): IRRELEVANT to my arm. My gauge is a pure trigram QAP
  objective over the fitted surfaces; `grep -c oxey` = 0 in model_norm.py + both drivers. oxey-style/
  roll/redir appear only as REPORTED columns in blend-report.json, never as inputs.
- CORRECTION 2 direction CONFIRMED: AALTO fold count is 4 (azerty/dvorak/qwerty/qwertz) in both
  bistrokes31_v1 (my count 2202 rows) and tristrokes31_cond_v1 (16643). Adopted: my AALTO side's
  55,404 pids are NOT 55,404 independent units for layout-level generalization → makes my AMENDMENT 2
  caveat (AALTO interval is a LOWER bound on uncertainty) MORE warranted.
- 🔴 **BUT ITS "9 distinct participants" IS WRONG — COMMUNITY HAS 7.** Verified 2 ways: (a) distinct
  pids in bistrokes_community.tsv = 7 → {200001..200007}; (b) ingest_report.json's registered `pids`
  map has exactly 7. MECHANISM: 9 = naive `label.rsplit('#',1)[1]` over the 12 labels, where three
  strings are the SAME PERSON (grzegorzkulesza, +pseudo, +rareboost). `+pseudo`/`+rareboost` are
  CORPUS TAGS after the submitter name — data/community/README.md says so verbatim. 9 counts
  label-variants; 7 counts humans. DIRECTION IS THE DANGEROUS ONE: 7<9 so "too few units" gets
  STRONGER. Also its row counts are off by one (2201/16642 vs my 2202/16643; NOT a header line).
  Same fact my AMENDMENT 1 corrected the OTHER way: 7 = whole file, 4 = the fitted rowStagger subset.
  **The SCOPE, not the count, is what must travel with the number.**
  → Relayed to parent as a PROGRESS (non-terminal) message. Audit:
  state/normgauge/artifacts/BRIEF-CORRECTION-AUDIT.md
NOTHING in any anchor, weight or blend changed: my participant counts were emitted in-code from the
stroke tables, never taken from the brief.

## Current State (FINAL)
- **Status:** done
- **Task:** complete — all 4 deliverables shipped, committed on branch `normgauge`, unpushed.
- **Next action:** none — send terminal DONE callback to keybo-optimization.
- **Blocked on:** nothing. Pushing / raising a CR is the human gate (my brief forbids both).

### R14 🟢 FULL SUITE rc=0 (sentinel /tmp/ngC.sentinel), zero failures
Includes the 2 pre-existing tests my flag had broken. 11 commits: 64c9ddf prereg → 156bd47 code →
8387f1c anchors → d517811/11755aa amendments → ff47694 drivers → 8105bec weights → c6f9932 results →
afb83e8 self-kill → aba7c69 CLI → 9d6167c report+audit → 12f4a45 sentinels → 5a90247 sixth self-kill.

### R15 🔴 SIXTH SELF-KILL: P10 is UNSCOREABLE — a borrowed-ruler comparison inside the anti-borrowing section
P10 predicted my sd would be "within 2x of 0.0492-0.0995". My sd is in NORMALIZED BLEND UNITS; those
are ms/char. The ratio has no truth value. My 0.000576 is not 85x smaller than 0.0492 — it is not on
that axis. The quadruple rule was satisfied by my MEASUREMENT and violated by my PREDICTION ABOUT it.

## Improvement Proposals
1. **`.agent/knowledge/workflow/agent-tool-quirks.md`** — add: *"A size-preserving source mutation
   applied and reverted within one mtime second leaves a `.pyc` CPython considers VALID (it validates
   on (mtime truncated to seconds, size)), so a mutation harness can execute the WRONG bytecode in
   either direction — reporting a mutant 'caught' that never ran, or 'survived' when the original ran.
   Any mutate-restore harness must unlink `__pycache__/<mod>.*.pyc`. `testkit`'s restore-to-green
   check is what catches it; without that check it is silent."* Verified: pyc recorded mtime
   1785288965 / size 24429 exactly matching the restored source.
2. **`keybo/testkit.py`** — consider a shipped `write_module_and_invalidate(path, text)` helper, since
   every mutation harness in this repo needs it and the failure is invisible.
3. **Campaign practice** — the wrong-constant sweep should extend to PREDICTIONS, not just results:
   my P10 registered a cross-scale comparison that could never be scored, and no gate caught it
   because a prediction is not a number anyone re-derives.
4. **Scope, not count, must travel with a participant/fold number.** Three different true answers
   coexist here (7 = whole community file, 4 = the fitted rowStagger subset, 9 = naive label-variants),
   and two separate agents got it wrong in opposite directions on the same day.
