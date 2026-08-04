# HYBRIDB-1 / TRIAXIS-1 — pre-registration

Committed **before any decision-bearing number of either arm exists**. The causal order is
verifiable in git timestamps: the only numbers that exist at this commit are (i) `repro.json`, a
pure REPRODUCTION of 20 numbers three siblings already published (nothing new, nothing
decision-bearing), and (ii) `timeprobe.py`'s wall-clock timings, which decide budget and nothing
else.

Two arms. ARM 1 is cheap and can kill ARM 2's premise, so it runs first.

---

## §0 What each arm maximizes, and what would make me abandon it

**ARM 1 (HYBRIDB-1).** hybrid-B is the one hybrid EXPLOIT-1 singled out and never trained: 18
columns (interp.1's 10 ordinals + the served ROW and FINGER one-hots), null space cut 71% at a
MAXCORR of 0.7079 that clears INTERPFRAME-1's registered 0.7850 bar. Its accuracy is unmeasured
and its exploitability is unmeasured. **The maximand is a HONEST VERDICT, not a pass.** The
outcome that would make hybrid-B adoptable-for-scoring is *accuracy-neutral AND non-exploitable*;
the outcome I consider more likely is registered in §4 as H_A.

**ARM 2 (TRIAXIS-1).** Resolve an inconsistency between two of the parent's own claims, WITH
NUMBERS, before building anything. Then build the thing the resolved axis actually points at —
which may be a TOOL change and may be nothing.

**Abandon rules, registered now:**
* **A1.** If hybrid-B fails the high-wpm non-regression gate STRUCTURALLY, that is the ARM-1
  headline and I do **not** build variants hunting for a pass. One registered variant, measured,
  reported. (I still run the exploitability probe — a frame that fails the gate AND is exploitable
  is a strictly stronger negative, and the probe is ~5 minutes.)
* **A2.** If hybrid-B is still exploitable, it is **not adoptable for scoring** and I say so
  plainly. "Less exploitable" is not "safe".
* **A3.** No fishing across hybrids. hybrid-A and hybrid-C are NOT trained. If a reader wants
  them, that is a separate registration.

---

## §1 ARM 1 — the frame under test, defined so it cannot drift

`hybrid-B` := `BIGRAM_INTERP_FEATURE_NAMES` (10) ++ `["bottom","home","top"]` ++
`["pinky","ring","middle","index","lateral"]` (8 served one-hots, selected BY NAME from
`BIGRAM_FEATURE_NAMES` and asserted present). 18 columns. Stamped with a **NEW** version constant
`FEATURE_VERSION_HYBRIDB = f"{FEATURE_VERSION}+hybrid-b.1"`; `FEATURE_VERSION` is **not** edited,
`_BIGRAM_PLACEMENT_NAMES` and `_TRIGRAM_LEVEL_NAMES` (the version-locked shared prefixes) are
**not** touched, and `data/models/k31/` is read-only. A fifth disjoint model population.

**Monotone constraints — registered NOW because the choice is decision-bearing.** The ten interp
columns keep `BIGRAM_INTERP_MONOTONE` verbatim. The eight added one-hots are **UNCONSTRAINED**,
for two reasons that are not post-hoc:
1. `{bottom, home, top}` are exactly collinear (they sum to 1 on every letter key), so any
   sign assignment over all three is self-contradictory: raising `bottom` while holding the other
   two fixed is not a reachable perturbation.
2. ADJ-2 PINKY-MONO measured a constrained **binary indicator** on this repo learning exactly zero
   magnitude, and INTERPFRAME-1 §5 explicitly scoped its non-reproduction of that to *graded
   geometry* columns. These eight are binary indicators.

⚠ **Registered CONSEQUENCE, so it cannot later read as a discovery:** M4 MONOFRAC (mass on
verified-monotone columns) **must** fall below interp.1's 1.0000 and below INTERPFRAME-1's ≥0.90
bar, because 8 of 18 columns carry no constraint. I register the *prediction* that hybrid-B FAILS
M4 and that this is a structural property of the design, not a measurement about the world. What is
*not* predetermined is the MAGNITUDE — the share of attribution mass the unconstrained one-hots
attract — and that is the number worth reporting.

**⚠ A CORRECTION TO MY BRIEF I AM REGISTERING BEFORE MEASURING.** My brief says hybrid-B clears
"INTERPFRAME-1's MAXCORR bar" and treats that as clearing the interpretability half. It clears
**one of the seven bars INTERPFRAME-1 registered**. hybrid-B was screened on M1 alone because
EXPLOIT-1's §g screen was model-free and the other six need a trained model. I now have the trained
model, so I measure **all seven** (§3) — and I have already measured, in `repro.json`, that
hybrid-B has **2 pairs |r|>0.7 vs interp.1's 1**, so it is already known to be worse than interp.1
on a sub-metric of the very bar it "clears". Calling M1 "the interpretability bar" would be the
kind of label-vs-referent substitution this campaign keeps catching.

## §2 ARM 1 accuracy — bars, and the exonerating outcome

Instrument: the shipped `validate()` LOLO harness, driven exactly as INTERPFRAME-1's `lolo.py`
drove it. 4 folds (azerty, dvorak, qwerty, qwertz) × 3 seeds (0,1,2), matched seeds/folds/params,
`ROW_STAGGERED_31`, `bistrokes31_v1.tsv`, wpm 90, `n_jobs=8`. Arms:

| arm | frame | why |
|---|---|---|
| **CUR** | served 20c | the incumbent, **re-measured here**, not borrowed |
| **HYBRIDB** | hybrid-B 18c | the candidate |
| **INTERP** | interp.1 10c | the published reference point, re-measured on the same run |

All three re-measured in ONE run so the three-way comparison is on identical folds/seeds.

Deltas are **PAIRED PER-FOLD** (MOR-FIX-1): per (fold, seed) difference, then the per-fold mean.
**Never a mean of ratios.** Reported: ρ, ρ/ceiling, wmae, umae, τ_heldout.

**Registered decision rules:**
* **B1 — THE GATE (the one refusal).** `require_no_high_wpm_regression_in_report`-style
  per-fold comparison against **CUR's own per-bucket ρ on the same fold** (INTERPFRAME-1's
  corrected baseline; a pooled cross-fold baseline made the gate refuse the incumbent, which
  measures fold heterogeneity, not the candidate). STRUCTURAL = regresses on EVERY seed of a fold
  ⇒ **REFUSAL**. Noise-only = some seeds ⇒ reported, no veto. **GATE CONTROL, mandatory: the gate
  must PASS CUR against CUR's own per-fold ρ. If it refuses the incumbent, every candidate verdict
  from it is uninterpretable and I report that instead of a verdict.**
* **B2 — RANK.** hybrid-B is rank-neutral iff mean paired Δρ vs CUR ≥ −0.005 **and** τ_heldout is
  1.0 on all 3 seeds. (−0.005 is ~10× interp.1's measured −0.00047 and ~0.6% of the absolute
  ρ≈0.815 — a threshold set to be *loose*, because the interesting failure is magnitude, not rank.)
* **B3 — MAGNITUDE.** Reported as the number, no pass/fail bar (INTERPFRAME-1 §0's convention:
  interpretability is the maximand and CLOSING-2 measured nine widening arms to accuracy-null).
  The registered PREDICTION is in §4 (H_B).

**EXONERATING OUTCOME for ARM 1 accuracy, stated so the arm can lose:** hybrid-B **passes** B1
(no structural high-wpm regression on any fold) and **passes** B2. That outcome is available and
would be a genuinely new result — the first frame in this campaign to buy interpretability without
the structural gate refusal. I register in §4 that I expect it to *pass* B1, i.e. **my registered
expectation is the one that makes the arm interesting, so the arm is not built to condemn.**

## §3 ARM 1 interpretability — all seven bars, measured not screened

INTERPFRAME-1's own instrument (`agent-artifacts/interpframe/metrics.py`), loaded BY PATH (a plain
`import metrics` picks up that directory's `_boot.py`, which shadows mine and pins the wrong
worktree). Weighting grid: **flagship-c3**, INTERPFRAME-1's own — a MAXCORR read on qwerty-C30M
gives 0.9556 vs the published 0.9813 purely from the grid.

| id | metric | dir | registered bar | interp.1's value |
|----|--------|-----|----------------|-----------------:|
| M1 | MAXCORR | ↓ | ≤ 0.7850 | 0.7037 |
| M1b | MEANCORR | ↓ | reported | 0.1572 |
| M2 | CONSTFRAC | ↓ | == 0 exactly | 0.0000 |
| M3 | SPLITPAIRS | ↓ | < 7 | 2 |
| M4 | MONOFRAC | ↑ | ≥ 0.90 | 1.0000 |
| M5 | SIGNSTAB / ρ | ↑ | ≥ 0.9000 / ρ ≥ 0.8737 | 1.0000 / 0.9394 |
| M6 | SEEDSTAB unanimity | ↑ | ≥ 0.8000 | 1.0000 |

M3's `same_property_groups` needs a registered grouping for the new frame. Registered here, with
the SAME generosity as the two existing lists (a grouping that split hybrid-B's related columns
apart to flatter it would make M3 meaningless): interp.1's three groups **unchanged**, plus
`{bottom, home, top}` and `{pinky, ring, middle, index, lateral}` (the served frame's own two
groups, verbatim), plus the CROSS groups that are the whole point of a hybrid —
`{bottom_bias, bottom, home, top}` and `{finger_load, pinky, ring, middle, index, lateral}` and
`{off_home_column, lateral}` — because an ordinal and the one-hots it was built to *replace* are by
construction the same property, and a hybrid that carries both is exactly where a same-property
sign conflict would appear.

## §4 THE REGISTERED EXPECTATIONS — each stated so it can LOSE

EXPLOIT-1 registered H_RANKPRESERVE as the more likely outcome and lost; that is the standard.

* **H_A (ARM 1 gate) — I expect hybrid-B to PASS B1.** Reasoning: INTERPFRAME-1's H2 established
  the gate failure is NOT the dropped `wpm` column (adding it back bought +0.005 wmae and refused
  the same buckets), and H3 established the mechanism is the RESOLUTION FLOOR. hybrid-B cuts the
  floor-at-mean from 2.0448 to 0.2545 ms (−88%) and restores 573 of the served frame's 765 distinct
  rows. If the gate failure is resolution-driven, an 88% floor cut should clear it. **If hybrid-B
  fails the gate anyway, H_A loses and the finding is that the high-wpm gate failure is NOT
  resolution-driven — which would contradict the surviving mechanism of INTERPFRAME-1 §j and is the
  more valuable outcome.**
* **H_B (ARM 1 magnitude) — I expect wmae to land closer to CUR than to interp.1, specifically
  Δwmae vs CUR < +2.0 ms** (interp.1's was +5.7654). Same reasoning as H_A. Registered as a number
  so it can be wrong.
* **H_C (ARM 1 exploitability) — I expect hybrid-B to be STILL EXPLOITABLE in the B channel.**
  Reasoning: R2 measured a ZERO-MODEL-ERROR control on interp.1 still exploitable at 2.6× its own
  floor, so exploitability is a property of the collapse, and hybrid-B retains 28.8% of interp.1's
  searchable null space (0.9377 vs 3.2565 ms) against a B-channel floor of ~0.147. **The
  exonerating outcome is available and would be the genuinely new result: gap ≤ floor.**
* **H_D (ARM 2 axis) — I expect "split-pairs YES, resolution NO"**, i.e. the parent's FM5 claim
  holds on the split-pairs axis and is REFUTED on the resolution axis, making both of the parent's
  statements individually true and the *word* "worst" the thing that was overloaded. **The
  outcomes that would refute H_D: (a) the 51/3.0465 split-pairs figure does not reproduce, or
  (b) it reproduces but is not actually WORSE than the bigram frame once normalized per column /
  per pair — 51 raw conflicts over 46 columns is not obviously worse than 7 over 20 until the
  normalization is chosen, and I register the normalization BELOW rather than after seeing it.**

## §5 ARM 1 exploitability — EXPLOIT-1's design, reused not reinvented

Reuse `agent-artifacts/goodhart/exploit.py`'s design verbatim, with the interp surface replaced by
the hybrid-B surface:

* **Operationalization.** Optimize AGAINST the hybrid-B surface; score the winner on the SERVED /
  shipped gauge. `gap = trusted(best-of-12 HYBRIDB) − trusted(best-of-12 SERVED)`; positive = worse.
  Best-of-N selected on each arm's **own** objective (what a real campaign does), reported always on
  the trusted surface.
* **Two channels, both registered up front.** **G** = the reported gauge (T2+Tcond, the adoption
  question); **B** = the bigram table alone (maximum sensitivity on the channel hybrid-B replaces).
  Arms differ in EXACTLY ONE object: the 961-entry T2.
* **Rule.** EXPLOITABLE iff `gap > floor`; **EXONERATED iff `gap ≤ floor`**; `gap < 0` is a
  stronger exoneration and is reported as such. **Margin-vs-floor BEFORE any p-value.**
* **MY OWN MEASURED FLOOR — no borrowed constant.** p95 of the SERVED control arm's own
  disjoint-half best-of-12 disagreement, 2000 splits, same design AND same data volume. (EXPLOIT-1's
  0.147085 / 1.081648 were measured on ITS run; a floor must match the comparison's data volume, so
  mine is re-measured even though the design is identical. I report both and flag any difference.)
* **24 seeds per arm per channel**, `SimulatedAnnealing(alpha=0.999)` + `two_opt`, ONE `search()`
  function for both arms so the polish is provably symmetric.
* **M4 WITHIN-GROUP ADVERSE SELECTION**, EXPLOIT-1's non-circular mechanism test, on hybrid-B's
  grouping: mass-weighted `T2_served[cell] − classmean(T2_served)`, reading ONLY the truth's table
  and the grouping. EXPLOIT-1 measured interp-optimal **+0.1882** vs served-optimal **−0.1754**
  (opposite signs). **Registered prediction: the same sign split appears on hybrid-B but SMALLER in
  magnitude, roughly in proportion to the 71% null-space cut.** A NULL here (same sign on both arms,
  or |Δ| at noise) would be evidence the null-space cut removed the mechanism.
* **INVARIANT 5 — no metric that is an algebraic function of its own outcome.** EXPLOIT-1's M2 was
  an identity (1.4e-14) because its bigram weight was the trigram table's own marginal. I use the
  same weighting, so **M2 will be the same identity on my run** and I register it NOW as a
  DECOMPOSITION (it prices the illusion), never as evidence. I will verify the identity numerically
  and report the residual.
* **INVARIANT-4 aborting checks**, every one aborting rather than warning: (1) trusted gauge parity
  vs `TimeSurface.card()` ≤ 1e-12 rel; (2) the models ARE hybrid-B models — stamp, 18 columns,
  monotone tuple recorded, asserted per seed, retrained on MY tree (a filename is not a
  provenance); (3) the two surfaces actually DIFFER; (4) the collapse is in the SURFACE the search
  queries, not merely in the feature matrix (max within-class `T2_hybridb` spread == 0 to 1e-9);
  (5) ONE `search()` for both arms; (6) `assert_tree()` on every run.
* **BOOTSTRAP the verdict's stability** (which seeds fill each arm's block, 4000 resamples, at
  n=3/6/12) and report it BESIDE the registered verdict, per INVARIANT 6. The registered verdict
  stands as recorded regardless of what the bootstrap says (goalpost discipline); its fragility is
  reported, not substituted.

## §6 ARM 2 — the axis question, with the normalization registered BEFORE measuring

The parent has written both of these, and both are its own:
* **(i)** "the trigram channel is where the defect measured WORST — 51 split pairs, 3.0465 ms/char"
  (INTERPFRAME-1 §a, M3 row).
* **(ii)** "the TRIGRAM frame is BETTER RESOLVED than the served bigram frame (0.9401 vs 0.7960;
  largest group 2 vs 4) ⇒ FM5's 'trigram is worst' does NOT hold on the resolution axis"
  (FRAMEDIAG-1 §e1).

**Registered procedure, in this order:**
1. **Reproduce (ii)** — done already in `repro.json`, model-free: 0.9401 vs 0.7960, largest group
   2 vs 4, all exact. 🟢
2. **Reproduce (i)** from the shipped `shap_diff` on the SAME pair INTERPFRAME-1 used
   (flagship-c3 → graphite, wpm 90, blend-v1), both channels, using the shipped
   `same_property_groups`. Report the reproduced counts and conflict masses.
3. **NORMALIZE, three ways, ALL REGISTERED NOW so none can be picked after seeing the numbers:**
   * **N1 raw count** — 51 vs 7 (what the parent quoted).
   * **N2 per same-property PAIR OPPORTUNITY** — conflicts ÷ the number of same-property pairs the
     registered grouping actually contains for that frame. A frame with 46 columns and bg1_/bg2_
     mirror pairs has far more opportunities than one with 20, so the raw count is partly a column
     count. **This is the normalization I consider decisive.**
   * **N3 conflict mass as a share of the channel's own |attribution| mass** — 3.0465 ms/char in a
     channel worth +2.1953 ms/char gap is a different statement from the same mass in a channel
     worth +0.9981.
4. **VERDICT RULE:** the trigram channel is "worst on the split-pairs axis" iff it is worse on
   **N2 AND N3**, not merely on N1. If N1 says worse but N2/N3 do not, the honest statement is that
   the 51 is a COLUMN-COUNT artifact and the parent's (i) does not survive normalization either —
   in which case **neither** of the parent's two claims supports building a trigram frame, and I say
   so and build nothing.
5. **Only then** decide FRAME vs TOOL vs NOTHING, per §7.

## §7 ARM 2 — what gets built, decided by the axis answer, registered as a mapping not a plan

| axis answer | what I build |
|---|---|
| split-pairs worst (N2 ∧ N3) **and** resolution not worst | **TOOL**: grouped/Owen-Shapley attribution over blocks in `keybo compare`/`shap-diff`. Both TCOND-1 and INTERPFRAME-1 flagged this as the right fix for credit-splitting. NOT a new frame — a frame fix cannot address non-unique credit across correlated columns. |
| resolution worst | a trigram interp frame would be justified; **but** (ii) is already measured and refutes this, so this row is registered only to make the mapping complete |
| neither survives normalization | **NOTHING is built.** I report the axis resolution and stop. A null that says "the target the brief named does not exist" is the result. |

**Registered NON-CLAIM:** interp.1 is bigram-only (`BIGRAM_INTERP_FEATURE_NAMES` has no trigram
counterpart), so there is **no** trigram interp frame to test — building one is a separate,
unregistered arm and I will not do it inside this one.

**The ASYMMETRY finding** (TCOND-1 verified: key `a`'s absolute placement appears in NO column;
`bg1_*` = placement(a,b) describing key **b**, `bg2_*` = placement(b,c) describing key **c**) is
pursued as a MEASUREMENT regardless of the axis answer, because it is cheap, model-free, and it
bears on whether a "fix" would destroy a real distinction. Registered quantities: (a) the exact set
of columns in which each of the three keys' absolute placement appears; (b) whether the trigram
frame's rows are invariant to any permutation that changes key `a`'s position while holding b,c —
i.e. **is `a`'s placement genuinely unrecoverable, or recoverable through the relational columns?**
Registered prediction: `a`'s ABSOLUTE placement (row/finger one-hots) is absent but `a`'s position
is partly recoverable from `bg1_dx/dy/distance` + `sg_*` GIVEN b and c, so the frame is asymmetric
in ABSOLUTE description while retaining RELATIONAL information — a weaker and more precise claim
than "key a is invisible", and one that could be refuted by finding two cells that differ only in
`a` and share a feature row.

## §8 Invariants I bind myself to

1. **Every floor MEASURED, none borrowed** — including re-measuring EXPLOIT-1's search-seed floor
   at my own data volume, and using the weighted **MEDIAN** for any wmae floor (FRAMEDIAG-1 §c2:
   the mean minimizes squared, not absolute, error; the published 2.2399 was 12.20% above the true
   1.9964).
2. **NO SELF-GENERATED TARGETS, and the tautology flagged where it is unavoidable.** T2 is built by
   `TableBigramScorer` on the SERVED frame, so the served frame's floor against T2 is **0 by
   construction** (FRAMEDIAG-1 §c1) and I will **not** quote the served-vs-hybrid floor pair as a
   two-frame contrast. hybrid-B did not generate T2, so hybrid-B's floor against T2 IS a real
   measurement — but hybrid-B **contains** all 8 of the served one-hots, so I register the open
   question of whether partial containment partially tautologizes it, and I will report
   `target_is_self_generated` from the shipped diagnostic for every frame I floor.
3. **MUTATION-TEST every new assertion, with `pytest -B` AND an explicit `__pycache__` purge BEFORE
   AND AFTER each mutation** (FM4-1: a `.bak` restored in the same second at the same byte size
   satisfies CPython's (mtime, size) `.pyc` check and runs MUTATED BYTECODE against RESTORED
   SOURCE). Every mutation asserted to have CHANGED the file (else `NOT-APPLIED`, never a false
   RED), and `rc` taken directly from pytest, never from a pipe tail.
4. **No assertion whose subject cannot vary.** If a field is only ever checked at its default, I
   force the other value somewhere (FRAMEDIAG-1's M20/M21b).
5. **Nothing adopted or promoted.** `FEATURE_VERSION` untouched; `data/models/k31/` read-only;
   `layouts.py` untouched; no searched board named as a contender (they are experimental
   artifacts); no CODE pushed; no branch merged or deleted.
6. **Confidence-tag everything**, and report margin-vs-floor before any p-value.

---

*Registered by `hybridtri`, a subagent of `keybo-optimization`, on branch `hybridtri` off
`framediag` (7b5362c). Nothing in this document was written after seeing a number it constrains.*
